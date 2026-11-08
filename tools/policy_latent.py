"""The actor on latents: a trained frame encoder's latent tokens in place of all the tokens the first actor read (policy_model.Actor).

    ~/zkml/bin/python tools/policy_train.py sim --run /mnt/d/zk-learned/runs/NAME --latent-encoder /mnt/d/zk-learned/runs/encoder-questions/final/vae-last.pt

* **The encoder is frozen.**  It's trained on its own (tools/frame_vae.py), and sees true names whatever names the actor's dropped this game.
* **It reads frames in order, as GPT-2 reads text.**  Each frame's latent tokens attend to their own frame's and those of the frames before it, up to `WINDOW_SECONDS`
  back, with a learned bias by how far back.  So one forward pass over a run of frames gives every decision in it, and training takes a segment of decisions at once, from
  the latents kept at collection.  Nothing is unrolled: the only thing kept between decisions is the latents of the frames before.
* **It's exact in play as in training.**  With a window of W frames a layer reaches W - 1 frames further back, so a decision depends on `reach` = LAYERS x (W - 1) frames
  before it, and play keeps those; a segment in training starts from the same frames.  Frames from before the game began are a learned embedding.
* **Each own unit reads what that made for its frame,** with a query of its own raw features, so it knows exactly where it is and what it is.
* **Pointers score against embeddings of the raw features** of what they point at, so a target's exact place is there for the choosing, whatever the latent blurred.
* **A key ends in sine codes of its place,** and a pointer's query ends in those of the unit's own, each band scaled by a gain the unit chooses.  Their dot product is then
  a sum over bands of gain x cos(band x (difference in place)): a learned function of how far away and in which direction, which a dot product of raw coordinates can't give.

`units` from `read` is B x U x (D + KEY): each token's reading, 0 for those that didn't read, then its key.  It keeps one tensor so the trainer can treat it as the first actor's.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_model as pm  # noqa: E402
from frame_jepa import CrossBlock  # noqa: E402
from frame_vae import LatentEncoder, load_vae  # noqa: E402
from zkbot.policy import featurize as fz, net as npnet, rawframe as rf, tables  # noqa: E402

D = 128
HEADS = 4
LAYERS = 2
BANDS = 6                   # as the encoder's: the finest wave is 1/16 of the map's -1..1
CODE = 4 * BANDS            # sin and cos of each of x and z
KEY = D + CODE
UNIT_PLACE = (fz._I, fz._I + 2)
REGION_PLACE = (0, 2)
WINDOW_SECONDS = 8.0        # how far back a frame's tokens attend, in each layer
DECISIONS_A_SECOND = 2.0    # at decide_every 1


def window_frames(decide_every: int = 1) -> int:
    """Frames a token attends to, its own included."""
    return max(2, round(WINDOW_SECONDS * DECISIONS_A_SECOND / decide_every) + 1)


class TimeBlock(nn.Module):
    """A GPT-2 block, with a bias on each head by how many frames back a key is.  Queries may be some of the keys' tokens: those of the newest frame, in play."""

    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.age = nn.Parameter(torch.zeros(heads, window))

    def bias(self, query_frame: torch.Tensor, key_frame: torch.Tensor) -> torch.Tensor:
        """heads x Q x K: the age bias, and -inf for keys after the query or out of the window."""
        age = query_frame[:, None] - key_frame[None, :]
        seen = (age >= 0) & (age < self.age.shape[1])
        return self.age[:, age.clamp(0, self.age.shape[1] - 1)].masked_fill(~seen, float("-inf"))

    def forward(self, x: torch.Tensor, keys: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, q_count, d = x.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(self.ln1(keys)).chunk(2, -1)
        q = self.q(self.ln1(x))
        o = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=bias.to(q.dtype))
        x = x + self.out(o.transpose(1, 2).reshape(b, q_count, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))


class LatentActor(nn.Module):
    recurrent = False
    uses_pairs = False

    def __init__(self, vocabulary: int, encoder: LatentEncoder, window: int = window_frames()):
        super().__init__()
        self.encoder, self.window = encoder, window
        self.reach = LAYERS * (window - 1)
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(BANDS, dtype=torch.float32), persistent=False)
        self.latent_in = nn.Linear(encoder.latent, D)
        self.slots = nn.Parameter(torch.randn(encoder.tokens, D) / math.sqrt(D))     # which latent token this is
        self.before = nn.Parameter(torch.randn(encoder.tokens, D) / math.sqrt(D))    # a frame from before the game began
        self.blocks = nn.ModuleList(TimeBlock(D, HEADS, window) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        self.name_embed = nn.Embedding(vocabulary, npnet.NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, npnet.ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, npnet.ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, D)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + CODE + npnet.NAME_DIM + 2 * npnet.ID_DIM, D)
        self.unit_in1 = nn.Linear(D, D)
        self.reg_in = nn.Linear(fz.REGION_DIM + CODE, D)
        self.read_unit = CrossBlock(D, HEADS)
        self.key_unit = nn.Linear(D, D)
        self.key_region = nn.Linear(D, D)
        self.head_kind = nn.Linear(D, rf.KIND_COUNT)
        self.q_enemy = nn.Linear(D, D)
        self.q_region = nn.Linear(D, D)
        self.gain_enemy = nn.Linear(D, 2 * BANDS)
        self.gain_region = nn.Linear(D, 2 * BANDS)
        self.q_type = nn.Linear(D, D)
        table = tables.table()
        self.type_key = nn.Linear(tables.TYPE_DIM, D)
        self.register_buffer("produce_stats", torch.from_numpy(table.stats[table.producible].copy()), persistent=False)
        self.fine0 = nn.Linear(D + KEY, D)
        self.fine1 = nn.Linear(D, fz.FINE_OPTIONS)
        self.route_scale = nn.Parameter(torch.tensor(npnet.ROUTE_SCALE_START))
        nn.init.normal_(self.head_kind.weight, std=npnet.HEAD_GAIN / math.sqrt(D))
        nn.init.zeros_(self.head_kind.bias)
        with torch.no_grad():
            self.head_kind.bias[rf.KEEP] = npnet.KEEP_START

    def code(self, place: torch.Tensor) -> torch.Tensor:
        """... x 2 to ... x CODE: every sine, then every cosine."""
        at = place.unsqueeze(-1) * self.bands
        return torch.cat((at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    # Frames in order ---------------------------------------------------------------------------------------------------------------------------------

    def sequence(self, frames: torch.Tensor, present: torch.Tensor, newest_only: bool = False) -> torch.Tensor:
        """frames: B x F x tokens x latent, oldest first.  present: B x F, false for frames from before the game.
        Returns B x F x tokens x D, what each frame's tokens hold after reading those before; with newest_only, B x 1 x tokens x D."""
        count, tokens = frames.shape[1], frames.shape[2]
        h = torch.where(present[..., None, None], self.latent_in(frames) + self.slots, self.before.to(frames.dtype)).flatten(1, 2)
        frame = torch.arange(count, device=frames.device).repeat_interleave(tokens)
        for i, block in enumerate(self.blocks):
            if newest_only and i == len(self.blocks) - 1:
                h = block(h[:, -tokens:], h, block.bias(frame[-tokens:], frame))
            else:
                h = block(h, h, block.bias(frame, frame))
        return self.lnf(h).view(len(frames), -1, tokens, D)

    def initial_state(self, seats: int, regions: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """What a seat keeps between decisions, newest first: the latents of the `reach` frames before, and which of them there have been."""
        return torch.zeros(seats, self.reach, self.encoder.tokens, self.encoder.latent, device=device), torch.zeros(seats, self.reach, device=device)

    def decide(self, b, history: torch.Tensor, seen: torch.Tensor):
        """A decision while playing.  Returns what the heads read, what to keep for next time, and what training will need of this decision."""
        now = self.encoder(b).float()
        frames = torch.cat((history.flip(1), now[:, None]), 1)
        present = torch.cat((seen.flip(1), torch.ones_like(seen[:, :1])), 1) > 0.5
        units, reg_h = self.read(b, self.sequence(frames, present, newest_only=True)[:, -1])
        kept = torch.cat((now[:, None], history[:, :-1]), 1), torch.cat((torch.ones_like(seen[:, :1]), seen[:, :-1]), 1)
        return units, reg_h, *kept, {"latents": now}

    def segments(self, history: torch.Tensor, seen: torch.Tensor, latents: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Training's: segments of decisions, each from the state kept at its start, in one forward pass.
        history, seen: B x reach (x ...), as `decide` keeps them.  latents: B x T x tokens x latent, each decision's own, padded at the end.
        Returns sum(lengths) x tokens x D: what each decision's frame holds, segment by segment."""
        frames = torch.cat((history.flip(1), latents), 1)
        present = torch.cat((seen.flip(1) > 0.5, torch.ones(latents.shape[:2], dtype=torch.bool, device=latents.device)), 1)
        out = self.sequence(frames, present)[:, self.reach:]
        return torch.cat([out[i, :n] for i, n in enumerate(lengths)])

    # Units ------------------------------------------------------------------------------------------------------------------------------------------

    def read(self, b, now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """now: B x tokens x D, what each decision's frame holds."""
        place = b.unit_x[..., UNIT_PLACE[0]:UNIT_PLACE[1]]
        code = self.code(place)
        raw = torch.cat((b.unit_x, code, self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(raw), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        count = min(b.ctrl_mask.shape[1], hu.shape[1])
        everyone = torch.ones(now.shape[:2], dtype=torch.bool, device=now.device)
        me = self.read_unit(hu[:, :count], now.to(hu.dtype), everyone)
        units = torch.cat((F.pad(me, (0, 0, 0, hu.shape[1] - count)), self.key_unit(hu), code.to(hu.dtype)), -1)

        centre = self.code(b.reg_x[..., REGION_PLACE[0]:REGION_PLACE[1]])
        hr = self.reg_in(torch.cat((b.reg_x, centre), -1))
        return units, torch.cat((self.key_region(hr), centre.to(hr.dtype)), -1)

    def _score(self, query: torch.Tensor, gain: torch.Tensor, own: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scaled = gain.repeat(1, 1, 2) * own.to(gain.dtype)
        return (torch.cat((query, scaled), -1) @ keys.transpose(1, 2)).float() / math.sqrt(D)

    def heads(self, b, units: torch.Tensor, reg_h: torch.Tensor):
        """As `policy_model.Actor.heads`."""
        count = b.kind_mask.shape[1]
        me, keys = units[:, :count, :D], units[..., D:]
        own = self.code(b.unit_x[:, :count, UNIT_PLACE[0]:UNIT_PLACE[1]])
        kind = self.head_kind(me).float().masked_fill(~b.kind_mask, pm.ILLEGAL)
        region = self._score(self.q_region(me), self.gain_region(me), own, reg_h)
        region = region - F.softplus(self.route_scale).float() * b.route_bias
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, KEY))
        enemy = self._score(self.q_enemy(me), self.gain_enemy(me), own, enemy_h)
        enemy = enemy[:, None].expand(-1, b.ptr_enemy.shape[1], -1, -1).masked_fill(~b.ptr_enemy, pm.ILLEGAL)
        produce = (self.q_type(me) @ self.type_key(self.produce_stats).T).float() / math.sqrt(D)
        return kind, region, enemy, enemy_h, produce.masked_fill(~b.ptr_produce, pm.ILLEGAL)

    def fine(self, me: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fine1(F.gelu(self.fine0(torch.cat((me[..., :D], target), -1)), approximate="tanh")).float()


def make(vocabulary: int, encoder_path: str | pathlib.Path, decide_every: int = 1) -> LatentActor:
    return LatentActor(vocabulary, LatentEncoder(load_vae(pathlib.Path(encoder_path))), window_frames(decide_every))


def trainable_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [p for p in actor.parameters() if p.requires_grad]


def parameter_report(actor: LatentActor) -> dict[str, int]:
    trained = sum(p.numel() for p in trainable_parameters(actor))
    frozen = sum(p.numel() for p in actor.encoder.parameters())
    return {"trained": trained, "encoder": frozen, "total": trained + frozen}
