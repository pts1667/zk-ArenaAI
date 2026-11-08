"""The actor: it reads a trained frame encoder's latent tokens, frame after frame.  Trained by tools/policy_clone_train.py (cloning) and
tools/policy_train.py (PPO):

    ~/zkml/bin/python tools/policy_train.py --run /mnt/d/zk-learned/runs/NAME

* **The encoder is frozen.**  It's trained on its own (tools/frame_vae.py), and sees true names whatever names the actor's dropped this game.
* **It reads frames in order, as GPT-2 reads text, with what it did between them** (2026-11-08):

      [frame 1: latent tokens] [frame 1: action tokens] [frame 2: latent tokens] [frame 2: action tokens] ...

  A frame's latent tokens attend to their own frame's, and to the latent and action tokens of the frames before it, up to `WINDOW_SECONDS` back, with a learned bias by how
  far back.  Its action tokens say what the policy chose at that frame; they attend to that frame's latent tokens and everything before, so a decision never sees its own
  choice, and the next frame's does.  One forward pass over a run of frames gives every decision in it, and training takes a segment of decisions at once, from the latents
  kept at collection and the actions taken.  Nothing is unrolled: what's kept between decisions is each frame before's latents and action tokens.
* **Action tokens** (`action_tokens`): each own unit that was told something other than keep becomes a vector of what it is (its raw features' embedding), the kind, and
  what it was pointed at (an enemy's or a region's key, a fine option, a type to make).  `ACTION_TOKENS` learned queries read those, with one learned "nothing" always
  there to read, and a count of how many acted.  So a frame where the policy kept everything has action tokens too, which say so.  They're the policy's own choices only:
  the economy script's orders and the opening's aren't, and show only in the frames, as each unit's order.  In cloning, they're the script's labels.
* **It's exact in play as in training.**  With a window of W frames a layer reaches W - 1 frames further back, so a decision depends on `reach` = LAYERS x (W - 1) frames
  before it, and play keeps those; a segment in training starts from the same frames.  Frames from before the game began are a learned embedding.
* **Each own unit reads what that made for its frame,** with a query of its own raw features, so it knows exactly where it is and what it is.
* **Pointers score against embeddings of the raw features** of what they point at, so a target's exact place is there for the choosing, whatever the latent blurred.
* **A key ends in sine codes of its place,** and a pointer's query ends in those of the unit's own, each band scaled by a gain the unit chooses.  Their dot product is then
  a sum over bands of gain x cos(band x (difference in place)): a learned function of how far away and in which direction, which a dot product of raw coordinates can't give.

`units` from `read` is B x U x (D + KEY): each token's reading, 0 for those that didn't read, then its key, in one tensor.

What a seat keeps between decisions (`initial_state`, `decide`) is B x reach x RECORD per frame, newest first: the frame's latents, then its action tokens, flat, so the
trainer's state handling doesn't change.  `decide` leaves the newest frame's action tokens empty; once actions are chosen, `acted` fills them in.
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
from frame_model import CrossBlock  # noqa: E402
from frame_vae import LatentEncoder, load_vae  # noqa: E402
from zkbot.policy import featurize as fz, rawframe as rf, tables  # noqa: E402

ENCODER = "/mnt/d/zk-learned/runs/encoder-questions/final/vae-last.pt"     # the frozen frame encoder it reads
D = 128
HEADS = 4
LAYERS = 2
BANDS = 6                   # as the encoder's: the finest wave is 1/16 of the map's -1..1
CODE = 4 * BANDS            # sin and cos of each of x and z
KEY = D + CODE
UNIT_PLACE = (fz._I, fz._I + 2)
REGION_PLACE = (0, 2)
WINDOW_SECONDS = 8.0        # how far back a frame's tokens attend, in each layer
ACTION_TOKENS = 4           # what the policy did at a frame, as tokens after that frame's
DECISIONS_A_SECOND = 2.0    # at decide_every 1


def window_frames(decide_every: int = 1) -> int:
    """Frames a token attends to, its own included."""
    return max(2, round(WINDOW_SECONDS * DECISIONS_A_SECOND / decide_every) + 1)


class TimeBlock(nn.Module):
    """A GPT-2 block, with a bias on each head by how many frames back a key is.  Queries may be some of the keys' tokens: those of the newest frame, in play."""

    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        self.heads = heads
        self.act_key = nn.Parameter(torch.zeros(heads))     # added for keys that are action tokens
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.age = nn.Parameter(torch.zeros(heads, window))

    def bias(self, query_frame: torch.Tensor, key_frame: torch.Tensor, query_act: torch.Tensor, key_act: torch.Tensor) -> torch.Tensor:
        """heads x Q x K: the age bias, and -inf for keys after the query or out of the window.  A frame's latent tokens come before its
        action tokens: they don't see them."""
        age = query_frame[:, None] - key_frame[None, :]
        seen = (age >= 0) & (age < self.age.shape[1]) & ~((age == 0) & key_act[None, :] & ~query_act[:, None])
        out = self.age[:, age.clamp(0, self.age.shape[1] - 1)] + self.act_key[:, None, None] * key_act[None, None, :]
        return out.masked_fill(~seen, float("-inf"))

    def forward(self, x: torch.Tensor, keys: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, q_count, d = x.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(self.ln1(keys)).chunk(2, -1)
        q = self.q(self.ln1(x))
        o = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=bias.to(q.dtype))
        x = x + self.out(o.transpose(1, 2).reshape(b, q_count, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))


class LatentActor(nn.Module):
    def __init__(self, vocabulary: int, encoder: LatentEncoder, window: int = window_frames()):
        super().__init__()
        self.encoder, self.window = encoder, window
        self.reach = LAYERS * (window - 1)
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(BANDS, dtype=torch.float32), persistent=False)
        self.latent_in = nn.Linear(encoder.latent, D)
        self.slots = nn.Parameter(torch.randn(encoder.tokens, D) / math.sqrt(D))     # which latent token this is
        self.before = nn.Parameter(torch.randn(encoder.tokens, D) / math.sqrt(D))    # a frame from before the game began
        self.blocks = nn.ModuleList(TimeBlock(D, HEADS, window) for _ in range(LAYERS))
        self.act_slots = nn.Parameter(torch.randn(ACTION_TOKENS, D) / math.sqrt(D))      # which action token this is
        self.act_before = nn.Parameter(torch.randn(ACTION_TOKENS, D) / math.sqrt(D))     # what was done before the game began: nothing
        self.lnf = nn.LayerNorm(D)
        self.name_embed = nn.Embedding(vocabulary, pm.NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, pm.ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, pm.ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, D)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + CODE + pm.NAME_DIM + 2 * pm.ID_DIM, D)
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
        # What was done: a vector per unit that acted, read by learned queries.
        self.act_kind = nn.Embedding(rf.KIND_COUNT, D)
        self.act_unit = nn.Linear(D, D)
        self.act_enemy = nn.Linear(KEY, D)
        self.act_region = nn.Linear(KEY, D)
        self.act_fine = nn.Embedding(fz.FINE_OPTIONS + 1, D)
        self.act_type = nn.Linear(tables.TYPE_DIM, D)
        self.act_out = nn.Linear(D, D)
        self.act_nothing = nn.Parameter(torch.randn(1, D) / math.sqrt(D))
        self.act_queries = nn.Parameter(torch.randn(ACTION_TOKENS, D) / math.sqrt(D))
        self.act_count = nn.Linear(1, D)
        self.act_read = CrossBlock(D, HEADS)
        self.route_scale = nn.Parameter(torch.tensor(pm.ROUTE_SCALE_START))
        nn.init.normal_(self.head_kind.weight, std=pm.HEAD_GAIN / math.sqrt(D))
        nn.init.zeros_(self.head_kind.bias)
        with torch.no_grad():
            self.head_kind.bias[rf.KEEP] = pm.KEEP_START

    def code(self, place: torch.Tensor) -> torch.Tensor:
        """... x 2 to ... x CODE: every sine, then every cosine."""
        at = place.unsqueeze(-1) * self.bands
        return torch.cat((at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    # Frames in order ---------------------------------------------------------------------------------------------------------------------------------

    def sequence(self, frames: torch.Tensor, present: torch.Tensor, acted: torch.Tensor, newest_only: bool = False) -> torch.Tensor:
        """frames: B x F x tokens x latent, oldest first.  present: B x F, false for frames from before the game.  acted: B x F x
        ACTION_TOKENS x D, what was done at each frame (`action_tokens`); the newest's isn't read with newest_only.
        Returns B x F x tokens x D, what each frame's latent tokens hold after reading those before; with newest_only, B x 1 x tokens x D."""
        count, tokens, acts = frames.shape[1], frames.shape[2], acted.shape[2]
        state = torch.where(present[..., None, None], self.latent_in(frames) + self.slots, self.before.to(frames.dtype))
        done = torch.where(present[..., None, None], acted.to(state.dtype) + self.act_slots, self.act_before.to(state.dtype))
        per = tokens + acts
        h = torch.cat((state, done), 2).flatten(1, 2)
        frame = torch.arange(count, device=frames.device).repeat_interleave(per)
        is_act = (torch.arange(per, device=frames.device) >= tokens).repeat(count)
        for i, block in enumerate(self.blocks):
            if newest_only and i == len(self.blocks) - 1:
                q = slice(h.shape[1] - per, h.shape[1] - acts)
                h = block(h[:, q], h, block.bias(frame[q], frame, is_act[q], is_act))
            else:
                h = block(h, h, block.bias(frame, frame, is_act, is_act))
        if newest_only:
            return self.lnf(h).view(len(frames), 1, tokens, D)
        return self.lnf(h).view(len(frames), count, per, D)[:, :, :tokens]

    @property
    def record(self) -> int:
        """What's kept of a frame: its latents, then its action tokens, flat."""
        return self.encoder.tokens * self.encoder.latent + ACTION_TOKENS * D

    def _unpack(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cut = self.encoder.tokens * self.encoder.latent
        lead = history.shape[:-1]
        return history[..., :cut].reshape(*lead, self.encoder.tokens, self.encoder.latent), history[..., cut:].reshape(*lead, ACTION_TOKENS, D)

    def initial_state(self, seats: int, regions: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """What a seat keeps between decisions, newest first: the `reach` frames before (`record`), and which of them there have been."""
        return torch.zeros(seats, self.reach, self.record, device=device), torch.zeros(seats, self.reach, device=device)

    def decide(self, b, history: torch.Tensor, seen: torch.Tensor):
        """A decision while playing.  Returns what the heads read, what to keep for next time, and what training will need of this decision.
        The newest frame's action tokens in what's kept are empty until `acted`."""
        now = self.encoder(b).float()
        latents, acted = self._unpack(history)
        frames = torch.cat((latents.flip(1), now[:, None]), 1)
        acts = torch.cat((acted.flip(1), torch.zeros_like(acted[:, :1])), 1)
        present = torch.cat((seen.flip(1), torch.ones_like(seen[:, :1])), 1) > 0.5
        units, reg_h = self.read(b, self.sequence(frames, present, acts, newest_only=True)[:, -1])
        newest = torch.cat((now.flatten(1), torch.zeros(len(now), ACTION_TOKENS * D, device=now.device)), 1)
        kept = torch.cat((newest[:, None], history[:, :-1]), 1), torch.cat((torch.ones_like(seen[:, :1]), seen[:, :-1]), 1)
        return units, reg_h, *kept, {"latents": now}

    def acted(self, b, a: pm.Actions, history: torch.Tensor) -> torch.Tensor:
        """What `decide` kept, with what was just done filled in as the newest frame's action tokens."""
        out = history.clone()
        out[:, 0, self.encoder.tokens * self.encoder.latent:] = self.action_tokens(b, a).float().flatten(1)
        return out

    def segments(self, history: torch.Tensor, seen: torch.Tensor, latents: torch.Tensor, lengths: list[int], acted: torch.Tensor) -> torch.Tensor:
        """Training's: segments of decisions, each from the state kept at its start, in one forward pass.
        history, seen: B x reach (x ...), as `decide` keeps them.  latents: B x T x tokens x latent, each decision's own, padded at the end.
        acted: B x T x ACTION_TOKENS x D, what was done at each (`action_tokens`, from the actions taken).
        Returns sum(lengths) x tokens x D: what each decision's frame holds, segment by segment."""
        before, before_acted = self._unpack(history)
        frames = torch.cat((before.flip(1), latents), 1)
        acts = torch.cat((before_acted.flip(1).to(acted.dtype), acted), 1)
        present = torch.cat((seen.flip(1) > 0.5, torch.ones(latents.shape[:2], dtype=torch.bool, device=latents.device)), 1)
        out = self.sequence(frames, present, acts)[:, self.reach:]
        return torch.cat([out[i, :n] for i, n in enumerate(lengths)])

    def segment_actions(self, b, a: pm.Actions, lengths: list[int]) -> torch.Tensor:
        """`action_tokens` for decisions in segment order, laid out as `segments` takes them: B x T x ACTION_TOKENS x D."""
        flat = self.action_tokens(b, a)
        out = flat.new_zeros(len(lengths), max(lengths), ACTION_TOKENS, D)
        at = 0
        for i, n in enumerate(lengths):
            out[i, :n] = flat[at:at + n]
            at += n
        return out

    # What was done ----------------------------------------------------------------------------------------------------------------------------------

    def action_tokens(self, b, a: pm.Actions) -> torch.Tensor:
        """B x ACTION_TOKENS x D: what each decision's units were told, as `sequence` reads it.  Keep is nothing."""
        hu, keys, reg_h = self._raw(b)
        count = b.kind_mask.shape[1]
        kind = a.kind[:, :count]
        acting = b.ctrl_mask & (kind != rf.KEEP)
        e = self.act_kind(kind.clamp(0, rf.KIND_COUNT - 1)) + self.act_unit(hu[:, :count])
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, KEY))
        enemy = torch.gather(enemy_h, 1, a.enemy[:, :count].clamp_min(0)[..., None].expand(-1, -1, KEY))
        e = e + self.act_enemy(enemy) * (a.enemy[:, :count] >= 0)[..., None]
        region = torch.gather(reg_h, 1, a.region[:, :count].clamp_min(0)[..., None].expand(-1, -1, KEY))
        e = e + self.act_region(region) * (a.region[:, :count] >= 0)[..., None]
        e = e + self.act_fine((a.fine[:, :count] + 1).clamp(0, fz.FINE_OPTIONS))
        made = self.act_type(self.produce_stats[a.produce[:, :count].clamp_min(0)].to(e.dtype))
        e = e + made * (a.produce[:, :count] >= 0)[..., None]
        e = self.act_out(F.gelu(e, approximate="tanh"))
        tokens = torch.cat((self.act_nothing.to(e.dtype).expand(len(e), 1, D), e), 1)
        mask = torch.cat((torch.ones_like(acting[:, :1]), acting), 1)
        queries = self.act_queries.to(e.dtype) + self.act_count(torch.log1p(acting.sum(-1, keepdim=True).float()))[:, None].to(e.dtype)
        return self.act_read(queries, tokens, mask)

    # Units ------------------------------------------------------------------------------------------------------------------------------------------

    def _raw(self, b) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """From each token's own features, before any reading: units' embeddings, their keys, and the regions' keys."""
        place = b.unit_x[..., UNIT_PLACE[0]:UNIT_PLACE[1]]
        code = self.code(place)
        raw = torch.cat((b.unit_x, code, self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(raw), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        centre = self.code(b.reg_x[..., REGION_PLACE[0]:REGION_PLACE[1]])
        hr = self.reg_in(torch.cat((b.reg_x, centre), -1))
        return hu, torch.cat((self.key_unit(hu), code.to(hu.dtype)), -1), torch.cat((self.key_region(hr), centre.to(hr.dtype)), -1)

    def read(self, b, now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """now: B x tokens x D, what each decision's frame holds."""
        hu, keys, reg_h = self._raw(b)
        count = min(b.ctrl_mask.shape[1], hu.shape[1])
        everyone = torch.ones(now.shape[:2], dtype=torch.bool, device=now.device)
        me = self.read_unit(hu[:, :count], now.to(hu.dtype), everyone)
        return torch.cat((F.pad(me, (0, 0, 0, hu.shape[1] - count)), keys), -1), reg_h

    def _score(self, query: torch.Tensor, gain: torch.Tensor, own: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scaled = gain.repeat(1, 1, 2) * own.to(gain.dtype)
        return (torch.cat((query, scaled), -1) @ keys.transpose(1, 2)).float() / math.sqrt(D)

    def heads(self, b, units: torch.Tensor, reg_h: torch.Tensor):
        """Float32 logits with the illegal choices taken out: kind, region, enemy per pointer kind, the enemies' keys, and what to produce.
        Which regions are legal depends on the kind: policy_model.region_legal."""
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
