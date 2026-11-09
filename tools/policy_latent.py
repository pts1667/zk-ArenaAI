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
* **It acts as a player does, one order after another** (2026-11-08, after Pluto): a decision is up to `actions.SLOTS` actions, then done.  Each picks a unit, its
  anchor, and an order for it, which may go as a group order to the units about it (zkbot.policy.groups).  A small decoder (`plan`) reads the frame and the actions
  already chosen at this decision, so each action knows what the others before it did; its anchor then reads the frame for itself (`reading`), and the heads choose
  the order as they did when every unit chose its own.  A unit no action picks keeps what it's doing.
* **Action tokens** (`action_tokens`): each action becomes a vector of who it picked (its raw features' embedding), the kind, and what it was pointed at (an enemy's
  or a region's key, a fine option, a type to make), and for a group order its radius and roles (`action_vectors`, which the decoder reads too).  `ACTION_TOKENS`
  learned queries read those, with one learned "nothing" always there to read, and a count of how many there were.  So a frame where the policy did nothing has
  action tokens too, which say so.  They're the policy's own choices only: the economy script's orders and the opening's aren't, and show only in the frames, as each
  unit's order.  In cloning, they're the script's labels.
* **It's exact in play as in training.**  With a window of W frames a layer reaches W - 1 frames further back, so a decision depends on `reach` = LAYERS x (W - 1) frames
  before it, and play keeps those; a segment in training starts from the same frames.  Frames from before the game began are a learned embedding.
* **An action's anchor reads what that made for its frame,** with a query of its own raw features and what the action is about, so it knows exactly where it is
  and what it is.
* **Pointers score against embeddings of the raw features** of what they point at, so a target's exact place is there for the choosing, whatever the latent blurred.
* **A key ends in sine codes of its place,** and a pointer's query ends in those of the unit's own, each band scaled by a gain the unit chooses.  Their dot product is then
  a sum over bands of gain x cos(band x (difference in place)): a learned function of how far away and in which direction, which a dot product of raw coordinates can't give.

`context` gives what every action of a decision reads (`Context`): the frame as the sequence left it, every unit's raw embedding and key, and the regions' keys.

What a seat keeps between decisions (`initial_state`, `decide`) is B x reach x RECORD per frame, newest first: the frame's latents, then its action tokens, flat, so the
trainer's state handling doesn't change.  `decide` leaves the newest frame's action tokens empty; once actions are chosen, `acted` fills them in.
"""
from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_model as pm  # noqa: E402
from frame_model import CrossBlock  # noqa: E402
from frame_vae import LatentEncoder, load_vae  # noqa: E402
from zkbot.policy import actions as act, featurize as fz, groups as gr, rawframe as rf, tables  # noqa: E402

ENCODER = "/mnt/d/zk-learned/runs/encoder-big/final/vae-last.pt"     # the frozen frame encoder it reads (03-training.md 11.17)
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
PLAN_LAYERS = 2             # of the decoder that chooses a decision's actions one after another
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


class PlanBlock(nn.Module):
    """A decoder block: each action's slot reads the frame and the slots up to its own."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_t = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        b, count, d = x.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(torch.cat((self.ln_t(frame), self.ln_q(x)), 1)).chunk(2, -1)
        earlier = torch.ones(count, count, dtype=torch.bool, device=x.device).tril()
        seen = torch.cat((torch.ones(count, frame.shape[1], dtype=torch.bool, device=x.device), earlier), 1)
        o = F.scaled_dot_product_attention(split(self.q(self.ln_q(x))), split(k), split(v), attn_mask=seen)
        x = x + self.out(o.transpose(1, 2).reshape(b, count, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))


@dataclass
class Context:
    """What a decision's actions read: the frame as the sequence left it (B x tokens x D), every unit token's raw embedding (B x U x D)
    and key (B x U x KEY), and the regions' keys (B x R x KEY)."""
    now: torch.Tensor
    hu: torch.Tensor
    keys: torch.Tensor
    reg_h: torch.Tensor


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
        # A decision's actions, one after another: a slot reads the frame and the actions before it, and picks a unit or done.
        self.plan_start = nn.Parameter(torch.randn(1, D) / math.sqrt(D))
        self.plan_slots = nn.Parameter(torch.randn(act.SLOTS, D) / math.sqrt(D))
        self.plan_blocks = nn.ModuleList(PlanBlock(D, HEADS) for _ in range(PLAN_LAYERS))
        self.plan_ln = nn.LayerNorm(D)
        self.q_anchor = nn.Linear(D, KEY)
        self.head_done = nn.Linear(D, 1)
        self.slot_unit = nn.Linear(D, D)        # what the action is about, in its anchor's query
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
        # Group orders (zkbot.policy.groups): alone or which radius, by the unit and its kind; then which roles, by the radius too.
        roles = len(gr.roles()[0])
        self.register_buffer("choosable_roles", torch.tensor([bool((gr.choosable() >> r) & 1) for r in range(roles)]), persistent=False)
        self.group_kind = nn.Embedding(rf.KIND_COUNT, D)
        self.group0 = nn.Linear(D, D)
        self.group1 = nn.Linear(D, len(gr.GROUP_RADII) + 1)
        self.group_radius = nn.Embedding(len(gr.GROUP_RADII) + 1, D)
        self.roles1 = nn.Linear(D, roles)
        # What was done: a vector per unit that acted, read by learned queries.
        self.act_kind = nn.Embedding(rf.KIND_COUNT, D)
        self.act_unit = nn.Linear(D, D)
        self.act_enemy = nn.Linear(KEY, D)
        self.act_region = nn.Linear(KEY, D)
        self.act_fine = nn.Embedding(fz.FINE_OPTIONS + 1, D)
        self.act_type = nn.Linear(tables.TYPE_DIM, D)
        self.act_group = nn.Embedding(len(gr.GROUP_RADII) + 1, D)
        self.act_roles = nn.Linear(roles, D)
        self.act_out = nn.Linear(D, D)
        self.act_nothing = nn.Parameter(torch.randn(1, D) / math.sqrt(D))
        self.act_queries = nn.Parameter(torch.randn(ACTION_TOKENS, D) / math.sqrt(D))
        self.act_count = nn.Linear(1, D)
        self.act_read = CrossBlock(D, HEADS)
        self.route_scale = nn.Parameter(torch.tensor(pm.ROUTE_SCALE_START))
        for head in (self.head_kind, self.group1, self.head_done):
            nn.init.normal_(head.weight, std=pm.HEAD_GAIN / math.sqrt(D))
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.group1.bias[0] = pm.ALONE_START
            self.head_done.bias[0] = pm.DONE_START

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
        """A decision while playing.  Returns what its actions read (`Context`), what to keep for next time, and what training will need
        of this decision.  The newest frame's action tokens in what's kept are empty until `acted`."""
        now = self.encoder(b).float()
        latents, acted = self._unpack(history)
        frames = torch.cat((latents.flip(1), now[:, None]), 1)
        acts = torch.cat((acted.flip(1), torch.zeros_like(acted[:, :1])), 1)
        present = torch.cat((seen.flip(1), torch.ones_like(seen[:, :1])), 1) > 0.5
        ctx = self.context(b, self.sequence(frames, present, acts, newest_only=True)[:, -1])
        newest = torch.cat((now.flatten(1), torch.zeros(len(now), ACTION_TOKENS * D, device=now.device)), 1)
        kept = torch.cat((newest[:, None], history[:, :-1]), 1), torch.cat((torch.ones_like(seen[:, :1]), seen[:, :-1]), 1)
        return ctx, *kept, {"latents": now}

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

    def action_vectors(self, b, ctx: Context | None, a: pm.Actions) -> torch.Tensor:
        """B x K x D: each action as a vector, of who it picked and what it told them.  Those that picked none are no action: their
        vectors are there, and the readers leave them out."""
        hu, keys, reg_h = (ctx.hu, ctx.keys, ctx.reg_h) if ctx is not None else self._raw(b)
        at = a.anchor.clamp_min(0)
        e = self.act_kind(a.kind.clamp(0, rf.KIND_COUNT - 1)) + self.act_unit(torch.gather(hu, 1, at[..., None].expand(-1, -1, hu.shape[-1])))
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, KEY))
        enemy = torch.gather(enemy_h, 1, a.enemy.clamp_min(0)[..., None].expand(-1, -1, KEY))
        e = e + self.act_enemy(enemy) * (a.enemy >= 0)[..., None]
        region = torch.gather(reg_h, 1, a.region.clamp_min(0)[..., None].expand(-1, -1, KEY))
        e = e + self.act_region(region) * (a.region >= 0)[..., None]
        e = e + self.act_fine((a.fine + 1).clamp(0, fz.FINE_OPTIONS))
        made = self.act_type(self.produce_stats[a.produce.clamp_min(0)].to(e.dtype))
        e = e + made * (a.produce >= 0)[..., None]
        e = e + self.act_group((a.group + 1).clamp_min(0))
        e = e + self.act_roles(pm.role_bits(a.roles, self.act_roles.in_features).to(e.dtype)) * (a.group >= 0)[..., None]
        return self.act_out(F.gelu(e, approximate="tanh"))

    def action_tokens(self, b, a: pm.Actions, ctx: Context | None = None) -> torch.Tensor:
        """B x ACTION_TOKENS x D: what each decision's actions were, as `sequence` reads it.  Done is nothing.  Any number of actions,
        in any order: the tokens read them as a set."""
        e = self.action_vectors(b, ctx, a)
        acting = a.anchor >= 0
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

    def context(self, b, now: torch.Tensor) -> Context:
        """now: B x tokens x D, what each decision's frame holds."""
        hu, keys, reg_h = self._raw(b)
        return Context(now.to(hu.dtype), hu, keys, reg_h)

    # Actions, one after another -------------------------------------------------------------------------------------------------------------------

    def plan(self, ctx: Context, vectors: torch.Tensor) -> torch.Tensor:
        """vectors: B x k x D, the actions chosen so far (`action_vectors`).  Returns B x (k + 1) x D: what each of the slots up to the
        next reads, each from the frame and the actions before it."""
        count = vectors.shape[1] + 1
        x = torch.cat((self.plan_start.to(vectors.dtype).expand(len(vectors), 1, D), vectors), 1) + self.plan_slots[:count].to(vectors.dtype)
        for block in self.plan_blocks:
            x = block(x, ctx.now.to(x.dtype))
        return self.plan_ln(x)

    def anchor_logits(self, b, ctx: Context, h: torch.Tensor, taken: torch.Tensor) -> torch.Tensor:
        """h: B x K x D, slots from `plan`; taken: B x K x C, the units actions before each picked.  Returns float32 B x K x (1 + C):
        done, then each controllable unit, those that may not be picked (nothing to tell them, or picked already) taken out."""
        units = b.ctrl_mask.shape[1]
        score = (self.q_anchor(h) @ ctx.keys[:, :units].transpose(1, 2)).float() / math.sqrt(D)
        score = score.masked_fill(~(pm.anchors_legal(b)[:, None] & ~taken), pm.ILLEGAL)
        return torch.cat((self.head_done(h).float(), score), -1)

    def reading(self, ctx: Context, anchor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """B x K x D: each action's anchor reads the frame, as itself and with what the action is about (h, its slot)."""
        hu = torch.gather(ctx.hu, 1, anchor.clamp_min(0)[..., None].expand(-1, -1, ctx.hu.shape[-1]))
        everyone = torch.ones(ctx.now.shape[:2], dtype=torch.bool, device=hu.device)
        return self.read_unit(hu + self.slot_unit(h).to(hu.dtype), ctx.now.to(hu.dtype), everyone)

    def _score(self, query: torch.Tensor, gain: torch.Tensor, own: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scaled = gain.repeat(1, 1, 2) * own.to(gain.dtype)
        return (torch.cat((query, scaled), -1) @ keys.transpose(1, 2)).float() / math.sqrt(D)

    def heads(self, b, me: torch.Tensor, keys: torch.Tensor, reg_h: torch.Tensor):
        """For each action, of its anchor: float32 logits with the illegal choices taken out: kind, region, enemy per pointer kind, the
        enemies' keys, and what to produce.  b is as the anchors see it (policy_model.per_action), me what they read (B x K x D), keys
        every unit token's.  Which regions are legal depends on the kind: policy_model.region_legal."""
        count = me.shape[1]
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

    def _group_in(self, me: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return self.group0(me[..., :D]) + self.group_kind(kind.clamp(0, rf.KIND_COUNT - 1)).to(me.dtype)

    def group(self, me: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        """B x C x (1 + radii): alone, or a group order of each radius."""
        return self.group1(F.gelu(self._group_in(me, kind), approximate="tanh")).float()

    def roles(self, me: torch.Tensor, kind: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """B x C x roles: each role's logit of being in the group, given its radius (group: -1 for alone)."""
        h = self._group_in(me, kind) + self.group_radius((group + 1).clamp_min(0)).to(me.dtype)
        return self.roles1(F.gelu(h, approximate="tanh")).float()


def make(vocabulary: int, encoder_path: str | pathlib.Path, decide_every: int = 1) -> LatentActor:
    return LatentActor(vocabulary, LatentEncoder(load_vae(pathlib.Path(encoder_path))), window_frames(decide_every))


def trainable_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [p for p in actor.parameters() if p.requires_grad]


def parameter_report(actor: LatentActor) -> dict[str, int]:
    trained = sum(p.numel() for p in trainable_parameters(actor))
    frozen = sum(p.numel() for p in actor.encoder.parameters())
    return {"trained": trained, "encoder": frozen, "total": trained + frozen}
