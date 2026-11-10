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

import functools
import math
import pathlib
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_model as pm  # noqa: E402
from frame_model import CrossBlock  # noqa: E402
from frame_vae import LatentEncoder, load_vae  # noqa: E402
from zkbot.policy import actions as act, featurize as fz, groups as gr, rawframe as rf, tables  # noqa: E402

ENCODER = "/mnt/d/zk-learned/runs/encoder-big/final/vae-last.pt"     # the frozen frame encoder it reads (03-training.md, "The frame encoder")
HEADS = 4                   # a lone TimeBlock's, in tests
BANDS = 6                   # as the encoder's: the finest wave is 1/16 of the map's -1..1
CODE = 4 * BANDS            # sin and cos of each of x and z
UNIT_PLACE = (fz._I, fz._I + 2)
REGION_PLACE = (0, 2)
SITE_PLACE = (0, 2)         # a site's x, z, as -1..1 of the map (featurize)
FEAT_PLACE = (0, 2)         # a feature token's x, z, the same
WINDOW_SECONDS = 8.0        # how far back a frame's tokens attend, in each layer
ACTION_TOKENS = 4           # what the policy did at a frame, as tokens after that frame's
Z_DIM = 32                  # a side's strategy (`LatentActor.z_in`, policy_clone_data.side_z): known, then the knobs and counts
DECISIONS_A_SECOND = 2.0    # at decide_every 1


@dataclass(frozen=True)
class Size:
    """How big the actor is: its width, attention heads, layers of the time transformer, and of the decoder that chooses a
    decision's actions one after another.  Kept in each checkpoint (`asdict`), to build the same again."""
    d: int = 256
    heads: int = 8
    layers: int = 4
    plan_layers: int = 4


SIZE = Size()               # the user's, 2026-11-09: about 10M trained parameters (it was d 128, 4 heads, 2 and 2: 1.9M)
SMALL = Size(128, 4, 2, 2)


def window_frames(decide_every: int = 1) -> int:
    """Frames a token attends to, its own included."""
    return max(2, round(WINDOW_SECONDS * DECISIONS_A_SECOND / decide_every) + 1)


@dataclass(frozen=True)
class Frames:
    """How `sequence` lays tokens out, for flex attention: every frame `per` tokens, its first `tokens` latent, then action tokens;
    the queries start at token `offset` (with newest_only, the newest frame's); a key up to `last` frames back is seen."""
    per: int
    tokens: int
    last: int
    offset: int = 0

    def ages(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (q + self.offset) // self.per - k // self.per

    def is_act(self, i: torch.Tensor) -> torch.Tensor:
        return i % self.per >= self.tokens


@functools.lru_cache(maxsize=16)
def _frame_mask(frames: Frames, queries: int, keys: int, device: str) -> BlockMask:
    def seen(b, h, q, k):
        age = frames.ages(q, k)
        hidden = (age == 0) & frames.is_act(k) & ~frames.is_act(q + frames.offset)
        return (age >= 0) & (age <= frames.last) & ~hidden
    return create_block_mask(seen, None, None, queries, keys, device=device)


@torch.compiler.disable
def frame_mask(frames: Frames, queries: int, keys: int, device: torch.device) -> BlockMask:
    """`TimeBlock.bias`'s -inf as a block mask: flex attention skips the blocks nothing in is seen, ~93% of a cloning sample's.
    The same every step for a layout, so it's made once."""
    return _frame_mask(frames, queries, keys, str(device))


# Compiled, as flex attention must be to be fast: eager, it scores every pair.  Dynamic after a second shape (the held-out check's
# batches, a short game's sample), so shapes never pile up recompiles until it falls back to eager.
_flex = torch.compile(flex_attention)


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
        out = self._age_bias(query_frame, key_frame) + self.act_key[:, None, None] * key_act[None, None, :]
        return out.masked_fill(~seen, float("-inf"))

    def _age_bias(self, query_frame: torch.Tensor, key_frame: torch.Tensor) -> torch.Tensor:
        """heads x Q x K: the learned bias by how many frames back.  When each frame's tokens come together and as many of them, as in
        `sequence`, it's indexed a pair of frames at a time and spread to their tokens: indexed a pair of tokens at a time, backward put
        millions of gradients one by one into a table of a few dozen, which took 77% of a cloning step's GPU time (03-training.md, "Speed")."""
        last = self.age.shape[1] - 1
        qf, qn = torch.unique_consecutive(query_frame, return_counts=True)
        kf, kn = torch.unique_consecutive(key_frame, return_counts=True)
        if bool((qn == qn[0]).all() & (kn == kn[0]).all()):
            table = self.age[:, (qf[:, None] - kf[None, :]).clamp(0, last)]
            heads, qp, kp = table.shape[0], int(qn[0]), int(kn[0])
            return table[:, :, None, :, None].expand(-1, -1, qp, -1, kp).reshape(heads, len(query_frame), len(key_frame))
        return self.age[:, (query_frame[:, None] - key_frame[None, :]).clamp(0, last)]

    def frame_bias(self, query_frames: torch.Tensor, key_frames: torch.Tensor, query_act: torch.Tensor, key_act: torch.Tensor) -> torch.Tensor:
        """`bias`, where queries and keys are whole frames' tokens and every frame's are laid out alike, as in `sequence`: query_frames
        (nq) and key_frames (nk) are the frames, query_act (qp) and key_act (kp) which of a frame's tokens are action tokens.  Built a
        pair of frames at a time from tensor ops alone, nothing read back to the host, so that it compiles and graphs (`_age_bias`
        isn't: it checks the layout on the host)."""
        heads, last = self.age.shape[0], self.age.shape[1] - 1
        age = query_frames[:, None] - key_frames[None, :]
        out = self.age[:, age.clamp(0, last)][:, :, None, :, None] + (self.act_key[:, None] * key_act[None, :])[:, None, None, None, :]
        hidden = (age == 0)[:, None, :, None] & key_act[None, None, None, :] & ~query_act[None, :, None, None]
        seen = ((age >= 0) & (age <= last))[:, None, :, None] & ~hidden
        shape = (heads, len(query_frames), len(query_act), len(key_frames), len(key_act))
        out = out.expand(shape).masked_fill(~seen[None], float("-inf"))
        return out.reshape(heads, shape[1] * shape[2], shape[3] * shape[4])

    def bias_dims(self, frames: Frames, queries: int, keys: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """`frame_bias` without its mask, as dimensions to append to each head's queries (heads x Q x W + 1) and keys (K x W + 1),
        W = last + 1: a query holds its frame's age bias for keys of each frame mod W, then act_key; a key, its frame mod W one-hot,
        then whether it's an action token.  Their dot product is the bias for every key in the window, where the age is the frames
        between mod W.  So flex attention needs no score_mod: a score_mod's gradient into this table of a few dozen came a pair of
        tokens at a time, as atomic adds, and made a cloning step 2.4x slower than the dense bias (2026-11-09).  Here it's a sum
        a frame at a time, by autograd."""
        w, dev = frames.last + 1, self.age.device
        frame_of = lambda n, offset: (torch.arange(n, device=dev) + offset) // frames.per  # noqa: E731
        fq, fk = frame_of(queries, frames.offset), frame_of(keys, 0)
        first, end = frames.offset // frames.per, (frames.offset + queries - 1) // frames.per + 1
        span = torch.arange(first, end, device=dev)
        by_frame = self.age[:, (span[:, None] - torch.arange(w, device=dev)[None, :]) % w]       # heads x frames x W
        q_dims = torch.cat((by_frame.index_select(1, fq - first), self.act_key[:, None, None].expand(-1, queries, 1)), -1)
        is_act = (torch.arange(keys, device=dev) % frames.per >= frames.tokens)
        k_dims = torch.cat((F.one_hot(fk % w, w), is_act[:, None]), -1)
        return q_dims.to(dtype), k_dims.to(dtype)

    def forward(self, x: torch.Tensor, keys: torch.Tensor, bias: torch.Tensor | tuple[Frames, BlockMask]) -> torch.Tensor:
        """bias: heads x Q x K added to the scores (`bias`, `frame_bias`), or a layout and its block mask, for flex attention."""
        b, q_count, d = x.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(self.ln1(keys)).chunk(2, -1)
        q = self.q(self.ln1(x))
        if isinstance(bias, tuple):
            frames, mask = bias
            q, k, v = split(q), split(k), split(v)
            q_dims, k_dims = self.bias_dims(frames, q_count, keys.shape[1], q.dtype)
            # The scale multiplies the appended dot product too: the bias goes in times its inverse.
            q = torch.cat((q, (q_dims * math.sqrt(q.shape[-1]))[None].expand(b, -1, -1, -1)), -1)
            k = torch.cat((k, k_dims[None, None].expand(b, self.heads, -1, -1)), -1)
            o = _flex(q, k, v, block_mask=mask, scale=1 / math.sqrt(v.shape[-1]))
        else:
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
    and key (B x U x KEY), the regions' keys (B x R x KEY), the sites' (B x S x KEY) and the feature tokens' (B x F x KEY).  Then each
    controllable unit's key as an anchor (B x C x KEY), from its reading of the frame, and the logit that it's told anything at all
    this decision (B x C, float32; cloning's side loss)."""
    now: torch.Tensor
    hu: torch.Tensor
    keys: torch.Tensor
    reg_h: torch.Tensor
    site_h: torch.Tensor
    feat_h: torch.Tensor
    anchor_keys: torch.Tensor
    ordered: torch.Tensor


class LatentActor(nn.Module):
    def __init__(self, vocabulary: int, encoder: LatentEncoder, window: int = window_frames(), size: Size = SIZE):
        super().__init__()
        self.encoder, self.window, self.size = encoder, window, size
        self.d, self.key = size.d, size.d + CODE
        self.flex = False       # the time transformer's attention by flex attention (`compile_for_cloning`), CUDA only
        self.reach = size.layers * (window - 1)
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(BANDS, dtype=torch.float32), persistent=False)
        self.latent_in = nn.Linear(encoder.latent, self.d)
        self.slots = nn.Parameter(torch.randn(encoder.tokens, self.d) / math.sqrt(self.d))     # which latent token this is
        self.before = nn.Parameter(torch.randn(encoder.tokens, self.d) / math.sqrt(self.d))    # a frame from before the game began
        # The strategy the side plays (04-literature.md P8, C5; `policy_clone_data.side_z`), added to every frame's latent tokens.
        # Zero at first, and an unknown z is all zeros: so an actor trained without it plays as it did.
        # Made without drawing from the generator, so that every other weight is drawn as before it was added.
        with torch.random.fork_rng(devices=[]):
            self.z_in = nn.Linear(Z_DIM, self.d)
        nn.init.zeros_(self.z_in.weight)
        nn.init.zeros_(self.z_in.bias)
        self.blocks = nn.ModuleList(TimeBlock(self.d, size.heads, window) for _ in range(size.layers))
        self.act_slots = nn.Parameter(torch.randn(ACTION_TOKENS, self.d) / math.sqrt(self.d))      # which action token this is
        self.act_before = nn.Parameter(torch.randn(ACTION_TOKENS, self.d) / math.sqrt(self.d))     # what was done before the game began: nothing
        self.lnf = nn.LayerNorm(self.d)
        self.name_embed = nn.Embedding(vocabulary, pm.NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, pm.ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, pm.ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, self.d)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + CODE + pm.NAME_DIM + 2 * pm.ID_DIM, self.d)
        self.unit_in1 = nn.Linear(self.d, self.d)
        self.reg_in = nn.Linear(fz.REGION_DIM + CODE, self.d)
        self.read_unit = CrossBlock(self.d, size.heads)
        # Whom to tell (04-literature.md P1, P2): every controllable unit reads the frame once a decision, before any is picked, so
        # that its key as an anchor knows what's around it; and from that, whether it's told anything at all.
        self.unit_read = CrossBlock(self.d, size.heads)
        self.key_anchor = nn.Linear(self.d, self.d)
        self.head_ordered = nn.Linear(self.d, 1)
        # A decision's actions, one after another: a slot reads the frame and the actions before it, and picks a unit or done.
        self.plan_start = nn.Parameter(torch.randn(1, self.d) / math.sqrt(self.d))
        self.plan_slots = nn.Parameter(torch.randn(act.SLOTS, self.d) / math.sqrt(self.d))
        self.plan_blocks = nn.ModuleList(PlanBlock(self.d, size.heads) for _ in range(size.plan_layers))
        self.plan_ln = nn.LayerNorm(self.d)
        self.q_anchor = nn.Linear(self.d, self.key)
        self.head_done = nn.Linear(self.d, 1)
        self.slot_unit = nn.Linear(self.d, self.d)        # what the action is about, in its anchor's query
        self.key_unit = nn.Linear(self.d, self.d)
        self.key_region = nn.Linear(self.d, self.d)
        self.head_kind = nn.Linear(self.d, rf.KIND_COUNT)
        self.q_enemy = nn.Linear(self.d, self.d)
        self.q_region = nn.Linear(self.d, self.d)
        self.gain_enemy = nn.Linear(self.d, 2 * BANDS)
        self.gain_region = nn.Linear(self.d, 2 * BANDS)
        self.q_type = nn.Linear(self.d, self.d)
        table = tables.table()
        self.type_key = nn.Linear(tables.TYPE_DIM, self.d)
        self.register_buffer("produce_stats", torch.from_numpy(table.stats[table.producible].copy()), persistent=False)
        self.fine0 = nn.Linear(self.d + self.key, self.d)
        self.fine1 = nn.Linear(self.d, fz.FINE_OPTIONS)
        # A BUILD's place (02-policy.md, "Where an action points"): beside one of our units or a site, keyed as regions are, then where
        # beside it, read from that place's key as fine is from its target's.
        self.site_in = nn.Linear(fz.SITE_DIM + CODE, self.d)
        self.key_site = nn.Linear(self.d, self.d)
        self.q_place = nn.Linear(self.d, self.d)
        self.gain_place = nn.Linear(self.d, 2 * BANDS)
        self.offset0 = nn.Linear(self.d + self.key, self.d)
        self.offset1 = nn.Linear(self.d, fz.PLACE_OFFSETS)
        # A RECLAIM's wreck: one of the feature tokens, keyed as sites are.
        self.feat_in = nn.Linear(fz.FEATURE_DIM + CODE, self.d)
        self.key_feat = nn.Linear(self.d, self.d)
        self.q_wreck = nn.Linear(self.d, self.d)
        self.gain_wreck = nn.Linear(self.d, 2 * BANDS)
        # Group orders (zkbot.policy.groups): alone or which radius, by the unit and its kind; then which roles, by the radius too.
        roles = len(gr.roles()[0])
        self.register_buffer("choosable_roles", torch.tensor([bool((gr.choosable() >> r) & 1) for r in range(roles)]), persistent=False)
        self.group_kind = nn.Embedding(rf.KIND_COUNT, self.d)
        self.group0 = nn.Linear(self.d, self.d)
        self.group1 = nn.Linear(self.d, len(gr.GROUP_RADII) + 1)
        self.group_radius = nn.Embedding(len(gr.GROUP_RADII) + 1, self.d)
        self.roles1 = nn.Linear(self.d, roles)
        # What was done: a vector per unit that acted, read by learned queries.
        self.act_kind = nn.Embedding(rf.KIND_COUNT, self.d)
        self.act_unit = nn.Linear(self.d, self.d)
        self.act_enemy = nn.Linear(self.key, self.d)
        self.act_region = nn.Linear(self.key, self.d)
        self.act_fine = nn.Embedding(fz.FINE_OPTIONS + 1, self.d)
        self.act_place = nn.Linear(self.key, self.d)
        self.act_wreck = nn.Linear(self.key, self.d)
        self.act_offset = nn.Embedding(fz.PLACE_OFFSETS + 1, self.d)
        self.act_type = nn.Linear(tables.TYPE_DIM, self.d)
        self.act_group = nn.Embedding(len(gr.GROUP_RADII) + 1, self.d)
        self.act_roles = nn.Linear(roles, self.d)
        self.act_out = nn.Linear(self.d, self.d)
        self.act_nothing = nn.Parameter(torch.randn(1, self.d) / math.sqrt(self.d))
        self.act_queries = nn.Parameter(torch.randn(ACTION_TOKENS, self.d) / math.sqrt(self.d))
        self.act_count = nn.Linear(1, self.d)
        self.act_read = CrossBlock(self.d, size.heads)
        self.route_scale = nn.Parameter(torch.tensor(pm.ROUTE_SCALE_START))
        for head in (self.head_kind, self.group1, self.head_done):
            nn.init.normal_(head.weight, std=pm.HEAD_GAIN / math.sqrt(self.d))
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.group1.bias[0] = pm.ALONE_START
            self.head_done.bias[0] = pm.DONE_START

    def code(self, place: torch.Tensor) -> torch.Tensor:
        """... x 2 to ... x CODE: every sine, then every cosine."""
        at = place.unsqueeze(-1) * self.bands
        return torch.cat((at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    # Frames in order ---------------------------------------------------------------------------------------------------------------------------------

    def sequence(self, frames: torch.Tensor, present: torch.Tensor, acted: torch.Tensor, newest_only: bool = False,
                 z: torch.Tensor | None = None) -> torch.Tensor:
        """frames: B x F x tokens x latent, oldest first.  present: B x F, false for frames from before the game.  acted: B x F x
        ACTION_TOKENS x D, what was done at each frame (`action_tokens`); the newest's isn't read with newest_only.  z: B x Z_DIM,
        the side's strategy, or None for unknown.
        Returns B x F x tokens x D, what each frame's latent tokens hold after reading those before; with newest_only, B x 1 x tokens x D."""
        count, tokens, acts = frames.shape[1], frames.shape[2], acted.shape[2]
        state = torch.where(present[..., None, None], self.latent_in(frames) + self.slots, self.before.to(frames.dtype))
        # None is an unknown z, which cloning shows as all zeros: z_in's bias, learnt with the rest, still reaches every token.
        # (Leaving it out, as before 2026-11-10, gave a z-trained actor told nothing a state it had never been trained on.)
        added = self.z_in.bias[None] if z is None else self.z_in(z.to(self.z_in.weight.dtype))
        state = state + added[:, None, None, :].to(state.dtype)
        done = torch.where(present[..., None, None], acted.to(state.dtype) + self.act_slots, self.act_before.to(state.dtype))
        per = tokens + acts
        h = torch.cat((state, done), 2).flatten(1, 2)
        frame = torch.arange(count, device=frames.device)
        is_act = torch.arange(per, device=frames.device) >= tokens
        last = self.blocks[0].age.shape[1] - 1
        for i, block in enumerate(self.blocks):
            if newest_only and i == len(self.blocks) - 1:
                q = slice(h.shape[1] - per, h.shape[1] - acts)
                if self.flex:
                    layout = Frames(per, tokens, last, q.start)
                    bias = layout, frame_mask(layout, tokens, h.shape[1], h.device)
                else:
                    bias = block.frame_bias(frame[-1:], frame, is_act[:tokens], is_act)
                h = block(h[:, q], h, bias)
            elif self.flex:
                layout = Frames(per, tokens, last)
                h = block(h, h, (layout, frame_mask(layout, h.shape[1], h.shape[1], h.device)))
            else:
                h = block(h, h, block.frame_bias(frame, frame, is_act, is_act))
        if newest_only:
            return self.lnf(h).view(len(frames), 1, tokens, self.d)
        return self.lnf(h).view(len(frames), count, per, self.d)[:, :, :tokens]

    @property
    def record(self) -> int:
        """What's kept of a frame: its latents, then its action tokens, flat."""
        return self.encoder.tokens * self.encoder.latent + ACTION_TOKENS * self.d

    def _unpack(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cut = self.encoder.tokens * self.encoder.latent
        lead = history.shape[:-1]
        return history[..., :cut].reshape(*lead, self.encoder.tokens, self.encoder.latent), history[..., cut:].reshape(*lead, ACTION_TOKENS, self.d)

    def initial_state(self, seats: int, regions: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """What a seat keeps between decisions, newest first: the `reach` frames before (`record`), and which of them there have been."""
        return torch.zeros(seats, self.reach, self.record, device=device), torch.zeros(seats, self.reach, device=device)

    def decide(self, b, history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        """A decision while playing.  Returns what its actions read (`Context`), what to keep for next time, and what training will need
        of this decision.  The newest frame's action tokens in what's kept are empty until `acted`.  z: B x Z_DIM, or None (unknown)."""
        now = self.encoder(b).float()
        latents, acted = self._unpack(history)
        frames = torch.cat((latents.flip(1), now[:, None]), 1)
        acts = torch.cat((acted.flip(1), torch.zeros_like(acted[:, :1])), 1)
        present = torch.cat((seen.flip(1), torch.ones_like(seen[:, :1])), 1) > 0.5
        ctx = self.context(b, self.sequence(frames, present, acts, newest_only=True, z=z)[:, -1])
        newest = torch.cat((now.flatten(1), torch.zeros(len(now), ACTION_TOKENS * self.d, device=now.device)), 1)
        kept = torch.cat((newest[:, None], history[:, :-1]), 1), torch.cat((torch.ones_like(seen[:, :1]), seen[:, :-1]), 1)
        return ctx, *kept, {"latents": now}

    def acted(self, b, a: pm.Actions, history: torch.Tensor) -> torch.Tensor:
        """What `decide` kept, with what was just done filled in as the newest frame's action tokens."""
        out = history.clone()
        out[:, 0, self.encoder.tokens * self.encoder.latent:] = self.action_tokens(b, a).float().flatten(1)
        return out

    def segments(self, history: torch.Tensor, seen: torch.Tensor, latents: torch.Tensor, lengths: list[int], acted: torch.Tensor,
                 z: torch.Tensor | None = None) -> torch.Tensor:
        """Training's: segments of decisions, each from the state kept at its start, in one forward pass.
        history, seen: B x reach (x ...), as `decide` keeps them.  latents: B x T x tokens x latent, each decision's own, padded at the end.
        acted: B x T x ACTION_TOKENS x D, what was done at each (`action_tokens`, from the actions taken).
        Returns sum(lengths) x tokens x D: what each decision's frame holds, segment by segment."""
        before, before_acted = self._unpack(history)
        frames = torch.cat((before.flip(1), latents), 1)
        acts = torch.cat((before_acted.flip(1).to(acted.dtype), acted), 1)
        present = torch.cat((seen.flip(1) > 0.5, torch.ones(latents.shape[:2], dtype=torch.bool, device=latents.device)), 1)
        out = self.sequence(frames, present, acts, z=z)[:, self.reach:]
        return torch.cat([out[i, :n] for i, n in enumerate(lengths)])

    def segment_actions(self, b, a: pm.Actions, lengths: list[int]) -> torch.Tensor:
        """`action_tokens` for decisions in segment order, laid out as `segments` takes them: B x T x ACTION_TOKENS x D."""
        flat = self.action_tokens(b, a)
        out = flat.new_zeros(len(lengths), max(lengths), ACTION_TOKENS, self.d)
        at = 0
        for i, n in enumerate(lengths):
            out[i, :n] = flat[at:at + n]
            at += n
        return out

    # What was done ----------------------------------------------------------------------------------------------------------------------------------

    def action_vectors(self, b, ctx: Context | None, a: pm.Actions) -> torch.Tensor:
        """B x K x D: each action as a vector, of who it picked and what it told them.  Those that picked none are no action: their
        vectors are there, and the readers leave them out."""
        hu, keys, reg_h, site_h, feat_h = (ctx.hu, ctx.keys, ctx.reg_h, ctx.site_h, ctx.feat_h) if ctx is not None else self._raw(b)
        at = a.anchor.clamp_min(0)
        e = self.act_kind(a.kind.clamp(0, rf.KIND_COUNT - 1)) + self.act_unit(torch.gather(hu, 1, at[..., None].expand(-1, -1, hu.shape[-1])))
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, self.key))
        enemy = torch.gather(enemy_h, 1, a.enemy.clamp_min(0)[..., None].expand(-1, -1, self.key))
        e = e + self.act_enemy(enemy) * (a.enemy >= 0)[..., None]
        region = torch.gather(reg_h, 1, a.region.clamp_min(0)[..., None].expand(-1, -1, self.key))
        e = e + self.act_region(region) * (a.region >= 0)[..., None]
        e = e + self.act_fine((a.fine + 1).clamp(0, fz.FINE_OPTIONS))
        made = self.act_type(self.produce_stats[a.produce.clamp_min(0)].to(e.dtype))
        e = e + made * (a.produce >= 0)[..., None]
        e = e + self.act_group((a.group + 1).clamp_min(0))
        e = e + self.act_roles(pm.role_bits(a.roles, self.act_roles.in_features).to(e.dtype)) * (a.group >= 0)[..., None]
        # `place` is a BUILD's place or a RECLAIM's feature token: each gathers only where it's its kind's.
        builds, reclaims = (a.kind == rf.BUILD) & (a.place >= 0), (a.kind == rf.RECLAIM) & (a.place >= 0)
        if b.places is not None:
            place_h = self._places(b, keys, site_h)
            place = torch.gather(place_h, 1, torch.where(builds, a.place, 0)[..., None].expand(-1, -1, self.key))
            e = e + self.act_place(place) * builds[..., None]
        wreck = torch.gather(feat_h, 1, torch.where(reclaims, a.place, 0)[..., None].expand(-1, -1, self.key))
        e = e + self.act_wreck(wreck.to(e.dtype)) * reclaims[..., None]
        e = e + self.act_offset((a.offset + 1).clamp(0, fz.PLACE_OFFSETS))
        return self.act_out(F.gelu(e, approximate="tanh"))

    def action_tokens(self, b, a: pm.Actions, ctx: Context | None = None) -> torch.Tensor:
        """B x ACTION_TOKENS x D: what each decision's actions were, as `sequence` reads it.  Done is nothing.  Any number of actions,
        in any order: the tokens read them as a set."""
        e = self.action_vectors(b, ctx, a)
        acting = a.anchor >= 0
        tokens = torch.cat((self.act_nothing.to(e.dtype).expand(len(e), 1, self.d), e), 1)
        mask = torch.cat((torch.ones_like(acting[:, :1]), acting), 1)
        queries = self.act_queries.to(e.dtype) + self.act_count(torch.log1p(acting.sum(-1, keepdim=True).float()))[:, None].to(e.dtype)
        return self.act_read(queries, tokens, mask)

    # Units ------------------------------------------------------------------------------------------------------------------------------------------

    def _raw(self, b) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """From each token's own features, before any reading: units' embeddings, their keys, the regions' keys, the sites' and the
        feature tokens'."""
        place = b.unit_x[..., UNIT_PLACE[0]:UNIT_PLACE[1]]
        code = self.code(place)
        raw = torch.cat((b.unit_x, code, self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(raw), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        centre = self.code(b.reg_x[..., REGION_PLACE[0]:REGION_PLACE[1]])
        hr = self.reg_in(torch.cat((b.reg_x, centre), -1))
        at = self.code(b.site_x[..., SITE_PLACE[0]:SITE_PLACE[1]])
        hs = self.site_in(torch.cat((b.site_x, at), -1))
        lies = self.code(b.feat_x[..., FEAT_PLACE[0]:FEAT_PLACE[1]])
        hf = self.feat_in(torch.cat((b.feat_x, lies), -1))
        return (hu, torch.cat((self.key_unit(hu), code.to(hu.dtype)), -1), torch.cat((self.key_region(hr), centre.to(hr.dtype)), -1),
                torch.cat((self.key_site(hs), at.to(hs.dtype)), -1), torch.cat((self.key_feat(hf), lies.to(hf.dtype)), -1))

    def _places(self, b, keys: torch.Tensor, site_h: torch.Tensor) -> torch.Tensor:
        """B x P x KEY: the keys of where a BUILD can be (`Batch.places`): units of ours, then sites."""
        at = b.places[:, :, None].expand(-1, -1, self.key)
        unit = torch.gather(keys, 1, at.clamp_min(0))
        site = torch.gather(site_h.to(keys.dtype), 1, (-1 - at).clamp_min(0))
        return torch.where(at >= 0, unit, site)

    def context(self, b, now: torch.Tensor) -> Context:
        """now: B x tokens x D, what each decision's frame holds."""
        hu, keys, reg_h, site_h, feat_h = self._raw(b)
        now = now.to(hu.dtype)
        units = b.ctrl_mask.shape[1]
        everyone = torch.ones(now.shape[:2], dtype=torch.bool, device=now.device)
        read = self.unit_read(hu[:, :units], now, everyone)
        anchor_keys = torch.cat((self.key_anchor(read), keys[:, :units, self.d:]), -1)
        return Context(now, hu, keys, reg_h, site_h, feat_h, anchor_keys, self.head_ordered(read)[..., 0].float())

    # Actions, one after another -------------------------------------------------------------------------------------------------------------------

    def plan(self, ctx: Context, vectors: torch.Tensor) -> torch.Tensor:
        """vectors: B x k x D, the actions chosen so far (`action_vectors`).  Returns B x (k + 1) x D: what each of the slots up to the
        next reads, each from the frame and the actions before it."""
        count = vectors.shape[1] + 1
        x = torch.cat((self.plan_start.to(vectors.dtype).expand(len(vectors), 1, self.d), vectors), 1) + self.plan_slots[:count].to(vectors.dtype)
        for block in self.plan_blocks:
            x = block(x, ctx.now.to(x.dtype))
        return self.plan_ln(x)

    def anchor_logits(self, b, ctx: Context, h: torch.Tensor, taken: torch.Tensor) -> torch.Tensor:
        """h: B x K x D, slots from `plan`; taken: B x K x C, the units actions before each picked.  Returns float32 B x K x (1 + C):
        done, then each controllable unit, those that may not be picked (nothing to tell them, or picked already) taken out."""
        score = (self.q_anchor(h) @ ctx.anchor_keys.transpose(1, 2)).float() / math.sqrt(self.d)
        score = score.masked_fill(~(pm.anchors_legal(b)[:, None] & ~taken), pm.ILLEGAL)
        return torch.cat((self.head_done(h).float(), score), -1)

    def reading(self, ctx: Context, anchor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """B x K x D: each action's anchor reads the frame, as itself and with what the action is about (h, its slot)."""
        hu = torch.gather(ctx.hu, 1, anchor.clamp_min(0)[..., None].expand(-1, -1, ctx.hu.shape[-1]))
        everyone = torch.ones(ctx.now.shape[:2], dtype=torch.bool, device=hu.device)
        return self.read_unit(hu + self.slot_unit(h).to(hu.dtype), ctx.now.to(hu.dtype), everyone)

    def _score(self, query: torch.Tensor, gain: torch.Tensor, own: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scaled = gain.repeat(1, 1, 2) * own.to(gain.dtype)
        return (torch.cat((query, scaled), -1) @ keys.transpose(1, 2)).float() / math.sqrt(self.d)

    def heads(self, b, me: torch.Tensor, ctx: Context):
        """For each action, of its anchor: float32 logits with the illegal choices taken out: kind, region, enemy per pointer kind, the
        enemies' keys, what to produce or build, a BUILD's place, the places' keys, and a RECLAIM's feature token.  b is as the anchors see it
        (policy_model.per_action), me what they read (B x K x D).  Which regions are legal depends on the kind:
        policy_model.region_legal."""
        keys, reg_h = ctx.keys, ctx.reg_h
        count = me.shape[1]
        own = self.code(b.unit_x[:, :count, UNIT_PLACE[0]:UNIT_PLACE[1]])
        kind = self.head_kind(me).float().masked_fill(~b.kind_mask, pm.ILLEGAL)
        region = self._score(self.q_region(me), self.gain_region(me), own, reg_h)
        region = region - F.softplus(self.route_scale).float() * b.route_bias
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, self.key))
        enemy = self._score(self.q_enemy(me), self.gain_enemy(me), own, enemy_h)
        enemy = enemy[:, None].expand(-1, b.ptr_enemy.shape[1], -1, -1).masked_fill(~b.ptr_enemy, pm.ILLEGAL)
        produce = (self.q_type(me) @ self.type_key(self.produce_stats).T).float() / math.sqrt(self.d)
        place_h = self._places(b, keys, ctx.site_h)
        place = self._score(self.q_place(me), self.gain_place(me), own, place_h)
        place = place.masked_fill(~b.place_mask[:, None], pm.ILLEGAL)
        wreck = self._score(self.q_wreck(me), self.gain_wreck(me), own, ctx.feat_h).masked_fill(~b.feat_mask[:, None], pm.ILLEGAL)
        return kind, region, enemy, enemy_h, produce.masked_fill(~b.ptr_produce, pm.ILLEGAL), place, place_h, wreck

    def fine(self, me: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fine1(F.gelu(self.fine0(torch.cat((me[..., :self.d], target), -1)), approximate="tanh")).float()

    def offset(self, me: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        """B x K x PLACE_OFFSETS: where beside its chosen place (B x K x KEY) a BUILD goes."""
        return self.offset1(F.gelu(self.offset0(torch.cat((me[..., :self.d], place), -1)), approximate="tanh")).float()

    def _group_in(self, me: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return self.group0(me[..., :self.d]) + self.group_kind(kind.clamp(0, rf.KIND_COUNT - 1)).to(me.dtype)

    def group(self, me: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        """B x C x (1 + radii): alone, or a group order of each radius."""
        return self.group1(F.gelu(self._group_in(me, kind), approximate="tanh")).float()

    def roles(self, me: torch.Tensor, kind: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """B x C x roles: each role's logit of being in the group, given its radius (group: -1 for alone)."""
        h = self._group_in(me, kind) + self.group_radius((group + 1).clamp_min(0)).to(me.dtype)
        return self.roles1(F.gelu(h, approximate="tanh")).float()


def make(vocabulary: int, encoder_path: str | pathlib.Path, decide_every: int = 1, size: Size = SIZE) -> LatentActor:
    return LatentActor(vocabulary, LatentEncoder(load_vae(pathlib.Path(encoder_path))), window_frames(decide_every), size)


def compile_for_cloning(actor: LatentActor, mode: str, flex: bool = True) -> None:
    """torch.compile the time transformer (`sequence`), whose shapes in cloning are the same every step: samples x (reach + length)
    frames x tokens.  "reduce-overhead" replays it as CUDA graphs too.  With flex, its attention is flex attention, which skips the
    blocks of keys out of the window instead of scoring every pair.  Not for play or PPO, whose shapes change with every batch:
    compiling there gained nothing and once hung (03-training.md, "Dead ends")."""
    actor.flex = flex
    if mode != "none":
        actor.sequence = torch.compile(actor.sequence, mode=mode, dynamic=False)


ANSWERS = ("head_kind", "head_done", "head_ordered", "fine1", "offset1", "group1", "roles1", "z_in")   # z_in: zero at first, Muon would jump it


def answers(name: str, layer: nn.Linear) -> bool:
    """The layers that give the heads' logits, and those reading a single number: Muon leaves them to AdamW (frame_vae.Optimisers)."""
    return name in ANSWERS or layer.in_features == 1


def trainable_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [p for p in actor.parameters() if p.requires_grad]


def parameter_report(actor: LatentActor) -> dict[str, int]:
    trained = sum(p.numel() for p in trainable_parameters(actor))
    frozen = sum(p.numel() for p in actor.encoder.parameters())
    return {"trained": trained, "encoder": frozen, "total": trained + frozen}
