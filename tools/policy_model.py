"""What the actors share: featurizer output batched for PyTorch, the heads' sampling and scoring, and the critic.

Batches are padded per token group, so every decision in a batch has the layout

    units (Umax) | regions (Rmax) | sites (Smax) | features (Fmax) | global | memory state

`collate` builds them from featurizer output, and is numpy so a data worker can run it.  The last token is the first actor's
recurrent memory, which went with it on 2026-11-08 (git tag `rules-bot-final` has it); it stays as a token kind no actor reads, so
the layout and saved weights hold.  The actor is tools/policy_latent.py.
"""
from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot.policy import actions as act, featurize as fz, groups as gr, rawframe as rf  # noqa: E402

ILLEGAL = -1e9
NAME_DIM = 16
ID_DIM = 8                   # each of ally team and player
DONE_START = 3.0             # an untrained actor's first action is done more often than not, unless it has 20 units or more
ALONE_START = 2.0            # an untrained group head mostly sends orders alone
ROUTE_SCALE_START = -3.0     # softplus gives 0.05 nats per walking second: near regions first, and a far one is still sampled
HEAD_GAIN = 0.01             # an untrained head's weights: it says only what its bias says
KINDS_WITH_REGION, KINDS_WITH_FINE = fz.KINDS_WITH_REGION, fz.KINDS_WITH_FINE

JOINED_FIELDS = ("unit_x", "unit_name", "unit_ally", "unit_player", "hidden_ally", "hidden_player", "unit_mask", "reg_x", "reg_mask", "site_x", "site_mask", "feat_x", "feat_mask", "glob_x", "tok_kind", "ctrl_mask", "kind_mask", "ptr_enemy",
                 "enemy_tokens", "enemy_mask", "ptr_region", "ptr_jump", "ptr_wait", "ptr_produce", "route_bias", "hidden_x", "hidden_name", "hidden_mask",
                 "places", "place_mask")
HALF_FIELDS = ("unit_x", "reg_x", "route_bias", "hidden_x")    # 71% of a cloning sample's bytes, as float32
# What only the heads read (and the anchor pointer, ctrl_mask): the frame encoder and the action tokens don't.  Cloning sends these for
# the decisions it counts alone, as the critic's hidden units for none.
HEAD_FIELDS = ("ctrl_mask", "kind_mask", "ptr_enemy", "ptr_region", "ptr_jump", "ptr_wait", "ptr_produce", "route_bias")
CRITIC_FIELDS = ("hidden_x", "hidden_name", "hidden_mask", "hidden_ally", "hidden_player")


@dataclass
class Batch:
    """All arrays are numpy until `to(device)`."""
    unit_x: object
    unit_name: object
    unit_mask: object
    reg_x: object
    reg_mask: object
    site_x: object
    site_mask: object
    feat_x: object
    feat_mask: object
    glob_x: object
    tok_kind: object
    key_mask: object
    ctrl_mask: object       # B x C
    kind_mask: object       # B x C x KIND_COUNT
    ptr_enemy: object       # B x len(fz.ENEMY_POINTER_KINDS) x C x E
    enemy_tokens: object    # B x E
    enemy_mask: object      # B x E
    ptr_region: object      # B x C x R
    ptr_jump: object        # B x C x R
    ptr_wait: object        # B x C x R
    ptr_produce: object     # B x C x T
    route_bias: object      # B x C x R
    hidden_x: object        # B x H x UNIT_DIM, critic only
    hidden_name: object     # B x H
    hidden_mask: object     # B x H
    unit_ally: object       # B x U
    unit_player: object     # B x U
    hidden_ally: object     # B x H
    hidden_player: object   # B x H
    true_name: object = None    # B x U: before this game's names were dropped, for an encoder trained on them
    # Where a BUILD can be, B x P (padded where place_mask is false), in the order of a decision's place choice (actions.place_points):
    # beside each of these unit tokens of ours, then each site, as -1 - its index.  So a decision's place index means the same in any
    # batch, however its units and sites are padded.
    places: object = None
    place_mask: object = None

    def to(self, device: str, only: frozenset[str] | None = None) -> "Batch":
        """Numpy arrays or tensors (pinned ones copy without waiting); a field sent as float16 (`half`) arrives as float32."""
        out = {}
        for name, value in self.__dict__.items():
            if value is None or (only is not None and name not in only):
                out[name] = None
                continue
            t = (value if isinstance(value, torch.Tensor) else torch.from_numpy(value)).to(device, non_blocking=True)
            out[name] = t.float() if t.dtype == torch.float16 else t
        return Batch(**out)

    def half(self) -> "Batch":
        """The big float fields as float16, to send between processes at half the bytes (04-literature.md S2); `to` makes them
        float32 again.  Positions lose up to 1/4,096 of the map's half-width."""
        return replace(self, **{f: getattr(self, f).astype(np.float16) for f in HALF_FIELDS if getattr(self, f) is not None})

    def rows(self, index: torch.Tensor) -> "Batch":
        """These decisions of a batch on the device, padded as the batch is."""
        return Batch(**{name: None if v is None else v.index_select(0, index) for name, v in self.__dict__.items()})

    def without(self, names: tuple[str, ...]) -> "Batch":
        return replace(self, **dict.fromkeys(names))

    def split(self, rows: list[int]) -> tuple["Batch", "Batch"]:
        """For cloning, in numpy: every frame as the encoder and the action tokens read it, and the `HEAD_FIELDS` of these rows alone.
        `rows` of the first, on the device, `with_heads` the second, is what the heads read."""
        heads = Batch(**{f: getattr(self, f)[rows] if f in HEAD_FIELDS else None for f in Batch.__dataclass_fields__})
        return self.without(HEAD_FIELDS + CRITIC_FIELDS), heads

    def with_heads(self, heads: "Batch") -> "Batch":
        """These decisions with the `HEAD_FIELDS` of `heads` (the same decisions, on the device).  Their last dimension, regions or
        enemy tokens, is padded out to this batch's where theirs were fewer."""
        last = {"ptr_enemy": self.enemy_tokens.shape[1], "ptr_region": self.reg_x.shape[1], "ptr_jump": self.reg_x.shape[1],
                "ptr_wait": self.reg_x.shape[1], "route_bias": self.reg_x.shape[1]}
        out = {}
        for name in HEAD_FIELDS:
            v = getattr(heads, name)
            width = last.get(name, v.shape[-1])
            if width != v.shape[-1]:
                wide = v.new_zeros(*v.shape[:-1], width)
                wide[..., :v.shape[-1]] = v
                v = wide
            out[name] = v
        return replace(self, **out)


WIDTHS = ("u", "r", "s", "f", "c", "e", "h", "p")      # a batch's padded widths: units, regions, sites, features, controllable, enemy
                                                        # tokens, hidden units, places


def widths(items: list[fz.Tensors]) -> dict[str, int]:
    """The widths `collate` pads these to, each at least 1."""
    return {"u": max(max(len(t.unit_x) for t in items), 1), "r": max(max(len(t.reg_x) for t in items), 1),
            "s": max(max(len(t.site_x) for t in items), 1), "f": max(max(len(t.feat_x) for t in items), 1),
            "c": max(max(t.controllable for t in items), 1), "e": max(max(len(t.enemy_tokens) for t in items), 1),
            "h": max(max(len(t.hidden_x) for t in items), 1), "p": max(max(len(t.place_tokens) + len(t.site_x) for t in items), 1)}


def collate(items: list[fz.Tensors], regions: int = 0, sizes: dict[str, int] | None = None, rows: int = 0) -> Batch:
    """regions: pad to at least this many, so batches of one map's decisions line up.  sizes: pad each group to at least these
    `WIDTHS`; rows: pad to this many decisions, the rest no frame at all (masked out but for the global token).  Padding changes
    nothing a decision's own outputs are, as every reader masks it: a CUDA graph of a fixed shape plays any batch that fits
    (policy_graphs.py)."""
    b = max(len(items), rows)
    w = widths(items)
    w["r"] = max(w["r"], regions)
    if sizes is not None:
        w = {k: max(v, sizes.get(k, 0)) for k, v in w.items()}
        w["u"] = max(w["u"], w["c"])        # the controllable units are the first unit tokens
    u, r, s, f, c, e, h, p = (w[k] for k in WIDTHS)
    n = u + r + s + f + 2
    unit_x = np.zeros((b, u, fz.UNIT_DIM), np.float32)
    unit_name = np.zeros((b, u), np.int64)
    unit_mask = np.zeros((b, u), bool)
    unit_ally = np.zeros((b, u), np.int64)
    unit_player = np.zeros((b, u), np.int64)
    reg_x = np.zeros((b, r, fz.REGION_DIM), np.float32)
    reg_mask = np.zeros((b, r), bool)
    site_x = np.zeros((b, s, fz.SITE_DIM), np.float32)
    site_mask = np.zeros((b, s), bool)
    feat_x = np.zeros((b, f, fz.FEATURE_DIM), np.float32)
    feat_mask = np.zeros((b, f), bool)
    glob_x = np.zeros((b, fz.GLOBAL_DIM), np.float32)
    tok_kind = np.zeros((b, n), np.int64)
    tok_kind[:, u:u + r] = fz.TK_REGION
    tok_kind[:, u + r:u + r + s] = fz.TK_SITE
    tok_kind[:, u + r + s:u + r + s + f] = fz.TK_FEATURE
    tok_kind[:, -2] = fz.TK_GLOBAL
    tok_kind[:, -1] = fz.TK_MEMORY_STATE
    true_name = np.zeros((b, u), np.int64)
    key_mask = np.zeros((b, n), bool)
    key_mask[:, -2:] = True
    ctrl_mask = np.zeros((b, c), bool)
    kind_mask = np.zeros((b, c, rf.KIND_COUNT), bool)
    ptr_enemy = np.zeros((b, len(fz.ENEMY_POINTER_KINDS), c, e), bool)
    places = np.zeros((b, p), np.int64)
    place_mask = np.zeros((b, p), bool)
    enemy_tokens = np.zeros((b, e), np.int64)
    enemy_mask = np.zeros((b, e), bool)
    ptr_region = np.zeros((b, c, r), bool)
    ptr_jump = np.zeros((b, c, r), bool)
    ptr_wait = np.zeros((b, c, r), bool)
    ptr_produce = np.zeros((b, c, items[0].ptr_produce.shape[1]), bool)
    route_bias = np.zeros((b, c, r), np.float32)
    hidden_x = np.zeros((b, h, fz.UNIT_DIM), np.float32)
    hidden_name = np.zeros((b, h), np.int64)
    hidden_mask = np.zeros((b, h), bool)
    hidden_ally = np.zeros((b, h), np.int64)
    hidden_player = np.zeros((b, h), np.int64)
    for i, t in enumerate(items):
        nu, nr, ns, nc, ne = len(t.unit_x), len(t.reg_x), len(t.site_x), t.controllable, len(t.enemy_tokens)
        unit_x[i, :nu], unit_name[i, :nu], unit_mask[i, :nu] = t.unit_x, t.unit_name, True
        true_name[i, :nu] = t.unit_name if t.true_name is None else t.true_name
        unit_ally[i, :nu], unit_player[i, :nu] = t.unit_ally, t.unit_player
        reg_x[i, :nr], reg_mask[i, :nr] = t.reg_x, True
        site_x[i, :ns], site_mask[i, :ns] = t.site_x, True
        nf = len(t.feat_x)
        feat_x[i, :nf], feat_mask[i, :nf] = t.feat_x, True
        glob_x[i] = t.glob_x
        tok_kind[i, :nu] = t.tok_kind[:nu]
        key_mask[i, np.concatenate((np.arange(nu), u + np.arange(nr), u + r + np.arange(ns), u + r + s + np.arange(nf), [u + r + s + f]))] = True
        ctrl_mask[i, :nc] = True
        kind_mask[i, :nc] = t.kind_mask
        if ne:
            ptr_enemy[i, :, :nc, :ne] = t.ptr_enemy
            enemy_tokens[i, :ne] = t.enemy_tokens
            enemy_mask[i, :ne] = True
        ptr_region[i, :nc, :nr] = t.ptr_region
        ptr_jump[i, :nc, :nr] = t.ptr_jump
        ptr_wait[i, :nc, :nr] = t.ptr_wait
        ptr_produce[i, :nc] = t.ptr_produce
        route_bias[i, :nc, :nr] = t.route_bias
        nh = len(t.hidden_x)
        hidden_x[i, :nh], hidden_name[i, :nh], hidden_mask[i, :nh] = t.hidden_x, t.hidden_name, True
        hidden_ally[i, :nh], hidden_player[i, :nh] = t.hidden_ally, t.hidden_player
        n_place = len(t.place_tokens) + ns
        places[i, :n_place], place_mask[i, :n_place] = np.concatenate((t.place_tokens, -1 - np.arange(ns))), True
    return Batch(unit_x, unit_name, unit_mask, reg_x, reg_mask, site_x, site_mask, feat_x, feat_mask, glob_x, tok_kind, key_mask,
                 ctrl_mask, kind_mask, ptr_enemy, enemy_tokens, enemy_mask, ptr_region, ptr_jump, ptr_wait, ptr_produce, route_bias, hidden_x, hidden_name, hidden_mask,
                 unit_ally, unit_player, hidden_ally, hidden_player, true_name, places=places, place_mask=place_mask)


def stage(arrays: list[np.ndarray], device: str) -> list[torch.Tensor]:
    """These arrays on the device, in one copy: packed into one pinned buffer, sent without waiting, and viewed there.  A copy from
    pageable memory waits for the GPU's queue, so a step's hundreds of small copies made the host wait at each (2026-11-09)."""
    if not str(device).startswith("cuda"):
        return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]
    offsets, at = [], 0
    for a in arrays:
        offsets.append(at)
        at += -(-a.nbytes // 16) * 16       # each starts on 16 bytes, so that any dtype can view it
    host = torch.empty(max(at, 16), dtype=torch.uint8, pin_memory=True)
    flat = host.numpy()
    for a, o in zip(arrays, offsets):
        flat[o:o + a.nbytes] = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    there = host.to(device, non_blocking=True)
    return [there[o:o + a.nbytes].view(torch.from_numpy(a[:0].reshape(-1)).dtype).view(a.shape) for a, o in zip(arrays, offsets)]


def join(batches: list[Batch], device: str | None = None) -> Batch:
    """Every decision of these batches as one.  Each token group is padded on its own, so the groups still line up.  True names
    (what the frame encoder reads, apart from the names the actor may read) are kept if every batch has them.

    device: joined there, as tensors: each batch's own arrays are copied over, and the padding between them is made there.  A
    cloning step's frames padded to its biggest on the host were ~180 MB, mostly padding, zeroed and copied (04-literature.md S2)."""
    out: dict[str, object] = {name: None for name in Batch.__dataclass_fields__}
    names = all(getattr(b, "true_name", None) is not None for b in batches)
    fields = {}
    for name in JOINED_FIELDS + (("true_name",) if names else ()):
        parts = [getattr(b, name) for b in batches]
        if all(p is None for p in parts):       # a field left out (`HEAD_FIELDS`, `CRITIC_FIELDS`)
            continue
        if name == "tok_kind":      # the critic reads the units' only
            parts = [p[:, :b.unit_x.shape[1]] for p, b in zip(parts, batches)]
        fields[name] = parts
    if device is not None:      # every field's parts in one copy
        sent = iter(stage([p for parts in fields.values() for p in parts], device))
    for name, parts in fields.items():
        shape = tuple(int(s) for s in (sum(len(p) for p in parts), *np.max([p.shape[1:] for p in parts], axis=0)))
        if device is not None:
            whole = torch.zeros(shape, dtype=torch.from_numpy(parts[0][:0]).dtype, device=device)
            parts = [next(sent) for _ in parts]
        else:
            whole = np.zeros(shape, parts[0].dtype)
        at = 0
        for p in parts:
            whole[(slice(at, at + len(p)), *map(slice, p.shape[1:]))] = p
            at += len(p)
        out[name] = whole
    return Batch(**out)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """F.scaled_dot_product_attention, a head size that isn't a multiple of 8 padded with zeros on the GPU: PyTorch's efficient
    kernels need one, and the frame encoder's (144 wide in 4 heads: 36) took the math kernel, 5 times slower at a cloning pass's
    sizes (2026-11-11).  The zeros change no score, the scale is the head size's, and the result's padding is dropped."""
    size = q.shape[-1]
    if size % 8 == 0 or not q.is_cuda:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    pad = lambda t: F.pad(t, (0, -size % 8))  # noqa: E731
    return F.scaled_dot_product_attention(pad(q), pad(k), pad(v), attn_mask=mask, scale=1.0 / math.sqrt(size))[..., :size]


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, h: torch.Tensor, bias: torch.Tensor | None, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).chunk(3, -1)
        shape = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        if bias is None:
            mask = key_mask[:, None, None, :]
        else:
            mask = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        o = attention(shape(q), shape(k), shape(v), mask)
        h = h + self.out(o.transpose(1, 2).reshape(b, n, d))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h)), approximate="tanh"))


class CrossBlock(nn.Module):
    """Queries reading tokens."""

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

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, q_count, d = queries.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(self.ln_t(tokens)).chunk(2, -1)
        o = attention(split(self.q(self.ln_q(queries))), split(k), split(v), key_mask[:, None, None, :])
        queries = queries + self.out(o.transpose(1, 2).reshape(b, q_count, d))
        return queries + self.ff2(F.gelu(self.ff1(self.ln2(queries)), approximate="tanh"))


assert rf.KIND_COUNT <= 31, "kind_in's mask is an int32's bits"


def kind_in(kind: torch.Tensor, kinds: tuple[int, ...]) -> torch.Tensor:
    """Whether each kind is one of these: a bit of a mask, nothing copied to the device.  `torch.isin` against a tensor made each
    call copied it there each time, and each copy made the host wait for the GPU; a table kept on the device was built on the host
    inside a compiled graph (`--compile dynamic`, 2026-11-11)."""
    mask = sum(1 << k for k in set(kinds))
    return ((mask >> kind.clamp(0, rf.KIND_COUNT - 1).to(torch.int32)) & 1).bool() & (kind >= 0)


def region_legal(b: Batch, scores: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """The region scores, with what each unit's kind can't go to taken out."""
    reach = kind_in(kind, fz.REACH_KINDS)[:, :, None]
    legal = torch.where(reach, b.ptr_jump, torch.where((kind == rf.WAIT_AT_BEACON)[:, :, None], b.ptr_wait, b.ptr_region))
    return scores.masked_fill(~legal, ILLEGAL)


def _masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return -(p * logp).sum(-1)


def _pick(logits: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, -1).gather(-1, choice[..., None])[..., 0]


TEMPERATURE = 1.0       # what sampling divides the logits by: below 1 plays surer.  Only for testing a policy (policy_clone_eval):
                        # the log-probabilities stay the untempered ones, which PPO needs


def _sample(logits: torch.Tensor) -> torch.Tensor:
    # Gumbel: no multinomial, whose checks dislike -1e9 under bf16 rounding.
    u = torch.rand_like(logits).clamp_min(1e-9)
    return (logits / TEMPERATURE - torch.log(-torch.log(u))).argmax(-1)


class Critic(nn.Module):
    """A separate encoder over the privileged frame.  Never in the bot.  Learnt queries read every token, as the frame encoder's do
    (frame_model.FrameEncoder), each round a reading and then the queries attending to each other, and the value is read from the
    global token, which is one of the queries.  So its cost grows with the tokens, not their square: to 2026-11-12 it was
    self-attention over every token, about 5M pairs a layer at the caps (the user).  Before the queries read them, each token has a
    feed-forward of its own.  Its size grew with the actor's (the user, 2026-11-09): it was width 128, 2 layers of 4 heads."""
    D = 256
    LAYERS = 4
    HEADS = 8
    QUERIES = 32

    def __init__(self, vocabulary: int, d: int = D, layers: int = LAYERS, heads: int = HEADS, bins: int = 0, span: float = 3.0,
                 queries: int = QUERIES):
        """bins: above 0, the value is a distribution over that many values evenly from -span to span, trained by HL-Gauss
        (`hl_gauss`; 04-literature.md P7), and the value is its mean; else one number.  layers: rounds of reading; queries: besides
        the global token's."""
        super().__init__()
        self.size = {"d": d, "layers": layers, "heads": heads, "queries": queries, **({"bins": bins, "span": span} if bins else {})}
        self.bins = bins
        self.name_embed = nn.Embedding(vocabulary, 8)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, d)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, 4)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, 4)
        self.unit_in = nn.Linear(fz.UNIT_DIM + 16, d)
        self.reg_in = nn.Linear(fz.REGION_DIM, d)
        self.site_in = nn.Linear(fz.SITE_DIM, d)
        self.feat_in = nn.Linear(fz.FEATURE_DIM, d)
        self.glob_in = nn.Linear(fz.GLOBAL_DIM, d)
        self.token_ln = nn.LayerNorm(d)
        self.token_ff1 = nn.Linear(d, 2 * d)
        self.token_ff2 = nn.Linear(2 * d, d)
        self.queries = nn.Parameter(torch.randn(queries, d) / math.sqrt(d))
        self.reads = nn.ModuleList(CrossBlock(d, heads) for _ in range(layers))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.value = nn.Linear(d, bins or 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        if bins:
            self.register_buffer("centres", torch.linspace(-span, span, bins), persistent=False)

    def hl_gauss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """B: the cross-entropy of the value's distribution (B x bins logits) against a Gaussian about each target, its standard
        deviation 0.75 of a bin, as each bin holds it (Farebrother et al. 2024)."""
        width = float(self.centres[1] - self.centres[0])
        edges = torch.cat((self.centres - width / 2, self.centres[-1:] + width / 2))
        t = target.float().clamp(float(edges[0]), float(edges[-1]))[:, None]
        cdf = torch.special.ndtr((edges[None] - t) / (0.75 * width))
        p = cdf[:, 1:] - cdf[:, :-1]
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return -(p * F.log_softmax(logits.float(), -1)).sum(-1)

    def forward(self, b: Batch, logits: bool = False) -> torch.Tensor:
        """B values; with logits (and bins), B x bins logits of the value's distribution."""
        hu = self.unit_in(torch.cat((b.unit_x, self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1))
        hh = self.unit_in(torch.cat((b.hidden_x, self.name_embed(b.hidden_name), self.ally_embed(b.hidden_ally), self.player_embed(b.hidden_player)), -1)) + self.kind_embed.weight[fz.TK_HIDDEN]
        hr = self.reg_in(b.reg_x) + self.kind_embed.weight[fz.TK_REGION]
        hs = self.site_in(b.site_x) + self.kind_embed.weight[fz.TK_SITE]
        hf = self.feat_in(b.feat_x) + self.kind_embed.weight[fz.TK_FEATURE]
        hg = self.glob_in(b.glob_x)[:, None]
        hu = hu + self.kind_embed(b.tok_kind[:, :hu.shape[1]])
        tokens = torch.cat((hg, hu, hh, hr, hs, hf), 1)
        tokens = tokens + self.token_ff2(F.gelu(self.token_ff1(self.token_ln(tokens)), approximate="tanh"))
        mask = torch.cat((torch.ones_like(b.unit_mask[:, :1]), b.unit_mask, b.hidden_mask, b.reg_mask, b.site_mask, b.feat_mask), 1)
        h = torch.cat((tokens[:, :1], self.queries.to(tokens.dtype).expand(len(tokens), -1, -1)), 1)
        everyone = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for read, block in zip(self.reads, self.blocks):
            h = block(read(h, tokens, mask), None, everyone)
        out = self.value(self.lnf(h[:, 0])).float()
        if not self.bins:
            return out[:, 0]
        return out if logits else (out.softmax(-1) * self.centres).sum(-1)


@dataclass
class Actions:
    """A decision's actions, B x K: each picks a controllable unit, its anchor (-1 for done, and for every slot after it), and
    an order for it.  The rest is as the anchor would choose for itself: -1 where a field isn't used, kind KEEP where there's no
    action.  logp: each action's, 0 after done."""
    anchor: torch.Tensor
    kind: torch.Tensor
    enemy: torch.Tensor
    region: torch.Tensor
    fine: torch.Tensor
    produce: torch.Tensor
    group: torch.Tensor     # index into groups.GROUP_RADII, -1 alone or unused
    roles: torch.Tensor     # the group's roles as bits, 0 if none
    place: torch.Tensor     # a BUILD's: what it's beside (Batch.places); a RECLAIM's: which feature token; -1 unused
    offset: torch.Tensor    # a BUILD's: where beside it, of fz.PLACE_OFFSETS; -1 unused
    logp: torch.Tensor


def actions_from(fields: dict, device: str) -> Actions:
    """From numpy arrays of `actions.SLOT_FIELDS` (and logp, or zeros), each B x K."""
    sent = stage([np.asarray(fields[f], dtype=np.int64) for f in act.SLOT_FIELDS], device)     # in one copy
    t = dict(zip(act.SLOT_FIELDS, sent))
    logp = torch.as_tensor(np.asarray(fields["logp"]), dtype=torch.float32, device=device) if "logp" in fields else torch.zeros(t["anchor"].shape, device=device)
    return Actions(**t, logp=logp)


def chosen(a: Actions) -> torch.Tensor:
    """B x K: the slots that were chosen: every one up to and including done."""
    return torch.cat((torch.ones_like(a.anchor[:, :1], dtype=torch.bool), a.anchor[:, :-1] >= 0), 1)


def role_bits(roles: torch.Tensor, count: int) -> torch.Tensor:
    """... to ... x count bools."""
    return ((roles[..., None] >> torch.arange(count, device=roles.device)) & 1).bool()


def _bits_logp(logits: torch.Tensor, bits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Each role in or out on its own: the log-probability of these, over the legal roles."""
    return (torch.where(bits, F.logsigmoid(logits), F.logsigmoid(-logits)) * legal).sum(-1)


def _bits_entropy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Mean over the legal roles of each one's entropy."""
    p = torch.sigmoid(logits)
    e = -(p * F.logsigmoid(logits) + (1.0 - p) * F.logsigmoid(-logits))
    return (e * legal).sum(-1) / legal.sum().clamp_min(1)


def _groupable(kind: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
    return kind_in(kind, gr.GROUP_KINDS) & on


def acting_kinds(kind_mask: torch.Tensor) -> torch.Tensor:
    """What an action may tell a unit: anything legal but keep."""
    out = kind_mask.clone()
    out[..., rf.KEEP] = False
    return out


def anchors_legal(b: Batch) -> torch.Tensor:
    """B x C: the units an action may pick, those that may be told something."""
    return b.ctrl_mask & acting_kinds(b.kind_mask).any(-1)


def per_action(b: Batch, anchor: torch.Tensor) -> Batch:
    """The batch as each action's anchor sees it: the per-unit masks of the unit each of the K picked, as if they were K units.
    Only what the heads read is taken; the rest is the batch's own.  An action that picked none (done) is no unit."""
    i = anchor.clamp_min(0)

    def take(x: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, i.view(*i.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))

    enemy = torch.gather(b.ptr_enemy, 2, i[:, None, :, None].expand(-1, b.ptr_enemy.shape[1], -1, b.ptr_enemy.shape[3]))
    return replace(b, ctrl_mask=anchor >= 0, kind_mask=acting_kinds(take(b.kind_mask)), ptr_enemy=enemy, ptr_region=take(b.ptr_region),
                   ptr_jump=take(b.ptr_jump), ptr_wait=take(b.ptr_wait), ptr_produce=take(b.ptr_produce), route_bias=take(b.route_bias),
                   unit_x=take(b.unit_x))


def _choices(actor: nn.Module, v: Batch, me: torch.Tensor, ctx, given: Actions | None = None):
    """For each action, given its anchor (`per_action`'s v, and what the anchor read, me): a kind, and then what the kind needs.
    Sampled, or with `given`, those scored.  Returns the choices, each action's log-probability of them, and each head's entropy with
    the actions it applies to.  ctx: the decision's `Context` (keys, reg_h, site_h)."""
    def choose(logits: torch.Tensor, field: str) -> torch.Tensor:
        return _sample(logits) if given is None else getattr(given, field).clamp_min(0)

    def choose_where(logits: torch.Tensor, field: str, uses: torch.Tensor) -> torch.Tensor:
        """A field that means something else for another kind (place: a BUILD's place, a RECLAIM's feature token): given, it's
        read only where this kind uses it, so that another kind's index can't run past these logits."""
        return _sample(logits) if given is None else torch.where(uses, getattr(given, field), 0).clamp_min(0)

    reg_h = ctx.reg_h
    kind_l, region_l, enemy_l, enemy_h, produce_l, place_l, place_h, wreck_l = actor.heads(v, me, ctx)
    on = v.ctrl_mask
    kind = choose(kind_l, "kind")
    logp = _pick(kind_l, kind)
    ent = {"kind": (_masked_entropy(kind_l), on & (v.kind_mask.sum(-1) > 1))}
    uses_region = kind_in(kind, KINDS_WITH_REGION) & on
    region_l = region_legal(v, region_l, kind)
    region = choose(region_l, "region")
    logp = logp + uses_region * _pick(region_l, region)
    ent["region"] = (_masked_entropy(region_l), uses_region)
    width = reg_h.shape[-1]
    target = torch.gather(reg_h, 1, region[:, :, None].expand(-1, -1, width))
    # The enemy head per kind: only a kind with an enemy pointer uses one.  Every pointer kind is scored whether any action has it or
    # not: asking first made the host wait for the GPU, four times a call.  A row of another kind counts for nothing (here is 0, and
    # its logits are finite where masked).
    enemy = torch.full_like(kind, -1)
    enemy_entropy = torch.zeros_like(logp)
    for slot, k in enumerate(fz.ENEMY_POINTER_KINDS):
        here = (kind == k) & on
        choice = choose(enemy_l[:, slot], "enemy")
        logp = logp + here * _pick(enemy_l[:, slot], choice)
        enemy = torch.where(here, choice, enemy)
        enemy_entropy = torch.where(here, _masked_entropy(enemy_l[:, slot]), enemy_entropy)
        target = torch.where(here[:, :, None], torch.gather(enemy_h, 1, choice[:, :, None].expand(-1, -1, width)), target)
    ent["enemy"] = (enemy_entropy, enemy >= 0)
    fine_l = actor.fine(me, target)
    fine = choose(fine_l, "fine")
    uses_fine = kind_in(kind, KINDS_WITH_FINE) & on
    logp = logp + uses_fine * _pick(fine_l, fine)
    ent["fine"] = (_masked_entropy(fine_l), uses_fine)
    produce = choose(produce_l, "produce")
    uses_produce = ((kind == rf.PRODUCE) | (kind == rf.BUILD)) & on      # what a factory makes, or a constructor starts
    logp = logp + uses_produce * _pick(produce_l, produce)
    ent["produce"] = (_masked_entropy(produce_l), uses_produce)
    # A BUILD's place: beside which of our units or sites, then where beside it.
    builds = (kind == rf.BUILD) & on
    place = choose_where(place_l, "place", builds)
    logp = logp + builds * _pick(place_l, place)
    ent["place"] = (_masked_entropy(place_l), builds)
    offset_l = actor.offset(me, torch.gather(place_h, 1, place[:, :, None].expand(-1, -1, place_h.shape[-1])))
    offset = choose_where(offset_l, "offset", builds)
    logp = logp + builds * _pick(offset_l, offset)
    ent["offset"] = (_masked_entropy(offset_l), builds)
    # A RECLAIM's feature token, in the same field.
    reclaims = (kind == rf.RECLAIM) & on
    wreck = choose_where(wreck_l, "place", reclaims)
    logp = logp + reclaims * _pick(wreck_l, wreck)
    ent["wreck"] = (_masked_entropy(wreck_l), reclaims)
    # Alone or as a group order, and then the group's roles.
    groupable = _groupable(kind, on)
    group_l = actor.group(me, kind)
    radius = _sample(group_l) if given is None else (given.group + 1).clamp_min(0)
    logp = logp + groupable * _pick(group_l, radius)
    ent["group"] = (_masked_entropy(group_l), groupable)
    group = torch.where(groupable, radius - 1, -1)
    roles_l = actor.roles(me, kind, group)
    legal = actor.choosable_roles
    if given is None:
        bits = (torch.rand_like(roles_l) < torch.sigmoid(roles_l)) & legal
    else:
        bits = role_bits(given.roles, roles_l.shape[-1])
    in_group = group >= 0
    logp = logp + in_group * _bits_logp(roles_l, bits, legal)
    ent["roles"] = (_bits_entropy(roles_l, legal), in_group)
    roles = torch.where(in_group, (bits.long() << torch.arange(bits.shape[-1], device=bits.device)).sum(-1), 0)
    out = {"kind": torch.where(on, kind, rf.KEEP), "enemy": enemy, "region": region.masked_fill(~uses_region, -1),
           "fine": fine.masked_fill(~uses_fine, -1), "produce": produce.masked_fill(~uses_produce, -1), "group": group, "roles": roles,
           "place": torch.where(builds, place, torch.where(reclaims, wreck, -1)), "offset": offset.masked_fill(~builds, -1)}
    return out, logp.masked_fill(~on, 0.0), ent


def _taken_before(anchor: torch.Tensor, units: int) -> torch.Tensor:
    """B x K x C: the units actions before each have picked.  Compared with each unit's index, not `F.one_hot`, whose count of
    classes is a number on the host: compiled, it made each count of units a graph of its own (2026-11-11)."""
    hot = (anchor[..., None] == torch.arange(units, device=anchor.device)) & (anchor >= 0)[..., None]
    return (torch.cumsum(hot.long(), 1) - hot.long()) > 0


@dataclass
class Sampling:
    """A decision's actions part-way through being chosen (`sample_steps`).  Each step writes its slot of each in place, so that
    the steps can be split among CUDA graphs that read and write the same memory (policy_graphs.py)."""
    out: dict[str, torch.Tensor]    # actions.SLOT_FIELDS, B x slots
    logp: torch.Tensor              # B x slots
    taken: torch.Tensor             # B x C: the units picked so far
    going: torch.Tensor             # B: not yet done
    vectors: torch.Tensor           # B x slots x D: the actions so far (`action_vectors`), the first k filled

    def actions(self) -> Actions:
        return Actions(**self.out, logp=self.logp)


def start_sampling(b: Batch, ctx, slots: int = act.SLOTS) -> Sampling:
    count, units = b.ctrl_mask.shape
    device = b.ctrl_mask.device
    return Sampling({f: torch.full((count, slots), act.UNSET[f], dtype=torch.long, device=device) for f in act.SLOT_FIELDS},
                    torch.zeros(count, slots, device=device), torch.zeros(count, units, dtype=torch.bool, device=device),
                    torch.ones(count, dtype=torch.bool, device=device), ctx.hu.new_zeros(count, slots, ctx.hu.shape[-1]))


def sample_steps(actor: nn.Module, b: Batch, ctx, s: Sampling, steps: range) -> None:
    """The actions of these slots, one after another, each reading those before it.  A decision already done stays done: past
    done a step changes nothing."""
    units = b.ctrl_mask.shape[1]
    for k in steps:
        h = actor.plan(ctx, s.vectors[:, :k])[:, -1:]
        anchor_l = actor.anchor_logits(b, ctx, h, s.taken[:, None])
        pick = _sample(anchor_l)
        anchor = torch.where(s.going[:, None], pick - 1, -1)
        lp = _pick(anchor_l, pick).masked_fill(~s.going[:, None], 0.0)
        choice, lp_rest, _ = _choices(actor, per_action(b, anchor), actor.reading(ctx, anchor, h), ctx)
        s.out["anchor"][:, k] = anchor[:, 0]
        for f, value in choice.items():
            s.out[f][:, k] = value[:, 0]
        s.logp[:, k] = (lp + lp_rest)[:, 0]
        this = Actions(anchor, **choice, logp=lp)
        s.vectors[:, k] = actor.action_vectors(b, ctx, this)[:, 0].to(s.vectors.dtype)
        s.taken |= F.one_hot(anchor[:, 0].clamp_min(0), units).bool() & (anchor >= 0)
        s.going &= anchor[:, 0] >= 0


def sample_actions(actor: nn.Module, b: Batch, ctx, slots: int = act.SLOTS, every_slot: bool = False) -> Actions:
    """A decision's actions, one after another, each reading those before it, until done or `slots` of them.

    every_slot: all `slots` steps, never asking the GPU whether every decision is done.  Past done a step changes nothing, so the
    actions are the same, given the same random numbers."""
    s = start_sampling(b, ctx, slots)
    for k in range(slots):
        sample_steps(actor, b, ctx, s, range(k, k + 1))
        if not every_slot and not s.going.any():
            break
    return s.actions()


def evaluate_actions(actor: nn.Module, b: Batch, ctx, a: Actions, labelled: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Log-probability of stored actions, B x K, 0 past done; and each head's entropy with the actions it applies to.  All the
    actions at once, each reading those before it, as when they were chosen.

    labelled: B x C, for cloning a decision whose actions came in no order of their own (a script's): an action's anchor then scores
    as any unit labelled and not yet picked, whichever is next in `a`, and done once they all are."""
    units = b.ctrl_mask.shape[1]
    vectors = actor.action_vectors(b, ctx, a)
    h = actor.plan(ctx, vectors[:, :-1])
    taken = _taken_before(a.anchor, units)
    anchor_l = actor.anchor_logits(b, ctx, h, taken)
    scores = F.log_softmax(anchor_l, -1)
    logp = scores.gather(-1, (a.anchor + 1)[..., None])[..., 0]
    if labelled is not None:
        remaining = labelled[:, None] & ~taken
        anywhere = torch.logsumexp(scores[..., 1:].masked_fill(~remaining, float("-inf")), -1)
        logp = torch.where(a.anchor >= 0, anywhere, logp)
    valid = chosen(a)
    ent = {"anchor": (_masked_entropy(anchor_l), valid)}
    _, rest, more = _choices(actor, per_action(b, a.anchor), actor.reading(ctx, a.anchor, h), ctx, given=a)
    ent.update(more)
    return (logp + rest).masked_fill(~valid, 0.0), ent


def teacher_logits(actor: nn.Module, b: Batch, ctx, a: Actions) -> tuple[torch.Tensor, torch.Tensor]:
    """The anchor logits (B x K x (1 + C)) and kind logits (B x K x KIND_COUNT) each of the actions had, given those before it, as
    `evaluate_actions` scores them: for measuring what cloning has learned."""
    h = actor.plan(ctx, actor.action_vectors(b, ctx, a)[:, :-1])
    anchor_l = actor.anchor_logits(b, ctx, h, _taken_before(a.anchor, b.ctrl_mask.shape[1]))
    return anchor_l, actor.heads(per_action(b, a.anchor), actor.reading(ctx, a.anchor, h), ctx)[0]


def each_unit_heads(actor: nn.Module, b: Batch, ctx):
    """The heads' logits for every controllable unit, as if each were the first action's anchor: B x C rows, as `heads` gives them.
    For looking at what the policy makes of each unit."""
    anchor = torch.arange(b.ctrl_mask.shape[1], device=b.ctrl_mask.device).expand(b.ctrl_mask.shape).masked_fill(~b.ctrl_mask, -1)
    h = actor.plan(ctx, ctx.hu.new_zeros(len(anchor), 0, ctx.hu.shape[-1])).expand(-1, anchor.shape[1], -1)
    return actor.heads(per_action(b, anchor), actor.reading(ctx, anchor, h), ctx)
