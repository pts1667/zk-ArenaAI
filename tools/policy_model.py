"""What the actors share: featurizer output batched for PyTorch, the heads' sampling and scoring, and the critic.

Batches are padded per token group, so every decision in a batch has the layout

    units (Umax) | regions (Rmax) | sites (Smax) | features (Fmax) | global | memory state

`collate` builds them from featurizer output, and is numpy so a data worker can run it.  The last token is the first actor's
recurrent memory, which went with it on 2026-11-08 (git tag `rules-bot-final` has it); it stays as a token kind no actor reads, so
the layout and saved weights hold.  The actor is tools/policy_latent.py.
"""
from __future__ import annotations

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
                 "enemy_tokens", "enemy_mask", "ptr_region", "ptr_jump", "ptr_wait", "ptr_produce", "route_bias", "hidden_x", "hidden_name", "hidden_mask")


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
    ptr_enemy: object       # B x 3 x C x E
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

    def to(self, device: str, only: frozenset[str] | None = None) -> "Batch":
        out = {}
        for name, value in self.__dict__.items():
            wanted = value is not None and (only is None or name in only)
            out[name] = torch.from_numpy(value).to(device, non_blocking=True) if wanted else None
        return Batch(**out)


def collate(items: list[fz.Tensors], regions: int = 0) -> Batch:
    """regions: pad to at least this many, so batches of one map's decisions line up."""
    b = len(items)
    u = max(max(len(t.unit_x) for t in items), 1)
    r = max(max(len(t.reg_x) for t in items), 1, regions)
    s = max(max(len(t.site_x) for t in items), 1)
    f = max(max(len(t.feat_x) for t in items), 1)
    c = max(max(t.controllable for t in items), 1)
    e = max(max(len(t.enemy_tokens) for t in items), 1)
    h = max(max(len(t.hidden_x) for t in items), 1)
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
    ptr_enemy = np.zeros((b, 3, c, e), bool)
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
    return Batch(unit_x, unit_name, unit_mask, reg_x, reg_mask, site_x, site_mask, feat_x, feat_mask, glob_x, tok_kind, key_mask,
                 ctrl_mask, kind_mask, ptr_enemy, enemy_tokens, enemy_mask, ptr_region, ptr_jump, ptr_wait, ptr_produce, route_bias, hidden_x, hidden_name, hidden_mask,
                 unit_ally, unit_player, hidden_ally, hidden_player, true_name)


def join(batches: list[Batch]) -> Batch:
    """Every decision of these batches as one.  Each token group is padded on its own, so the groups still line up."""
    out: dict[str, object] = {name: None for name in Batch.__dataclass_fields__}
    for name in JOINED_FIELDS:
        parts = [getattr(b, name) for b in batches]
        if name == "tok_kind":      # the critic reads the units' only
            parts = [p[:, :b.unit_x.shape[1]] for p, b in zip(parts, batches)]
        whole = np.zeros((sum(len(p) for p in parts), *np.max([p.shape[1:] for p in parts], axis=0)), parts[0].dtype)
        at = 0
        for p in parts:
            whole[(slice(at, at + len(p)), *map(slice, p.shape[1:]))] = p
            at += len(p)
        out[name] = whole
    return Batch(**out)


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
        o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), attn_mask=mask)
        h = h + self.out(o.transpose(1, 2).reshape(b, n, d))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h)), approximate="tanh"))


def region_legal(b: Batch, scores: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """The region scores, with what each unit's kind can't go to taken out."""
    reach = torch.isin(kind, torch.tensor((rf.JUMP, rf.THROW), device=kind.device))[:, :, None]
    legal = torch.where(reach, b.ptr_jump, torch.where((kind == rf.WAIT_AT_BEACON)[:, :, None], b.ptr_wait, b.ptr_region))
    return scores.masked_fill(~legal, ILLEGAL)


def _masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return -(p * logp).sum(-1)


def _pick(logits: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, -1).gather(-1, choice[..., None])[..., 0]


def _sample(logits: torch.Tensor) -> torch.Tensor:
    # Gumbel: no multinomial, whose checks dislike -1e9 under bf16 rounding.
    u = torch.rand_like(logits).clamp_min(1e-9)
    return (logits - torch.log(-torch.log(u))).argmax(-1)


class Critic(nn.Module):
    """A separate encoder over the privileged frame.  Never in the bot."""
    D = 128
    LAYERS = 2

    def __init__(self, vocabulary: int):
        super().__init__()
        d = self.D
        self.name_embed = nn.Embedding(vocabulary, 8)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, d)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, 4)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, 4)
        self.unit_in = nn.Linear(fz.UNIT_DIM + 16, d)
        self.reg_in = nn.Linear(fz.REGION_DIM, d)
        self.site_in = nn.Linear(fz.SITE_DIM, d)
        self.feat_in = nn.Linear(fz.FEATURE_DIM, d)
        self.glob_in = nn.Linear(fz.GLOBAL_DIM, d)
        self.blocks = nn.ModuleList(Block(d, 4) for _ in range(self.LAYERS))
        self.lnf = nn.LayerNorm(d)
        self.value = nn.Linear(d, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, b: Batch) -> torch.Tensor:
        hu = self.unit_in(torch.cat((b.unit_x, self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1))
        hh = self.unit_in(torch.cat((b.hidden_x, self.name_embed(b.hidden_name), self.ally_embed(b.hidden_ally), self.player_embed(b.hidden_player)), -1)) + self.kind_embed.weight[fz.TK_HIDDEN]
        hr = self.reg_in(b.reg_x) + self.kind_embed.weight[fz.TK_REGION]
        hs = self.site_in(b.site_x) + self.kind_embed.weight[fz.TK_SITE]
        hf = self.feat_in(b.feat_x) + self.kind_embed.weight[fz.TK_FEATURE]
        hg = self.glob_in(b.glob_x)[:, None]
        hu = hu + self.kind_embed(b.tok_kind[:, :hu.shape[1]])
        h = torch.cat((hg, hu, hh, hr, hs, hf), 1)
        mask = torch.cat((torch.ones_like(b.unit_mask[:, :1]), b.unit_mask, b.hidden_mask, b.reg_mask, b.site_mask, b.feat_mask), 1)
        for block in self.blocks:
            h = block(h, None, mask)
        return self.value(self.lnf(h[:, 0]))[:, 0].float()


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
    logp: torch.Tensor


def actions_from(fields: dict, device: str) -> Actions:
    """From numpy arrays of `actions.SLOT_FIELDS` (and logp, or zeros), each B x K."""
    t = {f: torch.as_tensor(np.asarray(fields[f]), dtype=torch.int64, device=device) for f in act.SLOT_FIELDS}
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
    return torch.isin(kind, torch.tensor(gr.GROUP_KINDS, device=kind.device)) & on


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


def _choices(actor: nn.Module, v: Batch, me: torch.Tensor, keys: torch.Tensor, reg_h: torch.Tensor, given: Actions | None = None):
    """For each action, given its anchor (`per_action`'s v, and what the anchor read, me): a kind, and then what the kind needs.
    Sampled, or with `given`, those scored.  Returns the choices, each action's log-probability of them, and each head's entropy with
    the actions it applies to."""
    def choose(logits: torch.Tensor, field: str) -> torch.Tensor:
        return _sample(logits) if given is None else getattr(given, field).clamp_min(0)

    kind_l, region_l, enemy_l, enemy_h, produce_l = actor.heads(v, me, keys, reg_h)
    on = v.ctrl_mask
    kind = choose(kind_l, "kind")
    logp = _pick(kind_l, kind)
    ent = {"kind": (_masked_entropy(kind_l), on & (v.kind_mask.sum(-1) > 1))}
    uses_region = torch.isin(kind, torch.tensor(KINDS_WITH_REGION, device=kind.device)) & on
    region_l = region_legal(v, region_l, kind)
    region = choose(region_l, "region")
    logp = logp + uses_region * _pick(region_l, region)
    ent["region"] = (_masked_entropy(region_l), uses_region)
    width = reg_h.shape[-1]
    target = torch.gather(reg_h, 1, region[:, :, None].expand(-1, -1, width))
    # The enemy head per kind: only a kind with an enemy pointer uses one.
    enemy = torch.full_like(kind, -1)
    enemy_entropy = torch.zeros_like(logp)
    for slot, k in enumerate(fz.ENEMY_POINTER_KINDS):
        here = (kind == k) & on
        if here.any():
            choice = choose(enemy_l[:, slot], "enemy")
            logp = logp + here * _pick(enemy_l[:, slot], choice)
            enemy = torch.where(here, choice, enemy)
            enemy_entropy = torch.where(here, _masked_entropy(enemy_l[:, slot]), enemy_entropy)
            target = torch.where(here[:, :, None], torch.gather(enemy_h, 1, choice[:, :, None].expand(-1, -1, width)), target)
    ent["enemy"] = (enemy_entropy, enemy >= 0)
    fine_l = actor.fine(me, target)
    fine = choose(fine_l, "fine")
    uses_fine = torch.isin(kind, torch.tensor(KINDS_WITH_FINE, device=kind.device)) & on
    logp = logp + uses_fine * _pick(fine_l, fine)
    ent["fine"] = (_masked_entropy(fine_l), uses_fine)
    produce = choose(produce_l, "produce")
    uses_produce = (kind == rf.PRODUCE) & on
    logp = logp + uses_produce * _pick(produce_l, produce)
    ent["produce"] = (_masked_entropy(produce_l), uses_produce)
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
           "fine": fine.masked_fill(~uses_fine, -1), "produce": produce.masked_fill(~uses_produce, -1), "group": group, "roles": roles}
    return out, logp.masked_fill(~on, 0.0), ent


def _taken_before(anchor: torch.Tensor, units: int) -> torch.Tensor:
    """B x K x C: the units actions before each have picked."""
    hot = F.one_hot(anchor.clamp_min(0), units).bool() & (anchor >= 0)[..., None]
    return (torch.cumsum(hot.long(), 1) - hot.long()) > 0


def sample_actions(actor: nn.Module, b: Batch, ctx, slots: int = act.SLOTS) -> Actions:
    """A decision's actions, one after another, each reading those before it, until done or `slots` of them."""
    count, units = b.ctrl_mask.shape
    device = b.ctrl_mask.device
    out = {f: torch.full((count, slots), act.UNSET[f], dtype=torch.long, device=device) for f in act.SLOT_FIELDS}
    logp = torch.zeros(count, slots, device=device)
    taken = torch.zeros(count, units, dtype=torch.bool, device=device)
    going = torch.ones(count, dtype=torch.bool, device=device)
    vectors = ctx.hu.new_zeros(count, 0, ctx.hu.shape[-1])
    for k in range(slots):
        h = actor.plan(ctx, vectors)[:, -1:]
        anchor_l = actor.anchor_logits(b, ctx, h, taken[:, None])
        pick = _sample(anchor_l)
        anchor = torch.where(going[:, None], pick - 1, -1)
        lp = _pick(anchor_l, pick).masked_fill(~going[:, None], 0.0)
        choice, lp_rest, _ = _choices(actor, per_action(b, anchor), actor.reading(ctx, anchor, h), ctx.keys, ctx.reg_h)
        out["anchor"][:, k] = anchor[:, 0]
        for f, value in choice.items():
            out[f][:, k] = value[:, 0]
        logp[:, k] = (lp + lp_rest)[:, 0]
        this = Actions(anchor, **choice, logp=lp)
        vectors = torch.cat((vectors, actor.action_vectors(b, ctx, this).to(vectors.dtype)), 1)
        taken = taken | (F.one_hot(anchor[:, 0].clamp_min(0), units).bool() & (anchor >= 0))
        going = going & (anchor[:, 0] >= 0)
        if not going.any():
            break
    return Actions(**out, logp=logp)


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
    _, rest, more = _choices(actor, per_action(b, a.anchor), actor.reading(ctx, a.anchor, h), ctx.keys, ctx.reg_h, given=a)
    ent.update(more)
    return (logp + rest).masked_fill(~valid, 0.0), ent


def teacher_logits(actor: nn.Module, b: Batch, ctx, a: Actions) -> tuple[torch.Tensor, torch.Tensor]:
    """The anchor logits (B x K x (1 + C)) and kind logits (B x K x KIND_COUNT) each of the actions had, given those before it, as
    `evaluate_actions` scores them: for measuring what cloning has learned."""
    h = actor.plan(ctx, actor.action_vectors(b, ctx, a)[:, :-1])
    anchor_l = actor.anchor_logits(b, ctx, h, _taken_before(a.anchor, b.ctrl_mask.shape[1]))
    return anchor_l, actor.heads(per_action(b, a.anchor), actor.reading(ctx, a.anchor, h), ctx.keys, ctx.reg_h)[0]


def each_unit_heads(actor: nn.Module, b: Batch, ctx):
    """The heads' logits for every controllable unit, as if each were the first action's anchor: B x C rows, as `heads` gives them.
    For looking at what the policy makes of each unit."""
    anchor = torch.arange(b.ctrl_mask.shape[1], device=b.ctrl_mask.device).expand(b.ctrl_mask.shape).masked_fill(~b.ctrl_mask, -1)
    h = actor.plan(ctx, ctx.hu.new_zeros(len(anchor), 0, ctx.hu.shape[-1])).expand(-1, anchor.shape[1], -1)
    return actor.heads(per_action(b, anchor), actor.reading(ctx, anchor, h), ctx.keys, ctx.reg_h)
