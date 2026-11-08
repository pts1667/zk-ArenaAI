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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot.policy import featurize as fz, rawframe as rf  # noqa: E402

ILLEGAL = -1e9
NAME_DIM = 16
ID_DIM = 8                   # each of ally team and player
KEEP_START = 2.0
ROUTE_SCALE_START = -3.0     # softplus gives 0.05 nats per walking second: near regions first, and a far one is still sampled
HEAD_GAIN = 0.01             # an untrained kind head says only what KEEP_START says
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
    kind: torch.Tensor      # B x C
    enemy: torch.Tensor     # B x C, -1 if unused
    region: torch.Tensor    # B x C, -1 if unused
    fine: torch.Tensor      # B x C, -1 if unused
    produce: torch.Tensor   # B x C, -1 if unused
    logp: torch.Tensor      # B x C


def sample_actions(actor: nn.Module, b: Batch, units: torch.Tensor, reg_h: torch.Tensor) -> Actions:
    kind_l, region_l, enemy_l, enemy_h, produce_l = actor.heads(b, units, reg_h)
    kind = _sample(kind_l)
    logp = _pick(kind_l, kind)
    uses_region = torch.isin(kind, torch.tensor(KINDS_WITH_REGION, device=kind.device))
    region_l = region_legal(b, region_l, kind)
    region = _sample(region_l)
    logp = logp + uses_region * _pick(region_l, region)
    # The enemy head per kind: only a kind with an enemy pointer uses one.
    enemy = torch.full_like(kind, -1)
    width = reg_h.shape[-1]
    target = torch.gather(reg_h, 1, region[:, :, None].expand(-1, -1, width))
    for slot, k in enumerate(fz.ENEMY_POINTER_KINDS):
        here = (kind == k) & b.ctrl_mask
        if here.any():
            choice = _sample(enemy_l[:, slot])
            logp = logp + here * _pick(enemy_l[:, slot], choice)
            enemy = torch.where(here, choice, enemy)
            target = torch.where(here[:, :, None], torch.gather(enemy_h, 1, choice[:, :, None].expand(-1, -1, width)), target)
    me = units[:, :kind.shape[1]]
    fine_l = actor.fine(me, target)
    fine = _sample(fine_l)
    uses_fine = torch.isin(kind, torch.tensor(KINDS_WITH_FINE, device=kind.device))
    logp = logp + uses_fine * _pick(fine_l, fine)
    produce = _sample(produce_l)
    uses_produce = kind == rf.PRODUCE
    logp = logp + uses_produce * _pick(produce_l, produce)
    off = ~b.ctrl_mask
    return Actions(kind, enemy.masked_fill(off, -1), region.masked_fill(~uses_region | off, -1), fine.masked_fill(~uses_fine | off, -1),
                   produce.masked_fill(~uses_produce | off, -1), logp.masked_fill(off, 0.0))


def evaluate_actions(actor: nn.Module, b: Batch, units: torch.Tensor, reg_h: torch.Tensor, a: Actions) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Log-probability of stored actions, per unit; and each head's entropy with the mask of units it applies to."""
    kind_l, region_l, enemy_l, enemy_h, produce_l = actor.heads(b, units, reg_h)
    legal_choices = b.kind_mask.sum(-1) > 1
    logp = _pick(kind_l, a.kind)
    ent = {"kind": (_masked_entropy(kind_l), b.ctrl_mask & legal_choices)}
    region_l = region_legal(b, region_l, a.kind)
    uses_region = a.region >= 0
    region = a.region.clamp_min(0)
    logp = logp + uses_region * _pick(region_l, region)
    ent["region"] = (_masked_entropy(region_l), uses_region)
    width = reg_h.shape[-1]
    target = torch.gather(reg_h, 1, region[:, :, None].expand(-1, -1, width))
    uses_enemy = a.enemy >= 0
    enemy = a.enemy.clamp_min(0)
    enemy_entropy = torch.zeros_like(logp)
    for slot, k in enumerate(fz.ENEMY_POINTER_KINDS):
        here = (a.kind == k) & uses_enemy
        if here.any():
            logp = logp + here * _pick(enemy_l[:, slot], enemy)
            enemy_entropy = torch.where(here, _masked_entropy(enemy_l[:, slot]), enemy_entropy)
    target = torch.where(uses_enemy[:, :, None], torch.gather(enemy_h, 1, enemy[:, :, None].expand(-1, -1, width)), target)
    ent["enemy"] = (enemy_entropy, uses_enemy)
    uses_fine = a.fine >= 0
    fine_l = actor.fine(units[:, :a.kind.shape[1]], target)
    logp = logp + uses_fine * _pick(fine_l, a.fine.clamp_min(0))
    ent["fine"] = (_masked_entropy(fine_l), uses_fine)
    uses_produce = a.produce >= 0
    logp = logp + uses_produce * _pick(produce_l, a.produce.clamp_min(0))
    ent["produce"] = (_masked_entropy(produce_l), uses_produce)
    return logp.masked_fill(~b.ctrl_mask, 0.0), ent

