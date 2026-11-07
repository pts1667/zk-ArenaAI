"""The policy network as numpy: the live forward pass, with no PyTorch in the bot.

tools/policy_model.py is the same network in PyTorch for training, with the same weight names, and tools/policy_parity.py
checks that the two agree.  Weights are the torch layout: a linear's weight is out x in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import featurize as fz
from . import rawframe as rf
from . import tables

D, LAYERS, HEADS = 192, 4, 4
FFN = 4 * D
GLOBAL_STATE, REGION_STATE = 384, 32
NAME_DIM = 16
ID_DIM = 8                  # each of ally team and player
RBF_DIM = 16
POINTERS = ("enemy", "own", "region", "site", "type", "feature")
KEEP_START = 2.0
ROUTE_SCALE_START = -3.0    # softplus gives 0.05 nats per walking second: near regions first, and a far one is still sampled
HEAD_GAIN = 0.01          # an untrained kind head says only what KEEP_START says
FEATURE_VERSION = 5
ACTION_VERSION = 2          # the kinds and heads: head_kind's width, and the produce head


def weight_shapes(vocabulary: int) -> dict[str, tuple[int, ...]]:
    """Every weight and buffer by name.  vocabulary: names plus the unknown row."""
    out: dict[str, tuple[int, ...]] = {"name_embed.weight": (vocabulary, NAME_DIM), "kind_embed.weight": (fz.TOKEN_KINDS, D),
                                       "rbf": (fz.NUM_BUCKETS, RBF_DIM), "pair_w": (RBF_DIM, LAYERS * HEADS), "route_scale": (),
                                       "ally_embed.weight": (fz.MAX_ALLY_TEAMS + 1, ID_DIM), "player_embed.weight": (fz.MAX_PLAYERS + 1, ID_DIM)}

    def linear(name: str, n_in: int, n_out: int) -> None:
        out[f"{name}.weight"], out[f"{name}.bias"] = (n_out, n_in), (n_out,)

    def norm(name: str, n: int) -> None:
        out[f"{name}.weight"], out[f"{name}.bias"] = (n,), (n,)

    linear("unit_in0", fz.UNIT_DIM + NAME_DIM + 2 * ID_DIM, D)
    linear("unit_in1", D, D)
    linear("reg_in", fz.REGION_DIM, D)
    linear("site_in", fz.SITE_DIM, D)
    linear("feat_in", fz.FEATURE_DIM, D)
    linear("glob_in", fz.GLOBAL_DIM, D)
    linear("reg_mem", REGION_STATE, D)
    linear("mem_in", GLOBAL_STATE, D)
    linear("gate_w", 15, 1)
    linear("gate_m", D, D)
    for i in range(LAYERS):
        norm(f"blocks.{i}.ln1", D)
        linear(f"blocks.{i}.qkv", D, 3 * D)
        linear(f"blocks.{i}.out", D, D)
        norm(f"blocks.{i}.ln2", D)
        linear(f"blocks.{i}.ff1", D, FFN)
        linear(f"blocks.{i}.ff2", FFN, D)
    norm("lnf", D)
    for name, n_in, n_state in (("gru_g", D, GLOBAL_STATE), ("gru_r", D, REGION_STATE)):
        linear(f"{name}.ih", n_in, 3 * n_state)
        linear(f"{name}.hh", n_state, 3 * n_state)
    linear("head_kind", D, rf.KIND_COUNT)
    for p in POINTERS:
        linear(f"q_{p}", D, D)
    linear("fine0", 2 * D, D)
    linear("fine1", D, fz.FINE_OPTIONS)
    linear("type_key", tables.TYPE_DIM, D)
    return out


def random_weights(vocabulary: int, seed: int = 0) -> dict[str, np.ndarray]:
    """An untrained network: for tests, and for measuring latency before anything has been trained."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in weight_shapes(vocabulary).items():
        if name == "route_scale":
            out[name] = np.array(ROUTE_SCALE_START, np.float32)
        elif name.endswith(".bias") or name == "pair_w":
            out[name] = np.zeros(shape, np.float32)
        elif ".ln" in name or name == "lnf.weight":
            out[name] = np.ones(shape, np.float32)
        elif name == "rbf":
            out[name] = rng.standard_normal(shape).astype(np.float32)
        else:
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            gain = HEAD_GAIN if name == "head_kind.weight" else 1.0
            out[name] = (gain * rng.standard_normal(shape) / math.sqrt(max(1, fan_in))).astype(np.float32)
    out["head_kind.bias"][rf.KEEP] = KEEP_START
    return out


def _linear(w: dict[str, np.ndarray], name: str, x: np.ndarray) -> np.ndarray:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def _gelu(x: np.ndarray) -> np.ndarray:
    # torch's approximate="tanh", since numpy has no erf.
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _norm(w: dict[str, np.ndarray], name: str, x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * w[f"{name}.weight"] + w[f"{name}.bias"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _gru(w: dict[str, np.ndarray], name: str, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    new_i, gate_i, reset_i = np.split(_linear(w, f"{name}.ih", x), 3, axis=-1)
    new_h, gate_h, reset_h = np.split(_linear(w, f"{name}.hh", h), 3, axis=-1)
    reset = _sigmoid(reset_h + reset_i)
    gate = _sigmoid(gate_h + gate_i)
    new = np.tanh(new_i + new_h * reset)
    return (h - new) * gate + new


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class State:
    """The recurrent memory: reset at game start, and advanced only by decisions the net really made."""
    glob: np.ndarray
    regions: np.ndarray

    @classmethod
    def initial(cls, regions: int) -> "State":
        return cls(np.zeros(GLOBAL_STATE, np.float32), np.zeros((regions, REGION_STATE), np.float32))


@dataclass
class Encoded:
    tokens: np.ndarray            # N x D, in tensor order then the memory token
    units: np.ndarray             # U x D
    regions: np.ndarray           # R x D
    state: State


@dataclass
class Decision:
    kind: np.ndarray              # C: the kind each unit chose
    enemy: np.ndarray             # C: index into enemy_tokens, or -1
    region: np.ndarray            # C: region index, or -1
    fine: np.ndarray              # C: fine option, or -1
    produce: np.ndarray           # C: place in table.producible, or -1
    logp: np.ndarray              # C
    kinds: np.ndarray = field(default_factory=lambda: np.zeros(0))   # C x KIND_COUNT, probabilities, for the logs
    state: State | None = None


def encode(w: dict[str, np.ndarray], bias_table: np.ndarray, t: fz.Tensors, state: State) -> Encoded:
    units = np.concatenate((t.unit_x, w["name_embed.weight"][t.unit_name], w["ally_embed.weight"][t.unit_ally], w["player_embed.weight"][t.unit_player]), axis=1)
    hu = _linear(w, "unit_in1", _gelu(_linear(w, "unit_in0", units)))
    hr = _linear(w, "reg_in", t.reg_x) + _linear(w, "reg_mem", state.regions)
    if len(t.gate_a):
        # Gate message layer: each region hears its neighbours, weighted by what the gate between them is like.
        a = _linear(w, "gate_w", t.gate_x)[:, 0]
        adjacency = np.zeros((len(hr), len(hr)), np.float32)
        adjacency[t.gate_a, t.gate_b] = a
        adjacency[t.gate_b, t.gate_a] = a
        hr = hr + _linear(w, "gate_m", adjacency @ hr)
    hs = _linear(w, "site_in", t.site_x)
    hf = _linear(w, "feat_in", t.feat_x)
    hg = _linear(w, "glob_in", t.glob_x)[None, :]
    hm = _linear(w, "mem_in", state.glob)[None, :]
    h = np.concatenate((hu, hr, hs, hf, hg, hm), axis=0)
    kinds = np.concatenate((t.tok_kind, np.array([fz.TK_MEMORY_STATE], np.int8)))
    h = h + w["kind_embed.weight"][kinds]
    n = len(h)
    # One attention bias for all layers and heads, from the table made at load.
    buckets = np.full((n, n), fz.MEMORY_BUCKET, np.int64)
    buckets[:n - 1, :n - 1] = t.pair_bucket
    bias = bias_table[buckets]              # N x N x (L*H)
    per_head = D // HEADS
    scale = 1.0 / math.sqrt(per_head)
    for layer in range(LAYERS):
        qkv = _linear(w, f"blocks.{layer}.qkv", _norm(w, f"blocks.{layer}.ln1", h))
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(n, HEADS, per_head).transpose(1, 0, 2)
        k = k.reshape(n, HEADS, per_head).transpose(1, 0, 2)
        v = v.reshape(n, HEADS, per_head).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) * scale + bias[:, :, layer * HEADS:(layer + 1) * HEADS].transpose(2, 0, 1)
        o = _softmax(scores) @ v
        h = h + _linear(w, f"blocks.{layer}.out", o.transpose(1, 0, 2).reshape(n, D))
        h = h + _linear(w, f"blocks.{layer}.ff2", _gelu(_linear(w, f"blocks.{layer}.ff1", _norm(w, f"blocks.{layer}.ln2", h))))
    h = _norm(w, "lnf", h)
    u, r = len(hu), len(hr)
    new_glob = _gru(w, "gru_g", h[u + r + len(hs) + len(hf)], state.glob)
    new_regions = _gru(w, "gru_r", h[u:u + r], state.regions)
    return Encoded(h, h[:u], h[u:u + r], State(new_glob, new_regions))


def kind_logits(w: dict[str, np.ndarray], enc: Encoded, t: fz.Tensors) -> np.ndarray:
    logits = _linear(w, "head_kind", enc.units[:t.controllable])
    return np.where(t.kind_mask, logits, -1e9)


def region_logits(w: dict[str, np.ndarray], enc: Encoded, t: fz.Tensors, kind: np.ndarray | None = None) -> np.ndarray:
    """What's legal goes by the kind each unit chose: none given, the regions it can walk to."""
    q = _linear(w, "q_region", enc.units[:t.controllable])
    scores = q @ enc.regions.T / math.sqrt(D)
    # Near regions first, until the policy learns otherwise.
    scores = scores - _softplus(float(w["route_scale"])) * t.route_bias
    legal = t.ptr_region
    if kind is not None:
        legal = np.where(np.isin(kind, (rf.JUMP, rf.THROW))[:, None], t.ptr_jump, np.where(kind[:, None] == rf.WAIT_AT_BEACON, t.ptr_wait, t.ptr_region))
    return np.where(legal, scores, -1e9)


def enemy_logits(w: dict[str, np.ndarray], enc: Encoded, t: fz.Tensors) -> np.ndarray:
    """len(ENEMY_POINTER_KINDS) x C x E."""
    q = _linear(w, "q_enemy", enc.units[:t.controllable])
    scores = q @ enc.units[t.enemy_tokens].T / math.sqrt(D)
    return np.where(t.ptr_enemy, scores[None, :, :], -1e9)


def produce_logits(w: dict[str, np.ndarray], enc: Encoded, t: fz.Tensors) -> np.ndarray:
    """C x T.  A type's key comes from its catalog stats, so a new unit is a new key."""
    table = tables.table()
    q = _linear(w, "q_type", enc.units[:t.controllable])
    scores = q @ _linear(w, "type_key", table.stats[table.producible]).T / math.sqrt(D)
    return np.where(t.ptr_produce, scores, -1e9)


def fine_logits(w: dict[str, np.ndarray], unit: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.concatenate((unit, target), axis=-1)
    return _linear(w, "fine1", _gelu(_linear(w, "fine0", x)))


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def _sample(p: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """Per row: a draw from p, or the most likely if there's no generator."""
    if rng is None:
        return p.argmax(axis=-1)
    u = rng.random(p.shape[:-1])[..., None]
    return (p.cumsum(axis=-1) < u).sum(axis=-1).clip(0, p.shape[-1] - 1)


class Policy:
    """One forward pass per decision, with the sampling that follows it."""

    def __init__(self, weights: dict[str, np.ndarray], temperature: float = 1.0):
        self.w = weights
        self.temperature = temperature
        self.bias_table = weights["rbf"] @ weights["pair_w"]

    def decide(self, t: fz.Tensors, state: State, rng: np.random.Generator | None = None) -> Decision:
        w = self.w
        enc = encode(w, self.bias_table, t, state)
        c = t.controllable
        kind = np.zeros(c, np.int64)
        enemy, region, fine, produce = np.full(c, -1, np.int64), np.full(c, -1, np.int64), np.full(c, -1, np.int64), np.full(c, -1, np.int64)
        logp = np.zeros(c, np.float32)
        if not c:
            return Decision(kind, enemy, region, fine, produce, logp, np.zeros((0, rf.KIND_COUNT), np.float32), enc.state)
        temperature = self.temperature
        kinds = _softmax(kind_logits(w, enc, t) / temperature)
        kind = _sample(kinds, rng)
        rows = np.arange(c)
        logp += np.log(np.maximum(kinds[rows, kind], 1e-30))
        target_h = np.zeros((c, D), np.float32)
        wants_region = np.isin(kind, fz.KINDS_WITH_REGION)
        if wants_region.any():
            idx = np.flatnonzero(wants_region)
            p = _softmax(region_logits(w, enc, t, kind)[idx] / temperature)
            region[idx] = _sample(p, rng)
            logp[idx] += np.log(np.maximum(p[np.arange(len(idx)), region[idx]], 1e-30))
            target_h[idx] = enc.regions[region[idx]]
        wants_enemy = np.isin(kind, fz.ENEMY_POINTER_KINDS)
        if wants_enemy.any():
            logits = enemy_logits(w, enc, t)
            for slot, k in enumerate(fz.ENEMY_POINTER_KINDS):
                idx = np.flatnonzero(kind == k)
                if not len(idx):
                    continue
                p = _softmax(logits[slot][idx] / temperature)
                enemy[idx] = _sample(p, rng)
                logp[idx] += np.log(np.maximum(p[np.arange(len(idx)), enemy[idx]], 1e-30))
                target_h[idx] = enc.units[t.enemy_tokens[enemy[idx]]]
        wants_fine = np.isin(kind, fz.KINDS_WITH_FINE)
        if wants_fine.any():
            idx = np.flatnonzero(wants_fine)
            p = _softmax(fine_logits(w, enc.units[idx], target_h[idx]) / temperature)
            fine[idx] = _sample(p, rng)
            logp[idx] += np.log(np.maximum(p[np.arange(len(idx)), fine[idx]], 1e-30))
        idx = np.flatnonzero(kind == rf.PRODUCE)
        if len(idx):
            p = _softmax(produce_logits(w, enc, t)[idx] / temperature)
            produce[idx] = _sample(p, rng)
            logp[idx] += np.log(np.maximum(p[np.arange(len(idx)), produce[idx]], 1e-30))
        return Decision(kind, enemy, region, fine, produce, logp, kinds, enc.state)


def load(path: Any, table_hash: str | None = None) -> dict[str, np.ndarray]:
    """Weights from an exported .npz; raises if they don't fit this code."""
    with np.load(path) as d:
        version = int(d["meta_feature_version"]) if "meta_feature_version" in d else -1
        if version != FEATURE_VERSION:
            raise ValueError(f"{path}: feature version {version}, this code uses {FEATURE_VERSION}")
        actions = int(d["meta_action_version"]) if "meta_action_version" in d else -1
        if actions != ACTION_VERSION:
            raise ValueError(f"{path}: action version {actions}, this code uses {ACTION_VERSION}")
        if table_hash is not None and str(d["meta_catalog_hash"]) != table_hash:
            raise ValueError(f"{path}: trained on catalog {d['meta_catalog_hash']}, the game's is {table_hash}")
        return {k: d[k].astype(np.float32) for k in d.files if not k.startswith("meta_")}
