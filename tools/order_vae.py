"""The order autoencoder: a side's commands at a decision (zkbot.policy.selections) to a small code and back
(docs/learned-policy/05-order-autoencoder.md; the user's, 2026-11-19).  It reads no frame: only the commands.

    order_vae.py train --data DIR --out DIR [--steps 20000] [--batch 2048] [--beta 0.1] [--d 192] [--codes 8] [--latent 16]
    order_vae.py eval --model DIR/order-vae.pt --data DIR

A transformer reads the decision's commands, one token each, and K learned queries among them become the code: K x latent
numbers, each a mean and a variance, sampled in training and pulled towards a unit Gaussian (--beta), since a policy has to
produce codes and not only read them.  The decoder's S slot queries read the code alone and say each command: whether there
is one, and its fields, each the likeliest that's plausible (selections.able).  Trained once on the scripts' commands
(order_data.py) with plausible made-up ones mixed in, then frozen: `Coder` is what play uses.

The defaults are vae-3's, which the user chose on 2026-11-19 after vae-1 (--beta 0.01 --empty 0.1 --codes 4) and vae-2
(--beta 0.1 --empty 0.02 --codes 4): "vae-2", "bigger code should be fine", "commands should be plausible/valid".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bots"), str(ROOT / "sim" / "py")]
from zkbot.policy import groups as gr, selections as sel  # noqa: E402
from zkbot.policy.tables import table  # noqa: E402

FIELDS = ("present", "roles", "radius", "portion", "kind", "type", "extent", "centre", "place")
# The most of the card this process may hold: past the card the Windows driver spills into system memory instead of failing
# (AGENTS.md, "Traps").  Training at 2,048 decisions a step holds 3.9 GB.  And the most decisions that go through the nets at
# once outside training: on 2026-11-19 `evaluate` sent 100,000 in one batch, which spilled and filled the machine's memory.
# ORDER_VAE_CARD_GB gives another cap, for two of these on the card at once.
CARD_GB = float(os.environ.get("ORDER_VAE_CARD_GB", 6.0))
CHUNK = 4096


def cap_card(device: str) -> None:
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, CARD_GB * 2**30 / torch.cuda.mem_get_info()[1]))
KIND_INDEX = {k: i for i, k in enumerate(sel.KINDS)}


@dataclasses.dataclass(frozen=True)
class Config:
    d: int = 192
    heads: int = 4
    layers: int = 3
    codes: int = 8              # K: code tokens.  4 held the scripts' decisions of up to 4 commands, 8 hold those of up to 8
    latent: int = 16            # numbers a code token
    bands: int = 8              # of a place's Fourier features: periods from the map down to 1 / 128 of it
    slots: int = sel.SLOTS
    roles: int = 0              # filled from the tables when made
    types: int = 0
    beta: float = 0.1
    place_weight: float = 100.0     # a nat for a place 1% of the map off


def sizes(cfg: Config) -> Config:
    return dataclasses.replace(cfg, roles=len(gr.roles()[0]), types=len(table().producible))


def to_arrays(decisions: list[np.ndarray], slots: int = sel.SLOTS) -> dict[str, np.ndarray]:
    """Decisions (each its commands, selections.COMMAND_DTYPE) as padded arrays, a row a decision and a column a slot: `present`,
    and each field as the nets take it (kind: its place in selections.KINDS; type and extent: the last class for none)."""
    n, types, radii = len(decisions), len(table().producible), len(sel.RADII)
    out = {"present": np.zeros((n, slots), bool), "roles": np.zeros((n, slots), np.int64), "radius": np.zeros((n, slots), np.int64),
           "portion": np.zeros((n, slots), np.int64), "kind": np.zeros((n, slots), np.int64), "type": np.full((n, slots), types, np.int64),
           "extent": np.full((n, slots), radii, np.int64), "centre": np.zeros((n, slots, 2), np.float32), "place": np.zeros((n, slots, 2), np.float32)}
    kind_index = np.full(256, -1, np.int64)
    kind_index[list(sel.KINDS)] = np.arange(len(sel.KINDS))
    for i, c in enumerate(decisions):
        c = c[:slots]
        k = len(c)
        out["present"][i, :k] = True
        out["roles"][i, :k], out["radius"][i, :k], out["portion"][i, :k] = c["roles"], c["radius"], c["portion"]
        out["kind"][i, :k] = kind_index[c["kind"]]
        out["type"][i, :k] = np.where(c["type"] >= 0, c["type"], types)
        out["extent"][i, :k] = np.where(c["extent"] >= 0, c["extent"], radii)
        out["centre"][i, :k, 0], out["centre"][i, :k, 1] = c["cx"], c["cz"]
        out["place"][i, :k, 0], out["place"][i, :k, 1] = c["x"], c["z"]
    return out


def to_commands(a: dict[str, np.ndarray]) -> list[np.ndarray]:
    """`to_arrays` the other way: each row's commands up to its first empty slot, with the fields a kind doesn't have unset."""
    types, radii = len(table().producible), len(sel.RADII)
    kinds = np.asarray(sel.KINDS)
    placed, typed = np.isin(kinds, sel.PLACED), np.isin(kinds, sel.TYPED)
    out = []
    for i in range(len(a["present"])):
        absent = np.flatnonzero(~a["present"][i])
        k = int(absent[0]) if len(absent) else a["present"].shape[1]
        c = np.zeros(k, sel.COMMAND_DTYPE)
        ki = a["kind"][i, :k]
        c["roles"], c["radius"], c["portion"], c["kind"] = a["roles"][i, :k], a["radius"][i, :k], a["portion"][i, :k], kinds[ki]
        c["cx"], c["cz"] = a["centre"][i, :k, 0], a["centre"][i, :k, 1]
        c["x"], c["z"] = np.where(placed[ki], a["place"][i, :k, 0], 0.0), np.where(placed[ki], a["place"][i, :k, 1], 0.0)
        c["type"] = np.where(typed[ki] & (a["type"][i, :k] < types), a["type"][i, :k], -1)
        c["extent"] = np.where(np.isin(kinds[ki], sel.AREAS) & (a["extent"][i, :k] < radii), a["extent"][i, :k], -1)
        out.append(c)
    return out


def made_up(rng: np.random.Generator, n: int, slots: int, roles: int, types: int) -> dict[str, np.ndarray]:
    """A command made up for every slot of n decisions, as `to_arrays` gives them (the user, 2026-11-19: "randomly/synthetically
    generated actions to ensure coverage of cases/orders not seen in the script", and "commands should be plausible/valid").
    Its kind is any of selections.KINDS alike.  Its roles are among those with a type that can be told the kind
    (selections.able): one of them half the time, and else that one with each other of them at three in ten.  Its type, where
    the kind has one, is any that the first of its roles can build or make.  Its radius is any alike, with a half or a quarter
    of a circle two times in five; its extent and places any alike, where its kind has them."""
    can, makes = sel.able()
    assert can.shape[1] == roles and makes.shape[2] == types
    shape = (n, slots)
    kind = rng.integers(0, len(sel.KINDS), shape)
    kinds = np.asarray(sel.KINDS)[kind]
    able = can[kind]                                                # n x slots x roles
    first = _one_of(rng, able)
    one = np.eye(roles, dtype=bool)[first]
    more = (rng.random((n, slots, roles)) < 0.3) & (rng.random(shape) < 0.5)[..., None] & able
    radius = rng.integers(0, len(sel.RADII), shape)
    return {"present": np.ones(shape, bool), "roles": ((one | more).astype(np.int64) << np.arange(roles)).sum(-1), "radius": radius,
            "portion": np.where(radius > 0, rng.choice(len(sel.PORTIONS), shape, p=(0.6, 0.2, 0.2)), 0), "kind": kind,
            "type": np.where(np.isin(kinds, sel.TYPED), _one_of(rng, makes[kind, first]), types),
            "extent": np.where(np.isin(kinds, sel.AREAS), rng.integers(0, len(sel.RADII), shape), len(sel.RADII)),
            "centre": rng.random((n, slots, 2), np.float32),
            "place": np.where(np.isin(kinds, sel.PLACED)[..., None], rng.random((n, slots, 2), np.float32), np.float32(0.0))}


def _one_of(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """One of each row's set members along the last axis, any alike (0 where a row has none)."""
    return (rng.random(mask.shape, np.float32) * mask).argmax(-1)


def plausible(a: dict[str, np.ndarray]) -> np.ndarray:
    """Of decisions as `to_arrays` gives them, which slots hold a command that's plausible (selections.able): every role it
    selects has a type that can be told its kind, and its type, if its kind has one, is one that some role of its can make."""
    can, makes = sel.able()
    hot = ((a["roles"][..., None] >> np.arange(can.shape[1])) & 1).astype(bool)
    fits = (hot & ~can[a["kind"]]).sum(-1) == 0
    typed = np.isin(np.asarray(sel.KINDS)[a["kind"]], sel.TYPED)
    made = np.take_along_axis((makes[a["kind"]] & hot[..., None]).any(-2), np.minimum(a["type"], makes.shape[2] - 1)[..., None], -1)[..., 0]
    return hot.any(-1) & fits & (~typed | (made & (a["type"] < makes.shape[2])))


def _where(mask: np.ndarray, a: np.ndarray, b) -> np.ndarray:
    """a where a slot's mask is set, else b; the places' two numbers go with their slot."""
    return np.where(mask[..., None] if a.ndim == 3 else mask, a, b).astype(a.dtype)


def synthetic(rng: np.random.Generator, n: int, slots: int, roles: int, types: int) -> dict[str, np.ndarray]:
    """n decisions wholly made up (`made_up`), of 1 to `slots` commands: three in ten any number alike, the rest few, as the
    scripts' are."""
    count = np.where(rng.random(n) < 0.3, rng.integers(1, slots + 1, n), np.minimum(rng.geometric(0.4, n), slots))
    present = np.arange(slots)[None] < count[:, None]
    # An empty slot as to_arrays leaves one.
    none = {"present": False, "roles": 0, "radius": 0, "portion": 0, "kind": 0, "type": types, "extent": len(sel.RADII), "centre": 0.0, "place": 0.0}
    return {k: _where(present, v, none[k]) for k, v in made_up(rng, n, slots, roles, types).items()}


def spliced(rng: np.random.Generator, rows: dict[str, np.ndarray], roles: int, types: int, share: float = 0.3) -> dict[str, np.ndarray]:
    """Decisions with `share` of their commands swapped for made-up ones, each in its own slot: a script's decision but for a
    command no script gave."""
    swap = rows["present"] & (rng.random(rows["present"].shape) < share)
    made = made_up(rng, *rows["present"].shape, roles, types)
    return {k: _where(swap, made[k], v) for k, v in rows.items()}


def fourier(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x in [0, 1], ... x k -> ... x k (1 + 2 bands): itself, and the sine and cosine of it at `bands` octaves."""
    f = x[..., None] * (math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype))
    return torch.cat((x[..., None], f.sin(), f.cos()), -1).flatten(-2)


class OrderVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, radii = cfg.d, len(sel.RADII)
        self.roles_in = nn.Linear(cfg.roles, d, bias=False)
        self.radius_in, self.portion_in = nn.Embedding(radii, d), nn.Embedding(len(sel.PORTIONS), d)
        self.kind_in, self.type_in, self.extent_in = nn.Embedding(len(sel.KINDS), d), nn.Embedding(cfg.types + 1, d), nn.Embedding(radii + 1, d)
        self.place_in = nn.Linear(4 * (1 + 2 * cfg.bands), d)
        self.slot_in = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
        self.join = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.code_query = nn.Parameter(torch.randn(cfg.codes, d) * 0.02)
        layer = lambda cls: cls(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)    # noqa: E731
        self.reader = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), cfg.layers, enable_nested_tensor=False)
        self.to_code = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * cfg.latent))
        self.from_code = nn.Linear(cfg.latent, d)
        self.code_at = nn.Parameter(torch.randn(cfg.codes, d) * 0.02)
        self.slot_query = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
        self.writer = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), cfg.layers)
        self.out = nn.LayerNorm(d)
        widths = {"present": 1, "roles": cfg.roles, "radius": radii, "portion": len(sel.PORTIONS), "kind": len(sel.KINDS), "type": cfg.types,
                  "extent": radii, "centre": 2, "place": 2}
        self.heads = nn.ModuleDict({f"to_{name}": nn.Linear(d, width) for name, width in widths.items()})      # "type" is a Module's own
        self.register_buffer("role_bits", torch.arange(cfg.roles), persistent=False)
        # What a command can say (selections.able): a kind's roles, and what each role builds or makes under it.
        can, makes = sel.able()
        self.register_buffer("can", torch.as_tensor(can), persistent=False)
        self.register_buffer("makes", torch.as_tensor(makes).float(), persistent=False)
        self.typed = tuple(KIND_INDEX[k] for k in sel.TYPED)
        # Training's alone: the two transformers in half precision on the card (a step is 93 ms for 161 at 2,048 decisions).
        # What they give is read in float32 always: a place in half precision is a 500th of the map off.
        self.amp = False

    def _trunk(self, x: torch.Tensor):
        return torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.amp and x.device.type == "cuda")

    def role_hot(self, roles: torch.Tensor) -> torch.Tensor:
        return ((roles[..., None] >> self.role_bits) & 1).float()

    def encode(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch of decisions (`to_arrays`, as tensors) -> each code's mean and log variance, B x K x latent."""
        tokens = (self.roles_in(self.role_hot(b["roles"])) + self.radius_in(b["radius"]) + self.portion_in(b["portion"]) + self.kind_in(b["kind"])
                  + self.type_in(b["type"]) + self.extent_in(b["extent"])
                  + self.place_in(fourier(torch.cat((b["centre"], b["place"]), -1), self.cfg.bands)) + self.slot_in)
        n = len(tokens)
        hidden = torch.cat((torch.zeros(n, self.cfg.codes, dtype=torch.bool, device=tokens.device), ~b["present"]), 1)
        with self._trunk(tokens):
            x = torch.cat((self.code_query.expand(n, -1, -1).to(tokens.dtype), self.join(tokens)), 1)
            read = self.reader(x, src_key_padding_mask=hidden)[:, :self.cfg.codes]
        mean, logvar = self.to_code(read.float()).chunk(2, -1)
        return mean, logvar.clamp(-12.0, 4.0)

    def decode(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """B x K x latent -> each slot's logits by field (`present`: B x S), and `centre` and `place` in [0, 1]."""
        with self._trunk(code):
            memory = self.from_code(code) + self.code_at
            written = self.writer(self.slot_query.expand(len(code), -1, -1).to(memory.dtype), memory)
        h = self.out(written.float())
        out = {name[3:]: head(h) for name, head in self.heads.items()}
        out["present"] = out["present"].squeeze(-1)
        out["centre"], out["place"] = out["centre"].sigmoid(), out["place"].sigmoid()
        return out

    def say(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """A code's decision as `to_arrays` has one: every head's likeliest, among what's plausible (the user, 2026-11-19:
        "commands should be plausible/valid"; selections.able).  Of the roles, those with a type that can be told the kind
        said, and the likeliest of them if none is likely; of the types, one that a role said can build or make."""
        o = self.decode(code)
        said = {"present": o["present"] > 0.0, "centre": o["centre"], "place": o["place"]}
        for name in ("radius", "portion", "kind", "extent"):
            said[name] = o[name].argmax(-1)
        roles = o["roles"].masked_fill(~self.can[said["kind"]], -torch.inf)
        hot = (roles > 0.0) | F.one_hot(roles.argmax(-1), self.cfg.roles).bool()
        said["roles"] = (hot.long() << self.role_bits).sum(-1)
        made = torch.zeros_like(o["type"], dtype=torch.bool)
        for k in self.typed:
            made = made | ((said["kind"] == k)[..., None] & (hot.float() @ self.makes[k] > 0.0))
        said["type"] = o["type"].masked_fill(~made, -torch.inf).argmax(-1)      # 0 for a kind with none: to_commands leaves it out
        return said


def reconstruction(model: OrderVAE, o: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What a decoding (`OrderVAE.decode`) is off from the decisions it should say (`to_arrays`, as tensors), by field, in nats a
    decision: whether each slot has a command, and of those that should the fields their kind has, a place at
    `Config.place_weight` a whole map off.  The autoencoder's own loss but for its pull to the Gaussian, and what an actor
    cloned from the scripts learns its code by (policy_code_clone.py)."""
    cfg = model.cfg
    there = b["present"]
    n = len(there)
    kinds = torch.as_tensor(sel.KINDS, device=there.device)[b["kind"]]
    placed = there & torch.isin(kinds, torch.as_tensor(sel.PLACED, device=there.device))
    typed = there & (b["type"] < cfg.types)
    wide = there & (b["extent"] < len(sel.RADII))

    def ce(name: str, where: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(o[name][where], b[name][where], reduction="sum") / n if where.any() else o["present"].sum() * 0.0

    return {"present": F.binary_cross_entropy_with_logits(o["present"], there.float(), reduction="sum") / n,
            "roles": F.binary_cross_entropy_with_logits(o["roles"][there], model.role_hot(b["roles"])[there], reduction="sum") / n,
            "radius": ce("radius", there), "portion": ce("portion", there), "kind": ce("kind", there), "type": ce("type", typed), "extent": ce("extent", wide),
            "centre": cfg.place_weight * (o["centre"] - b["centre"]).abs()[there].sum() / n,
            "place": cfg.place_weight * (o["place"] - b["place"]).abs()[placed].sum() / n}


def loss(model: OrderVAE, b: dict[str, torch.Tensor], beta: float, sample: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean, logvar = model.encode(b)
    code = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
    parts = reconstruction(model, model.decode(code), b)
    total = sum(parts.values())
    parts["kl"] = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).sum() / len(mean)
    return total + beta * parts["kl"], parts


class Data:
    """A recording (order_data.py) as arrays in memory, a share of it held out by part, and empty decisions mixed in."""

    def __init__(self, data: str | pathlib.Path, slots: int = sel.SLOTS, held_out: float = 0.1):
        import order_data
        parts = order_data.load(data)
        assert parts, f"no parts in {data}"
        keep = max(1, round(len(parts) * held_out))
        self.train = self._arrays(parts[:-keep] or parts, slots)
        self.test = self._arrays(parts[-keep:], slots)

    @staticmethod
    def _arrays(parts: list[dict], slots: int) -> dict[str, np.ndarray]:
        decisions, w, h = [], [], []
        for p in parts:
            decisions += np.split(p["commands"], np.cumsum(p["counts"].astype(np.int64))[:-1]) if len(p["counts"]) else []
            w.append(p["map_w"])
            h.append(p["map_h"])
        out = to_arrays(decisions, slots)
        out["map"] = np.stack((np.concatenate(w), np.concatenate(h)), 1).astype(np.float32)
        return out

    @staticmethod
    def batch(arrays: dict[str, np.ndarray], index: np.ndarray, device: str, empty: float = 0.0, rng: np.random.Generator | None = None,
              made: float = 0.0, cfg: Config | None = None) -> dict[str, torch.Tensor]:
        """empty: the share of the batch's decisions made empty.  made: the share that isn't the scripts' own: half of it
        decisions wholly made up (`synthetic`), in recorded ones' place, and half recorded ones with some of their commands
        swapped for made-up ones (`spliced`)."""
        rows = {k: v[index] for k, v in arrays.items() if k != "map"}
        if rng is not None and made > 0.0:
            m = int(rng.binomial(len(index), made / 2.0))
            for k, v in synthetic(rng, m, cfg.slots, cfg.roles, cfg.types).items():
                rows[k][:m] = v
            m2 = m + int(rng.binomial(len(index) - m, made / (2.0 - made)))
            for k, v in spliced(rng, {k: v[m:m2] for k, v in rows.items()}, cfg.roles, cfg.types).items():
                rows[k][m:m2] = v
        b = {k: torch.as_tensor(v) for k, v in rows.items()}
        if rng is not None and empty > 0.0:
            b["present"][torch.as_tensor(rng.random(len(index)) < empty)] = False
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def _quantile(x: torch.Tensor, q: float) -> float:
    """Of at most a million of them; nothing of none."""
    return float(x[:1_000_000].quantile(q)) if len(x) else float("nan")


@torch.no_grad()
def measure(model: OrderVAE, arrays: dict[str, np.ndarray], device: str, batch: int = CHUNK, most: int = 200_000, noise: float = 0.0) -> dict[str, float]:
    """How well held-out decisions come back from their codes' means: by field, of the commands there; a place's error in elmos.
    In float32, as it's played.  noise: a unit Gaussian's draw times this added to each code first: what a policy whose
    spread is that much says when it means the decision."""
    amp, model.amp = model.amp, False
    model.eval()
    n = min(len(arrays["present"]), most)
    tally = Agreement(model.cfg)
    kl = 0.0
    for at in range(0, n, batch):
        index = np.arange(at, min(at + batch, n))
        b = Data.batch(arrays, index, device)
        mean, logvar = model.encode(b)
        tally.add(model.say(mean + noise * torch.randn_like(mean) if noise else mean), b, torch.as_tensor(arrays["map"][index], device=device))
        kl += float((0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)).sum())
    out = {**tally.rates(), "kl_nats": kl / max(tally.sums["decisions"], 1.0)}
    model.train()
    model.amp = amp
    return out


class Agreement:
    """How far what's said (`OrderVAE.say`) agrees with the decisions it should say (`to_arrays`, as tensors), summed over
    batches: by field, of the commands in both; a place's error in elmos, on each decision's own map."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sums: dict[str, float] = {}
        self.centre_err: list[torch.Tensor] = []
        self.place_err: list[torch.Tensor] = []

    def add(self, said: dict[str, torch.Tensor], b: dict[str, torch.Tensor], size: torch.Tensor) -> None:
        """size: each decision's map, its width and height in elmos."""
        device = size.device
        size = size[:, None, :]
        there = b["present"]
        prefix = said["present"].long().cumprod(1).bool()
        count_right = prefix.sum(1) == there.sum(1)
        both = there & prefix
        placed = both & torch.isin(torch.as_tensor(sel.KINDS, device=device)[b["kind"]], torch.as_tensor(sel.PLACED, device=device))
        typed, wide = both & (b["type"] < self.cfg.types), both & (b["extent"] < len(sel.RADII))
        ce = ((said["centre"] - b["centre"]) * size).norm(dim=-1)
        pe = ((said["place"] - b["place"]) * size).norm(dim=-1)
        right = {"kind": said["kind"] == b["kind"], "roles": said["roles"] == b["roles"], "radius": said["radius"] == b["radius"],
                 "portion": said["portion"] == b["portion"]}
        whole = both & right["kind"] & right["roles"] & right["radius"] & right["portion"] & (ce < 100.0) & (~placed | (pe < 100.0))
        whole &= ~typed | (said["type"] == b["type"])
        whole &= ~wide | (said["extent"] == b["extent"])
        add = {"decisions": float(len(there)), "count right": float(count_right.sum()), "commands": float(there.sum()), "commands said": float(both.sum()),
               "said in all": float(prefix.sum()), "type right": float((said["type"] == b["type"])[typed].sum()), "typed": float(typed.sum()),
               "extent right": float((said["extent"] == b["extent"])[wide].sum()), "wide": float(wide.sum()),
               "commands whole": float(whole.sum()),
               "decisions whole": float((count_right & (whole | ~there).all(1)).sum())}
        add.update({f"{k} right": float(v[both].sum()) for k, v in right.items()})
        for k, v in add.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.centre_err.append(ce[both].float().cpu())
        self.place_err.append(pe[placed].float().cpu())

    def rates(self) -> dict[str, float]:
        sums = self.sums
        ce, pe = torch.cat(self.centre_err), torch.cat(self.place_err)
        c = max(sums["commands said"], 1.0)
        return {"count_right": sums["count right"] / sums["decisions"], "commands_said": sums["commands said"] / max(sums["commands"], 1.0),
                "kind_right": sums["kind right"] / c, "roles_right": sums["roles right"] / c, "radius_right": sums["radius right"] / c,
                "portion_right": sums["portion right"] / c, "type_right": sums["type right"] / max(sums["typed"], 1.0),
                "extent_right": sums["extent right"] / max(sums["wide"], 1.0),
                "centre_elmos_median": _quantile(ce, 0.5), "centre_elmos_p90": _quantile(ce, 0.9),
                "place_elmos_median": _quantile(pe, 0.5), "place_elmos_p90": _quantile(pe, 0.9),
                "commands_whole": sums["commands whole"] / max(sums["commands"], 1.0), "decisions_whole": sums["decisions whole"] / sums["decisions"]}


@torch.no_grad()
def from_prior(model: OrderVAE, device: str, n: int = 20_000, spread: float = 1.0) -> dict[str, float]:
    """What codes drawn from the unit Gaussian say: how many commands, and of which kinds.  spread: the Gaussian's, round 0."""
    model.eval()
    parts = [model.say(spread * torch.randn(min(CHUNK, n - at), model.cfg.codes, model.cfg.latent, device=device)) for at in range(0, n, CHUNK)]
    said = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    prefix = said["present"].long().cumprod(1).bool()
    kinds = torch.bincount(said["kind"][prefix], minlength=len(sel.KINDS)).float()
    out = {"commands a decision": float(prefix.sum()) / n, "empty": float((~prefix.any(1)).float().mean()),
           "one unit": float((said["radius"][prefix] == 0).float().mean()) if prefix.any() else 0.0}
    out.update({f"kind {sel.KINDS[i]}": float(kinds[i] / kinds.sum().clamp_min(1.0)) for i in range(len(sel.KINDS))})
    model.train()
    return out


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_card(device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = sizes(Config(d=args.d, codes=args.codes, latent=args.latent, beta=args.beta, layers=args.layers))
    data = Data(args.data, cfg.slots)
    n = len(data.train["present"])
    print(f"{n} decisions to learn from, {len(data.test['present'])} held out; {int(data.train['present'].sum())} commands", flush=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = OrderVAE(cfg).to(device)
    model.amp = True
    print(f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * min(s / args.steps, 1.0))))
    log = (out / "log.jsonl").open("w")
    began = time.time()
    for step in range(args.steps + 1):
        b = Data.batch(data.train, rng.integers(0, n, args.batch), device, empty=args.empty, rng=rng, made=args.synthetic, cfg=cfg)
        total, parts = loss(model, b, cfg.beta * min(1.0, step / warm))
        opt.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.report == 0 or step == args.steps:
            row = {"step": step, "loss": float(total), **{k: float(v) for k, v in parts.items()}, "seconds": time.time() - began}
            if step % (args.report * 10) == 0 or step == args.steps:
                row["held_out"] = measure(model, data.test, device)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "held_out"}), flush=True)
            if "held_out" in row:
                print("  held out:", json.dumps({k: round(v, 4) for k, v in row["held_out"].items()}), flush=True)
    model.amp = False       # measured as it's played
    save(model, out / "order-vae.pt")
    # What no script gave, a place's error as on a map of 8,192 elmos a side: decisions wholly made up, those of them with 4
    # commands or fewer, and held-out decisions with some commands swapped.
    check = np.random.default_rng(args.seed + 1)
    made = synthetic(check, 50_000, cfg.slots, cfg.roles, cfg.types)
    few = made["present"].sum(1) <= 4
    held = {k: v[:50_000] for k, v in data.test.items()}
    mixed = {**spliced(check, {k: v for k, v in held.items() if k != "map"}, cfg.roles, cfg.types), "map": held["map"]}
    side = np.full((50_000, 2), 8192.0, np.float32)
    final = {"held_out": measure(model, data.test, device), "made_up": measure(model, {**made, "map": side}, device),
             "made_up_few": measure(model, {**{k: v[few] for k, v in made.items()}, "map": side[few]}, device),
             "spliced": measure(model, mixed, device), "prior": from_prior(model, device)}
    (out / "summary.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final, indent=1))


def save(model: OrderVAE, path: str | pathlib.Path) -> None:
    torch.save({"config": dataclasses.asdict(model.cfg), "model": model.state_dict(), "kinds": list(sel.KINDS), "roles": list(gr.roles()[0]),
                "radii": list(sel.RADII), "portions": list(sel.PORTIONS)}, path)


def load(path: str | pathlib.Path, device: str = "cpu") -> OrderVAE:
    """The trained autoencoder, frozen.  It was made for the commands as they were: their kinds, roles and radii must be the code's."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["kinds"] == list(sel.KINDS) and saved["roles"] == list(gr.roles()[0]) and saved["radii"] == list(sel.RADII), "the commands have changed"
    model = OrderVAE(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)


class Coder:
    """The frozen autoencoder for play: decisions' commands to codes and back, in numpy."""

    def __init__(self, path: str | pathlib.Path, device: str = "cpu"):
        self.model, self.device = load(path, device), device

    @property
    def shape(self) -> tuple[int, int]:
        return self.model.cfg.codes, self.model.cfg.latent

    @torch.no_grad()
    def encode(self, decisions: list[np.ndarray]) -> np.ndarray:
        """Each decision's code: its mean.  n x K x latent."""
        b = {k: torch.as_tensor(v).to(self.device) for k, v in to_arrays(decisions, self.model.cfg.slots).items()}
        return self.model.encode(b)[0].cpu().numpy()

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> list[np.ndarray]:
        said = self.model.say(torch.as_tensor(codes, dtype=torch.float32, device=self.device))
        return to_commands({k: v.cpu().numpy() for k, v in said.items()})


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_card(device)
    model = load(args.model, device)
    data = Data(args.data, model.cfg.slots)
    torch.manual_seed(1)
    out = {"held_out": measure(model, data.test, device), "prior": from_prior(model, device)}
    # The codes' own spread, and what's left of a decision when its code is off by a policy's spread.
    n = min(len(data.test["present"]), 100_000)
    codes = [model.encode(Data.batch(data.test, np.arange(at, min(at + CHUNK, n)), device)) for at in range(0, n, CHUNK)]
    mean, logvar = (torch.cat([c[i].cpu() for c in codes]) for i in (0, 1))
    out["codes"] = {"mean_abs": float(mean.abs().mean()), "mean_std": float(mean.std(0).mean()), "posterior_std": float((0.5 * logvar).exp().mean())}
    for spread in (0.1, 0.3, 0.5, 0.75):
        drawn = from_prior(model, device, spread=spread)
        out[f"round 0, spread {spread}"] = {k: round(v, 3) for k, v in drawn.items() if not k.startswith("kind") or v >= 0.05}
    for noise in (0.03, 0.1, 0.3, 1.0):      # 0.1 is the actor's least spread (policy_code.LOG_STD_LEAST)
        m = measure(model, data.test, device, noise=noise)
        out[f"noise {noise}"] = {k: round(m[k], 4) for k in ("count_right", "kind_right", "roles_right", "type_right", "centre_elmos_median",
                                                             "place_elmos_median", "commands_whole", "decisions_whole")}
    # By how many commands a decision has: what a code holds (vae-1's 64 numbers held four commands and not sixteen).
    count = data.test["present"].sum(1)
    for least, most in ((1, 1), (2, 4), (5, 8), (9, 15), (16, 16)):
        of = (count >= least) & (count <= most)
        m = measure(model, {k: v[np.flatnonzero(of)[:60_000]] for k, v in data.test.items()}, device)
        out[f"commands {least} to {most}"] = {"share": round(float(of.mean()), 4), **{k: round(m[k], 4) for k in (
            "kind_right", "roles_right", "type_right", "centre_elmos_median", "place_elmos_median", "commands_whole", "decisions_whole", "kl_nats")}}
    if device == "cuda":
        out["card_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)     # the most it held, of CARD_GB
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--beta", type=float, default=Config.beta)
    t.add_argument("--d", type=int, default=192)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--codes", type=int, default=Config.codes)
    t.add_argument("--latent", type=int, default=16)
    t.add_argument("--empty", type=float, default=0.02, help="the share of training decisions made empty")
    t.add_argument("--synthetic", type=float, default=0.3, help="the share of training decisions that aren't the scripts' own: half "
                   "wholly made up, half with some commands swapped for made-up ones")
    t.add_argument("--report", type=int, default=200)
    t.add_argument("--seed", type=int, default=1)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
