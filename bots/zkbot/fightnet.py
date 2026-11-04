"""Learned fight evaluator (ML stage 1): entity-level fight logging,
featurisation shared with the trainer, and numpy inference.

Why: the hand-tuned power ratio (local_balance) ignores range, turrets,
air, terrain and unit abilities.  Engagements where it rated us 2.7x
stronger traded 0.3 metal killed per metal lost (h121).

Logging (FightLogMixin, flag `nofightlog` turns it off): every 2 s, for up
to MAX_OPEN separated fight sites, one snapshot of every unit within
RADIUS of the site, ours and theirs (mobile, static, air, remembered enemy
structures, radar blips), each as

  [side, name, dx, dz, dh, hp_frac, flags]          (see FLAG_* below)

and, HORIZONS seconds later, the metal each side lost among those units:
cost x min(hp fraction at snapshot, damage taken since / max HP).  Damage
comes from UNIT_DAMAGED (topic 9, ours) and ENEMY_DAMAGED (topic 17,
theirs), so enemy losses are measured, not guessed from vanished units.
One JSON line per snapshot goes to fights-team<N>.jsonl.

Model: a small set transformer over the entities (tools/train_fightnet.py,
PyTorch).  Weights are exported to data/fightnet.npz and evaluated here in
numpy, so the bot needs no PyTorch.  Outputs per horizon: the expected
fraction of our and of their snapshot value lost.
"""
from __future__ import annotations

import json
import math
import os
import zlib
from typing import Any

from . import catalog, units
from .params import FLAGS

try:
    import numpy as np
except ImportError:  # the trainer always has numpy; the bot does too
    np = None

RADIUS = 900.0
SNAP_EVERY = 60          # frames
HORIZONS = (10, 20)      # seconds
MAX_OPEN = 10
MAX_ENTS = 96
SITE_SEPARATION = 500.0
CONTACT = 700.0

FLAG_STATIC = 1
FLAG_AIR = 2
FLAG_RETREAT = 4
FLAG_BLIP = 8            # radar only: type may be unknown
FLAG_UNBUILT = 16
FLAG_MEMORY = 32         # remembered structure, not currently seen

RETREAT_BRANCHES = ("retreat_new", "retreating")

# Decision thresholds on the predicted exchange (their metal lost + 50) /
# (ours + 50) over the longest horizon, flag `fightnet`.
from .params import P  # noqa: E402
# Defaults sit at quantiles of the d1-trained model's predictions over the
# logged fight sites (which regress toward the mean: 10/25/50/75/90% at
# 0.36/0.50/0.69/0.92/1.2): attack ~top 20%, retreat early in the bottom 25%.
TARGET_EXCH = P("fightnet.target_exchange", 1.0, 0.3, 4.0, "attack a target only if the moved army is predicted to trade at least this")
HOLD_EXCH = P("fightnet.hold_exchange", 0.6, 0.2, 3.0, "meet an attack on our ground if the predicted trade is at least this")
CHASE_EXCH = P("fightnet.chase_exchange", 0.7, 0.2, 3.0, "chase a local target only if the fight there trades at least this")
RETREAT_WIN_EXCH = P("fightnet.retreat_win", 1.2, 0.8, 5.0, "predicted trade above which damaged units stay longest")
RETREAT_LOSE_EXCH = P("fightnet.retreat_lose", 0.5, 0.2, 1.5, "predicted trade below which damaged units leave earliest")

ROLES = ("raid", "scout", "riot", "skirm", "assault", "snipe", "arty", "aa",
         "antiheavy", "bomb", "builder", "static", "other")
TAGS = ("regen", "cloak", "shield", "jump", "emp", "slow", "disarm", "fire",
        "gravity", "kamikaze", "manualfire", "drones", "stealth", "armor", "capture")
TYPE_DIM = 13 + len(ROLES) + len(TAGS)
DYN_DIM = 11
N_HASH = 512             # name embedding buckets; 0 = unknown


# ------------------------------------------------------------ featurising

_TYPE_CACHE: dict[str, list[float]] = {}


# Engine helpers that are not combat value: a terraform unit is listed at
# 100,000 metal and vanishes when its terraform completes, which logged 17%
# of the first fight snapshots with a huge "loss".
NOT_UNITS = frozenset(("terraunit",))


def type_features(name: str) -> list[float]:
    """Static per-type vector from the unit dump: works for unit types the
    model never saw, so balance changes and new units are not blind spots."""
    v = _TYPE_CACHE.get(name)
    if v is not None:
        return v
    d = units.by_name(name) or {}
    s = catalog.stats(name) if d else None
    if not d or s is None:
        v = [0.0] * TYPE_DIM
        v[12] = 1.0                      # unknown type
        _TYPE_CACHE[name] = v
        return v
    is_def = 1.0 if units.is_defense(d) else 0.0
    role = catalog.infer_role(name) if d.get("isMobile") else "static"
    if role not in ROLES:
        role = "other"
    t = catalog.tags(name)
    v = [
        math.log1p(s["cost"]) / 7.0,
        math.log1p(s["hp"]) / 9.0,
        math.log1p(s["dps_ground"]) / 7.0,
        math.log1p(s["dps_air"]) / 7.0,
        s["range"] / 1000.0,
        s["speed"] / 200.0,
        s["aoe"] / 200.0,
        s["air"],
        0.0 if d.get("isMobile") else 1.0,
        1.0 if units.is_builder(d) else 0.0,
        is_def,
        s["unarmed"],
        0.0,                              # unknown-type marker
    ]
    v += [1.0 if role == r else 0.0 for r in ROLES]
    v += [1.0 if g in t else 0.0 for g in TAGS]
    _TYPE_CACHE[name] = v
    return v


def name_bucket(name: str) -> int:
    if not name or name.startswith("def") or name == "?":
        return 0
    return 1 + zlib.crc32(name.encode()) % (N_HASH - 1)


def dyn_features(side: int, dx: float, dz: float, dh: float, hp: float, flags: int) -> list[float]:
    dist = math.hypot(dx, dz)
    return [
        1.0 if side == 0 else -1.0,
        dx / RADIUS, dz / RADIUS, dist / RADIUS,
        max(-3.0, min(3.0, dh / 100.0)),
        hp,
        1.0 if flags & FLAG_STATIC else 0.0,
        1.0 if flags & FLAG_RETREAT else 0.0,
        1.0 if flags & FLAG_BLIP else 0.0,
        1.0 if flags & FLAG_UNBUILT else 0.0,
        1.0 if flags & FLAG_MEMORY else 0.0,
    ]


def featurize(ents: list) -> tuple[list[list[float]], list[int], list[int], list[float]]:
    """Entity list -> (feature rows, name buckets, sides, values).  value is
    cost x hp fraction: what each entity can still lose."""
    rows, ids, sides, vals = [], [], [], []
    for side, name, dx, dz, dh, hp, flags in ents[:MAX_ENTS]:
        tf = type_features(name)
        rows.append(tf + dyn_features(side, dx, dz, dh, hp, flags))
        ids.append(name_bucket(name) if not flags & FLAG_BLIP or not name.startswith("def") else 0)
        sides.append(side)
        c = float((units.by_name(name) or {}).get("costM") or 0.0)
        vals.append(c * max(0.0, min(1.0, hp)))
    return rows, ids, sides, vals


# ------------------------------------------------------------ inference

class FightNet:
    """Numpy forward pass of the trained set transformer.  Must mirror
    tools/train_fightnet.py:Net exactly (pre-LN encoder, tanh GELU)."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k] for k in z.files}
        self.layers = int(self.w["meta_layers"])
        self.heads = int(self.w["meta_heads"])
        self.horizons = [int(h) for h in self.w["meta_horizons"]]

    @staticmethod
    def _ln(x, g, b):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * g + b

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

    def forward(self, rows, ids, sides) -> np.ndarray:
        w = self.w
        x = np.asarray(rows, dtype=np.float32)
        h = self._gelu(x @ w["inp.0.weight"].T + w["inp.0.bias"])
        h = h @ w["inp.2.weight"].T + w["inp.2.bias"]
        h = h + w["emb.weight"][np.asarray(ids)]
        n, d = h.shape
        hd = d // self.heads
        for i in range(self.layers):
            p = f"enc.{i}."
            a = self._ln(h, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = a @ w[p + "qkv.weight"].T + w[p + "qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(n, self.heads, hd).transpose(1, 0, 2)
            k = k.reshape(n, self.heads, hd).transpose(1, 0, 2)
            v = v.reshape(n, self.heads, hd).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            att = e / e.sum(-1, keepdims=True)
            o = (att @ v).transpose(1, 0, 2).reshape(n, d)
            h = h + o @ w[p + "out.weight"].T + w[p + "out.bias"]
            a = self._ln(h, w[p + "ln2.weight"], w[p + "ln2.bias"])
            f = self._gelu(a @ w[p + "ff1.weight"].T + w[p + "ff1.bias"])
            h = h + f @ w[p + "ff2.weight"].T + w[p + "ff2.bias"]
        h = self._ln(h, w["lnf.weight"], w["lnf.bias"])
        sides = np.asarray(sides)
        pooled = []
        for s_ in (0, 1):
            m = sides == s_
            pooled.append(h[m].mean(0) if m.any() else np.zeros(d, dtype=np.float32))
            pooled.append(h[m].max(0) if m.any() else np.zeros(d, dtype=np.float32))
        z = np.concatenate(pooled)
        z = self._gelu(z @ w["head.0.weight"].T + w["head.0.bias"])
        out = z @ w["head.2.weight"].T + w["head.2.bias"]
        return 1.0 / (1.0 + np.exp(-out))       # [our_frac@h..., their_frac@h...]


_NET: dict[str, Any] = {}


def load_net() -> FightNet | None:
    if "net" in _NET:
        return _NET["net"]
    path = os.environ.get("ZKBOT_FIGHTNET") or os.path.join(os.path.dirname(__file__), "data", "fightnet.npz")
    net = None
    if np is not None and os.path.exists(path):
        try:
            net = FightNet(path)
        except Exception:
            net = None
    _NET["net"] = net
    return net


# ------------------------------------------------------------ bot side

class FightLogMixin:
    """Entity snapshots + measured outcomes, and the evaluator hook."""

    # ----- damage ledgers (cumulative, by unit id)

    def note_damage_events(self, obs: dict) -> None:
        own = self.__dict__.setdefault("own_dmg_taken", {})
        foe = self.__dict__.setdefault("enemy_dmg_taken", {})
        for ev in obs.get("events") or []:
            if not isinstance(ev, dict):
                continue
            topic = ev.get("topic")
            if topic not in (9, 17) or ev.get("paralyzer"):
                continue
            try:
                dmg = float(ev.get("damage") or 0.0)
                uid = int(ev.get("unitId") if topic == 9 else ev.get("enemyId"))
            except (TypeError, ValueError):
                continue
            if dmg <= 0.0:
                continue
            led = own if topic == 9 else foe
            led[uid] = led.get(uid, 0.0) + dmg

    # ----- entity snapshot

    def fight_entities(self, center: tuple[float, float], radius: float = RADIUS,
                       commit: bool = False) -> tuple[list, dict]:
        """Entities within radius of center, nearest first.  Returns the
        entity rows and {uid: (side, cost, max_hp, hp_frac)} for outcome
        tracking.  commit=True clears our retreat flags: "if we fight"."""
        cx, cz = center
        hmap = getattr(self.map, "ground_height", None)
        h0 = hmap(cx, cz) if hmap else 0.0
        branch = getattr(self, "unit_branch", {})
        r2 = radius * radius
        cand = []

        def add(side, u, extra):
            if u.name in NOT_UNITS:
                return
            dx, dz = u.x - cx, u.z - cz
            d2 = dx * dx + dz * dz
            if d2 > r2:
                return
            flags = extra
            if not u.is_mobile:
                flags |= FLAG_STATIC
            if u.is_air:
                flags |= FLAG_AIR
            if not u.is_finished:
                flags |= FLAG_UNBUILT
            cand.append((d2, side, u, flags, dx, dz))

        for uid, u in self.units.items():
            extra = 0
            if not commit and branch.get(uid) in RETREAT_BRANCHES:
                extra |= FLAG_RETREAT
            add(0, u, extra)
        seen = set()
        for e in self.enemies.values():
            seen.add(e.uid)
            add(1, e, 0)
        for uid, (e, _f) in getattr(self, "enemy_structures", {}).items():
            if uid not in seen:
                seen.add(uid)
                add(1, e, FLAG_MEMORY)
        for uid, e in self.radar.items():
            if uid not in seen:
                add(1, e, FLAG_BLIP)
        cand.sort(key=lambda c: c[0])
        ents, track = [], {}
        for _d2, side, u, flags, dx, dz in cand[:MAX_ENTS]:
            dh = (hmap(u.x, u.z) - h0) if hmap else 0.0
            hp = round(u.alive_frac, 3) if not flags & FLAG_BLIP else 1.0
            ents.append([side, u.name, round(dx), round(dz), round(dh), hp, flags])
            if not flags & (FLAG_BLIP | FLAG_MEMORY):
                track[u.uid] = (side, units.cost(u.data), max(1.0, u.max_health), u.alive_frac)
        return ents, track

    # ----- logger

    def fightlog_step(self) -> None:
        if "nofightlog" in FLAGS:
            return
        if self.frame - getattr(self, "_fl_frame", -10**9) < SNAP_EVERY:
            return
        self._fl_frame = self.frame
        open_ = self.__dict__.setdefault("_fl_open", [])
        still = []
        for snap in open_:
            self._fl_collect(snap)
            if self.frame - snap["frame"] >= HORIZONS[-1] * 30:
                self._fl_write(snap)
            else:
                still.append(snap)
        open_[:] = still
        if len(open_) >= MAX_OPEN:
            return
        # Sites: enemy combat units and defences in contact with our units.
        foes = [e for e in self.enemies.values()
                if (e.is_mobile and not e.is_builder and self.is_dangerous(e))
                or (not e.is_mobile and units.is_defense(e.data))]
        foes += [e for uid, (e, _f) in getattr(self, "enemy_structures", {}).items()
                 if uid not in self.enemies and units.is_defense(e.data)]
        if not foes:
            return
        mine = [u for u in self.units.values()
                if u.is_finished and ((u.is_mobile and u.is_combat) or (not u.is_mobile and units.is_defense(u.data)))]
        if not mine:
            return
        used: set[int] = set()
        for seed in sorted(foes, key=lambda e: -units.cost(e.data)):
            if seed.uid in used or len(open_) >= MAX_OPEN:
                continue
            cl = [e for e in foes if e.uid not in used and self.dist(e.pos, seed.pos) < SITE_SEPARATION]
            used.update(e.uid for e in cl)
            c = (sum(e.x for e in cl) / len(cl), sum(e.z for e in cl) / len(cl))
            near = [u for u in mine if self.dist(u.pos, c) < CONTACT]
            if not near:
                continue
            # Static vs static (two turret lines in range of nothing) is not
            # a fight anyone chose.
            if all(not u.is_mobile for u in near) and all(not e.is_mobile for e in cl):
                continue
            if any(self.dist(c, s["pos"]) < SITE_SEPARATION for s in open_):
                continue
            ents, track = self.fight_entities(c)
            own_led = getattr(self, "own_dmg_taken", {})
            foe_led = getattr(self, "enemy_dmg_taken", {})
            open_.append({
                "frame": self.frame,
                "pos": c,
                "ents": ents,
                "track": track,
                "dmg0": {uid: (own_led if s == 0 else foe_led).get(uid, 0.0) for uid, (s, *_r) in track.items()},
                "out": {},
            })

    def _fl_collect(self, snap: dict) -> None:
        age = (self.frame - snap["frame"]) / 30.0
        for h in HORIZONS:
            if str(h) in snap["out"] or age < h:
                continue
            own_led = getattr(self, "own_dmg_taken", {})
            foe_led = getattr(self, "enemy_dmg_taken", {})
            lost = [0.0, 0.0]
            for uid, (side, cost, mhp, frac0) in snap["track"].items():
                led = own_led if side == 0 else foe_led
                dmg = led.get(uid, 0.0) - snap["dmg0"].get(uid, 0.0)
                frac = min(frac0, dmg / mhp)
                if side == 0 and uid not in self.units:
                    frac = frac0          # our unit died: certain
                lost[side] += cost * max(0.0, frac)
            snap["out"][str(h)] = [round(lost[0]), round(lost[1])]

    def _fl_write(self, snap: dict) -> None:
        self._fl_collect(snap)
        rec = {
            "t": round(snap["frame"] / 30.0),
            "pos": [round(v) for v in snap["pos"]],
            "ents": snap["ents"],
            "out": snap["out"],
        }
        f = getattr(self, "_fl_file", None)
        if f is None:
            try:
                path = os.path.join(os.environ.get("ZK_ARENA_OUT", "."), f"fights-team{self.team_id}.jsonl")
                f = self._fl_file = open(path, "a", encoding="utf8", buffering=1)
            except OSError:
                self._fl_file = False
                return
        if f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ----- evaluator

    def fight_eval(self, center: tuple[float, float], commit: bool = True) -> dict | None:
        """Predicted metal lost by each side within the longest horizon if the
        fight at center goes on (commit=True: as if nobody retreats).
        Cached per 200-elmo cell for 1 s."""
        net = load_net()
        if net is None:
            return None
        key = (int(center[0] // 200), int(center[1] // 200), commit)
        cache = self.__dict__.setdefault("_fe_cache", {})
        hit = cache.get(key)
        if hit is not None and self.frame - hit[0] < 30:
            return hit[1]
        ents, _track = self.fight_entities(center, commit=commit)
        if not any(e[0] == 0 for e in ents) or not any(e[0] == 1 for e in ents):
            res = None
        else:
            rows, ids, sides, vals = featurize(ents)
            p = net.forward(rows, ids, sides)
            n = len(net.horizons)
            ours_v = sum(v for v, s in zip(vals, sides) if s == 0)
            theirs_v = sum(v for v, s in zip(vals, sides) if s == 1)
            res = {
                "our_value": ours_v,
                "their_value": theirs_v,
                "our_lost": float(p[n - 1]) * ours_v,
                "their_lost": float(p[2 * n - 1]) * theirs_v,
            }
            res["exchange"] = (res["their_lost"] + 50.0) / (res["our_lost"] + 50.0)
        if len(cache) > 400:
            cache.clear()
        cache[key] = (self.frame, res)
        return res

    def fight_eval_moved(self, target: tuple[float, float], uids: list[int]) -> dict | None:
        """Predicted outcome if the given units of ours fought at target:
        their formation (offsets from its median, squeezed to fit) is moved
        there and replaces whatever of ours is already near it."""
        net = load_net()
        if net is None or not uids:
            return None
        group = [self.units[i] for i in uids if i in self.units]
        if not group:
            return None
        key = ("mv", int(target[0] // 200), int(target[1] // 200), len(group) // 3)
        cache = self.__dict__.setdefault("_fe_cache", {})
        hit = cache.get(key)
        if hit is not None and self.frame - hit[0] < 60:
            return hit[1]
        xs = sorted(u.x for u in group)
        zs = sorted(u.z for u in group)
        mx, mz = xs[len(xs) // 2], zs[len(zs) // 2]
        ents, _t = self.fight_entities(target, commit=True)
        ids = {i for i in uids}
        ents = [e for e in ents if e[0] == 1]
        squeeze = 0.6 * RADIUS
        for u in group:
            dx, dz = u.x - mx, u.z - mz
            d = math.hypot(dx, dz)
            if d > squeeze:
                dx, dz = dx * squeeze / d, dz * squeeze / d
            # Approach from our side: shift the formation toward home.
            hx, hz = self.base_pos[0] - target[0], self.base_pos[1] - target[1]
            hn = math.hypot(hx, hz) or 1.0
            dx += hx / hn * 250.0
            dz += hz / hn * 250.0
            ents.append([0, u.name, round(dx), round(dz), 0, round(u.alive_frac, 3), 0])
        for uid, u in self.units.items():
            if uid not in ids and not u.is_mobile and self.dist(u.pos, target) < RADIUS:
                ents.append([0, u.name, round(u.x - target[0]), round(u.z - target[1]), 0,
                             round(u.alive_frac, 3), FLAG_STATIC])
        res = None
        if any(e[0] == 1 for e in ents):
            ents.sort(key=lambda e: e[2] * e[2] + e[3] * e[3])
            rows, nid, sides, vals = featurize(ents)
            p = net.forward(rows, nid, sides)
            n = len(net.horizons)
            ours_v = sum(v for v, s in zip(vals, sides) if s == 0)
            theirs_v = sum(v for v, s in zip(vals, sides) if s == 1)
            res = {"our_value": ours_v, "their_value": theirs_v,
                   "our_lost": float(p[n - 1]) * ours_v, "their_lost": float(p[2 * n - 1]) * theirs_v}
            res["exchange"] = (res["their_lost"] + 50.0) / (res["our_lost"] + 50.0)
        cache[key] = (self.frame, res)
        return res
