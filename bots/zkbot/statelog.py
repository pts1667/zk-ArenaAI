"""Whole-game state snapshots (ML stage 2: value model; stage 4: dense reward).

Every 10 s one JSON line to state-team<N>.jsonl: economy, our and the
known enemy force by role, structures by kind, cumulative trade, and a
GRID x GRID map of value (ours/theirs, mobile/static).  Labels are added
offline (tools/train_value.py): the game outcome from results.csv and the
future metal-production ratio from the demo's team stats.  Flag
`nostatelog` turns it off.

The layout follows arXiv 2501.03832 (a transformer over a time series of
spatial feature planes predicting the winner), with Zero-K's continuous
map pooled into a coarse grid and the economy as extra per-step features.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from . import units
from .params import FLAGS
from .fightnet import NOT_UNITS
from .threat import combat_power

EVERY = 300                   # frames
GRID = 8
ROLE_KEYS = ("raid", "scout", "riot", "skirm", "assault", "snipe", "arty", "aa",
             "antiheavy", "bomb", "builder", "combat", "other")


def _role(u) -> str:
    r = units.role_of(u.data)
    return r if r in ROLE_KEYS else "other"


def _kind(u) -> str:
    d = u.data
    if units.is_mex(d):
        return "mex"
    if units.is_factory(d):
        return "factory"
    if units.is_defense(d):
        return "defense"
    if units.is_energy(d):
        return "energy"
    return "other_static"


SPEND_ARMY = ("raid", "scout", "riot", "skirm", "assault", "snipe", "arty", "aa", "antiheavy", "bomb",
              "combat", "other")


def spend_category(d: dict | None) -> str:
    """Spend category of a unit def, shared with tools/observe_dataset.py so
    our realised spending and CircuitAI's are bucketed the same way."""
    if not d:
        return "unknown"
    if units.is_mex(d):
        return "mex"
    if units.is_factory(d):
        return "factory"
    if units.is_defense(d):
        return "defense"
    if units.is_energy(d):
        return "energy"
    if not d.get("isMobile"):
        return "other_static"
    if units.is_builder(d):
        return "builder"
    if d.get("isAir"):
        return "air"
    r = units.role_of(d)
    return "army_" + (r if r in SPEND_ARMY else "other")


class StateLogMixin:
    def note_spending(self) -> None:
        """Cumulative metal committed per spend category: each own unit's
        full cost once, when it first appears (started), dead or not."""
        if self.frame - getattr(self, "_sp_frame", -10**9) < 15:
            return
        self._sp_frame = self.frame
        seen = getattr(self, "_sp_seen", None)
        if seen is None:
            seen = self._sp_seen = set()
            self.spent = Counter()
        for uid, u in self.units.items():
            if uid not in seen:
                seen.add(uid)
                if self.frame > 60 and u.name not in NOT_UNITS:   # the start commander is free
                    cat = spend_category(u.data)
                    self.spent[cat] += units.cost(u.data)
                    if cat.startswith("army_"):
                        log = getattr(self, "army_spend_log", None)
                        if log is None:
                            log = self.army_spend_log = []
                        log.append((self.frame, cat[5:], units.cost(u.data)))

    def recent_army_spend(self, window: int = 7200) -> Counter:
        """Metal started per army role in the last `window` frames."""
        log = getattr(self, "army_spend_log", [])
        cutoff = self.frame - window
        if log and log[0][0] < cutoff - 1800:
            self.army_spend_log = log = [e for e in log if e[0] >= cutoff]
        out = Counter()
        for f, role, c in log:
            if f >= cutoff:
                out["raid" if role == "scout" else role] += c
        return out

    def statelog_step(self) -> None:
        if "nostatelog" in FLAGS and "spendmix" not in FLAGS:
            return
        self.note_spending()
        if "nostatelog" in FLAGS:
            return
        if self.frame - getattr(self, "_sl_frame", -10**9) < EVERY:
            return
        self._sl_frame = self.frame
        W = float(self.map.width or 1.0)
        H = float(self.map.height or 1.0)
        grid = [[[0.0] * 4 for _ in range(GRID)] for _ in range(GRID)]

        def cell(u):
            return (min(GRID - 1, max(0, int(u.z / H * GRID))), min(GRID - 1, max(0, int(u.x / W * GRID))))

        own_mob, own_st, own_pow = Counter(), Counter(), 0.0
        for u in self.units.values():
            if not u.is_finished or u.name in NOT_UNITS:
                continue
            c = units.cost(u.data)
            i, j = cell(u)
            if u.is_mobile:
                own_mob[_role(u)] += c
                own_pow += combat_power(u)
                grid[i][j][0] += c
            else:
                own_st[_kind(u)] += c
                own_st["n_" + _kind(u)] += 1
                grid[i][j][1] += c
        foe_mob, foe_st, foe_pow = Counter(), Counter(), 0.0
        recent = self.frame - 1800
        seen = set()
        for uid, (e, f) in getattr(self, "enemy_memory", {}).items():
            if f < recent or not e.is_mobile:
                continue
            seen.add(uid)
            c = units.cost(e.data)
            foe_mob[_role(e)] += c
            foe_pow += combat_power(e)
            i, j = cell(e)
            grid[i][j][2] += c
        for e in self.enemies.values():
            if e.uid in seen or not e.is_mobile:
                continue
            seen.add(e.uid)
            c = units.cost(e.data)
            foe_mob[_role(e)] += c
            foe_pow += combat_power(e)
            i, j = cell(e)
            grid[i][j][2] += c
        for uid, (e, _f) in getattr(self, "enemy_structures", {}).items():
            c = units.cost(e.data)
            foe_st[_kind(e)] += c
            foe_st["n_" + _kind(e)] += 1
            i, j = cell(e)
            grid[i][j][3] += c
        rec = {
            "t": round(self.frame / 30.0),
            "m_inc": round(self.metal_income(), 2),
            "e_inc": round(self.energy_income(), 1),
            "m_cur": round(self.metal_current()),
            "e_cur": round(self.energy_current()),
            "own_mob": {k: round(v) for k, v in own_mob.items()},
            "own_st": {k: round(v) for k, v in own_st.items()},
            "own_pow": round(own_pow),
            "foe_mob": {k: round(v) for k, v in foe_mob.items()},
            "foe_st": {k: round(v) for k, v in foe_st.items()},
            "foe_pow": round(foe_pow),
            "blips": len(self.radar),
            "lost": round(getattr(self, "own_lost_value", 0.0)),
            "killed": round(getattr(self, "enemy_killed_value", 0.0)),
            "spent": {k: round(v) for k, v in getattr(self, "spent", {}).items()},
            "grid": [[[round(v) for v in c] for c in row] for row in grid],
            "swap": 1 if self.dist(self.base_pos, (0.0, 0.0)) > self.dist(self.base_pos, (W, H)) else 0,
        }
        f = getattr(self, "_sl_file", None)
        if f is None:
            try:
                path = os.path.join(os.environ.get("ZK_ARENA_OUT", "."), f"state-team{self.team_id}.jsonl")
                f = self._sl_file = open(path, "a", encoding="utf8", buffering=1)
            except OSError:
                self._sl_file = False
                return
        if f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ------------------------------------------------------------ featurising
# Shared with tools/train_value.py.

def _l(v: float) -> float:
    import math
    return math.log1p(max(0.0, float(v))) / 8.0


def state_vector(rec: dict) -> list[float]:
    """Per-step features: time, economy, force by role, structures, trade."""
    om, fm = rec.get("own_mob") or {}, rec.get("foe_mob") or {}
    os_, fs = rec.get("own_st") or {}, rec.get("foe_st") or {}
    v = [rec.get("t", 0) / 1800.0, _l(rec.get("m_inc", 0) * 60), _l(rec.get("e_inc", 0) * 60),
         _l(rec.get("m_cur", 0)), _l(rec.get("e_cur", 0)), _l(rec.get("own_pow", 0)), _l(rec.get("foe_pow", 0)),
         _l(rec.get("blips", 0) * 50), _l(rec.get("lost", 0)), _l(rec.get("killed", 0))]
    v += [_l(om.get(r, 0)) for r in ROLE_KEYS]
    v += [_l(fm.get(r, 0)) for r in ROLE_KEYS]
    for side in (os_, fs):
        for k in ("mex", "factory", "defense", "energy", "other_static"):
            v += [_l(side.get(k, 0)), side.get("n_" + k, 0) / 20.0]
    return v


STATE_DIM = 10 + 2 * len(ROLE_KEYS) + 2 * 5 * 2


def state_grid(rec: dict) -> list[float]:
    """GRID x GRID x 4 value planes, flipped so our base is always at (0, 0)."""
    g = rec.get("grid") or [[[0, 0, 0, 0]] * GRID] * GRID
    if rec.get("swap"):
        g = [row[::-1] for row in g[::-1]]
    return [_l(c) for row in g for cell in row for c in cell]
