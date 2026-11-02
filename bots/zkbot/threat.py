"""Grid threat map and enemy-structure memory.

One numpy grid (64-elmo cells) holds the ground DPS that known enemies can
put on each cell: visible units, radar contacts, recently seen mobile units
(position decays out after a short time) and remembered static defences
(kept until they are seen destroyed or their spot is seen empty).  A second
grid holds our own combat DPS so local fights can be judged as
"our power vs their power" instead of unit counts.

Every module that used to ask "is any enemy within R of this point" can use
``threat_at`` / ``path_threat`` / ``safest_point_near`` instead.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from . import units
from .models import UnitInfo

CELL = 64.0
_WEAPONS_PATH = pathlib.Path(__file__).resolve().parent / "data" / "weapons.json"
try:
    WEAPONS: dict[str, dict] = json.loads(_WEAPONS_PATH.read_text(encoding="utf8"))
except Exception:  # pragma: no cover
    WEAPONS = {}

_COMMANDER_DEFAULT = {"dps_ground": 110.0, "dps_air": 0.0, "range": 420.0, "aoe": 0.0}


def weapon_info(u: UnitInfo) -> dict:
    w = WEAPONS.get(u.name)
    if w is not None:
        return w
    name = u.name.lower()
    if "dyntrainer" in name or "comm" in name or name.startswith("dyn"):
        return _COMMANDER_DEFAULT
    rng = float(u.data.get("maxWeaponRange") or 0.0)
    if rng <= 0.0:
        return {"dps_ground": 0.0, "dps_air": 0.0, "range": 0.0, "aoe": 0.0}
    # Unknown armed unit: assume DPS proportional to cost.
    return {"dps_ground": max(20.0, units.cost(u.data) * 0.5), "dps_air": 0.0, "range": rng, "aoe": 0.0}


def ground_dps(u: UnitInfo) -> float:
    return float(weapon_info(u).get("dps_ground") or 0.0)


def weapon_range(u: UnitInfo) -> float:
    w = weapon_info(u)
    return float(w.get("range") or u.data.get("maxWeaponRange") or 0.0)


def combat_power(u: UnitInfo) -> float:
    """Lanchester-style strength: sqrt(dps * effective hp)."""
    dps = ground_dps(u)
    if dps <= 0.0:
        return 0.0
    hp = u.health if u.health > 1.0 else float(u.data.get("health") or u.max_health or 1.0)
    hp = max(1.0, hp)
    return math.sqrt(dps * hp)


class ThreatMixin:
    # ------------------------------------------------------------- building

    def _threat_shape(self) -> tuple[int, int]:
        w = max(1, int(math.ceil(max(1.0, self.map.width) / CELL)))
        h = max(1, int(math.ceil(max(1.0, self.map.height) / CELL)))
        return h, w

    def _stamp(self, grid: np.ndarray, pos: tuple[float, float], radius: float, value: float) -> None:
        if value <= 0.0 or radius <= 0.0:
            return
        h, w = grid.shape
        cx, cz = pos[0] / CELL, pos[1] / CELL
        r = radius / CELL
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
        z0, z1 = max(0, int(cz - r)), min(h, int(cz + r) + 1)
        if x1 <= x0 or z1 <= z0:
            return
        xs = np.arange(x0, x1) + 0.5 - cx
        zs = np.arange(z0, z1) + 0.5 - cz
        d2 = zs[:, None] ** 2 + xs[None, :] ** 2
        mask = d2 <= r * r
        grid[z0:z1, x0:x1] += mask * value

    def note_enemy_structure(self, e: UnitInfo) -> None:
        if e.is_mobile:
            return
        mem = getattr(self, "enemy_structures", None)
        if mem is None:
            mem = {}
            self.enemy_structures = mem
        mem[e.uid] = (e, self.frame)

    def _forget_seen_empty_structures(self) -> None:
        mem = getattr(self, "enemy_structures", {})
        if not mem:
            return
        viewers = [
            (u.pos, float(u.data.get("losRadius") or 0.0) * 0.7)
            for u in self.units.values()
            if u.is_finished and float(u.data.get("losRadius") or 0.0) > 0
        ]
        for uid, (e, _f) in list(mem.items()):
            if uid in self.enemies:
                continue
            for p, r in viewers:
                if r > 50 and (p[0] - e.x) ** 2 + (p[1] - e.z) ** 2 < r * r:
                    mem.pop(uid, None)
                    break

    def forget_vanished_enemies(self) -> None:
        """Drop remembered mobile enemies whose last position is deep inside
        our current LOS but which are not visible: they died (the
        enemy-destroyed event carries no id) or moved, and either way the
        remembered position is wrong."""
        viewers = [
            (u.pos, float(u.data.get("losRadius") or 0.0) * 0.6)
            for u in self.units.values()
            if u.is_finished and float(u.data.get("losRadius") or 0.0) > 100
        ]
        if not viewers:
            return
        for uid, (e, f) in list(self.enemy_memory.items()):
            if uid in self.enemies or uid in self.radar or self.frame - f < 20:
                continue
            for p, r in viewers:
                if (p[0] - e.x) ** 2 + (p[1] - e.z) ** 2 < r * r:
                    self.enemy_memory.pop(uid, None)
                    break

    def update_threat_map(self) -> None:
        if self.map.width <= 0:
            return
        shape = self._threat_shape()
        grid = np.zeros(shape, dtype=np.float32)
        ours = np.zeros(shape, dtype=np.float32)
        air = np.zeros(shape, dtype=np.float32)
        self._air_build = air
        seen: set[int] = set()
        now = self.frame
        for e in list(self.enemies.values()) + list(self.radar.values()):
            if e.uid in seen:
                continue
            seen.add(e.uid)
            if not e.is_mobile and e.uid in self.enemies:
                self.note_enemy_structure(e)
            self._stamp_enemy(grid, e, 0.0)
        # Recently seen mobile enemies: still somewhere near where we saw them.
        for uid, (e, f) in self.enemy_memory.items():
            if uid in seen or not e.is_mobile:
                continue
            age = (now - f) / 30.0
            if age > 15.0:
                continue
            seen.add(uid)
            self._stamp_enemy(grid, e, age)
        if now % 90 < 15:
            self._forget_seen_empty_structures()
        if now % 30 < 15:
            self.forget_vanished_enemies()
        for uid, (e, _f) in getattr(self, "enemy_structures", {}).items():
            if uid in seen:
                continue
            self._stamp_enemy(grid, e, 0.0)
        for uid in self.army:
            u = self.units.get(uid)
            if u is None:
                continue
            self._stamp(ours, u.pos, weapon_range(u) + 120.0, ground_dps(u))
        for u in self.units.values():
            if units.is_defense(u.data) and not u.is_mobile and u.is_finished:
                self._stamp(ours, u.pos, weapon_range(u) + 32.0, ground_dps(u))
        self.threat_grid = grid
        self.air_grid = air
        self.own_grid = ours
        self.threat_frame_built = now

    def _hostile(self, e: UnitInfo) -> bool:
        """Mobile combat units, commanders and armed static defences."""
        return self.is_dangerous(e) or (units.is_defense(e.data) and weapon_range(e) > 0)

    def _stamp_enemy(self, grid: np.ndarray, e: UnitInfo, age: float) -> None:
        if not self._hostile(e):
            return
        if e.is_air and e.name.startswith("drone"):
            return
        if not e.is_finished and e.uid in self.enemies:
            return
        info = weapon_info(e)
        dps = float(info.get("dps_ground") or 0.0)
        air_dps = float(info.get("dps_air") or 0.0)
        rng = weapon_range(e)
        if e.uid in self.radar and e.uid not in self.enemies and e.def_id == 0:
            dps, rng = 60.0, 400.0
        speed = float(e.data.get("speed") or 0.0)
        margin = 48.0 if not e.is_mobile else min(300.0, 60.0 + speed * (1.2 + age * 0.25))
        weight = 1.0 if age <= 0 else max(0.2, 1.0 - age / 15.0)
        air_grid = getattr(self, "_air_build", None)
        if air_grid is not None and air_dps > 0 and not e.is_air:
            self._stamp(air_grid, e.pos, rng + margin, air_dps * weight)
        if dps > 0.0:
            self._stamp(grid, e.pos, rng + margin, dps * weight)

    # -------------------------------------------------------------- queries

    def _cell(self, pos: tuple[float, float]) -> tuple[int, int] | None:
        grid = getattr(self, "threat_grid", None)
        if grid is None:
            return None
        h, w = grid.shape
        ix = min(w - 1, max(0, int(pos[0] / CELL)))
        iz = min(h - 1, max(0, int(pos[1] / CELL)))
        return iz, ix

    def threat_at(self, pos: tuple[float, float]) -> float:
        c = self._cell(pos)
        if c is None:
            return 0.0
        return float(self.threat_grid[c])

    def own_power_at(self, pos: tuple[float, float]) -> float:
        c = self._cell(pos)
        if c is None:
            return 0.0
        return float(self.own_grid[c])

    def threat_in_radius(self, pos: tuple[float, float], radius: float) -> float:
        grid = getattr(self, "threat_grid", None)
        if grid is None:
            return 0.0
        h, w = grid.shape
        cx, cz = pos[0] / CELL, pos[1] / CELL
        r = radius / CELL
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
        z0, z1 = max(0, int(cz - r)), min(h, int(cz + r) + 1)
        if x1 <= x0 or z1 <= z0:
            return 0.0
        return float(grid[z0:z1, x0:x1].max())

    def path_threat(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Max threat on the straight segment a->b (sampled per cell)."""
        grid = getattr(self, "threat_grid", None)
        if grid is None:
            return 0.0
        n = max(2, int(self.dist(a, b) / CELL) + 1)
        best = 0.0
        for i in range(n + 1):
            f = i / n
            p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            best = max(best, self.threat_at(p))
        return best

    def safest_point_near(
        self,
        pos: tuple[float, float],
        toward: tuple[float, float] | None = None,
        radius: float = 700.0,
    ) -> tuple[float, float] | None:
        grid = getattr(self, "threat_grid", None)
        if grid is None:
            return None
        best = None
        best_score = 1e18
        for ring in (radius * 0.5, radius):
            for k in range(12):
                a = k * math.pi / 6.0
                p = (pos[0] + math.cos(a) * ring, pos[1] + math.sin(a) * ring)
                if not (40 < p[0] < self.map.width - 40 and 40 < p[1] < self.map.height - 40):
                    continue
                score = self.threat_at(p) * 10.0 + self.path_threat(pos, p) * 4.0
                if toward is not None:
                    score += self.dist(p, toward) * 0.15
                if score < best_score:
                    best_score = score
                    best = p
        if toward is not None and self.threat_at(toward) <= 0.0 and self.path_threat(pos, toward) <= 0.0:
            return toward
        return best

    def local_balance(self, pos: tuple[float, float], radius: float = 650.0) -> tuple[float, float]:
        """(our combat power, enemy combat power) of units within radius."""
        r2 = radius * radius
        ours = 0.0
        for uid in self.army:
            u = self.units.get(uid)
            if u is not None and (u.x - pos[0]) ** 2 + (u.z - pos[1]) ** 2 <= r2:
                ours += combat_power(u)
        for u in self.units.values():
            if units.is_defense(u.data) and not u.is_mobile and u.is_finished:
                if (u.x - pos[0]) ** 2 + (u.z - pos[1]) ** 2 <= r2:
                    ours += combat_power(u)
        theirs = 0.0
        for e in self.contacts().values():
            if not self._hostile(e) or e.is_air and e.name.startswith("drone"):
                continue
            if (e.x - pos[0]) ** 2 + (e.z - pos[1]) ** 2 <= r2:
                theirs += combat_power(e)
        for uid, (e, _f) in getattr(self, "enemy_structures", {}).items():
            if uid in self.enemies:
                continue
            if (e.x - pos[0]) ** 2 + (e.z - pos[1]) ** 2 <= r2:
                theirs += combat_power(e)
        return ours, theirs
