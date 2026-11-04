"""Keeping builders out of lava and void, using the terrain model.

The engine paths ground units through lava at a cost, so a constructor sent to
metal on an isolated island walks in and burns.  Built only on maps with harmful
liquid (flag `avoidlava`), so other maps pay nothing for it.
"""
from __future__ import annotations

import base64
import time
from typing import Any, Callable, Iterable

import numpy as np

from . import units

# The bridge's unit defs don't carry the move def, so go by unit family.  Anything
# else (cloak, shield, jump, commanders) walks like a kbot.
FAMILIES = (("spider", "spider"), ("tank", "tank"), ("veh", "tank"), ("hover", "hover"),
            ("amph", "amph"), ("ship", "ship"), ("sub", "ship"))
NOT_SET = -12345.0
FOOTPRINT = 40.0              # elmos around a building site that must be clear


def move_class(name: str) -> str | None:
    """TerrainMap movement class of a unit; None if it flies."""
    if units.is_air(units.by_name(name)):
        return None
    for prefix, cls in FAMILIES:
        if name.startswith(prefix):
            return cls
    return "kbot"


def _grid(res: dict[str, Any]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(res["data_b64"]), dtype="<f4")
    return raw.reshape(int(res["height"]), int(res["width"])).copy()


class Hazards:
    def __init__(self, terrain, log: Callable[[str], None] = lambda _msg: None):
        self.terrain = terrain
        self.log = log
        self._areas: dict[tuple[str, int, int], int | None] = {}
        self._skipped = -1

    @classmethod
    def build(cls, client, game_map, log: Callable[[str], None]) -> "Hazards | None":
        """Reads live heights, since gadgets can move the terrain after the bootstrap
        snapshot.  Any failure leaves the bot playing as it would without the flag."""
        from .terrain import TerrainMap

        t0 = time.time()
        try:
            height = _grid(client.query({"type": "height_map"}, timeout=30.0))
            slope = _grid(client.query({"type": "slope_map"}, timeout=30.0))
            t1 = time.time()
            # Set only where Zero-K's map_lava gadget runs, which spares hovers.
            level = client.query({"type": "game_rules_param", "key": "lavaLevel", "defaultFloat": NOT_SET})
            hover_safe = float(level.get("floatValue", NOT_SET)) != NOT_SET
            terrain = TerrainMap(height, slope, spots=[(s.x, s.z, s.income) for s in game_map.spots],
                                 water_damage=game_map.water_damage, hover_safe=hover_safe)
        except Exception as exc:
            log(f"avoidlava: no terrain ({exc!r}); playing without it")
            return None
        s = terrain.summary()
        log(f"avoidlava: {terrain.liquid} damage={terrain.water_damage:.0f} hover_safe={hover_safe} "
            f"query={t1 - t0:.2f}s build={time.time() - t1:.2f}s regions={s['regions']} chokes={s['chokes']} "
            f"ramps={s['ramps']} walls={s['walls']:.3f} below0={s['water']:.3f} "
            f"h_mean={float(height.mean()):.2f} slope_mean={float(slope.mean()):.5f}")
        return cls(terrain, log)

    def _area(self, cls: str, x: float, z: float) -> int | None:
        key = (cls, int(x) // 16, int(z) // 16)
        if key not in self._areas:
            found = self.terrain.area_of(cls, x, z)
            self._areas[key] = None if found is None else found[0]
        return self._areas[key]

    def can_walk(self, name: str, pos: tuple[float, float], target: tuple[float, float]) -> bool:
        """Can this unit get there without entering lava or void?"""
        cls = move_class(name)
        if cls is None:
            return True
        there = self._area(cls, *target)
        if there is None:
            return False
        here = self._area(cls, *pos)
        return here is None or here == there   # a unit on odd ground gets to pick its way off

    def site_bad(self, x: float, z: float) -> bool:
        """Is a building here in or against the liquid?"""
        at = self.terrain.hazard_at
        return any(at(x + dx, z + dz) for dx, dz in
                   ((0.0, 0.0), (-FOOTPRINT, 0.0), (FOOTPRINT, 0.0), (0.0, -FOOTPRINT), (0.0, FOOTPRINT)))

    def spots_in_reach(self, spots: Iterable, walkers: Iterable) -> list:
        """The metal spots that some mobile builder can get to."""
        spots = list(spots)
        walkers = list(walkers)
        out = [s for s in spots if any(self.can_walk(u.name, u.pos, (s.x, s.z)) for u in walkers)]
        skipped = len(spots) - len(out)
        if skipped != self._skipped:
            self._skipped = skipped
            self.log(f"avoidlava: {skipped} of {len(spots)} free spots out of reach")
        return out
