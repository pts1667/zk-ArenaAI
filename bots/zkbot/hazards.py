"""The map's terrain model from live heights, for the policy's map bundle (policy/runtime.py), and each unit's movement class."""
from __future__ import annotations

import base64
import time
from typing import Any, Callable

import numpy as np

from . import units

# The bridge's unit defs don't carry the move def, so go by unit family.  Anything
# else (cloak, shield, jump, commanders) walks like a kbot.
FAMILIES = (("spider", "spider"), ("tank", "tank"), ("veh", "tank"), ("hover", "hover"),
            ("amph", "amph"), ("ship", "ship"), ("sub", "ship"))
NOT_SET = -12345.0


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

    @classmethod
    def build(cls, client, game_map, log: Callable[[str], None]) -> "Hazards | None":
        """Reads live heights, since gadgets can move the terrain after the bootstrap snapshot.  None on any failure."""
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
            log(f"terrain: none ({exc!r})")
            return None
        s = terrain.summary()
        log(f"terrain: {terrain.liquid} damage={terrain.water_damage:.0f} hover_safe={hover_safe} "
            f"query={t1 - t0:.2f}s build={time.time() - t1:.2f}s regions={s['regions']} chokes={s['chokes']} "
            f"ramps={s['ramps']} walls={s['walls']:.3f} below0={s['water']:.3f} "
            f"h_mean={float(height.mean()):.2f} slope_mean={float(slope.mean()):.5f}")
        return cls(terrain, log)
