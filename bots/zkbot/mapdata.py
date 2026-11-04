"""Small amount of map analysis used for factory choice and safe building."""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .params import flag


@dataclass
class MetalSpot:
    x: float
    z: float
    income: float
    index: int
    status: str = "free"          # free, building, ours, enemy, blocked
    task_id: str | None = None


@dataclass
class MapData:
    width: float = 0.0
    height: float = 0.0
    heightmap: np.ndarray | None = None
    water_fraction: float = 0.0
    water_damage: float = 0.0
    water_is_harmful: bool = False
    spots: list[MetalSpot] = field(default_factory=list)
    geos: list[tuple[float, float]] = field(default_factory=list)
    wind_min: float = 0.0
    wind_max: float = 0.0
    tidal: float = 0.0
    start_boxes: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return self.width * 0.5, self.height * 0.5

    @property
    def wind_avg(self) -> float:
        if self.wind_min > 0 or self.wind_max > 0:
            return (self.wind_min + self.wind_max) * 0.5
        return 0.0

    def set_map_features(self, data: dict[str, Any]) -> None:
        self.spots.clear()
        self.geos.clear()
        for i, spot in enumerate(data.get("spots") or []):
            try:
                pos = spot.get("pos") or [0, 0, 0]
                resource = (spot.get("resource") or "Metal").lower()
                if resource != "metal":
                    continue
                self.spots.append(
                    MetalSpot(
                        x=float(pos[0]),
                        z=float(pos[2]),
                        income=float(spot.get("income") or 0.0),
                        index=len(self.spots),
                    )
                )
            except Exception:
                continue
        for feat in data.get("features") or []:
            if (feat.get("name") or "").lower().startswith("geovent"):
                try:
                    pos = feat.get("pos") or [0, 0, 0]
                    self.geos.append((float(pos[0]), float(pos[2])))
                except Exception:
                    pass
        self.water_damage = float(data.get("waterDamage") or 0.0)
        self.water_is_harmful = bool(data.get("waterIsHarmful") or False)

    def set_heightmap(self, data: dict[str, Any]) -> None:
        try:
            raw = base64.b64decode(data["data_b64"])
            arr = np.frombuffer(raw, dtype="<f4")
            w = int(data["width"])
            h = int(data["height"])
            if w * h != arr.size:
                return
            self.heightmap = arr.reshape((h, w)).copy()
            # Spring water level is 0; only count meaningful water.
            self.water_fraction = float((self.heightmap < -1.0).mean())
        except Exception:
            self.heightmap = None

    def favours_hover(self, ground: float) -> bool:
        """Is this a water map, or a base on the seabed?  Lava and void read as
        water in the heights, but hovers burn on one and can't enter the other."""
        if flag("safefac") and self.water_is_harmful:
            return False
        return ground < -1.0 or self.water_fraction > 0.25

    def ground_height(self, x: float, z: float) -> float:
        if self.heightmap is None or self.width <= 0 or self.height <= 0:
            return 0.0
        h, w = self.heightmap.shape
        ix = min(w - 1, max(0, int(x / max(1.0, self.width) * w)))
        iz = min(h - 1, max(0, int(z / max(1.0, self.height) * h)))
        return float(self.heightmap[iz, ix])

    def is_water(self, x: float, z: float) -> bool:
        return self.ground_height(x, z) < -1.0

    def nearest_spots(self, x: float, z: float, limit: int = 10) -> list[tuple[float, MetalSpot]]:
        rows = [(math.hypot(s.x - x, s.z - z), s) for s in self.spots]
        rows.sort(key=lambda p: p[0])
        return rows[:limit]

    def is_near_water(self, x: float, z: float, radius: float = 600.0) -> bool:
        if self.heightmap is None:
            return self.water_fraction > 0.05
        h, w = self.heightmap.shape
        sx = self.width / w
        sz = self.height / h
        x0 = max(0, int((x - radius) / max(1.0, self.width) * w))
        x1 = min(w, int((x + radius) / max(1.0, self.width) * w) + 1)
        z0 = max(0, int((z - radius) / max(1.0, self.height) * h))
        z1 = min(h, int((z + radius) / max(1.0, self.height) * h) + 1)
        if x1 <= x0 or z1 <= z0:
            return False
        return bool((self.heightmap[z0:z1, x0:x1] < -1.0).any())
