"""What the bootstrap says of the map: its size, metal spots and liquid, for the host and the policy's map bundle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetalSpot:
    x: float
    z: float
    income: float
    index: int
    status: str = "free"          # ours, enemy or free, as the host last saw it


@dataclass
class MapData:
    width: float = 0.0
    height: float = 0.0
    water_damage: float = 0.0
    spots: list[MetalSpot] = field(default_factory=list)
    start_boxes: dict[str, Any] = field(default_factory=dict)

    def set_map_features(self, data: dict[str, Any]) -> None:
        self.spots.clear()
        for spot in data.get("spots") or []:
            try:
                pos = spot.get("pos") or [0, 0, 0]
                if (spot.get("resource") or "Metal").lower() != "metal":
                    continue
                self.spots.append(MetalSpot(x=float(pos[0]), z=float(pos[2]), income=float(spot.get("income") or 0.0), index=len(self.spots)))
            except Exception:
                continue
        self.water_damage = float(data.get("waterDamage") or 0.0)
