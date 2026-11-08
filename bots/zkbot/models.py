"""A unit as the host keeps it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import units


@dataclass
class UnitInfo:
    uid: int
    def_id: int
    name: str
    data: dict[str, Any]
    x: float
    y: float
    z: float
    health: float
    max_health: float
    build_progress: float
    is_being_built: bool
    experience: float = 0.0
    first_seen: int = 0
    finished_frame: int = -1

    @property
    def pos(self) -> tuple[float, float]:
        return self.x, self.z

    @property
    def alive_frac(self) -> float:
        return self.health / max(1.0, self.max_health)

    @property
    def is_finished(self) -> bool:
        return (not self.is_being_built) and self.build_progress >= 0.985

    @property
    def is_factory(self) -> bool:
        return units.is_factory(self.data)

    @property
    def is_builder(self) -> bool:
        return units.is_builder(self.data)

    @property
    def is_mobile(self) -> bool:
        return bool(self.data.get("isMobile"))

    @property
    def is_combat(self) -> bool:
        return units.is_combat(self.data)

    @property
    def is_air(self) -> bool:
        return units.is_air(self.data)
