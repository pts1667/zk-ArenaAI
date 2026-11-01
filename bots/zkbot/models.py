"""Data structures shared by the bot modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import units


def _int(value: float) -> int:
    return int(round(float(value)))


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

@dataclass
class Task:
    key: str
    kind: str
    def_id: int
    x: float
    z: float
    priority: float
    facing: int = 0
    assignees: set[int] = field(default_factory=set)
    created: float = 0.0
    placed: bool = False
    placement_attempts: int = 0
    last_placement_frame: int = -10**9
    spot_index: int | None = None
    anchor: tuple[float, float] | None = None
    max_assignees: int = 1
    unplaceable: bool = False
    created_frame: int = 0
