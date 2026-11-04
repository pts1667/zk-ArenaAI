"""Answering a small raid without recalling the army (flag `nearbuild`).

A Flea at an extractor put the whole army in defend mode.  A threat that what's
already nearby can handle, counting a combat unit about to leave the factory,
stays with the responders.  While there's a threat, a factory building a combat
unit gets high build priority, so that unit arrives first when metal is short.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo
from .params import FLAGS, P
from .threat import combat_power, ground_dps

CMD_PRIORITY = 34220          # Zero-K unit_priority gadget
PRIORITY_NORMAL, PRIORITY_HIGH = 1, 2

SOON_SECONDS = P("nearbuild.soon", 5.0, 1.0, 15.0, "a unit finishing within this counts as already here")
FACTORY_RADIUS = P("nearbuild.factory_radius", 150.0, 60.0, 300.0, "a nanoframe this close is in that factory")
THREAT_RADIUS = P("nearbuild.threat_radius", 900.0, 300.0, 1500.0, "units this close to a threat can answer it")
COMMANDER_RADIUS = P("nearbuild.commander_radius", 1200.0, 300.0, 2500.0, "commander this close to a threat can answer it")
RECENT_FRAMES = P("nearbuild.recent", 300, 30, 1800, "frames a threat keeps a factory on high priority")
RATE_FRAMES = 15              # frames between build rate measurements


class NearBuildMixin:
    def factory_builds(self) -> list[tuple[UnitInfo, UnitInfo]]:
        """(factory, combat unit it's building)."""
        frames = [u for u in self.units.values()
                  if u.is_being_built and u.is_combat and not u.is_builder]
        out = []
        for f in self.units.values():
            if not (f.is_factory and f.is_finished):
                continue
            near = [u for u in frames if self.dist(u.pos, f.pos) < FACTORY_RADIUS]
            if near:
                out.append((f, max(near, key=lambda u: u.build_progress)))
        return out

    def build_seconds_left(self, factory: UnitInfo, u: UnitInfo) -> float:
        """Seconds until done: what the factory could do, or what it's really doing
        if that's slower, as it is when metal is short."""
        left = max(0.0, 1.0 - u.build_progress)
        speed = float(factory.data.get("buildSpeed") or 0.0)
        seconds = left * float(u.data.get("buildTime") or 0.0) / speed if speed > 0.0 else math.inf
        marks = self.__dict__.setdefault("_build_marks", {})
        mark = marks.get(u.uid)
        if mark is None:
            marks[u.uid] = (self.frame, u.build_progress, None)
            return seconds
        frame, progress, rate = mark
        if self.frame - frame >= RATE_FRAMES:
            rate = (u.build_progress - progress) / (self.frame - frame)
            marks[u.uid] = (self.frame, u.build_progress, rate)
        if rate is None:
            return seconds
        return max(seconds, left / rate / 30.0) if rate > 0.0 else math.inf

    def imminent_power(self) -> float:
        """Combat power about to leave the factories."""
        total = 0.0
        for f, u in self.factory_builds():
            if self.build_seconds_left(f, u) <= SOON_SECONDS:
                # Its health is still low, so go by what it will be.
                total += math.sqrt(max(0.0, ground_dps(u)) * float(u.data.get("health") or 0.0))
        return total

    def minor_threat(self, pos: tuple[float, float]) -> bool:
        """Can the units around this threat, and any about to appear, handle it?"""
        if "nearbuild" not in FLAGS:
            return False
        ours, theirs = self.local_balance(pos, THREAT_RADIUS)
        for uid in self.spearhead:
            u = self.units.get(uid)
            if u is not None and self.dist(u.pos, pos) <= THREAT_RADIUS:
                ours -= combat_power(u)      # they aren't coming back for it
        commander = self.units.get(self.commander_id)
        if commander is not None and self.dist(commander.pos, pos) <= COMMANDER_RADIUS:
            ours += combat_power(commander)
        return theirs <= ours + self.imminent_power() + self.patrol_help_power(pos)

    def threat_active(self) -> bool:
        if self.threat_pos is not None and self.frame - self.threat_frame < RECENT_FRAMES:
            return True
        return bool(self.update_raid_warnings())

    def update_factory_priority(self) -> None:
        if "nearbuild" not in FLAGS:
            return
        boosted = self.__dict__.setdefault("boosted_factories", set())
        threat = self.threat_active()
        building = {f.uid for f, _u in self.factory_builds()} if threat else set()
        for uid in sorted(building - boosted):
            self._set_priority(uid, PRIORITY_HIGH)
            boosted.add(uid)
            self.log(f"factory {uid} on high priority: threat with a combat unit in production")
        for uid in sorted(boosted - building):
            if uid in self.units:
                self._set_priority(uid, PRIORITY_NORMAL)
            boosted.discard(uid)

    def _set_priority(self, uid: int, level: int) -> None:
        self.queue({"type": "custom", "unitId": int(uid), "cmdId": CMD_PRIORITY, "params": [float(level)]})
