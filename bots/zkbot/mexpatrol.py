"""Raiders that guard the economy and do nothing else (flag `mexpatrol`).

About a fifth of the raiders, at least two by 1:30, walk between our extractors and
answer whatever attacks them or our constructors.  They never join a raid or the
main army, so a Flea at an extractor is theirs to deal with and everyone else carries
on.  The first raiders built (`spearhead`) are never taken for this.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo
from .params import FLAGS, P

SHARE = P("mexpatrol.share", 0.22, 0.0, 0.5, "share of raiders on patrol")
MIN_RAIDERS = P("mexpatrol.min", 2, 0, 6, "raiders on patrol, at least")
DUE_SECONDS = P("mexpatrol.due", 90.0, 30.0, 300.0, "the minimum is in place by then")
LEAD_SECONDS = P("mexpatrol.lead", 30.0, 0.0, 120.0, "production starts this long before it's due")
THREAT_RADIUS = P("mexpatrol.threat_radius", 900.0, 300.0, 2000.0, "enemy this close to an extractor or constructor")
ENEMY_RADIUS = P("mexpatrol.enemy_radius", 500.0, 200.0, 1000.0, "defence counted around a threat")
HELP_RADIUS = P("mexpatrol.help_radius", 2000.0, 500.0, 4000.0, "patrols this close to a threat count against it")
REACHED = P("mexpatrol.reached", 200.0, 80.0, 400.0, "elmos that count as having visited an extractor")
MEX_CLEAR = P("mexpatrol.mex_clear", 500.0, 200.0, 1000.0, "walk to no extractor with an enemy this close")
PATROL_SPEED = 120.0          # elmos a second, to weigh a walk against time unvisited


class MexPatrolMixin:
    mex_patrol: set[int] = frozenset()

    def update_mex_patrol(self, raider_uids: list[int], spear: set[int]) -> set[int]:
        """Membership lasts until the raider dies.  The closest to home fill gaps."""
        if "mexpatrol" not in FLAGS:
            return set()
        alive = set(raider_uids)
        keep = {uid for uid in self.mex_patrol if uid in alive}
        want = max(int(MIN_RAIDERS), round(SHARE * len(alive)))
        spare = sorted((uid for uid in alive - keep - spear if uid not in self.retreating),
                       key=lambda uid: self.dist(self.units[uid].pos, self.base_pos))
        keep |= set(spare[:max(0, want - len(keep))])
        self.mex_patrol = keep
        return keep

    def patrol_short(self) -> bool:
        """The minimum patrol is coming due and the raiders for it don't exist."""
        return ("mexpatrol" in FLAGS and self.frame_time() >= DUE_SECONDS - LEAD_SECONDS
                and len(self.mex_patrol) < MIN_RAIDERS)

    def own_mexes(self) -> list[UnitInfo]:
        return [u for u in self.units.values() if u.name == "staticmex" and u.is_finished]

    def economy_threats(self) -> list[tuple[float, float]]:
        """Where enemies are at our extractors and constructors, this frame."""
        cache = self.__dict__.get("_patrol_threats")
        if cache is not None and cache[0] == self.frame:
            return cache[1]
        assets = [u.pos for u in self.units.values()
                  if (u.name == "staticmex" and u.is_finished) or (u.is_builder and u.is_mobile)]
        found = [w["pos"] for w in self.update_raid_warnings()]
        for e in self.contacts().values():
            if (self.is_dangerous(e) and e.is_mobile and not e.is_air
                    and any(self.dist(e.pos, a) < THREAT_RADIUS for a in assets)):
                found.append(e.pos)
        self._patrol_threats = (self.frame, found)
        return found

    def patrol_can_take(self, pos: tuple[float, float]) -> bool:
        value = sum(units.cost(self.units[uid].data) for uid in self.mex_patrol
                    if uid in self.units and self.dist(self.units[uid].pos, pos) < HELP_RADIUS)
        return self.raid_defense_value(pos, ENEMY_RADIUS) <= value

    def patrol_help_power(self, pos: tuple[float, float]) -> float:
        """Patrols that can reach a threat: the army needn't come back for it."""
        from .threat import combat_power

        return sum(combat_power(self.units[uid]) for uid in self.mex_patrol
                   if uid in self.units and self.dist(self.units[uid].pos, pos) < HELP_RADIUS)

    def next_patrol_stop(self, uid: int, pos: tuple[float, float]) -> tuple[float, float]:
        """The extractor longest unvisited, allowing for the walk."""
        visited = self.__dict__.setdefault("patrol_visited", {})
        stops = self.__dict__.setdefault("patrol_stops", {})
        taken = {s for other, s in stops.items() if other != uid}
        mexes = {m.uid: m for m in self.own_mexes()}
        for mid, m in mexes.items():
            if self.dist(pos, m.pos) < REACHED:
                visited[mid] = self.frame
        current = stops.get(uid)
        if current in mexes and visited.get(current, -1) != self.frame and self.enemy_near(mexes[current].pos, MEX_CLEAR) is None:
            return mexes[current].pos
        best, best_score = None, -math.inf
        for mid, m in mexes.items():
            if mid in taken or self.enemy_near(m.pos, MEX_CLEAR) is not None:
                continue
            score = (self.frame - visited.get(mid, 0)) / 30.0 - self.dist(pos, m.pos) / PATROL_SPEED
            if score > best_score:
                best, best_score = mid, score
        if best is None:
            stops.pop(uid, None)
            return self.base_pos
        stops[uid] = best
        return mexes[best].pos

    def beh_mex_patrol(self, u: UnitInfo, p) -> str | None:
        if u.uid not in self.mex_patrol:
            return None
        targets = [t for t in self.economy_threats() if self.patrol_can_take(t)]
        if targets:
            self.order_fight(u.uid, min(targets, key=lambda t: self.dist(t, u.pos)))
            return "patrol_respond"
        self.order_fight(u.uid, self.next_patrol_stop(u.uid, u.pos))
        return "patrol"
