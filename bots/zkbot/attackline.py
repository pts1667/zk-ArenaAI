"""Getting the army to the fight, and in the right order.

`fastgather`: regrouping stopped the attack whenever under 60% of the group stood
within 700 elmos, and the front walked back to the middle.  The radius is wider
now, the group meets ahead of its middle, on the way to the target, and a regroup
is limited in time.

`assaultfirst`: everything was sent to the target at once, so the faster and more
fragile units got there first.  Near the fight, assaults lead and the others hold
behind them.  A hold point stays put until the front has clearly moved on, so
followers aren't given a new order every few seconds.
"""
from __future__ import annotations

from . import units
from .models import UnitInfo
from .params import FLAGS, P
from .threat import weapon_range

Pos = tuple[float, float]

STICK_FRAMES = P("attack.stick", 600, 0, 3600, "frames the main group keeps a target")
GATHER_RADIUS = P("gather.radius", 1100.0, 500.0, 2500.0, "members this close to the centre count as gathered")
GATHER_SHARE = P("gather.share", 0.5, 0.2, 0.9, "share of the group that must be gathered")
REGROUP_FRAMES = P("gather.regroup", 600, 150, 1800, "frames a regroup lasts at most")
REGROUP_REST = P("gather.rest", 1200, 0, 3600, "frames of attacking before the next regroup")
REGROUP_DIST = P("gather.regroup_dist", 900.0, 400.0, 2000.0, "units this far from the meeting point go to it")
MEET_SHARE = P("gather.meet_share", 0.3, 0.0, 0.8, "share of the way to the target the group meets at")
MEET_MAX = P("gather.meet_max", 600.0, 0.0, 2000.0, "elmos towards the target the group meets at most")
RAID_STAGE_RADIUS = P("gather.raid_stage", 1400.0, 500.0, 3000.0, "raiders this close to the stage count as there")
SQUAD_READY_RADIUS = P("gather.squad_ready", 800.0, 300.0, 1500.0, "squads this close to the meeting point are ready")
SQUAD_WAIT = P("gather.squad_wait", 450, 90, 1800, "frames squads wait for each other at most")
SQUAD_STANDOFF = 750.0        # outside the prize's defence

FRONT_ROLES = ("assault", "heavy")
MIN_FRONT = P("line.min_front", 2, 1, 6, "units needed to count as the front")
FRONT_RADIUS = P("line.front_radius", 1500.0, 500.0, 4000.0, "units further than this from the group aren't its front")
RIOT_BEHIND = P("line.riot_behind", 90.0, 0.0, 400.0, "elmos riots keep behind the front")
OTHER_BEHIND = P("line.other_behind", 170.0, 0.0, 500.0, "elmos other units keep behind the front")
ARTY_SHARE = P("line.arty_share", 0.8, 0.3, 1.0, "share of its range artillery keeps behind the front")
NEAR_TARGET = P("line.near_target", 1800.0, 500.0, 5000.0, "the order matters this close to the target")
NEAR_ENEMY = P("line.near_enemy", 1200.0, 400.0, 3000.0, "or with an enemy this close to the front")
HOLD_MOVED = P("line.hold_moved", 300.0, 100.0, 800.0, "the front moves this far before a hold point is updated")


class AttackLineMixin:
    # --- stickytarget ---------------------------------------------------------
    def kept_attack_target(self) -> Pos | None:
        """The main group's target, kept for a while unless it's been found empty.  The
        choice was remade every step, and the whole army's destination jumped with it."""
        if "stickytarget" not in FLAGS:
            return None
        kept = self.__dict__.get("main_target")
        if kept is None or self.frame - kept[1] >= STICK_FRAMES:
            return None
        key = (int(kept[0][0] // 200), int(kept[0][1] // 200))
        return None if key in self.__dict__.get("attack_blacklist", {}) else kept[0]

    def note_attack_target(self, target: Pos | None) -> None:
        kept = self.__dict__.get("main_target")
        if target is None:
            self.main_target = None
        elif kept is None or kept[0] != target:
            self.main_target = (target, self.frame)

    # --- fastgather -----------------------------------------------------------
    def gather_terms(self) -> tuple[float, float]:
        """Radius, and share of the group within it, that make the group gathered."""
        return (GATHER_RADIUS, GATHER_SHARE) if "fastgather" in FLAGS else (700.0, 0.6)

    def regroup_allowed(self, wanted: bool) -> bool:
        """A regroup lasts a limited time, then the attack carries on for a while."""
        if "fastgather" not in FLAGS:
            return wanted
        if self.frame < self.__dict__.get("regroup_until", -1):
            return wanted
        if not wanted or self.frame < self.__dict__.get("regroup_rest_until", -1):
            return False
        self.regroup_until = self.frame + REGROUP_FRAMES
        self.regroup_rest_until = self.regroup_until + REGROUP_REST
        return True

    def meeting_point(self, center: Pos, target: Pos | None) -> Pos:
        """Part of the way from the group's middle to its target."""
        d = self.dist(center, target) if target is not None else 0.0
        if d < 1.0:
            return center
        along = min(MEET_MAX, MEET_SHARE * d)
        return center[0] + (target[0] - center[0]) / d * along, center[1] + (target[1] - center[1]) / d * along

    def squad_meeting_point(self, centers: list[Pos], target: Pos) -> Pos:
        """Between the squads and on towards the target, outside its defence."""
        mx = sum(c[0] for c in centers) / len(centers)
        mz = sum(c[1] for c in centers) / len(centers)
        d = self.dist((mx, mz), target)
        if d < 1.0:
            return mx, mz
        # Half the walk left before the standoff; backwards if already too close.
        left = d - SQUAD_STANDOFF
        along = left * 0.5 if left > 0.0 else left
        return mx + (target[0] - mx) / d * along, mz + (target[1] - mz) / d * along

    # --- assaultfirst ---------------------------------------------------------
    def attack_front(self, members: list[int], center: Pos, target: Pos) -> tuple[tuple[str, ...], float] | None:
        """(roles at the front, how far they are from the target), if the order matters now."""
        cache = self.__dict__.get("_attack_front")
        if cache is not None and cache[0] == self.frame:
            return cache[1]
        found = None
        for roles in (FRONT_ROLES, ("riot",)):
            front = [self.units[uid] for uid in members
                     if units.role_of(self.units[uid].data) in roles
                     and self.dist(self.units[uid].pos, center) < FRONT_RADIUS]
            if len(front) >= MIN_FRONT:
                front.sort(key=lambda u: self.dist(u.pos, target))
                lead = front[len(front) // 3]
                d = self.dist(lead.pos, target)
                if d < NEAR_TARGET or self.enemy_near(lead.pos, NEAR_ENEMY) is not None:
                    found = (roles, d)
                break
        self._attack_front = (self.frame, found)
        return found

    def behind_by(self, u: UnitInfo) -> float:
        role = units.role_of(u.data)
        if role == "riot":
            return RIOT_BEHIND
        if role in ("arty", "heavyarty"):
            return max(OTHER_BEHIND, ARTY_SHARE * weapon_range(u))
        return OTHER_BEHIND

    def hold_behind_front(self, u: UnitInfo, members: list[int], center: Pos, target: Pos) -> Pos | None:
        """Where this unit waits for the front, or None if it should carry on."""
        holds = self.__dict__.setdefault("attack_holds", {})
        front = self.attack_front(members, center, target)
        if front is None or units.role_of(u.data) in front[0]:
            holds.pop(u.uid, None)
            return None
        limit = front[1] + self.behind_by(u)
        held = holds.get(u.uid)
        if held is not None and limit > held[1] - HOLD_MOVED and self.dist(held[2], target) < 1.0:
            return held[0]
        d = self.dist(u.pos, target)
        if d > limit or d < 1.0:
            holds.pop(u.uid, None)
            return None
        point = (target[0] + (u.x - target[0]) / d * limit, target[1] + (u.z - target[1]) / d * limit)
        holds[u.uid] = (point, limit, target)
        return point
