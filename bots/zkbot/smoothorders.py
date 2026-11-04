"""Fewer repeated orders, so units keep their momentum (flag `smoothorders`).

Every new order makes the engine ask for a new path, and a unit slows down while
it waits for one.  Orders went out every 3 seconds even when nothing had changed,
and sooner if the destination shifted by 150 elmos, however far away it was.  Now
an order stands until the destination moves by a good share of the distance left,
with a slow refresh in case the engine lost it.

Most orders that went out were not small shifts but plans changing their mind:
destinations jumping 1000+ elmos between patrol stops, raid targets and attack
targets.  So a unit sent somewhere by one of those plans goes on for a few seconds
before another of them may redirect it.  Reactions (retreating, kiting, raider
micro, shooting something) aren't held back.

The counts in the "army mode" log line measure this, with or without the flag.
"""
from __future__ import annotations

import math

from .params import FLAGS, P

MIN_SHIFT = P("smooth.min_shift", 150.0, 50.0, 400.0, "a shift this small never needs a new order")
SHIFT_SHARE = P("smooth.shift_share", 0.25, 0.0, 0.6, "share of the distance left the destination may shift")
REFRESH_FRAMES = P("smooth.refresh", 450, 90, 1800, "frames before repeating an unchanged order")
SWITCH_FRAMES = P("smooth.switch", 60, 0, 300, "frames before changing move to fight, or back, for the same place")
COMMIT_FRAMES = P("smooth.commit", 150, 0, 600, "frames a plan's order stands before a plan may redirect the unit")
ARRIVED = P("smooth.arrived", 200.0, 50.0, 500.0, "a unit this close to its destination is free for its next order")
ATTACK_COMMIT_FRAMES = P("smooth.attack_commit", 60, 0, 300, "frames a plan leaves a unit shooting a living target")
PLANNED = frozenset(("beh_mex_patrol", "beh_home_guard", "beh_respond", "beh_search", "beh_raider_squad",
                     "beh_ball_gather", "beh_regroup", "beh_defend", "beh_attack", "beh_rally"))
SOON_FRAMES = 90              # for the counts only: the time a unit needs to get going again
JUMP = 1000.0                 # for the counts only
ATTACK_REFRESH_FRAMES = P("smooth.attack_refresh", 300, 120, 900, "frames before repeating an attack on the same target")


class SmoothOrdersMixin:
    def order_settled(self, uid: int, typ: str, pos: tuple[float, float], old: tuple | None) -> bool:
        """Does the unit's current order already do this well enough?"""
        if old is None:
            return False
        age = self.frame - old[4]
        shift = math.hypot(old[1] - pos[0], old[2] - pos[1])
        u = self.units.get(uid)
        if self.order_committed(u, typ, old, age):
            return True
        left = self.dist(u.pos, pos) if u is not None else 0.0
        tolerance = max(MIN_SHIFT, SHIFT_SHARE * left)
        if old[0] == typ:
            return shift < tolerance and age < REFRESH_FRAMES
        return old[0] in ("move", "fight") and typ in ("move", "fight") and shift < tolerance and age < SWITCH_FRAMES

    def order_committed(self, u, typ: str, old: tuple, age: int) -> bool:
        """Still on the way where a plan sent it, and only a plan wants it elsewhere."""
        if self.__dict__.get("order_source") not in PLANNED or age >= COMMIT_FRAMES or u is None:
            return False
        if old[0] == "attack":
            # A target just out of reach, then back in, flipped attack and fight.
            return age < ATTACK_COMMIT_FRAMES and old[3] in self.__dict__.get("enemies", {})
        if old[0] not in ("move", "fight") or typ not in ("move", "fight"):
            return False
        return self.dist(u.pos, (old[1], old[2])) > ARRIVED

    def release_orders(self, uids) -> None:
        """These units' next orders go out at once: the plan has changed for good reason."""
        for uid in uids:
            self.unit_order.pop(uid, None)

    def attack_refresh_frames(self) -> int:
        return ATTACK_REFRESH_FRAMES if "smoothorders" in FLAGS else 120

    def count_order(self, typ: str, old: tuple | None, pos: tuple[float, float] | None = None,
                    target: int | None = None) -> None:
        """Why orders go out: first, repeat, shift, jump (a shift of 1000+), retarget or
        switch of kind, in total and by the behaviour that gave the order."""
        if old is None:
            why = "first"
        elif old[0] != typ:
            why = "switch"
        elif target is not None:
            why = "repeat" if old[3] == target else "retarget"
        elif pos is None or math.hypot(old[1] - pos[0], old[2] - pos[1]) < 1.0:
            why = "repeat"
        elif math.hypot(old[1] - pos[0], old[2] - pos[1]) < JUMP:
            why = "shift"
        else:
            why = "jump"
        counts = self.__dict__.setdefault("order_counts", {})
        counts[why] = counts.get(why, 0) + 1
        if old is not None and len(old) > 4 and self.frame - old[4] < SOON_FRAMES:
            counts["soon"] = counts.get("soon", 0) + 1
        source = f"{self.__dict__.get('order_source', '?').removeprefix('beh_')}:{why}"
        by = self.__dict__.setdefault("order_sources", {})
        by[source] = by.get(source, 0) + 1

    def top_order_sources(self, n: int = 10) -> dict[str, int]:
        by = self.__dict__.get("order_sources", {})
        return dict(sorted(by.items(), key=lambda kv: -kv[1])[:n])
