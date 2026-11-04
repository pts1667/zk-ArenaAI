"""Answering armored heavies such as the Crab (flag `armorcounter`).

What makes one: slow, expensive, a wide blast and an armored form.  Its blast melts
the raiders and skirmishers that beat other slow heavies, such as the Grizzly with
its narrow shot.  It is countered from outside its range (Slings, Phantoms) or by
overwhelming it with assaults (Knights), so production turns to those, raiders stay
away, and the main group goes in only when those units alone outweigh it.
"""
from __future__ import annotations

from functools import lru_cache

from . import catalog, units
from .models import UnitInfo
from .params import FLAGS, P
from .threat import weapon_range

Pos = tuple[float, float]

MAX_SPEED = P("armored.speed", 60.0, 30.0, 90.0, "armored heavies are slower than this")
MIN_COST = P("armored.cost", 800.0, 300.0, 3000.0, "and cost at least this")
MIN_AOE = P("armored.aoe", 100.0, 30.0, 400.0, "and have a blast this wide")
SEEN_FRAMES = P("armored.seen", 9000, 900, 36000, "frames one counts after it was last seen")
PHANTOM_INCOME = P("armored.phantom_income", 15.0, 5.0, 40.0, "metal income per Phantom")
HOLD_SHARE = P("armored.hold", 0.92, 0.6, 1.0, "share of its range a unit shells from")
# Shares of the army while any are about; assault for mass, artillery for range.
MIX = {"raid": 0.12, "skirm": 0.10, "riot": 0.04, "assault": 0.44, "arty": 0.30}
# How much of each role's metal counts against one.  Raiders only feed its blast.
WEIGHT = {"raid": 0.0, "scout": 0.0, "bomb": 0.0, "aa": 0.1, "riot": 0.3, "skirm": 0.5,
          "assault": 1.2, "heavy": 1.2, "arty": 1.0, "snipe": 1.5}


@lru_cache(maxsize=None)
def is_armored_heavy(name: str) -> bool:
    s = catalog.stats(name)
    return (s is not None and "armor" in catalog.tags(name) and 0.0 < s["speed"] < MAX_SPEED
            and s["cost"] >= MIN_COST and s["aoe"] >= MIN_AOE)


class HeavyCounterMixin:
    def armored_heavies(self) -> int:
        """How many we think are about.  They outlast the 60 s enemy memory: one out of
        sight for a minute is still there, and production mustn't flip away from it."""
        if "armorcounter" not in FLAGS:
            return 0
        seen = self.__dict__.setdefault("armored_seen", {})
        for e in self.contacts().values():
            if is_armored_heavy(e.name):
                seen[e.uid] = self.frame
        for uid, frame in list(seen.items()):
            if self.frame - frame > SEEN_FRAMES:
                seen.pop(uid)
        return len(seen)

    def armored_counter_mix(self, desired: dict[str, float]) -> dict[str, float]:
        if not self.armored_heavies():
            return desired
        out = dict(desired)
        out.update(MIX)
        return out

    def armored_phantom_quota(self) -> int:
        """Phantoms for it, without waiting for the usual time and income."""
        count = self.armored_heavies()
        if not count:
            return 0
        return min(count + 1, int(self.metal_income() / PHANTOM_INCOME))

    def ball_is_armored(self, ball: dict | None) -> bool:
        return ball is not None and bool(ball.get("armored"))

    def armored_ball_value(self, uids) -> float:
        total = 0.0
        for uid in uids:
            u = self.units.get(uid)
            if u is not None:
                total += units.cost(u.data) * WEIGHT.get(units.role_of(u.data), 0.5) * max(0.3, u.alive_frac)
        return total

    def shell_point(self, u: UnitInfo, ball: dict) -> Pos | None:
        """Where a unit that outranges the ball fires from, instead of waiting out of range."""
        reach = weapon_range(u)
        if not self.ball_is_armored(ball) or reach <= ball["range"]:
            return None
        bx, bz = ball["pos"]
        d = self.dist(u.pos, ball["pos"])
        if d < 1.0:
            return None
        hold = reach * HOLD_SHARE
        return bx + (u.x - bx) / d * hold, bz + (u.z - bz) / d * hold

    def closes_with_armored_ball(self, u: UnitInfo, p) -> bool:
        """Committed against it, assaults walk in: everything outranges them on the way."""
        ball = p.ball.ball
        return (self.ball_is_armored(ball) and p.posture.attack_target == ball["pos"]
                and units.role_of(u.data) in ("assault", "heavy"))
