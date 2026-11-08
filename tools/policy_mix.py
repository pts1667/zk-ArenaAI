"""What a side's factory builds next, and when that changes.

The sim's factory keeps building the type it was last told to, with no queue, and a PRODUCE order only sets the type for the unit
after the one in progress.  So a choice is made each time a unit starts: the script works out what the factory should build next,
and orders a PRODUCE only if that's not the type the factory is on.  Each such order is a choice a recorder can read
(`SimEnv.script_record`), and no other order changes what the factory builds.

The shares follow CircuitAI's factory.json, which gives each factory its unit mix by time, with the user's Zero-K rules on top.
Raiders come first and in numbers: riots only start at about four minutes, however many raiders there are.  Skirmishers
wait until slow units that can be kited (riots and assaults) have been seen.  AA waits until the enemy has been seen with air.
Artillery comes late, and a scout every few minutes from two.  Each unit is chosen by its catalog role, never by name: the
cheapest unit on the factory's roster with that role that the sim models.

A role is built when it's furthest behind its share of the units the side has started.  Losses don't change the mix, so a side that
loses its raiders keeps building raiders in the same proportion as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from zkbot import catalog as bot_catalog

if TYPE_CHECKING:
    from policy_script import ScriptData

SCOUT = "scout"
# Each role's share by minute, interpolated between the points and flat past the last.  Raiders are the whole factory to begin with;
# the other roles wait for the minutes CircuitAI's early tiers also wait for.
SCHEDULE: dict[str, tuple[tuple[float, float], ...]] = {
    "raid": ((0.0, 1.0), (4.0, 1.0), (10.0, 0.5), (20.0, 0.4), (60.0, 0.35)),
    "riot": ((0.0, 0.0), (4.0, 0.0), (10.0, 0.3), (60.0, 0.3)),
    "assault": ((0.0, 0.0), (6.0, 0.0), (12.0, 0.25), (60.0, 0.25)),
    "arty": ((0.0, 0.0), (12.0, 0.0), (18.0, 0.1), (60.0, 0.1)),
}
SKIRM_SHARE = 0.25          # once the enemy has been seen with a riot or an assault: slow units that can be kited
AA_SHARE = 0.12             # once the enemy has been seen with air
SNIPE_SHARE = 0.10          # once the enemy has been seen with a slow heavy, and from SNIPE_FROM on
SNIPE_FROM = 480.0         # seconds
SCOUTS_FROM = (120.0, 600.0)     # one scout from 2 minutes, a second from 10: the odd one, as CircuitAI has them
ROLE_ORDER = ("raid", "riot", "assault", "skirm", "arty", "aa", "snipe")     # ties in the deficit go to the first


def scout_quota(seconds: float) -> int:
    return sum(seconds >= t for t in SCOUTS_FROM)


@dataclass
class Seen:
    """What the enemy has been seen fielding.  It stays seen: a side that has met air keeps its AA."""
    air: bool = False
    kitable: bool = False
    heavy: bool = False


def shares(seconds: float, seen: Seen) -> dict[str, float]:
    """The target share of each role at this time, before any role that can't be built is left out."""
    minutes = seconds / 60.0
    out = {role: float(np.interp(minutes, [m for m, _ in points], [w for _, w in points])) for role, points in SCHEDULE.items()}
    if seen.kitable:
        out["skirm"] = SKIRM_SHARE
    if seen.air:
        out["aa"] = AA_SHARE
    if seen.heavy and seconds >= SNIPE_FROM:
        out["snipe"] = SNIPE_SHARE
    return out


@dataclass
class Mix:
    """One side's production: what it has started, and what it has seen."""
    data: "ScriptData"
    made: dict[str, int] = field(default_factory=dict)
    seen: Seen = field(default_factory=Seen)
    _menus: dict[int, dict[str, int]] = field(default_factory=dict)

    def observe(self, enemies: np.ndarray) -> None:
        """Note what the enemy fields.  Units whose type isn't known count for nothing."""
        if not len(enemies):
            return
        t = self.data.table
        rows = t.rows(enemies["def_id"])
        mobile = t.mobile[rows]
        self.seen.air |= bool((mobile & t.air[rows]).any())
        self.seen.kitable |= bool(self.data.kitable[rows].any())
        self.seen.heavy |= bool(self.data.slow_heavy[rows].any())

    def menu(self, factory_row: int) -> dict[str, int]:
        """The table row of the cheapest unit this factory builds for each role, among those the sim models."""
        if factory_row not in self._menus:
            t, data = self.data.table, self.data
            best: dict[str, int] = {}
            for name in bot_catalog.roster(t.names[factory_row]):
                row = data.row_of_name.get(name)
                if row is None or not data.modelled[row]:
                    continue
                role = data.role_of[row]
                if role not in best or t.cost[row] < t.cost[best[role]]:
                    best[role] = row
            self._menus[factory_row] = best
        return self._menus[factory_row]

    def choose(self, factory_row: int, seconds: float) -> int | None:
        """The table row of the unit this factory should start next, or None when none of its roles is due."""
        menu = self.menu(factory_row)
        if SCOUT in menu and self.made.get(SCOUT, 0) < scout_quota(seconds):
            return menu[SCOUT]
        share = {role: w for role, w in shares(seconds, self.seen).items() if w > 0.0 and role in menu}
        if not share:
            return None
        total = sum(share.values())
        started = sum(self.made.get(role, 0) for role in ROLE_ORDER)

        def behind(role: str) -> float:
            return share[role] / total * (started + 1) - self.made.get(role, 0)

        return menu[max(share, key=lambda role: (behind(role), -ROLE_ORDER.index(role)))]

    def record(self, role: str) -> None:
        """A unit of this role has started."""
        self.made[role] = self.made.get(role, 0) + 1
