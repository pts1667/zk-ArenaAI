"""What the scripts share: `ScriptData`, what the catalog says each kind of unit can do, and the fighters' states and groups.

The army script (policy_army.py) plays the fighting on a side the policy doesn't.  The first, raider-only fight script, built on
the same states, is now only frame_world.py's base for the frames the encoder learns from (`FightScript` there).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import catalog as bot_catalog, units  # noqa: E402
from zkbot.policy import tables  # noqa: E402
from policy_opening import modelled  # noqa: E402

HEAVY_COST = 500.0          # elmos a second: slow heavies are what snipers go for
HEAVY_RIOT_COST = 1000.0    # riot heavies (Sumo, Dante, Crab): raiders never go near these
SLOW_SPEED = 60.0

RETREAT_BELOW = 0.4         # of health.  The rules bot's are 0.30 for cheap units and 0.45 for others.
RECOVERED = 0.9
ARRIVED = 500.0             # elmos from the enemy's start
GROUPS_FROM = 180.0         # seconds of game time
GROUP_FIRST_METAL = 400.0   # at about 17 metal a second, these take 25 s to gather at 3:00 and about 3 minutes at the largest
GROUP_METAL_PER_MINUTE = 200.0
GROUP_MOST_METAL = 3000.0
LONGEST_WAIT = 180.0        # seconds, so that a side that has lost its income or its factory doesn't hold its last fighters for ever

NEW, WAITING, SENT, RETREATING = 0, 1, 2, 3


def group_metal(seconds: float) -> float:
    """What the fighters waiting must be worth before they go.  Nothing before groups start."""
    if seconds < GROUPS_FROM:
        return 0.0
    return min(GROUP_MOST_METAL, GROUP_FIRST_METAL + GROUP_METAL_PER_MINUTE * (seconds - GROUPS_FROM) / 60.0)


HURTS_NOT = 4 | 64 | 256       # sim/include/zksim/capi.h ZK_W_PARALYZER, ZK_W_SLOW_ONLY, ZK_W_DISARM_ONLY: weapons that don't take health


class ScriptData:
    """What the data says about each kind of unit.  Shared by all the scripts of an environment.  Arrays are one longer than the
    table, with the unknown type at the end, which is all False."""

    def __init__(self, table: tables.TypeTable, cat: zksim.Catalog):
        self.table = table
        self.fights = table.combat & table.mobile
        n = len(table.combat)
        self.regenerates = np.zeros(n, bool)
        self.modelled = np.zeros(n, bool)           # the sim does everything the unit does
        self.catalog_row = np.full(n, -1, np.int32)  # its index in the sim's catalog, which orders are given by
        self.extractor = np.zeros(n, bool)
        self.defence = np.zeros(n, bool)
        self.kitable = np.zeros(n, bool)            # a riot or an assault: slow, and can be kited
        self.heavy_riot = np.zeros(n, bool)
        self.slow_heavy = np.zeros(n, bool)
        self.row_of_name = {name: row for row, name in enumerate(table.names)}
        self.role_of = [bot_catalog.role(name) for name in table.names] + ["other"]
        for row, name in enumerate(table.names):
            d = units.by_name(name) or {}
            roles = bot_catalog.roles(name)
            if name in cat.index:
                i = cat.index[name]
                self.regenerates[row] = cat.units[i]["idle_regen"] > 0.0
                self.catalog_row[row] = i
                self.modelled[row] = modelled(cat, name)
            self.extractor[row] = units.is_mex(d)
            self.defence[row] = units.is_defense(d) and bool(table.armed[row])
            self.kitable[row] = bool({"riot", "assault"} & set(roles))
            self.heavy_riot[row] = "riot" in roles and float(table.cost[row]) >= HEAVY_RIOT_COST
            self.slow_heavy[row] = float(table.cost[row]) >= HEAVY_COST and float(table.speed[row]) < SLOW_SPEED
        # What a silo should make: the missile that hurts most, of those that hurt (not the stunning or slowing ones).
        self.best_missile: dict[int, int] = {}
        for row in np.flatnonzero(table.produces.any(axis=1) & ~table.factory):
            best, most = None, 0.0
            for m in table.producible[table.produces[row]]:
                name = table.names[int(m)]
                if name not in cat.index:
                    continue
                for w in cat.units[cat.index[name]]["mount_weapon"]:
                    if w < 0:
                        continue
                    weapon = cat.weapons[int(w)]
                    hurts = float(np.max(weapon["damage"])) if not weapon["flags"] & (HURTS_NOT) else 0.0
                    if hurts > most:
                        best, most = int(m), hurts
            if best is not None:
                self.best_missile[int(row)] = best
        mobile = table.mobile & table.combat
        self.kitable &= mobile
        self.heavy_riot &= mobile
        self.slow_heavy &= mobile

    def row_of_index(self, catalog_index: int) -> int:
        """The table row of a sim catalog index."""
        return int(np.flatnonzero(self.catalog_row == catalog_index)[0])
