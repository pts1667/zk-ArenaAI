"""The fight script: what plays the fighting on a side the policy doesn't.

Each fighter is in one of three states, and the side remembers whether it has reached the enemy's start.

  sent        attack-moves to the enemy's start, and once any of the side's fighters has got there, to the enemy's nearest building
  waiting     attack-moves to the rally point, until those waiting are worth a group: from 3:00 on, a fighter joins this before it's sent
  retreating  moves to the rally point at low health if it regenerates, until it has recovered

Groups are in metal rather than units, so that they take as long to gather whichever factory the side drew.

It knows where the enemy's buildings are without scouting for them: it's a training opponent, and what it's for is finishing games.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf, tables  # noqa: E402

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


class ScriptData:
    """What the data says about each kind of unit.  Shared by all the scripts of an environment."""

    def __init__(self, table: tables.TypeTable, cat: zksim.Catalog):
        self.table = table
        self.fights = table.combat & table.mobile
        self.regenerates = np.zeros(len(table.combat), bool)
        for row, name in enumerate(table.names):
            if name in cat.index:
                self.regenerates[row] = cat.units[cat.index[name]]["idle_regen"] > 0.0


def _nearest(x: np.ndarray, z: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, where the nearest of the others is."""
    i = np.hypot(others["x"][None, :] - x[:, None], others["z"][None, :] - z[:, None]).argmin(axis=1)
    return others["x"][i], others["z"][i]


@dataclass
class FightScript:
    """One side of one game."""
    data: ScriptData
    rally: tuple[float, float]
    enemy_start: tuple[float, float]
    state: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int8))     # by uid
    waiting_since: float | None = None      # when the first of those waiting began to
    reached_start: bool = False

    def _rejoining(self, seconds: float) -> int:
        return WAITING if seconds >= GROUPS_FROM else SENT

    def _release_group(self, states: np.ndarray, costs: np.ndarray, seconds: float) -> None:
        waiting = states == WAITING
        if not waiting.any():
            self.waiting_since = None
            return
        if self.waiting_since is None:
            self.waiting_since = seconds
        if costs[waiting].sum() >= group_metal(seconds) or seconds - self.waiting_since >= LONGEST_WAIT:
            states[waiting] = SENT
            self.waiting_since = None

    def _targets(self, fighters: np.ndarray, states: np.ndarray, enemies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where each fighter that's been sent should be heading."""
        x, z = fighters["x"], fighters["z"]
        tx, tz = np.full(len(x), self.enemy_start[0], np.float32), np.full(len(x), self.enemy_start[1], np.float32)
        sent = states == SENT
        if not self.reached_start:
            self.reached_start = bool((np.hypot(x[sent] - tx[sent], z[sent] - tz[sent]) < ARRIVED).any())
        if self.reached_start and sent.any() and len(enemies):
            buildings = enemies[~self.data.table.mobile[self.data.table.rows(enemies["def_id"])]]
            tx[sent], tz[sent] = _nearest(x[sent], z[sent], buildings if len(buildings) else enemies)
        return tx, tz

    def orders(self, own: np.ndarray, enemies: np.ndarray, seconds: float) -> np.ndarray:
        """Orders for this side's fighters, given all of its own records and all of the enemy's."""
        rows = self.data.table.rows(own["def_id"])
        finished = self.data.fights[rows] & ((own["flags"] & rf.BEING_BUILT) == 0)
        fighters, rows = own[finished], rows[finished]
        out = np.zeros(len(fighters), zksim.ORDER_DTYPE)
        if not len(fighters):
            self.waiting_since = None
            return out
        uids = fighters["uid"]
        if uids.max() >= len(self.state):
            self.state = np.resize(self.state, 2 * int(uids.max()) + 2)
            self.state[len(self.state) // 2:] = NEW
        states = self.state[uids]
        rejoining = self._rejoining(seconds)
        states[states == NEW] = rejoining
        hurt = fighters["hp_frac"]
        states[self.data.regenerates[rows] & (hurt < RETREAT_BELOW)] = RETREATING
        states[(states == RETREATING) & (hurt >= RECOVERED)] = rejoining
        self._release_group(states, self.data.table.cost[rows], seconds)
        tx, tz = self._targets(fighters, states, enemies)
        at_rally = states != SENT
        tx[at_rally], tz[at_rally] = self.rally
        self.state[uids] = states
        out["unit"], out["target"] = uids, -1
        out["kind"] = np.where(states == RETREATING, rf.MOVE, rf.FIGHT)
        out["x"], out["z"] = tx, tz
        return out
