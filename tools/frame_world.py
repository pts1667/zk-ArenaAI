"""What frames are collected from has in common: which maps, where things can stand on them, and a side that does more than the training script's.

Roster      what can be on the board: factories and their units, and what builders can build
BaseScript  a side's factories and builders: changing what's produced, taking spots, building energy, turrets and factories, reclaiming, helping
FightScript the first fight script, raider-only: fighters attack-move to the enemy's start, in groups by metal from 3:00, and retreat to
            regenerate.  It knows where the enemy's buildings are without scouting: it's a training opponent, there to finish games.
FrameFight  the fight script, for units that can't all go to the same places: a ship's target is the nearest water to it
FrameGames  whole games on any map, with every factory: on the ground, in the air, on the sea
"""
from __future__ import annotations

import pathlib
import sys
import zlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_sim_env as se  # noqa: E402
from policy_opening import Footing, Ground, Opening, footprints  # noqa: E402
from policy_script import ARRIVED, GROUPS_FROM, LONGEST_WAIT, NEW, RECOVERED, RETREAT_BELOW, RETREATING, SENT, WAITING, ScriptData, group_metal  # noqa: E402
import zksim  # noqa: E402
from zkbot import catalog as bot_catalog, units as bot_units  # noqa: E402
from zkbot.policy import bundle as bd, rawframe as rf, tables  # noqa: E402

WATER_MAP = 0.03            # of a map under water, for its pool weight and for thinking of its factories
WATER_MAP_WEIGHT = 1.5      # how much likelier a map with water is drawn: frames with ships are what games on the old maps had none of
EXTRACTOR_REACH = 64.0      # sim/src/economy.cpp


def pool_weights(maps: list) -> np.ndarray:
    """Real and made-up maps half each, if there are both, and maps with water likelier."""
    kinds = [facts.kind for _, _, facts in maps]
    weights = np.array([WATER_MAP_WEIGHT if facts.water >= WATER_MAP else 1.0 for _, _, facts in maps])
    for kind in set(kinds):
        chosen = np.array([k == kind for k in kinds])
        weights[chosen] /= weights[chosen].sum()
    return weights / weights.sum()


def held_out_maps(maps: list) -> list[int]:
    """Maps that nothing fits on, to see how well things carry over to a new one.  The one every engine test is on, and about a tenth of a large pool, chosen by name so that it doesn't move."""
    return [i for i, (_, _, facts) in enumerate(maps) if "titanduel" in facts.name or (len(maps) > 20 and zlib.crc32(facts.name.encode()) % 10 == 0)]


class Roster:
    """What can be on the board, from the catalog: what factories build, and what their builders and commanders can.  Rows of the type table unless it says otherwise."""

    def __init__(self, table: tables.TypeTable, cat: zksim.Catalog):
        self.table, self.cat = table, cat
        usable = lambda name: name in cat.index  # noqa: E731  what the sim doesn't model the weapon of can still stand there and be seen
        self.factories = [r for r, name in enumerate(table.names) if table.factory[r] and usable(name) and any(usable(n) for n in bot_catalog.roster(name))]
        self.by_factory = {r: np.array([table.names.index(n) for n in bot_catalog.roster(table.names[r]) if usable(n)]) for r in self.factories}
        self.mobile = np.unique(np.concatenate(list(self.by_factory.values())))
        self.commanders = cat_commanders(cat)
        commanders = [table.names.index(cat.units[i]["name"]) for i in self.commanders if cat.units[i]["name"] in table.names]
        builders = [r for r in self.mobile if table.builder[r]] + commanders
        self.options: dict[int, np.ndarray] = {}
        for r in builders:
            named = {o.get("name") if isinstance(o, dict) else bot_units.name_of(o) for o in (bot_units.by_name(table.names[r]) or {}).get("buildOptions") or []}
            self.options[int(r)] = np.array(sorted(table.names.index(n) for n in named if n in table.names and usable(n) and not table.mobile[table.names.index(n)]), np.int64)
        self.structures = np.unique(np.concatenate(list(self.options.values())))
        self.is_extractor = extractors(table)
        self.extractor = int(np.flatnonzero(self.is_extractor)[0])
        self.half = np.zeros((len(table.names) + 1, 2), np.float32)     # of each type's footprint, along x and z
        self.footing: dict[int, Footing] = {}
        self.is_energy = np.zeros(len(table.names) + 1, bool)
        for r, name in enumerate(table.names):
            if name in cat.index:
                u = cat.units[cat.index[name]]
                self.footing[r] = Footing.of(u)
                self.half[r] = self.footing[r].half_x, self.footing[r].half_z
                self.is_energy[r] = not table.mobile[r] and u["energy_make"] > 0.0
        self.is_turret = ~table.mobile & table.armed & ~table.factory
        self.is_silo = np.zeros(len(table.names) + 1, bool)
        for r, name in enumerate(table.names):
            self.is_silo[r] = name in cat.index and cat.units[cat.index[name]]["missile_pads"] > 0
        self.missiles = np.array([i for i, u in enumerate(cat.units) if u["flags"] & zksim.MISSILE], np.int32)
        self.fighters = {r: units[table.combat[units] & ~table.builder[units]] for r, units in self.by_factory.items()}
        self.fighters = {r: (f if len(f) else self.by_factory[r]) for r, f in self.fighters.items()}
        self.makers = {r: units[table.builder[units]] for r, units in self.by_factory.items()}

    def catalog_index(self, rows: np.ndarray) -> np.ndarray:
        return np.array([self.cat.index[self.table.names[r]] for r in rows], np.int32)

    def factories_for(self, ground: Ground) -> tuple[list[int], np.ndarray]:
        """The factories that suit a map, wherever on it they are, and how likely each is: see Ground.suits."""
        if ground.factory_choices is not None:
            return ground.factory_choices
        rows, weights = [], []
        for r in self.factories:
            classes = [int(c) for c in self.table.move_class[self.fighters[r]] if c != tables.NO_CLASS]
            usual = max(set(classes), key=lambda c: (classes.count(c), c)) if classes else tables.NO_CLASS
            if ground.suits(self.footing[r], usual):
                rows.append(r)
                weights.append(ground.chance(usual))
        ground.factory_choices = rows, np.array(weights) / max(sum(weights), 1.0)
        return ground.factory_choices


def cat_commanders(cat: zksim.Catalog) -> list[int]:
    return [i for i, u in enumerate(cat.units) if u["flags"] & zksim.COMMANDER]


def extractors(table: tables.TypeTable) -> np.ndarray:
    """By row of the type table."""
    out = np.zeros(len(table.names) + 1, bool)
    for r, name in enumerate(table.names):
        out[r] = bool(bot_units.is_mex(bot_units.by_name(name)))
    return out


def _nearest(x: np.ndarray, z: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, where the nearest of the others is."""
    i = np.hypot(others["x"][None, :] - x[:, None], others["z"][None, :] - z[:, None]).argmin(axis=1)
    return others["x"][i], others["z"][i]


@dataclass
class FightScript:
    """One side of one game.  A fighter is sent (to the enemy's start, then its nearest building once any has got there), waiting at
    the rally until those waiting are worth a group, or retreating to the rally to regenerate."""
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


@dataclass
class FrameFight(FightScript):
    """As the fight script, but each sends its units somewhere their class can be, a side whose units can't reach the enemy's start looks for its buildings after a while,
    and what can jump does, whenever it's charged: frames should have jumpers in the air, and waiting to jump again.  So with the rest of what a few units can do: a Lobster throws
    whoever's around it along the way, a Newton or a Sumo goes between pushing and pulling, and a Djinn leaves a beacon at the rally and goes ahead for others to come over to."""
    ground: Ground | None = None
    beacons: dict[int, tuple[float, float, float]] = field(default_factory=dict)    # a Djinn's uid: where its beacon went, and when
    switched: int = 0

    GIVE_UP_ON_START = 360.0     # seconds
    JUMP_FROM = 150.0            # elmos from where it's going: past this it jumps there
    JUMP_SHORT = 0.95            # of its range: on its own, an order to go exactly that far would walk a bit first
    SWITCH_EVERY = 60.0          # seconds, between pushing and pulling
    BEACON_TAKES = 15.0          # seconds a Djinn stands there for one to appear
    BEACON_AGAIN = 90.0          # seconds, if it's lost one
    BEACON_FOUND = 40.0          # elmos from where it went
    STAND_AT = 0.5               # of the way from the rally to the enemy
    ARRIVED_THERE = 150.0        # elmos
    COME_OVER_FROM = 400.0       # elmos from a beacon

    def orders(self, own: np.ndarray, enemies: np.ndarray, seconds: float) -> np.ndarray:
        self.reached_start = self.reached_start or seconds >= self.GIVE_UP_ON_START
        out = super().orders(own, enemies, seconds)
        switches = self._switch(own, seconds)
        if not len(out):
            return switches
        table = self.data.table
        rows = table.rows(own["def_id"])
        classes = table.move_class[rows[self.data.fights[rows] & ((own["flags"] & rf.BEING_BUILT) == 0)]]
        for c in np.unique(classes):
            if c == tables.NO_CLASS:
                continue
            at = np.flatnonzero(classes == c)
            off = at[~self.ground.can(int(c), out["x"][at], out["z"][at])]
            for i in off:
                place = self.ground.near(int(c), float(out["x"][i]), float(out["z"][i]))
                if place is not None:
                    out["x"][i], out["z"][i] = place
        fighters = own[self.data.fights[rows] & ((own["flags"] & rf.BEING_BUILT) == 0)]
        self._jump(fighters, out)
        self._throw(fighters, own, out)
        return np.concatenate((self._teleport(fighters, own, out, seconds), switches))

    def _throw(self, fighters: np.ndarray, own: np.ndarray, out: np.ndarray) -> None:
        """What can throw and has someone to throw does, along the way."""
        data, table = self.data, self.data.table
        rows = table.rows(fighters["def_id"])
        ready = (data.throw_radius[rows] > 0.0) & ((fighters["flags"] & rf.JUMP_READY) != 0)
        if not ready.any():
            return
        own_rows = table.rows(own["def_id"])
        throwable = own[table.mobile[own_rows] & (table.move_class[own_rows] != tables.NO_CLASS) & ((own["flags"] & rf.BEING_BUILT) == 0)]
        for i in np.flatnonzero(ready):
            dx, dz = float(out["x"][i] - fighters["x"][i]), float(out["z"][i] - fighters["z"][i])
            away = float(np.hypot(dx, dz))
            if away < self.JUMP_FROM:
                continue
            near = np.hypot(throwable["x"] - fighters["x"][i], throwable["z"] - fighters["z"][i]) <= data.throw_radius[rows[i]]
            if not (near & (throwable["uid"] != fighters["uid"][i])).any():
                continue
            hop = min(away, float(data.throw_reach[rows[i]]) * self.JUMP_SHORT) / away
            x, z = np.array([fighters["x"][i] + dx * hop], np.float32), np.array([fighters["z"][i] + dz * hop], np.float32)
            if self.ground.can(int(table.move_class[rows[i]]), x, z)[0]:
                out["kind"][i], out["x"][i], out["z"][i], out["target"][i] = rf.MANUAL_FIRE, x[0], z[0], -1

    def _teleport(self, fighters: np.ndarray, own: np.ndarray, out: np.ndarray, seconds: float) -> np.ndarray:
        """A Djinn leaves a beacon where this side's units gather and stands halfway to the enemy, and some of those about the beacon wait there to come over."""
        data, table = self.data, self.data.table
        rows = table.rows(fighters["def_id"])
        djinns = np.flatnonzero(data.teleports[rows])
        if not len(djinns):
            return out
        own_rows = table.rows(own["def_id"])
        beacons = own[data.beacon[own_rows]]
        keep = np.ones(len(out), bool)
        for i in djinns:
            uid, c = int(fighters["uid"][i]), int(table.move_class[rows[i]])
            placed = self.beacons.get(uid)
            beacon = None
            if placed is not None and len(beacons):
                apart = np.hypot(beacons["x"] - placed[0], beacons["z"] - placed[1])
                if apart.min() <= self.BEACON_FOUND:
                    beacon = beacons[int(np.argmin(apart))]
            if beacon is None:
                if placed is not None and seconds - placed[2] < self.BEACON_AGAIN:
                    keep[i] = seconds - placed[2] >= self.BEACON_TAKES
                    continue
                x, z = self.ground.near(c, *self.rally) or self.rally
                self.beacons[uid] = float(x), float(z), seconds
                out[i] = (uid, zksim.ORDER_PLACE_BEACON, x, z, -1)
                continue
            x = self.rally[0] + self.STAND_AT * (self.enemy_start[0] - self.rally[0])
            z = self.rally[1] + self.STAND_AT * (self.enemy_start[1] - self.rally[1])
            x, z = self.ground.near(c, float(x), float(z)) or (fighters["x"][i], fighters["z"][i])
            if np.hypot(fighters["x"][i] - x, fighters["z"][i] - z) > self.ARRIVED_THERE:
                out["kind"][i], out["x"][i], out["z"][i], out["target"][i] = rf.MOVE, x, z, -1
                continue
            keep[i] = False
            if not fighters["flags"][i] & rf.IDLE:
                continue
            around = (np.hypot(fighters["x"] - beacon["x"], fighters["z"] - beacon["z"]) <= self.COME_OVER_FROM) & (fighters["uid"] % 2 == 0) & ~data.teleports[rows]
            out["kind"][around], out["target"][around] = zksim.ORDER_WAIT_AT_BEACON, int(beacon["uid"])
        return out[keep]

    def _switch(self, own: np.ndarray, seconds: float) -> np.ndarray:
        """What both pushes and pulls goes from one to the other every so often, not all at once."""
        turn = int(seconds / self.SWITCH_EVERY)
        if turn == self.switched:
            return np.zeros(0, zksim.ORDER_DTYPE)
        self.switched = turn
        units = own[self.data.switches[self.data.table.rows(own["def_id"])] & ((own["flags"] & rf.BEING_BUILT) == 0)]
        out = np.zeros(len(units), zksim.ORDER_DTYPE)
        out["unit"], out["kind"], out["target"] = units["uid"], zksim.ORDER_ON_OFF, (units["uid"] + turn) % 2
        return out

    def _jump(self, fighters: np.ndarray, out: np.ndarray) -> None:
        """Turns the orders of those charged into a jump along the way."""
        table = self.data.table
        rows = table.rows(fighters["def_id"])
        for i in np.flatnonzero(table.jump[rows] & ((fighters["flags"] & rf.JUMP_READY) != 0)):
            dx, dz = float(out["x"][i] - fighters["x"][i]), float(out["z"][i] - fighters["z"][i])
            away = float(np.hypot(dx, dz))
            if away < self.JUMP_FROM:
                continue
            hop = min(away, float(table.jump_range[rows[i]]) * self.JUMP_SHORT) / away
            x, z = np.array([fighters["x"][i] + dx * hop], np.float32), np.array([fighters["z"][i] + dz * hop], np.float32)
            if self.ground.can(int(table.move_class[rows[i]]), x, z)[0]:
                out["kind"][i], out["x"][i], out["z"][i] = rf.JUMP, x[0], z[0]


def fight_data(table: tables.TypeTable, cat: zksim.Catalog) -> ScriptData:
    """Builders have work of their own, so the fight script leaves them be."""
    data = ScriptData(table, cat)
    data.fights = data.fights & ~table.builder
    data.throw_radius, data.throw_reach = table.throw_radius, table.throw_reach
    data.teleports, data.beacon, data.switches = table.teleports, table.beacon, table.switches
    return data


# A side's economy ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class BaseScript:
    """One side of one game: what its factories make, and what its builders do when they've nothing to do.  There's no plan in it.  It's for frames to look like someone's playing."""
    roster: Roster
    ground: Ground
    base: tuple[float, float]
    enemy: tuple[float, float]
    rng: np.random.Generator
    clock: float = 0.0              # seconds a scenario claims had passed when it began
    change_at: dict[int, float] = field(default_factory=dict)      # factory: when it makes something else

    PRODUCE_FOR = (30.0, 120.0)     # seconds
    FIRST_COST = 200.0              # what a builder will start, plus this much for each second of the game
    COST_PER_SECOND = 3.0
    MOST_FACTORIES = 3
    REACH = 1000.0                  # elmos a builder will go to help or reclaim

    def _next_unit(self, factory: int, own_rows: np.ndarray, seconds: float) -> int:
        roster, rng = self.roster, self.rng
        makers = roster.makers[factory]
        wanted_builders = 1 + min(4.0, (seconds + self.clock) / 180.0)
        if len(makers) and roster.table.builder[own_rows].sum() < wanted_builders:
            return int(roster.catalog_index(makers[int(rng.integers(len(makers)))][None])[0])
        fighters = roster.fighters[factory]
        chances = 1.0 / np.sqrt(roster.table.cost[fighters])
        return int(roster.catalog_index(fighters[int(rng.choice(len(fighters), p=chances / chances.sum()))][None])[0])

    def _build(self, uid: int, row: int, near: tuple[float, float], taken: np.ndarray) -> tuple | None:
        place = self.ground.site(self.rng, self.roster.footing[row], near[0], near[1], taken)
        if place is None:
            return None
        return uid, rf.BUILD, place[0], place[1], self.roster.cat.index[self.roster.table.names[row]]

    def _job(self, unit: np.void, row: int, own: np.ndarray, own_rows: np.ndarray, free: np.ndarray, features: np.ndarray, taken: np.ndarray, seconds: float) -> tuple | None:
        roster, rng, table = self.roster, self.rng, self.roster.table
        uid, x, z = int(unit["uid"]), float(unit["x"]), float(unit["z"])
        c = int(table.move_class[row])
        options = roster.options.get(row, np.zeros(0, np.int64))
        options = options[table.cost[options] <= self.FIRST_COST + self.COST_PER_SECOND * (seconds + self.clock)]
        options = np.array([o for o in options if self.ground.has_room(roster.footing[int(o)].low, roster.footing[int(o)].high)], np.int64)
        away = np.hypot(own["x"] - x, own["z"] - z)
        started = np.flatnonzero((own["flags"] & rf.BEING_BUILT != 0) & ~table.mobile[own_rows] & (away <= self.REACH))
        factories = np.flatnonzero(table.factory[own_rows] & (away <= self.REACH))
        lying = features[(features["metal"] > 0.0) & (np.hypot(features["x"] - x, features["z"] - z) <= self.REACH)] if len(features) else features
        nearest_free = free[np.argsort(np.hypot(free[:, 0] - x, free[:, 1] - z))[:8]] if len(free) else free
        spots = [s for s in nearest_free if c == tables.NO_CLASS or self.ground.joined(c, (x, z), (float(s[0]), float(s[1])))]
        jobs: list[tuple[str, float]] = []
        if len(started):
            jobs.append(("finish", 3.0))
        if spots and roster.extractor in options:
            jobs.append(("extractor", 4.0))
        for kind, chosen, chance in (("energy", roster.is_energy[options], 2.0), ("turret", roster.is_turret[options], 1.5), ("other", ~(roster.is_energy[options] | roster.is_turret[options] | roster.is_extractor[options] | table.factory[options]), 0.5)):
            if chosen.any():
                jobs.append((kind, chance))
        fit_factories = np.array([o for o in options if table.factory[o] and int(o) in self._usable_factories()], np.int64)
        if len(fit_factories) and table.factory[own_rows].sum() < self.MOST_FACTORIES:
            jobs.append(("factory", 0.5))
        if len(lying):
            jobs.append(("reclaim", 1.0))
        if len(factories):
            jobs.append(("assist", 0.5))
        if not jobs:
            return None
        chances = np.array([w for _, w in jobs])
        job = jobs[int(rng.choice(len(jobs), p=chances / chances.sum()))][0]
        if job == "finish":
            return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][started[int(rng.integers(len(started)))]])
        if job == "assist":
            return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][factories[int(rng.integers(len(factories)))]])
        if job == "extractor":
            s = spots[0]
            return uid, rf.BUILD, float(s[0]), float(s[1]), roster.cat.index[table.names[roster.extractor]]
        if job == "reclaim":
            f = lying[int(np.argmin(np.hypot(lying["x"] - x, lying["z"] - z)))]
            return uid, rf.RECLAIM, float(f["x"]), float(f["z"]), int(f["feat_id"])
        if job == "factory":
            return self._build(uid, int(fit_factories[int(rng.integers(len(fit_factories)))]), self.base, taken)
        chosen = {"energy": roster.is_energy[options], "turret": roster.is_turret[options]}.get(job)
        if chosen is None:
            chosen = ~(roster.is_energy[options] | roster.is_turret[options] | roster.is_extractor[options] | table.factory[options])
        pick = options[chosen]
        near = self.base
        if job == "turret" and rng.random() < 0.5:
            t = float(rng.uniform(0.1, 0.5))
            near = (self.base[0] + t * (self.enemy[0] - self.base[0]), self.base[1] + t * (self.enemy[1] - self.base[1]))
        return self._build(uid, int(pick[int(rng.integers(len(pick)))]), near, taken)

    def _usable_factories(self) -> set[int]:
        return set(self.roster.factories_for(self.ground)[0])

    def orders(self, own: np.ndarray, enemies: np.ndarray, features: np.ndarray, seconds: float, busy: int = -1, allies: np.ndarray | None = None) -> np.ndarray:
        """own: a side's own records.  enemies: the others', as this is a script and spots are all it uses them for.  busy: a uid not
        to touch.  allies: its allies' records, whose spots it leaves and whose buildings it keeps off."""
        roster, table = self.roster, self.roster.table
        rows = table.rows(own["def_id"])
        finished = (own["flags"] & rf.BEING_BUILT) == 0
        out: list[tuple] = []
        for i in np.flatnonzero(finished & table.factory[rows]):
            uid = int(own["uid"][i])
            if seconds >= self.change_at.get(uid, 0.0):
                self.change_at[uid] = seconds + float(self.rng.uniform(*self.PRODUCE_FOR))
                out.append((uid, rf.PRODUCE, 0.0, 0.0, self._next_unit(int(rows[i]), rows, seconds)))
        for i in np.flatnonzero(finished & roster.is_silo[rows]):
            uid = int(own["uid"][i])
            if seconds >= self.change_at.get(uid, 0.0):
                self.change_at[uid] = seconds + float(self.rng.uniform(*self.PRODUCE_FOR))
                out.append((uid, rf.PRODUCE, 0.0, 0.0, int(self.rng.choice(roster.missiles))))
        idle = np.flatnonzero(finished & table.builder[rows] & table.mobile[rows] & ((own["flags"] & rf.IDLE) != 0) & (own["uid"] != busy))
        if len(idle):
            spots = self.ground.bundle.spot_list
            others = enemies if allies is None or not len(allies) else np.concatenate((enemies, allies))
            other_rows = table.rows(others["def_id"])
            holders = np.concatenate((own[roster.is_extractor[rows]][["x", "z"]], others[roster.is_extractor[other_rows]][["x", "z"]]))
            free = spots
            if len(holders) and len(spots):
                free = spots[np.hypot(spots[:, 0, None] - holders["x"], spots[:, 1, None] - holders["z"]).min(axis=1) > EXTRACTOR_REACH]
            # Our side of the map.
            free = free[np.hypot(free[:, 0] - self.base[0], free[:, 1] - self.base[1]) <= 1.2 * np.hypot(free[:, 0] - self.enemy[0], free[:, 1] - self.enemy[1])]
            built = ~table.mobile[rows]
            taken = np.concatenate((footprints(own["x"][built], own["z"][built], roster.half[rows[built], 0], roster.half[rows[built], 1]),
                                    footprints(spots[:, 0], spots[:, 1], *roster.half[roster.extractor])))
            if allies is not None and len(allies):
                ally_rows = table.rows(allies["def_id"])
                theirs = ~table.mobile[ally_rows]
                taken = np.concatenate((taken, footprints(allies["x"][theirs], allies["z"][theirs], roster.half[ally_rows[theirs], 0], roster.half[ally_rows[theirs], 1])))
            for i in idle:
                job = self._job(own[i], int(rows[i]), own, rows, free, features, taken, seconds)
                if job is None:
                    continue
                out.append(job)
                if job[1] == rf.BUILD:
                    taken = np.concatenate((taken, footprints(job[2], job[3], *roster.half[table.names.index(roster.cat.units[job[4]]["name"])])))
                    if roster.is_extractor[table.names.index(roster.cat.units[job[4]]["name"])]:
                        free = free[np.hypot(free[:, 0] - job[2], free[:, 1] - job[3]) > EXTRACTOR_REACH]
        return np.array(out, zksim.ORDER_DTYPE)


class BaseOrders:
    """For an environment whose teams have a BaseScript as well as a fight script: in `bases`, by game, one for each team, None
    for a team without (the chickens)."""
    BASE_EVERY = 8      # decisions: builders don't wait long for work

    def script_orders(self, g: int, side: int) -> np.ndarray:
        fights = super().script_orders(g, side)
        if self.decisions[g] % self.BASE_EVERY or self.bases[g][side] is None:
            return fights
        own, enemies, allies = self._sides_of(g, side)
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        base = self.bases[g][side].orders(own, enemies, self._team_units(g)[side][1], seconds, self._busy(g, side), allies)
        return np.concatenate((fights, base)) if len(base) else fights

    def _busy(self, g: int, side: int) -> int:
        return -1


# Whole games -----------------------------------------------------------------------------------------------------------------------------------------


class WideOpening(Opening):
    """The opening with every factory a side can place: the first thing it makes is its raider, or if it has none, its cheapest fighter."""
    favour_water = True     # frames with ships are what games on the old maps had none of

    def __init__(self, cat: zksim.Catalog, roster: Roster, decide_every: int = 1):
        super().__init__(cat, decide_every)
        table = roster.table
        self.factories = []
        for i, u in enumerate(cat.units):
            name = u["name"]
            if not u["flags"] & zksim.PLOPPABLE or name not in table.names or table.names.index(name) not in roster.by_factory:
                continue
            raider = bot_catalog.pick(name, "raid")
            if raider is None or raider not in cat.index:
                fighters = roster.fighters[table.names.index(name)]
                raider = table.names[int(fighters[np.argmin(table.cost[fighters])])]
            self.factories.append((i, cat.index[raider]))
        self._factory_footing = None    # of these factories, not the opening's

    def _fallback(self, ground: Ground, start: tuple[float, float]) -> int:
        """No way there on this map for anything it can place: at least the air will do."""
        return next(i for i, (_, first) in enumerate(self.factories) if self.cat.units[first]["flags"] & zksim.AIR)


class FrameGames(BaseOrders, se.SimEnv):
    """Games from the start on any map, with both sides' factories and builders at work.  What frames of a game are, as near as a script can get."""
    own_scripts = True      # FrameFight and BaseScript, not the C++ army and economy scripts

    def __init__(self, games: int, seed: int, threads: int = 16, maps: list | None = None, weights: np.ndarray | None = None):
        super().__init__(games, seed, ["scripts"], threads=threads, maps=maps, weights=weights)
        self.roster = Roster(self.table, se.catalog())
        self.opening = WideOpening(se.catalog(), self.roster)
        self.script_data = fight_data(self.table, se.catalog())
        self.bases: list = [None] * games

    def _start_commanders(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, box0: se.StartBox, box1: se.StartBox, x0: float, z0: float, x1: float, z1: float) -> None:
        commanders = self.opening.commanders
        types = [commanders[int(self.rng.integers(len(commanders)))] for _ in range(2)]
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.array(types), np.array([0, 1]), np.array([x0, x1]), np.array([z0, z1]),
                         commander=(0, 1), base=(x0, z0, x1, z1), sites=bundle.sites())
        self.batch.economy(g, bundle.spot_list)
        ground = self.grounds[id(bundle)]
        self.seats[g] = (self.opening.seat(self.rng, 0, ground, (x0, z0), box1.point), self.opening.seat(self.rng, 1, ground, (x1, z1), box0.point))
        self.scripts[g] = (FrameFight(self.script_data, self.seats[g][0].plan.rally, (x1, z1), ground=ground), FrameFight(self.script_data, self.seats[g][1].plan.rally, (x0, z0), ground=ground))
        self.bases[g] = (BaseScript(self.roster, ground, (x0, z0), (x1, z1), self.rng), BaseScript(self.roster, ground, (x1, z1), (x0, z0), self.rng))

    def _busy(self, g: int, side: int) -> int:
        """The commander, until the opening's over."""
        seats = self.seats[g]
        return side if seats is not None and not seats[side].done else -1
