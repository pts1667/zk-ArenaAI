"""What frames are collected from has in common: which maps, where things can stand on them, and a side that does more than the training script's.

Roster      what can be on the board: factories and their units, and what builders can build
BaseScript  a side's factories and builders: changing what's produced, taking spots, building energy, turrets and factories, reclaiming, helping
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
from policy_opening import Footing, Ground, Opening, Seat, modelled  # noqa: E402
from policy_script import FightScript, ScriptData  # noqa: E402
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
        usable = lambda name: name in cat.index and modelled(cat, name)  # noqa: E731
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
        self.radius = np.zeros(len(table.names) + 1, np.float32)
        self.footing: dict[int, Footing] = {}
        self.is_energy = np.zeros(len(table.names) + 1, bool)
        for r, name in enumerate(table.names):
            if name in cat.index:
                u = cat.units[cat.index[name]]
                self.radius[r] = u["footprint_radius"]
                self.footing[r] = Footing.of(u)
                self.is_energy[r] = not table.mobile[r] and u["energy_make"] > 0.0
        self.is_turret = ~table.mobile & table.armed & ~table.factory
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


@dataclass
class FrameFight(FightScript):
    """As the fight script, but each sends its units somewhere their class can be, a side whose units can't reach the enemy's start looks for its buildings after a while,
    and what can jump does, whenever it's charged: frames should have jumpers in the air, and waiting to jump again."""
    ground: Ground | None = None

    GIVE_UP_ON_START = 360.0     # seconds
    JUMP_FROM = 150.0            # elmos from where it's going: past this it jumps there
    JUMP_SHORT = 0.95            # of its range: on its own, an order to go exactly that far would walk a bit first

    def orders(self, own: np.ndarray, enemies: np.ndarray, seconds: float) -> np.ndarray:
        self.reached_start = self.reached_start or seconds >= self.GIVE_UP_ON_START
        out = super().orders(own, enemies, seconds)
        if not len(out):
            return out
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
        self._jump(own[self.data.fights[rows] & ((own["flags"] & rf.BEING_BUILT) == 0)], out)
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

    def orders(self, own: np.ndarray, enemies: np.ndarray, features: np.ndarray, seconds: float, busy: int = -1) -> np.ndarray:
        """own: a side's own records.  enemies: the other's, as this is a script and spots are all it uses them for.  busy: a uid not to touch."""
        roster, table = self.roster, self.roster.table
        rows = table.rows(own["def_id"])
        finished = (own["flags"] & rf.BEING_BUILT) == 0
        out: list[tuple] = []
        for i in np.flatnonzero(finished & table.factory[rows]):
            uid = int(own["uid"][i])
            if seconds >= self.change_at.get(uid, 0.0):
                self.change_at[uid] = seconds + float(self.rng.uniform(*self.PRODUCE_FOR))
                out.append((uid, rf.PRODUCE, 0.0, 0.0, self._next_unit(int(rows[i]), rows, seconds)))
        idle = np.flatnonzero(finished & table.builder[rows] & table.mobile[rows] & ((own["flags"] & rf.IDLE) != 0) & (own["uid"] != busy))
        if len(idle):
            spots = self.ground.bundle.spot_list
            other_rows = table.rows(enemies["def_id"])
            holders = np.concatenate((own[roster.is_extractor[rows]][["x", "z"]], enemies[roster.is_extractor[other_rows]][["x", "z"]]))
            free = spots
            if len(holders) and len(spots):
                free = spots[np.hypot(spots[:, 0, None] - holders["x"], spots[:, 1, None] - holders["z"]).min(axis=1) > EXTRACTOR_REACH]
            # Our side of the map.
            free = free[np.hypot(free[:, 0] - self.base[0], free[:, 1] - self.base[1]) <= 1.2 * np.hypot(free[:, 0] - self.enemy[0], free[:, 1] - self.enemy[1])]
            built = ~table.mobile[rows]
            taken = np.concatenate((np.column_stack((own["x"][built], own["z"][built], roster.radius[rows[built]])),
                                    np.column_stack((spots[:, 0], spots[:, 1], np.full(len(spots), self.roster.footing[self.roster.extractor].radius))))).astype(np.float32)
            for i in idle:
                job = self._job(own[i], int(rows[i]), own, rows, free, features, taken, seconds)
                if job is None:
                    continue
                out.append(job)
                if job[1] == rf.BUILD:
                    taken = np.concatenate((taken, np.array([[job[2], job[3], roster.radius[table.names.index(roster.cat.units[job[4]]["name"])]]], np.float32)))
                    if roster.is_extractor[table.names.index(roster.cat.units[job[4]]["name"])]:
                        free = free[np.hypot(free[:, 0] - job[2], free[:, 1] - job[3]) > EXTRACTOR_REACH]
        return np.array(out, zksim.ORDER_DTYPE)


class BaseOrders:
    """For an environment whose sides have a BaseScript as well as a fight script: in `bases`, by game, one for each side."""
    BASE_EVERY = 8      # decisions: builders don't wait long for work

    def script_orders(self, g: int, side: int) -> np.ndarray:
        fights = super().script_orders(g, side)
        if self.decisions[g] % self.BASE_EVERY:
            return fights
        frame = self.batch.frame(g, side)
        own, enemy = frame.units, self.batch.frame(g, 1 - side).units
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        base = self.bases[g][side].orders(own[own["player"] == side], enemy[enemy["player"] == 1 - side], frame.features, seconds, self._busy(g, side))
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
            if raider is None or raider not in cat.index or not modelled(cat, raider):
                fighters = roster.fighters[table.names.index(name)]
                raider = table.names[int(fighters[np.argmin(table.cost[fighters])])]
            self.factories.append((i, cat.index[raider]))
        self.factory_radius = max(cat.units[f]["footprint_radius"] for f, _ in self.factories)

    def _fallback(self, ground: Ground, start: tuple[float, float]) -> int:
        """No way there on this map for anything it can place: at least the air will do."""
        return next(i for i, (_, first) in enumerate(self.factories) if self.cat.units[first]["flags"] & zksim.AIR)


class FrameGames(BaseOrders, se.SimEnv):
    """Games from the start on any map, with both sides' factories and builders at work.  What frames of a game are, as near as a script can get."""

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
