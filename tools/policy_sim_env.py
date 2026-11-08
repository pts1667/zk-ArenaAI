"""The simulator as an environment for tools/policy_train.py, with the same interface as the toy's.

A game starts from a commander in each of two of the map's start boxes.  Both sides play the opening of policy_opening.py by rule,
so what the policy controls is the raiders it produces.  On the other side, a script controls the fighters and the factory: the army script of policy_army.py by default, or the raider-only fight script of policy_script.py, for comparisons.  The kinds the policy has are those the sim obeys for fighting units; S-M7 adds the rest.

What each script ordered at each decision is in `script_record`, and its production choices are readable with `script_production`.

Passing `units` gives the older start instead, two mirrored armies and no economy, which checks and benchmarks still use.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import catalog as bot_catalog  # noqa: E402
from zkbot.policy import bundle as bd, rawframe as rf, tables  # noqa: E402
from policy_army import ArmyScript, make_script  # noqa: E402
from policy_eco import ECO_EVERY, Aimer, EcoData, Economy  # noqa: E402
from policy_opening import Ground, Opening, Seat, is_buildable, modelled  # noqa: E402
from policy_script import ScriptData  # noqa: E402

KINDS = (rf.KEEP, rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.JUMP, rf.CLOAK_ON, rf.CLOAK_OFF,
         rf.PRODUCE, rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON)
TICKS_PER_DECISION = 3
TIME_LIMIT_DECISIONS = 2400         # 20 minutes
VALUE_CONSTANT = 1000.0
SCRIPT_EVERY = 6                    # decisions: as often as the policy's commit lock lets it re-order
SPREAD = 220.0
MOST_COST = 600.0
EARLY_END_DECISIONS = 240           # 2 minutes: from then on, a side this far ahead in metal value has won
EARLY_END_RATIO = 2.5
EARLY_END_LEAD = 10000.0            # and this much ahead in metal, so that a small game isn't called on a few raiders
# 03-training.md section 3.  Other exported maps stay out of training: playing pads every seat to the map with the most regions.
TRAINING_MAPS = ("agalta_v1_0", "faultline_v1_0", "frostycove_v1_13", "red_comet_remake_1_7", "titanduel_2_2")

AT_THE_START_POINT = 0.5             # of starts: where our bot and a player who doesn't choose start.  The rest are anywhere in the box.
START_GRID = 32.0                    # elmos: starts snap to this, so that the opening's plans can be kept
START_ROOM = 48.0                    # elmos of level ground round a start
START_TRIES = 64

_POOL: list[tuple[zksim.Map, bd.Bundle, "MapFacts"]] = []
_CATALOG: zksim.Catalog | None = None
_TYPES: dict[str, np.ndarray] = {}


def catalog() -> zksim.Catalog:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = zksim.Catalog()
    return _CATALOG


def _inside(polygon: list[list[float]], x: float, z: float) -> bool:
    """By counting the edges a line from the point crosses."""
    inside = False
    for (ax, az), (bx, bz) in zip(polygon, polygon[1:] + polygon[:1]):
        if (az > z) != (bz > z) and x < ax + (z - az) * (bx - ax) / (bz - az):
            inside = not inside
    return inside


@dataclass(frozen=True)
class StartBox:
    """A side's place on a map, from tools/sim_engine.py facts."""
    polygons: tuple[list[list[float]], ...]
    point: tuple[float, float]      # the one Zero-K recommends

    def pick(self, rng: np.random.Generator, bundle: bd.Bundle) -> tuple[float, float]:
        if rng.random() < AT_THE_START_POINT:
            return self.point
        corners = np.array([v for polygon in self.polygons for v in polygon])
        low, high = corners.min(axis=0), corners.max(axis=0)
        for _ in range(START_TRIES):
            x, z = np.round(rng.uniform(low, high) / START_GRID) * START_GRID
            x, z = float(x), float(z)
            if any(_inside(polygon, x, z) for polygon in self.polygons) and is_buildable(bundle, x, z, START_ROOM):
                return x, z
        return self.point


@dataclass(frozen=True)
class MapFacts:
    """What a game says about a map, from tools/sim_engine.py facts."""
    layouts: list[list[StartBox]]       # some maps draw theirs at random when they load, among these
    features: np.ndarray                # N x 6: x, z, metal, hp, radius, def.  What's reclaimable from the start.
    name: str = ""
    kind: str = "real"                  # or "made": tools/map_gen.py
    water: float = 0.0                  # of the map under the plane
    liquid: str = "water"
    reach: dict[str, np.ndarray] | None = field(default=None, repr=False)   # by class: which part of the map each cell connects to, -1 for none


def _layouts(facts: dict) -> list[list[StartBox]]:
    return [[StartBox(tuple(b["polygons"]), (float(b["startpoints"][0][0]), float(b["startpoints"][0][1]))) for b in boxes] for boxes in facts["layouts"]]


def _map_facts(slug: str) -> MapFacts:
    facts = json.loads((zksim.DATA / "map_facts.json").read_text(encoding="utf8"))[slug]
    return MapFacts(_layouts(facts), np.array(facts["features"], np.float32).reshape(-1, 6), name=slug)


def load_pool(directory: pathlib.Path | str) -> list[tuple[zksim.Map, bd.Bundle, MapFacts]]:
    """All the maps of a directory that tools/map_gen.py wrote to, in the order of their names."""
    out = []
    for path in sorted(pathlib.Path(directory).glob("*.facts.json")):
        slug = path.name[:-len(".facts.json")]
        facts = json.loads(path.read_text(encoding="utf8"))
        with np.load(path.with_name(f"{slug}.reach.npz")) as d:
            reach = {k: d[k] for k in d.files}
        out.append((zksim.Map.load(path.with_name(f"{slug}.npz")), bd.Bundle.load(path.with_name(f"{slug}.bundle.npz")),
                    MapFacts(_layouts(facts), np.array(facts["features"], np.float32).reshape(-1, 6), slug, facts["kind"], float(facts["water"]), facts["liquid"], reach)))
    return out


def pool() -> list[tuple[zksim.Map, bd.Bundle, MapFacts]]:
    """The training maps, exported by tools/sim_data.py maps.  One object per map, so forked workers can key on it."""
    if not _POOL:
        for slug in TRAINING_MAPS:
            bundle = bd.Bundle.load(zksim.DATA / "maps" / f"{slug}.bundle.npz")
            _POOL.append((zksim.Map.load(zksim.DATA / "maps" / f"{slug}.npz"), bundle, _map_facts(slug)))
    return _POOL


def sim_orders(orders: list) -> np.ndarray:
    """The policy's orders as the sim's.  Its kinds up to PRODUCE have the same numbers; a thrower's fire at the ground, and a switch's target, are the sim's own."""
    table, cat = tables.table(), catalog()
    out = np.zeros(len(orders), zksim.ORDER_DTYPE)
    for i, o in enumerate(orders):
        kind, target = o.kind, o.target
        if kind == rf.THROW:
            kind = zksim.ORDER_MANUAL_FIRE
        elif kind == rf.PUSH or kind == rf.PULL:
            kind, target = zksim.ORDER_ON_OFF, int(kind == rf.PUSH)
        elif kind == rf.PLACE_BEACON:
            kind = zksim.ORDER_PLACE_BEACON
        elif kind == rf.WAIT_AT_BEACON:
            kind = zksim.ORDER_WAIT_AT_BEACON
        elif kind == rf.PRODUCE:
            target = cat.index[table.names[target]]
        out[i] = (o.uid, kind, o.x, o.z, target)
    return out


def policy_orders(rows: np.ndarray) -> list:
    """`sim_orders` the other way: what a script told the sim, in the policy's terms, so it can be read as the policy's choices.
    A manual fire with no unit to aim at is a throw, as the policy has no other ground-aimed one."""
    from zkbot.policy import actions
    table, cat = tables.table(), catalog()
    out = []
    for o in rows:
        kind, target = int(o["kind"]), int(o["target"])
        if kind == zksim.ORDER_MANUAL_FIRE and target < 0:
            kind = rf.THROW
        elif kind == zksim.ORDER_ON_OFF:
            kind = rf.PUSH if target else rf.PULL
        elif kind == zksim.ORDER_PLACE_BEACON:
            kind = rf.PLACE_BEACON
        elif kind == zksim.ORDER_WAIT_AT_BEACON:
            kind = rf.WAIT_AT_BEACON
        elif kind == zksim.ORDER_PRODUCE:
            target = table.names.index(cat.units[target]["name"])
        out.append(actions.Order(int(o["unit"]), kind, float(o["x"]), float(o["z"]), target))
    return out


def army_types(table: tables.TypeTable) -> dict[str, np.ndarray]:
    """Ground and water fighters of ordinary cost, by the data rather than by name.  Ships need a sea this start doesn't look for."""
    if not _TYPES:
        cat = catalog()
        # The factory roster of 00-overview, not every def there is: no chickens, and no test units.
        built = {n for r, f in enumerate(table.names) if table.factory[r] for n in bot_catalog.roster(f)}
        rows = [r for r, name in enumerate(table.names) if name in cat.index and name in built and table.combat[r] and table.mobile[r] and not table.air[r]
                and table.can_hit_ground[r] and table.cost[r] <= MOST_COST and table.move_class[r] != bd.MOVE_CLASSES.index("ship") and modelled(cat, name)]
        _TYPES["rows"] = np.array(rows)
        _TYPES["index"] = np.array([cat.index[table.names[r]] for r in rows], np.int32)
        _TYPES["classes"] = table.move_class[rows].astype(np.int32)
    return _TYPES


def place(bundle: bd.Bundle, rng: np.random.Generator, classes: np.ndarray, cx: float, cz: float) -> tuple[np.ndarray, np.ndarray]:
    """Round a point, each unit somewhere its class can stand."""
    n = len(classes)
    x, z = np.full(n, cx, np.float32), np.full(n, cz, np.float32)
    todo = np.arange(n)
    rows, cols = bundle.passable.shape
    for _ in range(12):
        if not len(todo):
            break
        tx = np.clip(cx + rng.normal(0.0, SPREAD, len(todo)), 8.0, bundle.map_w - 8.0)
        tz = np.clip(cz + rng.normal(0.0, SPREAD, len(todo)), 8.0, bundle.map_h - 8.0)
        i = np.minimum((tz / bd.PASS_CELL).astype(np.int32), rows - 1)
        j = np.minimum((tx / bd.PASS_CELL).astype(np.int32), cols - 1)
        good = (bundle.passable[i, j] >> classes[todo]) & 1 == 1
        x[todo[good]], z[todo[good]] = tx[good], tz[good]
        todo = todo[~good]
    return x, z


def _far_ahead(value: np.ndarray, other: np.ndarray) -> np.ndarray:
    return (value > EARLY_END_RATIO * other) & (value - other >= EARLY_END_LEAD)


class SimEnv:
    eval_opponents = ("fight",)

    def __init__(self, games: int, seed: int, modes: list[str], units: int | None = None, threads: int = 16, decide_every: int = 1, maps: list | None = None,
                 weights: np.ndarray | None = None, scripts: tuple[str, str] = ("army", "army"), economy: bool = True, script_impl: str = "cpp",
                 script_threads: int | None = None):
        """maps: as `pool` or `load_pool` give, and weights how likely each is, if not evenly.  scripts: the kind of script each side
        plays, as policy_army.make_script names them.  When the two differ, each game gives them to the sides at random, so that
        the games compare the two whichever side has which.  economy: whether constructors and big buildings are run for every
        side (policy_eco.py); without it, a side's economy is its opening, as before 2026-11-08.  script_impl: "cpp" (the default) runs
        the army and economy scripts as libzkscripts (policy_scripts_cpp.py) on `script_threads` threads (`threads` if not given),
        "python" as the Python modules.  The opening is Python either way."""
        if script_impl not in ("python", "cpp"):
            raise ValueError(f"no script implementation called {script_impl!r}")
        if script_impl == "cpp" and set(scripts) != {"army"}:
            raise ValueError("the C++ scripts are the army script's")
        self.script_impl = script_impl
        self.count, self.modes, self.units = games, modes, units
        self.maps = pool() if maps is None else maps
        self.grounds = {id(b): Ground(b, f) for _, b, f in self.maps}
        self.weights = weights
        self.self_play_share = 0.0      # of games starting from now on, whatever `modes` says
        self.ended: dict[int, dict] = {}    # a seat's last finished game, for a test to note
        self.script_pair = scripts
        self.script_kinds: list[tuple[str, str]] = [scripts] * games
        self.script_record: dict[int, np.ndarray] = {}      # seat -> the orders its script gave this decision, as the sim takes them
        self.outcomes: list[dict] = []                      # every side of every finished game, for a measurement to take and clear
        self.opening = Opening(catalog(), decide_every) if units is None else None
        self.seats: list[tuple[Seat, Seat] | None] = [None] * games
        self.scripts: list[tuple[object, object] | None] = [None] * games
        # In slots of TICKS_PER_DECISION.  Game time stays what it was: the time limit, and how often the script re-orders.
        self.ticks = TICKS_PER_DECISION * decide_every
        self.time_limit = TIME_LIMIT_DECISIONS // decide_every
        self.early_end = EARLY_END_DECISIONS // decide_every
        self.script_every = max(1, SCRIPT_EVERY // decide_every)
        self.rng = np.random.default_rng(seed)
        self.seed_stream = np.random.SeedSequence(seed)
        self.batch = zksim.Batch(catalog(), games, threads=threads)
        self.table = tables.table()
        self.script_data = ScriptData(self.table, catalog())
        self.eco_data = EcoData(self.table, catalog())
        self.economies: list[tuple[Economy, Economy] | None] = [None] * games     # constructors and big buildings, for every side: policy_eco.py
        self.economy = economy
        self.cpp = None         # policy_scripts_cpp.Scripts, which plays the scripts of every game when script_impl is "cpp"
        if script_impl == "cpp":
            import policy_scripts_cpp
            self.cpp_data = policy_scripts_cpp.ScriptsData(self.script_data, self.eco_data)
            self.cpp = policy_scripts_cpp.Scripts(self.cpp_data, games, threads if script_threads is None else script_threads)
        self.seat_count = games * 2
        self.max_regions = max(b.regions for _, b, _ in self.maps)
        self.mode = [modes[0]] * games
        self.bundle = [self.maps[0][1]] * games
        self.starts = np.zeros((games, 4), np.float32)
        self.decisions = np.zeros(games, np.int32)
        self.last_raw = np.zeros(games)
        self.delta = np.zeros(games)
        self.result = np.zeros(games)
        self.pending: dict[int, list] = {}
        self.done: list[tuple[int, str, float]] = []
        self._replace: list[int] = []

    def bundles(self) -> list[bd.Bundle]:
        return [b for _, b, _ in self.maps]

    def begin(self) -> None:
        for g in range(self.count):
            self._reset(g)
        self.batch.step(ticks=0)
        self.last_raw[:] = self._raw()

    def _draw_map(self) -> tuple[zksim.Map, bd.Bundle, MapFacts]:
        if self.weights is None:
            return self.maps[int(self.rng.integers(len(self.maps)))]
        return self.maps[int(self.rng.choice(len(self.maps), p=self.weights))]

    def _reset(self, g: int) -> None:
        world_map, bundle, facts = self._draw_map()
        boxes = facts.layouts[int(self.rng.integers(len(facts.layouts)))]
        self.mode[g] = self.modes[int(self.rng.integers(len(self.modes)))]
        if self.rng.random() < self.self_play_share:
            self.mode[g] = "mirror"
        kinds = self.script_pair
        if kinds[0] != kinds[1] and self.rng.random() < 0.5:
            kinds = (kinds[1], kinds[0])
        self.script_kinds[g] = kinds
        # As Zero-K's "allshuffle": who gets which box is by chance, and on a map with more boxes than sides some stay empty.
        own, other = self.rng.choice(len(boxes), 2, replace=False)
        x0, z0 = boxes[own].pick(self.rng, bundle)
        x1, z1 = boxes[other].pick(self.rng, bundle)
        if self.opening is None:
            self._start_armies(g, world_map, bundle, x0, z0, x1, z1)
        else:
            self._start_commanders(g, world_map, bundle, boxes[own], boxes[other], x0, z0, x1, z1)
        self.batch.add_features(g, facts.features)
        self.bundle[g] = bundle
        self.starts[g] = (x0, z0, x1, z1)
        self.decisions[g] = 0

    def _start_armies(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, x0: float, z0: float, x1: float, z1: float) -> None:
        types = army_types(self.table)
        pick = self.rng.integers(0, len(types["index"]), self.units)
        ax, az = place(bundle, self.rng, types["classes"][pick], x0, z0)
        bx, bz = place(bundle, self.rng, types["classes"][pick], x1, z1)
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.tile(types["index"][pick], 2), np.repeat([0, 1], self.units),
                         np.concatenate((ax, bx)), np.concatenate((az, bz)), base=(x0, z0, x1, z1), sites=bundle.sites())
        kinds = self.script_kinds[g]
        if self.cpp is not None:
            self.scripts[g] = None
            self.cpp.reset(g, self.grounds[id(bundle)], (x0, z0, x1, z1), (x0, z0, x1, z1), economy=False, aimer=False)
            return
        self.scripts[g] = (make_script(kinds[0], self.script_data, (x0, z0), (x1, z1)), make_script(kinds[1], self.script_data, (x1, z1), (x0, z0)))

    def _start_commanders(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, box0: StartBox, box1: StartBox, x0: float, z0: float, x1: float, z1: float) -> None:
        """Handles 0 and 1 are the commanders, so a side's first own record is its commander while it lives."""
        commanders = self.opening.commanders
        types = [commanders[int(self.rng.integers(len(commanders)))] for _ in range(2)]
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.array(types), np.array([0, 1]), np.array([x0, x1]), np.array([z0, z1]),
                         commander=(0, 1), base=(x0, z0, x1, z1), sites=bundle.sites())
        self.batch.economy(g, bundle.spot_list)
        # A side plans towards where the enemy's box says it should be: what a player knows, then.
        ground = self.grounds[id(bundle)]
        self.seats[g] = (self.opening.seat(self.rng, 0, ground, (x0, z0), box1.point), self.opening.seat(self.rng, 1, ground, (x1, z1), box0.point))
        kinds, seats = self.script_kinds[g], self.seats[g]
        if self.cpp is not None:
            # The same draws from the environment's generator as the Python economies take, so that both play the same games' starts.
            seeds = tuple(int(self.rng.integers(1 << 62)) for _ in range(2)) if self.economy else (0, 0)
            self.scripts[g] = None
            self.cpp.reset(g, ground, (x0, z0, x1, z1), (*seats[0].plan.rally, *seats[1].plan.rally), economy=self.economy, aimer=True, seeds=seeds)
            return
        self.scripts[g] = tuple(
            make_script(kinds[side], self.script_data, seats[side].plan.rally, (x1, z1) if side == 0 else (x0, z0),
                        self.script_data.row_of_index(seats[side].raider_type), Aimer(self.eco_data))
            for side in (0, 1))
        if self.economy:
            self.economies[g] = tuple(
                Economy(self.eco_data, ground, side, base, enemy, np.random.default_rng(int(self.rng.integers(1 << 62))))
                for side, base, enemy in ((0, (x0, z0), (x1, z1)), (1, (x1, z1), (x0, z0))))

    def _opening_orders(self, g: int, side: int) -> np.ndarray | None:
        """The commander's next step, if it has finished this one or run out of time on it."""
        seats = self.seats[g]
        if seats is None or seats[side].done:
            return None
        seat, n = seats[side], int(self.batch.counts[g, side])
        units = self.batch.records[g, side, :n]
        if not n or units["uid"][0] != side:
            seat.done = True        # the commander is dead
            return None
        if not units["flags"][0] & rf.IDLE and self.decisions[g] < seat.deadline:
            return None
        orders = seat.advance(units, int(self.decisions[g]))
        return np.array(orders, zksim.ORDER_DTYPE) if orders else None

    def _economy_orders(self, g: int, side: int) -> np.ndarray | None:
        """Constructors' and big buildings' orders, for either kind of side.  Not the policy's, so not recorded as the script's."""
        economies = self.economies[g]
        if economies is None or self.decisions[g] % ECO_EVERY:
            return None
        mine = self.batch.frame(g, side)
        units, theirs = mine.units, self.batch.frame(g, 1 - side).units
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        seats = self.seats[g]
        busy = side if seats is not None and not seats[side].done else -1      # the commander, while the opening has it
        glob = mine.glob
        out = economies[side].orders(units[units["player"] == side], theirs[theirs["player"] == 1 - side],
                                     float(glob[rf.G["metal_income"]]), seconds, busy, float(glob[rf.G["metal"]]), float(glob[rf.G["metal_storage"]]),
                                     float(glob[rf.G["energy"]]), float(glob[rf.G["energy_storage"]]))
        return out if len(out) else None

    def _is_policy(self, seat: int) -> bool:
        mode = self.mode[seat // 2]
        return mode == "mirror" or (seat % 2 == 0 and mode != "scripts")

    def policy_seats(self) -> list[int]:
        return list(range(self.seat_count))

    def policy_seats_to_act(self) -> list[int]:
        return [s for s in range(self.seat_count) if self._is_policy(s)]

    def bundle_of(self, seat: int) -> bd.Bundle:
        return self.bundle[seat // 2]

    def frame(self, seat: int) -> rf.Frame:
        return self.batch.frame(seat // 2, seat % 2)

    def give_orders(self, seat: int, orders: list) -> None:
        self.pending[seat] = orders

    def script_orders(self, g: int, side: int) -> np.ndarray:
        if self.decisions[g] % self.script_every:
            return np.zeros(0, zksim.ORDER_DTYPE)
        mine = self.batch.frame(g, side)
        own, enemy = mine.units, self.batch.frame(g, 1 - side).units
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        script = self.scripts[g][side]
        if isinstance(script, ArmyScript):
            theirs = own["player"] != side
            seen = own["uid"][theirs & np.isin(own["knowledge"], (rf.KNOW_LOS, rf.KNOW_RADAR))]
            return script.orders(own[own["player"] == side], enemy[enemy["player"] == 1 - side], seconds, float(mine.glob[rf.G["metal_income"]]),
                                 seen)
        return script.orders(own[own["player"] == side], enemy[enemy["player"] == 1 - side], seconds)

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def _raw(self) -> np.ndarray:
        a, b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        return (a - b) / (a + b + VALUE_CONSTANT)

    def script_production(self, seat: int) -> list[tuple[int, str]]:
        """This decision's production choices by a script-controlled seat: (factory uid, unit name) for each PRODUCE it gave."""
        names = catalog().units
        return [(int(o["unit"]), names[int(o["target"])]["name"]) for o in self.script_record.get(seat, np.zeros(0, zksim.ORDER_DTYPE))
                if o["kind"] == zksim.ORDER_PRODUCE]

    def _scripts_cpp(self) -> tuple[np.ndarray, np.ndarray]:
        """Every game's orders with the scripts in C++: the policy's and the opening's from here, the scripts' from libzkscripts, in
        the order the Python path gives them.  Fills `script_record` as it does."""
        n = 2 * self.count
        flags, busy = np.zeros(n, np.int32), np.full(n, -1, np.int32)
        pre, post = [], []
        pre_counts, post_counts = np.zeros(n, np.int32), np.zeros(n, np.int32)
        for g in range(self.count):
            seats = self.seats[g]
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part = sim_orders(orders)
                    pre.append(part)
                    pre_counts[seat] = len(part)
                elif not self._is_policy(seat):
                    flags[seat] = 1         # policy_scripts_cpp.SEAT_ARMY
                opening = self._opening_orders(g, side)
                if opening is not None:
                    post.append(opening)
                    post_counts[seat] = len(opening)
                if seats is not None and not seats[side].done:
                    busy[seat] = side       # the commander, while the opening has it
        join = lambda parts, counts: (np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE),  # noqa: E731
                                      np.concatenate(([0], np.cumsum(counts))).astype(np.int32))
        orders, offsets, army, army_offsets = self.cpp.step(self.batch, self.decisions, flags, busy, self.ticks, self.script_every, ECO_EVERY,
                                                            pre=join(pre, pre_counts) if pre else None, post=join(post, post_counts) if post else None)
        self.script_record = {int(s): army[army_offsets[s]:army_offsets[s + 1]] for s in np.flatnonzero(flags)}
        return orders, offsets

    def step(self, progress: float) -> None:
        if self.cpp is not None:
            all_orders, offsets = self._scripts_cpp()
            self.batch.step(all_orders, offsets, ticks=self.ticks)
            self._after_step()
            return
        parts, offsets = [], [0]
        self.script_record = {}
        for g in range(self.count):
            total = 0
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part = sim_orders(orders)
                elif not self._is_policy(seat):
                    part = self.script_orders(g, side)
                    self.script_record[seat] = part
                else:
                    part = None
                if part is not None:
                    parts.append(part)
                    total += len(part)
                for extra in (self._opening_orders(g, side), self._economy_orders(g, side)):
                    if extra is not None:
                        parts.append(extra)
                        total += len(extra)
            offsets.append(offsets[-1] + total)
        all_orders = np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE)
        self.batch.step(all_orders, np.array(offsets, np.int32), ticks=self.ticks)
        self._after_step()

    def _made(self, g: int, side: int) -> dict:
        """What the side's army script has started, by role, for the outcome."""
        if self.cpp is not None:
            return self.cpp.made(g, side)
        script = self.scripts[g][side]
        return dict(script.mix.made) if isinstance(script, ArmyScript) else {}

    def _after_step(self) -> None:
        self.decisions += 1
        raw = self._raw()
        self.delta = raw - self.last_raw
        self.last_raw = raw
        self.done, self._replace = [], []
        self.result[:] = 0.0
        alive_a, alive_b = self.batch.stats[:, 0, 1] > 0, self.batch.stats[:, 1, 1] > 0
        value_a, value_b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        late = self.decisions >= self.early_end
        wins_a = (alive_a & ~alive_b) | (late & _far_ahead(value_a, value_b))
        wins_b = (alive_b & ~alive_a) | (late & _far_ahead(value_b, value_a))
        for g in np.flatnonzero(~(alive_a & alive_b) | wins_a | wins_b | (self.decisions >= self.time_limit)):
            g = int(g)
            won_a, won_b = bool(wins_a[g]), bool(wins_b[g])
            self.result[g] = 1.0 if won_a else (-1.0 if won_b else 0.0)
            for side, won, lost in ((0, won_a, won_b), (1, won_b, won_a)):
                seat = g * 2 + side
                facts = self._game_facts(g, side, won, lost)
                if self._is_policy(seat):
                    self.done.append((seat, self.mode[g], 1.0 if won else (0.0 if lost else 0.5)))
                    self.ended[seat] = facts
                self.outcomes.append({**facts, "kind": "policy" if self._is_policy(seat) else self.script_kinds[g][side],
                                      "made": self._made(g, side)})
            self._replace.append(g)

    def _game_facts(self, g: int, side: int, won: bool, lost: bool) -> dict:
        """A finished game as seen from one side, for the record of a test."""
        stats = self.batch.stats[g]
        seats = self.seats[g]
        names = catalog().units
        slugs = {id(b): f.name for _, b, f in self.maps}
        return {
            "map": slugs.get(id(self.bundle[g]), "?"), "seconds": float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND,
            "result": "won" if won else ("lost" if lost else "drew"),
            "how": "eliminated" if stats[0, 1] == 0 or stats[1, 1] == 0 else ("lead" if won or lost else "limit"),
            "own_value": float(stats[side, 0]), "enemy_value": float(stats[1 - side, 0]),
            "own_factory": names[seats[side].factory_type]["name"] if seats else "", "enemy_factory": names[seats[1 - side].factory_type]["name"] if seats else "",
        }

    def reward(self, seat: int, beta: float) -> float:
        g = seat // 2
        sign = 1.0 if seat % 2 == 0 else -1.0
        return float(sign * ((1.0 - beta) * self.delta[g] + beta * self.result[g]))

    def finished_seats(self) -> list[tuple[int, str, float]]:
        """Seats whose game just ended.  The call also starts those games afresh, so read the reward first."""
        out = self.done
        for g in self._replace:
            self._reset(g)
            self.batch.refresh(g)
            self.last_raw[g] = self._raw()[g]
        self._replace, self.done = [], []
        return out
