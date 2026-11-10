"""The simulator as an environment for tools/policy_train.py, with the same interface as the toy's.

A game starts from a commander in each of two of the map's start boxes.  Both sides play the opening of policy_opening.py by rule,
so what the policy controls is what it produces, and its constructors.  On the other side, the army script (C++, sim/scripts) controls the fighters and the factory, and the economy script (also C++) runs every side's constructors: the policy's too, so on its side both tell them, unless it runs its own economy (`policy_economy_share`).  The kinds the policy has are those the sim obeys (`KINDS`), BUILD and ASSIST among them; a BUILD goes where it fits (`place_build`).

What each script ordered at each decision is in `script_record`, and its production choices are readable with `script_production`.

Passing `units` gives the older start instead, two mirrored armies and no economy, which checks and benchmarks still use.

Passing `variants=True` gives each side's army script its own knobs at the start of each game (policy_variants.py), kept in `styles`.
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
from zkbot.policy import bundle as bd, groups, rawframe as rf, tables  # noqa: E402
from policy_eco import ECO_EVERY, EcoData  # noqa: E402
from policy_opening import Ground, Opening, Seat, is_buildable, modelled  # noqa: E402
from policy_script import ScriptData  # noqa: E402
import policy_variants as pv  # noqa: E402

KINDS = (rf.KEEP, rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.JUMP, rf.CLOAK_ON, rf.CLOAK_OFF,
         rf.PRODUCE, rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.BUILD, rf.ASSIST, rf.RECLAIM)
BUILD_SPREADS = (16.0, 32.0, 64.0, 128.0)     # elmos: how far round a BUILD's point a place is looked for, nearest first
BUILD_TRIES = 8
TICKS_PER_DECISION = 3
TIME_LIMIT_DECISIONS = 2400         # 20 minutes
VALUE_CONSTANT = 1000.0
INCOME_CONSTANT = 5.0               # metal a second: the income balance's, as VALUE_CONSTANT is the value balance's
DIVERSITY_DECAY = 0.999     # a decision, of the counts of what the policy's sides have made: their half-life is 700 decisions of the batch
SCRIPT_EVERY = 6                    # decisions: as often as the policy's commit lock lets it re-order
SPREAD = 220.0
MOST_COST = 600.0
EARLY_END_DECISIONS = 240           # 2 minutes: from then on, a side this far ahead in metal value has won
EARLY_END_RATIO = 2.5
EARLY_END_LEAD = 10000.0            # and this much ahead in metal, so that a small game isn't called on a few raiders
# 01-simulator.md, "Data".  Other exported maps stay out of training: playing pads every seat to the map with the most regions.
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


_SHARED: dict[int, tuple[list, dict, object]] = {}     # id(maps) -> (maps, their Grounds, the C++ scripts' data with every map added)


def share_maps(maps: list) -> None:
    """Makes, for every SimEnv on this map list, each map's `Ground` warmed with what the scripts ask of it, and the C++ scripts'
    data with every map added.  A recorder calls it before its workers fork, so that they share these (about 6 MB a map) rather
    than each make its own as it comes to a map: 12 workers on the 298-map pool came to 1.6 GB each."""
    import policy_scripts_cpp
    table = tables.table()
    data = policy_scripts_cpp.ScriptsData(ScriptData(table, catalog()), EcoData(table, catalog()))
    grounds = {id(b): Ground(b, f) for _, b, f in maps}
    for ground in grounds.values():
        ground.warm(data.footings)
        data.map_index(ground)
    _SHARED[id(maps)] = (maps, grounds, data)


def sim_orders(orders: list) -> np.ndarray:
    """The policy's orders as the sim's.  Its kinds up to PRODUCE have the same numbers; a thrower's fire at the ground, and a switch's target, are the sim's own.
    A group order is its unit's, followed by a groups.GROUP_MARK record with the radius as x and the roles as target, as the scripts record theirs:
    `split_groups` makes the sim's group orders of them."""
    table, cat = tables.table(), catalog()
    out = np.zeros(len(orders) + sum(o.radius > 0.0 for o in orders), zksim.ORDER_DTYPE)
    i = 0
    for o in orders:
        kind, target = o.kind, o.target
        if kind == rf.THROW:
            kind = zksim.ORDER_MANUAL_FIRE
        elif kind == rf.PUSH or kind == rf.PULL:
            kind, target = zksim.ORDER_ON_OFF, int(kind == rf.PUSH)
        elif kind == rf.PLACE_BEACON:
            kind = zksim.ORDER_PLACE_BEACON
        elif kind == rf.WAIT_AT_BEACON:
            kind = zksim.ORDER_WAIT_AT_BEACON
        elif kind == rf.PRODUCE or kind == rf.BUILD:
            target = cat.index.get(table.names[target], -1)
        out[i] = (o.uid, kind, o.x, o.z, target)
        i += 1
        if o.radius > 0.0:
            out[i] = (o.uid, groups.GROUP_MARK, o.radius, 0.0, o.roles)
            i += 1
    return out


def split_groups(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orders with group marks (`sim_orders`) as the sim takes them: the unit orders, and the group orders (zksim.GROUP_DTYPE)."""
    mark = rows["kind"] == groups.GROUP_MARK
    if not mark.any():
        return rows, np.zeros(0, zksim.GROUP_DTYPE)
    at = np.flatnonzero(mark)
    anchor = rows[at - 1]
    out = np.zeros(len(at), zksim.GROUP_DTYPE)
    out["anchor"], out["roles"], out["radius"] = anchor["unit"], rows["target"][at].astype(np.uint32), rows["x"][at]
    out["kind"], out["x"], out["z"], out["target"] = anchor["kind"], anchor["x"], anchor["z"], anchor["target"]
    alone = ~mark
    alone[at - 1] = False
    return rows[alone], out


def policy_orders(rows: np.ndarray) -> list:
    """`sim_orders` the other way: what a script told the sim, in the policy's terms, so it can be read as the policy's choices.
    A manual fire with no unit to aim at is a throw, as the policy has no other ground-aimed one.  A group mark makes the order before it a group order."""
    from dataclasses import replace
    from zkbot.policy import actions
    table, cat = tables.table(), catalog()
    out = []
    for o in rows:
        kind, target = int(o["kind"]), int(o["target"])
        if kind == groups.GROUP_MARK:
            out[-1] = replace(out[-1], radius=float(o["x"]), roles=target)
            continue
        if kind == zksim.ORDER_MANUAL_FIRE and target < 0:
            kind = rf.THROW
        elif kind == zksim.ORDER_ON_OFF:
            kind = rf.PUSH if target else rf.PULL
        elif kind == zksim.ORDER_PLACE_BEACON:
            kind = rf.PLACE_BEACON
        elif kind == zksim.ORDER_WAIT_AT_BEACON:
            kind = rf.WAIT_AT_BEACON
        elif kind == zksim.ORDER_PRODUCE or kind == zksim.ORDER_BUILD:
            target = table.names.index(cat.units[target]["name"])
        out.append(actions.Order(int(o["unit"]), kind, float(o["x"]), float(o["z"]), target))
    return out


def army_types(table: tables.TypeTable) -> dict[str, np.ndarray]:
    """Ground and water fighters of ordinary cost, by the data rather than by name.  Ships need a sea this start doesn't look for."""
    if not _TYPES:
        cat = catalog()
        # The factories' rosters (`bot_catalog.roster`), not every def there is: no chickens, and no test units.
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
    # A subclass that plays its own Python scripts (frame_world.FrameGames, frame_data.ScenarioEnv) sets this: then `scripts[g]`
    # holds one for each side, with `orders(own, enemy, seconds)`, and nothing but those and the opening runs a side.  Otherwise
    # the army and economy scripts are C++ (policy_scripts_cpp.py).
    own_scripts = False

    def __init__(self, games: int, seed: int, modes: list[str], units: int | None = None, threads: int = 16, decide_every: int = 1, maps: list | None = None,
                 weights: np.ndarray | None = None, economy: bool = True, script_threads: int | None = None, variants: bool = False):
        """maps: as `pool` or `load_pool` give, and weights how likely each is, if not evenly.  economy: whether the economy script
        runs constructors and big buildings for every side (policy_eco.py); without it, a side's economy is its opening, as before
        2026-11-08.  The army and economy scripts run in C++ on `script_threads` threads (`threads` if not given); the opening is
        Python.  variants: whether each side's army script draws its own knobs at each game's start (policy_variants.py); otherwise
        every side plays the standard script."""
        self.count, self.modes, self.units = games, modes, units
        self.maps = pool() if maps is None else maps
        shared = _SHARED.get(id(self.maps))
        shared = shared if shared is not None and shared[0] is self.maps else None
        self.grounds = shared[1] if shared else {id(b): Ground(b, f) for _, b, f in self.maps}
        self.weights = weights
        self.self_play_share = 0.0      # of games starting from now on, whatever `modes` says
        # The policy decides every this many of the sim's decisions, which step in between: at decide_every 1 and act_every 6, the
        # sim plays as the cloning data was recorded and the policy decides when the scripts look (policy_clone_eval).
        self.act_every = 1
        # The army script runs for the policy's seats too, only advising: its orders are labels (`army_orders`) and aren't carried out.
        # For learning from what it would have done in the policy's games (DAgger, PPO's cloning loss; 04-literature.md P3, P7).
        self.advise = False
        # Of games starting from now on, the share where the policy's seats run their own economy: the economy script only advises
        # them (its orders in `eco_orders` alone).  In the rest it builds for them as before, after the policy's orders.
        self.policy_economy_share = 0.0
        self.policy_economy = np.zeros(games, bool)
        # A small zero-sum reward for income (metal, as `_raw` is for value), on top of the value balance: 04-literature.md C7.
        self.income_weight = 0.0
        self.last_income = np.zeros(games)
        self.income_delta = np.zeros(games)
        self.ended: dict[int, dict] = {}    # a seat's last finished game, for a test to note
        self.script_kinds: list[tuple[str, str]] = [("army", "army")] * games
        self.variants_on = variants
        self.styles: list[tuple[pv.Variant, pv.Variant] | None] = [None] * games    # each side's army script knobs, when variants are on
        self.script_record: dict[int, np.ndarray] = {}      # seat -> the orders its script gave this decision, as the sim takes them
        # The same with the C++ scripts, over every seat: (orders, where each seat's start).  None with a subclass's own scripts.
        self.army_orders: tuple[np.ndarray, np.ndarray] | None = None
        # The C++ economy script's orders this decision, the same way: labels for cloning the economy (03-training.md, "Script data").
        self.eco_orders: tuple[np.ndarray, np.ndarray] | None = None
        # Whether the C++ army script sends what it can as group orders (capi.h zk_group_rec), which play exactly as the same orders
        # sent unit by unit.  They're in army_orders, as the policy would give them (policy_orders).
        self.group_orders = True
        self.outcomes: list[dict] = []                      # every side of every finished game, for a measurement to take and clear
        self.capture = None     # a policy_replay.Capture, told of every step: a test's games kept for the dashboard
        self.opening = Opening(catalog(), decide_every) if units is None else None
        self.seats: list[tuple[Seat, Seat] | None] = [None] * games
        self.scripts: list[tuple[object, object] | None] = [None] * games     # a subclass's own scripts: `own_scripts`
        # In slots of TICKS_PER_DECISION.  Game time stays what it was: the time limit, and how often the script re-orders.
        self.ticks = TICKS_PER_DECISION * decide_every
        self.time_limit = TIME_LIMIT_DECISIONS // decide_every
        self.early_end = EARLY_END_DECISIONS // decide_every
        self.script_every = max(1, SCRIPT_EVERY // decide_every)
        self.eco_every = max(1, ECO_EVERY // decide_every)
        self.rng = np.random.default_rng(seed)
        self.seed_stream = np.random.SeedSequence(seed)
        self.batch = zksim.Batch(catalog(), games, threads=threads)
        self.table = tables.table()
        self.script_data = ScriptData(self.table, catalog())
        self.eco_data = EcoData(self.table, catalog())
        self.economy = economy
        self.cpp = None         # policy_scripts_cpp.Scripts, which plays the army and economy scripts of every game
        if not self.own_scripts:
            import policy_scripts_cpp
            self.cpp_data = shared[2] if shared else policy_scripts_cpp.ScriptsData(self.script_data, self.eco_data)
            self.cpp = policy_scripts_cpp.Scripts(self.cpp_data, games, threads if script_threads is None else script_threads)
        self.batch.set_roles(self._role_of_type())
        self.seat_count = games * 2
        self.max_regions = max(b.regions for _, b, _ in self.maps)
        self.mode = [modes[0]] * games
        self.bundle = [self.maps[0][1]] * games
        self.starts = np.zeros((games, 4), np.float32)
        self.decisions = np.zeros(games, np.int32)
        self.last_raw = np.zeros(games)
        self.delta = np.zeros(games)
        self.result = np.zeros(games)
        # The diversity bonus (`_diversity`, after Pluto's): each side's this decision, and what's behind it.
        self.diversity = 0.0                        # a unit's bonus at most; 0 is none
        self.bonus = np.zeros((games, 2))
        self.makeable = np.zeros(len(self.table.mobile), bool)     # by type-table row: what a factory can make that moves
        self.makeable[self.table.producible] = True
        self.makeable &= np.asarray(self.table.mobile, bool)
        self.made = np.zeros(len(self.makeable))       # the policy's sides' finished units of those, by row, decayed
        self.known: list[list[np.ndarray | None]] = [[None, None] for _ in range(games)]     # each side's finished units so far
        self.pending: dict[int, list] = {}
        self.done: list[tuple[int, str, float]] = []
        self._replace: list[int] = []

    def _role_of_type(self) -> np.ndarray:
        """Each sim type's role (groups.roles), by the type-table row it is, for the sim's group orders."""
        _, role_of = groups.roles()
        rows = np.asarray(self.script_data.catalog_row)
        out = np.full(len(catalog().units), -1, np.int32)
        has = np.flatnonzero(rows >= 0)
        out[rows[has]] = role_of[has]
        return out

    def bundles(self) -> list[bd.Bundle]:
        return [b for _, b, _ in self.maps]

    def begin(self) -> None:
        for g in range(self.count):
            self._reset(g)
        self.batch.step(ticks=0)
        self.last_raw[:] = self._raw()
        self.last_income[:] = self._income()

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
        # Drawn only when it's on, so that the generator's draws, and so the games, are as before when it isn't.
        self.policy_economy[g] = self.policy_economy_share > 0.0 and self.rng.random() < self.policy_economy_share
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
        self.known[g] = [None, None]

    def _draw_styles(self, g: int) -> tuple[pv.Variant, pv.Variant] | None:
        """Each side's army script knobs for game `g`: drawn from the environment's generator when variants are on, else None."""
        self.styles[g] = (pv.draw(self.rng), pv.draw(self.rng)) if self.variants_on else None
        return self.styles[g]

    def _start_armies(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, x0: float, z0: float, x1: float, z1: float) -> None:
        types = army_types(self.table)
        pick = self.rng.integers(0, len(types["index"]), self.units)
        ax, az = place(bundle, self.rng, types["classes"][pick], x0, z0)
        bx, bz = place(bundle, self.rng, types["classes"][pick], x1, z1)
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.tile(types["index"][pick], 2), np.repeat([0, 1], self.units),
                         np.concatenate((ax, bx)), np.concatenate((az, bz)), base=(x0, z0, x1, z1), sites=bundle.sites())
        styles = self._draw_styles(g)
        if self.cpp is not None:
            self.cpp.reset(g, self.grounds[id(bundle)], (x0, z0, x1, z1), (x0, z0, x1, z1), economy=False, aimer=False, variants=styles)

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
        seats = self.seats[g]
        styles = self._draw_styles(g)
        if self.cpp is not None:
            seeds = tuple(int(self.rng.integers(1 << 62)) for _ in range(2)) if self.economy else (0, 0)
            self.cpp.reset(g, ground, (x0, z0, x1, z1), (*seats[0].plan.rally, *seats[1].plan.rally), economy=self.economy, aimer=True, seeds=seeds,
                           variants=styles)

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

    def _is_policy(self, seat: int) -> bool:
        mode = self.mode[seat // 2]
        return mode == "mirror" or (seat % 2 == 0 and mode != "scripts")

    def policy_seats(self) -> list[int]:
        return list(range(self.seat_count))

    def policy_seats_to_act(self) -> list[int]:
        """The policy's seats that decide now: every `act_every` decisions of their game."""
        return [s for s in range(self.seat_count) if self._is_policy(s) and self.decisions[s // 2] % self.act_every == 0]

    def bundle_of(self, seat: int) -> bd.Bundle:
        return self.bundle[seat // 2]

    def frame(self, seat: int) -> rf.Frame:
        return self.batch.frame(seat // 2, seat % 2)

    def give_orders(self, seat: int, orders: list) -> None:
        self.pending[seat] = orders

    def place_build(self, seat: int, row: int, x: float, z: float) -> tuple[float, float] | None:
        """Where a BUILD of this type-table row, told to (x, z), goes: an extractor on the free spot nearest the point; anything else
        at the nearest place to it that its footing fits, clear of our structures and of every metal spot (as the economy script
        keeps clear, eco.cpp `_taken`).  None if the sim can't build the type (no catalog record, a geothermal vent's) or it fits
        nowhere."""
        ed = self.eco_data
        if row >= len(ed.catalog_row) or ed.catalog_row[row] < 0 or ed.needs_place[row] or row not in ed.footing:
            return None
        f = self.frame(seat)
        if ed.extractor[row]:
            free = f.sites[f.sites["status"] == rf.SITE_FREE]
            if not len(free):
                return None
            k = int(np.argmin(np.hypot(free["x"] - x, free["z"] - z)))
            return float(free["x"][k]), float(free["z"][k])
        own = f.units[f.units["player"] == seat % 2]
        rows = self.table.rows(own["def_id"])
        still = ~self.table.mobile[rows]
        spots = np.asarray(self.bundle_of(seat).spot_list, np.float32).reshape(-1, 3)
        taken = np.concatenate((np.stack((own["x"][still], own["z"][still], ed.radius[rows[still]]), 1),
                                np.stack((spots[:, 0], spots[:, 1], np.full(len(spots), ed.radius[ed.extractor].max())), 1)))
        ground, footing = self.grounds[id(self.bundle_of(seat))], ed.footing[row]

        def fits(px: float, pz: float) -> bool:
            return ground.fits(px, pz, footing) and bool((np.hypot(taken[:, 0] - px, taken[:, 1] - pz) >= taken[:, 2] + footing.radius).all())

        # The point itself if it fits, else the nearest that does, looking further out a ring at a time.
        if fits(x, z):
            return x, z
        for spread in BUILD_SPREADS:
            for _ in range(BUILD_TRIES):
                px, pz = x + float(self.rng.normal(0.0, spread)), z + float(self.rng.normal(0.0, spread))
                if fits(px, pz):
                    return px, pz
        return ground.site(self.rng, footing, x, z, taken)

    def _placed(self, seat: int, rows: np.ndarray) -> np.ndarray:
        """The policy's BUILD orders put where they fit (`place_build`), and those that fit nowhere dropped.  rows: `sim_orders`'."""
        build = np.flatnonzero(rows["kind"] == zksim.ORDER_BUILD)
        if not len(build):
            return rows
        keep = np.ones(len(rows), bool)
        names = catalog().units
        for i in build:
            target = int(rows["target"][i])
            at = None if target < 0 else self.place_build(seat, self.table.names.index(names[target]["name"]), float(rows["x"][i]), float(rows["z"][i]))
            if at is None:
                keep[i] = False
            else:
                rows["x"][i], rows["z"][i] = at
        return rows[keep]

    def script_orders(self, g: int, side: int) -> np.ndarray:
        """A subclass's own script's orders for this side (`own_scripts`)."""
        if self.decisions[g] % self.script_every:
            return np.zeros(0, zksim.ORDER_DTYPE)
        mine = self.batch.frame(g, side)
        own, enemy = mine.units, self.batch.frame(g, 1 - side).units
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        return self.scripts[g][side].orders(own[own["player"] == side], enemy[enemy["player"] == 1 - side], seconds)

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def _raw(self) -> np.ndarray:
        a, b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        return (a - b) / (a + b + VALUE_CONSTANT)

    def _income(self) -> np.ndarray:
        """Side 0's metal income against side 1's, -1 to 1, as `_raw` is for value."""
        a, b = self.batch.globs[:, 0, rf.G["metal_income"]], self.batch.globs[:, 1, rf.G["metal_income"]]
        return (a - b) / (a + b + INCOME_CONSTANT)

    def script_production(self, seat: int) -> list[tuple[int, str]]:
        """This decision's production choices by a script-controlled seat: (factory uid, unit name) for each PRODUCE it gave."""
        names = catalog().units
        return [(int(o["unit"]), names[int(o["target"])]["name"]) for o in self.script_record.get(seat, np.zeros(0, zksim.ORDER_DTYPE))
                if o["kind"] == zksim.ORDER_PRODUCE]

    def _scripts_cpp(self) -> tuple[np.ndarray, np.ndarray]:
        """Every game's orders: the policy's and the opening's from here, the scripts' from libzkscripts.  Fills `script_record`
        with the army script's."""
        n = 2 * self.count
        flags, busy = np.zeros(n, np.int32), np.full(n, -1, np.int32)
        pre, post = [], []
        pre_counts, post_counts = np.zeros(n, np.int32), np.zeros(n, np.int32)
        policy_groups: list[list[np.ndarray]] = [[] for _ in range(self.count)]
        for g in range(self.count):
            seats = self.seats[g]
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part, grouped = split_groups(self._placed(seat, sim_orders(orders)))
                    pre.append(part)
                    pre_counts[seat] = len(part)
                    policy_groups[g].append(grouped)
                elif not self._is_policy(seat):
                    flags[seat] = 1         # policy_scripts_cpp.SEAT_ARMY
                if self._is_policy(seat):
                    flags[seat] |= (2 if self.advise else 0) | (4 if self.policy_economy[g] else 0)    # SEAT_ARMY_LABEL, SEAT_ECO_LABEL
                opening = self._opening_orders(g, side)
                if opening is not None:
                    post.append(opening)
                    post_counts[seat] = len(opening)
                if seats is not None and not seats[side].done:
                    busy[seat] = side       # the commander, while the opening has it
        join = lambda parts, counts: (np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE),  # noqa: E731
                                      np.concatenate(([0], np.cumsum(counts))).astype(np.int32))
        orders, offsets, army, army_offsets, groups, group_offsets = self.cpp.step(
            self.batch, self.decisions, flags, busy, self.ticks, self.script_every, self.eco_every, pre=join(pre, pre_counts) if pre else None,
            post=join(post, post_counts) if post else None, group_orders=self.group_orders)
        self.script_record = {int(s): army[army_offsets[s]:army_offsets[s + 1]] for s in np.flatnonzero(flags & 1)}
        self.army_orders = (army, army_offsets)
        self.eco_orders = self.cpp.eco_orders(self.count)
        if any(len(p) for parts in policy_groups for p in parts):      # the policy's group orders join the scripts'
            whole = [np.concatenate([groups[group_offsets[g]:group_offsets[g + 1]], *policy_groups[g]]) for g in range(self.count)]
            groups = np.concatenate(whole)
            group_offsets = np.concatenate(([0], np.cumsum([len(w) for w in whole]))).astype(np.int32)
        return orders, offsets, groups, group_offsets

    def step(self, progress: float) -> None:
        if self.cpp is not None:
            all_orders, offsets, groups, group_offsets = self._scripts_cpp()
            self.batch.step(all_orders, offsets, ticks=self.ticks, groups=groups, group_offsets=group_offsets)
            self._after_step()
            return
        # A subclass's own scripts.
        parts, offsets, group_parts, group_offsets = [], [0], [], [0]
        self.script_record = {}
        for g in range(self.count):
            total = grouped = 0
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part, some = split_groups(sim_orders(orders))
                    group_parts.append(some)
                    grouped += len(some)
                elif not self._is_policy(seat):
                    part = self.script_orders(g, side)
                    self.script_record[seat] = part
                else:
                    part = None
                if part is not None:
                    parts.append(part)
                    total += len(part)
                extra = self._opening_orders(g, side)
                if extra is not None:
                    parts.append(extra)
                    total += len(extra)
            offsets.append(offsets[-1] + total)
            group_offsets.append(group_offsets[-1] + grouped)
        all_orders = np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE)
        all_groups = np.concatenate(group_parts) if group_parts else np.zeros(0, zksim.GROUP_DTYPE)
        self.batch.step(all_orders, np.array(offsets, np.int32), ticks=self.ticks, groups=all_groups if len(all_groups) else None,
                        group_offsets=np.array(group_offsets, np.int32) if len(all_groups) else None)
        self._after_step()

    def _made(self, g: int, side: int) -> dict:
        """What the side's army script has started, by role, for the outcome."""
        return self.cpp.made(g, side) if self.cpp is not None else {}

    def rarity(self, rows: np.ndarray) -> np.ndarray:
        """0 to 1 for each type-table row: 0 for a type the policy's sides make at least as often as the average of what factories
        can make, and towards 1 the rarer it is, on a log scale (1 at a share of 1 / N squared, N those types: 122)."""
        n = int(self.makeable.sum())
        share = (self.made[rows] + 1.0) / (self.made.sum() + n)
        return np.clip(np.log(1.0 / (n * share)) / np.log(n), 0.0, 1.0)

    def _diversity(self) -> None:
        """Pluto's diversity bonus, for the units the policy chooses to make: each unit a factory could make that a side finishes
        this decision is worth `diversity` x its type's `rarity`.  Rarity is counted over the policy's sides only; both sides earn the bonus, and `reward`
        takes it zero-sum.  A side's units at its first decision of a game, the commander's among them, count as made already."""
        self.bonus[:] = 0.0
        if not self.diversity:
            return
        self.made *= DIVERSITY_DECAY
        for g in range(self.count):
            for side in (0, 1):
                u = self.batch.records[g, side, :int(self.batch.counts[g, side])]
                done = u["uid"][(u["player"] == side) & ((u["flags"] & rf.BEING_BUILT) == 0)].astype(np.int64)
                known = self.known[g][side]
                self.known[g][side] = np.union1d(done, known) if known is not None else np.unique(done)
                if known is None:
                    continue
                fresh = np.isin(u["uid"], done[~np.isin(done, known)]) & (u["player"] == side)
                rows = self.table.rows(u["def_id"][fresh])
                rows = rows[self.makeable[rows]]
                if not len(rows):
                    continue
                self.bonus[g, side] = self.diversity * float(self.rarity(rows).sum())
                if self._is_policy(g * 2 + side):
                    np.add.at(self.made, rows, 1.0)

    def _after_step(self) -> None:
        self.decisions += 1
        raw = self._raw()
        self.delta = raw - self.last_raw
        self.last_raw = raw
        if self.income_weight:
            income = self._income()
            self.income_delta = income - self.last_income
            self.last_income = income
        self._diversity()
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
        if self.capture is not None:
            self.capture.after_step(self)

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
            "variant": self.styles[g][side].name if self.styles[g] else "standard",
        }

    def reward(self, seat: int, beta: float) -> float:
        """Zero-sum: side 0's is the change in the value balance and its diversity bonus less side 1's, fading by beta into the result;
        side 1's is the opposite."""
        g = seat // 2
        sign = 1.0 if seat % 2 == 0 else -1.0
        shaped = self.delta[g] + self.bonus[g, 0] - self.bonus[g, 1] + self.income_weight * self.income_delta[g]
        return float(sign * ((1.0 - beta) * shaped + beta * self.result[g]))

    def finished_seats(self) -> list[tuple[int, str, float]]:
        """Seats whose game just ended.  The call also starts those games afresh, so read the reward first."""
        out = self.done
        for g in self._replace:
            self._reset(g)
            self.batch.refresh(g)
            self.last_raw[g] = self._raw()[g]
            self.last_income[g] = self._income()[g]
        self._replace, self.done = [], []
        return out
