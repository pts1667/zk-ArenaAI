"""The simulator as an environment for tools/policy_train.py, with the same interface as the toy's.

A game starts from a commander in each of two of the map's start boxes.  Both sides play the opening of policy_opening.py by rule,
so what the policy controls is the raiders it produces.  The fight script of policy_script.py controls them on its side of a game.  The kinds the policy has are those the sim obeys for fighting units; S-M7 adds the rest.

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
from policy_opening import Ground, Opening, Seat, is_buildable, modelled  # noqa: E402
from policy_script import FightScript, ScriptData  # noqa: E402

KINDS = (rf.KEEP, rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.JUMP, rf.CLOAK_ON, rf.CLOAK_OFF)
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

    def __init__(self, games: int, seed: int, modes: list[str], units: int | None = None, threads: int = 16, decide_every: int = 1, maps: list | None = None, weights: np.ndarray | None = None):
        """maps: as `pool` or `load_pool` give, and weights how likely each is, if not evenly."""
        self.count, self.modes, self.units = games, modes, units
        self.maps = pool() if maps is None else maps
        self.grounds = {id(b): Ground(b, f) for _, b, f in self.maps}
        self.weights = weights
        self.self_play_share = 0.0      # of games starting from now on, whatever `modes` says
        self.ended: dict[int, dict] = {}    # a seat's last finished game, for a test to note
        self.opening = Opening(catalog(), decide_every) if units is None else None
        self.seats: list[tuple[Seat, Seat] | None] = [None] * games
        self.scripts: list[tuple[FightScript, FightScript] | None] = [None] * games
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
        self.scripts[g] = (FightScript(self.script_data, (x0, z0), (x1, z1)), FightScript(self.script_data, (x1, z1), (x0, z0)))

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
        self.scripts[g] = (FightScript(self.script_data, self.seats[g][0].plan.rally, (x1, z1)), FightScript(self.script_data, self.seats[g][1].plan.rally, (x0, z0)))

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
        own, enemy = self.batch.frame(g, side).units, self.batch.frame(g, 1 - side).units
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        return self.scripts[g][side].orders(own[own["player"] == side], enemy[enemy["player"] == 1 - side], seconds)

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def _raw(self) -> np.ndarray:
        a, b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        return (a - b) / (a + b + VALUE_CONSTANT)

    def step(self, progress: float) -> None:
        parts, offsets = [], [0]
        for g in range(self.count):
            total = 0
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part = np.array([(o.uid, o.kind, o.x, o.z, o.target) for o in orders], zksim.ORDER_DTYPE)
                elif not self._is_policy(seat):
                    part = self.script_orders(g, side)
                else:
                    part = None
                if part is not None:
                    parts.append(part)
                    total += len(part)
                opening = self._opening_orders(g, side)
                if opening is not None:
                    parts.append(opening)
                    total += len(opening)
            offsets.append(offsets[-1] + total)
        all_orders = np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE)
        self.batch.step(all_orders, np.array(offsets, np.int32), ticks=self.ticks)
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
                if self._is_policy(seat):
                    self.done.append((seat, self.mode[g], 1.0 if won else (0.0 if lost else 0.5)))
                    self.ended[seat] = self._game_facts(g, side, won, lost)
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
