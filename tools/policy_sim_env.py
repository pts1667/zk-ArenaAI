"""The simulator as an environment for tools/policy_train.py, with the same interface as the toy's.

A game starts from a commander for each team in the map's start boxes.  Every team plays the opening of policy_opening.py by rule,
so what the policy controls is what it produces, and its constructors.  On the other side, the army script (C++, sim/scripts) controls the fighters and the factory, and the economy script (also C++) runs every side's constructors: the policy's too, so on its side both tell them, unless it runs its own economy (`policy_economy_share`).  The kinds the policy has are those the sim obeys (`KINDS`), BUILD and ASSIST among them; a BUILD goes where it fits (`place_build`).

Games of several teams (`kinds`): each game is drawn a kind, a 1v1, a team game (two alliances of 2 to 4 players), a
free-for-all (3 to 8 alliances of one) or a chicken game (1 to 4 players against Zero-K's chicken defense, which the sim plays,
at a difficulty drawn each game), and then a map whose `modes` hold it.  A seat is `game * slots + team`; with 2 slots, as a
batch of 1v1s has, that's `game * 2 + side` as it always was, and a 1v1 plays exactly as it did.  The reward is zero-sum over
the alliances (`reward`); an alliance with nothing left that keeps it in the game is out, its seats finished at once, and the
game goes on while two live (`_after_step`).  The chickens have no seat, and are out when their queen dies.

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
import map_modes  # noqa: E402
from policy_eco import ECO_EVERY, EcoData  # noqa: E402
from policy_opening import Ground, Opening, Seat, apart, footprints, is_buildable, modelled  # noqa: E402
from policy_script import ScriptData  # noqa: E402
import policy_variants as pv  # noqa: E402

KINDS = (rf.KEEP, rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.JUMP, rf.CLOAK_ON, rf.CLOAK_OFF,
         rf.PRODUCE, rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.BUILD, rf.ASSIST, rf.RECLAIM,
         rf.LOAD, rf.UNLOAD, rf.DROP)
BUILD_SPREADS = (16.0, 32.0, 64.0, 128.0)     # elmos: how far round a BUILD's point a place is looked for, nearest first
BUILD_TRIES = 8
TICKS_PER_DECISION = 3
TIME_LIMIT_DECISIONS = 9600         # 80 minutes (the user, 2026-11-10; it was 20): chicken defense keeps Zero-K's own timings
# Room for a frame's unit records, which a full frame cuts silently: a scripted 1v1 to 80 minutes held 1,292 in one frame
# (2026-11-10), past the batch's default 1,024.  64 games of 2 sides cost 42 MB at this, against 10.5 MB.
UNIT_RECORDS = 4096
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
NEAREST_START = 1024.0               # elmos: how far from a start point nobody can stand on a place is looked for
SPREAD_CANDIDATES = 16               # places in a box tried for each ally past its start points, the one farthest from the rest kept

# Kinds of game (the user, 2026-11-10), each on the maps whose `modes` hold it.
GAME_KINDS = ("1v1", "teams", "ffa", "chicken")
TEAM_SIDE = (2, map_modes.MOST_A_SIDE)          # players an alliance of a team game has, up to the map's modes["teams"]
FFA_ALLIANCES = (3, map_modes.MOST_ALLIANCES)   # alliances of one a free-for-all has, up to the map's modes["ffa"]
# A chicken game: these many players in one alliance, all in one box, against Zero-K's chicken defense (the sim's spawner), whose
# team comes after theirs in an alliance of its own and has no seat.  Their difficulty is drawn each game from weights by
# zksim.CHICKEN_DIFFICULTIES' names (SimEnv's `difficulties`): these until the scripts' measures say what they can beat.
CHICKEN_PLAYERS = (1, 4)
CHICKEN_DIFFICULTY_WEIGHTS = {"Very Easy": 1.0, "Easy": 1.0, "Normal": 1.0}
# Which of zksim's stats says a team is still in the game: from ABI 20 the live units that keep it in (not the Flea, the bombs,
# terraunit: Zero-K's game_over.lua), before it every live unit.
KEEP_STAT = 3 if zksim.STATS > 3 else 1

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
    points: tuple[tuple[float, float], ...] = ()    # every start point the box lists, `point` first; just `point` if not given

    def __post_init__(self):
        if not self.points:
            object.__setattr__(self, "points", (self.point,))

    def at_point(self, bundle: bd.Bundle, point: tuple[float, float], move_class: int | None) -> tuple[float, float]:
        """Where a commander of this move class starts when it's to start at one of the box's points: there, or, if it
        can't stand there, at the nearest place in the box with room for a base.  Esker Creek's points are on a ridge only
        spiders climb, and a commander put there never moved (2026-11-11); Zero-K drops it at the nearest spot it can
        stand on (start_unit_setup.lua, GetClosestValidSpawnSpot).  No class: there."""
        rows, cols = bundle.passable.shape
        cell = bundle.passable[min(int(point[1] / bd.PASS_CELL), rows - 1), min(int(point[0] / bd.PASS_CELL), cols - 1)]
        if move_class is None or (int(cell) >> move_class) & 1:
            return point
        gx, gz = (round(v / START_GRID) * START_GRID for v in point)
        for ring in range(1, int(NEAREST_START / START_GRID) + 1):
            steps = [k * START_GRID for k in range(-ring, ring + 1)]
            square = {(gx + dx, gz + dz) for dx in steps for dz in (steps[0], steps[-1])} | {(gx + dx, gz + dz) for dz in steps for dx in (steps[0], steps[-1])}
            found = [(float(x), float(z)) for x, z in square if any(_inside(polygon, x, z) for polygon in self.polygons) and is_buildable(bundle, x, z, START_ROOM)]
            if found:
                return min(found, key=lambda p: (np.hypot(p[0] - point[0], p[1] - point[1]), p))
        return point

    def pick(self, rng: np.random.Generator, bundle: bd.Bundle, move_class: int | None = None) -> tuple[float, float]:
        """A start for the box's one player, whose commander is of this move class (`at_point`)."""
        if rng.random() < AT_THE_START_POINT:
            return self.at_point(bundle, self.point, move_class)
        place = self._somewhere(rng, bundle)
        return self.at_point(bundle, self.point, move_class) if place is None else place

    def _somewhere(self, rng: np.random.Generator, bundle: bd.Bundle) -> tuple[float, float] | None:
        """A place inside the box with room for a base, or None if none turned up."""
        corners = np.array([v for polygon in self.polygons for v in polygon])
        low, high = corners.min(axis=0), corners.max(axis=0)
        for _ in range(START_TRIES):
            x, z = np.round(rng.uniform(low, high) / START_GRID) * START_GRID
            x, z = float(x), float(z)
            if any(_inside(polygon, x, z) for polygon in self.polygons) and is_buildable(bundle, x, z, START_ROOM):
                return x, z
        return None

    def picks(self, rng: np.random.Generator, bundle: bd.Bundle, n: int, move_class: int | None = None) -> list[tuple[float, float]]:
        """Starts for an alliance of n in the box: its start points, drawn among them, where it lists enough; else every point it
        lists and the rest apart inside it, each the farthest from those before of SPREAD_CANDIDATES places.  A point is where
        a commander of this move class would start for it (`at_point`)."""
        if len(self.points) >= n:
            return [self.at_point(bundle, self.points[int(i)], move_class) for i in rng.choice(len(self.points), n, replace=False)]
        out = [self.at_point(bundle, p, move_class) for p in self.points]
        while len(out) < n:
            found = [p for p in (self._somewhere(rng, bundle) for _ in range(SPREAD_CANDIDATES)) if p is not None]
            if not found:
                found = [self.point]        # a box with no room, which no pool map has: together at its point
            gaps = [min(np.hypot(x - ox, z - oz) for ox, oz in out) for x, z in found]
            out.append(found[int(np.argmax(gaps))])
        return out


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
    # Which kinds of game the map holds (tools/map_modes.py): {"1v1": bool, "teams": most a side, "ffa": most alliances,
    # "chickens": bool, "chickens_or_teams": bool}.  Its facts file's, or else from its boxes and name.
    modes: dict = field(default_factory=dict)
    # Maps of one group weigh as one map wherever they're drawn: a generator's maps of one real map's layout, as Violet Rampart's
    # three seeds of each number of bases (18 of the pool's 34 real free-for-all maps, 2026-11-10).  "" for a map of its own.
    group: str = ""

    def __post_init__(self):
        if not self.modes:      # from the boxes and the name, as on a map of unknown size: 2 a side if it's a team map
            boxes = max((len(b) for b in self.layouts), default=0)
            points = max((len(box.points) for b in self.layouts for box in b), default=1)
            object.__setattr__(self, "modes", map_modes.modes(map_modes.Known(boxes, points, names=(self.name,))))


def _layouts(facts: dict) -> list[list[StartBox]]:
    return [[StartBox(tuple(b["polygons"]), (float(b["startpoints"][0][0]), float(b["startpoints"][0][1])), tuple((float(x), float(z)) for x, z in b["startpoints"]))
             for b in boxes] for boxes in facts["layouts"]]


def _map_facts(slug: str) -> MapFacts:
    facts = json.loads((zksim.DATA / "map_facts.json").read_text(encoding="utf8"))[slug]
    area = float(facts.get("width", 0.0)) * float(facts.get("height", 0.0)) / (map_modes.SIDE * map_modes.SIDE)
    return MapFacts(_layouts(facts), np.array(facts["features"], np.float32).reshape(-1, 6), name=slug, modes=facts.get("modes") or map_modes.of_facts({"name": slug, "layouts": facts["layouts"]}, area))


def load_pool(directory: pathlib.Path | str) -> list[tuple[zksim.Map, bd.Bundle, MapFacts]]:
    """All the maps of a directory that tools/map_gen.py wrote to, in the order of their names."""
    out = []
    for path in sorted(pathlib.Path(directory).glob("*.facts.json")):
        slug = path.name[:-len(".facts.json")]
        facts = json.loads(path.read_text(encoding="utf8"))
        with np.load(path.with_name(f"{slug}.reach.npz")) as d:
            reach = {k: d[k] for k in d.files}
        bundle = bd.Bundle.load(path.with_name(f"{slug}.bundle.npz"))
        # A pool written before modes were, from its boxes and names.
        modes = facts.get("modes") or map_modes.of_facts(facts, bundle.map_w * bundle.map_h / (map_modes.SIDE * map_modes.SIDE))
        style = facts.get("style") or {}
        group = f"{style['generator']}:{style.get('bases', '')}" if style.get("generator") else ""
        out.append((zksim.Map.load(path.with_name(f"{slug}.npz")), bundle,
                    MapFacts(_layouts(facts), np.array(facts["features"], np.float32).reshape(-1, 6), slug, facts["kind"], float(facts["water"]), facts["liquid"], reach, modes,
                             group)))
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
    grounds = {id(b): Ground(b, f, m.height) for m, b, f in maps}
    for ground in grounds.values():
        ground.warm(data.footings)
        data.map_index(ground)
    _SHARED[id(maps)] = (maps, grounds, data)


# An air transport's kinds, the policy's to the sim's and back: their numbers differ.
_TRANSPORT_ORDERS = {rf.LOAD: zksim.ORDER_LOAD, rf.UNLOAD: zksim.ORDER_UNLOAD, rf.DROP: zksim.ORDER_DROP}
_TRANSPORT_KINDS = {order: kind for kind, order in _TRANSPORT_ORDERS.items()}


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
        elif kind in _TRANSPORT_ORDERS:
            kind = _TRANSPORT_ORDERS[kind]
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
        elif kind in _TRANSPORT_KINDS:
            kind = _TRANSPORT_KINDS[kind]
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


def parse_kinds(items) -> dict[str, float]:
    """`--kinds 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1` (or one string of them) as {kind: weight}; a kind without a weight weighs 1."""
    if isinstance(items, str):
        items = items.split()
    out = {}
    for item in items:
        kind, _, weight = item.partition(":")
        if kind not in GAME_KINDS:
            raise ValueError(f"no kind of game {kind!r}: there are {', '.join(GAME_KINDS)}")
        out[kind] = float(weight) if weight else 1.0
    return out


def _difficulty_index(name) -> int:
    """A chicken difficulty by zksim.CHICKEN_DIFFICULTIES' index or name, in any case, with - or _ for its spaces."""
    if isinstance(name, (int, np.integer)) or str(name).isdigit():
        index = int(name)
        if not 0 <= index < len(zksim.CHICKEN_DIFFICULTIES):
            raise ValueError(f"no chicken difficulty {index}: there are 0 to {len(zksim.CHICKEN_DIFFICULTIES) - 1}")
        return index
    plain = [d.lower() for d in zksim.CHICKEN_DIFFICULTIES]
    said = str(name).lower().replace("-", " ").replace("_", " ")
    if said not in plain:
        raise ValueError(f"no chicken difficulty {name!r}: there are {', '.join(zksim.CHICKEN_DIFFICULTIES)}")
    return plain.index(said)


def difficulty_weights(weights) -> dict[int, float]:
    """Chicken difficulties' weights as {index: weight}, from a dict by name or index, or from `--chicken-difficulties easy:1
    normal:0.5` (a difficulty without a weight weighs 1)."""
    if isinstance(weights, dict):
        pairs = list(weights.items())
    else:
        pairs = []
        for item in weights.split() if isinstance(weights, str) else weights:
            name, _, w = item.partition(":")
            pairs.append((name, float(w) if w else 1.0))
    out = {_difficulty_index(name): float(w) for name, w in pairs if float(w) > 0}
    if not out:
        raise ValueError("no chicken difficulty with a weight above 0")
    return out


def slots_for(kinds: dict[str, float]) -> int:
    """The teams a game the batch needs room for, for the biggest game of these kinds: a chicken game's chickens take one."""
    need = {"1v1": 2, "teams": 2 * TEAM_SIDE[1], "ffa": FFA_ALLIANCES[1], "chicken": CHICKEN_PLAYERS[1] + 1}
    return max(need[k] for k, w in kinds.items() if w > 0)


def holds(facts: MapFacts, kind: str, slots: int) -> bool:
    """Whether a game of this kind can be played on the map in a batch with this many slots a game."""
    modes = facts.modes
    if kind == "1v1":
        return bool(modes.get("1v1"))
    if kind == "teams":
        return modes.get("teams", 0) >= TEAM_SIDE[0] and slots >= 2 * TEAM_SIDE[0] and any(len(b) >= 2 for b in facts.layouts)
    if kind == "ffa":
        return modes.get("ffa", 0) >= FFA_ALLIANCES[0] and slots >= FFA_ALLIANCES[0] and any(len(b) >= FFA_ALLIANCES[0] for b in facts.layouts)
    if kind == "chicken":       # chicken maps and team maps (map_modes); a pool written before that key, from the two it joins
        either = modes.get("chickens_or_teams", bool(modes.get("chickens")) or modes.get("teams", 0) > 0)
        return bool(either) and slots >= CHICKEN_PLAYERS[0] + 1 and any(len(b) for b in facts.layouts)
    raise ValueError(kind)


class SimEnv:
    eval_opponents = ("fight",)
    # A subclass that plays its own Python scripts (frame_world.FrameGames, frame_data.ScenarioEnv) sets this: then `scripts[g]`
    # holds one for each side, with `orders(own, enemy, seconds)`, and nothing but those and the opening runs a side.  Otherwise
    # the army and economy scripts are C++ (policy_scripts_cpp.py).
    own_scripts = False

    def __init__(self, games: int, seed: int, modes: list[str], units: int | None = None, threads: int = 16, decide_every: int = 1, maps: list | None = None,
                 weights: np.ndarray | None = None, economy: bool = True, script_threads: int | None = None, variants: bool = False,
                 teams: int | None = None, kinds: dict[str, float] | None = None, difficulties=None):
        """maps: as `pool` or `load_pool` give, and weights how likely each is, if not evenly.  economy: whether the economy script
        runs constructors and big buildings for every side (policy_eco.py); without it, a side's economy is its opening, as before
        2026-11-08.  The army and economy scripts run in C++ on `script_threads` threads (`threads` if not given); the opening is
        Python.  variants: whether each side's army script draws its own knobs at each game's start (policy_variants.py); otherwise
        every side plays the standard script.  kinds: how likely each kind of game is (GAME_KINDS; `parse_kinds`), 1v1s only if
        None.  teams: the slots a game, for the biggest game of those kinds if None (`slots_for`): 2 for 1v1s.  difficulties: how
        likely each chicken difficulty is, by name or index (`difficulty_weights`), CHICKEN_DIFFICULTY_WEIGHTS if None."""
        self.count, self.modes, self.units = games, modes, units
        # Of chicken games starting from now on, by zksim.CHICKEN_DIFFICULTIES' index.
        self.chicken_difficulties = difficulty_weights(CHICKEN_DIFFICULTY_WEIGHTS if difficulties is None else difficulties)
        self.maps = pool() if maps is None else maps
        self.kinds = {k: float(w) for k, w in ({"1v1": 1.0} if kinds is None else kinds).items() if w > 0}
        unknown = set(self.kinds) - set(GAME_KINDS)
        if unknown or not self.kinds:
            raise ValueError(f"kinds of game {sorted(unknown) or 'none'}: there are {', '.join(GAME_KINDS)}")
        if units is not None and set(self.kinds) != {"1v1"}:
            raise ValueError("the older start (units) is two armies: 1v1s only")
        self.slots = slots_for(self.kinds) if teams is None else int(teams)
        assert 2 <= self.slots <= zksim.MAX_TEAMS, self.slots
        self.weights = weights
        self._kind_names = list(self.kinds)
        self._kind_p = np.array(list(self.kinds.values())) / sum(self.kinds.values())
        self._map_draws = {kind: self._map_choices(kind) for kind in self.kinds}
        shared = _SHARED.get(id(self.maps))
        shared = shared if shared is not None and shared[0] is self.maps else None
        self.grounds = shared[1] if shared else {id(b): Ground(b, f, m.height) for m, b, f in self.maps}
        self.self_play_share = 0.0      # of games starting from now on, whatever `modes` says
        # The policy decides every this many of the sim's decisions, which step in between: at decide_every 1 and act_every 6, the
        # sim plays as the cloning data was recorded and the policy decides when the scripts look (policy_clone_eval).
        self.act_every = 1
        # Every game starts on a slot that act_every divides, so that all the policy's seats decide together: one forward pass a
        # decision rather than one for each of act_every phases, each of which costs a sequential sampling loop (sample_actions) about
        # the same whatever its batch (2026-11-10: 89% of PPO's collection, at 13.6 seats a pass).  A game that ends between such
        # slots waits, ended and untouched, for the next to start afresh: so each game's own count, which its scripts and opening go
        # by, is as before.  The diversity bonus is then counted once a decision, at the last slot it covers.
        self.align = False
        self.slot = 0                       # steps since begin
        self.waiting = np.zeros(games, bool)    # ended, waiting for such a slot to start afresh
        # The army script runs for the policy's seats too, only advising: its orders are labels (`army_orders`) and aren't carried out.
        # For learning from what it would have done in the policy's games (DAgger, PPO's cloning loss; 04-literature.md P3, P7).
        self.advise = False
        # Of games starting from now on, the share where the policy's seats run their own economy: the economy script only advises
        # them (its orders in `eco_orders` alone).  In the rest it builds for them as before, after the policy's orders.
        self.policy_economy_share = 0.0
        self.policy_economy = np.zeros(games, bool)
        # A small zero-sum reward for income (metal, as `_raw` is for value), on top of the value balance: 04-literature.md C7.
        self.income_weight = 0.0
        self.last_income = np.zeros((games, self.slots))       # by alliance, as `last_raw`
        self.income_delta = np.zeros((games, self.slots))
        # A small reward for short games (the user, 2026-11-10): an alliance that goes out pays this x the share of the time limit
        # left / (alliances - 1) (`_time_left`), and the survivors at the game's end share what was paid, as the result.  Not faded by
        # beta, so that a sooner win is worth more all through a run.
        self.speed_weight = 0.0
        # Of games starting from now on, the share where the lead rule (`_far_ahead`) can end the game; drawn at each one's start
        # into `lead_rule`.  PPO lowers it, so that the policy learns to finish games (the user, 2026-11-10); tests, recording and
        # evals keep it whole.
        self.lead_share = 1.0
        self.lead_rule = np.ones(games, bool)
        self.ended: dict[int, dict] = {}    # a seat's last finished game, for a test to note
        self.script_kinds: list[tuple[str, ...]] = [("army", "army")] * games      # each team's
        self.variants_on = variants
        self.styles: list[tuple[pv.Variant, ...] | None] = [None] * games    # each team's army script knobs, when variants are on
        self.script_record: dict[int, np.ndarray] = {}      # seat -> the orders its script gave this decision, as the sim takes them
        # The same with the C++ scripts, over every seat: (orders, where each seat's start).  None with a subclass's own scripts.
        self.army_orders: tuple[np.ndarray, np.ndarray] | None = None
        # The C++ economy script's orders this decision, the same way: labels for cloning the economy (03-training.md, "Script data").
        self.eco_orders: tuple[np.ndarray, np.ndarray] | None = None
        # Whether the C++ army script sends what it can as group orders (capi.h zk_group_rec), which play exactly as the same orders
        # sent unit by unit.  They're in army_orders, as the policy would give them (policy_orders).
        self.group_orders = True
        # Every team of every game, as its alliance goes out or the game ends (`_game_facts`, with "game", and "kind": who played
        # it), for a measurement to take and clear.
        self.outcomes: list[dict] = []
        self.capture = None     # a policy_replay.Capture, told of every step: a test's games kept for the dashboard
        self.opening = Opening(catalog(), decide_every) if units is None else None
        self.seats: list[tuple[Seat, ...] | None] = [None] * games       # each team's opening
        self.scripts: list[tuple | None] = [None] * games     # a subclass's own scripts, one a team: `own_scripts`
        self._units_of: dict[int, tuple[int, list]] = {}     # `_team_units`, by game, with the step it was made at
        # In slots of TICKS_PER_DECISION.  Game time stays what it was: the time limit, and how often the script re-orders.
        self.ticks = TICKS_PER_DECISION * decide_every
        self.time_limit = TIME_LIMIT_DECISIONS // decide_every
        self.early_end = EARLY_END_DECISIONS // decide_every
        self.script_every = max(1, SCRIPT_EVERY // decide_every)
        self.eco_every = max(1, ECO_EVERY // decide_every)
        self.rng = np.random.default_rng(seed)
        self.seed_stream = np.random.SeedSequence(seed)
        self.batch = zksim.Batch(catalog(), games, threads=threads, cap=UNIT_RECORDS, teams=self.slots)
        self.table = tables.table()
        self.script_data = ScriptData(self.table, catalog())
        # Where a commander can stand, for its start (StartBox.at_point): they're all of one class, amphibious walkers.
        classes = {int(self.table.move_class[self.script_data.row_of_index(c)]) for c in self.opening.commanders} if self.opening else set()
        self.commander_class = classes.pop() if len(classes) == 1 else None
        self.eco_data = EcoData(self.table, catalog())
        self.economy = economy
        self.cpp = None         # policy_scripts_cpp.Scripts, which plays the army and economy scripts of every game
        if not self.own_scripts:
            import policy_scripts_cpp
            self.cpp_data = shared[2] if shared else policy_scripts_cpp.ScriptsData(self.script_data, self.eco_data)
            self.cpp = policy_scripts_cpp.Scripts(self.cpp_data, games, threads if script_threads is None else script_threads, teams=self.slots)
        self.batch.set_roles(self._role_of_type())
        self.seat_count = games * self.slots
        self.max_regions = max(b.regions for _, b, _ in self.maps)
        self.mode = [modes[0]] * games
        self.bundle = [self.maps[0][1]] * games
        self.starts = np.zeros((games, 2 * self.slots), np.float32)     # x, z of each team's start
        self.decisions = np.zeros(games, np.int32)
        # Each game's teams (`_new_teams`): its kind, how many teams it has (the first of its slots), each one's alliance (-1 for an
        # unused slot), how many alliances, and which are still in.  Until a game says otherwise, a 1v1.
        self._rows = np.arange(games)
        self.kind = ["1v1"] * games
        self.n_teams = np.full(games, 2, np.int32)
        self.ally = np.full((games, self.slots), -1, np.int32)
        self.n_allies = np.full(games, 2, np.int32)
        # A chicken game's chickens: their team (-1 in other games), its difficulty, and the sim's chicken_state of it after the
        # last step, which says when the queen is dead.
        self.chickens = np.full(games, -1, np.int32)
        self.difficulty = np.full(games, -1, np.int32)
        self.chicken_states: dict[int, dict[str, float]] = {}
        self.alive = np.zeros((games, self.slots), bool)        # by alliance
        self.paying = np.zeros((games, self.slots), bool)       # the alliances in at the start of the step just taken
        self.facing = np.zeros((games, self.slots), np.int32)   # by team: the enemy team its opening plans towards
        # Each team's result, set when its alliance goes out or the game ends; `result` is team 0's at its game's end, and 0 else.
        self.outcome = np.zeros((games, self.slots))
        self.result = np.zeros(games)
        # By alliance: the value balance (`_raw`) at the last step and its change; and the result and the speed term paid at the
        # step just taken, with what the alliances out so far have paid, which the survivors share at the end.
        self.last_raw = np.zeros((games, self.slots))
        self.delta = np.zeros((games, self.slots))
        self.paid = np.zeros((games, self.slots))
        self.speed_paid = np.zeros((games, self.slots))     # as shares of the time limit: `reward` weighs them by speed_weight
        self.speed_pot = np.zeros(games)
        for g in range(games):
            self._new_teams(g)
        # The diversity bonus (`_diversity`, after Pluto's): each team's this decision, and what's behind it.
        self.diversity = 0.0                        # a unit's bonus at most; 0 is none
        self.bonus = np.zeros((games, self.slots))
        self.makeable = np.zeros(len(self.table.mobile), bool)     # by type-table row: what a factory can make that moves
        self.makeable[self.table.producible] = True
        self.makeable &= np.asarray(self.table.mobile, bool)
        self.made = np.zeros(len(self.makeable))       # the policy's sides' finished units of those, by row, decayed
        self.known: list[list[np.ndarray | None]] = [[None] * self.slots for _ in range(games)]     # each team's finished units so far
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
            self._new_teams(g)
            self._reset(g)
        self.batch.step(ticks=0)
        self.last_raw[:] = self._raw()
        self.last_income[:] = self._income()

    def _new_teams(self, g: int, kind: str = "1v1", ally_of=(0, 1)) -> None:
        """Game g's teams, each one's alliance: a 1v1 unless told otherwise (as a subclass's own `_reset` leaves them).  In a chicken
        game the chickens are the last team."""
        n = len(ally_of)
        self.kind[g] = kind
        self.n_teams[g] = n
        self.ally[g] = -1
        self.ally[g, :n] = ally_of
        self.n_allies[g] = max(ally_of) + 1
        self.alive[g] = False
        self.alive[g, :self.n_allies[g]] = True
        self.outcome[g] = 0.0
        self.speed_pot[g] = 0.0
        self.chickens[g] = n - 1 if kind == "chicken" else -1
        self.difficulty[g] = -1
        self.chicken_states.pop(g, None)
        self.script_kinds[g] = tuple("chickens" if t == self.chickens[g] else "army" for t in range(n))

    def game_of(self, seat: int) -> int:
        return seat // self.slots

    def team_of(self, seat: int) -> int:
        return seat % self.slots

    def plays(self, g: int, team: int) -> str | None:
        """Who plays a team of game g: "policy", "script" or "chickens"; None for an unused slot."""
        if team >= self.n_teams[g]:
            return None
        if team == self.chickens[g]:
            return "chickens"
        return "policy" if self._is_policy(g * self.slots + team) else "script"

    def _map_choices(self, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
        """The maps a game of this kind is drawn from and how likely each is: `weights` among the maps that hold it, a group's maps
        sharing one map's weight (MapFacts.group).  None where that's every map of the pool, none grouped, which draws as before."""
        held = np.array([holds(f, kind, self.slots) for _, _, f in self.maps])
        groups = np.array([f.group for _, _, f in self.maps])
        if held.all() and not (groups != "").any():
            return None
        w = (np.ones(len(self.maps)) if self.weights is None else np.array(self.weights, np.float64)) * held
        for name in set(groups[held & (groups != "")]):
            members = held & (groups == name)
            w[members] /= members.sum()
        index = np.flatnonzero(w > 0)
        if not len(index):
            raise ValueError(f"no map of the pool holds a game of kind {kind!r} in {self.slots} slots")
        return index, w[index] / w[index].sum()

    def _draw_map(self, kind: str = "1v1") -> tuple[zksim.Map, bd.Bundle, MapFacts]:
        choices = self._map_draws[kind]
        if choices is not None:
            index, p = choices
            return self.maps[int(index[self.rng.choice(len(index), p=p)])]
        if self.weights is None:
            return self.maps[int(self.rng.integers(len(self.maps)))]
        return self.maps[int(self.rng.choice(len(self.maps), p=self.weights))]

    def _draw_kind(self) -> str:
        """Drawn only where there's a choice, so that a 1v1-only environment's draws, and so its games, are as they were."""
        if len(self._kind_names) == 1:
            return self._kind_names[0]
        return self._kind_names[int(self.rng.choice(len(self._kind_names), p=self._kind_p))]

    def _draw_difficulty(self) -> int:
        keys = list(self.chicken_difficulties)
        w = np.array([self.chicken_difficulties[k] for k in keys], np.float64)
        return keys[int(self.rng.choice(len(keys), p=w / w.sum()))]

    def _sizes(self, kind: str, facts: MapFacts, boxes: list[StartBox]) -> list[int]:
        """Each player alliance's players: two alliances of one (nothing drawn), two of the same 2 to 4, 3 to 8 of one, or against
        the chickens one of 1 to 4."""
        if kind == "1v1":
            return [1, 1]
        if kind == "teams":
            most = min(int(facts.modes["teams"]), self.slots // 2, TEAM_SIDE[1])
            return [int(self.rng.integers(TEAM_SIDE[0], most + 1))] * 2
        if kind == "chicken":
            return [int(self.rng.integers(CHICKEN_PLAYERS[0], min(CHICKEN_PLAYERS[1], self.slots - 1) + 1))]
        most = min(int(facts.modes["ffa"]), self.slots, len(boxes), FFA_ALLIANCES[1])
        return [1] * int(self.rng.integers(FFA_ALLIANCES[0], most + 1))

    def _draw_boxes(self, kind: str, facts: MapFacts) -> list[StartBox]:
        """One of the map's layouts.  A free-for-all needs a box an alliance: on a map that draws its layout, one with enough."""
        layouts = facts.layouts if kind != "ffa" else [b for b in facts.layouts if len(b) >= FFA_ALLIANCES[0]]
        return layouts[int(self.rng.integers(len(layouts)))]

    def _draw_starts(self, kind: str, bundle: bd.Bundle, facts: MapFacts, boxes: list[StartBox]) -> tuple[list, list, list[int], np.ndarray]:
        """A game's teams: each one's start, where the others take it to be (`known`) and its alliance; and each player alliance's
        box, as an index.  As Zero-K's "allshuffle": which alliance gets which box is by chance, and on a map with more boxes than
        alliances some stay empty.  On a chicken map the players take the first box: it's theirs ("Robot Defenders", with the
        map's metal; the other is the hive's), and Zero-K doesn't shuffle a game of one player alliance (start_boxes.lua), so
        that's the one they get.  An alliance of one starts as a 1v1's side does; allies at their box's start points, or apart
        in it (StartBox.picks).  Where the others plan towards is the box's point, or a start point where a team has one of its
        own.  The chickens, last in a chicken game, have no start: their burrows come up away from the players.  Their place,
        which the players' openings and scripts take as the enemy's start, is the map's middle."""
        sizes = self._sizes(kind, facts, boxes)
        if kind == "chicken" and facts.modes.get("chickens"):
            chosen = np.arange(len(sizes))
        else:
            chosen = self.rng.choice(len(boxes), len(sizes), replace=False)
        starts, known, ally_of = [], [], []
        for a, (b, n) in enumerate(zip(chosen, sizes)):
            box = boxes[b]
            if n == 1:
                starts.append(box.pick(self.rng, bundle, self.commander_class))
                known.append(box.point)
            else:
                for p in box.picks(self.rng, bundle, n, self.commander_class):
                    starts.append(p)
                    known.append(p if p in box.points else box.point)
            ally_of += [a] * n
        if kind == "chicken":
            middle = (bundle.map_w / 2.0, bundle.map_h / 2.0)
            starts.append(middle)
            known.append(middle)
            ally_of.append(1)
        return starts, known, ally_of, chosen

    def _reset(self, g: int) -> None:
        kind = self._draw_kind()
        world_map, bundle, facts = self._draw_map(kind)
        boxes = self._draw_boxes(kind, facts)
        self.mode[g] = self.modes[int(self.rng.integers(len(self.modes)))]
        if self.rng.random() < self.self_play_share:
            self.mode[g] = "mirror"
        # Drawn only when it's on, so that the generator's draws, and so the games, are as before when it isn't.
        self.policy_economy[g] = self.policy_economy_share > 0.0 and self.rng.random() < self.policy_economy_share
        # The same: whole, nothing is drawn.  Chicken defense has no such rule.
        self.lead_rule[g] = (self.lead_share >= 1.0 or self.rng.random() < self.lead_share) and kind != "chicken"
        starts, known, ally_of, chosen = self._draw_starts(kind, bundle, facts, boxes)
        difficulty = self._draw_difficulty() if kind == "chicken" else -1
        self._new_teams(g, kind, ally_of)
        self.difficulty[g] = difficulty
        (x0, z0), (x1, z1) = starts[0], starts[1]
        if self.opening is None:
            self._start_armies(g, world_map, bundle, x0, z0, x1, z1)
        elif kind == "1v1":
            self._start_commanders(g, world_map, bundle, boxes[chosen[0]], boxes[chosen[1]], x0, z0, x1, z1)
        else:
            self._start_teams(g, world_map, bundle, starts, known)
        self.batch.add_features(g, facts.features)
        self.bundle[g] = bundle
        self.starts[g] = 0.0
        self.starts[g, :2 * len(starts)] = [v for s in starts for v in s]
        self.decisions[g] = 0
        self.known[g] = [None] * self.slots

    def _draw_styles(self, g: int) -> tuple[pv.Variant, ...] | None:
        """Each team's army script knobs for game `g`: drawn from the environment's generator when variants are on, else None.  The
        chickens, whom no script plays, have the standard's."""
        chickens = int(self.chickens[g])
        self.styles[g] = tuple(pv.STANDARD if t == chickens else pv.draw(self.rng) for t in range(int(self.n_teams[g]))) if self.variants_on else None
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
        """A 1v1's start, which a subclass may play its own way: each side plans towards where the enemy's box says it should be,
        what a player knows then."""
        self._start_teams(g, world_map, bundle, [(x0, z0), (x1, z1)], [box0.point, box1.point])

    def _start_teams(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, starts: list[tuple[float, float]], known: list[tuple[float, float]]) -> None:
        """A commander for each team, in team order, so that handle t is team t's and a team's first own record is its commander
        while it lives.  Each team's opening plans towards the nearest enemy's start as the others know it (`known`), and takes
        the spots nearer it than its allies' starts.  In a chicken game the chickens, the last team, get no commander, no opening
        and no script: the sim spawns and orders them (Batch.reset's `chickens`), and `starts` has the map's middle for them."""
        n = len(starts)
        ally_of = [int(a) for a in self.ally[g, :n]]
        chickens = int(self.chickens[g])
        players = [t for t in range(n) if t != chickens]
        commanders = self.opening.commanders
        types = [commanders[int(self.rng.integers(len(commanders)))] for _ in players]
        base = tuple(v for s in starts for v in s)
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.array(types), np.array(players), np.array([starts[t][0] for t in players]),
                         np.array([starts[t][1] for t in players]), commander=tuple(-1 if t == chickens else t for t in range(n)), base=base,
                         sites=bundle.sites(), ally_of=ally_of, chickens=None if chickens < 0 else (chickens, int(self.difficulty[g])))
        self.batch.economy(g, bundle.spot_list)
        ground = self.grounds[id(bundle)]
        seats = []
        for t in range(n):
            enemies = [u for u in range(n) if ally_of[u] != ally_of[t]]
            facing = min(enemies, key=lambda u: (np.hypot(known[u][0] - starts[t][0], known[u][1] - starts[t][1]), u))
            self.facing[g, t] = facing
            if t == chickens:
                seats.append(None)
                continue
            allies = tuple((starts[u], u < t) for u in range(n) if u != t and ally_of[u] == ally_of[t])
            seats.append(self.opening.seat(self.rng, t, ground, starts[t], known[facing], allies))
        self.seats[g] = tuple(seats)
        styles = self._draw_styles(g)
        if self.cpp is not None:
            seeds = tuple(int(self.rng.integers(1 << 62)) for _ in range(n)) if self.economy else (0,) * n
            rally = tuple(v for t, s in enumerate(seats) for v in (starts[t] if s is None else s.plan.rally))
            self.cpp.reset(g, ground, base, rally, economy=self.economy, aimer=True, seeds=seeds, variants=styles, ally_of=ally_of,
                           script_teams=None if chickens < 0 else players)

    def _opening_orders(self, g: int, team: int) -> np.ndarray | None:
        """The commander's next step, if it has finished this one or run out of time on it."""
        seats = self.seats[g]
        if seats is None or seats[team] is None or seats[team].done:
            return None
        seat, n = seats[team], int(self.batch.counts[g, team])
        units = self.batch.records[g, team, :n]
        if not n or units["uid"][0] != team:
            seat.done = True        # the commander is dead
            return None
        if not units["flags"][0] & rf.IDLE and self.decisions[g] < seat.deadline:
            return None
        orders = seat.advance(units, int(self.decisions[g]))
        return np.array(orders, zksim.ORDER_DTYPE) if orders else None

    def _is_policy(self, seat: int) -> bool:
        """In mirror the policy plays every player's team; in fight team 0, the scripts every other, its allies too (a simulated
        co-op); in scripts none.  An unused slot is no one's, nor are the chickens."""
        g, team = divmod(seat, self.slots)
        if team >= self.n_teams[g] or team == self.chickens[g]:
            return False
        mode = self.mode[g]
        return mode == "mirror" or (team == 0 and mode != "scripts")

    def policy_seats(self) -> list[int]:
        """Every player's seat the games have now, whoever plays it: who does can change from game to game, as can how many teams
        a slot's game has.  The chickens have none."""
        return [g * self.slots + t for g in range(self.count) for t in range(int(self.n_teams[g])) if t != self.chickens[g]]

    def policy_seats_to_act(self) -> list[int]:
        """The policy's seats that decide now: every `act_every` decisions of their game, while their alliance is in it and the
        team has a unit that keeps it in.  A team whose allies play on without it waits for their end, which its last decision's
        reward goes on to take."""
        teams = np.arange(self.slots)
        mirror = np.array([m == "mirror" for m in self.mode])
        scripts = np.array([m == "scripts" for m in self.mode])
        players = (teams < self.n_teams[:, None]) & (teams != self.chickens[:, None])
        policy = players & (mirror[:, None] | ((teams == 0) & ~scripts[:, None]))
        live = np.take_along_axis(self.alive, np.maximum(self.ally, 0), 1) & (self.batch.stats[:, :, KEEP_STAT] > 0)
        now = (self.decisions % self.act_every == 0) & ~self.waiting
        return np.flatnonzero((policy & live & now[:, None]).ravel()).tolist()

    def bundle_of(self, seat: int) -> bd.Bundle:
        return self.bundle[seat // self.slots]

    def frame(self, seat: int) -> rf.Frame:
        return self.batch.frame(seat // self.slots, seat % self.slots)

    def give_orders(self, seat: int, orders: list) -> None:
        self.pending[seat] = orders

    def place_build(self, seat: int, row: int, x: float, z: float) -> tuple[float, float] | None:
        """Where a BUILD of this type-table row, told to (x, z), goes: an extractor on the free spot nearest the point; anything else
        at the nearest place to it the game would let it be built (`Ground.fits`: its footprint's rectangle, on level ground),
        clear of our structures' footprints and our allies' and of every metal spot (as the economy script keeps clear, eco.cpp
        `_taken`).  None if the sim can't build the type (no catalog record, a geothermal
        vent's) or it fits nowhere."""
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
        g, team = divmod(seat, self.slots)
        own = f.units[(f.units["player"] == team) | (f.units["ally_team"] == self.ally[g, team])]
        rows = self.table.rows(own["def_id"])
        still = ~self.table.mobile[rows]
        spots = np.asarray(self.bundle_of(seat).spot_list, np.float32).reshape(-1, 3)
        taken = np.concatenate((footprints(own["x"][still], own["z"][still], ed.half[rows[still], 0], ed.half[rows[still], 1]),
                                footprints(spots[:, 0], spots[:, 1], ed.spot_half, ed.spot_half)))
        ground, footing = self.grounds[id(self.bundle_of(seat))], ed.footing[row]

        def fits(px: float, pz: float) -> bool:
            return ground.fits(px, pz, footing) and apart(taken, px, pz, footing.half_x, footing.half_z)

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

    def _team_units(self, g: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each team of game g's own units, as its own frame has them, and the features it sees: for a subclass's scripts, which
        know where everything is.  Made once a step."""
        kept = self._units_of.get(g)
        if kept is None or kept[0] != self.slot:
            frames = [self.batch.frame(g, t) for t in range(int(self.n_teams[g]))]
            kept = self._units_of[g] = (self.slot, [(f.units[f.units["player"] == t], f.features) for t, f in enumerate(frames)])
        return kept[1]

    def _sides_of(self, g: int, team: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A team's own units, every enemy team's (whether it sees them or not), and its allies'."""
        units = self._team_units(g)
        ally = self.ally[g]
        enemies = [u for t, (u, _) in enumerate(units) if ally[t] != ally[team]]
        allies = [u for t, (u, _) in enumerate(units) if t != team and ally[t] == ally[team]]
        own = units[team][0]
        return own, enemies[0] if len(enemies) == 1 else np.concatenate(enemies), np.concatenate(allies) if allies else own[:0]

    def script_orders(self, g: int, side: int) -> np.ndarray:
        """A subclass's own script's orders for this team (`own_scripts`); none for a team with no script, as the chickens."""
        script = self.scripts[g][side]
        if script is None or self.decisions[g] % self.script_every:
            return np.zeros(0, zksim.ORDER_DTYPE)
        own, enemies, _ = self._sides_of(g, side)
        seconds = float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        return script.orders(own, enemies, seconds)

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def _by_alliance(self, per_team: np.ndarray) -> np.ndarray:
        """games x slots: a per-team quantity summed over each alliance's teams."""
        out = np.zeros(per_team.shape, per_team.dtype)
        for t in range(self.slots):
            a = self.ally[:, t]
            used = a >= 0
            out[self._rows[used], a[used]] += per_team[used, t]
        return out

    def _others(self, v: np.ndarray, live: np.ndarray) -> np.ndarray:
        """games x slots: for each alliance, the total of the other live alliances' v."""
        out = np.zeros_like(v)
        for a in range(self.slots):
            them = live.copy()
            them[:, a] = False
            out[:, a] = np.where(them, v, 0).sum(1)
        return out

    def _balance(self, per_team: np.ndarray, constant: float) -> np.ndarray:
        """games x slots, by alliance: each live alliance's total less the mean of the other live alliances', over all of theirs and
        `constant`; 0 for the rest.  It sums to zero over the live alliances: with A of them, (A share - 1) / (A - 1) where share
        is (its own + constant / A) / (all + constant), which at two is (A - B) / (A + B + constant) as it always was."""
        v = np.where(self.alive, self._by_alliance(per_team), 0)
        others = self._others(v, self.alive)
        n = np.maximum(self.alive.sum(1) - 1, 1).astype(v.dtype)
        return np.where(self.alive, (v - others / n[:, None]) / (v.sum(1) + constant)[:, None], 0)

    def _raw(self) -> np.ndarray:
        """Each live alliance's value balance, -1 to 1 (`_balance`)."""
        return self._balance(self.batch.stats[:, :, 0], VALUE_CONSTANT)

    def _income(self) -> np.ndarray:
        """Each live alliance's metal income against the others', -1 to 1, as `_raw` is for value."""
        return self._balance(self.batch.globs[:, :, rf.G["metal_income"]], INCOME_CONSTANT)

    def script_production(self, seat: int) -> list[tuple[int, str]]:
        """This decision's production choices by a script-controlled seat: (factory uid, unit name) for each PRODUCE it gave."""
        names = catalog().units
        return [(int(o["unit"]), names[int(o["target"])]["name"]) for o in self.script_record.get(seat, np.zeros(0, zksim.ORDER_DTYPE))
                if o["kind"] == zksim.ORDER_PRODUCE]

    def _scripts_cpp(self) -> tuple[np.ndarray, np.ndarray]:
        """Every game's orders: the policy's and the opening's from here, the scripts' from libzkscripts.  Fills `script_record`
        with the army script's.  The chickens' seat, which the sim orders, stays empty as an unused slot's does; what the scripts
        are told of a chicken game is in `chickens`, `difficulty` and `chicken_states` (the queen), for `cpp.step` below."""
        n = self.slots * self.count         # by seat; an unused slot's stay empty
        flags, busy = np.zeros(n, np.int32), np.full(n, -1, np.int32)
        pre, post = [], []
        pre_counts, post_counts = np.zeros(n, np.int32), np.zeros(n, np.int32)
        policy_groups: list[list[np.ndarray]] = [[] for _ in range(self.count)]
        for g in range(self.count):
            seats = self.seats[g]
            for team in range(int(self.n_teams[g])):
                if team == self.chickens[g]:
                    continue
                seat = g * self.slots + team
                orders = self.pending.pop(seat, None)
                policy = self._is_policy(seat)
                if orders is not None:
                    part, grouped = split_groups(self._placed(seat, sim_orders(orders)))
                    pre.append(part)
                    pre_counts[seat] = len(part)
                    policy_groups[g].append(grouped)
                elif not policy:
                    flags[seat] = 1         # policy_scripts_cpp.SEAT_ARMY
                if policy:
                    flags[seat] |= (2 if self.advise else 0) | (4 if self.policy_economy[g] else 0)    # SEAT_ARMY_LABEL, SEAT_ECO_LABEL
                opening = self._opening_orders(g, team)
                if opening is not None:
                    post.append(opening)
                    post_counts[seat] = len(opening)
                if seats is not None and seats[team] is not None and not seats[team].done:
                    busy[seat] = team       # the commander, while the opening has it
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
            for side in range(int(self.n_teams[g])):
                seat = g * self.slots + side
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
        """Pluto's diversity bonus, for the units the policy chooses to make: each unit a factory could make that a team finishes
        this decision is worth `diversity` x its type's `rarity`.  Rarity is counted over the policy's teams only; every team earns
        the bonus, and `reward` takes it zero-sum over the alliances.  A team's units at its first decision of a game, the
        commander's among them, count as made already."""
        self.bonus[:] = 0.0
        if not self.diversity:
            return
        window = self.act_every if self.align else 1
        if self.slot % window:
            return
        self.made *= DIVERSITY_DECAY ** window
        for g in range(self.count):
            for side in range(int(self.n_teams[g])):
                if side == self.chickens[g]:        # nothing they have is made in a factory
                    continue
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
                if self._is_policy(g * self.slots + side):
                    np.add.at(self.made, rows, 1.0)

    def _after_step(self) -> None:
        """Rewards' parts by alliance, and who's out.  An alliance with no team that has a unit keeping it in is out: its teams'
        result is -1 / (A - 1), A the game's alliances, and their seats finish now.  The game goes on while two alliances are in,
        to the time limit, unless a lead ends it (`_far_ahead` against the total of the rest, which puts every other one out).  Then
        the S alliances still in share what the others lost, (A - S) / ((A - 1) S) each: 1 for a sole winner, 0 when all are in at
        the limit; and if the last went out together, they're those S.  At A = 2 these are the 1, -1 and 0 of a 1v1.  In a chicken
        game the chickens are in until their queen dies, whatever units they have: so the players win then, lose when they're out,
        and draw at the time limit; it has no lead rule."""
        self.decisions += 1
        self.slot += 1
        self.paying[:] = self.alive
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
        self.paid[:] = 0.0
        self.speed_paid[:] = 0.0
        live = self.alive
        still = live & self._by_alliance((self.batch.stats[:, :, KEEP_STAT] > 0).astype(np.int32)).astype(bool)
        for g in np.flatnonzero(self.chickens >= 0):
            state = self.chicken_states[int(g)] = self.batch.chicken_state(int(g))
            a = self.ally[g, self.chickens[g]]
            still[g, a] = live[g, a] and state["queen"] != 2       # 2: dead
        value = np.where(live, self._by_alliance(self.batch.stats[:, :, 0]), 0)
        late = (self.decisions >= self.early_end) & self.lead_rule
        leads = still & late[:, None] & _far_ahead(value, self._others(value, live))
        falls = live & ~still
        over = self.decisions >= self.time_limit
        for g in np.flatnonzero((falls.any(1) | leads.any(1) | over) & ~self.waiting):
            self._settle(int(g), falls[g], leads[g], bool(over[g]))
        if self.capture is not None:
            self.capture.after_step(self)

    def _settle(self, g: int, falls: np.ndarray, leads: np.ndarray, over: bool) -> None:
        """Game g's alliances that go out now, and its end if it ends: what each is paid, each team's outcome, its facts, and the
        policy's seats that finish (`done`)."""
        live = self.alive[g]
        if leads.any():
            survivors, out, ends = np.flatnonzero(leads)[:1], np.flatnonzero(live & ~leads), True
        elif not (live & ~falls).any():         # the last went out together
            survivors, out, ends = np.flatnonzero(falls), np.zeros(0, np.int64), True
        else:
            survivors, out = np.flatnonzero(live & ~falls), np.flatnonzero(falls)
            ends = len(survivors) == 1 or over
        a_count = int(self.n_allies[g])
        lost = -1.0 / (a_count - 1)
        decided = []
        for a in out:
            self.paid[g, a] = lost
            self.speed_paid[g, a] = self._time_left(g) * lost
            self.speed_pot[g] -= self.speed_paid[g, a]
            self.alive[g, a] = False
            decided.append((a, lost))
        if ends:
            share = (a_count - len(survivors)) / ((a_count - 1) * len(survivors))
            for a in survivors:
                self.paid[g, a] = share
                self.speed_paid[g, a] = self.speed_pot[g] / len(survivors)
                decided.append((a, share))
        elif len(out):
            # The rest's balances are over the alliances still in from now on.
            self.last_raw[g] = self._raw()[g]
            self.last_income[g] = self._income()[g]
        result_of = dict(decided)
        teams = [t for t in range(int(self.n_teams[g])) if int(self.ally[g, t]) in result_of]
        for t in teams:
            self.outcome[g, t] = result_of[int(self.ally[g, t])]
        for t in teams:
            r = float(self.outcome[g, t])
            seat = g * self.slots + t
            won, beaten = r >= 1.0, r < 0.0
            facts = self._game_facts(g, t, won, beaten)
            policy = self._is_policy(seat)
            if policy:
                self.done.append((seat, self.mode[g], 1.0 if won else (0.0 if beaten else 0.5)))
                self.ended[seat] = facts
            self.outcomes.append({**facts, "kind": "policy" if policy else self.script_kinds[g][t], "made": self._made(g, t), "game": g})
        if ends:
            self.result[g] = self.outcome[g, 0]
            self._replace.append(g)

    def _game_facts(self, g: int, team: int, won: bool, lost: bool) -> dict:
        """A game as it ended for one team, for the record of a test: `enemy_value` is every other alliance's, `enemy_factory` the
        one its opening planned against's.  "eliminated" if its alliance or every other was, "lead" if it won or lost otherwise,
        else "limit"; in a chicken game "eliminated" if the players were, "queen" if the queen died, else "limit".  The chickens'
        team has no factory and "chickens" for a variant, and a chicken game says its difficulty."""
        stats = self.batch.stats[g]
        seats = self.seats[g]
        names = catalog().units
        slugs = {id(b): f.name for _, b, f in self.maps}
        n, ally = int(self.n_teams[g]), self.ally[g]
        chickens = int(self.chickens[g])
        mine = [t for t in range(n) if ally[t] == ally[team]]
        enemies = [t for t in range(n) if ally[t] != ally[team]]
        gone = lambda teams: all(stats[t, KEEP_STAT] == 0 for t in teams)  # noqa: E731
        enemy = int(self.facing[g, team]) if n > 2 else 1 - team
        factory = lambda t: names[seats[t].factory_type]["name"] if seats and seats[t] is not None else ""  # noqa: E731
        if chickens < 0:
            how = "eliminated" if gone(mine) or gone(enemies) else ("lead" if won or lost else "limit")
        else:
            queen_dead = self.chicken_states.get(g, {}).get("queen") == 2
            how = "eliminated" if gone([t for t in range(n) if t != chickens]) else "queen" if queen_dead else "limit"
        return {
            "map": slugs.get(id(self.bundle[g]), "?"), "seconds": float(self.decisions[g]) * self.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND,
            "result": "won" if won else ("lost" if lost else "drew"),
            "how": how,
            "own_value": float(stats[team, 0]), "enemy_value": float(stats[enemies, 0].sum()),
            "own_factory": factory(team), "enemy_factory": factory(enemy),
            "variant": "chickens" if team == chickens else self.styles[g][team].name if self.styles[g] else "standard",
            "lead_rule": bool(self.lead_rule[g]),       # whether a lead could have ended it
            "game_kind": self.kind[g], "team": team, "ally": int(ally[team]), "teams": n, "alliances": int(self.n_allies[g]),
            "outcome": float(self.outcome[g, team]),
            **({"difficulty": zksim.CHICKEN_DIFFICULTIES[int(self.difficulty[g])]} if chickens >= 0 else {}),
        }

    def _time_left(self, g: int) -> float:
        """The speed term: going out now costs speed_weight x this share of the time limit left x the result an out alliance gets,
        -1 / (A - 1), on top of that result, and the survivors share what was paid at the end, as the result.  Zero-sum as it is."""
        return 1.0 - float(self.decisions[g]) / self.time_limit

    def _bonuses(self, g: int) -> list[float]:
        """Game g's diversity bonus this decision, by alliance: its teams' together."""
        out = [0.0] * int(self.n_allies[g])
        for t in range(int(self.n_teams[g])):
            out[int(self.ally[g, t])] += self.bonus[g, t]
        return out

    def reward(self, seat: int, beta: float) -> float:
        """Zero-sum over the alliances, and every team of one gets its alliance's: the change in its value balance (`_raw`) and its
        diversity bonus less the mean of the other live alliances', fading by beta into the result paid at this step (`_after_step`),
        and the speed term (`_time_left`), which doesn't fade.  In a game of two alliances, alliance 1's is alliance 0's turned round,
        as side 1's was side 0's in a 1v1."""
        g, team = divmod(seat, self.slots)
        a = int(self.ally[g, team])
        if a < 0:
            return 0.0
        bonus = self._bonuses(g)
        if self.n_allies[g] == 2:
            sign, a = (1.0 if a == 0 else -1.0), 0
            shaped = self.delta[g, 0] + bonus[0] - bonus[1] + self.income_weight * self.income_delta[g, 0]
        else:
            if not self.paying[g, a]:
                return 0.0
            them = [b for b in np.flatnonzero(self.paying[g]) if b != a]
            div = bonus[a] - sum(bonus[b] for b in them) / max(len(them), 1)
            sign = 1.0
            shaped = self.delta[g, a] + div + self.income_weight * self.income_delta[g, a]
        result = self.paid[g, a]
        out = (1.0 - beta) * shaped + beta * result
        if self.speed_weight and self.speed_paid[g, a]:
            out += self.speed_weight * self.speed_paid[g, a]
        return float(sign * out)

    def finished_seats(self) -> list[tuple[int, str, float]]:
        """Seats whose game just ended, or whose alliance went out of it.  The call also starts the ended games afresh, so read the
        reward first; with `align`, they start at the next slot that act_every divides, and wait ended till then."""
        out = self.done
        if self._replace:
            self.waiting[self._replace] = True
        if not self.align or self.slot % self.act_every == 0:
            for g in np.flatnonzero(self.waiting):
                g = int(g)
                self._new_teams(g)
                self._reset(g)
                self.batch.refresh(g)
                self.last_raw[g] = self._raw()[g]
                self.last_income[g] = self._income()[g]
            self.waiting[:] = False
        self._replace, self.done = [], []
        return out
