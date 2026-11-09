"""Frames for the frame encoder (frame_model.py, trained by frame_vae.py): what a seat saw, from three sources, on all the training maps.

Script games are real games: both sides play the opening, a script for factories and builders, and the fight script, with every factory there is, on the map's own terms.
Scenarios are built at random but could have happened, so they reach what script games don't: other unit types, big armies, a fight in someone's base, a base that's been fought over.
Each is of a flavour (`FLAVOURS`): as any game, or what the army script seldom or never makes: air battles, air raids on a base with AA, fleets,
heavy units, swarms of cheap ones.
Army games are the C++ army and economy scripts against each other (policy_sim_env, with variants): what the policy is cloned from and plays.

Both are on maps from a pool, real and made up (tools/map_gen.py), with and without water:

    python tools/frame_data.py collect --maps /mnt/d/zk-learned/maps --out /mnt/d/zk-learned/frames/NAME --script-frames 1000000 --scenario-frames 1000000 --army-frames 1000000 --games 256 --workers 8

Each shard is an uncompressed npz of concatenated tokens and offsets.  Links between frames (a seat's frame a moment later) stay within a shard.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_sim_env as se  # noqa: E402
from frame_world import BaseOrders, BaseScript, EXTRACTOR_REACH as SPOT_REACH, WATER_MAP, FrameFight, FrameGames, Ground, Roster, extractors, fight_data, held_out_maps, pool_weights  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

SOURCES = ("script", "scenario", "army")
PROBES = ("own_value", "enemy_value", "own_fighters", "enemy_fighters", "enemy_known", "own_x", "own_z", "enemy_x", "enemy_z", "commander_hp", "metal", "seconds",
          "own_air", "enemy_air", "own_ships", "enemy_ships", "sea_share", "wreck_metal")
# Token groups as stored: name, feature size.  Floats go to 16 bits, as these are all small numbers and it halves the disk.
GROUPS = (("unit", fz.UNIT_DIM), ("reg", fz.REGION_DIM), ("site", fz.SITE_DIM), ("feat", fz.FEATURE_DIM))
LONGEST_SCENARIO_START_SECONDS = 1200.0     # what a scenario's clock can claim: a game's time limit


@dataclass(frozen=True)
class Capture:
    """When a source takes frames, in decisions (half seconds)."""
    every: int
    later: int = 4          # each seat again this long after, for the temporal view
    warm: int = 0           # none before this: a scenario's first moments have no orders or velocities

    def __post_init__(self) -> None:
        if not 0 < self.later < self.every:
            raise ValueError(f"a later frame {self.later} decisions on, every {self.every}: the two would coincide")


@dataclass(frozen=True)
class Flavour:
    """A kind of scenario: what each side's army is drawn from (`army_pools`, or "mixed" for as in any game), and whether the first
    raids the second's base.  The sides are swapped at random.  value, structures: each side's, in place of ScenarioRules.side_value and
    structure_share."""
    name: str
    armies: tuple[str, str] = ("mixed", "mixed")
    raid: bool = False
    value: tuple[float, float] | None = None
    needs_water: bool = False
    structures: tuple[tuple[float, float], tuple[float, float]] | None = None


FLAVOURS = ((Flavour("mixed"), 0.4),
            (Flavour("air battle", ("air", "air_aa"), value=(3000.0, 40000.0), structures=((0.0, 0.1), (0.0, 0.2))), 0.2),
            (Flavour("air raid", ("air", "aa_base"), raid=True, value=(3000.0, 40000.0), structures=((0.0, 0.1), (0.3, 0.6))), 0.1),
            (Flavour("fleets", ("ships", "ships"), value=(1000.0, 30000.0), needs_water=True), 0.1),
            (Flavour("heavies", ("heavy", "mixed"), value=(4000.0, 40000.0)), 0.1),
            (Flavour("swarms", ("swarm", "swarm"), value=(1500.0, 15000.0)), 0.1))
HEAVY_FROM = 1200.0     # metal: a heavy unit
SWARM_UP_TO = 120.0     # metal: a swarm's


@dataclass(frozen=True)
class ScenarioRules:
    """What a random scenario can hold.  Costs are metal."""
    side_value: tuple[float, float] = (150.0, 40000.0)  # log-uniform
    most_units: int = 240           # a side
    extractors: tuple[int, int] = (0, 8)
    structure_share: tuple[float, float] = (0.0, 0.5)   # uniform, of a side's value
    most_structure_cost: float = 6500.0     # up to a Singularity and the big guns, and no superweapons
    has_factory: float = 0.8
    second_factory_from: float = 3000.0     # of a side's value: where another's likely, as a side that big has been at it a while
    third_factory_from: float = 12000.0
    wrecks: float = 0.6             # of sides that have lost things
    most_wrecks: int = 40
    wreck_share: float = 0.5        # of a side's value at most, that has been lost
    heap: float = 0.25              # of wrecks that have been shot at, which leaves less
    scouted: tuple[float, float] = (0.0, 1.0)   # uniform: how much of the enemy's base a side has seen before
    has_commander: float = 0.85
    keeps_to_one_factory: float = 0.7   # or draws from every factory's roster
    clusters: tuple[int, int] = (1, 3)
    full_health: float = 0.6        # of units; the rest uniformly hurt
    unfinished: float = 0.1         # of structures; uniformly part built
    decisions: int = 48             # a scenario plays this long, then a new one starts
    capture: Capture = Capture(every=12, later=4, warm=4)
    income_per_second: float = 15.0  # what a side's value says about its game's clock
    flavours: tuple[tuple[Flavour, float], ...] = FLAVOURS
    aa_turret_share: tuple[float, float] = (0.3, 0.7)    # of an AA base's structure budget


# Frames as a model reads them -------------------------------------------------------------------------------------------------------------------------


def with_site_status(frame: rf.Frame, table, is_extractor: np.ndarray) -> rf.Frame:
    """Who holds each spot, as far as this seat knows.  The live bot's Brain says this; the sim's records leave every spot free."""
    if not len(frame.sites):
        return frame
    u = frame.units
    known = u[(u["knowledge"] != rf.KNOW_HIDDEN) & is_extractor[table.rows(u["def_id"])]]
    sites = frame.sites.copy()
    sites["status"] = rf.SITE_FREE
    if len(known):
        d = np.hypot(sites["x"][:, None] - known["x"][None], sites["z"][:, None] - known["z"][None])
        nearest = d.argmin(axis=1)
        held = d[np.arange(len(sites)), nearest] <= SPOT_REACH
        who = known[nearest]
        ally = int(frame.glob[rf.G["ally_team"]])
        status = np.where(who["ally_team"] != ally, rf.SITE_ENEMY, np.where(who["flags"] & rf.BEING_BUILT, rf.SITE_BUILDING, rf.SITE_OURS))
        sites["status"] = np.where(held, status, rf.SITE_FREE)
    return replace(frame, sites=sites)


def probe_targets(frame: rf.Frame, bundle: bd.Bundle, table) -> np.ndarray:
    """In the order of PROBES: what a frame's latent might tell, for checking it.  The enemy's value, fighters and place include what this seat can't see."""
    u = frame.units
    g = frame.glob
    rows = table.rows(u["def_id"])
    own = u["player"] == int(g[rf.G["player"]])
    enemy = u["ally_team"] != int(g[rf.G["ally_team"]])
    fighting = table.combat[rows] & table.mobile[rows] & ((u["flags"] & rf.BEING_BUILT) == 0)
    known = enemy & (u["knowledge"] != rf.KNOW_HIDDEN)
    air = table.air[rows]
    ships = table.move_class[rows] == bd.MOVE_CLASSES.index("ship")
    costs = table.cost[rows]

    def centre(mask: np.ndarray) -> tuple[float, float]:
        if not mask.any():
            return 0.0, 0.0
        return float(u["x"][mask].mean() / bundle.map_w * 2.0 - 1.0), float(u["z"][mask].mean() / bundle.map_h * 2.0 - 1.0)

    return np.array((np.log1p(table.cost[rows][own].sum()), np.log1p(table.cost[rows][enemy].sum()), np.log1p((own & fighting).sum()), np.log1p((enemy & fighting).sum()),
                     np.log1p(known.sum()), *centre(own), *centre(enemy), g[rf.G["commander_hp"]], np.log1p(max(0.0, float(g[rf.G["metal"]]))),
                     g[rf.G["frame"]] / zksim.FRAMES_PER_SECOND / 1000.0,
                     np.log1p(costs[own & air].sum()), np.log1p(costs[enemy & air].sum()), np.log1p(costs[own & ships].sum()), np.log1p(costs[enemy & ships].sum()),
                     float(bundle.sea.mean()) if bundle.regions else 0.0, np.log1p(frame.features["metal"].sum())), np.float32)


@dataclass(frozen=True)
class Meta:
    map: int
    source: int
    game: int
    seat: int
    decision: int


class ShardWriter:
    """Frames in, shards out.  `add` gives each frame its index in the current shard, so that `link` can tie a seat's frame to a later one."""

    def __init__(self, out: pathlib.Path, prefix: str, frames_per_shard: int = 2000):
        self.out, self.prefix, self.size = out, prefix, frames_per_shard
        out.mkdir(parents=True, exist_ok=True)
        self.shards = self.total = 0
        self._clear()

    def _clear(self) -> None:
        self.items: list[tuple[fz.Tensors, Meta, np.ndarray]] = []
        self.later: list[int] = []
        self.generation = getattr(self, "generation", 0) + 1

    def add(self, tensors: fz.Tensors, meta: Meta, probes: np.ndarray) -> tuple[int, int]:
        """Returns the shard's generation and the frame's index in it: a link is good only within a generation."""
        self.items.append((tensors, meta, probes))
        self.later.append(-1)
        self.total += 1
        return self.generation, len(self.items) - 1

    def link(self, earlier: tuple[int, int], later: tuple[int, int]) -> None:
        if earlier[0] == later[0] == self.generation:
            self.later[earlier[1]] = later[1]

    def flush_if_full(self) -> None:
        if len(self.items) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.items:
            return
        columns: dict[str, np.ndarray] = {}
        tensors = [t for t, _, _ in self.items]
        for name, _ in GROUPS:
            x = [getattr(t, f"{name}_x") for t in tensors]
            columns[f"{name}_x"] = np.concatenate(x).astype(np.float16)
            columns[f"{name}_at"] = np.cumsum([0] + [len(a) for a in x], dtype=np.int64)
        columns["unit_name"] = np.concatenate([t.unit_name for t in tensors]).astype(np.int16)
        columns["unit_ally"] = np.concatenate([t.unit_ally for t in tensors]).astype(np.int8)
        columns["unit_player"] = np.concatenate([t.unit_player for t in tensors]).astype(np.int8)
        columns["unit_kind"] = np.concatenate([t.tok_kind[:len(t.unit_x)] for t in tensors]).astype(np.int8)
        columns["glob_x"] = np.stack([t.glob_x for t in tensors]).astype(np.float16)
        columns["probes"] = np.stack([p for _, _, p in self.items])
        for name in ("map", "source", "game", "seat", "decision"):
            columns[name] = np.array([getattr(m, name) for _, m, _ in self.items], np.int32)
        columns["later"] = np.array(self.later, np.int32)
        np.savez(self.out / f"{self.prefix}-{self.shards:05d}.npz", **columns)
        self.shards += 1
        self._clear()


# Random feasible scenarios ------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Side:
    """Rows of the type table, except the commander, which is the catalog's."""
    commander: int      # catalog index, or -1
    commander_at: tuple[float, float]
    rows: np.ndarray
    x: np.ndarray
    z: np.ndarray
    hp: np.ndarray
    build: np.ndarray
    wrecks: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), np.float32))    # what it has lost: as MapFacts.features


def _spend(rng: np.random.Generator, choices: np.ndarray, cost: np.ndarray, budget: float, most: int) -> np.ndarray:
    """Random choices, until the budget's gone.  Whatever would pass it is skipped."""
    out: list[int] = []
    left = budget
    affordable = choices[cost[choices] <= left]
    while len(affordable) and len(out) < most:
        r = int(rng.choice(affordable))
        out.append(r)
        left -= float(cost[r])
        affordable = affordable[cost[affordable] <= left]
    return np.array(out, np.int64)


def _on_passable(bundle: bd.Bundle, x: np.ndarray, z: np.ndarray, classes: np.ndarray) -> np.ndarray:
    rows, cols = bundle.passable.shape
    i = np.minimum((z / bd.PASS_CELL).astype(np.int32), rows - 1)
    j = np.minimum((x / bd.PASS_CELL).astype(np.int32), cols - 1)
    return (bundle.passable[i, j] >> classes) & 1 == 1


def _place_units(rng: np.random.Generator, ground: Ground, table, rows: np.ndarray, cx: float, cz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round a point, or for a class that can't be there, round the nearest place it can: ships off the shore, and walkers on it.  What finds nowhere is left out.  Returns rows, x, z."""
    bundle = ground.bundle
    x, z = np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)
    kept = np.ones(len(rows), bool)
    for c in np.unique(table.move_class[rows]):
        at = np.flatnonzero(table.move_class[rows] == c)
        if c == tables.NO_CLASS:
            x[at] = np.clip(cx + rng.normal(0.0, se.SPREAD, len(at)), 8.0, bundle.map_w - 8.0)
            z[at] = np.clip(cz + rng.normal(0.0, se.SPREAD, len(at)), 8.0, bundle.map_h - 8.0)
            continue
        centre = (cx, cz) if ground.can(int(c), np.array([cx]), np.array([cz]))[0] else ground.near(int(c), cx, cz)
        if centre is None:
            kept[at] = False
            continue
        classes = np.full(len(at), int(c), np.int32)
        x[at], z[at] = se.place(bundle, rng, classes, centre[0], centre[1])
        kept[at] = _on_passable(bundle, x[at], z[at], classes)
    return rows[kept], x[kept], z[kept]


def _structure_place(rng: np.random.Generator, ground: Ground, roster: Roster, row: int, own: tuple[float, float], enemy: tuple[float, float]) -> tuple[float, float]:
    """Factories and the like go at home.  A turret goes there too, or out by one of its side's spots, or on the way to the enemy."""
    if not roster.is_turret[row]:
        return own
    pick = rng.random()
    spots = ground.bundle.spot_list
    if pick < 0.25 and len(spots):
        near = spots[np.argsort(np.hypot(spots[:, 0] - own[0], spots[:, 1] - own[1]))[:8]]
        s = near[int(rng.integers(len(near)))]
        return float(s[0]), float(s[1])
    if pick < 0.5:
        t = float(rng.uniform(0.15, 0.6))
        return own[0] + t * (enemy[0] - own[0]), own[1] + t * (enemy[1] - own[1])
    return own


def _place_structures(rng: np.random.Generator, ground: Ground, roster: Roster, rows: np.ndarray, own: tuple[float, float], enemy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extractors on the nearest free spots, the rest where their data allows, none on top of another.  What finds nowhere is left out."""
    kept, xs, zs = [], [], []
    spots = ground.bundle.spot_list[np.argsort(np.hypot(ground.bundle.spot_list[:, 0] - own[0], ground.bundle.spot_list[:, 1] - own[1]))]
    taken = np.column_stack((spots[:, 0], spots[:, 1], np.full(len(spots), roster.footing[roster.extractor].radius))).astype(np.float32)
    used = np.zeros(len(spots), bool)
    for r in rows:
        r = int(r)
        found = None
        if roster.is_extractor[r]:
            s = np.flatnonzero(~used)
            if len(s):
                used[s[0]] = True
                found = (float(spots[s[0], 0]), float(spots[s[0], 1]))
        else:
            at = _structure_place(rng, ground, roster, r, own, enemy)
            found = ground.site(rng, roster.footing[r], at[0], at[1], taken, 2.0 * se.SPREAD)
        if found is not None:
            kept.append(r)
            xs.append(found[0])
            zs.append(found[1])
            if not roster.is_extractor[r]:
                taken = np.concatenate((taken, np.array([[found[0], found[1], roster.footing[r].radius]], np.float32)))
    return np.array(kept, np.int64), np.array(xs, np.float32), np.array(zs, np.float32)


def _army_place(rng: np.random.Generator, bundle: bd.Bundle, own: tuple[float, float], enemy: tuple[float, float]) -> tuple[float, float]:
    """A cluster's centre: at home, at the enemy's, somewhere between, or anywhere."""
    pick = rng.integers(4)
    if pick == 0:
        return own
    if pick == 1:
        return enemy
    if pick == 2:
        t = float(rng.random())
        return own[0] + t * (enemy[0] - own[0]), own[1] + t * (enemy[1] - own[1])
    return float(rng.uniform(0.0, bundle.map_w)), float(rng.uniform(0.0, bundle.map_h))


def _draw_wrecks(rng: np.random.Generator, rules: ScenarioRules, roster: Roster, ground: Ground, pool: np.ndarray, own: tuple[float, float], enemy: tuple[float, float], value: float) -> np.ndarray:
    """What a side has lost so far, where it fell: bits of its own kind of army, mostly on the way to the fight.  A wreck's metal and health are the catalog's."""
    cat, table = roster.cat, roster.table
    if rng.random() >= rules.wrecks or not len(pool):
        return np.zeros((0, 6), np.float32)
    lost = _spend(rng, pool, table.cost, value * float(rng.uniform(0.0, rules.wreck_share)), rules.most_wrecks)
    out = []
    t = float(rng.uniform(0.2, 0.8))
    middle = own[0] + t * (enemy[0] - own[0]), own[1] + t * (enemy[1] - own[1])
    for r in lost:
        u = cat.units[cat.index[table.names[int(r)]]]
        metal, hp, feature = (u["heap_metal"], u["heap_hp"], u["heap_feature"]) if rng.random() < rules.heap else (u["wreck_metal"], u["wreck_hp"], u["wreck_feature"])
        if metal <= 0.0:
            continue
        cx, cz = middle if rng.random() < 0.7 else _army_place(rng, ground.bundle, own, enemy)
        x, z = float(cx + rng.normal(0.0, 2.0 * se.SPREAD)), float(cz + rng.normal(0.0, 2.0 * se.SPREAD))
        c = int(table.move_class[int(r)])
        # A plane can come down anywhere.  Others fell where they could be.
        if c != tables.NO_CLASS and not ground.can(c, np.array([x]), np.array([z]))[0]:
            place = ground.near(c, x, z)
            if place is None:
                continue
            x, z = place
        if 0.0 < x < ground.bundle.map_w and 0.0 < z < ground.bundle.map_h:
            out.append((x, z, metal, hp, u["footprint_radius"], feature))
    return np.array(out, np.float32).reshape(-1, 6)


def army_pools(roster: Roster) -> dict[str, np.ndarray]:
    """What a flavour's army is drawn from, rows of the type table: fighters that fly, ground AA, both, ships, heavy fighters, cheap
    ones; and AA turrets for an AA base."""
    from zkbot import catalog as bot_catalog
    from zkbot.policy import groups
    table = roster.table
    names, role_of = groups.roles()
    aa = role_of[:len(role_of)] == names.index("aa")
    mobile = roster.mobile
    fighter = mobile[table.combat[mobile] & ~table.builder[mobile]]
    air, ships = fighter[table.air[fighter]], fighter[table.move_class[fighter] == bd.MOVE_CLASSES.index("ship")]
    ground_aa = fighter[aa[fighter] & ~table.air[fighter] & (table.move_class[fighter] != bd.MOVE_CLASSES.index("ship"))]
    walks = fighter[~table.air[fighter] & (table.move_class[fighter] != bd.MOVE_CLASSES.index("ship"))]
    return {"air": air, "air_aa": np.union1d(air, ground_aa), "aa_base": np.union1d(walks, ground_aa), "ships": ships,
            "heavy": walks[table.cost[walks] >= HEAVY_FROM], "swarm": walks[table.cost[walks] <= SWARM_UP_TO],
            "aa_turrets": np.array([r for r in roster.structures if roster.is_turret[r] and _shoots_only_air(bot_catalog.stats(table.names[r]))], np.int64)}


def _shoots_only_air(stats: dict | None) -> bool:
    """As zkbot.catalog.stat_role tells a mobile AA unit, for a turret."""
    return stats is not None and stats["dps_ground"] <= 0.5 < stats["dps_air"]


def draw_side(rng: np.random.Generator, rules: ScenarioRules, roster: Roster, ground: Ground, own: tuple[float, float], enemy: tuple[float, float],
              army: str = "mixed", pools: dict[str, np.ndarray] | None = None, raid: bool = False, values: tuple[float, float] | None = None,
              structures: tuple[float, float] | None = None) -> Side:
    """army: what its army is drawn from (`army_pools`), "mixed" as in any game.  raid: its army is at the enemy's base."""
    table = roster.table
    bundle = ground.bundle
    value = float(np.exp(rng.uniform(*np.log(values or rules.side_value))))
    for_structures = value * float(rng.uniform(*(structures or rules.structure_share)))
    factories, chances = roster.factories_for(ground)
    cheap = np.array([r for r in roster.structures if table.cost[r] <= rules.most_structure_cost and not table.factory[r] and ground.has_room(roster.footing[int(r)].low, roster.footing[int(r)].high)], np.int64)
    # A base starts with a factory and something to pay for it, so that most scenarios look like one.  A side with a lot has more of them.
    count = 0
    if factories and rng.random() < rules.has_factory:
        count = 1 + int(value >= rules.second_factory_from and rng.random() < 0.7) + int(value >= rules.third_factory_from and rng.random() < 0.7)
    own_factories = np.array([], np.int64)
    if count:
        own_factories = rng.choice(np.array(factories), count, p=chances)
        own_factories = own_factories[np.cumsum(table.cost[own_factories]) <= for_structures]
    extractor_count = int(rng.integers(rules.extractors[0], rules.extractors[1] + 1))
    wanted = [own_factories, np.full(extractor_count, roster.extractor, np.int64)]
    if army == "aa_base" and pools is not None:
        turrets = pools["aa_turrets"]
        turrets = turrets[np.isin(turrets, cheap)]
        wanted.append(_spend(rng, turrets, table.cost, for_structures * float(rng.uniform(*rules.aa_turret_share)), rules.most_units // 8))
    wanted.append(_spend(rng, cheap, table.cost, for_structures - sum(float(table.cost[w].sum()) for w in wanted), rules.most_units // 4))
    built, bx, bz = _place_structures(rng, ground, roster, np.concatenate(wanted), own, enemy)
    parts = [(built, bx, bz)]
    pool = np.unique(np.concatenate([roster.by_factory[f] for f in factories])) if factories else roster.mobile
    if factories and rng.random() < rules.keeps_to_one_factory:
        one = int(own_factories[int(rng.integers(len(own_factories)))]) if len(own_factories) else int(rng.choice(np.array(factories), p=chances))
        pool = roster.by_factory[one]
    if army != "mixed" and pools is not None and len(pools[army]):
        pool = pools[army]
    clusters = int(rng.integers(rules.clusters[0], rules.clusters[1] + 1))
    room = rules.most_units - len(built)
    for _ in range(clusters):
        if raid:
            t = float(rng.uniform(0.75, 1.0))
            cx, cz = own[0] + t * (enemy[0] - own[0]), own[1] + t * (enemy[1] - own[1])
        else:
            cx, cz = _army_place(rng, bundle, own, enemy)
        picked = _spend(rng, pool, table.cost, (value - for_structures) / clusters, room // clusters)
        if len(picked):
            parts.append(_place_units(rng, ground, table, picked, cx, cz))
    rows = np.concatenate([p[0] for p in parts]).astype(np.int64)
    x = np.concatenate([p[1] for p in parts]).astype(np.float32)
    z = np.concatenate([p[2] for p in parts]).astype(np.float32)
    hurt = rng.random(len(rows)) >= rules.full_health
    hp = np.where(hurt, rng.uniform(0.1, 1.0, len(rows)), 1.0).astype(np.float32)
    mobile = table.mobile[rows]
    build = np.where(~mobile & (rng.random(len(rows)) < rules.unfinished), rng.uniform(0.1, 1.0, len(rows)), 1.0).astype(np.float32)
    commander = roster.commanders[int(rng.integers(len(roster.commanders)))] if rng.random() < rules.has_commander or not len(rows) else -1
    return Side(commander, own, rows, x, z, hp, build, _draw_wrecks(rng, rules, roster, ground, pool, own, enemy, value))


class ScenarioEnv(BaseOrders, se.SimEnv):
    """Games that start in the middle: both sides drawn by `draw_side`, playing from there, and over after a short while."""
    own_scripts = True      # FrameFight and BaseScript, not the C++ army and economy scripts

    def __init__(self, games: int, seed: int, rules: ScenarioRules = ScenarioRules(), threads: int = 16, maps: list | None = None, weights: np.ndarray | None = None):
        self.rules = rules
        self.clock = np.zeros(games, np.float32)    # the frames a scenario's clock claims have passed
        self.scouted: list[tuple[np.ndarray, np.ndarray] | None] = [None] * games
        super().__init__(games, seed, ["scripts"], units=1, threads=threads, maps=maps, weights=weights)
        self.roster = Roster(self.table, se.catalog())
        self.pools = army_pools(self.roster)
        self.script_data = fight_data(self.table, se.catalog())
        self.grounds = {id(b): Ground(b, f) for _, b, f in self.maps}
        self.bases: list = [None] * games
        self.flavour: list[str] = [""] * games
        self.time_limit = rules.decisions
        self.early_end = rules.decisions + 1

    def _reset(self, g: int) -> None:
        world_map, bundle, facts = self._draw_map()
        ground = self.grounds[id(bundle)]
        boxes = facts.layouts[int(self.rng.integers(len(facts.layouts)))]
        one, other = self.rng.choice(len(boxes), 2, replace=False)
        x0, z0 = boxes[one].pick(self.rng, bundle)
        x1, z1 = boxes[other].pick(self.rng, bundle)
        flavours, weights = zip(*self.rules.flavours)
        flavour = flavours[int(self.rng.choice(len(flavours), p=np.array(weights) / sum(weights)))]
        if flavour.needs_water and facts.water < WATER_MAP:
            flavour = Flavour("mixed")
        self.flavour[g] = flavour.name
        armies, raid = flavour.armies, (flavour.raid, False)
        structures = flavour.structures or (None, None)
        if self.rng.random() < 0.5:
            armies, raid, structures = armies[::-1], raid[::-1], structures[::-1]
        sides = (draw_side(self.rng, self.rules, self.roster, ground, (x0, z0), (x1, z1), armies[0], self.pools, raid[0], flavour.value, structures[0]),
                 draw_side(self.rng, self.rules, self.roster, ground, (x1, z1), (x0, z0), armies[1], self.pools, raid[1], flavour.value, structures[1]))
        self.mode[g] = "scripts"
        count = self._load(g, world_map, bundle, sides, (x0, z0, x1, z1))
        self.batch.add_features(g, np.concatenate((facts.features, sides[0].wrecks, sides[1].wrecks)))
        self.scripts[g] = (FrameFight(self.script_data, (x0, z0), (x1, z1), ground=ground), FrameFight(self.script_data, (x1, z1), (x0, z0), ground=ground))
        self.bundle[g] = bundle
        self.starts[g] = (x0, z0, x1, z1)
        self.decisions[g] = 0
        value = max(float(self.table.cost[s.rows].sum()) for s in sides)
        seconds = min(value / self.rules.income_per_second, LONGEST_SCENARIO_START_SECONDS)
        self.clock[g] = seconds * zksim.FRAMES_PER_SECOND
        self.bases[g] = (BaseScript(self.roster, ground, (x0, z0), (x1, z1), self.rng, seconds), BaseScript(self.roster, ground, (x1, z1), (x0, z0), self.rng, seconds))
        share = self.rng.uniform(*self.rules.scouted, 2)
        self.scouted[g] = (self.rng.random(count) < share[0], self.rng.random(count) < share[1])

    def _load(self, g: int, world_map: zksim.Map, bundle: bd.Bundle, sides: tuple[Side, Side], base: tuple[float, float, float, float]) -> int:
        """Commanders first, as handles 0 and 1 where both live, so that a side's first own record is its commander as in a game.  Returns how many units."""
        types, side_of, xs, zs, hps, builds = [], [], [], [], [], []
        handles = [-1, -1]
        for s, side in enumerate(sides):
            if side.commander >= 0:
                handles[s] = len(types)
                types.append(side.commander)
                side_of.append(s)
                xs.append(side.commander_at[0])
                zs.append(side.commander_at[1])
                hps.append(1.0)
                builds.append(1.0)
        index = [self.roster.catalog_index(s.rows) for s in sides]
        type_ = np.concatenate((np.array(types, np.int32), *index))
        side_arr = np.concatenate((np.array(side_of, np.int32), np.zeros(len(index[0]), np.int32), np.ones(len(index[1]), np.int32)))
        x = np.concatenate((np.array(xs, np.float32), sides[0].x, sides[1].x))
        z = np.concatenate((np.array(zs, np.float32), sides[0].z, sides[1].z))
        hp = np.concatenate((np.array(hps, np.float32), sides[0].hp, sides[1].hp))
        build = np.concatenate((np.array(builds, np.float32), sides[0].build, sides[1].build))
        heading = self.rng.uniform(0.0, 2.0 * np.pi, len(type_)).astype(np.float32)
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), type_, side_arr, x, z, heading=heading, hp=hp, build=build,
                         commander=(handles[0], handles[1]), base=base, sites=bundle.sites())
        self.batch.economy(g, bundle.spot_list, metal=float(self.rng.uniform(0.0, 500.0)), energy=float(self.rng.uniform(0.0, 500.0)))
        return len(type_)

    def frame(self, seat: int) -> rf.Frame:
        """A scenario starts with nothing seen, where a side in a game that far on would remember some of the enemy's buildings."""
        frame = super().frame(seat)
        g = seat // 2
        glob = frame.glob.copy()
        glob[rf.G["frame"]] += self.clock[g]
        u = frame.units
        hidden = u["knowledge"] == rf.KNOW_HIDDEN
        if hidden.any():
            seen_before = self.scouted[g][seat % 2]
            uids = u["uid"]
            remembered = hidden & ~self.table.mobile[self.table.rows(u["def_id"])] & (uids < len(seen_before)) & seen_before[np.minimum(uids, len(seen_before) - 1)]
            if remembered.any():
                u = u.copy()
                u["knowledge"][remembered] = rf.KNOW_MEMORY
                u["flags"][remembered] |= rf.TYPE_KNOWN
                u["vx"][remembered] = u["vz"][remembered] = 0.0
        return replace(frame, units=u, glob=glob)


# Collecting ------------------------------------------------------------------------------------------------------------------------------------------------


class Collector:
    """Plays an environment and takes frames of both seats of each game as `capture` says, each game at its own moments."""

    def __init__(self, env: se.SimEnv, capture: Capture, writer: ShardWriter, source: str, seed: int, first_game: int = 0):
        self.env, self.capture, self.writer = env, capture, writer
        self.source = SOURCES.index(source)
        self.is_extractor = extractors(env.table)
        self.map_of = {id(b): i for i, (_, b, _) in enumerate(env.maps)}
        self.seen = np.zeros(len(PROBES), np.int64)     # frames in which each probe wasn't nothing
        self.taken = 0
        rng = np.random.default_rng(seed)
        self.phase = rng.integers(0, capture.every, env.count)
        self.game = first_game + np.arange(env.count)
        self.next_game = first_game + env.count
        self.outcomes: dict[int, float] = {}
        self.waiting: dict[tuple[int, int], tuple[int, int]] = {}   # (game, seat): the frame its later one should link to

    def _take(self, g: int, side: int) -> tuple[int, int]:
        env = self.env
        seat = g * 2 + side
        bundle = env.bundle_of(seat)
        frame = with_site_status(env.frame(seat), env.table, self.is_extractor)
        tensors = fz.featurize(frame, bundle, env.table, economy=True)
        meta = Meta(self.map_of[id(bundle)], self.source, int(self.game[g]), side, int(env.decisions[g]))
        probes = probe_targets(frame, bundle, env.table)
        self.seen += probes != 0.0
        self.taken += 1
        return self.writer.add(tensors, meta, probes)

    def _step_captures(self) -> None:
        env, capture = self.env, self.capture
        on = (env.decisions + self.phase) % capture.every
        for g in np.flatnonzero((env.decisions >= capture.warm) & ((on == 0) | (on == capture.later % capture.every))):
            g = int(g)
            is_later = bool(on[g] != 0)
            for side in (0, 1):
                key = (int(self.game[g]), side)
                at = self._take(g, side)
                if is_later and key in self.waiting:
                    self.writer.link(self.waiting.pop(key), at)
                elif not is_later:
                    self.waiting[key] = at

    def run(self, frames: int) -> None:
        env = self.env
        env.begin()
        goal = self.writer.total + frames
        while self.writer.total < goal:
            self._step_captures()
            self.writer.flush_if_full()
            env.step(0.0)
            for g in env._replace:
                self.outcomes[int(self.game[g])] = float(env.result[g])
                self.game[g] = self.next_game
                self.next_game += 1
            env.finished_seats()
        self.writer.flush()


def write_manifest(out: pathlib.Path, outcomes: dict[int, float], facts: dict) -> None:
    """How each game ended, by side 0's result, and what made these frames.  A game still going when collecting stopped has none."""
    games = np.array(sorted(outcomes), np.int64)
    np.savez(out / "outcomes.npz", game=games, result=np.array([outcomes[int(g)] for g in games], np.float32))
    (out / "manifest.json").write_text(json.dumps(facts, indent=1), encoding="utf8")


@dataclass(frozen=True)
class Part:
    """One process's share of a collection."""
    out: pathlib.Path
    source: str
    frames: int
    games: int
    seed: int
    threads: int
    script_every: int
    rules: ScenarioRules
    frames_per_shard: int
    number: int = 0


GAMES_APART = 1 << 23       # of numbers for each part's games.  Game numbers don't overlap between the parts of a collection.
SOURCES_APART = 1 << 29     # and each source's, so that three fit in the int32 they're stored in
_FORKED_MAPS: list = []     # as forked processes see them, so that none has to load the pool


def _collect_part(part: Part, maps: list | None = None) -> tuple[dict[int, float], float, np.ndarray, int]:
    maps = _FORKED_MAPS if maps is None else maps
    weights = pool_weights(maps)
    seed = part.seed + 1000 * part.number + SOURCES.index(part.source)
    if part.source == "script":
        env, capture = FrameGames(part.games, seed, part.threads, maps, weights), Capture(part.script_every)
    elif part.source == "army":
        env, capture = se.SimEnv(part.games, seed, ["scripts"], threads=part.threads, maps=maps, weights=weights, variants=True), Capture(part.script_every)
    else:
        env, capture = ScenarioEnv(part.games, seed, part.rules, part.threads, maps, weights), part.rules.capture
    first_game = SOURCES.index(part.source) * SOURCES_APART + part.number * GAMES_APART
    prefix = part.source if not part.number else f"{part.source}-w{part.number}"
    started = time.time()
    collector = Collector(env, capture, ShardWriter(part.out, prefix, part.frames_per_shard), part.source, seed, first_game=first_game)
    collector.run(part.frames)
    return collector.outcomes, collector.writer.total / (time.time() - started), collector.seen, collector.taken


def collect(out: pathlib.Path, script_frames: int, scenario_frames: int, games: int = 128, seed: int = 1, threads: int = 16, script_every: int = 30, rules: ScenarioRules = ScenarioRules(),
            frames_per_shard: int = 2000, maps: list | None = None, workers: int = 1, army_frames: int = 0) -> dict:
    """Plays until there are this many frames of each, on these maps or else the training maps.  `workers` processes share the frames, the games and the threads, for what takes the time is making frames, in Python."""
    global _FORKED_MAPS
    maps = se.pool() if maps is None else maps
    outcomes: dict[int, float] = {}
    rates, shares = {}, {}
    for source, frames in (("script", script_frames), ("scenario", scenario_frames), ("army", army_frames)):
        if not frames:
            continue
        parts = [Part(out, source, frames // workers, max(1, games // workers), seed, max(1, threads // workers), script_every, rules, frames_per_shard, k) for k in range(workers)]
        if workers == 1:
            results = [_collect_part(parts[0], maps)]
        else:
            _FORKED_MAPS = maps
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                results = pool.map(_collect_part, parts)
        for got, _, _, _ in results:
            outcomes.update(got)
        rates[source] = sum(r[1] for r in results)
        taken = max(sum(r[3] for r in results), 1)
        shares[source] = {name: float(n) / taken for name, n in zip(PROBES, sum(r[2] for r in results))}
    facts = {"maps": [f.name for _, _, f in maps], "kinds": [f.kind for _, _, f in maps], "water": [f.water for _, _, f in maps], "weights": pool_weights(maps).tolist(), "held_out": held_out_maps(maps),
             "probes": PROBES, "sources": SOURCES, "frames_per_second": rates, "seen_in": shares, "seed": seed, "feature_version": fz.FEATURE_VERSION}
    write_manifest(out, outcomes, facts)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=pathlib.Path, required=True)
    c.add_argument("--maps", type=pathlib.Path, help="a directory tools/map_gen.py wrote to.  Otherwise the training maps.")
    c.add_argument("--script-frames", type=int, default=0)
    c.add_argument("--scenario-frames", type=int, default=0)
    c.add_argument("--army-frames", type=int, default=0, help="from the C++ army and economy scripts' games, with variants")
    c.add_argument("--games", type=int, default=128)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--script-every", type=int, default=30)
    c.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    maps = se.load_pool(args.maps) if args.maps else None
    print(json.dumps({k: v for k, v in collect(args.out, args.script_frames, args.scenario_frames, args.games, args.seed, args.threads, args.script_every, maps=maps, workers=args.workers, army_frames=args.army_frames).items() if k not in ("maps", "kinds", "water", "weights")}, indent=1))


if __name__ == "__main__":
    main()
