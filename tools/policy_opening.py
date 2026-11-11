"""The opening every team of a sim game plays by rule.

The commander places its factory, free and at once, which then repeats its raider.  The commander goes round the metal spots near
its start, an extractor and a solar beside it at each, and after that helps the factory for good.  What's picked by chance is which
commander and which factory, per team, among the factories that suit the map.  Commanders, factories and raiders are chosen by what the data says they are.

A team faces one enemy, the one whose start is nearest (policy_sim_env picks it), and its spots are those no nearer that enemy's
start than its own and, among allies, nearer its own start than any ally's, a tie going to the lower team: as the scripts share
them out (sim/scripts).
"""
from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import catalog as bot_catalog, terrain  # noqa: E402
from zkbot.policy import bundle as bd, rawframe as rf  # noqa: E402
from zkbot.policy.tables import NO_CLASS  # noqa: E402

MEX, SOLAR = "staticmex", "energysolar"
NEAR_SPOTS = 1300.0        # elmos from the start: past this is expansion, not the opening
MOST_SPOTS = 6             # and no more than this many, the nearest: Agalta's start has 10 in that reach
STEP_SECONDS = 120.0        # a step that takes longer than this is given up on, so a site out of reach can't end the opening
SITE_GAP = 8.0              # elmos between footprints
LINK_REACH = 150.0          # a solar this near an extractor still links to it: pylon ranges 100 and 50
FACTORY_NEAREST, FACTORY_FARTHEST, RING = 64.0, 800.0, 32.0
RALLY_AHEAD = 200.0         # elmos past the factory's edge, clear of where its units come out
RALLY_ROOM = 48.0
TURNS = 16
GROUND = (1 << bd.MOVE_CLASSES.index("kbot")) | (1 << bd.MOVE_CLASSES.index("tank"))
SHIP, HOVER, AMPH = bd.MOVE_CLASSES.index("ship"), bd.MOVE_CLASSES.index("hover"), bd.MOVE_CLASSES.index("amph")
WET = (1 << HOVER) | (1 << AMPH)
ROOMY_CLASS = 0.04          # of a map a factory's units need to get about on
GOOD_REACH = 0.5            # of the metal the best class could get to from the same place: less than this, and that factory is a mistake here
SEA_SHARE = 0.2             # of a map one sea has to be for ships
SEA_NEAR = 1500.0           # elmos from a base to that sea
SEA_MAP = 0.15              # of a map under water: past this, factories that use the water are likelier
WATER_WEIGHT = 3.0          # ... by this
SITE_TRIES = 24
SITE_SPREAD = 300.0
SAMPLED_CELLS = 4096
NEAR_MEMO = 64.0            # elmos: `near` answers once for each square of this size


def modelled(cat: zksim.Catalog, name: str) -> bool:
    """Not a unit whose point is something the sim doesn't do yet (capture, mines, pushing, throwing)."""
    return not any(cat.weapons[w]["flags"] & zksim.W_UNMODELLED for w in cat.units[cat.index[name]]["mount_weapon"])


@dataclass(frozen=True)
class Build:
    type: int
    x: float
    z: float


@dataclass(frozen=True)
class Plan:
    """Where one side builds, whichever factory it picks: their footprints are alike."""
    factory: tuple[float, float]
    builds: tuple[Build, ...]
    rally: tuple[float, float]      # where the fight script gathers fighters and sends the hurt


def is_buildable(bundle: bd.Bundle, x: float, z: float, radius: float) -> bool:
    """Whether a footprint here is on the map and on ground that both walkers and vehicles can use."""
    if x < radius or z < radius or x > bundle.map_w - radius or z > bundle.map_h - radius:
        return False
    rows, cols = bundle.passable.shape
    r0, r1 = int((z - radius) / bd.PASS_CELL), min(int((z + radius) / bd.PASS_CELL) + 1, rows)
    c0, c1 = int((x - radius) / bd.PASS_CELL), min(int((x + radius) / bd.PASS_CELL) + 1, cols)
    return bool(np.all((bundle.passable[r0:r1, c0:c1] & GROUND) == GROUND))


def apart(taken: np.ndarray, x: float, z: float, half_x: float, half_z: float, gap: float = 0.0) -> bool:
    """Whether a footprint of these half sizes here overlaps none of those taken (N x 4: x, z, half x, half z), with a gap
    between if one's asked.  Footprints are rectangles along the map's axes, as the game's are (a build is given no facing), and
    in the game two may touch."""
    return not len(taken) or bool(((np.abs(taken[:, 0] - x) >= taken[:, 2] + half_x + gap) | (np.abs(taken[:, 1] - z) >= taken[:, 3] + half_z + gap)).all())


def footprints(x, z, half_x, half_z) -> np.ndarray:
    """N x 4 for `apart`: x, z, half x, half z, each one number or one for each."""
    columns = np.broadcast_arrays(np.atleast_1d(np.asarray(x, np.float32)), np.asarray(z, np.float32), np.asarray(half_x, np.float32), np.asarray(half_z, np.float32))
    return np.column_stack(columns).astype(np.float32).reshape(-1, 4)


def _towards(x: float, z: float, tx: float, tz: float, distance: float) -> list[tuple[float, float]]:
    """Points this far from (x, z), starting with the one facing (tx, tz) and going round both ways."""
    facing = math.atan2(tz - z, tx - x)
    turns = sorted(range(TURNS), key=lambda k: min(k, TURNS - k))
    return [(x + math.cos(facing + k * 2 * math.pi / TURNS) * distance, z + math.sin(facing + k * 2 * math.pi / TURNS) * distance) for k in turns]


def nearby_spots(spots: np.ndarray, x: float, z: float, ex: float, ez: float, allies: tuple = ()) -> np.ndarray:
    """The spots to take, on our side of the map, each next to the last.  allies: ((x, z), lower) for each ally, `lower` whether
    its team is the lower one, which a spot as near both goes to."""
    own = np.hypot(spots[:, 0] - x, spots[:, 1] - z)
    mine = (own <= NEAR_SPOTS) & (own <= np.hypot(spots[:, 0] - ex, spots[:, 1] - ez))
    for (ax, az), lower in allies:
        theirs = np.hypot(spots[:, 0] - ax, spots[:, 1] - az)
        mine &= (own < theirs) if lower else (own <= theirs)
    near = spots[mine]
    near = near[np.argsort(np.hypot(near[:, 0] - x, near[:, 1] - z), kind="stable")[:MOST_SPOTS]]
    route, left, at = [], list(range(len(near))), (x, z)
    while left:
        i = min(left, key=lambda k: (math.hypot(near[k, 0] - at[0], near[k, 1] - at[1]), k))
        left.remove(i)
        route.append(near[i])
        at = (float(near[i, 0]), float(near[i, 1]))
    return np.array(route, np.float32).reshape(-1, 3)


def make_plan(ground: "Ground", start: tuple[float, float], enemy: tuple[float, float], factory: "Footing",
              mex: int, mex_footing: "Footing", solar: int, solar_footing: "Footing", allies: tuple = ()) -> Plan:
    """factory: a footing as large as any factory's that might go there.  allies: as `nearby_spots` takes them.
    Each goes where the game would let it be built (`Ground.fits`: its footprint's rectangle, on level ground) and both walkers
    and vehicles can get about, a gap from the next."""
    bundle = ground.bundle
    x, z = start
    spots = np.asarray(bundle.spot_list, np.float32).reshape(-1, 3)

    def free(p: tuple[float, float], footing: "Footing", taken: np.ndarray) -> bool:
        return (is_buildable(bundle, *p, max(footing.half_x, footing.half_z)) and ground.fits(*p, footing)
                and apart(taken, *p, footing.half_x, footing.half_z, SITE_GAP))

    # The factory keeps off each spot by an extractor and a solar beside it, whichever side that ends up on.
    keep_off = footprints(spots[:, 0], spots[:, 1], mex_footing.half_x + SITE_GAP + 2 * solar_footing.half_x, mex_footing.half_z + SITE_GAP + 2 * solar_footing.half_z)
    reach = max(factory.half_x, factory.half_z)
    site = None
    for distance in np.arange(reach + FACTORY_NEAREST, FACTORY_FARTHEST, RING):
        site = next((p for p in _towards(x, z, *enemy, float(distance)) if free(p, factory, keep_off)), None)
        if site:
            break
    site = site or start        # nowhere clear in reach, which no training map does
    taken = np.concatenate((footprints(site[0], site[1], factory.half_x, factory.half_z), footprints(spots[:, 0], spots[:, 1], mex_footing.half_x, mex_footing.half_z)))
    builds: list[Build] = []
    for spot in nearby_spots(bundle.spot_list, x, z, *enemy, allies):
        sx, sz = float(spot[0]), float(spot[1])
        builds.append(Build(mex, sx, sz))
        # Right beside it, unless the ground there is too steep.
        beside = None
        for distance in np.arange(max(mex_footing.half_x + solar_footing.half_x, mex_footing.half_z + solar_footing.half_z) + SITE_GAP, LINK_REACH, RING / 2):
            beside = next((p for p in _towards(sx, sz, x, z, float(distance)) if free(p, solar_footing, taken)), None)
            if beside:
                break
        if beside:
            taken = np.concatenate((taken, footprints(beside[0], beside[1], solar_footing.half_x, solar_footing.half_z)))
            builds.append(Build(solar, *beside))
    rally = next((p for distance in np.arange(reach + RALLY_AHEAD, FACTORY_FARTHEST, RING)
                  for p in _towards(*site, *enemy, float(distance)) if is_buildable(bundle, *p, RALLY_ROOM)), site)
    return Plan(site, tuple(builds), rally)


@dataclass
class Seat:
    """One team's way through its opening.  `team` is its player and its commander's handle too (policy_sim_env adds the
    commanders first, in team order)."""
    team: int
    plan: Plan
    factory_type: int
    factory_def: int
    raider_type: int
    step_decisions: int
    next_build: int = -1        # -1: the factory comes first
    factory_uid: int = -1
    deadline: int = 0
    done: bool = False

    def advance(self, units: np.ndarray, decision: int) -> list[tuple[int, int, float, float, int]]:
        """The orders to give now.  Call when the commander has nothing to do, or when its step ran out of time."""
        commander = self.team
        self.deadline = decision + self.step_decisions
        if self.next_build == -1:
            self.next_build = 0
            return [(commander, rf.BUILD, *self.plan.factory, self.factory_type)]
        out = []
        if self.factory_uid < 0:
            built = units[(units["def_id"] == self.factory_def) & (units["player"] == self.team) & ((units["flags"] & rf.BEING_BUILT) == 0)]
            if len(built):
                self.factory_uid = int(built["uid"][0])
                out.append((self.factory_uid, rf.PRODUCE, 0.0, 0.0, self.raider_type))
        if self.next_build < len(self.plan.builds):
            b = self.plan.builds[self.next_build]
            self.next_build += 1
            out.append((commander, rf.BUILD, b.x, b.z, b.type))
            return out
        self.done = True
        if self.factory_uid >= 0:
            out.append((commander, rf.ASSIST, 0.0, 0.0, self.factory_uid))
        return out


class Ground:
    """One map's cells, for placing and for routes.  Everything's made when first asked for, as most maps of a pool are in use for only some of the time."""

    def __init__(self, bundle: bd.Bundle, facts: Any, height: np.ndarray | None = None):
        """height: the sim map's (zksim.Map.height), the ground at each cell of `passable`.  Without it, level ground is judged
        by the bundle's coarser heights."""
        self.bundle, self.facts = bundle, facts
        self.rows, self.cols = bundle.passable.shape
        self._height = np.ascontiguousarray(height, np.float32) if height is not None and height.shape == bundle.passable.shape else None
        self.water = facts.liquid == "water"
        self._masks: dict[tuple[float, float], np.ndarray] = {}
        self._cells: dict[tuple[float, float], np.ndarray] = {}
        self._class_cells: dict[int, np.ndarray] = {}
        self._shares: dict[int, float] = {}
        self._labels: dict[int, np.ndarray] = {}
        self._reach: dict[tuple[int, int], float] = {}
        self._near: dict[tuple[int, int, int], tuple[float, float]] = {}
        self.factory_choices: tuple[list[int], np.ndarray] | None = None   # Roster.factories_for's answer

    def mask(self, low: float, high: float) -> np.ndarray:
        """Where a structure allowed these depths of water can stand.  Below water it goes on the sea bed or floats, and either way, the ground's no business of ours."""
        key = (low, high)
        if key not in self._masks:
            scale = int(self.bundle.cell / bd.PASS_CELL)
            height = np.repeat(np.repeat(self.bundle.height, scale, 0), scale, 1)[:self.rows, :self.cols]
            depth = -height
            bits = self.bundle.passable
            standing = np.where(depth <= 0.0, (bits & GROUND) == GROUND, ((bits & WET) != 0) & self.water)
            self._masks[key] = (depth >= low) & (depth <= high) & standing
        return self._masks[key]

    def _sites(self, low: float, high: float) -> np.ndarray:
        """N x 2: x, z of a sample of those cells."""
        key = (low, high)
        if key not in self._cells:
            cells = np.argwhere(self.mask(low, high))
            cells = cells[np.random.default_rng(len(cells)).permutation(len(cells))[:SAMPLED_CELLS]]
            self._cells[key] = (cells[:, ::-1].astype(np.float32) + 0.5) * float(bd.PASS_CELL)
        return self._cells[key]

    def warm(self, footings: list[tuple[float, float]]) -> None:
        """Makes now what the scripts ask of every map: each footing's mask and sites, and each move class's parts.  For maps shared by
        processes forked after (policy_sim_env.share_maps), which then don't each make their own."""
        for low, high in footings:
            self._sites(low, high)
        for c in range(len(bd.MOVE_CLASSES)):
            self.labels(c)
            self.share(c)
            self._main_cells(c)

    def has_room(self, low: float, high: float) -> bool:
        return len(self._sites(low, high)) > 0

    @property
    def heights(self) -> np.ndarray:
        """The ground's height at each cell of `passable`, float32."""
        if self._height is None:
            scale = int(self.bundle.cell / bd.PASS_CELL)
            self._height = np.ascontiguousarray(np.repeat(np.repeat(self.bundle.height, scale, 0), scale, 1)[:self.rows, :self.cols], np.float32)
        return self._height

    def fits(self, x: float, z: float, footing: "Footing") -> bool:
        """Whether the game would let it be built here, but for what else stands there (CGameHelper::TestUnitBuildSquare): its
        footprint's rectangle on the map, on ground or in water its data allows, and level: every cell under it within its
        maxHeightDif of the ground at its middle.  One whose middle is under water floats or sits on the sea bed, and isn't asked."""
        hx, hz = footing.half_x, footing.half_z
        if x < hx or z < hz or x > self.bundle.map_w - hx or z > self.bundle.map_h - hz:
            return False
        r0, r1 = int((z - hz) / bd.PASS_CELL), min(int((z + hz) / bd.PASS_CELL) + 1, self.rows)
        c0, c1 = int((x - hx) / bd.PASS_CELL), min(int((x + hx) / bd.PASS_CELL) + 1, self.cols)
        if not self.mask(footing.low, footing.high)[r0:r1, c0:c1].all():
            return False
        if footing.height_dif <= 0.0:
            return True
        heights = self.heights
        middle = heights[min(int(z / bd.PASS_CELL), self.rows - 1), min(int(x / bd.PASS_CELL), self.cols - 1)]
        if middle < 0.0:
            return True
        under = heights[r0:r1, c0:c1]
        return bool(under.max() - middle <= footing.height_dif and middle - under.min() <= footing.height_dif)

    def site(self, rng: np.random.Generator, footing: "Footing", x: float, z: float, taken: np.ndarray, spread: float = SITE_SPREAD) -> tuple[float, float] | None:
        """Somewhere near this point it can be built (`fits`), clear of what's taken (N x 4, `footprints`).  If there's nowhere near, as for a shipyard at a base inland, the nearest it can find."""
        def clear(px: float, pz: float) -> bool:
            return apart(taken, px, pz, footing.half_x, footing.half_z)

        for _ in range(SITE_TRIES):
            px, pz = float(x + rng.normal(0.0, spread)), float(z + rng.normal(0.0, spread))
            if self.fits(px, pz, footing) and clear(px, pz):
                return px, pz
        cells = self._sites(footing.low, footing.high)
        if not len(cells):
            return None
        for i in np.argsort(np.hypot(cells[:, 0] - x, cells[:, 1] - z))[:SITE_TRIES]:
            px, pz = float(cells[i, 0]), float(cells[i, 1])
            if self.fits(px, pz, footing) and clear(px, pz):
                return px, pz
        return None

    def labels(self, move_class: int) -> np.ndarray:
        """Which parts of the map connect, for this class: -1 where it can't go.  The map's own if it came with them."""
        if move_class not in self._labels:
            reach = self.facts.reach
            name = bd.MOVE_CLASSES[move_class]
            if reach is not None and name in reach:
                self._labels[move_class] = reach[name]
            else:
                self._labels[move_class] = terrain.label_runs(((self.bundle.passable >> move_class) & 1).astype(bool))[0].astype(np.int16)
        return self._labels[move_class]

    def share(self, move_class: int) -> float:
        if move_class not in self._shares:
            self._shares[move_class] = float(((self.bundle.passable >> move_class) & 1).mean())
        return self._shares[move_class]

    def _main_cells(self, move_class: int) -> np.ndarray:
        """A sample of the cells of the largest part this class can go about on."""
        if move_class not in self._class_cells:
            labels = self.labels(move_class)
            part = self._largest_part(move_class)
            if part >= 0:
                cells = np.argwhere(labels == part)
                cells = cells[np.random.default_rng(len(cells)).permutation(len(cells))[:2 * SAMPLED_CELLS]]
            else:
                cells = np.zeros((0, 2), np.int64)
            self._class_cells[move_class] = (cells[:, ::-1].astype(np.float32) + 0.5) * float(bd.PASS_CELL)
        return self._class_cells[move_class]

    def near(self, move_class: int, x: float, z: float) -> tuple[float, float] | None:
        """The nearest place to this one on the main part of this class's ground, or the point itself if it's already somewhere the class can be."""
        key = (move_class, int(x / NEAR_MEMO), int(z / NEAR_MEMO))
        if key not in self._near:
            cells = self._main_cells(move_class)
            if not len(cells):
                return None
            i = int(np.argmin(np.hypot(cells[:, 0] - x, cells[:, 1] - z)))
            self._near[key] = float(cells[i, 0]), float(cells[i, 1])
        return self._near[key]

    def can(self, move_class: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        i = np.clip((z / bd.PASS_CELL).astype(np.int32), 0, self.rows - 1)
        j = np.clip((x / bd.PASS_CELL).astype(np.int32), 0, self.cols - 1)
        return (self.bundle.passable[i, j] >> move_class) & 1 == 1

    def part_at(self, move_class: int, x: float, z: float) -> int:
        """Which part of its ground the class would be on, at this point or the nearest."""
        labels = self.labels(move_class)
        i, j = min(max(int(z / bd.PASS_CELL), 0), self.rows - 1), min(max(int(x / bd.PASS_CELL), 0), self.cols - 1)
        if labels[i, j] >= 0:
            return int(labels[i, j])
        place = self.near(move_class, x, z)
        if place is None:
            return -1
        return int(labels[min(int(place[1] / bd.PASS_CELL), self.rows - 1), min(int(place[0] / bd.PASS_CELL), self.cols - 1)])

    def joined(self, move_class: int, a: tuple[float, float], b: tuple[float, float]) -> bool:
        part = self.part_at(move_class, *a)
        return part >= 0 and part == self.part_at(move_class, *b)

    def _largest_part(self, move_class: int) -> int:
        labels = self.labels(move_class)
        usable = labels[labels >= 0]
        return int(np.bincount(usable).argmax()) if len(usable) else -1

    def metal_reach(self, move_class: int, at: tuple[float, float] | None = None) -> float:
        """How much of the map's metal is on the part of its ground this class would be on at this point, or with no point, its largest part."""
        part = self.part_at(move_class, *at) if at is not None else self._largest_part(move_class)
        key = (move_class, part)
        if key not in self._reach:
            spots = self.bundle.spot_list
            if not len(spots):
                self._reach[key] = 1.0
            else:
                i = np.clip((spots[:, 1] / bd.PASS_CELL).astype(np.int32), 0, self.rows - 1)
                j = np.clip((spots[:, 0] / bd.PASS_CELL).astype(np.int32), 0, self.cols - 1)
                self._reach[key] = float((self.labels(move_class)[i, j] == part).mean()) if part >= 0 else 0.0
        return self._reach[key]

    def suits(self, footing: "Footing", move_class: int, start: tuple[float, float] | None = None, enemy: tuple[float, float] | None = None) -> bool:
        """Whether a factory of these units makes sense here, for one who starts here and has this enemy; with neither, anywhere on the map.
        Water by itself says little: it's decoration on Akilon, and on Sailaway land units get to 2% of the metal.  So what counts is the metal a class can get to, next
        to what the best of them could from there, and that it can get to the enemy.  move_class: the one its fighters mostly have, NO_CLASS for air, which goes anywhere."""
        if not self.has_room(footing.low, footing.high):
            return False
        if move_class == NO_CLASS:
            return True
        if self.share(move_class) < ROOMY_CLASS:
            return False
        if move_class == SHIP:
            return self._sea_suits(start, enemy)
        if start is not None and enemy is not None and not self.joined(move_class, start, enemy):
            return False
        best = max(self.metal_reach(c, start) for c in range(len(bd.MOVE_CLASSES)) if c != SHIP)
        return self.metal_reach(move_class, start) >= GOOD_REACH * best

    def _sea_suits(self, start: tuple[float, float] | None, enemy: tuple[float, float] | None) -> bool:
        """Ships don't get to metal, so it's the sea: one big enough to fight on, with both bases at its shore."""
        labels = self.labels(SHIP)
        if (labels == self._largest_part(SHIP)).mean() < SEA_SHARE:
            return False
        for at in (start, enemy):
            if at is not None:
                place = self.near(SHIP, *at)
                if place is None or np.hypot(place[0] - at[0], place[1] - at[1]) > SEA_NEAR:
                    return False
        return True

    def chance(self, move_class: int) -> float:
        """How much likelier: where there's much water, what uses it and what flies over it."""
        share = self.facts.water if self.facts.water > 0.0 else float((self.bundle.height < 0.0).mean())
        return WATER_WEIGHT if self.water and share >= SEA_MAP and (move_class == NO_CLASS or move_class in (SHIP, HOVER, AMPH)) else 1.0


@dataclass(frozen=True)
class Footing:
    """What a structure's data says about where it can be built.  The game tests its footprint, a rectangle, and not the circle
    units keep out of (`radius`, half the footprint's width): a Cloakbot Factory's is 112 by 160 elmos, its radius 56."""
    radius: float
    low: float      # water depth, below ground for land: a shipyard must have 15, a solar can't have any
    high: float
    half_x: float = 0.0         # half the footprint, in elmos; the radius both ways if not said
    half_z: float = 0.0
    height_dif: float = 0.0     # how far the ground under it may be above or below the ground at its middle (the game's maxHeightDif); 0 for any

    def __post_init__(self) -> None:
        if self.half_x <= 0.0:
            object.__setattr__(self, "half_x", self.radius)
        if self.half_z <= 0.0:
            object.__setattr__(self, "half_z", self.radius)

    @classmethod
    def of(cls, unit: dict) -> "Footing":
        """From a catalog record."""
        return cls(float(unit["footprint_radius"]), float(unit["min_water_depth"]), float(unit["max_water_depth"]),
                   float(unit.get("foot_x", 0.0)), float(unit.get("foot_z", 0.0)), float(unit.get("max_height_dif", 0.0)))

    def over(self, other: "Footing") -> "Footing":
        """One that covers both, for a place either might go: the larger footprint, the stricter ground."""
        dif = min(self.height_dif, other.height_dif) if self.height_dif > 0.0 and other.height_dif > 0.0 else max(self.height_dif, other.height_dif)
        return Footing(max(self.radius, other.radius), max(self.low, other.low), min(self.high, other.high),
                       max(self.half_x, other.half_x), max(self.half_z, other.half_z), dif)


class Opening:
    """What the data offers, and each map's plans.  Shared by all the games of an environment."""
    favour_water = False

    def __init__(self, cat: zksim.Catalog, decide_every: int = 1):
        self.cat, self.decide_every = cat, decide_every
        self.commanders = [i for i, u in enumerate(cat.units) if u["flags"] & zksim.COMMANDER]
        # One factory for each raider a side could open with, on the ground.  Which of them suit a map is for `choices`.
        self.factories: list[tuple[int, int]] = []
        for i, u in enumerate(cat.units):
            if not u["flags"] & zksim.PLOPPABLE:
                continue
            raider = bot_catalog.pick(u["name"], "raid")
            if raider and raider in cat.index and modelled(cat, raider):
                r = cat.units[cat.index[raider]]
                if not r["flags"] & zksim.AIR and r["move_class"] != zksim.MOVE_CLASSES.index("ship"):
                    self.factories.append((i, cat.index[raider]))
        self.mex, self.solar = cat.index[MEX], cat.index[SOLAR]
        self._plans: dict[tuple, Plan] = {}
        self._factory_footing: Footing | None = None

    @property
    def factory_footing(self) -> Footing:
        """As large as any of the factories that go on land, and wanting ground as level as the strictest: a plan's factory site
        is whichever's."""
        if self._factory_footing is None:
            footings = [Footing.of(self.cat.units[f]) for f, _ in self.factories]
            land = [f for f in footings if f.low <= 0.0] or footings
            out = land[0]
            for f in land[1:]:
                out = out.over(f)
            self._factory_footing = out
        return self._factory_footing

    def plan(self, ground: "Ground", start: tuple[float, float], enemy: tuple[float, float], allies: tuple = ()) -> Plan:
        key = (id(ground.bundle), start, enemy, allies)
        if key not in self._plans:
            u = self.cat.units
            self._plans[key] = make_plan(ground, start, enemy, self.factory_footing, self.mex, Footing.of(u[self.mex]), self.solar,
                                         Footing.of(u[self.solar]), allies)
        return self._plans[key]

    def choices(self, ground: Ground, start: tuple[float, float], enemy: tuple[float, float]) -> tuple[list[int], np.ndarray]:
        """Which of the factories suit this start, by what they open with, and how likely each is."""
        chosen, chances = [], []
        for k, (factory, first) in enumerate(self.factories):
            move_class = int(self.cat.units[first]["move_class"])
            if ground.suits(Footing.of(self.cat.units[factory]), move_class, start, enemy):
                chosen.append(k)
                chances.append(ground.chance(move_class) if self.favour_water else 1.0)
        return chosen, np.array(chances) / max(sum(chances), 1e-9)

    def _fallback(self, ground: Ground, start: tuple[float, float]) -> int:
        """When none suits, the one that gets to most metal."""
        return max(range(len(self.factories)), key=lambda k: ground.metal_reach(int(self.cat.units[self.factories[k][1]]["move_class"]), start))

    def seat(self, rng: np.random.Generator, team: int, ground: Ground, start: tuple[float, float], enemy: tuple[float, float],
             allies: tuple = ()) -> Seat:
        """enemy: the start it plans towards, the nearest enemy's.  allies: as `nearby_spots` takes them, none in a 1v1."""
        chosen, chances = self.choices(ground, start, enemy)
        k = chosen[int(rng.choice(len(chosen), p=chances))] if chosen else self._fallback(ground, start)
        factory, raider = self.factories[k]
        plan = self.plan(ground, start, enemy, allies)
        footing = Footing.of(self.cat.units[factory])
        if footing.low > 0.0:       # a shipyard: as near as there's deep enough water
            place = ground.site(rng, footing, start[0], start[1], np.zeros((0, 4), np.float32))
            if place is not None:
                plan = replace(plan, factory=place)
        return Seat(team, plan, factory, int(self.cat.units[factory]["def_id"]), raider, int(STEP_SECONDS * 2) // self.decide_every)
