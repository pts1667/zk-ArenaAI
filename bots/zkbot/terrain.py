"""Terrain regions: a walkable-distance map graph built from the height data.

Everything here is computed once from arrays, with no engine or bot state, so
it can run offline on map files (tools/terrain_map.py) as well as in a game.

1. Passability.  Per movement class, the engine's own speed formulas
   (Sim/MoveTypes/MoveMath) over the slope and height maps, using the limits in
   Zero-K's gamedata/movedefs.lua.  Reachability uses the full 16-elmo grid;
   coarse cells would hide thin cliffs or close narrow passes.
2. Levels, ramps and chokes.  A level is a large flat area.  A ramp is a walkable
   slope joining two levels of different height; a slope with no level at its
   top goes nowhere and is not a ramp.  A choke is a narrow point, by distance
   to the nearest wall, opening into wider ground on both sides, with no short
   way around; a compound with four gates has four chokes.
3. Regions.  Seeded from metal clusters, start positions, large levels and open
   ground, then grown outward by walking cost, so cliffs and shorelines become
   boundaries and borders prefer to sit on chokes and ramps.  Land and open sea
   never share a region.
4. Graph.  Touching regions are linked by gates.  Each movement class has its
   own gates, with a width and a walking length, so "a ramp the tanks can use"
   and "a cliff only spiders can cross" are different things.  Regions carry
   altitude and terrain tags; gates carry choke, ramp and shore tags.

Water: the engine's water plane is hardcoded at height 0 (CGround::
GetWaterPlaneLevel).  Zero-K's `waterlevel` and `waterpreset` mod options move
the terrain instead (GG.Terraform_RaiseWater), so heights must be read after
that has run; the snapshot sent at AI load may predate it.  `water_level` is
subtracted from the heights up front, so nothing below depends on it being 0.

Harmful liquid is not water.  Any water damage makes the plane lava, or void
from the level where the engine itself stops ground units walking in (maps kill
there, The Hole by gadget at y < 20).  The engine would still path through lava
at a cost, and the units would burn, so no class may enter either one and they
make no sea regions.  A void's kill line can sit above the plane, so the ground
just above it counts too (VOID_MARGIN); lava's doesn't, as real low shelves sit
just above it (Carabanossi has 18% of its map at 10-20 elmos).  The one exception
is hover where Zero-K's map_lava gadget runs (`hover_safe`, known live
from the `lavaLevel` game rules param): it cancels their lava damage.  A map's
own lava gadget doesn't, as on Banana Republic.

Not modelled: map typemap speed multipliers and blocking features, which the
bridge does not expose, and lava that rises during a game.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

SQUARE = 8.0                  # elmos per height map square
SLOPE_CELL = 16.0             # elmos per slope map cell
MAX_WORK_CELLS = 90_000       # keeps the build to a second or two
PASS_SHARE = 1.0              # share of a work cell that must be passable; a
                              # lower value lets regions grow through thin cliffs
MAX_OPEN_REGIONS = 32         # regions with no metal; metal clusters always get one
MIN_POCKET_ELMOS2 = 128.0 * 128.0        # smaller passable areas don't count
MIN_OPEN_SEED_ELMOS2 = 512.0 * 512.0     # metal-less landmass or sea that gets a seed
MIN_OPEN_REGION_ELMOS2 = 640.0 * 640.0   # smaller metal-less regions join their neighbours
SEA_GROW_COST = 0.5           # open sea holds less to fight over: larger regions
CHOKE_WIDTH = 320.0           # elmos; a passage narrower than this may be a choke
CHOKE_SCALES = (320.0, 192.0, 128.0)  # passage widths looked for, widest first
CHOKE_OPENING = 1.6           # the ground beyond must be this much wider
CHOKE_AROUND = 1000.0         # elmos; a shorter way around makes it a mere gap
CHOKE_MAX_MOUTHS = 8          # of one narrow part, the largest
CHOKE_GROW_COST = 8.0         # growth cost in a choke, to put borders on them
BORDER_DETOUR = 2.0           # a narrow region border counts as a choke if going around costs this many times more

FLAT_SLOPE = 0.02             # engine slope; a gradient of about 0.2, above map noise
FLAT_SLOPE_TIGHT = 0.006      # re-split flat ground that still climbs a level's worth
LEVEL_MIN_ELMOS2 = 320.0 * 320.0         # flat ground smaller than this is no level
LEVEL_SEED_ELMOS2 = 640.0 * 640.0        # a metal-less level this big gets a region
RAMP_RISE = 40.0              # elmos between two levels for a slope to be a ramp
RAMP_MAX_LENGTH = 1000.0      # elmos; longer than this is rolling country
RAMP_MERGE_GAP = 384.0        # elmos; closer stretches of one hillside are one ramp
RAMP_GROW_COST = 3.0          # growth cost on ramps, to put borders on them

LAND, SEA, WALL = 1, 2, 0

# Engine limits above which water stops being movable (MoveDefHandler.cpp).
MAX_WATER_DAMAGE_GROUND = 1e3
MAX_WATER_DAMAGE_HOVER = 1e4
VOID_MARGIN = 20.0            # elmos; the kill line of the void gadget on The Hole and Hexian Arena


@dataclass(frozen=True)
class MoveClass:
    """One family of Zero-K move defs (gamedata/movedefs.lua)."""
    name: str
    kind: str                         # ground, hover or ship
    slope_deg: float = 0.0
    depth: float = 0.0                # ground/hover: max depth; ship: min depth
    slope_mod: float | None = None    # None: the engine default
    depth_coeffs: tuple[float, float] = (0.0, 0.0)   # quadratic, linear

    @property
    def max_slope(self) -> float:
        # MoveDefHandler::DegreesToMaxSlope
        return 1.0 - math.cos(math.radians(min(60.0, max(0.0, self.slope_deg)) * 1.5))

    @property
    def slope_mod_value(self) -> float:
        return 4.0 / (self.max_slope + 0.001) if self.slope_mod is None else self.slope_mod


_WADE = (0.0027, 0.02)        # common_depthmodparams
MOVE_CLASSES: dict[str, MoveClass] = {c.name: c for c in (
    MoveClass("kbot", "ground", 36.0, 22.0, depth_coeffs=_WADE),
    MoveClass("tank", "ground", 18.0, 22.0, slope_mod=20.0, depth_coeffs=_WADE),
    MoveClass("spider", "ground", 70.0, 22.0, depth_coeffs=_WADE),
    MoveClass("amph", "ground", 36.0, 5000.0),
    MoveClass("hover", "hover", 18.0, 5000.0, slope_mod=30.0),
    MoveClass("ship", "ship", depth=5.0),
)}


def slope_from_corners(corners: np.ndarray) -> np.ndarray:
    """The engine's slope map (CReadMap::UpdateSlopemap) from corner heights
    of shape (mapz + 1, mapx + 1); the result has one value per 2x2 squares."""
    c = np.asarray(corners, np.float64)
    d_tr = c[:-1, 1:] - c[:-1, :-1]
    d_bl = c[1:, :-1] - c[:-1, :-1]
    d_h = c[1:, :-1] - c[1:, 1:]
    d_v = c[:-1, 1:] - c[1:, 1:]
    top_left = SQUARE / np.sqrt(SQUARE * SQUARE + d_tr * d_tr + d_bl * d_bl)
    bottom_right = SQUARE / np.sqrt(SQUARE * SQUARE + d_h * d_h + d_v * d_v)
    faces = np.stack((top_left, bottom_right), axis=-1)
    h, w = faces.shape[:2]
    h -= h % 2
    w -= w % 2
    tiles = faces[:h, :w].reshape(h // 2, 2, w // 2, 2, 2).transpose(0, 2, 1, 3, 4).reshape(h // 2, w // 2, 8)
    avg = tiles.mean(axis=2)
    low = tiles.min(axis=2)
    lerp = low / avg           # "smooth it a bit, so small holes don't block huge tanks"
    return (1.0 - (low + (avg - low) * lerp)).astype(np.float32)


def liquid_kind(water_damage: float) -> str:
    """"water", "lava" or "void", from the engine's scaled damage per slow update."""
    if water_damage <= 0.0:
        return "water"
    return "lava" if water_damage < MAX_WATER_DAMAGE_GROUND else "void"


def hazard_line(water_damage: float) -> float:
    """Height relative to the plane below which a unit is harmed."""
    return VOID_MARGIN if liquid_kind(water_damage) == "void" else 0.0


def speed_field(mc: MoveClass, height: np.ndarray, slope: np.ndarray, water_damage: float = 0.0,
                hover_safe: bool = False) -> np.ndarray:
    """Speed multiplier per cell: CMoveMath's Ground/Hover/ShipSpeedMod, except
    that nothing enters a harmful liquid (module notes).  `height` is relative
    to the plane; 0 means impassable."""
    if mc.kind == "ship":
        out = (-height >= mc.depth).astype(np.float32)
    else:
        land_speed = (1.0 / (1.0 + slope * mc.slope_mod_value)).astype(np.float32)
        land_speed[slope > mc.max_slope] = 0.0
        if mc.kind == "hover":
            on_water = 0.0 if water_damage >= MAX_WATER_DAMAGE_HOVER else 1.0
            out = np.where(height < 0.0, on_water, land_speed).astype(np.float32)
            if hover_safe:
                return out
        else:
            depth = np.maximum(0.0, -height)
            scale = np.maximum(0.01, mc.depth_coeffs[0] * depth * depth + mc.depth_coeffs[1] * depth + 1.0)
            out = land_speed / scale
            out[depth > mc.depth] = 0.0
    if water_damage > 0.0:
        out[height < hazard_line(water_damage)] = 0.0
    return out.astype(np.float32)


def label_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected areas (4-neighbour) by row runs and union-find: far fewer
    runs than cells, so this stays quick on the full grid of a big map.
    Returns (label per cell, -1 outside; size per label)."""
    h, w = mask.shape
    edges = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows_s, cols_s = np.nonzero(edges == 1)
    _rows_e, cols_e = np.nonzero(edges == -1)
    n = len(rows_s)
    if n == 0:
        return np.full((h, w), -1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    row_list, start, end = rows_s.tolist(), cols_s.tolist(), cols_e.tolist()
    row_first = 0             # first run of the current row
    prev_first = prev_end = 0 # runs [prev_first, prev_end) are the previous row's
    current_row = -2
    for r in range(n):
        row = row_list[r]
        if row != current_row:
            if row == current_row + 1:
                prev_first, prev_end = row_first, r
            else:
                prev_first = prev_end = r
            row_first = r
            current_row = row
        k = prev_first
        while k < prev_end:
            if end[k] <= start[r]:
                k += 1
                prev_first = k    # it ends before this run, so before later ones too
                continue
            if start[k] >= end[r]:
                break
            ra, rb = find(k), find(r)
            if ra != rb:
                parent[rb] = ra
            k += 1
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    ids, compact = np.unique(roots, return_inverse=True)
    lengths = cols_e - cols_s
    sizes = np.bincount(compact, weights=lengths, minlength=len(ids)).astype(np.int32)
    out = np.full(h * w, -1, dtype=np.int32)
    offsets = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    out[np.repeat(rows_s * w + cols_s, lengths) + offsets] = np.repeat(compact, lengths)
    return out.reshape(h, w), sizes


def _block_mean(a: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return a
    h, w = a.shape[0] // b, a.shape[1] // b
    return a[:h * b, :w * b].reshape(h, b, w, b).mean(axis=(1, 3))


def _pool_speed(speed: np.ndarray, b: int) -> np.ndarray:
    """A work cell is passable when enough of it is; its speed is the mean of
    the passable part, so a narrow ramp keeps its own speed."""
    if b <= 1:
        return speed
    ok = speed > 0.0
    share = _block_mean(ok.astype(np.float32), b)
    total = _block_mean(speed, b)
    out = np.zeros_like(share, dtype=np.float32)
    keep = share >= PASS_SHARE
    out[keep] = total[keep] / share[keep]
    return out


@dataclass
class Level:
    """A large flat area at one height."""
    lid: int
    x: float
    z: float
    height: float
    area: float


@dataclass
class Ramp:
    """A walkable slope joining two levels."""
    rid: int
    low: int                      # level ids
    high: int
    x: float                      # the middle of the slope
    z: float
    width: float                  # across the middle, elmos
    length: float                 # level edge to level edge, elmos
    rise: float
    tanks: bool                   # gentle enough for tanks and hovers along its whole length
    low_region: int = -1
    high_region: int = -1


@dataclass
class Choke:
    """A narrow passage between wider ground."""
    cid: int
    x: float
    z: float
    width: float
    cls: str                      # kbot on land, ship for a strait at sea


@dataclass
class Gate:
    """One contiguous passable stretch of a border, for one movement class."""
    x: float
    z: float
    width: float
    length: float                 # walking cost, centre to centre, in elmos
    area: int = -1                # the connected passable area it belongs to
    ramp: int | None = None       # Ramp.rid when the gate sits on a ramp
    choke: int | None = None      # Choke.cid when the gate sits in a choke
    tags: set[str] = field(default_factory=set)   # choke, ramp, shore


@dataclass
class Edge:
    a: int
    b: int
    gates: dict[str, list[Gate]] = field(default_factory=dict)   # by movement class

    def best_gate(self, cls: str, area: int | None = None) -> Gate | None:
        gs = [g for g in self.gates.get(cls, ()) if area is None or g.area == area]
        return min(gs, key=lambda g: g.length) if gs else None

    def width(self, cls: str) -> float:
        return sum(g.width for g in self.gates.get(cls, ()))


@dataclass
class Region:
    rid: int
    x: float
    z: float
    domain: int                   # LAND or SEA
    spots: list[int] = field(default_factory=list)   # indices into the spot list
    income: float = 0.0
    cells: int = 0
    area: float = 0.0             # elmos^2, walls included
    height: float = 0.0           # mean walkable height above the water plane
    rel_height: float = 0.0       # versus neighbours at another height, by border length
    level: int = -1               # the level covering most of it
    wall_share: float = 0.0
    water_share: float = 0.0
    access: dict[str, float] = field(default_factory=dict)   # class -> passable share
    tags: set[str] = field(default_factory=set)   # sea shallow plateau terrace basin flat rough island coast
    neighbors: list[int] = field(default_factory=list)


class TerrainMap:
    def __init__(self, height: np.ndarray, slope: np.ndarray | None = None,
                 spots: list[tuple[float, float, float]] | None = None,
                 anchors: list[tuple[float, float]] | None = None,
                 water_level: float = 0.0, water_damage: float = 0.0, hover_safe: bool = False,
                 cluster_radius: float | None = None) -> None:
        """height: center heights per 8-elmo square, as the height_map query
        returns them.  slope: per 16-elmo cell, as the slope_map query returns
        it (derived from height if None).  spots: (x, z, income)."""
        h = np.asarray(height, np.float32) - float(water_level)
        if h.ndim != 2 or h.shape[0] < 4 or h.shape[1] < 4:
            raise ValueError(f"height map too small: {h.shape}")
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        self.water_level = float(water_level)
        # The engine's and the bridge's unit, per slow update.  Where map_lava runs
        # the plane burns by gadget, even if mapinfo gives it no damage of its own.
        water_damage = max(float(water_damage), 1.0 if hover_safe else 0.0)
        self.water_damage = water_damage
        self.liquid = liquid_kind(self.water_damage)
        self.hover_safe = bool(hover_safe)
        self.width = h.shape[1] * SQUARE
        self.depth = h.shape[0] * SQUARE
        self.spots = [(float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.0) for s in (spots or [])]
        if slope is None:
            slope = slope_from_corners(np.pad(h, ((0, 1), (0, 1)), mode="edge"))
        slope = np.nan_to_num(np.asarray(slope, np.float32))
        h16 = _block_mean(h, 2)
        rows = min(h16.shape[0], slope.shape[0])
        cols = min(h16.shape[1], slope.shape[1])
        h16, slope = h16[:rows, :cols], slope[:rows, :cols]

        # Work grid: 32 elmos, coarser only on huge maps.
        b = 2
        while (rows // b) * (cols // b) > MAX_WORK_CELLS:
            b *= 2
        self.block = b
        self.cell = SLOPE_CELL * b
        fine_speed = {name: speed_field(mc, h16, slope, water_damage, hover_safe) for name, mc in MOVE_CLASSES.items()}
        self.hazard = h16 < hazard_line(self.water_damage) if self.liquid != "water" else np.zeros_like(h16, dtype=bool)
        self.speed = {name: _pool_speed(s, b) for name, s in fine_speed.items()}
        self.height = _block_mean(h16, b).astype(np.float32)
        self.slope = _block_mean(slope, b).astype(np.float32)
        self.water = _block_mean((h16 < 0.0).astype(np.float32), b).astype(np.float32)
        self.h, self.w = self.height.shape
        self.min_pocket_fine = max(2, int(MIN_POCKET_ELMOS2 / (SLOPE_CELL * SLOPE_CELL)))
        self.min_seed_cells = max(2, int(MIN_OPEN_SEED_ELMOS2 / (self.cell * self.cell)))

        # Reachability on the full grid, trimmed to whole work cells.
        self.comp: dict[str, np.ndarray] = {}
        self.comp_size: dict[str, np.ndarray] = {}
        for name, s in fine_speed.items():
            self.comp[name], self.comp_size[name] = label_runs(s[:self.h * b, :self.w * b] > 0.0)

        land = self.speed["kbot"] > 0.0
        sea = ~land & (self.water >= 0.5) & ((self.speed["hover"] > 0.0) | (self.speed["ship"] > 0.0))
        self.domain = np.full((self.h, self.w), WALL, dtype=np.int8)
        self.domain[land] = LAND
        self.domain[sea] = SEA
        # Lava or void no unit stands in.  Each cell still has a nearest region so
        # that any position is answerable, but it says nothing about that region.
        self.burning = (self.domain == WALL) & (self.water >= 0.5) & (self.liquid != "water")

        self.cluster_radius = float(cluster_radius or max(420.0, min(680.0, math.hypot(self.width, self.depth) * 0.085)))
        self.levels: list[Level] = []
        self.ramps: list[Ramp] = []
        self.level_id = np.full((self.h, self.w), -1, dtype=np.int32)
        self.ramp_id = np.full((self.h, self.w), -1, dtype=np.int32)
        self.chokes: list[Choke] = []
        self.fine_label: dict[str, np.ndarray] = {}   # per class, on the full grid; see _fine_labels
        self.choke_id = np.full((self.h, self.w), -1, dtype=np.int32)
        self.regions: list[Region] = []
        self.edges: dict[tuple[int, int], Edge] = {}
        self.label = np.full((self.h, self.w), -1, dtype=np.int32)
        self._find_levels()
        self._find_ramps()
        self._find_chokes("kbot", self.domain == LAND)
        self._find_chokes("ship", self.speed["ship"] > 0.0)
        self._build_regions(anchors or [])
        self._build_edges()
        self._label_regions()

    # ------------------------------------------------------------- geometry

    def _cell_of(self, x: float, z: float) -> tuple[int, int]:
        return (min(self.h - 1, max(0, int(z / self.cell))), min(self.w - 1, max(0, int(x / self.cell))))

    def _centre_of(self, i: int, j: int) -> tuple[float, float]:
        return (j + 0.5) * self.cell, (i + 0.5) * self.cell

    def _snap(self, i: int, j: int, radius: int = 12) -> tuple[int, int] | None:
        """Nearest cell in a land or sea domain, for spots on steep ground."""
        if self.domain[i, j] != WALL:
            return i, j
        for r in range(1, radius + 1):
            best = None
            best_d = 1e18
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < self.h and 0 <= b < self.w and self.domain[a, b] != WALL:
                        d = di * di + dj * dj
                        if d < best_d:
                            best_d, best = d, (a, b)
            if best is not None:
                return best
        return None

    # ------------------------------------------------------ levels and ramps

    def _find_levels(self) -> None:
        dry = (self.domain == LAND) & (self.water < 0.5)
        min_cells = max(4, int(LEVEL_MIN_ELMOS2 / (self.cell * self.cell)))
        comp, sizes = label_runs(dry & (self.slope < FLAT_SLOPE))
        pieces = []
        for cid in np.nonzero(sizes >= min_cells)[0]:
            cells = comp == cid
            hs = self.height[cells]
            if np.percentile(hs, 95) - np.percentile(hs, 5) < RAMP_RISE:
                pieces.append(np.nonzero(cells))
                continue
            # Gently climbing ground: a stricter test finds the flats in it.
            sub, sub_sizes = label_runs(cells & (self.slope < FLAT_SLOPE_TIGHT))
            pieces += [np.nonzero(sub == k) for k in np.nonzero(sub_sizes >= min_cells)[0]]
        for cells in pieces:
            cz, cx = cells[0].mean(), cells[1].mean()
            k = int(np.argmin((cells[0] - cz) ** 2 + (cells[1] - cx) ** 2))
            lid = len(self.levels)
            x, z = self._centre_of(int(cells[0][k]), int(cells[1][k]))
            self.levels.append(Level(lid, x, z, float(np.median(self.height[cells])),
                                     float(len(cells[0]) * self.cell * self.cell)))
            self.level_id[cells] = lid

    @staticmethod
    def _chamfer(zero: np.ndarray) -> list[float]:
        """Distance in cells to the nearest True cell (two-pass chamfer).  The
        outermost row and column are not updated: callers pad by one."""
        h, w = zero.shape
        d = np.where(zero, 0.0, 1e9).ravel().tolist()
        ortho, diag = 1.0, math.sqrt(2.0)
        for i in range(1, h - 1):
            for u in range(i * w + 1, i * w + w - 1):
                if d[u]:
                    d[u] = min(d[u], d[u - 1] + ortho, d[u - w] + ortho, d[u - w - 1] + diag, d[u - w + 1] + diag)
        for i in range(h - 2, 0, -1):
            for u in range(i * w + w - 2, i * w, -1):
                if d[u]:
                    d[u] = min(d[u], d[u + 1] + ortho, d[u + w] + ortho, d[u + w + 1] + diag, d[u + w - 1] + diag)
        return d

    def _find_chokes(self, cls: str, passable: np.ndarray) -> None:
        """Narrow passages, by shape.  Open ground is wherever a disk fits; the
        passable rest is narrow.  A narrow part touching open ground in two
        separate places joins them, and the narrowest point of the widest path
        through it is a choke, unless there is a short way around that is wider
        than the choke.  A compound's gates all count: none is a way around
        another unless it is wider and close by.  Corners touch open ground in
        one place only, so they never count."""
        if not passable.any():
            return
        h, w = self.h + 2, self.w + 2
        pad = np.pad(passable, 1)
        clear = self._chamfer(~pad)
        arr = np.array(clear, dtype=np.float64).reshape(h, w)
        found: list[tuple[float, int]] = []
        for width in CHOKE_SCALES:
            limit = (width / self.cell + 0.5) / 2.0
            if limit < 1.5:           # too narrow for this grid to see
                continue
            reach = limit * CHOKE_OPENING
            core = arr > reach
            if not core.any():
                continue
            opened = pad & (np.array(self._chamfer(core), dtype=np.float64).reshape(h, w) <= reach)
            found += self._narrow_passages(pad, opened, clear, limit)
        kept: list[tuple[int, int, float]] = []
        for c, u in sorted(found):
            i, j = u // w - 1, u % w - 1
            if all((i - a) ** 2 + (j - b) ** 2 > (c + e + 2.0) ** 2 for a, b, e in kept):
                kept.append((i, j, c))
        for i, j, c in kept:
            cid = len(self.chokes)
            x, z = self._centre_of(i, j)
            self.chokes.append(Choke(cid, x, z, (2.0 * c - 0.5) * self.cell, cls))
            r = int(math.ceil(c + 1.0))
            zz, xx = np.ogrid[-i:self.h - i, -j:self.w - j]
            disk = (zz * zz + xx * xx <= r * r) & passable & (self.choke_id < 0)
            self.choke_id[disk] = cid

    def _narrow_passages(self, pad: np.ndarray, opened: np.ndarray, clear: list[float],
                         limit: float) -> list[tuple[float, int]]:
        """(half width in cells, padded flat index) of each passage at one scale."""
        h, w = pad.shape
        comp, sizes = label_runs(pad & ~opened)
        if not len(sizes):
            return []
        zs, xs = np.nonzero(comp >= 0)
        order = np.argsort(comp[zs, xs], kind="stable")
        ends = np.cumsum(sizes).tolist()
        out = []
        first = 0
        for k, last in enumerate(ends):
            idx = order[first:last]
            first = last
            i0, i1 = int(zs[idx].min()), int(zs[idx].max())
            j0, j1 = int(xs[idx].min()), int(xs[idx].max())
            # Open ground touching this part (i0 >= 1 and so on: the border is impassable).
            mine = comp[i0 - 1:i1 + 2, j0 - 1:j1 + 2] == k
            near = mine.copy()
            near[1:] |= mine[:-1]
            near[:-1] |= mine[1:]
            spread = near.copy()
            spread[:, 1:] |= near[:, :-1]
            spread[:, :-1] |= near[:, 1:]
            contact, contact_sizes = label_runs(spread & opened[i0 - 1:i1 + 2, j0 - 1:j1 + 2])
            if len(contact_sizes) < 2:
                continue
            biggest = np.argsort(-contact_sizes, kind="stable")[:CHOKE_MAX_MOUTHS].tolist()
            mouths = []
            for g in biggest:
                gz, gx = np.nonzero(contact == g)
                mouths.append(((gz + i0 - 1) * w + gx + j0 - 1).tolist())
            part = set((zs[idx] * w + xs[idx]).tolist())
            for m, mouth in enumerate(mouths[:-1]):
                for target in mouths[m + 1:]:
                    hit = self._widest_path(part, mouth, target, clear, w)
                    if hit is None or hit[0] > limit:
                        continue
                    if not self._way_around(mouth, target, clear, hit[0], w):
                        out.append(hit)
        return out

    @staticmethod
    def _steps(w: int) -> tuple[tuple[int, float, int, int], ...]:
        """(offset, length, the two orthogonal cells a diagonal step squeezes between)."""
        d = math.sqrt(2.0)
        return ((-1, 1.0, 0, 0), (1, 1.0, 0, 0), (-w, 1.0, 0, 0), (w, 1.0, 0, 0),
                (-w - 1, d, -w, -1), (-w + 1, d, -w, 1), (w - 1, d, w, -1), (w + 1, d, w, 1))

    def _widest_path(self, part: set[int], mouth: list[int], target: list[int], clear: list[float],
                     w: int) -> tuple[float, int] | None:
        """Through the narrow part, the path whose tightest point is widest:
        where a unit would actually go.  Returns its tightest point, the middle
        of it if the passage stays that narrow for a while."""
        steps = self._steps(w)
        best: dict[int, tuple[float, float]] = {}
        parent: dict[int, int] = {}
        q: list[tuple[float, float, int]] = []
        for c in mouth:
            for off, _len, _a, _b in steps[:4]:
                v = c + off
                if v in part and v not in best:
                    best[v] = (clear[v], 1.0)
                    heapq.heappush(q, (-clear[v], 1.0, v))
        goal = set(target)
        end = None
        while q:
            neg, dist, u = heapq.heappop(q)
            if best[u] != (-neg, dist):
                continue
            if any(u + off in goal for off, _l, _a, _b in steps[:4]):
                end = u
                break
            for off, length, a, b in steps:
                v = u + off
                if v not in part or (a and not (clear[u + a] and clear[u + b])):
                    continue
                cand = (min(-neg, clear[v]), dist + length)
                old = best.get(v)
                if old is None or cand[0] > old[0] or (cand[0] == old[0] and cand[1] < old[1]):
                    best[v] = cand
                    parent[v] = u
                    heapq.heappush(q, (-cand[0], cand[1], v))
        if end is None:
            return None
        path = [end]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        widths = [clear[c] for c in path]
        c = min(widths)
        at = widths.index(c)
        lo = hi = at
        while lo > 0 and widths[lo - 1] <= c + 0.5:
            lo -= 1
        while hi + 1 < len(path) and widths[hi + 1] <= c + 0.5:
            hi += 1
        return c, path[(lo + hi) // 2]

    def _way_around(self, mouth: list[int], target: list[int], clear: list[float], c: float, w: int) -> bool:
        """Can a unit get from one mouth to the other nearby, on ground wider than the choke?"""
        limit = CHOKE_AROUND / self.cell
        steps = self._steps(w)
        # Next to the mouth counts: it may itself be no wider than the choke.
        goal = {u + off for u in target for off, _l, _a, _b in steps} | set(target)
        best = dict.fromkeys(mouth, 0.0)
        q = [(0.0, u) for u in mouth]
        while q:
            dist, u = heapq.heappop(q)
            if dist > best[u]:
                continue
            for off, length, _a, _b in steps:
                v = u + off
                nd = dist + length
                if clear[v] > c and nd <= limit and nd < best.get(v, math.inf):
                    if v in goal:
                        return True
                    best[v] = nd
                    heapq.heappush(q, (nd, v))
        return False

    def _walk_from(self, starts: set[int], allowed: set[int]) -> dict[int, float]:
        """Plan distance from `starts` through `allowed` (no corner cutting)."""
        w = self.w
        ortho, diag = self.cell, self.cell * math.sqrt(2.0)
        dist = {s: 0.0 for s in starts}
        q = [(0.0, s) for s in starts]
        heapq.heapify(q)
        while q:
            d, u = heapq.heappop(q)
            if d > dist[u] or d > RAMP_MAX_LENGTH:
                continue
            x = u % w
            left = x > 0 and (u - 1) in allowed
            right = x < w - 1 and (u + 1) in allowed
            up = (u - w) in allowed
            down = (u + w) in allowed
            nbrs = []
            if left:
                nbrs.append((u - 1, ortho))
            if right:
                nbrs.append((u + 1, ortho))
            if up:
                nbrs.append((u - w, ortho))
                if left and (u - w - 1) in allowed:
                    nbrs.append((u - w - 1, diag))
                if right and (u - w + 1) in allowed:
                    nbrs.append((u - w + 1, diag))
            if down:
                nbrs.append((u + w, ortho))
                if left and (u + w - 1) in allowed:
                    nbrs.append((u + w - 1, diag))
                if right and (u + w + 1) in allowed:
                    nbrs.append((u + w + 1, diag))
            for v, step in nbrs:
                nd = d + step
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(q, (nd, v))
        return dist

    def _find_ramps(self) -> None:
        """Walkable non-level ground touching two levels of different height,
        cut at the halfway line so separate slopes count separately."""
        if len(self.levels) < 2:
            return
        w = self.w
        slopes = (self.domain == LAND) & (self.level_id < 0)
        comp, _sizes = label_runs(slopes)
        lv = self.level_id
        idx = np.arange(self.h * w).reshape(self.h, w)
        touch: dict[int, dict[int, set[int]]] = {}
        for a, cell_ids, other in ((comp[:, :-1], idx[:, :-1], lv[:, 1:]), (comp[:, 1:], idx[:, 1:], lv[:, :-1]),
                                   (comp[:-1, :], idx[:-1, :], lv[1:, :]), (comp[1:, :], idx[1:, :], lv[:-1, :])):
            m = (a >= 0) & (other >= 0)
            for c, cell, level in zip(a[m].tolist(), cell_ids[m].tolist(), other[m].tolist()):
                touch.setdefault(c, {}).setdefault(level, set()).add(cell)
        flat_comp = comp.ravel()
        found = []
        for cid, by_level in touch.items():
            lids = sorted(by_level, key=lambda l: self.levels[l].height)
            if len(lids) < 2 or self.levels[lids[-1]].height - self.levels[lids[0]].height < RAMP_RISE:
                continue
            allowed = set(np.nonzero(flat_comp == cid)[0].tolist())
            dist = {l: self._walk_from(by_level[l], allowed) for l in lids}
            for i, low in enumerate(lids):
                for high in lids[i + 1:]:
                    rise = self.levels[high].height - self.levels[low].height
                    if rise < RAMP_RISE:
                        continue
                    da, db = dist[low], dist[high]
                    total = {c: da[c] + db[c] for c in da if c in db}
                    if not total or min(total.values()) > RAMP_MAX_LENGTH:
                        continue
                    slack = min(total.values()) * 1.3 + 2.0 * self.cell
                    mid = {c for c, t in total.items() if t <= slack and abs(da[c] - db[c]) <= self.cell}
                    for section in self._clusters(mid, int(RAMP_MERGE_GAP / self.cell)):
                        found.append((rise, low, high, section, total, by_level[low], by_level[high]))
        found.sort(key=lambda f: -f[0])
        flat_ramp = self.ramp_id.ravel()
        for rise, low, high, section, total, starts, ends in found:
            length = min(total[c] for c in section)
            zs, xs = np.divmod(np.array(section), w)
            k = int(np.argmin((zs - zs.mean()) ** 2 + (xs - xs.mean()) ** 2))
            # A bigger climb already runs through here: same hillside, not a new ramp.
            if flat_ramp[int(section[k])] >= 0:
                continue
            x, z = self._centre_of(int(zs[k]), int(xs[k]))
            width = (math.hypot(int(xs.max() - xs.min()), int(zs.max() - zs.min())) + 1.0) * self.cell
            corridor = self._corridor(section, total, length)
            rid = len(self.ramps)
            self.ramps.append(Ramp(rid, low, high, x, z, width, length, rise, self._tanks_cross(corridor, starts, ends)))
            for c in corridor:
                if flat_ramp[c] < 0:
                    flat_ramp[c] = rid

    def _corridor(self, section: list[int], total: dict[int, float], length: float) -> set[int]:
        """The slope belonging to one ramp: on a short way between the two
        levels, and near this ramp's middle rather than another's further
        along the same hillside."""
        cells = np.array([c for c, t in total.items() if t <= length * 1.3 + 2.0 * self.cell])
        zs, xs = np.divmod(cells, self.w)
        sample = np.array(section[:: max(1, len(section) // 64)])
        sz, sx = np.divmod(sample, self.w)
        d2 = (zs[:, None] - sz[None, :]) ** 2 + (xs[:, None] - sx[None, :]) ** 2
        reach = max(2.0, length * 0.65 / self.cell)
        return set(cells[d2.min(axis=1) <= reach * reach].tolist())

    def _clusters(self, cells: set[int], gap: int = 1) -> list[list[int]]:
        """Groups of flat cell indices no more than `gap` cells apart."""
        w = self.w
        steps = range(-gap, gap + 1)
        left = set(cells)
        out = []
        while left:
            start = left.pop()
            group = [start]
            stack = [start]
            while stack:
                u = stack.pop()
                x = u % w
                for dz in steps:
                    for dx in steps:
                        if not 0 <= x + dx < w:
                            continue
                        v = u + dz * w + dx
                        if v in left:
                            left.remove(v)
                            group.append(v)
                            stack.append(v)
            out.append(group)
        return out

    def _tanks_cross(self, corridor: set[int], starts: set[int], ends: set[int]) -> bool:
        flat = (self.speed["tank"] > 0.0).ravel()
        w = self.w
        open_cells = {c for c in corridor if flat[c]}
        stack = [c for c in starts if c in open_cells]
        seen = set(stack)
        while stack:
            u = stack.pop()
            if u in ends:
                return True
            x = u % w
            for v, ok in ((u - 1, x > 0), (u + 1, x < w - 1), (u - w, True), (u + w, True)):
                if ok and v in open_cells and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    # -------------------------------------------------------------- growth

    def _grow(self, seeds: list[tuple[int, int]], dist: list[float], owner: list[int], cost: list[float],
              same_owner_only: bool = False) -> None:
        """Multi-source shortest walk in place.  seeds: (flat index, owner).
        Regions never cross between land and sea."""
        h, w = self.h, self.w
        dom = self._dom_flat
        q: list[tuple[float, int]] = []
        for idx, who in seeds:
            if dist[idx] > 0.0 or owner[idx] < 0:
                dist[idx] = 0.0
                owner[idx] = who
                q.append((0.0, idx))
        heapq.heapify(q)
        diag = math.sqrt(2.0) * self.cell
        ortho = self.cell
        last = h * w - w
        while q:
            d, u = heapq.heappop(q)
            if d > dist[u]:
                continue
            du = dom[u]
            cu = cost[u]
            who = owner[u]
            x = u % w
            left = x > 0 and dom[u - 1] == du
            right = x < w - 1 and dom[u + 1] == du
            up = u >= w and dom[u - w] == du
            down = u < last and dom[u + w] == du
            nbrs = []
            if left:
                nbrs.append((u - 1, ortho))
            if right:
                nbrs.append((u + 1, ortho))
            if up:
                nbrs.append((u - w, ortho))
                if left and dom[u - w - 1] == du:
                    nbrs.append((u - w - 1, diag))
                if right and dom[u - w + 1] == du:
                    nbrs.append((u - w + 1, diag))
            if down:
                nbrs.append((u + w, ortho))
                if left and dom[u + w - 1] == du:
                    nbrs.append((u + w - 1, diag))
                if right and dom[u + w + 1] == du:
                    nbrs.append((u + w + 1, diag))
            for v, step in nbrs:
                if same_owner_only and owner[v] != who:
                    continue
                nd = d + step * 0.5 * (cu + cost[v])
                if nd < dist[v]:
                    dist[v] = nd
                    if not same_owner_only:
                        owner[v] = who
                    heapq.heappush(q, (nd, v))

    def _border_pairs(self, owner: np.ndarray):
        """(u, v) flat indices of 4-neighbour cells with different owners."""
        w = self.w
        idx = np.arange(self.h * w).reshape(self.h, w)
        rows = np.nonzero((owner[:, :-1] != owner[:, 1:]) & (owner[:, :-1] >= 0) & (owner[:, 1:] >= 0))
        cols = np.nonzero((owner[:-1, :] != owner[1:, :]) & (owner[:-1, :] >= 0) & (owner[1:, :] >= 0))
        return (np.concatenate((idx[rows], idx[cols])),
                np.concatenate((idx[rows[0], rows[1] + 1], idx[cols[0] + 1, cols[1]])))

    def _build_regions(self, anchors: list[tuple[float, float]]) -> None:
        n = self.h * self.w
        self._dom_flat = self.domain.ravel().tolist()
        kbot = self.speed["kbot"].ravel()
        # Open sea has no slope to pay for; land costs the time a bot needs.
        walk_cost = np.where(kbot > 0.0, 1.0 / np.maximum(kbot, 0.1), 1.0)
        grow_cost = np.where(self.ramp_id.ravel() >= 0, walk_cost * RAMP_GROW_COST, walk_cost)
        grow_cost = np.where(self.domain.ravel() == SEA, grow_cost * SEA_GROW_COST, grow_cost)
        grow_cost = np.where(self.choke_id.ravel() >= 0, grow_cost * CHOKE_GROW_COST, grow_cost).tolist()
        walk_cost = walk_cost.tolist()
        flat_level = self.level_id.ravel()

        # Spots to cells.  Each usable spot starts as its own seed.
        spot_cells: list[int | None] = []
        for x, z, _inc in self.spots:
            c = self._snap(*self._cell_of(x, z))
            spot_cells.append(None if c is None else c[0] * self.w + c[1])
        live = [i for i, c in enumerate(spot_cells) if c is not None]

        # Single linkage on walking distance: the Voronoi border of the spots
        # holds the shortest link of every cluster, so one sweep is enough.
        parent = list(range(len(self.spots)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def same_altitude(i: int, j: int) -> bool:
            a, b = flat_level[spot_cells[i]], flat_level[spot_cells[j]]
            return a < 0 or b < 0 or a == b or abs(self.levels[a].height - self.levels[b].height) < RAMP_RISE

        if live:
            dist = [math.inf] * n
            owner = [-1] * n
            self._grow([(spot_cells[i], i) for i in live], dist, owner, grow_cost)
            oa = np.array(owner, dtype=np.int32).reshape(self.h, self.w)
            us, vs = self._border_pairs(oa)
            for u, v in zip(us.tolist(), vs.tolist()):
                if (self._dom_flat[u] == self._dom_flat[v] and dist[u] + dist[v] + self.cell <= self.cluster_radius
                        and same_altitude(owner[u], owner[v])):
                    ra, rb = find(owner[u]), find(owner[v])
                    if ra != rb:
                        parent[rb] = ra

        clusters: dict[int, list[int]] = {}
        for i in live:
            clusters.setdefault(find(i), []).append(i)
        groups = list(clusters.values())
        seeds_of: list[list[int]] = [[spot_cells[i] for i in g] for g in groups]
        spots_of: list[list[int]] = groups

        def grow_all() -> tuple[list[float], list[int]]:
            d = [math.inf] * n
            o = [-1] * n
            self._grow([(c, r) for r, cells in enumerate(seeds_of) for c in cells], d, o, grow_cost)
            return d, o

        dist, owner = grow_all()

        # Start positions that no cluster already covers.
        for ax, az in anchors:
            c = self._snap(*self._cell_of(ax, az))
            if c is None:
                continue
            idx = c[0] * self.w + c[1]
            if dist[idx] > self.cluster_radius * 1.15:
                seeds_of.append([idx])
                spots_of.append([])
                self._grow([(idx, len(seeds_of) - 1)], dist, owner, grow_cost)

        # A large level with nothing on it is still a place to hold.
        taken = {int(flat_level[c]) for cells in seeds_of for c in cells}
        first_open = len(seeds_of)
        for lv in sorted(self.levels, key=lambda l: -l.area):
            if lv.lid in taken or lv.area < LEVEL_SEED_ELMOS2 or len(seeds_of) - first_open >= MAX_OPEN_REGIONS // 2:
                continue
            i, j = self._cell_of(lv.x, lv.z)
            idx = i * self.w + j
            seeds_of.append([idx])
            spots_of.append([])
            self._grow([(idx, len(seeds_of) - 1)], dist, owner, grow_cost)

        # Open ground and any water or island with no metal: repeatedly seed
        # the walkable cell farthest from every region, then re-centre those
        # seeds, so open regions don't sit in a corner of their own area.
        dom = self.domain.ravel()
        reach = self.cluster_radius * 1.8
        added: list[int] = []
        walkable = np.nonzero(dom != WALL)[0]
        if walkable.size:
            flat_comp = self._domain_components().ravel()[walkable]
            _ids, inverse, counts = np.unique(flat_comp, return_inverse=True, return_counts=True)
            big = counts[inverse] >= self.min_seed_cells
            while len(seeds_of) - first_open < MAX_OPEN_REGIONS:
                # Unreached cells first (no region on that landmass), then far ones.
                d = np.where(big, np.array(dist, dtype=np.float64)[walkable], -1.0)
                k = int(np.argmax(d))
                if d[k] <= reach:
                    break
                idx = int(walkable[k])
                added.append(len(seeds_of))
                seeds_of.append([idx])
                spots_of.append([])
                self._grow([(idx, len(seeds_of) - 1)], dist, owner, grow_cost)
        if added:
            oa = np.array(owner, dtype=np.int32)
            for r in added:
                cells = np.nonzero(oa == r)[0]
                if cells.size == 0:
                    continue
                zs, xs = np.divmod(cells, self.w)
                cz, cx = zs.mean(), xs.mean()
                seeds_of[r] = [int(cells[int(np.argmin((zs - cz) ** 2 + (xs - cx) ** 2))])]
            dist, owner = grow_all()

        # Small regions with nothing on them only clutter the graph: let their
        # neighbours have the ground.  A whole landmass or sea always keeps one.
        oa = np.array(owner, dtype=np.int32)
        min_cells = MIN_OPEN_REGION_ELMOS2 / (self.cell * self.cell)
        domain_comp = self._domain_components().ravel()
        held = {int(domain_comp[c]) for r, cells in enumerate(seeds_of) if spots_of[r] for c in cells}
        drop = []
        for r in sorted(range(len(seeds_of)), key=lambda q: np.count_nonzero(oa == q), reverse=True):
            if spots_of[r] or not seeds_of[r]:
                continue
            area_id = int(domain_comp[seeds_of[r][0]])
            if np.count_nonzero(oa == r) < min_cells and area_id in held:
                drop.append(r)
            else:
                held.add(area_id)
        if drop:
            for r in drop:
                seeds_of[r] = []
            dist, owner = grow_all()

        # Keep only regions that own something, in a stable order.
        oa = np.array(owner, dtype=np.int32)
        remap: dict[int, int] = {}
        for r in range(len(seeds_of)):
            cells = np.nonzero(oa == r)[0]
            if cells.size:
                remap[r] = len(self.regions)
                zs, xs = np.divmod(cells, self.w)
                if spots_of[r]:
                    sx = sum(self.spots[i][0] for i in spots_of[r]) / len(spots_of[r])
                    sz = sum(self.spots[i][1] for i in spots_of[r]) / len(spots_of[r])
                else:
                    sx, sz = self._centre_of(*divmod(seeds_of[r][0], self.w))
                # The centre must be a cell of the region: a cluster's mean
                # can land on a cliff or in another region.
                pi, pj = self._cell_of(sx, sz)
                k = int(np.argmin((zs - pi) ** 2 + (xs - pj) ** 2))
                cx, cz = self._centre_of(int(zs[k]), int(xs[k]))
                self.regions.append(Region(rid=remap[r], x=cx, z=cz, domain=int(dom[cells[k]]),
                                           spots=list(spots_of[r]),
                                           income=sum(self.spots[i][2] for i in spots_of[r])))
        label = np.full(n, -1, dtype=np.int32)
        for r, new in remap.items():
            label[oa == r] = new
        self.walk_label = label.reshape(self.h, self.w).copy()
        self.label = self._fill_walls(label.reshape(self.h, self.w))

        # Walking cost from each region's centre, inside that region only.
        centre_dist = [math.inf] * n
        walk_owner = self.walk_label.ravel().tolist()
        centres = [self._cell_of(r.x, r.z) for r in self.regions]
        self._grow([(i * self.w + j, r.rid) for (i, j), r in zip(centres, self.regions)],
                   centre_dist, walk_owner, walk_cost, same_owner_only=True)
        self.centre_dist = np.array(centre_dist, dtype=np.float64).reshape(self.h, self.w)

    def _domain_components(self) -> np.ndarray:
        """Connected same-domain areas, walls excluded."""
        out = np.full((self.h, self.w), -1, dtype=np.int32)
        offset = 0
        for d in (LAND, SEA):
            comp, sizes = label_runs(self.domain == d)
            m = comp >= 0
            out[m] = comp[m] + offset
            offset += len(sizes)
        return out

    def _fill_walls(self, label: np.ndarray) -> np.ndarray:
        """Every cell belongs to a region, so a position is always answerable."""
        out = label.copy()
        if not self.regions:
            return out
        q = deque(zip(*np.nonzero(out >= 0)))
        while q:
            i, j = q.popleft()
            r = out[i, j]
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < self.h and 0 <= b < self.w and out[a, b] < 0:
                    out[a, b] = r
                    q.append((a, b))
        return out

    # --------------------------------------------------------------- edges

    def _fine_labels(self, name: str) -> np.ndarray:
        """Region of each full-grid cell this class can stand on.  Cells under a
        work-grid wall take the region they are walkably joined to, not the
        nearest one: the far flank of a thin wall is on the far side."""
        b = self.block
        comp = self.comp[name]
        fine = np.repeat(np.repeat(self.walk_label, b, axis=0), b, axis=1)
        fine[comp < 0] = -1
        h, w = fine.shape
        out = fine.ravel().tolist()
        walkable = (comp >= 0).ravel().tolist()
        todo = (comp >= 0) & (fine < 0)
        near = np.zeros_like(todo)
        near[1:] |= todo[:-1]
        near[:-1] |= todo[1:]
        near[:, 1:] |= todo[:, :-1]
        near[:, :-1] |= todo[:, 1:]
        q = deque(np.nonzero((near & (fine >= 0)).ravel())[0].tolist())
        while q:
            u = q.popleft()
            r = out[u]
            x = u % w
            if x > 0 and walkable[u - 1] and out[u - 1] < 0:
                out[u - 1] = r
                q.append(u - 1)
            if x < w - 1 and walkable[u + 1] and out[u + 1] < 0:
                out[u + 1] = r
                q.append(u + 1)
            if u >= w and walkable[u - w] and out[u - w] < 0:
                out[u - w] = r
                q.append(u - w)
            if u < (h - 1) * w and walkable[u + w] and out[u + w] < 0:
                out[u + w] = r
                q.append(u + w)
        return np.array(out, dtype=np.int16).reshape(h, w)

    def _crossings(self, name: str) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
        """Work-cell pairs on each region border that this class can really
        cross: full-grid neighbours on both sides, same large passable area.
        Keyed by (region, region, area), since a region can hold two areas
        that don't connect (a sea between two beaches, for a bot)."""
        b = self.block
        fine = self.fine_label[name] = self._fine_labels(name)
        comp = self.comp[name]
        big = self.comp_size[name] >= self.min_pocket_fine
        out: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
        w = self.w
        for la, lb, ca, cb, shift in ((fine[:, :-1], fine[:, 1:], comp[:, :-1], comp[:, 1:], (0, 1)),
                                      (fine[:-1, :], fine[1:, :], comp[:-1, :], comp[1:, :], (1, 0))):
            m = (la != lb) & (la >= 0) & (lb >= 0) & (ca == cb)
            if not m.any():
                continue
            m[m] = big[ca[m]]
            zs, xs = np.nonzero(m)
            us = (zs // b) * w + xs // b
            vs = ((zs + shift[0]) // b) * w + (xs + shift[1]) // b
            for a, c, area, u, v in zip(la[m].tolist(), lb[m].tolist(), ca[m].tolist(), us.tolist(), vs.tolist()):
                if a > c:
                    a, c, u, v = c, a, v, u
                out.setdefault((a, c, area), set()).add((u, v))
        return {k: sorted(v) for k, v in out.items()}

    def _build_edges(self) -> None:
        if len(self.regions) < 2:
            return
        flat_centre = self.centre_dist.ravel()
        flat_dom = self.domain.ravel()
        for name in MOVE_CLASSES:
            for (a, b, area), pairs in self._crossings(name).items():
                gates = self._split_gates(a, b, area, pairs, flat_centre, flat_dom)
                edge = self.edges.get((a, b))
                if edge is None:
                    edge = self.edges[(a, b)] = Edge(a, b)
                    self.regions[a].neighbors.append(b)
                    self.regions[b].neighbors.append(a)
                edge.gates.setdefault(name, []).extend(gates)
        # A gate on a choke is one.  So is a way between regions too narrow for
        # the grid to see as a choke, if going around is long; regions touching
        # at a corner with a third beside them are not.
        for e in self.edges.values():
            for name, gates in e.gates.items():
                for g in gates:
                    if g.choke is not None:
                        g.tags.add("choke")
                    elif e.width(name) < CHOKE_WIDTH and self._graph_cost(
                            e.a, e.b, name, g.area, skip=(e.a, e.b), limit=g.length * BORDER_DETOUR) is None:
                        g.tags.add("choke")

    def _graph_cost(self, a: int, b: int, cls: str, area: int, skip: tuple[int, int] | None = None,
                    limit: float = math.inf) -> float | None:
        """Cheapest way between two regions for one passable area, or None."""
        best = {a: 0.0}
        q = [(0.0, a)]
        while q:
            d, u = heapq.heappop(q)
            if u == b:
                return d
            if d > best.get(u, math.inf):
                continue
            for v in self.regions[u].neighbors:
                key = (min(u, v), max(u, v))
                if key == skip:
                    continue
                gate = self.edges[key].best_gate(cls, area)
                if gate is None:
                    continue
                nd = d + gate.length
                if nd <= limit and nd < best.get(v, math.inf):
                    best[v] = nd
                    heapq.heappush(q, (nd, v))
        return None

    def _split_gates(self, a: int, b: int, area: int, pairs, flat_centre, flat_dom) -> list[Gate]:
        w = self.w
        mids = {}
        for u, v in pairs:
            zu, xu = divmod(u, w)
            zv, xv = divmod(v, w)
            mids[(zu + zv, xu + xv)] = (u, v)    # doubled coordinates of the midpoint
        seen = set()
        ra, rb = self.regions[a], self.regions[b]
        flat_ramp = self.ramp_id.ravel()
        flat_choke = self.choke_id.ravel()
        out = []
        for start in mids:
            if start in seen:
                continue
            seen.add(start)
            group = []
            stack = [start]
            while stack:
                m = stack.pop()
                group.append(m)
                for dz in (-2, -1, 0, 1, 2):
                    for dx in (-2, -1, 0, 1, 2):
                        k = (m[0] + dz, m[1] + dx)
                        if k in mids and k not in seen:
                            seen.add(k)
                            stack.append(k)
            zs = [m[0] for m in group]
            xs = [m[1] for m in group]
            # Halved midpoints are in cell coordinates (+0.5 for the cell centre).
            width = max(self.cell, math.hypot(max(xs) - min(xs), max(zs) - min(zs)) * 0.5 * self.cell)
            mz, mx = sum(zs) / len(zs), sum(xs) / len(xs)
            near = min(group, key=lambda m: (m[0] - mz) ** 2 + (m[1] - mx) ** 2)
            gx, gz = (near[1] * 0.5 + 0.5) * self.cell, (near[0] * 0.5 + 0.5) * self.cell
            best = math.inf
            ramp = choke = None
            for m in group:
                u, v = mids[m]
                best = min(best, float(flat_centre[u] + flat_centre[v]) + self.cell)
                for c in (u, v):
                    if flat_ramp[c] >= 0:
                        ramp = int(flat_ramp[c])
                    if flat_choke[c] >= 0:
                        choke = int(flat_choke[c])
            if not math.isfinite(best):
                # Crossing through ground a bot can't walk: fall back to straight lines.
                best = math.hypot(gx - ra.x, gz - ra.z) + math.hypot(gx - rb.x, gz - rb.z)
            tags = set()
            u, v = mids[near]
            if flat_dom[u] != flat_dom[v] and WALL not in (flat_dom[u], flat_dom[v]):
                tags.add("shore")
            if ramp is not None:
                tags.add("ramp")
            out.append(Gate(gx, gz, width, best, area, ramp, choke, tags))
        return out

    # -------------------------------------------------------------- labels

    def _label_regions(self) -> None:
        if not self.regions:
            return
        b = self.block
        kcomp = self.comp["kbot"][::b, ::b][:self.h, :self.w]
        main_land = int(np.argmax(self.comp_size["kbot"])) if len(self.comp_size["kbot"]) else -1
        all_land = self.height[self.domain == LAND]
        relief = float(np.percentile(all_land, 95) - np.percentile(all_land, 5)) if all_land.size else 0.0
        rel_limit = max(30.0, relief * 0.12)
        for r in self.regions:
            own = self.label == r.rid
            walk = self.walk_label == r.rid
            r.cells = int(own.sum())
            r.area = r.cells * self.cell * self.cell
            ground = own & ~self.burning
            r.wall_share = float((self.domain[ground] == WALL).mean()) if ground.any() else 0.0
            r.water_share = float(self.water[ground].mean()) if ground.any() else 0.0
            r.height = float(self.height[walk].mean()) if walk.any() else float(self.height[own].mean())
            levels = self.level_id[walk]
            levels = levels[levels >= 0]
            if levels.size:
                r.level = int(np.bincount(levels).argmax())
            for name in MOVE_CLASSES:
                r.access[name] = float((self.speed[name][own] > 0.0).mean()) if r.cells else 0.0
        for ramp in self.ramps:
            i, j = self._cell_of(self.levels[ramp.low].x, self.levels[ramp.low].z)
            ramp.low_region = int(self.label[i, j])
            i, j = self._cell_of(self.levels[ramp.high].x, self.levels[ramp.high].z)
            ramp.high_region = int(self.label[i, j])
        # Height against the neighbours, weighted by how much border they share.
        weight: dict[int, list[tuple[int, float]]] = {r.rid: [] for r in self.regions}
        for (a, c), e in self.edges.items():
            w = max(e.width(k) for k in e.gates)
            weight[a].append((c, w))
            weight[c].append((a, w))
        for r in self.regions:
            if r.domain == SEA:
                r.tags.add("sea")
                continue
            if r.water_share >= 0.3:
                r.tags.add("shallow")
            # Only neighbours at another height say anything about this one;
            # those on the same level would average a plateau away.
            lands = [(self.regions[o].height - r.height, w) for o, w in weight[r.rid] if self.regions[o].domain == LAND]
            other = [(d, w) for d, w in lands if abs(d) >= rel_limit]
            if other:
                r.rel_height = -sum(d * w for d, w in other) / max(1.0, sum(w for _d, w in other))
                lower, higher = any(d < 0 for d, _w in other), any(d > 0 for d, _w in other)
                if lower and not higher:
                    r.tags.add("plateau")
                elif higher and not lower:
                    r.tags.add("basin")
                else:
                    r.tags.add("terrace")
            walk = self.walk_label == r.rid
            if walk.any():
                # Tanks climb half the slope bots do, so their share tells
                # open ground from broken ground.
                tank_share = float((self.speed["tank"][walk] > 0.0).mean())
                if tank_share >= 0.85:
                    r.tags.add("flat")
                elif tank_share < 0.6:
                    r.tags.add("rough")
                comps = kcomp[walk]
                comps = comps[comps >= 0]
                if comps.size and int(np.bincount(comps).argmax()) != main_land:
                    r.tags.add("island")
            for o, _w in weight[r.rid]:
                if self.regions[o].domain == SEA:
                    r.tags.add("coast")
                    break

    # ------------------------------------------------------------- queries

    def region_at(self, x: float, z: float) -> Region | None:
        if not self.regions:
            return None
        return self.regions[int(self.label[self._cell_of(x, z)])]

    def hazard_at(self, x: float, z: float) -> bool:
        """Is this point in or right beside lava or void?"""
        i = min(self.hazard.shape[0] - 1, max(0, int(z / SLOPE_CELL)))
        j = min(self.hazard.shape[1] - 1, max(0, int(x / SLOPE_CELL)))
        return bool(self.hazard[i, j])

    def region_for(self, cls: str, x: float, z: float) -> int | None:
        """Region a unit of this class is walkably in at a passable point: on
        the flank of a thin wall this can differ from region_at.  None on a
        strip no region reaches."""
        i = min(self.comp[cls].shape[0] - 1, max(0, int(z / SLOPE_CELL)))
        j = min(self.comp[cls].shape[1] - 1, max(0, int(x / SLOPE_CELL)))
        rid = int(self.fine_label[cls][i, j]) if cls in self.fine_label else -1
        return rid if rid >= 0 else None

    def can_reach(self, cls: str, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Can this movement class walk from a to b?  Positions on impassable
        ground use the nearest passable area, as units standing there get out."""
        ca = self.area_of(cls, *a)
        cb = self.area_of(cls, *b)
        return ca is not None and ca[0] == cb[0] if cb is not None else False

    def area_of(self, cls: str, x: float, z: float, radius: int = 16) -> tuple[int, float, float] | None:
        """(passable area, x, z of its nearest cell) for this class, or None."""
        comp = self.comp[cls]
        sizes = self.comp_size[cls]
        h, w = comp.shape
        i = min(h - 1, max(0, int(z / SLOPE_CELL)))
        j = min(w - 1, max(0, int(x / SLOPE_CELL)))
        i0, j0 = max(0, i - radius), max(0, j - radius)
        win = comp[i0:min(h, i + radius + 1), j0:min(w, j + radius + 1)]
        zs, xs = np.nonzero(win >= 0)
        if zs.size == 0:
            return None
        ids = win[zs, xs]
        keep = sizes[ids] >= self.min_pocket_fine
        if not keep.any():
            keep[:] = True
        kz, kx = zs[keep], xs[keep]
        k = int(np.argmin((kz + i0 - i) ** 2 + (kx + j0 - j) ** 2))
        return int(ids[keep][k]), (int(kx[k]) + j0 + 0.5) * SLOPE_CELL, (int(kz[k]) + i0 + 0.5) * SLOPE_CELL

    def route(self, a: tuple[float, float], b: tuple[float, float], cls: str = "kbot") -> tuple[float, list[Gate]] | None:
        """(walking cost, gates in order) from a to b, or None if the region
        graph has no way for this class."""
        pa, pb = self.area_of(cls, *a), self.area_of(cls, *b)
        if pa is None or pb is None or pa[0] != pb[0]:
            return None
        area = pa[0]
        # Regions of where the unit will actually be, for a start on bad ground.
        # One area is labelled throughout or not at all, so both or neither.
        ra, rb = self.region_for(cls, pa[1], pa[2]), self.region_for(cls, pb[1], pb[2])
        if ra == rb or ra is None or rb is None:
            return math.hypot(a[0] - b[0], a[1] - b[1]), []
        best = {ra: 0.0}
        prev: dict[int, tuple[int, Gate]] = {}
        q = [(0.0, ra)]
        while q:
            d, u = heapq.heappop(q)
            if u == rb:
                break
            if d > best.get(u, math.inf):
                continue
            for v in self.regions[u].neighbors:
                gate = self.edges[(min(u, v), max(u, v))].best_gate(cls, area)
                if gate is None:
                    continue
                nd = d + gate.length
                if nd < best.get(v, math.inf):
                    best[v] = nd
                    prev[v] = (u, gate)
                    heapq.heappush(q, (nd, v))
        if rb not in best:
            return None
        gates: list[Gate] = []
        n = rb
        while n != ra:
            n, g = prev[n]
            gates.append(g)
        gates.reverse()
        return best[rb], gates

    def summary(self) -> dict:
        return {
            "cell": self.cell,
            "regions": len(self.regions),
            "land": sum(1 for r in self.regions if r.domain == LAND),
            "sea": sum(1 for r in self.regions if r.domain == SEA),
            "edges": len(self.edges),
            "kbot_edges": sum(1 for e in self.edges.values() if "kbot" in e.gates),
            "chokes": len(self.chokes),
            "levels": len(self.levels),
            "ramps": len(self.ramps),
            "walls": float(((self.domain == WALL) & ~self.burning).mean()),
            "water": float(self.water.mean()),
            "liquid": self.liquid,
        }
