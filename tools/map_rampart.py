"""Violet Rampart Generator v1.4's terrain, from a seed and a number of bases: the map's own generator, ported from its Lua.

The map (Rafal[ZK], 2021) is a free-for-all for 3 to 11 players that makes its ground when a game starts, in
LuaRules/Gadgets/mapgen_violet_rampart_terrain.lua and TerrainGenerator/TerrainShapes/*.lua: a base for each player, a walled
spade on a handle, round a ring of lanes on a plateau 300 high, with a way down into a central arena; acid below everything else
at -200.  The archive's own height map is only a placeholder for this, so a pool map read from the archive isn't the ground a game
is played on.

Everything here follows the Lua step by step, with the same arithmetic in the same order, so that the same random draws make the
same map.  `LuaRandom` is the generator both draw from when they're compared (tools/map_gen.py's notes): the engine has its own, so
a game's seed makes some other of these maps; and the engine runs Lua in 32-bit floats, so a height-map corner right on an edge
may fall the other way in a game.  Left out: the type map, whose three terrain types all keep every speed (mapinfo.lua), and the
textures.  Metal spots are what map_metal_layout.lua hands to the game (`GG.mapgen_mexList`), at the incomes it gives them.
Nothing the map places can be reclaimed: its crystals and geothermal vents are decoration (feature_placer.lua, features/crystal.lua).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAP_SIZE = 12288.0                  # Game.mapSizeX and mapSizeZ: 24 x 24
SQUARE = 8.0                        # Game.squareSize: a height-map corner each
MAP_SQUARE = 1024.0                 # the Lua's MAP_SQUARE_SIZE, the blocks it tests a shape against before its corners
SQUARES = int(MAP_SIZE / MAP_SQUARE)
CORNERS = int(MAP_SIZE / SQUARE) + 1
HUGE = 1e6                          # DISTANCE_HUGE
EPSILON = 0.001                     # INTERSECTION_EPSILON
WATER_DAMAGE = 200.0                # mapinfo.lua, a second.  water_damage_by_footprint.lua scales it by a unit's footprint area / 16.
NAME = "Violet Rampart Generator v1.4"
# Elmos walking between spots of one terrain region (terrain.TerrainMap): under the 458 that its nearest spots are apart, so that
# each is a region, as on most maps (the pool's median is 1.2 spots a region).  The default for its size, 680, makes the evenly
# spaced spots one chain over the whole plateau: two regions in all, and five route segments a class.
CLUSTER_RADIUS = 420.0

# Geometry (the Lua's CONFIG).
LANE_MIN_FROM_CENTER = 900.0        # the limit for 3 bases
LANE_END_MIN_FROM_CENTER = 1600.0   # for 4 or 5
LANE_MIN_LENGTH = 1900.0            # from 6
LANE_WIDTH = 900.0
CENTER_WIDTH_MIN = 790.0
CENTER_WIDTH_FACTOR = 0.30
ENTRANCE_SHARE = 1.0 / 3.0
RAMP_SLOPE_DEGREES = 1.5 * 18       # tanks' steepest
RAMP_LENGTH_MULT = 1.20
FLAT_MIN_WIDTH = 64.0
LANE_MEX_OFFSET = 0.0
LANE_GEO_OFFSET = 0.25 * LANE_WIDTH
SPADE_MIN_ANGLE_DEGREES = 7.5
HANDLE_WIDTH, HANDLE_HEIGHT = 550.0, 1100.0
SPADE_WIDTH, SPADE_HEIGHT = 1200.0, 800.0
SPADE_PATH_RADIUS = 350.0
SPADE_VISUAL_OFFSET = 120.0
WALL_INNER = 8.0 + 4.0              # RAMPART_WALL_INNER_TEXTURE_WIDTH
WALL_TOTAL = WALL_INNER + 48.0
WALL_OUTER_TOTAL = WALL_TOTAL + 8.0
BOTTOM, CENTER_LOW, RAMPART, WALL_TOP, WALL_FOOT = -200.0, 20.0, 300.0, 370.0, 1.0
SPADE_MEXES, HANDLE_MEXES, LANE_MEXES = 5, 2, 3
SPADE_METAL, HANDLE_METAL, ENTRANCE_METAL, LANE_METAL = 2.0, 1.5, 3.0, 1.5
MIN_BASES_FOR_CENTER_GEO = 6
START_BOX_PADDING = 32.0
MOST_BASES = 11


class LuaRandom:
    """Park and Miller's generator, with Lua 5.1's math.random(), math.random(m, n) and math.randomseed: exact in doubles, so the
    Lua and this draw the same numbers.  Its first draws from a small seed are nearly multiples of the seed, so seeds should be
    large, as the map's own are (1 to 1,000,000)."""
    MODULUS, MULTIPLIER = 2147483647, 16807

    def __init__(self, seed: int):
        self.state = int(seed) % self.MODULUS or 1

    def random(self) -> float:
        self.state = self.state * self.MULTIPLIER % self.MODULUS
        return self.state / self.MODULUS

    def randint(self, m: int, n: int) -> int:
        return int(math.floor(self.random() * (n - m + 1))) + m


# Points and paths ------------------------------------------------------------------------------------------------------------------


def _unit(x: float, y: float) -> tuple[float, float]:
    mult = 1.0 / math.sqrt(x * x + y * y)
    return x * mult, y * mult


def _distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    dx, dy = q[0] - p[0], q[1] - p[1]
    return math.sqrt(dx * dx + dy * dy)


def _across(p, v, x, y):
    """LineCoordsDistance: how far from the line through p along v."""
    return np.abs(-v[0] * (y - p[1]) + v[1] * (x - p[0]))


def _along(p, v, x, y):
    """LineCoordsProjection: how far along v from p, either way."""
    return v[0] * (x - p[0]) + v[1] * (y - p[1])


def _reach(v: tuple[float, float], half: float) -> float:
    """How far a square of this half size reaches along v from its centre."""
    return abs(v[0] * half + v[1] * (half if v[0] * v[1] >= 0 else -half))


class Rotation:
    def __init__(self, cx: float, cy: float, angle: float):
        self.cx, self.cy, self.sin, self.cos = cx, cy, math.sin(angle), math.cos(angle)

    def __call__(self, p: tuple[float, float]) -> tuple[float, float]:
        dx, dy = p[0] - self.cx, p[1] - self.cy
        return self.cx + dx * self.cos - dy * self.sin, self.cy + dx * self.sin + dy * self.cos


class Line:
    def __init__(self, p1, p2):
        self.p1, self.length, self.front = p1, _distance(p1, p2), _unit(p2[0] - p1[0], p2[1] - p1[1])

    def at(self, advance: float) -> tuple[float, float]:
        return self.p1[0] + self.front[0] * advance, self.p1[1] + self.front[1] * advance


class Arc:
    def __init__(self, center, radius: float, start: float, span: float):
        self.center, self.radius, self.start, self.span, self.length = center, radius, start, span, span * radius

    def at(self, advance: float) -> tuple[float, float]:
        angle = self.start + (advance / self.length) * self.span
        return self.center[0] + self.radius * math.sin(angle), self.center[1] - self.radius * math.cos(angle)


class Path:
    def __init__(self, segments: list):
        self.segments, self.total = segments, 0.0
        for s in segments:
            self.total = self.total + s.length

    def at(self, advance: float) -> tuple[float, float]:
        if advance <= 0:
            return self.segments[0].at(0)
        if advance < self.total:
            for s in self.segments:
                if advance <= s.length:
                    return s.at(advance)
                advance = advance - s.length
        return self.segments[-1].at(self.segments[-1].length)

    def relative(self, share: float) -> tuple[float, float]:
        return self.at(share * self.total)

    def points(self, inner: int, ends: bool) -> list[tuple[float, float]]:
        out = [self.at(0)] if ends else []
        if inner >= 1:
            step = self.total / (inner + 1)
            out += [self.at(i * step) for i in range(1, inner + 1)]
        return out + ([self.at(self.total)] if ends else [])


# Shapes: each says the box it may touch, whether it may touch a map square, and what it does to the heights it covers ---------------


def _walled(h: np.ndarray, d: np.ndarray):
    """modifyHeightMapForWalledShape, given how far outside the shape's border each corner is."""
    covered = d <= WALL_OUTER_TOTAL
    edge = covered & (d > 0)
    new = h.copy()
    new[d <= 0] = RAMPART
    new[edge & (d < WALL_INNER)] = RAMPART      # the Lua sets it if it's above or below, so always
    new[edge & (d >= WALL_INNER) & (d <= WALL_TOTAL) & (h != RAMPART)] = WALL_TOP
    new[edge & (d > WALL_TOTAL) & (h < WALL_FOOT)] = WALL_FOOT
    return covered, new


def _onto(h: np.ndarray, inside: np.ndarray, ground):
    """modifyHeightMapForFlatShape and ...RampShape: up to the ground, or down to it from above the rampart."""
    return inside, np.where(inside & ((h < ground) | (RAMPART < h)), ground, h)


class _Rectangle:
    def __init__(self, p1, p2, width: float):
        self.p1, self.p2, self.width = p1, p2, width
        self.front = _unit(p2[0] - p1[0], p2[1] - p1[1])
        self.right = (-self.front[1], self.front[0])
        self.center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.half_w, self.half_h = width / 2, _distance(p1, p2) / 2

    def rotated(self, rot: Rotation):
        return type(self)(rot(self.p1), rot(self.p2), self.width)

    def _box(self, across: float, along: float):
        w, h = self.half_w + across, self.half_h + along
        rx = abs(self.front[0]) * h + abs(self.right[0]) * w
        ry = abs(self.front[1]) * h + abs(self.right[1]) * w
        return self.center[0] - rx, self.center[1] - ry, self.center[0] + rx, self.center[1] + ry

    def _hits(self, sx: int, sz: int, across: float, along: float) -> bool:
        x, y = (sx - 0.5) * MAP_SQUARE, (sz - 0.5) * MAP_SQUARE
        on_front, on_right = _along(self.center, self.front, x, y), _along(self.center, self.right, x, y)
        reach = _reach(self.front, MAP_SQUARE / 2)
        w, h = self.half_w + across + EPSILON, self.half_h + along + EPSILON
        return on_right - reach <= w and on_right + reach >= -w and on_front - reach <= h and on_front + reach >= -h


class WalledRectangle(_Rectangle):
    """RampartFullyWalledRectangle: the spade."""

    def box(self):
        return self._box(WALL_OUTER_TOTAL, WALL_OUTER_TOTAL)

    def hits(self, sx: int, sz: int) -> bool:
        return self._hits(sx, sz, WALL_OUTER_TOTAL, WALL_OUTER_TOTAL)

    def draw(self, h, x, y):
        return _walled(h, np.maximum(_across(self.center, self.front, x, y) - self.half_w, _across(self.center, self.right, x, y) - self.half_h))


class SideWalledRectangle(_Rectangle):
    """RampartVerticallyWalledRectangle: walls along its length only, open at its ends.  A handle, and the way into the arena."""

    def box(self):
        return self._box(WALL_OUTER_TOTAL, 0.0)

    def hits(self, sx: int, sz: int) -> bool:
        return self._hits(sx, sz, WALL_OUTER_TOTAL, 0.0)

    def draw(self, h, x, y):
        d = np.where(_across(self.center, self.right, x, y) <= self.half_h, _across(self.center, self.front, x, y) - self.half_w, HUGE)
        return _walled(h, d)


class _Circle:
    def __init__(self, center, radius: float, ground: float = RAMPART):
        self.center, self.radius, self.ground = center, radius, ground

    def rotated(self, rot: Rotation):
        return type(self)(rot(self.center), self.radius, self.ground)

    def _box(self, border: float):
        r = self.radius + border
        return self.center[0] - r, self.center[1] - r, self.center[0] + r, self.center[1] + r

    def _hits(self, sx: int, sz: int, border: float) -> bool:
        half = MAP_SQUARE / 2
        dx = max(0, abs(self.center[0] - (sx - 0.5) * MAP_SQUARE) - half)
        dy = max(0, abs(self.center[1] - (sz - 0.5) * MAP_SQUARE) - half)
        r = self.radius + border + EPSILON
        return dx * dx + dy * dy <= r * r

    def _from_centre(self, x, y):
        dx, dy = x - self.center[0], y - self.center[1]
        return np.sqrt(dx * dx + dy * dy)


class WalledCircle(_Circle):
    """RampartWalledCircle: the spade's round end."""

    def box(self):
        return self._box(WALL_OUTER_TOTAL)

    def hits(self, sx: int, sz: int) -> bool:
        return self._hits(sx, sz, WALL_OUTER_TOTAL)

    def draw(self, h, x, y):
        return _walled(h, self._from_centre(x, y) - self.radius)


class FlatCircle(_Circle):
    """RampartFlatCircle: the arena's floor."""

    def box(self):
        return self._box(0.0)

    def hits(self, sx: int, sz: int) -> bool:
        return self._hits(sx, sz, 0.0)

    def draw(self, h, x, y):
        return _onto(h, self._from_centre(x, y) <= self.radius, self.ground)


class _Trapezoid:
    """Between two parallel edges, width1 across at p1 and width2 at p2."""

    def __init__(self, p1, p2, width1: float, width2: float, ground: float = RAMPART, ground2: float = 0.0):
        self.p1, self.p2, self.width1, self.width2, self.ground, self.ground2 = p1, p2, width1, width2, ground, ground2
        self.front = _unit(p2[0] - p1[0], p2[1] - p1[1])
        self.right = (-self.front[1], self.front[0])
        self.center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        self.height = _distance(p1, p2)
        half1, half2 = width1 / 2, width2 / 2
        self.half_h = self.height / 2
        self.half_w = (half1 + half2) / 2
        self.widening = (half2 - half1) / self.height
        self.border_mult = math.sqrt(1.0 + self.widening * self.widening)
        f, r, c = self.front, self.right, self.center
        self.right_point = (c[0] + self.half_w * r[0], c[1] + self.half_w * r[1])
        self.left_point = (c[0] - self.half_w * r[0], c[1] - self.half_w * r[1])
        n = _unit(f[0] + self.widening * r[0], f[1] + self.widening * r[1])
        self.right_normal = (-n[1], n[0])
        n = _unit(f[0] - self.widening * r[0], f[1] - self.widening * r[1])
        self.left_normal = (n[1], -n[0])

    def rotated(self, rot: Rotation):
        return type(self)(rot(self.p1), rot(self.p2), self.width1, self.width2, self.ground, self.ground2)

    def box(self, across: float = 0.0, along: float = 0.0):
        h = self.half_h + along
        w = self.half_w + across * self.border_mult
        w1, w2 = w - h * self.widening, w + h * self.widening
        fx, fy = self.front[0] * h, self.front[1] * h
        x1, x2 = abs(self.right[0]) * w1, abs(self.right[0]) * w2
        y1, y2 = abs(self.right[1]) * w1, abs(self.right[1]) * w2
        c = self.center
        return (c[0] + min(-fx - x1, fx - x2), c[1] + min(-fy - y1, fy - y2),
                c[0] + max(-fx + x1, fx + x2), c[1] + max(-fy + y1, fy + y2))

    def hits(self, sx: int, sz: int, across: float = 0.0, along: float = 0.0) -> bool:
        x, y = (sx - 0.5) * MAP_SQUARE, (sz - 0.5) * MAP_SQUARE
        half = MAP_SQUARE / 2
        on_front = _along(self.center, self.front, x, y)
        on_right = _along(self.right_point, self.right_normal, x, y)
        on_left = _along(self.left_point, self.left_normal, x, y)
        h, w = self.half_h + along + EPSILON, across + EPSILON
        reach = _reach(self.front, half)
        return (on_front - reach <= h and on_front + reach >= -h and on_right - _reach(self.right_normal, half) <= w
                and on_left - _reach(self.left_normal, half) <= w)

    def _parts(self, x, y):
        return _across(self.center, self.front, x, y), _along(self.center, self.front, x, y)

    def _inside(self, x, y):
        off, on = self._parts(x, y)
        return (np.abs(on) <= self.half_h) & (off <= self.half_w + on * self.widening), on


class WalledTrapezoid(_Trapezoid):
    """RampartHorizontallyWalledTrapezoid: walls on its two parallel edges, open at its sides.  The lanes."""

    def box(self):
        return super().box(0.0, WALL_OUTER_TOTAL)

    def hits(self, sx: int, sz: int) -> bool:
        return super().hits(sx, sz, 0.0, WALL_OUTER_TOTAL)

    def draw(self, h, x, y):
        off, on = self._parts(x, y)
        return _walled(h, np.where(off <= self.half_w + on * self.widening, np.abs(on) - self.half_h, HUGE))


class WallTrapezoid(_Trapezoid):
    """RampartInternalWallTrapezoid without its outer texture: the arena's wall."""

    def draw(self, h, x, y):
        inside, _ = self._inside(x, y)
        return inside, np.where(inside, WALL_TOP, h)


class FlatTrapezoid(_Trapezoid):
    """RampartFlatTrapezoid: the step in front of the arena's wall."""

    def draw(self, h, x, y):
        return _onto(h, self._inside(x, y)[0], self.ground)


class RampTrapezoid(_Trapezoid):
    """RampartRampTrapezoid: from `ground` at p1 to `ground2` at p2.  Down into the arena."""

    def draw(self, h, x, y):
        inside, on = self._inside(x, y)
        return _onto(h, inside, (self.ground + self.ground2) / 2 + on * ((self.ground2 - self.ground) / self.height))


def _blocks(square: int) -> tuple[float, float]:
    """The corners a map square holds: the first holds corner 0 too."""
    return (0.0 if square == 1 else (square - 1) * MAP_SQUARE + SQUARE), square * MAP_SQUARE


def _draw(h: np.ndarray, shape) -> None:
    """GenerateHeightMapForShape: the corners of the shape's box, in the map squares it may touch, column by column."""
    x1, y1, x2, y2 = shape.box()
    bx1, by1 = max(0.0, math.ceil(x1 / SQUARE) * SQUARE), max(0.0, math.ceil(y1 / SQUARE) * SQUARE)
    bx2, by2 = min(MAP_SIZE, math.floor(x2 / SQUARE) * SQUARE), min(MAP_SIZE, math.floor(y2 / SQUARE) * SQUARE)
    sx1, sy1 = max(1, math.ceil(x1 / MAP_SQUARE)), max(1, math.ceil(y1 / MAP_SQUARE))
    sx2, sy2 = min(SQUARES, math.ceil(x2 / MAP_SQUARE)), min(SQUARES, math.ceil(y2 / MAP_SQUARE))
    for sx in range(sx1, sx2 + 1):
        hit = [sz for sz in range(sy1, sy2 + 1) if shape.hits(sx, sz)]
        if not hit:
            continue
        cx1, cx2 = _blocks(sx)
        ax, bx = max(cx1, bx1), min(bx2, cx2)
        az, bz = max(_blocks(hit[0])[0], by1), min(by2, _blocks(hit[-1])[1])
        if ax > bx or az > bz:
            continue
        i0, i1, j0, j1 = int(ax / SQUARE), int(bx / SQUARE), int(az / SQUARE), int(bz / SQUARE)
        x, y = np.meshgrid(np.arange(i0, i1 + 1) * SQUARE, np.arange(j0, j1 + 1) * SQUARE)
        part = h[j0:j1 + 1, i0:i1 + 1]
        covered, new = shape.draw(part, x, y)
        # The Lua leaves a column at the first corner outside after one inside.  The shapes are convex, so that skips nothing,
        # but it costs nothing to do the same.
        left = np.cumsum((np.cumsum(covered, 0) > 0) & ~covered, 0) > 0
        done = covered & ~left
        part[done] = new[done]


# The map --------------------------------------------------------------------------------------------------------------------------


@dataclass
class Rampart:
    bases: int
    seed: int
    corners: np.ndarray                             # heights at the height map's corners, [z, x]
    spots: list[tuple[float, float, float]]          # x, z, metal a second
    geos: list[tuple[float, float]]
    boxes: list[tuple[list[tuple[float, float]], tuple[float, float]]]     # each base's start box and start point, in base order
    initial_angle: float                            # how far the first base is turned round the centre
    spade_angle: float                              # how far each spade is turned on its handle


def _spade_angle(rng, span: float) -> float:
    angle = (-0.5 + rng.random()) * span
    return 0.0 if abs(angle) < math.radians(SPADE_MIN_ANGLE_DEGREES) else angle


def _offset(rng, p, most: float, v) -> tuple[float, float]:
    offset = (-1.0 + 2.0 * rng.random()) * most
    return p[0] + offset * v[0], p[1] + offset * v[1]


def _base(rng, bases: int, turn: float):
    """GenerateGeometryForSingleBase: one base's shapes, spots, geos and start box, at 12 o'clock.  And the arena's floor."""
    cx = cy = MAP_SIZE / 2
    half = turn / 2
    lane_end = max(LANE_MIN_FROM_CENTER / math.cos(half), LANE_END_MIN_FROM_CENTER, (LANE_MIN_LENGTH / 2) / math.sin(half))
    handle_y = cy - lane_end - ((LANE_WIDTH - HANDLE_WIDTH) / 2) / math.cos(half)
    anchor = (cx, handle_y)
    spade_angle = _spade_angle(rng, min(turn, math.radians(180) - turn))
    spade = Rotation(cx, handle_y, spade_angle)
    neck, end = handle_y - HANDLE_HEIGHT, handle_y - HANDLE_HEIGHT - SPADE_HEIGHT
    shapes = [SideWalledRectangle(anchor, spade((cx, neck)), HANDLE_WIDTH),
              WalledRectangle(spade((cx, neck)), spade((cx, end)), SPADE_WIDTH),
              WalledCircle(spade((cx, end)), SPADE_WIDTH / 2)]
    unique = []
    lane_start = (cx, cy - lane_end)
    lane_stop = Rotation(cx, cy, turn)(lane_start)
    lane = Rotation(cx, cy, turn / 2)
    lane_mid = lane_end * math.cos(half)
    outer = lane_mid + LANE_WIDTH / 2
    inner = lane_mid - LANE_WIDTH / 2
    centre_w = inner - (WALL_TOTAL + WALL_INNER)
    widen = max(0, centre_w - CENTER_WIDTH_MIN) * CENTER_WIDTH_FACTOR
    inner = lane_mid - (LANE_WIDTH / 2 + widen)
    centre_w = inner - (WALL_TOTAL + WALL_INNER)
    outer_edge, inner_edge = 2 * (outer * math.tan(half)), 2 * (inner * math.tan(half))
    centre_edge = 2 * (centre_w * math.tan(half))
    lane_shape = WalledTrapezoid(lane((cx, cy - outer)), lane((cx, cy - inner)), outer_edge + EPSILON, inner_edge + EPSILON)
    lane_right = lane_shape.front      # (sic): what the Lua offsets lane spots along
    shapes.append(lane_shape)
    ramp_mult = (1.0 / math.tan(math.radians(RAMP_SLOPE_DEGREES))) * RAMP_LENGTH_MULT
    ramp_length = min((RAMPART - CENTER_LOW) * ramp_mult, max(0, centre_w - FLAT_MIN_WIDTH))
    low = RAMPART - math.floor(ramp_length / ramp_mult)
    floor_w = centre_w - ramp_length
    if floor_w > 0:
        unique.append(FlatCircle((cx, cy), floor_w / math.cos(half), low))
    if ramp_length > 0:
        shapes.append(RampTrapezoid(lane((cx, cy - centre_w)), lane((cx, cy - floor_w)), centre_edge + EPSILON,
                                    2 * (floor_w * math.tan(half)) + EPSILON, RAMPART, low))
    entrance_start, entrance_end = cy - inner, cy - centre_w
    wall_start, wall_end = entrance_start + WALL_INNER, entrance_end - WALL_INNER
    inner_wall = 2 * ((centre_w + WALL_INNER) * math.tan(half))
    outer_wall = 2 * ((centre_w + WALL_TOTAL) * math.tan(half))
    entrance_w = inner_wall * ENTRANCE_SHARE - 2 * WALL_INNER
    shapes += [FlatTrapezoid(lane((cx, wall_end)), lane((cx, entrance_end)), inner_wall + EPSILON, centre_edge + EPSILON),
               WallTrapezoid(lane((cx, wall_start)), lane((cx, wall_end)), outer_wall + EPSILON, inner_wall + EPSILON),
               SideWalledRectangle(lane((cx, entrance_start)), lane((cx, entrance_end)), entrance_w)]
    # Resources, along paths round the spade, up the handle and along the lane.
    path_y = handle_y - HANDLE_HEIGHT - (SPADE_WIDTH / 2 - SPADE_PATH_RADIUS)
    spade_path = Path([Line(spade((cx - SPADE_PATH_RADIUS, path_y)), spade((cx - SPADE_PATH_RADIUS, end))),
                       Arc(spade((cx, end)), SPADE_PATH_RADIUS, spade_angle - math.radians(90), math.radians(180)),
                       Line(spade((cx + SPADE_PATH_RADIUS, end)), spade((cx + SPADE_PATH_RADIUS, path_y)))])
    handle_path = Path([Line(lane_start, anchor), Line(anchor, spade((cx, path_y)))])
    lane_path = Path([Line(lane_start, lane_stop)])
    spots = [(*p, SPADE_METAL) for p in spade_path.points(SPADE_MEXES - 2, True)]
    spots += [(*p, HANDLE_METAL) for p in handle_path.points(HANDLE_MEXES, False)]
    spots.append((*lane_start, ENTRANCE_METAL))
    spots += [(*_offset(rng, p, LANE_MEX_OFFSET, lane_right), LANE_METAL) for p in lane_path.points(LANE_MEXES, False)]
    n = rng.randint(0, SPADE_MEXES - 2)
    geos = [spade_path.relative((n + 1.0 / 3.0 + (1.0 / 3.0) * rng.random()) / (SPADE_MEXES - 1))]
    n = rng.randint(LANE_MEXES // 2, math.ceil(LANE_MEXES / 2))
    geos.append(_offset(rng, lane_path.relative((n + 1.0 / 3.0 + (1.0 / 3.0) * rng.random()) / (LANE_MEXES + 1)), LANE_GEO_OFFSET, lane_right))
    unique_geos = [(cx, cy)] if bases >= MIN_BASES_FOR_CENTER_GEO else []
    radius = SPADE_WIDTH / 2 - START_BOX_PADDING
    box = [spade((cx - radius, neck - START_BOX_PADDING))]
    box += Path([Arc(spade((cx, end)), radius, spade_angle - math.radians(90), math.radians(180))]).points(20 - 1, True)
    box.append(spade((cx + radius, neck - START_BOX_PADDING)))
    start = spade((cx, neck - SPADE_HEIGHT / 2 - SPADE_VISUAL_OFFSET))
    return unique, shapes, spots, unique_geos, geos, box, start, spade_angle


def _building_centre(v: float) -> float:
    """roundToBuildingCenter, with Spring's math.round (floor of x + 0.5)."""
    return (math.floor(((v / SQUARE) - 1) / 2 + 0.5) * 2 + 1) * SQUARE


def generate(bases: int, seed: int, rng=None) -> Rampart:
    """The map a game with this many bases makes, drawing from `rng` (LuaRandom(seed) if None) as the Lua does from math.random."""
    if not 3 <= bases <= MOST_BASES:
        raise ValueError(f"Violet Rampart has 3 to {MOST_BASES} bases, not {bases}")
    rng = LuaRandom(seed) if rng is None else rng
    cx = cy = MAP_SIZE / 2
    turn = math.radians(360) / bases
    initial = rng.random() * turn
    unique, shapes, spots, unique_geos, geos, box, start, spade_angle = _base(rng, bases, turn)
    # FixLastBaseClockwiseOrder: base A is the first clockwise from 12 o'clock.
    if Rotation(cx, cy, initial + (bases - 1) * turn)(start)[0] >= cx:
        initial = initial - turn
    turns = [Rotation(cx, cy, initial + i * turn) for i in range(bases)]
    drawn = unique + [s.rotated(r) for s in shapes for r in turns]
    corners = np.full((CORNERS, CORNERS), BOTTOM)
    for shape in drawn:
        _draw(corners, shape)
    out_spots = []
    for r in turns:
        for sx, sz, metal in spots:
            x, z = r((sx, sz))
            out_spots.append((_building_centre(x), _building_centre(z), metal))
    out_geos = [(_building_centre(x), _building_centre(z)) for x, z in unique_geos + [r(g) for r in turns for g in geos]]
    boxes = [([r(p) for p in box], r(start)) for r in turns]
    return Rampart(bases, seed, corners, out_spots, out_geos, boxes, initial, spade_angle)
