"""The maps frames are collected on: the real ones, and made-up ones with the same statistics, in a directory of their own since they're large.

    python tools/sim_engine.py facts ... --out D:/zk-learned/maps/real_facts.json    what a game says about each real map
    python tools/map_gen.py real --facts D:/zk-learned/maps/real_facts.json --out D:/zk-learned/maps
    python tools/map_gen.py generate --count 200 --out D:/zk-learned/maps
    python tools/map_gen.py look --out results/maps --count 12      pictures of what `generate` would make
    python tools/map_gen.py stats --dir D:/zk-learned/maps          how the two kinds compare

Run these on Windows: reading a real map's archive takes 7-Zip, and `look` takes matplotlib.

Made-up maps are heights like a map file's, so they go through what a real map does (terrain_map.build, sim_data.export_map), and a frame can't tell them apart by what it's made of.
A map is a landform (rolling, plateaus, ridges, craters, canyons), then water (none, lakes, a river, a central sea, a coast, islands, nearly all sea), then flat ground at the starts, then ramps until a walker
can get everywhere that matters.  Most maps are symmetric, as real ones are: fields are symmetrised before any sharp shaping, and whatever's stamped afterwards is stamped at both places.
Sizes, relief, spots and incomes follow what maps.json and the real maps' facts say.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import sim_data  # noqa: E402
import terrain_map as tm  # noqa: E402
from zkbot import terrain as tr  # noqa: E402

SIDE = 512.0                    # elmos in one of a map's size units
WALK_CLASSES = ("kbot", "tank")
RAMP_RUN = 3.7                  # elmos along a ramp per elmo of rise: 15 degrees, which vehicles can climb
MOST_RAMPS = 16
MOST_TRIES = 8
RIVER_DEEP = 60.0


@dataclass(frozen=True)
class Rules:
    """What a map is drawn from.  Sizes and densities are real maps' (maps.json: side 10, 14 and 20 at the 10th, 50th and 90th percentile; relief 224, 504 and 1000; spots 0.17, 0.31 and 0.51 per unit square)."""
    sides: tuple[int, ...] = (8, 10, 12, 14, 16, 18, 20)
    side_weights: tuple[float, ...] = (0.08, 0.16, 0.2, 0.22, 0.14, 0.1, 0.1)
    square: float = 0.7
    relief: tuple[float, float] = (200.0, 1000.0)       # log-uniform
    spots_per_unit: tuple[float, float] = (0.17, 0.5)
    income: tuple[float, float] = (1.5, 3.0)            # a map's usual spot.  Real spots are 1.4, 2.0 and 3.1 at the 1st, 50th and 99th percentile.
    four_boxes: float = 0.15
    lava: float = 0.08                                  # of maps with water
    lands: tuple[tuple[str, float], ...] = (("rolling", 0.3), ("plateaus", 0.3), ("ridges", 0.12), ("craters", 0.13), ("canyons", 0.15))
    waters: tuple[tuple[str, float], ...] = (("none", 0.6), ("lakes", 0.08), ("river", 0.08), ("central", 0.07), ("coast", 0.06), ("islands", 0.07), ("sea", 0.04))
    symmetries: tuple[tuple[str, float], ...] = (("rot180", 0.4), ("mirror", 0.3), ("diagonal", 0.15), ("none", 0.15))


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _choose(rng: np.random.Generator, table: tuple[tuple[str, float], ...]) -> str:
    return str(table[int(rng.choice(len(table), p=np.array([w for _, w in table]) / sum(w for _, w in table)))][0])


class Canvas:
    """A map's grid of height corners, and how it's symmetric: image(point) and reflect(field) agree."""

    def __init__(self, rng: np.random.Generator, width: float, depth: float, symmetry: str):
        self.rng, self.width, self.depth = rng, width, depth
        self.cols, self.rows = int(width / tr.SQUARE) + 1, int(depth / tr.SQUARE) + 1
        self.x, self.z = np.meshgrid(np.arange(self.cols, dtype=np.float32) * float(tr.SQUARE), np.arange(self.rows, dtype=np.float32) * float(tr.SQUARE))
        self.symmetry = symmetry
        if symmetry == "mirror":
            self.axis = "x" if width > depth or (width == depth and rng.random() < 0.5) else "z"
        else:
            self.axis = ""
        self.flip = bool(rng.random() < 0.5)    # which diagonal

    def image(self, x: float, z: float) -> tuple[float, float]:
        w, d = self.width, self.depth
        if self.symmetry == "rot180":
            return w - x, d - z
        if self.symmetry == "mirror":
            return (w - x, z) if self.axis == "x" else (x, d - z)
        if self.symmetry == "diagonal":
            return (w - z, w - x) if self.flip else (z, x)
        return x, z

    def reflect(self, a: np.ndarray) -> np.ndarray:
        if self.symmetry == "rot180":
            return a[::-1, ::-1]
        if self.symmetry == "mirror":
            return a[:, ::-1] if self.axis == "x" else a[::-1, :]
        if self.symmetry == "diagonal":
            return a[::-1, ::-1].T if self.flip else a.T
        return a

    def places(self, x: float, z: float) -> list[tuple[float, float]]:
        return [(x, z)] if self.symmetry == "none" else [(x, z), self.image(x, z)]

    def noise(self, smooth: float, scale: float = 4096.0) -> np.ndarray:
        """Unit variance, with amplitudes falling as frequency to the -smooth.  Cropped, so that its wrapping doesn't show."""
        rows, cols = int(self.rows * 1.25), int(self.cols * 1.25)
        fz = np.fft.fftfreq(rows, tr.SQUARE / scale)
        fx = np.fft.rfftfreq(cols, tr.SQUARE / scale)
        k = np.maximum(np.hypot(fz[:, None], fx[None, :]), 0.5)
        spectrum = k ** -smooth * (self.rng.normal(size=k.shape) + 1j * self.rng.normal(size=k.shape))
        a = np.fft.irfft2(spectrum, s=(rows, cols))[:self.rows, :self.cols].astype(np.float32)
        if self.symmetry != "none":
            a = a + self.reflect(a)
        return (a - a.mean()) / max(float(a.std()), 1e-6)

    def distance(self, x: float, z: float) -> np.ndarray:
        return np.hypot(self.x - x, self.z - z)

    def to_segment(self, a: tuple[float, float], b: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """How far every corner is from a line, and how far along it (0 to 1) its nearest point is."""
        dx, dz = b[0] - a[0], b[1] - a[1]
        t = np.clip(((self.x - a[0]) * dx + (self.z - a[1]) * dz) / max(dx * dx + dz * dz, 1.0), 0.0, 1.0)
        return np.hypot(self.x - (a[0] + t * dx), self.z - (a[1] + t * dz)), t

    def height_at(self, h: np.ndarray, x: float, z: float) -> float:
        return float(h[min(max(int(z / tr.SQUARE), 0), self.rows - 1), min(max(int(x / tr.SQUARE), 0), self.cols - 1)])

    def _even_out(self, h: np.ndarray) -> None:
        """A thing and its image touch where they cross the line between them, and what's drawn second would win."""
        if self.symmetry != "none":
            h[:] = 0.5 * (h + self.reflect(h))

    def flatten(self, h: np.ndarray, x: float, z: float, inner: float, outer: float, level: float) -> None:
        for px, pz in self.places(x, z):
            w = _smooth((outer - self.distance(px, pz)) / max(outer - inner, 1.0))
            h += (level - h) * w
        self._even_out(h)

    def ramp(self, h: np.ndarray, a: tuple[float, float], b: tuple[float, float], width: float, fade: float, low: float, high: float) -> None:
        pairs = [(a, b)] if self.symmetry == "none" else [(a, b), (self.image(*a), self.image(*b))]
        for p, q in pairs:
            away, t = self.to_segment(p, q)
            w = _smooth((width + fade - away) / fade)
            h += (low + (high - low) * t - h) * w
        self._even_out(h)


def _starts(c: Canvas, pairs: int) -> list[tuple[float, float]]:
    """Near the edges and well apart, each with its image.  A map without symmetry has its second start drawn likewise."""
    rng = c.rng
    wanted = 0.45 * math.hypot(c.width, c.depth)
    out: list[tuple[float, float]] = []
    for _ in range(pairs):
        for _ in range(256):
            band = rng.uniform(0.08, 0.28, 2)
            x = (band[0] if rng.random() < 0.5 else 1.0 - band[0]) * c.width if rng.random() < 0.6 else rng.uniform(0.1, 0.9) * c.width
            z = (band[1] if rng.random() < 0.5 else 1.0 - band[1]) * c.depth if rng.random() < 0.6 else rng.uniform(0.1, 0.9) * c.depth
            point = float(np.round(x / 32.0) * 32.0), float(np.round(z / 32.0) * 32.0)
            other = c.image(*point)
            if c.symmetry == "none":
                for _ in range(64):
                    other = float(np.round(rng.uniform(0.1, 0.9) * c.width / 32.0) * 32.0), float(np.round(rng.uniform(0.1, 0.9) * c.depth / 32.0) * 32.0)
                    if math.dist(point, other) >= wanted:
                        break
            if math.dist(point, other) >= wanted and all(math.dist(p, q) >= wanted * 0.5 for q in out for p in (point, other)):
                out += [point, other]
                break
    return out


# Landforms ------------------------------------------------------------------------------------------------------------------------------------------------


def _rolling(c: Canvas, relief: float, starts: list) -> np.ndarray:
    return relief * 0.27 * c.noise(c.rng.uniform(1.9, 2.5))


def _plateaus(c: Canvas, relief: float, starts: list) -> np.ndarray:
    rng = c.rng
    levels = int(rng.integers(2, 5))
    field = c.noise(rng.uniform(2.0, 2.8))
    sharp = rng.uniform(0.03, 0.12)
    h = np.zeros_like(field)
    for q in rng.uniform(0.2, 0.85, levels):
        h += relief / levels * _smooth((field - float(np.quantile(field[::8, ::8], q))) / sharp)
    return h + relief * 0.03 * c.noise(2.3)


def _ridges(c: Canvas, relief: float, starts: list) -> np.ndarray:
    rng = c.rng
    crest = 1.0 - np.abs(c.noise(rng.uniform(1.9, 2.4), rng.uniform(3000.0, 6000.0)))
    gaps = _smooth((c.noise(2.2) + 0.6) / 0.4)     # where the ridges are low enough to pass
    return relief * (0.7 * _smooth((crest - 0.7) / 0.15) * gaps + 0.08 * c.noise(2.2))


def _craters(c: Canvas, relief: float, starts: list) -> np.ndarray:
    rng = c.rng
    h = relief * 0.12 * c.noise(2.2)
    for _ in range(int(rng.integers(4, 20))):
        radius = float(rng.uniform(150.0, 800.0))
        depth = min(relief * 0.5, radius * float(rng.uniform(0.15, 0.4)))
        x, z = float(rng.uniform(0.0, c.width)), float(rng.uniform(0.0, c.depth))
        for px, pz in c.places(x, z):
            d = c.distance(px, pz) / radius
            h += depth * (0.35 * np.exp(-((d - 1.0) / 0.25) ** 2) - np.clip(1.0 - d * d, 0.0, 1.0))
    return h


def _canyons(c: Canvas, relief: float, starts: list) -> np.ndarray:
    """High ground, with lanes cut through it between the starts and some rooms."""
    rng = c.rng
    rooms = [(float(rng.uniform(0.1, 0.9) * c.width), float(rng.uniform(0.1, 0.9) * c.depth)) for _ in range(int(rng.integers(3, 8)))]
    points = [q for p in rooms for q in c.places(*p)] + list(starts)
    away = np.full_like(c.x, 1e9)
    for i, p in enumerate(points):
        near = sorted(points[:i] + points[i + 1:], key=lambda q: math.dist(p, q))[:2]
        for q in near:
            away = np.minimum(away, c.to_segment(p, q)[0])
        away = np.minimum(away, c.distance(*p) - float(rng.uniform(100.0, 300.0)))
    wiggle = 120.0 * c.noise(2.0, 1500.0)
    width = float(rng.uniform(100.0, 260.0))
    return relief * (0.6 * _smooth((away + wiggle - width) / float(rng.uniform(32.0, 128.0))) + 0.03 * c.noise(2.3))


LANDFORMS = {"rolling": _rolling, "plateaus": _plateaus, "ridges": _ridges, "craters": _craters, "canyons": _canyons}


# Water ---------------------------------------------------------------------------------------------------------------------------------------------------


def _sink(h: np.ndarray, share: float, deepest: float) -> np.ndarray:
    """Levels the plane so that this share of the map is under it, as deep as this at most, and the land keeps its shape."""
    h = h - float(np.quantile(h[::8, ::8], share))
    low = float(-h.min())
    if low > 0.0:
        h = np.where(h < 0.0, h * deepest / low, h)
    return h


def _from_centre(c: Canvas) -> np.ndarray:
    return np.hypot((c.x - c.width / 2.0) / (c.width / 2.0), (c.z - c.depth / 2.0) / (c.depth / 2.0)) / math.sqrt(2.0)


def _river(c: Canvas, h: np.ndarray, starts: list, relief: float) -> np.ndarray:
    """Across the way between the first two starts, as its own image, with fords where it runs shallow."""
    rng = c.rng
    h = h - float(np.quantile(h[::8, ::8], 0.02)) + 30.0
    a, b = starts[0], starts[1]
    length = max(math.dist(a, b), 1.0)
    nx, nz = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    wiggle = c.noise(2.0, 3000.0)
    if c.symmetry != "none":
        wiggle = 0.5 * (wiggle - c.reflect(wiggle))
    across = (c.x - c.width / 2.0) * nx + (c.z - c.depth / 2.0) * nz + float(rng.uniform(100.0, 500.0)) * wiggle
    width = float(rng.uniform(100.0, 320.0))
    fords = _smooth((c.noise(2.4, 2000.0) - 0.7) / 0.2)
    deep = RIVER_DEEP * (1.0 - fords) + 14.0 * fords
    w = _smooth(1.0 - np.abs(across) / (width + 64.0))
    # Land rises from the banks, so that the channel's its own valley.
    return h * (1.0 - w) + (-deep) * w


def _water(c: Canvas, h: np.ndarray, kind: str, starts: list, relief: float) -> np.ndarray:
    rng = c.rng
    deepest = float(np.exp(rng.uniform(np.log(40.0), np.log(250.0))))
    if kind == "none":
        return h - float(h.min()) + float(rng.uniform(20.0, 100.0))
    if kind == "lakes":
        return _sink(h, float(rng.uniform(0.04, 0.2)), deepest)
    if kind == "river":
        return _river(c, h, starts, relief)
    if kind == "central":
        return _sink(h + 1.5 * relief * _from_centre(c), float(rng.uniform(0.2, 0.5)), deepest)
    if kind == "coast":
        return _sink(h - 1.5 * relief * _from_centre(c), float(rng.uniform(0.2, 0.5)), deepest)
    if kind == "islands":
        return _sink(h, float(rng.uniform(0.6, 0.88)), deepest)
    return _sink(h, float(rng.uniform(0.94, 0.99)), deepest)    # sea


# Getting about ------------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Ground:
    """Where each class can go, on the 16-elmo cells that speed fields use."""
    height: np.ndarray
    slope: np.ndarray
    water_damage: float
    hover_safe: bool

    @classmethod
    def of(cls, corners: np.ndarray, water_damage: float, hover_safe: bool) -> "Ground":
        center = (corners[:-1, :-1] + corners[:-1, 1:] + corners[1:, :-1] + corners[1:, 1:]) * 0.25
        slope = np.nan_to_num(np.asarray(tr.slope_from_corners(corners), np.float32))
        height = tr._block_mean(center, 2)
        rows, cols = min(height.shape[0], slope.shape[0]), min(height.shape[1], slope.shape[1])
        return cls(height[:rows, :cols], slope[:rows, :cols], water_damage * tm.DAMAGE_PER_SLOW_UPDATE, hover_safe)

    def can(self, name: str) -> np.ndarray:
        return tr.speed_field(tr.MOVE_CLASSES[name], self.height, self.slope, self.water_damage, self.hover_safe) > 0.0


def _cell(x: float, z: float, shape: tuple[int, int]) -> tuple[int, int]:
    return min(int(z / 16.0), shape[0] - 1), min(int(x / 16.0), shape[1] - 1)


def _closest(rng: np.random.Generator, one: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Among samples of two lists of cells."""
    one = one[rng.permutation(len(one))[:2500]]
    other = other[rng.permutation(len(other))[:2500]]
    best, a, b = 1e18, one[0], other[0]
    for start in range(0, len(one), 512):
        block = one[start:start + 512]
        d = ((block[:, None] - other[None]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        if d[i, j] < best:
            best, a, b = float(d[i, j]), block[i], other[j]
    return a, b


def _join_up(c: Canvas, h: np.ndarray, starts: list, damage: float, hover_safe: bool) -> None:
    """Ramps from where a walker at the first start can get, to the largest place it can't, until any place left over is small."""
    rng = c.rng
    smallest = max(1200.0, 0.01 * c.rows * c.cols / 4.0)    # cells: 1% of the map
    width = float(rng.uniform(56.0, 128.0))
    for _ in range(MOST_RAMPS):
        can = Ground.of(h, damage, hover_safe).can("tank")
        labels, sizes = tr.label_runs(can)
        main = int(labels[_cell(*starts[0], can.shape)])
        if main < 0:    # the first start's pad itself is too steep, which only a very small pad on a hard slope does
            return
        others = [k for k in np.argsort(-sizes) if k != main and sizes[k] >= smallest]
        if not others:
            return
        a, b = _closest(rng, np.argwhere(labels == main), np.argwhere(labels == others[0]))
        ax, az, bx, bz = (a[1] + 0.5) * 16.0, (a[0] + 0.5) * 16.0, (b[1] + 0.5) * 16.0, (b[0] + 0.5) * 16.0
        high_a, high_b = max(c.height_at(h, ax, az), 8.0), max(c.height_at(h, bx, bz), 8.0)
        gap = max(math.hypot(bx - ax, bz - az), 1.0)
        short = RAMP_RUN * abs(high_a - high_b) + 64.0 - gap
        if short > 0.0:     # the ramp runs on into the ground at both ends
            ux, uz = (bx - ax) / gap, (bz - az) / gap
            ax, az, bx, bz = ax - ux * short / 2.0, az - uz * short / 2.0, bx + ux * short / 2.0, bz + uz * short / 2.0
            high_a, high_b = max(c.height_at(h, ax, az), 8.0), max(c.height_at(h, bx, bz), 8.0)
        c.ramp(h, (ax, az), (bx, bz), width, 48.0, high_a, high_b)


def _together(ground: Ground, name: str, starts: list) -> bool:
    can = ground.can(name)
    labels, _ = tr.label_runs(can)
    at = {int(labels[_cell(x, z, can.shape)]) for x, z in starts}
    return len(at) == 1 and -1 not in at


# Spots, boxes and things lying about --------------------------------------------------------------------------------------------------------------------


def _spots(c: Canvas, ground: Ground, starts: list, rules: Rules, water_share: float, walkers_only: bool) -> list[tuple[float, float, float]]:
    """Some round each start, as a real map has, and the rest spread about at a real map's density, some under water where there's water."""
    rng = c.rng
    can = ground.can("kbot") & ground.can("tank")
    wet = (ground.height < -5.0) & ground.can("hover") if not ground.water_damage else np.zeros_like(can)
    area = c.width * c.depth / (SIDE * SIDE)
    wanted = float(rng.uniform(*rules.spots_per_unit)) * area
    spacing = 0.65 * math.sqrt(c.width * c.depth / max(wanted, 1.0))
    usual = float(rng.uniform(*rules.income))
    out: list[tuple[float, float, float]] = []

    def add(x: float, z: float, room: float) -> bool:
        if not all(math.dist((x, z), s[:2]) >= room for s in out):
            return False
        income = round(float(np.clip(rng.normal(usual, 0.2), 0.8, 5.0)), 3)
        for px, pz in c.places(x, z):
            if not all(math.dist((px, pz), s[:2]) >= room * 0.5 for s in out):
                return False
        out.extend((px, pz, income) for px, pz in c.places(x, z))
        return True

    cells = np.argwhere(can)
    for sx, sz in starts[::1 if c.symmetry == "none" else 2]:
        near = cells[np.hypot((cells[:, 1] + 0.5) * 16.0 - sx, (cells[:, 0] + 0.5) * 16.0 - sz) <= 900.0]
        if not len(near):
            continue
        placed = 0
        for i, j in near[rng.permutation(len(near))[:64]]:
            placed += add((j + 0.5) * 16.0, (i + 0.5) * 16.0, 240.0)
            if placed >= int(rng.integers(3, 6)):
                break
    land = cells
    water = np.argwhere(wet) if water_share > 0.02 else np.zeros((0, 2), np.int64)
    for _ in range(int(wanted) * 8):
        if len(out) >= wanted:
            break
        pool = water if len(water) and rng.random() < 0.8 * water_share else land
        if not len(pool):
            pool = water
        i, j = pool[int(rng.integers(len(pool)))]
        add((j + 0.5) * 16.0, (i + 0.5) * 16.0, spacing)
    return out


def _box(c: Canvas, x: float, z: float) -> dict:
    """A rectangle round a start, a quarter of the map across or so.  As tools/sim_engine.py facts has them."""
    rng = c.rng
    half_x, half_z = float(rng.uniform(0.08, 0.18)) * c.width, float(rng.uniform(0.08, 0.18)) * c.depth
    x0, x1 = max(16.0, x - half_x), min(c.width - 16.0, x + half_x)
    z0, z1 = max(16.0, z - half_z), min(c.depth - 16.0, z + half_z)
    return {"polygons": [[[x0, z0], [x1, z0], [x1, z1], [x0, z1]]], "startpoints": [[x, z]]}


def _features(c: Canvas, ground: Ground) -> list[list[float]]:
    """Half of maps have nothing lying about.  Real features are 10, 50 and 100 metal at the 10th, 50th and 90th percentile, with 100 to 2000 health."""
    rng = c.rng
    if rng.random() < 0.5:
        return []
    can = np.argwhere(ground.can("kbot"))
    out = []
    metal = float(rng.choice((10.0, 25.0, 50.0, 100.0)))
    for _ in range(int(rng.integers(4, 70))):
        i, j = can[int(rng.integers(len(can)))]
        for px, pz in c.places((j + 0.5) * 16.0, (i + 0.5) * 16.0):
            out.append([px, pz, metal, float(rng.choice((100.0, 500.0, 2000.0))), float(rng.uniform(16.0, 40.0)), 1.0])
    return out


@dataclass
class Made:
    map: tm.MapFile
    layouts: list
    features: list
    style: dict


def make(seed: int, rules: Rules = Rules()) -> Made:
    """A map whose starts can reach each other.  What can't is drawn again."""
    for attempt in range(MOST_TRIES):
        rng = np.random.default_rng((seed, attempt))
        made = _attempt(rng, seed, rules)
        if made is not None:
            return made
    raise ValueError(f"seed {seed}: no good map in {MOST_TRIES} tries")


def _attempt(rng: np.random.Generator, seed: int, rules: Rules) -> Made | None:
    side = int(rng.choice(rules.sides, p=np.array(rules.side_weights) / sum(rules.side_weights)))
    other = side if rng.random() < rules.square else int(np.clip(side + rng.choice((-6, -4, -2, 2, 4)), rules.sides[0], rules.sides[-1]))
    symmetry = _choose(rng, rules.symmetries)
    if symmetry == "diagonal":
        other = side
    c = Canvas(rng, side * SIDE, other * SIDE, symmetry)
    pairs = 2 if rng.random() < rules.four_boxes and symmetry != "none" else 1
    starts = _starts(c, pairs)
    if len(starts) < 2:
        return None
    land, water = _choose(rng, rules.lands), _choose(rng, rules.waters)
    relief = float(np.exp(rng.uniform(*np.log(rules.relief))))
    h = _water(c, LANDFORMS[land](c, relief, starts), water, starts, relief)
    damage = float(rng.choice((100.0, 1000.0))) if water != "none" and rng.random() < rules.lava else 0.0
    hover_safe = bool(damage and rng.random() < 0.5)
    # Flat ground at each start, standing clear of any water.
    for x, z in starts[::1 if symmetry == "none" else 2]:
        inner = float(rng.uniform(220.0, 500.0))
        level = max(c.height_at(h, x, z), float(rng.uniform(12.0, 60.0)))
        c.flatten(h, x, z, inner, inner + float(rng.uniform(80.0, 240.0)) + 0.5 * abs(level), level)
    must_walk = water not in ("islands", "sea") and not (water == "central" and rng.random() < 0.5)
    if must_walk:
        _join_up(c, h, starts, damage, hover_safe)
    ground = Ground.of(h, damage, hover_safe)
    if must_walk and not _together(ground, "kbot", starts):
        return None
    if not must_walk and not (_together(ground, "amph", starts) and (hover_safe or not damage) and _together(ground, "hover", starts)):
        return None
    for x, z in starts:
        if not tm_buildable(ground, x, z):
            return None
    water_share = float((ground.height < 0.0).mean())
    spots = _spots(c, ground, starts, rules, 0.0 if damage else water_share, must_walk)
    if len(spots) < 6:
        return None
    boxes = []
    for x, z in starts[::1 if symmetry == "none" else 2]:
        box = _box(c, x, z)
        boxes.append(box)
        if symmetry != "none":
            image = c.image(x, z)
            boxes.append({"polygons": [[list(c.image(*v)) for v in box["polygons"][0]]], "startpoints": [list(image)]})
    mf = tm.MapFile(f"gen_{seed:05d}", h.astype(np.float32), np.zeros((c.rows // 2, c.cols // 2), np.uint8), water_damage=damage, hover_safe=hover_safe, spots=spots, spot_source="made")
    return Made(mf, [boxes], _features(c, ground), {"land": land, "water": water, "symmetry": symmetry, "relief": relief})


def tm_buildable(ground: Ground, x: float, z: float, room: float = 48.0) -> bool:
    """What tools/policy_opening.py asks of a start."""
    can = ground.can("kbot") & ground.can("tank")
    r0, r1 = max(int((z - room) / 16.0), 0), int((z + room) / 16.0) + 1
    c0, c1 = max(int((x - room) / 16.0), 0), int((x + room) / 16.0) + 1
    block = can[r0:r1, c0:c1]
    return block.size > 0 and bool(block.all())


# Writing maps out ---------------------------------------------------------------------------------------------------------------------------------------


def write_entry(mf: tm.MapFile, kind: str, layouts: list, features: list, out: pathlib.Path, style: dict | None = None) -> tr.TerrainMap:
    """Beside the sim's and the policy's files, which classes can reach each other, and what's known about the map."""
    terrain = sim_data.export_map(mf, mf.name, out)
    slug = sim_data._slug(mf.name)
    np.savez_compressed(out / f"{slug}.reach.npz", **{name: labels.astype(np.int16) for name, labels in terrain.comp.items()})
    summary = terrain.summary()
    facts = {"name": mf.name, "kind": kind, "layouts": layouts, "features": features, "water": float(summary["water"]), "liquid": summary["liquid"], "style": style or {}}
    (out / f"{slug}.facts.json").write_text(json.dumps(facts), "utf8")
    return terrain


def _quiet_write(mf: tm.MapFile, kind: str, layouts: list, features: list, out: pathlib.Path, style: dict | None) -> dict:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return write_entry(mf, kind, layouts, features, out, style).summary()


def _one_real(job: tuple[str, str, dict, str]) -> str:
    name, path, facts, out = job
    started = time.time()
    try:
        mf = tm.load_map(pathlib.Path(path), name)
        if mf.generated:
            return f"{name}: makes its terrain in the game, left out"
        layouts = [b for b in facts["layouts"] if len(b) >= 2]
        mf.spots = [tuple(s) for s in facts["spots"]]
        if not layouts or not mf.spots:
            return f"{name}: {len(layouts)} layouts and {len(mf.spots)} spots, left out"
        mf.spot_source = "engine"
        s = _quiet_write(mf, "real", layouts, facts["features"], pathlib.Path(out), None)
        return f"{name}: {s['land']} land and {s['sea']} sea regions, {s['water']:.0%} {s['liquid']}, {time.time() - started:.0f} s"
    except Exception as e:      # one map that won't go shouldn't cost the rest
        return f"{name}: failed, {type(e).__name__}: {e}"


def export_real(facts_file: pathlib.Path, out: pathlib.Path, workers: int) -> None:
    table = json.loads(facts_file.read_text("utf8"))
    jobs = [(name, str(path), table[sim_data._slug(name)], str(out)) for name, path in tm.find_maps([])
            if sim_data._slug(name) in table and not (out / f"{sim_data._slug(name)}.facts.json").exists()]
    print(f"{len(jobs)} real maps to export", flush=True)
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for line in pool.map(_one_real, jobs):
            print(line, flush=True)


def _one_made(job: tuple[int, str]) -> str:
    seed, out = job
    started = time.time()
    try:
        made = make(seed)
        s = _quiet_write(made.map, "made", made.layouts, made.features, pathlib.Path(out), made.style)
        return f"{made.map.name}: {made.style}, {made.map.corners.shape[1] // 64} x {made.map.corners.shape[0] // 64}, {len(made.map.spots)} spots, {s['land']} land and {s['sea']} sea regions, " \
               f"{s['walls']:.0%} walls, {s['water']:.0%} {s['liquid']}, {time.time() - started:.0f} s"
    except Exception as e:
        return f"gen_{seed:05d}: failed, {type(e).__name__}: {e}"


def generate(count: int, out: pathlib.Path, seed: int, workers: int) -> None:
    jobs = [(s, str(out)) for s in range(seed, seed + count) if not (out / f"gen_{s:05d}.facts.json").exists()]
    print(f"{len(jobs)} maps to make", flush=True)
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for line in pool.map(_one_made, jobs):
            print(line, flush=True)


def look(out: pathlib.Path, count: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s in range(seed, seed + count):
        started = time.time()
        made = make(s)
        made_at = time.time()
        terrain = tm.build(made.map)
        summary = terrain.summary()
        tm.render(f"{made.map.name} {made.style['land']}, {made.style['water']}, {made.style['symmetry']}", made.map, terrain, out / f"{made.map.name}.png")
        print(f"{made.map.name}: {made.style}, spots {len(made.map.spots)}, features {len(made.features)}, regions {summary['land']}+{summary['sea']}, levels {summary['levels']}, ramps {summary['ramps']}, chokes {summary['chokes']}, "
              f"walls {summary['walls']:.2f}, water {summary['water']:.2f}, made in {made_at - started:.1f} s, built in {time.time() - made_at:.1f} s", flush=True)


def stats(directory: pathlib.Path) -> None:
    """Medians by kind, to compare."""
    rows: dict[str, list[list[float]]] = {}
    sys.path.insert(0, str(ROOT / "bots"))
    from zkbot.policy import bundle as bd
    for path in sorted(directory.glob("*.facts.json")):
        facts = json.loads(path.read_text("utf8"))
        b = bd.Bundle.load(path.with_name(path.name.replace(".facts.json", ".bundle.npz")))
        area = b.map_w * b.map_h / (SIDE * SIDE)
        passable = [float(((b.passable >> c) & 1).mean()) for c in range(6)]
        rows.setdefault(facts["kind"], []).append([area, float(b.height.max() - b.height.min()), facts["water"], len(b.spot_list) / area, float(b.spot_list[:, 2].mean()), b.regions, float(b.sea.sum()), passable[0], passable[1], len(facts["features"])])
    names = ("area", "relief", "water", "spots/area", "income", "regions", "sea regions", "kbot", "tank", "features")
    for kind, table in rows.items():
        t = np.array(table)
        print(f"{kind}: {len(t)} maps")
        for i, name in enumerate(names):
            print(f"  {name:12s} " + " ".join(f"{v:8.2f}" for v in np.percentile(t[:, i], (10, 50, 90))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("real")
    r.add_argument("--facts", type=pathlib.Path, required=True)
    r.add_argument("--out", type=pathlib.Path, required=True)
    r.add_argument("--workers", type=int, default=10)
    g = sub.add_parser("generate")
    g.add_argument("--count", type=int, default=200)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", type=pathlib.Path, required=True)
    g.add_argument("--workers", type=int, default=10)
    k = sub.add_parser("look")
    k.add_argument("--count", type=int, default=12)
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "maps")
    s = sub.add_parser("stats")
    s.add_argument("--dir", type=pathlib.Path, required=True)
    a = parser.parse_args()
    if a.command == "real":
        export_real(a.facts, a.out, a.workers)
    elif a.command == "generate":
        generate(a.count, a.out, a.seed, a.workers)
    elif a.command == "look":
        look(a.out, a.count, a.seed)
    else:
        stats(a.dir)


if __name__ == "__main__":
    main()
