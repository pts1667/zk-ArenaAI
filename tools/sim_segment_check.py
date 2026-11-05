"""Do straight lines inside a segment arrive?  Random goals per class, one unit per world, on a map's real terrain.

    ~/zkml/bin/python tools/sim_segment_check.py [map file]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim" / "py"))
import zksim  # noqa: E402

PAIRS = 300
MIN_DISTANCE = 300.0
ARRIVED = 64.0
TIME_ALLOWANCE = 4.0     # of the straight walk at top speed on level ground
UNITS = {"kbot": "cloakraid", "tank": "tankassault", "spider": "spiderassault", "hover": "hoverraid"}


def walkable_areas(speed: np.ndarray) -> np.ndarray:
    """Connected walkable ground, stepping as the fields do, so sampled goals can be reached."""
    rows, cols = speed.shape
    area = np.full(speed.shape, -1, np.int32).ravel()
    open_cell = (speed > 0).ravel().tolist()
    label = area.tolist()
    count = 0
    for start in range(rows * cols):
        if not open_cell[start] or label[start] >= 0:
            continue
        label[start] = count
        todo = [start]
        while todo:
            i = todo.pop()
            cx, cz = i % cols, i // cols
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, nz = cx + dx, cz + dz
                    if 0 <= nx < cols and 0 <= nz < rows:
                        j = nz * cols + nx
                        if open_cell[j] and label[j] < 0:
                            label[j] = count
                            todo.append(j)
        count += 1
    return np.array(label, np.int32).reshape(rows, cols)


def line_blocked(speed: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    d = float(np.hypot(*(b - a)))
    for t in np.arange(16.0, d, 16.0) / d:
        p = a + (b - a) * t
        if speed[int(p[1] / 16), int(p[0] / 16)] <= 0:
            return True
    return False


def sample(rng, cells: np.ndarray, want) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(4000):
        i, j = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        a, b = (i[::-1] + 0.5) * 16.0, (j[::-1] + 0.5) * 16.0
        if np.hypot(*(b - a)) >= MIN_DISTANCE and want(i, j):
            return a, b
    return None


def walk(cat, world_map, unit: str, a, b, speed_per_frame: float) -> bool:
    w = zksim.World(cat, world_map)
    u = w.add(unit, 0, float(a[0]), float(a[1]))
    w.move(u, float(b[0]), float(b[1]))
    limit = int(TIME_ALLOWANCE * np.hypot(*(b - a)) / speed_per_frame / zksim.TICK_FRAMES) + 30
    for _ in range(limit):
        w.step(1)
        s = w.get(u)
        if np.hypot(s.x - b[0], s.z - b[1]) < ARRIVED:
            return True
    return False


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else zksim.DATA / "maps/titanduel_2_2.npz"
    cat = zksim.Catalog()
    world_map = zksim.Map.load(path)
    rng = np.random.default_rng(7)
    print(f"{pathlib.Path(path).stem}: arrived within {TIME_ALLOWANCE:g}x the straight walk, {PAIRS} goals each")
    print(f"  {'class':8s}{'same segment':>14s}{'line blocked':>14s}{'of those arrived':>18s}{'other segment':>15s}")
    for name, unit in UNITS.items():
        c = zksim.MOVE_CLASSES.index(name)
        speed, seg = world_map.speed[c], world_map.segments[c]
        area = walkable_areas(speed)
        big = np.bincount(seg[seg >= 0])
        cells = np.argwhere((seg >= 0) & (speed > 0) & (big[np.maximum(seg, 0)] > 64))
        top = cat.units[cat.index[unit]]["speed"]
        same = other = other_tried = blocked = blocked_ok = 0
        for _ in range(PAIRS):
            pair = sample(rng, cells, lambda i, j: seg[tuple(i)] == seg[tuple(j)])
            if pair is None:
                continue
            ok = walk(cat, world_map, unit, *pair, top)
            same += ok
            if line_blocked(speed, *pair):
                blocked += 1
                blocked_ok += ok
        for _ in range(PAIRS):
            pair = sample(rng, cells, lambda i, j: seg[tuple(i)] != seg[tuple(j)] and area[tuple(i)] == area[tuple(j)])
            if pair is not None:
                other_tried += 1
                other += walk(cat, world_map, unit, *pair, top)
        print(f"  {name:8s}{same / PAIRS:14.1%}{blocked / PAIRS:14.1%}{f'{blocked_ok}/{blocked}':>18s}{f'{other / other_tried:.1%}' if other_tried else 'none':>15s}")


if __name__ == "__main__":
    main()
