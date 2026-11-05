"""Routes between segments, with one centre per segment instead of a field per segment.

Segments are split until a segment's centre can see most of it, so a unit that can't see where it's going can nearly
always walk to its segment's centre.  From there the next segment's gate is in sight, and so on: a table of next hops is
all the pathfinding a run needs.  It's an approximation, by design: splitting stops at a limit, and the few tiles left
over go to a nearby centre that sees them, even another segment's, or else to the nearest in their own.  Numpy only,
so both the map export and the tests can use it.
"""
from __future__ import annotations

import heapq

import numpy as np

SIGHT_STEP = 0.5            # cells between tests along a line: finer than the sim's own, so its lines are clear too
CHUNK_SAMPLES = 4_000_000   # samples per batch of lines
SEED_CANDIDATES = 6         # possible centres tried for each new segment
SCORE_CELLS = 400           # cells each candidate is scored on
MIN_SEGMENT_CELLS = 16      # smaller leftovers join a neighbour, though its centre can't see them all
MAX_PARTS = 4               # a segment splits into at most this many
SEEN_SHARE = 0.9            # stop splitting once its centres see this share of a segment
LEFTOVER_CENTRES = 12       # how many of the nearest centres a leftover tile tries


def _sight(passable: np.ndarray, sx: int, sz: int, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Which of these cells have a clear line to the cell (sx, sz)."""
    dx, dz = (xs - sx).astype(np.float32), (zs - sz).astype(np.float32)
    steps = np.maximum(np.ceil(np.hypot(dx, dz) / SIGHT_STEP).astype(np.int32), 1)
    order = np.argsort(steps)
    out = np.zeros(len(xs), bool)
    start = 0
    while start < len(order):
        most = int(steps[order[min(start + 4096, len(order) - 1)]])
        end = start + max(1, min(4096, CHUNK_SAMPLES // max(most, 1)))
        batch = order[start:end]
        n = int(steps[batch].max())
        t = np.arange(1, n + 1, dtype=np.float32)[None, :] / steps[batch, None]
        inside = t <= 1.0
        px = np.floor(sx + 0.5 + dx[batch, None] * t).astype(np.int32)
        pz = np.floor(sz + 0.5 + dz[batch, None] * t).astype(np.int32)
        px = np.clip(px, 0, passable.shape[1] - 1)
        pz = np.clip(pz, 0, passable.shape[0] - 1)
        out[batch] = (passable[pz, px] | ~inside).all(axis=1)
        start = end
    return out


def split_by_sight(segments: np.ndarray, passable: np.ndarray, max_parts: int = MAX_PARTS, seen_share: float = SEEN_SHARE,
                   seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split each segment into parts visible from one cell, up to the limits; the rest go to the nearest centre.
    Returns the labels and each part's centre as (x, z)."""
    rng = np.random.default_rng(seed)
    out = np.full(segments.shape, -1, np.int16)
    centres: list[tuple[int, int]] = []
    left_x, left_z, left_fallback = [], [], []
    for s in np.unique(segments[segments >= 0]).tolist():
        zs, xs = np.nonzero(segments == s)
        size, first = len(xs), len(centres)
        while len(centres) == first or (len(xs) > size * (1.0 - seen_share) and len(centres) - first < max_parts):
            mid = int(np.argmin((xs - xs.mean()) ** 2 + (zs - zs.mean()) ** 2))
            candidates = [mid] + rng.integers(0, len(xs), SEED_CANDIDATES - 1).tolist()
            scored = rng.integers(0, len(xs), min(SCORE_CELLS, len(xs)))
            # The middle wins ties, so a segment that is already open keeps a centre near its centroid.
            scores = [int(_sight(passable, int(xs[c]), int(zs[c]), xs[scored], zs[scored]).sum()) for c in candidates]
            best = candidates[int(np.argmax(scores))]
            sx, sz = int(xs[best]), int(zs[best])
            seen = _sight(passable, sx, sz, xs, zs)
            seen[best] = True
            out[zs[seen], xs[seen]] = len(centres)
            centres.append((sx, sz))
            xs, zs = xs[~seen], zs[~seen]
        if len(xs):
            mine = np.array(centres[first:], np.float32)
            d2 = (xs[:, None] - mine[:, 0]) ** 2 + (zs[:, None] - mine[:, 1]) ** 2
            left_x.append(xs)
            left_z.append(zs)
            left_fallback.append(first + np.argmin(d2, axis=1))
    if left_x:
        _place_leftovers(out, passable, centres, np.concatenate(left_x), np.concatenate(left_z), np.concatenate(left_fallback))
    return out, centres


def _place_leftovers(out: np.ndarray, passable: np.ndarray, centres: list[tuple[int, int]], xs: np.ndarray, zs: np.ndarray, fallback: np.ndarray) -> None:
    """Each tile goes to the nearest centre that sees it, or failing that to its own segment's nearest."""
    out[zs, xs] = fallback
    where = np.array(centres, np.float32)
    tiers = min(LEFTOVER_CENTRES, len(centres))
    nearest = np.empty((len(xs), tiers), np.int32)
    for start in range(0, len(xs), 4096):
        d2 = (xs[start:start + 4096, None] - where[:, 0]) ** 2 + (zs[start:start + 4096, None] - where[:, 1]) ** 2
        nearest[start:start + 4096] = np.argsort(d2, axis=1)[:, :tiers]
    todo = np.arange(len(xs))
    for tier in range(tiers):
        chosen = nearest[todo, tier]
        placed = np.zeros(len(todo), bool)
        for centre in np.unique(chosen).tolist():
            group = np.nonzero(chosen == centre)[0]
            seen = _sight(passable, centres[centre][0], centres[centre][1], xs[todo[group]], zs[todo[group]])
            out[zs[todo[group[seen]]], xs[todo[group[seen]]]] = centre
            placed[group[seen]] = True
        todo = todo[~placed]
        if not len(todo):
            break


def _border_pairs(labels: np.ndarray):
    """Both directions of every 4-way step between two segments: from-cell and to-cell, as flat arrays."""
    rows, cols = labels.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    a = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
    b = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
    flat = labels.ravel()
    keep = (flat[a] != flat[b]) & (flat[a] >= 0) & (flat[b] >= 0)
    return np.concatenate([a[keep], b[keep]]), np.concatenate([b[keep], a[keep]])


def join_small(labels: np.ndarray, centres: list[tuple[int, int]], min_cells: int = MIN_SEGMENT_CELLS) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Small parts join the neighbour they share most border with; isolated ones lose their segment."""
    count = len(centres)
    if not count:
        return labels, centres
    flat = labels.ravel()
    sizes = np.bincount(flat[flat >= 0], minlength=count)
    src, dst = _border_pairs(labels)
    pairs, shared = np.unique(flat[src].astype(np.int64) * count + flat[dst], return_counts=True)
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for key, n in zip(pairs.tolist(), shared.tolist()):
        neighbours.setdefault(key // count, []).append((n, key % count))
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    total = sizes.tolist()
    for s in np.argsort(sizes).tolist():
        if sizes[s] >= min_cells:
            break
        touching = [(n, find(o)) for n, o in neighbours.get(s, []) if find(o) != find(s)]
        if touching:
            root, into = find(s), max(touching)[1]
            # The larger part keeps its centre.
            if total[root] > total[into]:
                root, into = into, root
            parent[root] = into
            total[into] += total[root]
    roots = [s for s in range(count) if find(s) == s and total[s] >= min_cells]
    new = {r: i for i, r in enumerate(roots)}
    table = np.array([new.get(find(s), -1) for s in range(count)] + [-1], np.int16)
    return table[labels], [centres[r] for r in roots]


def _walk_time(speed: np.ndarray, x0: float, z0: float, x1: float, z1: float) -> float:
    d = float(np.hypot(x1 - x0, z1 - z0))
    if d <= 0.0:
        return 0.0
    n = max(1, int(d))
    t = (np.arange(n, dtype=np.float32) + 1.0) / n
    px = np.clip(np.floor(x0 + (x1 - x0) * t).astype(np.int32), 0, speed.shape[1] - 1)
    pz = np.clip(np.floor(z0 + (z1 - z0) * t).astype(np.int32), 0, speed.shape[0] - 1)
    return float((d / n / np.maximum(speed[pz, px], 0.05)).sum())


def build_routes(labels: np.ndarray, centres: list[tuple[int, int]], speed: np.ndarray) -> dict[str, np.ndarray]:
    """Gates between touching segments and the first gate to take from any segment to any other.

    All positions are in cells.  next_edge[a, g] indexes the edge arrays, -1 where there's no way."""
    count = len(centres)
    passable = speed > 0
    centre = np.array(centres, np.float32).reshape(count, 2) + 0.5
    src, dst = _border_pairs(labels)
    flat = labels.ravel()
    cols = labels.shape[1]
    edge_from: list[int] = []
    edge_to: list[int] = []
    gates: list[tuple[float, float]] = []
    costs: list[float] = []
    if len(src):
        pair = flat[src].astype(np.int64) * count + flat[dst]
        order = np.argsort(pair, kind="stable")
        keys, starts = np.unique(pair[order], return_index=True)
        for key, first, last in zip(keys.tolist(), starts.tolist(), starts[1:].tolist() + [len(order)]):
            a, b = key // count, key % count
            cells = dst[order[first:last]]
            gx, gz = cells % cols, cells // cols
            near = np.hypot(gx + 0.5 - centre[a, 0], gz + 0.5 - centre[a, 1]) + np.hypot(gx + 0.5 - centre[b, 0], gz + 0.5 - centre[b, 1])
            # The gate is on the far side: the shortest way through that both centres can see, or failing that this one.
            seen = _sight(passable, centres[a][0], centres[a][1], gx, gz)
            both = seen & _sight(passable, centres[b][0], centres[b][1], gx, gz)
            usable = both if both.any() else seen
            if usable.any():
                near = np.where(usable, near, np.inf)
            k = int(np.argmin(near))
            x, z = float(gx[k]) + 0.5, float(gz[k]) + 0.5
            edge_from.append(a)
            edge_to.append(b)
            gates.append((x, z))
            costs.append(_walk_time(speed, centre[a, 0], centre[a, 1], x, z) + _walk_time(speed, x, z, centre[b, 0], centre[b, 1]))
    inbound: list[list[int]] = [[] for _ in range(count)]
    for e, b in enumerate(edge_to):
        inbound[b].append(e)
    next_edge = np.full((count, count), -1, np.int32)
    for goal in range(count):
        best = [float("inf")] * count
        best[goal] = 0.0
        heap = [(0.0, goal)]
        while heap:
            t, v = heapq.heappop(heap)
            if t > best[v]:
                continue
            for e in inbound[v]:
                u = edge_from[e]
                nt = t + costs[e]
                if nt < best[u]:
                    best[u] = nt
                    next_edge[u, goal] = e
                    heapq.heappush(heap, (nt, u))
    return {"centres": centre, "edge_to": np.array(edge_to, np.int32), "gates": np.array(gates, np.float32).reshape(-1, 2), "next_edge": next_edge}


def navigation(segments: np.ndarray, speed: np.ndarray, max_parts: int = MAX_PARTS, seen_share: float = SEEN_SHARE) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Final labels and routes for one class, from its first segments."""
    labels, centres = split_by_sight(segments, speed > 0, max_parts, seen_share)
    labels, centres = join_small(labels, centres)
    return labels, build_routes(labels, centres, speed)
