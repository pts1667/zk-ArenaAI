"""The static bundle: what stays true about a map for a whole game, in arrays.

Built by the same terrain.py on the same height arrays in both paths, so a region means the same thing to the sim and the
live bot.  Once built it's numpy only, so a sim worker or a data loader never needs the terrain code.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .rawframe import SITE_DTYPE

MOVE_CLASSES = ("kbot", "tank", "spider", "amph", "hover", "ship")
REGION_TAGS = ("sea", "shallow", "plateau", "terrace", "basin", "flat", "rough", "island", "coast")
GATE_TAGS = ("choke", "ramp", "shore")
GATE_DIM = 6 + 6 + len(GATE_TAGS)
PASS_CELL = 16.0
MAX_HOP = 6


@dataclass
class Bundle:
    map_w: float
    map_h: float
    cell: float                 # of label and height
    label: np.ndarray           # h x w, region per work cell; every cell has one
    height: np.ndarray          # h x w, mean ground height
    passable: np.ndarray        # rows x cols at PASS_CELL: one bit per move class
    centre: np.ndarray          # R x 2
    area: np.ndarray            # R
    region_height: np.ndarray   # R x 2: height, height against neighbours
    sea: np.ndarray             # R
    tags: np.ndarray            # R x len(REGION_TAGS)
    access: np.ndarray          # R x 6
    shares: np.ndarray          # R x 2: wall, water
    spots: np.ndarray           # R x 2: count, income
    gate_counts: np.ndarray     # R x 2: ramps, chokes
    edge_a: np.ndarray          # E
    edge_b: np.ndarray          # E
    gate_x: np.ndarray          # E x GATE_DIM: 6 widths, 6 walking lengths, tags
    walk: np.ndarray            # 6 x R x R: walking elmos between regions, inf if there's no way
    hops: np.ndarray            # R x R int8: gates between regions over all classes, up to MAX_HOP
    spot_list: np.ndarray       # S x 3: x, z, income; the sim's sites come from here, the live bot's from the Brain

    @property
    def regions(self) -> int:
        return len(self.centre)

    def _cell(self, x: Any, z: Any) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip((np.asarray(z) / self.cell).astype(np.int32), 0, self.label.shape[0] - 1)
        j = np.clip((np.asarray(x) / self.cell).astype(np.int32), 0, self.label.shape[1] - 1)
        return i, j

    def region_at(self, x: Any, z: Any) -> np.ndarray:
        return self.label[self._cell(x, z)]

    def ground_at(self, x: Any, z: Any) -> np.ndarray:
        return self.height[self._cell(x, z)]

    def save(self, path: Any) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Any) -> "Bundle":
        with np.load(path) as d:
            return cls(**{f.name: (float(d[f.name]) if f.name in ("map_w", "map_h", "cell") else d[f.name]) for f in fields(cls)})

    def sites(self) -> np.ndarray:
        """The site records of a game with every spot free."""
        out = np.zeros(len(self.spot_list), SITE_DTYPE)
        out["site_id"] = np.arange(len(out))
        out["x"], out["z"], out["income"] = self.spot_list[:, 0], self.spot_list[:, 1], self.spot_list[:, 2]
        out["region_id"] = self.region_at(out["x"], out["z"])
        return out


def _shortest(cost: np.ndarray) -> np.ndarray:
    """All pairs, as Floyd-Warshall: a map has 120 regions at most."""
    d = cost.copy()
    np.fill_diagonal(d, 0.0)
    for k in range(len(d)):
        d = np.minimum(d, d[:, k, None] + d[k, None, :])
    return d


def from_terrain(t: Any) -> Bundle:
    from ..terrain import SEA

    r = len(t.regions)
    walk = np.full((6, r, r), np.inf, np.float64)
    hop = np.full((r, r), np.inf, np.float64)
    edge_a, edge_b, gate_x = [], [], []
    gate_counts = np.zeros((r, 2), np.float32)
    for (a, b), edge in sorted(t.edges.items()):
        row = np.zeros(GATE_DIM, np.float32)
        ramp = choke = 0
        for c, name in enumerate(MOVE_CLASSES):
            gates = edge.gates.get(name) or ()
            if not gates:
                continue
            row[c] = sum(g.width for g in gates)
            row[6 + c] = min(g.length for g in gates)
            walk[c, a, b] = walk[c, b, a] = row[6 + c]
            for g in gates:
                for k, tag in enumerate(GATE_TAGS):
                    row[12 + k] = max(row[12 + k], float(tag in g.tags))
                ramp = max(ramp, int(g.ramp is not None))
                choke = max(choke, int(g.choke is not None))
        if not row[:6].any():
            continue
        hop[a, b] = hop[b, a] = 1.0
        edge_a.append(a)
        edge_b.append(b)
        gate_x.append(row)
        gate_counts[[a, b], 0] += ramp
        gate_counts[[a, b], 1] += choke
    passable = np.zeros(t.comp["kbot"].shape, np.uint8)
    for c, name in enumerate(MOVE_CLASSES):
        passable |= (t.comp[name] >= 0).astype(np.uint8) << c
    regions = t.regions
    return Bundle(
        map_w=float(t.width), map_h=float(t.depth), cell=float(t.cell),
        label=np.asarray(t.label, np.int16), height=np.asarray(t.height, np.float32), passable=passable,
        centre=np.array([(g.x, g.z) for g in regions], np.float32).reshape(r, 2),
        area=np.array([g.area for g in regions], np.float32),
        region_height=np.array([(g.height, g.rel_height) for g in regions], np.float32).reshape(r, 2),
        sea=np.array([g.domain == SEA for g in regions], np.float32),
        tags=np.array([[tag in g.tags for tag in REGION_TAGS] for g in regions], np.float32).reshape(r, len(REGION_TAGS)),
        access=np.array([[g.access.get(c, 0.0) for c in MOVE_CLASSES] for g in regions], np.float32).reshape(r, 6),
        shares=np.array([(g.wall_share, g.water_share) for g in regions], np.float32).reshape(r, 2),
        spots=np.array([(len(g.spots), g.income) for g in regions], np.float32).reshape(r, 2),
        gate_counts=gate_counts,
        edge_a=np.array(edge_a, np.int32), edge_b=np.array(edge_b, np.int32),
        gate_x=np.array(gate_x, np.float32).reshape(len(edge_a), GATE_DIM),
        walk=np.stack([_shortest(walk[c]) for c in range(6)]).astype(np.float32),
        hops=np.minimum(_shortest(hop), MAX_HOP).astype(np.int8),
        spot_list=np.array(t.spots, np.float32).reshape(len(t.spots), 3),
    )
