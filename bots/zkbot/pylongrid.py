"""Pylons that join far extractors to the overdrive grid (flag `pylongrid`).

Two units share a grid when they're closer than the sum of their link ranges
(Zero-K unit_mex_overdrive.lua, quadField:GetIntersections).  A generator reaches
63-150 elmos and an extractor 50, so only the extractors around the base ever
joined the generators; a pylon reaches 500.  At most one was ever built, wherever
the next reactor would go.

Overdrive gives an extractor sqrt(energy) * 0.25 of its income extra, so spreading
the same energy over more extractors yields more metal.  A pylon is built when
that extra metal pays for it soon enough.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo
from .params import P

PYLON = "energypylon"
PYLON_RANGE = 500.0
GENERATORS = ("energysolar", "energywind", "energyfusion", "energysingu", "energygeo", "energyheavygeo")
PYLON_COST = P("grid.pylon_cost", 200.0, 50.0, 500.0, "metal per pylon")
PAYBACK = P("grid.payback", 300.0, 60.0, 900.0, "seconds of extra metal a pylon must pay for itself within")
MARGIN = P("grid.margin", 50.0, 10.0, 150.0, "elmos of link range kept spare for placement")
PRIORITY = P("grid.priority", 1400.0, 500.0, 2000.0, "a pylon that only just pays; about an extractor's priority")
SAFE_RADIUS = P("grid.safe_radius", 300.0, 100.0, 800.0, "no threat this close to a pylon")
STEP_FRAMES = 300
LOG_FRAMES = 1800
DEFAULT_INCOME = 2.0          # metal a second from an extractor whose spot isn't known


def link_range(u: UnitInfo) -> float:
    return float((u.data.get("customParams") or {}).get("pylonrange") or 0.0)


def generation(grid: list[UnitInfo]) -> float:
    """What a grid's generators cost, which is close enough to what they make."""
    return sum(units.cost(u.data) for u in grid if u.name in GENERATORS)


def extractors(grid: list[UnitInfo]) -> list[UnitInfo]:
    return [u for u in grid if u.name == "staticmex"]


class PylonGridMixin:
    def grid_components(self) -> list[list[UnitInfo]]:
        """Our finished grid members, split into separate grids."""
        nodes = [u for u in self.units.values() if u.is_finished and link_range(u) > 0.0]
        parent = list(range(len(nodes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if self.dist(a.pos, nodes[j].pos) < link_range(a) + link_range(nodes[j]):
                    parent[find(i)] = find(j)
        groups: dict[int, list[UnitInfo]] = {}
        for i, u in enumerate(nodes):
            groups.setdefault(find(i), []).append(u)
        return list(groups.values())

    def main_grid(self, components: list[list[UnitInfo]]) -> list[UnitInfo] | None:
        best = max(components, key=generation, default=None)
        return best if best is not None and generation(best) > 0.0 else None

    def mex_income(self, m: UnitInfo) -> float:
        index = self.spot_index_at(m.pos) if hasattr(self, "spot_index_at") else None
        income = self.map.spots[index].income if index is not None else 0.0
        return income or DEFAULT_INCOME

    def overdrive_total(self, mexes: list[UnitInfo], energy: float) -> float:
        if not mexes:
            return 0.0
        return sum(self.mex_income(m) for m in mexes) * 0.25 * math.sqrt(energy / len(mexes))

    def overdrive_gain(self, grid: list[UnitInfo], added: list[UnitInfo]) -> float:
        """Metal a second gained by spreading this grid's energy over more extractors."""
        energy = max(self.od_energy + self.od_waste, 0.0)
        mexes = extractors(grid)
        return self.overdrive_total(mexes + added, energy) - self.overdrive_total(mexes, energy)

    def bridge(self, grid: list[UnitInfo], outside: list[UnitInfo]) -> tuple[UnitInfo, UnitInfo, int] | None:
        """(grid member, extractor outside it, pylons between them) for the shortest gap."""
        best, best_gap = None, math.inf
        for m in extractors(outside):
            for n in grid:
                gap = self.dist(n.pos, m.pos) - link_range(n) - link_range(m)
                if gap < best_gap:
                    best, best_gap = (n, m), gap
        if best is None:
            return None
        span = 2.0 * (PYLON_RANGE - MARGIN)
        return best[0], best[1], max(1, math.ceil(best_gap / span))

    def pylon_point(self, n: UnitInfo, m: UnitInfo, count: int) -> tuple[float, float]:
        """Where the next pylon goes on the line from the grid to the extractor."""
        d = self.dist(n.pos, m.pos) or 1.0
        reach_grid = link_range(n) + PYLON_RANGE - MARGIN
        along = reach_grid
        if count == 1:
            reach_mex = link_range(m) + PYLON_RANGE - MARGIN
            along = (max(0.0, d - reach_mex) + min(d, reach_grid)) / 2.0
        return n.x + (m.x - n.x) / d * along, n.z + (m.z - n.z) / d * along

    def pylon_links(self, x: float, z: float) -> bool:
        """Would a pylon here join the main grid?  Placement can drift."""
        grid = self.main_grid(self.grid_components())
        return grid is not None and any(
            math.hypot(x - n.x, z - n.z) < link_range(n) + PYLON_RANGE - 10.0 for n in grid
        )

    def pylon_option(self, grid: list[UnitInfo], component: list[UnitInfo]) -> tuple | None:
        """(metal a second per pylon, pylons, where the next goes, extractors joined), if worth it."""
        found = self.bridge(grid, component)
        if found is None:
            return None
        n, m, count = found
        added = extractors(component)
        gain = self.overdrive_gain(grid, added)
        if gain <= 0.0 or count * PYLON_COST / gain > PAYBACK:
            return None
        point = self.pylon_point(n, m, count)
        if self.threat_in_radius(point, SAFE_RADIUS) > 0.0 or not self.site_safe_for_builder(point, "pylon"):
            return None
        return gain / count, count, point, len(added)

    def ensure_grid_pylons(self) -> None:
        if self.frame - self.__dict__.get("grid_frame", -10**9) < STEP_FRAMES:
            return
        self.grid_frame = self.frame
        components = self.grid_components()
        grid = self.main_grid(components)
        if grid is None:
            return
        outside = [c for c in components if c is not grid and extractors(c)]
        self.log_grid(grid, outside)
        if any(t.kind == "pylon" for t in self.tasks.values()):
            return
        if any(u.name == PYLON and u.is_being_built for u in self.units.values()):
            return
        did = units.id_of(PYLON)
        options = [o for o in (self.pylon_option(grid, c) for c in outside) if o is not None]
        if did is None or not options:
            return
        rate, count, point, joined = max(options, key=lambda o: o[0])
        # At a fixed priority below extractors, these went stale with nobody to build them.
        priority = PRIORITY + max(0.0, PAYBACK - PYLON_COST / rate)
        self.add_task(f"pylon:{self.frame}", "pylon", did, point[0], point[1], priority, anchor=point)
        self.log(f"pylon at {tuple(round(v) for v in point)}: {count} to join {joined} extractors, "
                 f"+{rate * count:.2f} metal/s")

    def log_grid(self, grid: list[UnitInfo], outside: list[list[UnitInfo]]) -> None:
        if self.frame - self.__dict__.get("grid_log_frame", -10**9) < LOG_FRAMES:
            return
        self.grid_log_frame = self.frame
        apart = sum(len(extractors(c)) for c in outside)
        pylons = sum(1 for u in self.units.values() if u.name == PYLON and u.is_finished)
        self.log(f"grid extractors joined={len(extractors(grid))} apart={apart} pylons={pylons} "
                 f"od_metal={self.od_metal:.1f} od_energy={self.od_energy:.1f} waste={self.od_waste:.1f}")
