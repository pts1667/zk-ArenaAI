"""Terrain / metal-spot sectors and strategic decisions.

The bot used to reason in map-global terms: "nearest metal spot", "fixed rally
point", "threat position".  That works on an open test map but does not
generalise and it makes armies chase single targets instead of controlling
ground.  This module builds a small sector graph from the things that actually
matter on a Zero-K map:

* bundles of nearby metal spots,
* terrain height / slope / water around each bundle,
* the two start areas,
* current friendly and enemy strength in each sector.

The rest of the bot can then ask for the safest expansion sector, the most
threatened sector, a defensive high-ground position, or the weakest enemy
economy sector to attack.  It is deliberately cheap: dozens of sectors, not a
full per-cell influence simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import units
from .mapdata import MetalSpot


@dataclass
class Sector:
    sid: int
    x: float
    z: float
    spots: list[MetalSpot] = field(default_factory=list)
    income: float = 0.0
    distance_base: float = 0.0
    distance_enemy: float = 0.0
    height: float = 0.0
    height_var: float = 0.0
    slope: float = 0.0
    water: float = 0.0
    defense_score: float = 1.0
    owner: str = "neutral"       # ours, enemy, neutral, front
    threat: float = 0.0
    our_strength: float = 0.0
    enemy_strength: float = 0.0
    enemy_buildings: float = 0.0
    last_seen: int = -10**9
    frontier: bool = False
    # Defence spending model inputs: how much static/mobile value is here,
    # and how close the local defenders are to breaking.
    defense_value: float = 0.0
    mobile_value: float = 0.0
    strain: float = 0.0


class StrategyMixin:
    def ensure_strategy(self) -> None:
        if getattr(self, "sectors", None):
            return
        if self.map.width <= 0 or not self.map.spots:
            return
        self._build_sectors()

    # ------------------------------------------------------------- sector build

    def _build_sectors(self) -> None:
        spots = list(self.map.spots)
        if not spots:
            self.sectors = []
            return
        w, h = self.map.width, self.map.height
        # Cluster radius scales with map size, but is capped so 32-spot maps
        # produce a handful of meaningful sectors rather than one blob.
        radius = max(420.0, min(680.0, math.hypot(w, h) * 0.085))
        parent = list(range(len(spots)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(spots):
            for j in range(i + 1, len(spots)):
                b = spots[j]
                if math.hypot(a.x - b.x, a.z - b.z) <= radius:
                    union(i, j)
        clusters: dict[int, list[int]] = {}
        for i in range(len(spots)):
            clusters.setdefault(find(i), []).append(i)

        # Add the base and the estimated enemy start as their own anchors when
        # no metal cluster is close enough.  That keeps "home" and "enemy
        # economy" sectors visible on low-metal or water maps.
        anchors: list[tuple[str, float, float]] = [("base", *self.base_pos)]
        if self.enemy_base:
            anchors.append(("enemy", *self.enemy_base))

        sectors: list[Sector] = []
        for indices in clusters.values():
            sx = sum(spots[i].x for i in indices) / len(indices)
            sz = sum(spots[i].z for i in indices) / len(indices)
            s = Sector(
                sid=len(sectors),
                x=sx,
                z=sz,
                spots=[spots[i] for i in indices],
                income=sum(spots[i].income for i in indices),
            )
            sectors.append(s)
        for _kind, ax, az in anchors:
            if any(self.dist((ax, az), (s.x, s.z)) < radius * 1.15 for s in sectors):
                continue
            sectors.append(Sector(sid=len(sectors), x=ax, z=az, spots=[]))

        self.sectors = sectors
        self.sector_built_frame = self.frame
        self.sector_update_frame = -10**9
        for s in sectors:
            s.distance_base = self.dist((s.x, s.z), self.base_pos)
            s.distance_enemy = self.dist(
                (s.x, s.z),
                self.enemy_base or (w * 0.5, h * 0.5),
            )
            self._score_sector_terrain(s)
            for spot in s.spots:
                self.sector_of_spot[spot.index] = s.sid
        self.log(
            "sectors "
            + " ".join(
                f"#{s.sid}({s.x:.0f},{s.z:.0f},M{s.income:.0f},d{s.defense_score:.1f})"
                for s in sectors
            )
        )
        self.update_sector_state()

    def _score_sector_terrain(self, s: Sector) -> None:
        """Sample height/slope/water; high defensible ground gets a bonus."""
        if self.map.heightmap is None:
            s.defense_score = 1.0
            return
        h, w = self.map.heightmap.shape
        sx = self.map.width / max(1, w)
        sz = self.map.height / max(1, h)
        cx = int(s.x / max(1.0, self.map.width) * w)
        cz = int(s.z / max(1.0, self.map.height) * h)
        r = max(3, int(380.0 / max(1.0, self.map.width) * w))
        values: list[float] = []
        slopes: list[float] = []
        water = 0
        samples = 0
        for z in range(max(0, cz - r), min(h, cz + r + 1), 3):
            for x in range(max(0, cx - r), min(w, cx + r + 1), 3):
                v = float(self.map.heightmap[z, x])
                values.append(v)
                samples += 1
                if v < -1.0:
                    water += 1
                if x + 1 < w:
                    slopes.append(abs(v - float(self.map.heightmap[z, x + 1])))
                if z + 1 < h:
                    slopes.append(abs(v - float(self.map.heightmap[z + 1, x])))
        if not values:
            s.defense_score = 1.0
            return
        avg = sum(values) / len(values)
        var = sum((v - avg) ** 2 for v in values) / len(values)
        s.height = avg
        s.height_var = math.sqrt(var)
        s.water = water / max(1, samples)
        s.slope = sum(slopes) / max(1, len(slopes)) * 100.0
        # Relative high ground and rough ground are good; large water areas
        # and very steep cells are bad for ground armies and buildings.
        high = sum(1 for v in values if v > avg + 6.0) / len(values)
        s.defense_score = (
            1.0
            + high * 1.4
            + min(0.5, s.height_var / 40.0)
            + min(0.4, s.slope)
            - s.water * 1.2
        )
        s.defense_score = max(0.25, s.defense_score)

    # ------------------------------------------------------------- sector query

    def sector_for_pos(self, pos: tuple[float, float]) -> Sector | None:
        if not getattr(self, "sectors", None):
            return None
        best = None
        best_d = 1e18
        for s in self.sectors:
            d = self.dist(pos, (s.x, s.z))
            if d < best_d:
                best = s
                best_d = d
        return best

    def sector_of_metal_spot(self, index: int) -> Sector | None:
        sid = getattr(self, "sector_of_spot", {}).get(index)
        if sid is None:
            return None
        return self.sectors[sid] if 0 <= sid < len(self.sectors) else None

    def sector_spot_status(self, s: Sector) -> str:
        statuses = {spot.status for spot in s.spots}
        if "ours" in statuses:
            return "ours"
        if "enemy" in statuses:
            return "enemy"
        if "building" in statuses:
            return "building"
        return "neutral"

    def update_sector_state(self) -> None:
        if not getattr(self, "sectors", None):
            return
        if self.frame - getattr(self, "sector_update_frame", -10**9) < 20:
            return
        self.sector_update_frame = self.frame
        for s in self.sectors:
            s.threat = 0.0
            s.our_strength = 0.0
            s.enemy_strength = 0.0
            s.enemy_buildings = 0.0
            s.defense_value = 0.0
            s.mobile_value = 0.0
            s.strain = 0.0
            s.owner = "neutral"
            s.frontier = False
        # Friendly mobile value and static extractors.
        for u in self.units.values():
            s = self.sector_for_pos(u.pos)
            if s is None:
                continue
            val = max(1.0, units.cost(u.data) if u.is_combat else 25.0)
            s.our_strength += val
            if u.is_finished:
                if units.is_defense(u.data):
                    s.defense_value += units.cost(u.data)
                elif u.is_mobile and u.is_combat:
                    s.mobile_value += units.cost(u.data)
                if u.alive_frac < 0.78:
                    s.strain += (1.0 - u.alive_frac) * (1.5 if units.is_defense(u.data) else 1.0)
                if u.uid in self.retreating:
                    s.strain += 0.45
            if u.name == "staticmex" and u.is_finished:
                s.owner = "ours"
        # Enemy contacts: value, threat and structures.
        for e in self.contacts().values():
            if not self.is_dangerous(e):
                continue
            s = self.sector_for_pos(e.pos)
            if s is None:
                continue
            val = max(1.0, units.cost(e.data) if e.is_combat else 100.0)
            s.enemy_strength += val
            s.threat += val
            s.last_seen = self.frame
            if e.is_factory or units.is_defense(e.data) or e.is_builder:
                s.enemy_buildings += val
        # Spot ownership is also a strong signal even when the extractor is
        # currently out of LOS.
        for s in self.sectors:
            status = self.sector_spot_status(s)
            if status == "ours":
                s.owner = "ours"
            elif status == "enemy":
                s.owner = "enemy"
            # A sector between our base and the enemy start that has enemy
            # presence is the current front.
            if s.threat > 0 and s.distance_base < self.dist(
                self.base_pos, self.enemy_base or self.map.center
            ) * 0.95:
                s.frontier = True
            elif s.owner == "enemy" and s.distance_base < 2600.0:
                s.frontier = True

    # ------------------------------------------------------------- decisions

    def most_threatened_sector(self) -> Sector | None:
        self.update_sector_state()
        if not getattr(self, "sectors", None):
            return None
        best = None
        best_score = 0.0
        for s in self.sectors:
            if s.threat <= 0:
                continue
            score = s.threat * (0.7 + min(1.5, s.income / 10.0))
            if s.owner == "ours":
                score *= 1.4
            if s.distance_base < 1800.0:
                score *= 1.5
            if score > best_score:
                best_score = score
                best = s
        return best

    def best_defense_pos(self, s: Sector | None) -> tuple[float, float] | None:
        if s is None:
            return None
        # Prefer high ground near the sector centre.  Sample a few points and
        # pick the highest one that is not water.
        best = (s.x, s.z)
        if self.map.heightmap is not None:
            best_h = self.map.ground_height(s.x, s.z)
            for dx, dz in ((0, 0), (180, 0), (-180, 0), (0, 180), (0, -180),
                           (140, 140), (-140, 140), (140, -140), (-140, -140)):
                x, z = s.x + dx, s.z + dz
                if x < 40 or z < 40 or x > self.map.width - 40 or z > self.map.height - 40:
                    continue
                h = self.map.ground_height(x, z)
                if h > best_h and not self.map.is_water(x, z):
                    best_h = h
                    best = (x, z)
        return best

    def safe_expansion_sector(self) -> Sector | None:
        self.update_sector_state()
        if not getattr(self, "sectors", None):
            return None
        best = None
        best_score = 0.0
        for s in self.sectors:
            if not s.spots:
                continue
            if s.owner == "enemy" or s.threat > 0:
                continue
            if s.distance_base > 5200.0:
                continue
            score = (
                s.income * 18.0
                + s.defense_score * 55.0
                - s.distance_base * 0.055
                - s.distance_enemy * 0.015
            )
            if score > best_score:
                best_score = score
                best = s
        return best

    def best_attack_sector(self) -> Sector | None:
        self.update_sector_state()
        if not getattr(self, "sectors", None):
            return None
        best = None
        best_score = -1e18
        for s in self.sectors:
            if s.owner != "enemy" and s.threat <= 0:
                continue
            # Enemy economy and support structures are the best targets.
            value = s.income * 30.0 + s.enemy_buildings * 0.2 + 40.0
            score = value - s.distance_base * 0.05 - s.enemy_strength * 0.7
            if s.frontier:
                score += 60.0
            if score > best_score:
                best_score = score
                best = s
        return best

    def frontier_rally(self) -> tuple[float, float]:
        self.update_sector_state()
        if getattr(self, "sectors", None):
            owned = [s for s in self.sectors if s.owner == "ours" and s.distance_base > 400]
            if owned:
                s = max(owned, key=lambda q: q.distance_enemy)
                p = self.best_defense_pos(s)
                if p is not None:
                    return p
            front = [s for s in self.sectors if s.frontier]
            if front:
                s = max(front, key=lambda q: q.threat)
                p = self.best_defense_pos(s)
                if p is not None:
                    return p
        bx, bz = self.base_pos
        ex, ez = self.enemy_base or self.map.center
        return bx + (ex - bx) * 0.25, bz + (ez - bz) * 0.25
