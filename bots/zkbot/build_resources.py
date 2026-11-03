"""ResourceBuildMixin: Build tasks for metal and energy: extractors, generators, storage, pylons.

Split out of economy.py; EconomyMixin.economy_step calls these.
"""
from __future__ import annotations

import math

from . import units


class ResourceBuildMixin:
    def ensure_mex_tasks(self, limit: int | None = None) -> None:
        if not self.map.spots:
            return
        self.ensure_strategy()
        self.update_sector_state()
        if limit is None:
            t = self.frame_time()
            if t < 90:
                limit = 7
            elif t < 240:
                limit = 12
            elif t < 480:
                limit = 20
            else:
                limit = 32
        bx, bz = self.base_pos
        free = [
            s
            for s in self.map.spots
            if s.status == "free"
            and s.task_id is None
            and self.blocked_spot_until.get(s.index, -1) < self.frame
        ]
        # CircuitAI Hard expands to every safe metal spot almost immediately.
        # A tiny 1400-elmo comfort circle loses the economy race before the
        # first fight, so expand by map distance and only pull the radius in
        # when something is actually attacking the base.
        t = self.frame_time()
        if t < 120:
            safe_radius = 2400.0
        elif t < 240:
            safe_radius = 3200.0
        elif t < 360:
            safe_radius = 4000.0
        elif t < 600:
            safe_radius = 5000.0
        else:
            safe_radius = 6000.0
        if self.threat_pos is not None and self.frame - self.threat_frame < 300:
            if self.dist(self.threat_pos, self.base_pos) < 1600.0:
                safe_radius = min(safe_radius, 2200.0)
        if self.combat_count() < 4:
            safe_radius = min(safe_radius, 2600.0)
        free = [s for s in free if math.hypot(s.x - bx, s.z - bz) <= safe_radius]
        pending = sum(1 for t in self.tasks.values() if t.kind == "mex")
        room = max(0, limit - pending)
        room = min(room, max(6, self.builder_count() * 2) - pending)
        if room <= 0:
            return
        enemy_base = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        scored: list[tuple[float, MetalSpot]] = []
        for s in free:
            db = math.hypot(s.x - bx, s.z - bz)
            de = math.hypot(s.x - enemy_base[0], s.z - enemy_base[1])
            # Extractor income compounds; CircuitAI Hard's whole edge is
            # that it gets all of its safe mexes down first.  Keep extractors
            # above early turrets and generators, otherwise the opening metal
            # pool is spent on defense and the expansion stalls at one mex.
            early = 900.0 if t < 240 else (350.0 if t < 480 else 0.0)
            priority = 1450.0 + early - db * 0.045 + s.income * 20.0
            sector = self.sector_of_metal_spot(s.index)
            if sector is not None:
                # Sector-aware expansion: a bundle of mexes on defensible
                # high ground is worth more than a lone spot in a threatened
                # corridor.  This is the map-general replacement for the old
                # radius-only expansion rule.
                if sector.owner == "enemy":
                    priority -= 500.0
                if sector.threat > 0:
                    priority -= min(700.0, sector.threat * 2.5)
                    if t < 420:
                        priority -= 250.0
                if sector.frontier:
                    priority -= 120.0
                priority += sector.defense_score * 35.0
                priority += sector.income * 8.0
            if de < 1100:
                priority -= 90.0
            # Uncovered enemy DPS on the spot (including remembered turrets).
            risk = self.threat_in_radius((s.x, s.z), 160.0) - self.own_power_at((s.x, s.z))
            if risk > 0:
                priority -= min(900.0, 120.0 + risk * 4.0)
            scored.append((priority, s))
        scored.sort(key=lambda p: p[0], reverse=True)
        did = units.id_of("staticmex")
        if did is None:
            return
        for priority, s in scored[:room]:
            key = f"mex:{s.index}"
            if key in self.tasks:
                continue
            if any(
                u.name == "staticmex"
                and u.is_finished
                and self.dist(u.pos, (s.x, s.z)) < 130.0
                for u in self.units.values()
            ):
                continue
            self.add_task(
                key,
                "mex",
                did,
                s.x,
                s.z,
                priority,
                anchor=(s.x, s.z),
                spot_index=s.index,
            )
            s.task_id = key
            s.status = "building"

    def energy_anchor(self, kind: str = "energy") -> tuple[float, float]:
        """Keep new generators inside the base perimeter.

        The old version chased the nearest extractor cluster.  On a corner
        start that sent every constructor and every solar across the map,
        where raiders killed them before they finished.  Generators around
        the base still link to the first extractors; later pylons can bridge
        the overdrive grid.
        """
        bx, bz = self.base_pos
        radius = 165.0 if kind == "fusion" else 135.0
        angle = math.radians((self.energy_task_seq * 137.5 + 20.0) % 360.0)
        cx, cz = bx, bz
        if kind == "energy":
            # Pack solars around safe owned extractors near home so they join
            # the overdrive grid directly (an isolated base ring could not be
            # placed once the base mexes were gone).
            mexes = [
                u
                for u in self.units.values()
                if u.name == "staticmex"
                and u.is_finished
                and self.dist(u.pos, self.base_pos) < 1600.0
                and self.threat_at(u.pos) <= 0.0
            ]
            best = None
            best_score = 1e18
            for m in mexes:
                crowd = sum(
                    1
                    for g in self.units.values()
                    if g.name == "energysolar" and self.dist(g.pos, m.pos) < 200.0
                )
                if crowd >= 6:
                    continue
                score = self.dist(m.pos, self.base_pos) + crowd * 250.0
                if score < best_score:
                    best_score = score
                    best = m
            if best is not None:
                cx, cz = best.pos
                radius = 95.0
        x = cx + math.cos(angle) * radius
        z = cz + math.sin(angle) * radius
        x = min(max(40.0, x), max(41.0, self.map.width - 40.0))
        z = min(max(40.0, z), max(41.0, self.map.height - 40.0))
        return x, z

    def ensure_energy_tasks(self, force: int = 0) -> None:
        minc = self.metal_income()
        einc = self.energy_income()
        ecur = self.energy_current()
        estore = self.effective_storage(self.energy())
        # Do not overbuild 2-energy solar panels just because the energy bar
        # is not completely full.  On a 20-metal map 25-30 panels are plenty
        # for the overdrive grid; every extra pair costs a combat unit.
        # Overdrive turns surplus energy into metal, so the energy target
        # grows with time like CircuitAI's income factor (0.15 -> 7 over an
        # hour): ~0.7x metal income at 5 min, ~1.3x at 10, ~1.9x at 15.
        factor = 0.15 + 6.85 * min(1.0, self.frame_time() / 3600.0)
        # Overdrive pays per extractor with diminishing returns, so surplus
        # energy is only worth it with enough mexes to spread it over.
        mexes = self.mex_count()
        # Zero-K construction drains energy 1:1 with metal, so energy below
        # metal income caps spending and the rest of the metal piles up.
        base_e = minc * 1.1 + 4.0
        od_cap = base_e + 6.0 * mexes
        target = max(10.0, base_e, min(minc * factor, od_cap), self.energy_usage() + 8.0)
        m_float = self.metal_current() > 0.5 * self.effective_storage(self.metal())
        energy_bound = m_float and einc < minc and ecur < estore * 0.5
        deficit = target - einc
        pending = sum(1 for t in self.tasks.values() if t.kind in ("energy", "fusion"))
        solar_existing = sum(
            1
            for u in self.units.values()
            if u.name == "energysolar" and (u.is_finished or u.is_being_built)
        )
        max_solar = int(max(12.0, target * 0.5)) + 4
        max_pending = 5 if energy_bound else 3
        usage = self.energy_usage()
        critical = (
            ecur < max(80.0, estore * 0.12)
            or einc < usage
        )
        # The first couple of generators have to win the task queue.  On a
        # 350-metal start, assigning every constructor to extractor jobs first
        # leaves energy at zero for minutes; the factory then cannot replace
        # losses and the base collapses.  Once two solar exist, mexes take over.
        if solar_existing < 2 and (self.mex_count() >= 2 or ecur < 120.0):
            energy_priority = 2600.0
        elif solar_existing < 2:
            # Extractors first while the starting energy lasts.
            energy_priority = 1300.0
        elif critical and ecur < max(20.0, estore * 0.05):
            # Units and construction both consume energy.  With the bar at
            # zero the factory simply stops producing even when metal is
            # capped, so an empty bar has to outrank extractor expansion.
            energy_priority = 2600.0
        elif critical or energy_bound:
            # Energy-bound with metal floating: every solar converts idle
            # metal into build power, so it outranks expansion.
            energy_priority = 2200.0
        else:
            energy_priority = 1000.0
        # Priorities are stored on the task at creation.  Update existing
        # energy jobs so an emergency solar does not keep outranking mexes
        # after the emergency has passed.
        for t in self.tasks.values():
            if t.kind == "energy":
                t.priority = energy_priority
        if (
            force == 0
            and ecur > estore * 0.85
            and (self.od_waste >= 2.0 or einc >= target)
        ):
            # The bar is full, nothing is being wasted, and construction is
            # covered.  More generators would only starve metal expansion.
            return
        if deficit <= 0 and ecur > estore * 0.45 and force == 0:
            return
        # A fusion reactor is the efficient mid-game answer.  Wait until the
        # economy can actually pay for it, otherwise it stalls all expansion.
        if (
            minc >= 20
            and einc < max(target, self.energy_usage() + 6.0)
            and mexes >= 8 + 4 * sum(
                1 for u in self.units.values() if u.name == "energyfusion"
            )
            and (self.metal_current() > 400 or minc >= 25)
            and self.threat_at(self.base_pos) <= 0.0
            and not any(t.kind == "fusion" for t in self.tasks.values())
            and len(self.factory_list()) > 0
        ):
            did = units.id_of("energyfusion")
            if did:
                anchor = self.energy_anchor("fusion")
                fusion_priority = (
                    1450.0
                    if ecur < max(80.0, estore * 0.12) or einc < usage
                    else 1040.0
                )
                self.add_task(
                    f"fusion:{self.frame}:{self.energy_task_seq}",
                    "fusion",
                    did,
                    anchor[0],
                    anchor[1],
                    fusion_priority,
                    anchor=anchor,
                    max_assignees=2,
                )
                self.energy_task_seq += 1
                return
        if pending >= max_pending:
            return
        if solar_existing + pending >= max_solar:
            return
        did = units.id_of("energysolar")
        if did is None:
            return
        # Create enough jobs to cover the deficit.  Each solar is 2 E/s.
        count = 1
        if energy_bound:
            count = max(1, math.ceil(deficit / 2.0))
        count = min(count, max_pending - pending)
        if count <= 0:
            return
        for _ in range(count):
            anchor = self.energy_anchor("energy")
            self.add_task(
                f"energy:{self.frame}:{self.energy_task_seq}",
                "energy",
                did,
                anchor[0],
                anchor[1],
                energy_priority,
                anchor=anchor,
            )
            self.energy_task_seq += 1

    def ensure_storage_tasks(self) -> None:
        m = self.metal()
        e = self.energy()
        m_cur = float(m.get("current") or 0)
        m_stor = self.effective_storage(m)
        e_cur = float(e.get("current") or 0)
        e_stor = self.effective_storage(e)
        einc = self.energy_income()
        # Storage is not just a buffer: overdrive is strongest when the energy
        # bar is full, so a few storage buildings keep surplus energy useful.
        target = 1 + (1 if einc > 15 else 0) + (1 if einc > 40 else 0)
        existing = sum(
            1
            for u in self.units.values()
            if u.name == "staticstorage" and (u.is_finished or u.is_being_built)
        )
        pending = sum(1 for t in self.tasks.values() if t.kind == "storage")
        if existing + pending >= target:
            return
        if not (m_cur > m_stor * 0.90 or e_cur > e_stor * 0.85):
            return
        did = units.id_of("staticstorage")
        if did is None:
            return
        self.add_task(
            f"storage:{self.frame}",
            "storage",
            did,
            self.base_pos[0],
            self.base_pos[1],
            990.0,
            anchor=self.base_pos,
        )

    def ensure_pylon_tasks(self) -> None:
        # Energy waste while overdrive metal is low usually means generator
        # grids are not linked to the mex grids.  A pylon bridges 500 elmos.
        if self.od_waste < 4.0 or self.od_metal < 0.1:
            return
        if any(
            u.name == "energypylon" and (u.is_finished or u.is_being_built)
            for u in self.units.values()
        ):
            return
        if any(t.kind == "pylon" for t in self.tasks.values()):
            return
        did = units.id_of("energypylon")
        if did is None:
            return
        anchor = self.energy_anchor("fusion")
        self.add_task(
            f"pylon:{self.frame}",
            "pylon",
            did,
            anchor[0],
            anchor[1],
            720.0,
            anchor=anchor,
        )
