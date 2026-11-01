"""EconomyMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class EconomyMixin:
    def economy_step(self) -> None:
        self.defense_step()
        self.cleanup_tasks()
        self.ensure_mex_tasks()
        self.ensure_mex_defense_tasks()
        self.ensure_energy_tasks()
        self.ensure_storage_tasks()
        self.ensure_pylon_tasks()
        self.ensure_factory_tasks()
        self.ensure_strider_tasks()
        self.ensure_caretaker_tasks()
        self.ensure_radar_tasks()
        self.ensure_defense_tasks()
        self.ensure_front_defense_tasks()
        self.ensure_silo_tasks()
        self.ensure_seismic_tasks()
        self.production_step()
        self.assign_builders()

    # -------------------------------------------------------------- task setup

    def add_task(
        self,
        key: str,
        kind: str,
        def_id: int,
        x: float,
        z: float,
        priority: float,
        *,
        anchor: tuple[float, float] | None = None,
        spot_index: int | None = None,
        max_assignees: int = 1,
    ) -> Task:
        t = Task(
            key=key,
            kind=kind,
            def_id=int(def_id),
            x=float(x),
            z=float(z),
            priority=float(priority),
            created=self.frame_time(),
            created_frame=self.frame,
            anchor=anchor,
            spot_index=spot_index,
            max_assignees=max_assignees,
        )
        self.tasks[key] = t
        return t

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

    def ensure_mex_defense_tasks(self) -> None:
        """Put a range-appropriate turret on the forward extractors.

        A lone extractor is free metal for the first raider that finds it,
        but the old version always built a Lotus and built up to nine of
        them regardless of pressure.  The turret type now follows the range
        of recently attacking units, and the budget gate keeps static defence
        from crowding out mobile production.
        """
        if self.frame_time() < 150 or self.metal_income() < 8:
            return
        mexes = [
            u
            for u in self.units.values()
            if u.name == "staticmex" and u.is_finished
        ]
        if not mexes:
            return
        ex, ez = self.enemy_base or (
            self.map.width * 0.5,
            self.map.height * 0.5,
        )
        mexes.sort(key=lambda u: math.hypot(u.x - ex, u.z - ez))
        existing = [
            u
            for u in self.units.values()
            if units.is_defense(u.data) and not u.is_mobile and u.is_finished
        ]
        pending = [t for t in self.tasks.values() if t.kind == "mexdef"]
        desired = self.desired_defense_share()
        cap = max(2, min(8, int(2 + desired * 18)))
        if len(existing) + len(pending) >= cap:
            return
        for m in mexes[:6]:
            if self.frame_time() < 360 and self.dist(m.pos, self.base_pos) > 1500.0:
                continue
            if any(self.dist(m.pos, u.pos) < 300 for u in existing):
                continue
            if any(t.anchor and self.dist(m.pos, t.anchor) < 300 for t in pending):
                continue
            if len(pending) >= 3:
                break
            did = self.defense_unit_for(m.pos)
            if did is None:
                continue
            # 2200-metal anti-heavy turrets belong in the base line, not on
            # every forward extractor.  If enemy artillery is the only thing
            # that outranges the spot, leave the extractor to the mobile army.
            if units.name_of(did) == "turretantiheavy":
                continue
            cost = units.cost(units.by_id(did))
            sector = self.sector_for_pos(m.pos)
            pressure = self.sector_attack_pressure_value(sector) if sector else 0.0
            urgent = pressure > 0.35 or self.enemy_near(m.pos, 650.0) is not None
            if not self.defense_budget_ok(cost, urgent=urgent):
                continue
            self.log(
                f"mexdef {units.name_of(did)} at {tuple(round(v) for v in m.pos)} "
                f"req={self.recent_attacker_range_for(m.pos):.0f} urgent={urgent}"
            )
            self.add_task(
                f"mexdef:{self.frame}:{m.uid}",
                "mexdef",
                did,
                m.x,
                m.z,
                1450.0 + min(450.0, pressure * 28.0) if urgent else 1260.0,
                anchor=m.pos,
            )

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
        x = bx + math.cos(angle) * radius
        z = bz + math.sin(angle) * radius
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
        target = max(10.0, minc * 0.7 + 5.0, self.energy_usage() + 8.0)
        deficit = target - einc
        pending = sum(1 for t in self.tasks.values() if t.kind in ("energy", "fusion"))
        solar_existing = sum(
            1
            for u in self.units.values()
            if u.name == "energysolar" and (u.is_finished or u.is_being_built)
        )
        max_solar = int(max(12.0, target * 0.5)) + 4
        max_pending = 3
        usage = self.energy_usage()
        critical = (
            ecur < max(80.0, estore * 0.12)
            or einc < usage
        )
        # The first couple of generators have to win the task queue.  On a
        # 350-metal start, assigning every constructor to extractor jobs first
        # leaves energy at zero for minutes; the factory then cannot replace
        # losses and the base collapses.  Once two solar exist, mexes take over.
        if solar_existing < 2:
            energy_priority = 2600.0
        elif critical and ecur < max(20.0, estore * 0.05):
            # Units and construction both consume energy.  With the bar at
            # zero the factory simply stops producing even when metal is
            # capped, so an empty bar has to outrank extractor expansion.
            energy_priority = 2600.0
        elif critical:
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
            and self.od_waste < 2.0
            and einc >= usage * 0.8
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
            and einc < max(minc * 1.02, self.energy_usage() + 6.0)
            and (self.metal_current() > 600 or minc > 35)
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
        if pending == 0 and force == 0:
            count = max(count, 1)
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

    def ensure_factory_tasks(self) -> None:
        factories = [
            u
            for u in self.units.values()
            if u.is_factory and (u.is_finished or u.is_being_built)
        ]
        if self.factory_plop_sent and not factories:
            return
        minc = self.metal_income()
        target = 1
        t = self.frame_time()
        if (minc >= 18 and t > 180) or t > 300:
            target = 2
        if (minc >= 30 and t > 420) or t > 600:
            target = 3
        if (minc >= 45 and t > 720) or t > 900:
            target = 4
        pending = sum(1 for t in self.tasks.values() if t.kind == "factory")
        if len(factories) + pending >= target:
            return
        plane_def = units.id_of("factoryplane")
        has_plane = any(
            u.name == "factoryplane" and (u.is_finished or u.is_being_built)
            for u in self.units.values()
        )
        plane_pending = any(
            t.kind == "factory" and t.def_id == plane_def for t in self.tasks.values()
        )
        # Get a second ground factory first; it turns metal into units much
        # more efficiently than an early air plant.  Air is useful for
        # scouting and commander hunting once the economy can support it.
        want_plane = (
            target >= 4
            and minc >= 40
            and t > 900
            and not has_plane
            and not plane_pending
        )
        if want_plane:
            did = plane_def
        else:
            did = self.factory_def_id or units.id_of("factorycloak")
        if did is None:
            return
        bx, bz = self.base_pos
        cx, cz = self.map.width * 0.5, self.map.height * 0.5
        dx, dz = cx - bx, cz - bz
        length = math.hypot(dx, dz) or 1.0
        # Build inland, not into the map corner: several factory sites near the
        # edge are rejected by the engine and leave a builder reissuing forever.
        side = 1.0 if (self.factory_task_seq % 2 == 0) else -1.0
        px, pz = -dz / length, dx / length
        anchor = (
            bx + dx / length * 260.0 + side * px * 200.0,
            bz + dz / length * 260.0 + side * pz * 200.0,
        )
        margin = 360.0
        anchor = (
            min(max(margin, anchor[0]), self.map.width - margin),
            min(max(margin, anchor[1]), self.map.height - margin),
        )
        self.add_task(
            f"factory:{self.frame}:{self.factory_task_seq}",
            "factory",
            int(did),
            anchor[0],
            anchor[1],
            # A second factory is the cheapest way to convert a capped metal
            # bank into army.  At the old 1480 priority every energy/defence
            # job outranked it and the bot sat on 500 metal with one factory.
            2350.0,
            anchor=anchor,
            max_assignees=2,
        )
        self.factory_task_seq += 1

    def ensure_strider_tasks(self) -> None:
        if self.frame_time() < 720 or self.metal_income() < 35:
            return
        did = units.id_of("striderhub")
        if did is None:
            return
        if any(
            u.name == "striderhub" and (u.is_finished or u.is_being_built)
            for u in self.units.values()
        ):
            return
        if any(t.kind == "strider" for t in self.tasks.values()):
            return
        self.add_task(
            f"strider:{self.frame}",
            "strider",
            did,
            self.base_pos[0],
            self.base_pos[1],
            1520.0,
            anchor=self.base_pos,
            max_assignees=2,
        )

    def ensure_caretaker_tasks(self) -> None:
        minc = self.metal_income()
        factories = self.factory_list()
        if not factories or minc < 7:
            return
        did = units.id_of("staticcon")
        if did is None:
            return
        target_per_factory = 1
        if minc >= 25:
            target_per_factory = 2
        if minc >= 45:
            target_per_factory = 3
        for f in factories:
            existing = [
                u
                for u in self.units.values()
                if u.name == "staticcon"
                and (u.is_finished or u.is_being_built)
                and self.dist(u.pos, f.pos) < 750
            ]
            pending = [
                t
                for t in self.tasks.values()
                if t.kind == "caretaker"
                and t.anchor
                and self.dist(t.anchor, f.pos) < 750
            ]
            if len(existing) + len(pending) >= target_per_factory:
                continue
            key = f"caretaker:{f.uid}:{len(existing)+len(pending)}"
            if key in self.tasks:
                continue
            self.add_task(
                key,
                "caretaker",
                did,
                f.x + 70.0,
                f.z + 70.0,
                1040.0,
                anchor=f.pos,
            )

    def ensure_radar_tasks(self) -> None:
        if self.frame_time() < 20:
            return
        did = units.id_of("staticradar")
        if did is None:
            return
        existing = [
            u
            for u in self.units.values()
            if u.name == "staticradar" and (u.is_finished or u.is_being_built)
        ]
        pending = [t for t in self.tasks.values() if t.kind == "radar"]
        arty = self.artillery_units() if hasattr(self, "artillery_units") else []
        # Radar is the target feed for artillery.  Keep a few more towers
        # when we actually own guns that can use the blips.
        target = 3 + min(3, len(arty) // 2)
        if len(existing) + len(pending) >= target:
            return
        anchors: list[tuple[float, float]] = [self.base_pos]
        if self.frame_time() > 120:
            anchors.append(self.rally_point())
        if self.frame_time() > 240 and self.enemy_base:
            ours = [s for s in self.map.spots if s.status == "ours"]
            if ours:
                ex, ez = self.enemy_base
                front = min(ours, key=lambda sp: math.hypot(sp.x - ex, sp.z - ez))
                anchors.append((front.x, front.z))
        # With artillery, put a radar near the guns / attack sector so the
        # next target is already painted before the guns acquire it.
        if arty:
            cx = sum(u.x for u in arty) / len(arty)
            cz = sum(u.z for u in arty) / len(arty)
            sector = self.sector_for_pos((cx, cz))
            p = self.best_defense_pos(sector) if sector is not None else None
            anchors.append(p or (cx, cz))
            attack = self.best_attack_sector()
            if attack is not None:
                p = self.best_defense_pos(attack)
                if p is not None and self.dist(p, self.base_pos) < 3800.0:
                    anchors.append(p)
        for a in anchors:
            if a is None:
                continue
            if self.enemy_near(a, 500.0) is not None:
                continue
            # Do not send an unescorted builder to a distant forward point;
            # radar is useful, but not at the cost of the only constructor.
            if self.dist(a, self.base_pos) > 1900.0 and not any(
                self.dist(self.units[uid].pos, a) < 700.0
                for uid in self.army
                if uid in self.units
            ):
                continue
            # 2100-elmo radar radius; a tower within ~1900 already covers
            # this anchor.  Avoid building three redundant towers on one spot.
            if any(self.dist(a, u.pos) < 1900.0 for u in existing):
                continue
            if any(t.anchor and self.dist(a, t.anchor) < 1900.0 for t in pending):
                continue
            radar_priority = 1760.0 if arty else (1480.0 if not existing else 1320.0)
            self.add_task(
                f"radar:{self.frame}:{len(existing)+len(pending)}",
                "radar",
                did,
                a[0],
                a[1],
                radar_priority,
                anchor=a,
            )
            return

    def ensure_silo_tasks(self) -> None:
        if self.frame_time() < 600 or self.metal_income() < 40:
            return
        did = units.id_of("staticmissilesilo")
        if did is None:
            return
        if any(
            u.name == "staticmissilesilo" and (u.is_finished or u.is_being_built)
            for u in self.units.values()
        ):
            return
        if any(t.kind == "silo" for t in self.tasks.values()):
            return
        self.add_task(
            f"silo:{self.frame}",
            "silo",
            did,
            self.base_pos[0],
            self.base_pos[1],
            1070.0,
            anchor=self.base_pos,
            max_assignees=2,
        )

    def ensure_seismic_tasks(self) -> None:
        silos = [
            u
            for u in self.units.values()
            if u.name == "staticmissilesilo" and u.is_finished
        ]
        if not silos:
            return
        did = units.id_of("seismic")
        if did is None:
            return
        existing = sum(
            1
            for u in self.units.values()
            if u.name == "seismic" and (u.is_finished or u.is_being_built)
        )
        pending = sum(1 for t in self.tasks.values() if t.kind == "seismic")
        if existing + pending >= 1:
            return
        silo = silos[0]
        self.add_task(
            f"seismic:{self.frame}",
            "seismic",
            did,
            silo.x,
            silo.z,
            1180.0,
            anchor=silo.pos,
            max_assignees=1,
        )

    def ensure_front_defense_tasks(self) -> None:
        # Forward turrets are useful once there is an army and an economy to
        # spare.  Their type follows recent attacker range; the budget gate
        # prevents a turret line from taking metal from the mobile army.
        if self.frame_time() < 900 or self.combat_count() < 20:
            return
        mexes = [
            u
            for u in self.units.values()
            if u.name == "staticmex" and u.is_finished
        ]
        if not mexes:
            return
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        mexes.sort(key=lambda u: math.hypot(u.x - ex, u.z - ez))
        existing = [
            u
            for u in self.units.values()
            if units.is_defense(u.data) and not u.is_mobile and u.is_finished
        ]
        pending = [t for t in self.tasks.values() if t.kind == "frontdef"]
        desired = self.desired_defense_share()
        cap = max(1, min(5, int(1 + desired * 10)))
        if len(pending) >= cap:
            return
        for t in pending:
            t.priority = 1500.0 if self.enemy_near((t.x, t.z), 650) else 1260.0
        for m in mexes[:4]:
            if any(self.dist(m.pos, u.pos) < 450 for u in existing):
                continue
            if any(t.anchor and self.dist(m.pos, t.anchor) < 450 for t in pending):
                continue
            if len(pending) >= 2:
                break
            did = self.defense_unit_for(m.pos)
            if did is None:
                continue
            if units.name_of(did) == "turretantiheavy":
                continue
            cost = units.cost(units.by_id(did))
            urgent = self.enemy_near(m.pos, 650) is not None
            if not self.defense_budget_ok(cost, urgent=urgent):
                continue
            self.log(
                f"frontdef {units.name_of(did)} at {tuple(round(v) for v in m.pos)} "
                f"req={self.recent_attacker_range_for(m.pos):.0f} urgent={urgent}"
            )
            self.add_task(
                f"frontdef:{self.frame}:{m.uid}",
                "frontdef",
                did,
                m.x,
                m.z,
                1520.0 if urgent else 1260.0,
                anchor=m.pos,
            )

    def ensure_defense_tasks(self) -> None:
        threat = self.threat_pos if self.frame - self.threat_frame < 240 else None
        # A distant contact is the mobile army's problem.  Putting a turret
        # 2 km from home just feeds the builder to the next raider and leaves
        # the base without a line.
        if threat is not None and self.dist(threat, self.base_pos) > 1700.0:
            threat = None
        # AA is only justified after a real aircraft has been observed.
        # Commander drones and the radar-tower morph are excluded in
        # defense.note_air_contact(), so they never trigger this branch.
        if self.aircraft_seen():
            aa = sum(
                1
                for u in self.units.values()
                if u.name.startswith("turretaa") and (u.is_finished or u.is_being_built)
            )
            if aa == 0 and not any(t.kind == "aa" for t in self.tasks.values()):
                did = self.defense_unit_for(self.base_pos, aa=True)
                if did is not None:
                    cost = units.cost(units.by_id(did))
                    if self.defense_budget_ok(cost, urgent=self.aircraft_seen_recent()):
                        self.log(f"aa turret {units.name_of(did)} real aircraft seen")
                        self.add_task(
                            f"aa:{self.frame}",
                            "aa",
                            did,
                            self.base_pos[0],
                            self.base_pos[1],
                            1300.0 if self.aircraft_seen_recent() else 1180.0,
                            anchor=self.base_pos,
                        )
        if threat is None and self.frame_time() < 55:
            return
        t = self.frame_time()
        desired = self.desired_defense_share()
        # The defence cap now follows the measured pressure instead of a
        # fixed five-turret quota.
        defense_cap = max(1, min(8, int(1 + desired * 22)))
        if t < 120:
            defense_cap = min(defense_cap, 1)
        elif t < 240:
            defense_cap = min(defense_cap, 2)
        elif t < 420:
            defense_cap = min(defense_cap, 4)
        if sum(1 for q in self.tasks.values() if q.kind in ("defense", "aa")) >= defense_cap:
            return
        bx, bz = self.base_pos
        anchors: list[tuple[float, float]] = []
        threatened = self.most_threatened_sector()
        # Only fortify a sector we actually own near home.  Sending
        # constructors to a far neutral sector just feeds CircuitAI raiders
        # and leaves the base without turrets.
        if threatened is not None and (
            threatened.distance_base >= 2200.0 or threatened.owner != "ours"
        ):
            threatened = None
        sector_anchor = self.best_defense_pos(threatened)
        if sector_anchor is not None:
            # Hold the threatened sector's high ground instead of placing
            # every turret on a fixed ring around the start position.
            anchors.append(sector_anchor)
        elif threat:
            # Build the turret on the base side of the threat, not at the
            # threat itself where constructors are easy prey.
            dx, dz = threat[0] - bx, threat[1] - bz
            length = math.hypot(dx, dz) or 1.0
            anchors.append((bx + dx / length * 210.0, bz + dz / length * 210.0))
        else:
            # A compact perimeter of heavy turrets is what actually holds
            # against CircuitAI's first big push.  Lay it out toward the map
            # centre: the start box is often in a corner, and turrets placed
            # against the map edge can be unreachable or partially off-map.
            cx, cz = self.map.width * 0.5, self.map.height * 0.5
            dx, dz = cx - bx, cz - bz
            length = math.hypot(dx, dz) or 1.0
            tx, tz = dx / length, dz / length
            px, pz = -tz, tx
            for toward, side in ((170.0, 0.0), (110.0, 180.0), (110.0, -180.0), (300.0, 0.0)):
                x = bx + tx * toward + px * side
                z = bz + tz * toward + pz * side
                x = min(max(300.0, x), self.map.width - 300.0)
                z = min(max(300.0, z), self.map.height - 300.0)
                anchors.append((x, z))
        # With no contact, put one turret on a forward owned sector that is
        # still close enough to support; distant sectors are left to the army.
        if not anchors and getattr(self, "sectors", None):
            owned = [
                s
                for s in self.sectors
                if s.owner == "ours" and 500 < s.distance_base < 2000
            ]
            if owned:
                s = max(owned, key=lambda q: q.distance_enemy)
                p = self.best_defense_pos(s)
                if p is not None:
                    anchors.append(p)
        for a in anchors:
            if any(self.dist(a, p) < 140 for p in self.defense_built_near):
                continue
            if any(
                q.anchor and self.dist(a, q.anchor) < 140
                for q in self.tasks.values()
                if q.kind == "defense"
            ):
                continue
            did = self.defense_unit_for(a)
            if did is None:
                continue
            if units.name_of(did) == "turretantiheavy":
                antiheavy = sum(
                    1
                    for u in self.units.values()
                    if u.name == "turretantiheavy" and (u.is_finished or u.is_being_built)
                ) + sum(
                    1
                    for q in self.tasks.values()
                    if q.kind == "defense" and units.name_of(q.def_id) == "turretantiheavy"
                )
                if antiheavy >= 1:
                    continue
            cost = units.cost(units.by_id(did))
            urgent = (
                sector_anchor is not None
                and threatened is not None
            ) or (threat is not None and self.dist(threat, self.base_pos) < 1700.0)
            if not self.defense_budget_ok(cost, urgent=urgent):
                continue
            priority = 1650.0
            if sector_anchor is not None and threatened is not None:
                priority = 2500.0
            elif t < 240:
                priority = 1900.0
            elif t < 360 and len(self.defense_built_near) < 3:
                priority = 2200.0
            elif threat:
                priority = 1750.0
            self.log(
                f"defense turret {units.name_of(did)} at {tuple(round(v) for v in a)} "
                f"req={self.recent_attacker_range_for(a):.0f} urgent={urgent} "
                f"share={self.defense_spend_share():.2f}/{desired:.2f}"
            )
            self.add_task(
                f"defense:{self.frame}:{len(self.defense_built_near)}",
                "defense",
                did,
                a[0],
                a[1],
                priority,
                anchor=a,
                max_assignees=2,
            )
            break

    # --------------------------------------------------------------- task util

    def cleanup_tasks(self) -> None:
        for key, t in list(self.tasks.items()):
            # Drop assignees that no longer point at this task.
            for uid in list(t.assignees):
                if self.builder_task.get(uid) != key or uid not in self.units:
                    t.assignees.discard(uid)
            # Completed?
            if self.task_completed(t):
                self.finish_task(key)
                continue
            # If a building for this task already exists (even unfinished),
            # keep the task alive; abandoning it would leave a half-built
            # factory or reactor.
            for u in self.units.values():
                if u.def_id == t.def_id and not u.is_finished:
                    if self.dist(u.pos, (t.x, t.z)) < 220:
                        t.created_frame = self.frame
                        break
            age = self.frame - t.created_frame
            if t.unplaceable and age > 120:
                self.drop_task(key, "unplaceable")
            elif t.kind in ("energy", "fusion") and not t.assignees and age > 600:
                self.drop_task(key, "unassigned")
            elif not t.assignees and age > 6000:
                # Long walks and a busy constructor queue are normal; dropping
                # an expansion job after 60 seconds caused constant churn.
                self.drop_task(key, "stale")
            elif age > 9000 and not t.assignees:
                self.drop_task(key, "too old")

    def task_completed(self, t: Task) -> bool:
        # An unassigned energy job cannot be completed by a generator that
        # happened to finish nearby.  That was creating a stream of "task
        # done" messages while the solar count and income stayed flat.
        if t.kind in ("energy", "fusion") and not t.assignees:
            return False
        if t.kind == "mex" and t.spot_index is not None:
            if self.map.spots[t.spot_index].status == "ours":
                return True
            # Also accept an already built extractor at the task position.
            # This handles the case where the extractor was completed before
            # the task was created (for example after a status update race).
            for u in self.units.values():
                if u.name != "staticmex" or not u.is_finished:
                    continue
                if self.dist(u.pos, (t.x, t.z)) < 130.0:
                    return True
        if not t.placed:
            return False
        if t.kind == "energy":
            radius = 40.0
        elif t.kind == "fusion":
            radius = 120.0
        elif t.kind == "factory":
            radius = 190.0
        else:
            radius = 160.0
        for u in self.units.values():
            if not u.is_finished or u.def_id != t.def_id:
                continue
            # Only a building that appeared after the task was created can
            # complete it.  Using finished_frame here let an old generator
            # complete a long stream of new energy tasks as soon as its
            # finish was first observed, so the economy never grew.
            if t.kind in ("energy", "fusion"):
                if u.first_seen < t.created_frame:
                    continue
            elif u.finished_frame < t.created_frame:
                continue
            if self.dist(u.pos, (t.x, t.z)) < radius:
                return True
        return False

    def finish_task(self, key: str) -> None:
        t = self.tasks.pop(key, None)
        if not t:
            return
        for uid in list(t.assignees):
            if self.builder_task.get(uid) == key:
                self.builder_task.pop(uid, None)
                self.builder_idle[uid] = True
                self.builder_job_frame.pop(uid, None)
        if t.kind == "mex" and t.spot_index is not None:
            s = self.map.spots[t.spot_index]
            s.status = "ours"
            s.task_id = None
        if t.kind == "defense":
            self.defense_built_near.append((t.x, t.z))
        if t.kind in ("energy", "fusion", "factory", "caretaker", "radar"):
            self.log(f"task done {key}")

    def drop_task(self, key: str, reason: str) -> None:
        t = self.tasks.pop(key, None)
        if not t:
            return
        for uid in list(t.assignees):
            if self.builder_task.get(uid) == key:
                self.builder_task.pop(uid, None)
                self.builder_idle[uid] = True
                self.builder_job_frame.pop(uid, None)
        if t.kind == "mex" and t.spot_index is not None:
            s = self.map.spots[t.spot_index]
            s.status = "free"
            s.task_id = None
        self.log(f"drop task {key} ({reason}), attempts={t.placement_attempts}")
