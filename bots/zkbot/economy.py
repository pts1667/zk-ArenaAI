"""EconomyMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class EconomyMixin:
    def economy_step(self) -> None:
        self.cleanup_tasks()
        self.ensure_mex_tasks()
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
        if limit is None:
            t = self.frame_time()
            if t < 90:
                limit = 6
            elif t < 240:
                limit = 10
            elif t < 480:
                limit = 16
            else:
                limit = 32
        bx, bz = self.base_pos
        free = [s for s in self.map.spots if s.status == "free" and s.task_id is None]
        if self.combat_count() >= 24:
            safe_radius = 5000.0
        elif self.combat_count() >= 16:
            safe_radius = 2200.0
        else:
            safe_radius = 1400.0
        free = [s for s in free if math.hypot(s.x - bx, s.z - bz) <= safe_radius]
        pending = sum(1 for t in self.tasks.values() if t.kind == "mex")
        room = max(0, limit - pending)
        room = min(room, max(4, self.builder_count()) - pending)
        if room <= 0:
            return
        enemy_base = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        scored: list[tuple[float, MetalSpot]] = []
        for s in free:
            db = math.hypot(s.x - bx, s.z - bz)
            de = math.hypot(s.x - enemy_base[0], s.z - enemy_base[1])
            priority = 1450.0 - db * 0.045 + s.income * 20.0
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
        """Choose a spot next to a metal extractor so overdrive can link."""
        owned = [
            u
            for u in self.units.values()
            if u.name == "staticmex" and u.is_finished
        ]
        if not owned:
            fx, fz = self.base_pos
            # Anchor the first generators on the closest metal spot: that is
            # where the first extractor will go, so the overdrive grid links
            # immediately instead of wasting the early energy.
            spots = sorted(
                self.map.spots, key=lambda sp: math.hypot(sp.x - fx, sp.z - fz)
            )
            if spots and math.hypot(spots[0].x - fx, spots[0].z - fz) < 700.0:
                spot = spots[min(self.energy_task_seq, min(3, len(spots))) % len(spots)]
                angle = math.radians((self.energy_task_seq * 137.5 + 20.0) % 360.0)
                radius = 115.0
                return (
                    spot.x + math.cos(angle) * radius,
                    spot.z + math.sin(angle) * radius,
                )
            # Otherwise spread the first few generators around the base.
            angle = math.radians((self.energy_task_seq * 137.5 + 20.0) % 360.0)
            radius = 130.0 + 40.0 * (self.energy_task_seq % 3)
            return fx + math.cos(angle) * radius, fz + math.sin(angle) * radius

        # Prefer mex clusters close to the base: they are safer, their pylon
        # ranges overlap, and builders do not have to cross the map to link
        # another generator into the overdrive grid.
        near_base = [
            u for u in owned if self.dist(u.pos, self.base_pos) < 1800.0
        ]
        if near_base:
            owned = near_base

        def score(u: UnitInfo) -> float:
            density = sum(1 for v in owned if self.dist(u.pos, v.pos) < 450.0)
            return density * 100.0 - self.dist(u.pos, self.base_pos) * 0.04

        owned.sort(key=score, reverse=True)
        top = owned[: min(10, len(owned))]
        spot = top[self.energy_task_seq % len(top)]
        radius = 165.0 if kind == "fusion" else 115.0
        angle = math.radians((self.energy_task_seq * 137.5 + 35.0) % 360.0)
        x = spot.x + math.cos(angle) * radius
        z = spot.z + math.sin(angle) * radius
        x = min(max(40.0, x), max(41.0, self.map.width - 40.0))
        z = min(max(40.0, z), max(41.0, self.map.height - 40.0))
        return x, z

    def ensure_energy_tasks(self, force: int = 0) -> None:
        minc = self.metal_income()
        einc = self.energy_income()
        ecur = self.energy_current()
        estore = self.effective_storage(self.energy())
        target = max(8.0, minc * 0.8 + 4.0, self.energy_usage() + 10.0)
        deficit = target - einc
        pending = sum(1 for t in self.tasks.values() if t.kind in ("energy", "fusion"))
        max_pending = 3
        usage = self.energy_usage()
        critical = (
            ecur < max(80.0, estore * 0.12)
            or einc < usage
        )
        energy_priority = 2000.0 if critical else 1000.0
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
        did = units.id_of("energysolar")
        if did is None:
            return
        # Create enough jobs to cover the deficit.  Each solar is 2 E/s.
        count = max(force, min(2, int(math.ceil(max(0.0, deficit) / 2.0))))
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
        if minc >= 14 or self.frame_time() > 120:
            target = 2
        if minc >= 22 or self.frame_time() > 300:
            target = 3
        if minc >= 28 or self.frame_time() > 420:
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
        # The first extra factory is an air plant.  Air units can chase a
        # cloaked or cliff-hugging enemy commander that ground units cannot
        # corner, and they double as scouts.
        if target >= 2 and plane_def and not has_plane and not plane_pending:
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
            1480.0,
            anchor=anchor,
            max_assignees=2,
        )
        self.factory_task_seq += 1

    def ensure_strider_tasks(self) -> None:
        if self.frame_time() < 420 or self.metal_income() < 22:
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
        target = 3
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
        for a in anchors:
            if any(self.dist(a, u.pos) < 450 for u in existing):
                continue
            if any(t.anchor and self.dist(a, t.anchor) < 450 for t in pending):
                continue
            radar_priority = 1480.0 if not existing else 1300.0
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
        if self.frame_time() < 240 or self.combat_count() < 8:
            return
        did = units.id_of("turretlaser")
        if did is None:
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
            if u.name in (
                "turretlaser",
                "turretmissile",
                "turretaalaser",
                "turretgauss",
                "turretheavylaser",
            )
            and (u.is_finished or u.is_being_built)
        ]
        pending = [t for t in self.tasks.values() if t.kind == "frontdef"]
        for t in pending:
            t.priority = 1600.0 if self.enemy_near((t.x, t.z), 650) else 1480.0
        for m in mexes[:10]:
            if any(self.dist(m.pos, u.pos) < 450 for u in existing):
                continue
            if any(t.anchor and self.dist(m.pos, t.anchor) < 450 for t in pending):
                continue
            if len(pending) >= 8:
                break
            urgent = self.enemy_near(m.pos, 650) is not None
            self.add_task(
                f"frontdef:{self.frame}:{m.uid}",
                "frontdef",
                did,
                m.x,
                m.z,
                1600.0 if urgent else 1480.0,
                anchor=m.pos,
            )

    def ensure_defense_tasks(self) -> None:
        threat = self.threat_pos if self.frame - self.threat_frame < 240 else None
        air_threat = any(u.is_air for u in self.enemies.values())
        if air_threat:
            aa = sum(
                1
                for u in self.units.values()
                if u.name.startswith("turretaa") and (u.is_finished or u.is_being_built)
            )
            if aa == 0 and not any(t.kind == "aa" for t in self.tasks.values()):
                did = units.id_of("turretaalaser")
                if did:
                    self.add_task(
                        f"aa:{self.frame}",
                        "aa",
                        did,
                        self.base_pos[0],
                        self.base_pos[1],
                        1250.0,
                        anchor=self.base_pos,
                    )
        if threat is None and self.frame_time() < 55:
            return
        defense_cap = 5 if threat else 3
        if sum(1 for t in self.tasks.values() if t.kind in ("defense", "aa")) >= defense_cap:
            return
        bx, bz = self.base_pos
        anchors: list[tuple[float, float]] = []
        if threat:
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
        did = None
        if self.metal_income() >= 22 and self.frame_time() > 360:
            did = units.id_of("turretgauss")
        if did is None:
            did = units.id_of("turretlaser")
        if did is None:
            return
        for a in anchors:
            if any(self.dist(a, p) < 140 for p in self.defense_built_near):
                continue
            if any(
                t.anchor and self.dist(a, t.anchor) < 140
                for t in self.tasks.values()
                if t.kind == "defense"
            ):
                continue
            self.add_task(
                f"defense:{self.frame}:{len(self.defense_built_near)}",
                "defense",
                did,
                a[0],
                a[1],
                1750.0 if threat else 1650.0,
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
            elif not t.assignees and age > 6000:
                # Long walks and a busy constructor queue are normal; dropping
                # an expansion job after 60 seconds caused constant churn.
                self.drop_task(key, "stale")
            elif age > 9000 and not t.assignees:
                self.drop_task(key, "too old")

    def task_completed(self, t: Task) -> bool:
        if t.kind == "mex" and t.spot_index is not None:
            if self.map.spots[t.spot_index].status == "ours":
                return True
        if not t.placed:
            return False
        if t.kind == "energy":
            radius = 65.0
        elif t.kind == "fusion":
            radius = 150.0
        elif t.kind == "factory":
            radius = 190.0
        else:
            radius = 160.0
        for u in self.units.values():
            if not u.is_finished or u.def_id != t.def_id:
                continue
            # Only a building that finished after the task was created can
            # complete it.  This matters because several energy jobs may be
            # placed in one cluster.
            if u.finished_frame < t.created_frame:
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
