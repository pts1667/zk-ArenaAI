"""FacilityBuildMixin: Build tasks for production: factories, strider hub, air pads, caretakers.

Split out of economy.py; EconomyMixin.economy_step calls these.
"""
from __future__ import annotations

import math
import os
import random

from . import units


class FacilityBuildMixin:
    def ensure_factory_tasks(self) -> None:
        factories = [
            u
            for u in self.units.values()
            if u.is_factory and (u.is_finished or u.is_being_built)
        ]
        if self.factory_plop_sent and not factories:
            return
        minc = self.metal_income()
        t = self.frame_time()
        # One ground factory plus caretakers carries the game for a long
        # time.  The second factory comes at ~42 metal/s, or at 20-25 min
        # depending on map size, and it is an air plant (scouting and fire
        # support).  More factories only on a much bigger economy.
        t2 = self.second_factory_time()
        target = 1
        # Metal that keeps piling up despite extra caretakers means build
        # power is short: bring the second factory forward.
        floating = t >= 420 and getattr(self, "metal_float_frames", 0) >= 2700
        if minc >= 42 or t >= t2 or floating:
            target = 2
        if minc >= 75 and t >= t2:
            target = 3
        if minc >= 110:
            target = 4
        pending = sum(1 for q in self.tasks.values() if q.kind == "factory")
        # The strider hub is a factory too, but has its own rule.
        counted = [f for f in factories if f.name != "striderhub"]
        if len(counted) + pending >= target:
            return
        air_names = ("factorygunship", "factoryplane")
        has_air = any(f.name in air_names for f in factories) or any(
            q.kind == "factory" and units.name_of(q.def_id) in air_names
            for q in self.tasks.values()
        )
        if not has_air:
            did = units.id_of(self.air_factory_choice())
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
        # Rotate through several sites: a crowded base (dozens of solars) or a
        # wreck on the old factory site left the two fixed anchors blocked and
        # a lost ground factory was never rebuilt (30 min of capped metal).
        k = self.factory_task_seq
        offs = (35.0, -35.0, 70.0, -70.0, 0.0, 110.0, -110.0)
        ang = math.atan2(dz, dx) + math.radians(offs[k % len(offs)])
        dist = min(1400.0, 330.0 + 220.0 * (k // len(offs)) + 60.0 * (k % 3))
        anchor = (bx + math.cos(ang) * dist, bz + math.sin(ang) * dist)
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

    def second_factory_time(self) -> float:
        """20 min on small maps, up to 25 min on large ones (seconds)."""
        side = math.sqrt(max(1.0, self.map.width * self.map.height)) / 512.0
        if side <= 12.0:
            return 1200.0
        if side >= 18.0:
            return 1500.0
        return 1200.0 + (side - 12.0) / 6.0 * 300.0

    def air_factory_choice(self) -> str:
        """Second factory, chosen once per game: usually an airplane plant,
        often a gunship plant, occasionally a second ground factory.
        Enemy air forces planes (fighters).  ZKBOT_SECOND=<factory> pins it
        for tests."""
        forced = os.environ.get("ZKBOT_SECOND")
        if forced:
            return forced
        comp = self.enemy_composition()
        if self.aircraft_seen() and comp.get("air", 0) >= 4:
            return "factoryplane"
        choice = getattr(self, "second_factory_choice", None)
        if choice is None:
            r = random.random()
            if r < 0.55:
                choice = "factoryplane"
            elif r < 0.90:
                choice = "factorygunship"
            else:
                choice = self.factory_name or "factorycloak"
            self.second_factory_choice = choice
            self.log(f"second factory choice: {choice} (roll {r:.2f})")
        return choice

    def ensure_airpad_tasks(self) -> None:
        """Bombers rearm on pads; one pad per ~6 planes, at home."""
        if not any(f.name == "factoryplane" for f in self.factory_list()):
            return
        did = units.id_of("staticrearm")
        if did is None:
            return
        planes = sum(1 for u in self.units.values() if u.is_air and u.is_combat)
        target = 1 + planes // 6
        have = sum(
            1 for u in self.units.values() if u.name == "staticrearm" and (u.is_finished or u.is_being_built)
        ) + sum(1 for q in self.tasks.values() if q.kind == "airpad")
        if have >= min(3, target):
            return
        bx, bz = self.base_pos
        cx, cz = self.map.center
        d = math.hypot(cx - bx, cz - bz) or 1.0
        anchor = (bx + (cx - bx) / d * 120.0, bz + (cz - bz) / d * 120.0)
        self.add_task(
            f"airpad:{self.frame}", "airpad", did, anchor[0], anchor[1], 1300.0, anchor=anchor
        )

    def ensure_strider_tasks(self) -> None:
        if self.frame_time() < self.second_factory_time() or self.metal_income() < 80:
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
        # 180 metal of caretaker is two extractors in the opening; wait until
        # the expansion is under way.
        if not factories or minc < 12 or self.frame_time() < 150:
            return
        did = units.id_of("staticcon")
        if did is None:
            return
        # Factories + caretakers should be able to spend ~70% of income; a
        # lone factory is only 10 BP, so at 45 metal/s it left 30/s unspent.
        want_bp = 0.7 * minc / max(1, len(factories))
        target_per_factory = max(1, math.ceil((want_bp - 10.0) / 10.0))
        # Metal piling up means production build power is short: add
        # caretakers (CircuitAI allows up to 6 per factory).
        m = self.metal()
        float_frac = self.metal_current() / max(1.0, self.effective_storage(m))
        if float_frac > 0.5:
            self.metal_float_frames = getattr(self, "metal_float_frames", 0) + 15
        else:
            self.metal_float_frames = max(0, getattr(self, "metal_float_frames", 0) - 30)
        extra = min(4, getattr(self, "metal_float_frames", 0) // 900)
        target_per_factory = min(6, target_per_factory + extra)
        # Unspent metal outranks expansion: more extractors only add to the
        # pile while the army starves.
        if float_frac > 0.5:
            care_priority = 2050.0
        elif float_frac > 0.25:
            care_priority = 1600.0
        else:
            care_priority = 1250.0
        for t in self.tasks.values():
            if t.kind == "caretaker":
                t.priority = care_priority
        for f in factories:
            existing = [
                u
                for u in self.units.values()
                if u.name == "staticcon"
                and (u.is_finished or u.is_being_built)
                and self.dist(u.pos, f.pos) < 480
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
                care_priority,
                anchor=f.pos,
            )
