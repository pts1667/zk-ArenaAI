"""SupportBuildMixin: Build tasks for support structures: radar, missile silo, seismic detector.

Split out of economy.py; EconomyMixin.economy_step calls these.
"""
from __future__ import annotations

import math

from . import units


class SupportBuildMixin:
    def ensure_radar_tasks(self) -> None:
        if self.frame_time() < 120 and self.threat_pos is None:
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
        if self.frame_time() < 900 or self.metal_income() < 60:
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
