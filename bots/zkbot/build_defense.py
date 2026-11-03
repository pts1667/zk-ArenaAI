"""DefenseBuildMixin: Build tasks for static defence: extractor guards, front line, base turrets.

Split out of economy.py; EconomyMixin.economy_step calls these.
"""
from __future__ import annotations

import math

from . import units
from .params import FLAGS


class DefenseBuildMixin:
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
            # A spot that has already lost an extractor will be raided again:
            # its turret must beat new expansion, or it never gets built
            # (unsafe while raiders are there, outranked once they leave).
            idx = self.spot_index_at(m.pos) if hasattr(self, "spot_index_at") else None
            contested = idx is not None and getattr(self, "spot_mex_losses", {}).get(idx, 0) >= 1
            if "mexdefprio" not in FLAGS:
                prio = 1450.0 + min(450.0, pressure * 28.0) if urgent else 1260.0
            elif urgent:
                prio = 1650.0 + min(450.0, pressure * 28.0)
            elif contested:
                prio = 1550.0
            else:
                prio = 1260.0
            self.add_task(
                f"mexdef:{self.frame}:{m.uid}",
                "mexdef",
                did,
                m.x,
                m.z,
                prio,
                anchor=m.pos,
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
            want_aa = self.desired_static_aa()
            pending_aa = sum(1 for t in self.tasks.values() if t.kind == "aa")
            if aa + pending_aa < want_aa and pending_aa == 0:
                # First turret at home; further ones where aircraft actually
                # hurt us (fusions, factories, the army's staging area).
                anchor = self.base_pos
                if aa >= 1:
                    hot = self.air_damage_hotspot()
                    if hot is not None and self.dist(hot, self.base_pos) < 2600.0:
                        anchor = hot
                did = self.defense_unit_for(anchor, aa=True)
                if did is not None:
                    cost = units.cost(units.by_id(did))
                    hurt = self.recent_air_damage() > 1500.0
                    urgent = self.aircraft_seen_recent() or hurt
                    if self.defense_budget_ok(cost, urgent=urgent):
                        self.log(
                            f"aa turret {units.name_of(did)} #{aa + 1}/{want_aa} at "
                            f"{tuple(round(v) for v in anchor)} "
                            f"air_dmg={self.recent_air_damage():.0f}"
                        )
                        self.add_task(
                            f"aa:{self.frame}",
                            "aa",
                            did,
                            anchor[0],
                            anchor[1],
                            1650.0 if hurt else (1300.0 if self.aircraft_seen_recent() else 1180.0),
                            anchor=anchor,
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
                s = min(owned, key=lambda q: q.distance_enemy)
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
