"""DefenseBuildMixin: Build tasks for static defence: extractor guards, front line, base turrets.

Split out of economy.py; EconomyMixin.economy_step calls these.
"""
from __future__ import annotations

import math

from . import units
from .params import FLAGS, P
from .raidprobe import is_ground_turret

MEXLLT_AFTER = P("mexllt.after", 150.0, 0.0, 900.0, "seconds into the game a new extractor gets a turret first")
MEXLLT_NEAR = P("mexllt.near", 400.0, 150.0, 800.0, "a turret this close already covers the spot")
MEXLLT_OFFSET = P("mexllt.offset", 110.0, 70.0, 300.0, "elmos from the spot to the turret")
MEXLLT_PENDING = P("mexllt.pending", 2, 1, 6, "turrets of this kind waiting at once")
MEXLLT_GIVE_UP = P("mexllt.give_up", 2700, 300, 9000, "frames to wait for the turret before building the extractor anyway")
MEXLLT_SKIP = P("mexllt.skip", 5400, 300, 18000, "frames a spot then goes without asking again")
MEXLLT_MIN_MEX = P("mexllt.min_mex", 4, 0, 12, "with fewer extractors than this, income comes first")

MEX_DEF_CAP = P("def.mex_cap", 8, 2, 30, "max static defences while adding extractor turrets")
MEX_DEF_SCAN = P("def.mex_scan", 6, 2, 30, "forward-most extractors considered for a turret")
CLUSTER_DEF_SEC = P("def.cluster_sec", 23.0, 5.0, 60.0,
                    "flag circdef: turret value per owned cluster, in seconds of income")

# CircuitAI's land "porcupine" chain (data/config/build_chain.json): the
# turrets a defended metal cluster gets, in order, while its defence value is
# below the income-scaled cap.  Its caretakers, shield, radar and artillery
# entries are left out; AA entries are skipped until aircraft are seen.
CLUSTER_CHAIN = ("turretlaser", "turretlaser", "turretmissile", "turretheavylaser", "turretaalaser",
                 "turretmissile", "turretriot", "turretgauss", "turretmissile", "turretemp",
                 "turretaalaser", "turretriot", "turretheavylaser", "turretaalaser", "turretlaser",
                 "turretemp", "turretgauss", "turretmissile", "turretheavylaser", "turretaafar",
                 "turretaaheavy")


class DefenseBuildMixin:
    def ensure_cluster_defense_tasks(self) -> None:
        """Flag circdef: CircuitAI's per-cluster porcupine (MilitaryManager::
        MakeDefence).  Every owned metal cluster has a defence point whose
        turret value may grow to CLUSTER_DEF_SEC seconds of income
        (16 s x CircuitAI's map amount factor ~1.45); front clusters (enemy
        or recent attacks nearby, or a rich spot) walk the whole chain,
        quiet ones get one turret.  Unlike the share budget, the turret
        value grows with income and with the number of clusters held, which
        is how CircuitAI ends up putting 15-27% of its metal into turrets."""
        if "circdef" not in FLAGS or self.frame_time() < 60:
            return
        if self.frame - getattr(self, "_cdef_frame", -10**9) < 60:
            return
        self._cdef_frame = self.frame
        pending = [t for t in self.tasks.values() if t.kind == "clusterdef"]
        if len(pending) >= 3:
            return
        sectors = [s for s in getattr(self, "sectors", []) if s.spots and s.owner == "ours"]
        if not sectors:
            return
        income = min(self.metal_income(), self.energy_income())
        incomes = [sp.income for sp in self.map.spots] or [1.0]
        avg_i, max_i = sum(incomes) / len(incomes), max(incomes)
        air = self.aircraft_seen()
        defs = [u for u in self.units.values() if units.is_defense(u.data) and not u.is_mobile]
        start = getattr(self, "_cdef_next", 0) % len(sectors)
        for k in range(len(sectors)):
            s = sectors[(start + k) % len(sectors)]
            if any(t.anchor and self.dist(t.anchor, (s.x, s.z)) < 450 for t in pending):
                continue
            ours = sum(1 for sp in s.spots if getattr(sp, "status", "") == "ours") or 1
            max_cost = CLUSTER_DEF_SEC * income * max(1.0, ours / 3.0)
            rich = max_i > avg_i + 1.0 and max(sp.income for sp in s.spots) > (avg_i + max_i) * 0.5
            # CircuitAI: two neighbouring clusters under threat (or a rich
            # spot) make a cluster front line.
            near_threat = sum(
                1 for o in getattr(self, "sectors", [])
                if o is not s and o.owner != "ours" and (o.owner == "enemy" or o.threat > 0)
                and self.dist((o.x, o.z), (s.x, s.z)) < 1800.0
            )
            porc = rich or near_threat >= 2 or self.sector_attack_pressure_value(s) > 0.3
            point = self.best_defense_pos(s) or (s.x, s.z)
            if self.enemy_base:
                dx, dz = self.enemy_base[0] - point[0], self.enemy_base[1] - point[1]
                ln = math.hypot(dx, dz) or 1.0
                point = (point[0] + dx / ln * 120.0, point[1] + dz / ln * 120.0)
            have = sum(units.cost(u.data) for u in defs if self.dist(u.pos, point) < 420.0)
            total, n = 0.0, 0
            for name in CLUSTER_CHAIN:
                if name.startswith("turretaa") and not air:
                    continue
                if not porc and n >= 1:
                    break
                d = units.by_name(name)
                if d is None:
                    continue
                n += 1
                total += units.cost(d)
                if total <= have + 1.0:
                    continue
                if total >= max_cost:
                    break
                urgent = self.enemy_near(point, 700.0) is not None
                self.log(f"clusterdef {name} sector #{s.sid} porc={porc} have={have:.0f} cap={max_cost:.0f}")
                self.add_task(
                    f"clusterdef:{self.frame}:{s.sid}",
                    "clusterdef",
                    int(d["id"]),
                    point[0],
                    point[1],
                    # CircuitAI queues defence at HIGH priority; at 1300 the
                    # first turrets went stale behind extractor jobs.
                    1600.0 if urgent else 1450.0,
                    anchor=point,
                )
                self._cdef_next = start + k + 1
                return
        self._cdef_next = start + 1

    def ensure_mex_defense_tasks(self) -> None:
        """Put a range-appropriate turret on the forward extractors.

        A lone extractor is free metal for the first raider that finds it,
        but the old version always built a Lotus and built up to nine of
        them regardless of pressure.  The turret type now follows the range
        of recently attacking units, and the budget gate keeps static defence
        from crowding out mobile production.
        """
        if self.frame_time() < 150 or self.metal_income() < 8 or "circdef" in FLAGS:
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
        cap = max(2, min(MEX_DEF_CAP, int(2 + desired * 18)))
        if len(existing) + len(pending) >= cap:
            return
        for m in mexes[:MEX_DEF_SCAN]:
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

    def hold_unstarted_mexes(self) -> None:
        """Extractor jobs queued before 2:30 with nobody on them wait for a turret too."""
        if "mexllt" not in FLAGS:
            return
        for key, t in list(self.tasks.items()):
            # Only what was queued before the rule began: later jobs have been through
            # it, and redoing them had turrets and extractors take turns being dropped.
            if t.kind != "mex" or t.assignees or t.spot_index is None or t.created >= MEXLLT_AFTER:
                continue
            if any(u.is_being_built and u.def_id == t.def_id and self.dist(u.pos, (t.x, t.z)) < 130.0
                   for u in self.units.values()):
                continue
            if self.mex_waits_for_turret(self.map.spots[t.spot_index], t.priority):
                self.drop_task(key, "turret first")

    def mex_waits_for_turret(self, s, priority: float) -> bool:
        """From 2:30 an unclaimed spot gets a Lotus (LLT) next to it first (flag `mexllt`),
        unless a turret covers it already.  A spot where that fails gets its extractor
        anyway after a while."""
        if "mexllt" not in FLAGS or self.frame_time() < MEXLLT_AFTER:
            return False
        if self.mex_count() < MEXLLT_MIN_MEX:
            return False
        spot = (s.x, s.z)
        skip = self.__dict__.setdefault("mexllt_skip", {})
        if skip.get(s.index, -1) > self.frame:
            return False
        near = [u for u in self.units.values() if is_ground_turret(u) and self.dist(u.pos, spot) < MEXLLT_NEAR]
        asked = self.__dict__.setdefault("mexllt_asked", {})
        if any(u.is_finished for u in near):
            asked.pop(s.index, None)
            return False
        key = f"mexllt:{s.index}"
        if near:
            asked[s.index] = self.frame
            return True
        waiting = self.tasks.get(key)
        if waiting is not None and waiting.assignees:
            asked[s.index] = self.frame
            return True
        if self.frame - asked.setdefault(s.index, self.frame) > MEXLLT_GIVE_UP:
            asked.pop(s.index, None)
            skip[s.index] = self.frame + MEXLLT_SKIP
            self.drop_task(key, "extractor goes first")
            return False
        if key in self.tasks:
            return True
        did = units.id_of("turretlaser")
        pending = sum(1 for t in self.tasks.values() if t.kind == "mexllt")
        if did is None or pending >= MEXLLT_PENDING:
            return did is not None
        # On the enemy's side of the spot, and clear of the spot itself.
        ex, ez = self.enemy_base or self.map.center
        d = math.hypot(ex - s.x, ez - s.z) or 1.0
        x, z = s.x + (ex - s.x) / d * MEXLLT_OFFSET, s.z + (ez - s.z) / d * MEXLLT_OFFSET
        self.add_task(key, "mexllt", did, x, z, priority, anchor=(x, z))
        self.log(f"mexllt turret first at spot {s.index} ({s.x:.0f},{s.z:.0f})")
        return True

    def ensure_front_defense_tasks(self) -> None:
        # Forward turrets are useful once there is an army and an economy to
        # spare.  Their type follows recent attacker range; the budget gate
        # prevents a turret line from taking metal from the mobile army.
        if self.frame_time() < 900 or self.combat_count() < 20 or "circdef" in FLAGS:
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
