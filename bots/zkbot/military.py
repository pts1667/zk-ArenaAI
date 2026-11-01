"""MilitaryMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class MilitaryMixin:
    def is_raider(self, u: UnitInfo) -> bool:
        return units.role_of(u.data) in ("raid", "scout")

    def army_mode(self) -> str:
        enemy_count = (
            sum(1 for e in self.enemies.values() if self.is_dangerous(e))
            + len(self.radar)
        )
        combat = self.combat_count()
        if enemy_count > self.enemy_peak or self.frame - self.enemy_peak_frame > 1800:
            self.enemy_peak = enemy_count
            self.enemy_peak_frame = self.frame
        # Sector threat is the strategic view: a wave moving through a
        # frontier sector is defended as a group, not chased unit by unit.
        threatened = self.most_threatened_sector()
        if (
            threatened is not None
            and threatened.threat > 0
            and threatened.distance_base < 1800.0
            and threatened.owner == "ours"
        ):
            return "defend"
        # Defend only a real attack, or enemy units close enough to the base
        # / turret line.  Previously any distant contact put the whole army in
        # defend mode forever while CircuitAI freely rebuilt its economy.
        if self.threat_pos and self.frame - self.threat_frame < 900:
            near_base = self.dist(self.threat_pos, self.base_pos) < 1800.0
            if near_base or enemy_count >= max(4, int(combat * 0.60)):
                return "defend"
        if enemy_count >= 8 and combat > 0 and enemy_count > combat * 1.15:
            # Do not feed a smaller army into a larger one; mass first.
            return "gather"
        # Attack when we have a decisive mass advantage over the enemy peak,
        # even if a small raiding party is still visible somewhere.
        threshold = max(
            24,
            int(self.enemy_peak * 0.90) + 2,
            18 + int(self.frame_time() / 360),
        )
        recent_enemy = self.frame - self.last_enemy_seen_frame < 1800
        # Do not let one small visible raiding party pin a large army at the
        # rally point forever.  If the contact is clearly smaller than us, go
        # win the game instead of waiting for the map to become empty.
        small_contact = enemy_count <= max(6, int(combat * 0.30))
        if combat >= threshold and (not recent_enemy or small_contact):
            return "attack"
        return "gather"

    def assign_home_guard(self) -> None:
        """Keep a small defensive force at home while the main army attacks.

        Sector threats are useful, but if every combat unit marches to the
        enemy base, CircuitAI simply walks into the solar field.  The guard is
        made of AA / artillery / skirmishers first, then nearby units.
        """
        self.home_guard = {uid for uid in self.home_guard if uid in self.army}
        target = max(2, min(10, len(self.army) // 4))
        if len(self.home_guard) >= target:
            return
        candidates = [
            UID
            for UID in self.army
            if UID not in self.home_guard and UID in self.units
        ]
        if not candidates:
            return

        def priority(uid: int) -> tuple[int, float]:
            u = self.units[uid]
            role = units.role_of(u.data)
            if units.is_aa(u.data) or role == "aa":
                p = 0
            elif role == "skirm":
                p = 1
            elif role == "riot":
                p = 2
            elif role == "assault":
                p = 3
            elif role in ("arty", "heavyarty"):
                # Artillery belongs with the attacking force unless the base
                # is actually under threat; leaving all of it at home makes
                # every attack into a turret line fail.
                p = 4
            else:
                p = 6
            return p, self.dist(u.pos, self.base_pos)

        candidates.sort(key=priority)
        for uid in candidates[: target - len(self.home_guard)]:
            self.home_guard.add(uid)
        if self.home_guard:
            self.say(
                "home_guard",
                f"home guard {len(self.home_guard)}/{len(self.army)}",
            )

    def military_step(self) -> None:
        if not self.army:
            return
        self.assign_home_guard()
        mode = self.army_mode()
        center = self.army_center()
        enemy_commander = (
            self.enemies.get(self.enemy_commander_id) if self.enemy_commander_id else None
        )
        commander_memory = (
            self.enemy_memory.get(self.enemy_commander_id)
            if self.enemy_commander_id is not None and not self.enemy_commander_dead
            else None
        )
        focus_commander = False
        focus_pos: tuple[float, float] | None = None
        if self.enemy_commander_dead:
            enemy_commander = None
        if enemy_commander is not None and enemy_commander.is_finished:
            focus_pos = enemy_commander.pos
        elif commander_memory and self.frame - commander_memory[1] < 900:
            focus_pos = commander_memory[0].pos
        enemy_count = (
            sum(1 for e in self.enemies.values() if self.is_dangerous(e))
            + len(self.radar)
        )
        close_to_base = (
            focus_pos is not None and self.dist(focus_pos, self.base_pos) < 2000.0
        )
        weak_commander = (
            enemy_commander is not None and enemy_commander.alive_frac < 0.5
        )
        if (
            focus_pos is not None
            and self.combat_count() >= (7 if close_to_base else 12)
            and (
                (
                    close_to_base
                    and (
                        enemy_count <= self.combat_count() * 1.5
                        or enemy_count <= 8
                    )
                )
                or (weak_commander and close_to_base)
            )
        ):
            focus_commander = True
            self.hunt_commander = True
            if self.frame - self.last_attack_frame > 300:
                self.last_attack_frame = self.frame
                hp = ""
                if enemy_commander is not None:
                    hp = (
                        f" hp={enemy_commander.health:.0f}/"
                        f"{enemy_commander.max_health:.0f}"
                    )
                near = sum(
                    1
                    for uid in self.army
                    if uid in self.units
                    and self.dist(self.units[uid].pos, focus_pos) < 1200.0
                )
                self.log(
                    f"focus enemy commander at {tuple(round(v) for v in focus_pos)}"
                    f"{hp} army_near={near}"
                )

        if focus_pos is None:
            self.hunt_commander = False
        enemy_army_pos = None
        contacts = list(self.enemies.values()) + list(self.radar.values())
        if contacts:
            e = min(
                (
                    e
                    for e in contacts
                    if not e.is_builder or "dyntrainer" in e.name or "commander" in e.name
                ),
                key=lambda e: self.dist(e.pos, self.base_pos),
                default=None,
            )
            if e is not None:
                enemy_army_pos = e.pos

        threatened = self.most_threatened_sector()
        if threatened is not None and (
            threatened.distance_base >= 2000.0 or threatened.owner != "ours"
        ):
            threatened = None
        defense_anchor = self.best_defense_pos(threatened)
        if defense_anchor is None:
            defense_anchor = self.threat_pos
        attack_sector = self.best_attack_sector()
        attack_anchor = self.best_defense_pos(attack_sector)
        home_anchor = defense_anchor or self.base_pos
        raider_uids = [
            uid for uid in self.army if uid in self.units and self.is_raider(self.units[uid])
        ]
        raider_stage = self.rally_point()
        if self.enemy_near(raider_stage, 650.0) is not None:
            raider_stage = self.base_pos
        raider_group_limit = 3 if self.frame_time() < 420 else 5
        squad = raider_uids[:raider_group_limit]
        group_needed = 2 if self.frame_time() < 300 else 3
        # If nothing has been visible for a while, the enemy commander may be
        # cloaked or hiding.  Sweep the enemy half of the map with small groups
        # instead of letting a large army sit on an empty base.
        unknown_hunt = (
            focus_pos is None
            and self.combat_count() >= 10
            and self.frame_time() > 480
        )
        search_mode = unknown_hunt or (
            mode != "defend"
            and self.combat_count() >= 8
            and self.frame - self.last_enemy_seen_frame > 600
            and enemy_army_pos is None
        )
        search_pts = self.search_points() if search_mode else []
        group_active = (
            len(squad) >= group_needed
            and not focus_commander
            and not search_mode
            and mode != "attack"
            and not (
                mode == "defend"
                and self.threat_pos is not None
                and self.dist(self.threat_pos, self.base_pos) < 2200.0
            )
        )
        raid_target = None
        ready: list[int] = []
        if group_active:
            if (
                self.raid_group_target is None
                or self.frame - self.raid_group_frame > 600
                or self.dist(self.raid_group_target, self.base_pos) < 450.0
            ):
                self.raid_group_target = self.choose_raid_group_target(squad)
                self.raid_group_frame = self.frame
                if self.raid_group_target is not None:
                    self.log(
                        "raid group "
                        f"{len(squad)} -> {tuple(round(v) for v in self.raid_group_target)}"
                    )
            raid_target = self.raid_group_target
            if raid_target is None:
                group_active = False
            else:
                ready = [
                    uid
                    for uid in squad
                    if uid in self.units
                    and (
                        self.dist(self.units[uid].pos, raider_stage) < 900.0
                        or self.dist(self.units[uid].pos, raid_target) < 1000.0
                    )
                ]
                self.raid_group_size = len(ready)
        for uid in list(self.army):
            u = self.units.get(uid)
            if not u:
                continue
            # Cloak bombs are expendable: drive them straight at the most
            # valuable target instead of retreating or forming a firing line.
            if units.role_of(u.data) == "bomb":
                target = (
                    focus_pos
                    or enemy_army_pos
                    or (self.threat_pos if mode == "defend" else None)
                    or self.group_attack_target(center)
                    or self.enemy_base
                    or self.rally_point()
                )
                if target:
                    self.order_move(uid, target)
                continue
            # Retreat only when it is actually worth preserving the unit.
            # Retreat destinations are chosen to avoid enemy contact; a cheap
            # raider with no clean path home is allowed to trade itself.
            if uid in self.retreating and u.alive_frac > 0.70:
                self.retreating.discard(uid)
            if uid in self.retreating:
                self.order_move(uid, self.safe_retreat_point(u))
                continue
            should_retreat, retreat_dest = self.retreat_decision(u)
            if should_retreat and retreat_dest is not None:
                self.retreating.add(uid)
                self.order_move(uid, retreat_dest)
                continue
            self.retreating.discard(uid)
            if uid in self.home_guard and mode == "attack":
                # The guard holds the base / threatened owned sector while
                # the main force is away.  In defend mode everyone converges,
                # so this branch is intentionally limited to attack/gather.
                self.order_fight(uid, home_anchor)
                continue
            if focus_commander and focus_pos is not None:
                if enemy_commander is not None and enemy_commander.is_finished:
                    self.order_attack(uid, enemy_commander.uid)
                elif u.is_air or units.role_of(u.data) in (
                    "arty",
                    "assault",
                    "riot",
                    "skirm",
                    "heavy",
                ):
                    # The commander went out of LOS (often cloaked).  Fire at
                    # the last known position; area damage can still finish it.
                    self.order_attack_ground(uid, focus_pos)
                else:
                    self.order_fight(uid, focus_pos)
                continue
            if search_mode and search_pts:
                should_search = (
                    not unknown_hunt
                    or u.is_air
                    or self.is_raider(u)
                    or uid % 6 == 0
                )
                if should_search:
                    idx = (uid + self.frame // 900) % len(search_pts)
                    self.order_fight(uid, search_pts[idx])
                    continue
            if self.is_raider(u):
                if group_active and raid_target is not None:
                    if uid in ready:
                        self.order_fight(uid, raid_target)
                    else:
                        self.order_fight(uid, raider_stage)
                    continue
                if mode != "attack" and not (
                    mode == "defend"
                    and self.threat_pos is not None
                    and self.dist(self.threat_pos, self.base_pos) < 2200.0
                ):
                    self.order_fight(uid, raider_stage)
                    continue
            if mode in ("defend", "attack") and not search_mode:
                local = self.best_local_target(u)
                if local is not None:
                    role = units.role_of(u.data)
                    reach = 520.0
                    if role in ("arty", "heavyarty"):
                        reach = max(reach, float(u.data.get("maxWeaponRange") or 0.0) * 0.95)
                    near = self.dist(u.pos, local.pos) < reach
                    aa_air = (units.is_aa(u.data) or role == "aa") and local.is_air
                    if near or aa_air:
                        # Radar blips have no valid attack target id for most
                        # weapons; artillery can still shell the location.
                        if local.uid in self.radar and role in ("arty", "heavyarty"):
                            self.order_attack_ground(uid, local.pos)
                        else:
                            self.order_attack(uid, local.uid)
                        continue
            if mode == "defend" and defense_anchor is not None:
                # If the enemy ball is as large as ours, fall back to the
                # turret line instead of feeding units into it piecemeal at a
                # distant extractor.  Smaller raids are met in the field.
                # Hold a group anchor (high ground / sector centre).  Units
                # fight what comes to the line instead of each chasing a
                # different contact and arriving one at a time.
                self.order_fight(uid, defense_anchor)
                continue
            if enemy_army_pos and self.dist(u.pos, enemy_army_pos) < 750:
                self.order_fight(uid, enemy_army_pos)
                continue
            if mode == "attack":
                target = attack_anchor or self.group_attack_target(center)
                if target:
                    self.order_fight(uid, target)
            else:
                self.order_fight(uid, self.rally_point())

    def army_center(self) -> tuple[float, float] | None:
        if not self.army:
            return None
        xs = [self.units[uid].x for uid in self.army if uid in self.units]
        zs = [self.units[uid].z for uid in self.army if uid in self.units]
        if not xs:
            return None
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def retreat_point(self, u: UnitInfo) -> tuple[float, float]:
        bx, bz = self.base_pos
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        dx, dz = bx - ex, bz - ez
        length = math.hypot(dx, dz) or 1.0
        return bx + dx / length * 140.0, bz + dz / length * 140.0

    def rally_point(self) -> tuple[float, float]:
        if getattr(self, "sectors", None):
            return self.frontier_rally()
        bx, bz = self.base_pos
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        return bx + (ex - bx) * 0.22, bz + (ez - bz) * 0.22

    def search_points(self) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        w, h = self.map.width, self.map.height
        if w <= 0 or h <= 0:
            return pts
        # Dense enough that a cloaked commander is physically bumped into:
        # cloak is only revealed by proximity in Zero-K, so sparse waypoints
        # can miss a hiding commander forever.
        step = 400.0
        gx = step * 0.5
        while gx < w:
            gz = step * 0.5
            while gz < h:
                if self.dist((gx, gz), self.base_pos) > 1600.0:
                    pts.append((gx, gz))
                gz += step
            gx += step
        if self.enemy_base:
            ex, ez = self.enemy_base
            pts.insert(0, (ex, ez))
            for dx, dz in ((-220, 0), (220, 0), (0, -220), (0, 220)):
                pts.insert(1, (ex + dx, ez + dz))
        for s in self.map.spots:
            if self.dist((s.x, s.z), self.base_pos) > 1800:
                pts.append((s.x, s.z))
        return pts

    def group_attack_target(
        self, center: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if center is None:
            center = self.rally_point()
        # Prefer the enemy commander if it is close to our army or our base.
        ec = self.enemies.get(self.enemy_commander_id) if self.enemy_commander_id else None
        if ec is not None and ec.is_finished and not self.enemy_commander_dead:
            if self.dist(ec.pos, center) < 2200.0 or self.dist(ec.pos, self.base_pos) < 1800.0:
                return ec.pos
        if self.enemy_commander_id is not None and not self.enemy_commander_dead:
            remembered = self.enemy_memory.get(self.enemy_commander_id)
            if remembered and self.frame - remembered[1] < 1800:
                return remembered[0].pos
        # Otherwise hit the most valuable enemy thing near the army.
        best = None
        best_score = -1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            score = 0.0
            if not self.enemy_commander_dead and (
                "dyntrainer" in e.name or "commander" in e.name
            ):
                score += 1500.0
            if e.is_factory:
                score += 1200.0
            if e.is_builder:
                score += 800.0
            if e.name in ("staticmex", "staticcon"):
                score += 300.0
            if units.is_energy(e.data):
                score += 220.0
            if e.is_combat:
                score += 100.0
            d = self.dist(center, e.pos)
            score -= d * 0.20
            if score > best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        if self.enemy_army_pos and self.frame - self.enemy_army_frame < 600:
            return self.enemy_army_pos
        return self.enemy_base

    def pick_attack_target(self, u: UnitInfo) -> tuple[float, float] | None:
        best = None
        best_score = -1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            score = 0.0
            if e.is_factory:
                score += 500.0
            if e.name in ("staticmex", "staticcon"):
                score += 250.0
            if units.is_energy(e.data):
                score += 180.0
            if e.is_combat:
                score += 80.0
            score -= self.dist(u.pos, e.pos) * 0.15
            if score > best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        if self.enemy_army_pos and self.frame - self.enemy_army_frame < 600:
            return self.enemy_army_pos
        return self.enemy_base

    def pick_raid_target(self, u: UnitInfo) -> tuple[float, float] | None:
        best = None
        best_score = 1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            d = self.dist(u.pos, e.pos)
            value = 0.0
            if e.name in ("staticmex", "staticcon"):
                value = 350.0
            elif e.is_factory:
                value = 500.0
            elif units.is_energy(e.data):
                value = 250.0
            elif e.is_builder:
                value = 250.0
            else:
                continue
            score = d - value
            if score < best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        # Nothing is visible yet.  Send the raiders toward the enemy base so
        # they actually find the expansion/army instead of idling at rally.
        # This is how CircuitAI uses its opening raiders, and Loss-of-Sight
        # contacts are exactly what our own FOV/radar need.
        if self.enemy_base:
            ex, ez = self.enemy_base
            # Prefer the enemy-side metal spots.  This sends raiders into the
            # enemy economy even when no extractor is currently in LOS.
            side = [
                s
                for s in self.map.spots
                if s.status != "ours"
                and math.hypot(s.x - ex, s.z - ez) < 1900.0
            ]
            if side:
                side.sort(key=lambda s: math.hypot(s.x - u.x, s.z - u.z))
                return side[0].x, side[0].z
            return self.enemy_base
        return None

    def order_fight(self, uid: int, pos: tuple[float, float]) -> None:
        self._order(uid, "fight", pos)

    def order_move(self, uid: int, pos: tuple[float, float]) -> None:
        self._order(uid, "move", pos)

    def order_attack(self, uid: int, target_id: int) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == "attack" and old[3] == target_id and self.frame - old[4] < 120:
            return
        self.queue({"type": "attack", "unitId": int(uid), "targetId": int(target_id)})
        self.unit_order[uid] = ("attack", 0.0, 0.0, int(target_id), self.frame)

    def order_attack_ground(self, uid: int, pos: tuple[float, float]) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == "attack_ground":
            if math.hypot(old[1] - pos[0], old[2] - pos[1]) < 150 and self.frame - old[4] < 120:
                return
        self.queue(
            {
                "type": "custom",
                "unitId": int(uid),
                "cmdId": 20,
                "params": [float(pos[0]), 0.0, float(pos[1])],
            }
        )
        self.unit_order[uid] = (
            "attack_ground",
            float(pos[0]),
            float(pos[1]),
            None,
            self.frame,
        )

    def _order(self, uid: int, typ: str, pos: tuple[float, float]) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == typ:
            ox, oz = old[1], old[2]
            if math.hypot(ox - pos[0], oz - pos[1]) < 150 and self.frame - old[4] < 90:
                return
        self.queue({"type": typ, "unitId": int(uid), "pos": [float(pos[0]), 0.0, float(pos[1])]})
        self.unit_order[uid] = (typ, float(pos[0]), float(pos[1]), None, self.frame)

    # ------------------------------------------------------------- commander

    def ensure_commander_safe(self) -> None:
        if not self.commander:
            return
        u = self.commander
        # Hard-code the safe radius around home, not around "wherever the
        # commander happens to be".  CircuitAI regularly sends fast shield
        # raiders around the front to kill constructors and the commander.
        if self.dist(u.pos, self.base_pos) > 900.0:
            if self.frame - self.last_commander_retreat_frame < 30:
                return
            self.last_commander_retreat_frame = self.frame
            target = self.retreat_point(u)
            self.queue(
                {"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]]}
            )
            self.say(
                "commander_return",
                f"commander returning home from {tuple(round(v) for v in u.pos)}",
            )
            return
        danger = None
        best = 1e18
        for e in self.enemies.values():
            if not self.is_dangerous(e):
                continue
            d = self.dist(u.pos, e.pos)
            if d < 1000 and d < best:
                best = d
                danger = e
        if danger is not None and best < 900:
            nearby = [
                e
                for e in self.enemies.values()
                if self.is_dangerous(e) and self.dist(u.pos, e.pos) < 900
            ]
            is_enemy_commander = "dyntrainer" in danger.name or "commander" in danger.name
            if best < 450 and (
                (not is_enemy_commander and len(nearby) <= 1 and u.alive_frac > 0.75)
                or (
                    is_enemy_commander
                    and len(nearby) <= 1
                    and u.alive_frac > 0.7
                    and u.health > danger.health * 1.3
                )
            ):
                if self.frame - self.last_commander_retreat_frame < 45:
                    return
                self.last_commander_retreat_frame = self.frame
                if danger.is_finished:
                    self.order_attack(u.uid, danger.uid)
                else:
                    self.order_fight(u.uid, danger.pos)
                self.say(
                    "commander_defend",
                    f"commander defending against {danger.name} at "
                    f"{tuple(round(v) for v in danger.pos)}",
                )
                return
            if self.frame - self.last_commander_retreat_frame < 60:
                return
            self.last_commander_retreat_frame = self.frame
            target = self.retreat_point(u)
            self.queue(
                {"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]]}
            )
            self.say(
                "commander_danger",
                f"commander retreating from enemy at {tuple(round(v) for v in danger.pos)}",
            )

    # ----------------------------------------------------------------- summary
