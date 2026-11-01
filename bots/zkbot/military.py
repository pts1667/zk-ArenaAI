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
        if self.threat_pos and self.frame - self.threat_frame < 300:
            return "defend"
        enemy_count = (
            sum(1 for e in self.enemies.values() if self.is_dangerous(e))
            + len(self.radar)
        )
        if (
            enemy_count >= 8
            and self.combat_count() > 0
            and enemy_count > self.combat_count() * 1.15
            and self.enemy_army_pos
            and self.dist(self.enemy_army_pos, self.base_pos) < 4200
        ):
            # Do not feed a smaller army into a larger one; mass first.
            return "gather"
        threshold = 12 + int(self.frame_time() / 300)
        if self.combat_count() >= threshold:
            return "attack"
        return "gather"

    def military_step(self) -> None:
        if not self.army:
            return
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
            focus_pos is not None and self.dist(focus_pos, self.base_pos) < 1800.0
        )
        weak_commander = (
            enemy_commander is not None and enemy_commander.alive_frac < 0.5
        )
        if (
            focus_pos is not None
            and self.combat_count() >= 12
            and (
                (
                    close_to_base
                    and (
                        enemy_count <= self.combat_count() * 1.5
                        or enemy_count <= 8
                    )
                )
                or weak_commander
                or enemy_count <= self.combat_count() * 1.15
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

        raiders_sent = 0
        max_raiders = 2 if mode != "attack" else 0
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
            # Retreat badly damaged units.
            if u.alive_frac < 0.30 and self.frame - self.threat_frame < 600:
                self.retreating.add(uid)
            elif uid in self.retreating and u.alive_frac > 0.70:
                self.retreating.discard(uid)
            if uid in self.retreating:
                self.order_move(uid, self.retreat_point(u))
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
            if mode == "defend" and self.threat_pos:
                self.order_fight(uid, self.threat_pos)
                continue
            if enemy_army_pos and self.dist(u.pos, enemy_army_pos) < 750:
                self.order_fight(uid, enemy_army_pos)
                continue
            if mode == "attack":
                target = self.group_attack_target(center)
                if target:
                    self.order_fight(uid, target)
            else:
                # Gather at a forward rally point.  A couple of raiders may
                # poke at exposed enemy extractors, but the rest stay together.
                if (
                    self.is_raider(u)
                    and self.frame_time() > 90
                    and raiders_sent < max_raiders
                ):
                    target = self.pick_raid_target(u)
                    if target and self.dist(u.pos, target) > 400:
                        self.order_fight(uid, target)
                        raiders_sent += 1
                        continue
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
        return best

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
        danger = None
        best = 1e18
        for e in self.enemies.values():
            if not self.is_dangerous(e):
                continue
            d = self.dist(u.pos, e.pos)
            if d < 650 and d < best:
                best = d
                danger = e
        if danger is not None and best < 550:
            nearby = [
                e
                for e in self.enemies.values()
                if self.is_dangerous(e) and self.dist(u.pos, e.pos) < 900
            ]
            is_enemy_commander = "dyntrainer" in danger.name or "commander" in danger.name
            if (
                (not is_enemy_commander and len(nearby) <= 2 and u.alive_frac > 0.55)
                or (
                    is_enemy_commander
                    and len(nearby) <= 2
                    and u.alive_frac > 0.6
                    and u.health > danger.health * 1.15
                )
                or (is_enemy_commander and danger.alive_frac < 0.35 and u.alive_frac > 0.5)
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
