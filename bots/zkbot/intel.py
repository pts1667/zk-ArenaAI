"""IntelMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class IntelMixin:
    def find_commander(self, obs: dict[str, Any]) -> None:
        best = None
        for uid_s, raw in (obs.get("units") or {}).items():
            did = int(raw.get("defId") or 0)
            d = units.by_id(did) or {}
            name = d.get("name") or raw.get("name") or ""
            if "dyntrainer" in name or name.startswith("commander"):
                h = float(raw.get("health") or 0)
                if best is None or h > best[0]:
                    best = (h, int(uid_s), did)
        if best:
            self.commander_id = best[1]
            self.commander = self.make_unit(
                self.commander_id, best[2], obs["units"][str(best[1])]
            )

    def make_unit(
        self,
        uid: int,
        def_id: int,
        raw: dict[str, Any],
        first_seen: int | None = None,
        finished_frame: int | None = None,
    ) -> UnitInfo:
        d = units.by_id(def_id) or {}
        pos = raw.get("pos") or [0, 0, 0]
        return UnitInfo(
            uid=uid,
            def_id=def_id,
            name=d.get("name") or raw.get("name") or f"def{def_id}",
            data=d,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            health=float(raw.get("health") or 0.0),
            max_health=float(raw.get("maxHealth") or 1.0),
            build_progress=float(raw.get("buildProgress") or 0.0),
            is_being_built=bool(raw.get("isBeingBuilt") or False),
            experience=float(raw.get("experience") or 0.0),
            first_seen=self.frame if first_seen is None else int(first_seen),
            finished_frame=-1 if finished_frame is None else int(finished_frame),
        )

    def read_units(self, obs: dict[str, Any]) -> None:
        new_units: dict[int, UnitInfo] = {}
        for uid_s, raw in (obs.get("units") or {}).items():
            try:
                uid = int(uid_s)
                did = int(raw.get("defId") or 0)
                previous = self.units.get(uid)
                finished_frame = previous.finished_frame if previous else -1
                if bool(raw.get("isBeingBuilt") or False) is False and float(
                    raw.get("buildProgress") or 0.0
                ) >= 0.985:
                    if finished_frame < 0:
                        finished_frame = self.frame
                elif previous is not None and not previous.is_finished:
                    finished_frame = -1
                new_units[uid] = self.make_unit(
                    uid,
                    did,
                    raw,
                    first_seen=previous.first_seen if previous else self.frame,
                    finished_frame=finished_frame,
                )
            except Exception:
                continue
        old_ids = set(self.units)
        for uid, u in new_units.items():
            if uid in old_ids:
                continue
            if u.is_factory and u.is_finished:
                self.say(f"factory{uid}", f"factory complete: {u.name} id={uid}")
            elif u.name == "staticcon":
                self.say(f"caretaker{uid}", f"caretaker complete id={uid}")
            elif u.is_builder:
                self.log(f"new builder: {u.name} id={uid}")

        # Release jobs of dead builders.
        for uid in list(self.builder_task):
            if uid not in new_units:
                key = self.builder_task.pop(uid, None)
                if key and key in self.tasks:
                    self.tasks[key].assignees.discard(uid)
                self.builder_idle.pop(uid, None)
                self.builder_job_frame.pop(uid, None)
                self.log(f"builder #{uid} died; released {key}")
        self.units = new_units
        self.note_new_units(new_units, old_ids)
        if self.commander_id in self.units:
            self.commander = self.units[self.commander_id]
        for uid in list(self.retreating):
            if uid not in self.units:
                self.retreating.discard(uid)
        self.army = {uid for uid, u in self.units.items() if u.is_combat and u.is_finished}
        # Dedicated Gremlin spotters are controlled by scouting.py, not by the
        # normal army/home-guard logic.
        self.army -= getattr(self, "gremlin_scouts", set())
        self.raiders = {
            uid
            for uid in self.army
            if units.role_of(self.units[uid].data) in ("raid", "scout")
            and not self.units[uid].is_air
        }

    def read_enemies(self, obs: dict[str, Any]) -> None:
        now = self.frame
        enemies: dict[int, UnitInfo] = {}
        for uid_s, raw in (obs.get("enemies") or {}).items():
            try:
                uid = int(uid_s)
                did = int(raw.get("defId") or 0)
                u = self.make_unit(uid, did, raw)
                enemies[uid] = u
                self.enemy_memory[uid] = (u, now)
                seen = getattr(self, "enemy_seen_ids", None)
                if seen is None:
                    seen = self.enemy_seen_ids = {}
                seen.setdefault(u.name, set()).add(uid)
                self.note_air_contact(u)
                name = u.name.lower()
                if "dyntrainer" in name or name.startswith("commander") or "commander" in name:
                    self.enemy_commander_id = uid
                    if u.health <= 0:
                        self.enemy_commander_dead = True
                        self.enemy_commander_id = None
                        self.enemy_memory.pop(uid, None)
                # A roaming commander only stands in for the base until an
                # enemy factory has been seen.
                if u.is_factory or (
                    ("dyntrainer" in name or "commander" in name)
                    and not getattr(self, "enemy_factory_seen", False)
                ):
                    if u.is_factory:
                        self.enemy_factory_seen = True
                    if self.dist(u.pos, self.base_pos) > 700:
                        if now - self.enemy_base_seen_frame > 300 or self.enemy_base is None:
                            self.enemy_base = u.pos
                            self.enemy_base_seen_frame = now
                if not u.is_builder and not u.is_factory:
                    self.enemy_army_pos = u.pos
                    self.enemy_army_frame = now
            except Exception:
                continue
        self.enemies = enemies
        self.radar = {}
        for uid_s, raw in (obs.get("radarBlips") or {}).items():
            try:
                uid = int(uid_s)
                did = int(raw.get("defId") or 0)
                u = self.make_unit(uid, did, raw)
                self.radar[uid] = u
                self.enemy_memory[uid] = (u, now)
                self.note_air_contact(u)
                name = u.name.lower()
                if "dyntrainer" in name or "commander" in name:
                    hp_raw = raw.get("health")
                    if hp_raw is not None and float(hp_raw) <= 0:
                        self.enemy_commander_dead = True
                        self.enemy_commander_id = None
                        self.enemy_memory.pop(uid, None)
                    else:
                        self.enemy_commander_id = uid
                if u.is_factory and self.dist(u.pos, self.base_pos) > 700:
                    self.enemy_factory_seen = True
                    self.enemy_base = u.pos
                    self.enemy_base_seen_frame = now
                if not u.is_builder:
                    self.enemy_army_pos = u.pos
                    self.enemy_army_frame = now
            except Exception:
                continue
        if enemies or self.radar:
            self.last_enemy_seen_frame = now
        cutoff = now - 1800
        for uid in list(self.enemy_memory):
            if self.enemy_memory[uid][1] < cutoff:
                self.enemy_memory.pop(uid, None)
        if self.enemy_commander_id is not None and self.enemy_commander_id not in self.enemy_memory:
            self.enemy_commander_id = None

    def read_features(self, obs: dict[str, Any]) -> None:
        self._feature_cache = obs.get("features") or {}

    def update_spot_status(self) -> None:
        # Rebuild the status from scratch each time.  The old version reset
        # visible enemy extractors to "free", which made the expander and the
        # raiders think enemy metal spots were available.
        for s in self.map.spots:
            s.status = "free"
            s.task_id = None
        for t in self.tasks.values():
            if t.kind == "mex" and t.spot_index is not None:
                idx = int(t.spot_index)
                if 0 <= idx < len(self.map.spots):
                    self.map.spots[idx].status = "building"
                    self.map.spots[idx].task_id = t.key
        for u in self.units.values():
            if not u.is_finished or u.name != "staticmex":
                continue
            for s in self.map.spots:
                if self.dist((s.x, s.z), u.pos) <= 105:
                    s.status = "ours"
                    break
        remembered = [e for e, _f in getattr(self, "enemy_structures", {}).values()]
        for u in list(self.enemies.values()) + remembered:
            if u.name != "staticmex":
                continue
            for s in self.map.spots:
                if self.dist((s.x, s.z), u.pos) <= 105:
                    s.status = "enemy"
                    break

    def update_observation_rate(self) -> None:
        if not self.observation_setting_sent:
            return
        combat = (
            self.threat_pos is not None
            and self.frame - self.threat_frame < 300
        ) or self.hunt_commander
        if self.enemies:
            nearest = min(
                (self.dist(u.pos, self.base_pos) for u in self.enemies.values()),
                default=1e18,
            )
            combat = combat or nearest < 1500.0
        if combat:
            self.last_combat_frame = self.frame
        # 30 frames per second: 6 frames (0.2 s) is plenty for fights and
        # 10 for quiet periods.  The old 2-frame combat setting, which was
        # active most of the game, made the engine wait for Python 15 times
        # per game second and dominated wall time in long games.
        fight_n, calm_n = getattr(self, "block_setting", (6, 10))
        want = fight_n if self.frame - self.last_combat_frame < 6000 else calm_n
        if want != self.block_frames:
            self.block_frames = want
            try:
                self.client.set_settings({"block_n_frames": want})
                self.log(f"block_n_frames={want}")
            except Exception:
                pass

    def update_threat(self) -> None:
        threat = None
        best = 1e18
        mexes = [
            u.pos
            for u in self.units.values()
            if u.name == "staticmex" and u.is_finished
        ]
        for u in list(self.enemies.values()) + list(self.radar.values()):
            if u.is_builder and "dyntrainer" not in u.name and "commander" not in u.name:
                # Ordinary builders are not threats by themselves, but an
                # enemy commander is.
                if not u.is_combat:
                    continue
            d = self.dist(u.pos, self.base_pos)
            near_mex = any(self.dist(u.pos, p) < 600 for p in mexes)
            near_army = any(
                self.dist(u.pos, self.units[uid].pos) < 700
                for uid in self.army
                if uid in self.units
            )
            if (d < 2600 or near_mex or near_army) and d < best:
                best = d
                threat = u.pos
        if threat:
            self.threat_pos = threat
            self.threat_frame = self.frame

    # ----------------------------------------------------------------- economy

    def is_dangerous(self, e: UnitInfo) -> bool:
        # Commanders are builders, so is_combat is false for them, but they
        # are the most dangerous units on the map.
        return e.is_combat or "dyntrainer" in e.name or "commander" in e.name

    def enemy_near(self, pos: tuple[float, float], radius: float) -> UnitInfo | None:
        best = None
        best_d = radius
        for e in list(self.enemies.values()) + list(self.radar.values()):
            if not self.is_dangerous(e):
                continue
            d = self.dist(pos, e.pos)
            if d < best_d:
                best = e
                best_d = d
        return best

    def store_economy(self, obs: dict[str, Any]) -> None:
        self.game_info["_economy"] = obs.get("economy") or {}

    def metal(self) -> dict:
        return (self.game_info.get("_economy") or {}).get("Metal") or {}

    def energy(self) -> dict:
        return (self.game_info.get("_economy") or {}).get("Energy") or {}

    def metal_current(self) -> float:
        return float(self.metal().get("current") or 0.0)

    def metal_income(self) -> float:
        return float(self.metal().get("income") or 0.0)

    def energy_current(self) -> float:
        return float(self.energy().get("current") or 0.0)

    def energy_income(self) -> float:
        return float(self.energy().get("income") or 0.0)

    def energy_usage(self) -> float:
        return float(self.energy().get("usage") or 0.0)

    @staticmethod

    @staticmethod
    def effective_storage(resource: dict) -> float:
        raw = float(resource.get("storage") or 0.0)
        # Zero-K reserves 10000 hidden storage so that caretakers can spend in
        # chunks.  The visible cap is raw - 10000.
        return max(1.0, raw - 10000.0 if raw > 10000.0 else raw)

    def mex_count(self) -> int:
        return sum(1 for u in self.units.values() if u.name == "staticmex" and u.is_finished)

    def builder_count(self) -> int:
        # Count constructors that are still being built too.  Otherwise the
        # factory keeps seeing a below-target count while two extra builders
        # are already in its queue.
        return sum(
            1
            for u in self.units.values()
            if u.is_builder and u.is_mobile
        )

    def factory_list(self) -> list[UnitInfo]:
        return [u for u in self.units.values() if u.is_factory and u.is_finished]

    def combat_count(self) -> int:
        return len(self.army)
