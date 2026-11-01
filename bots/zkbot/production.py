"""ProductionMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class ProductionMixin:
    def desired_builder_count(self) -> int:
        t = self.frame_time()
        if t < 120:
            return 4
        if t < 300:
            return 5
        return 6

    def next_unit_for_factory(self, factory: UnitInfo) -> int | None:
        fname = factory.name or self.factory_name or "factorycloak"
        builders = self.builder_count()
        if fname == "striderhub":
            if self.metal_income() >= 30:
                roles = ["assault", "riot", "heavy", "arty", "assault"]
            else:
                roles = ["riot", "assault", "antiheavy"]
            role = roles[self.air_counter % len(roles)]
            self.air_counter += 1
            did = units.factory_unit(fname, role)
            if did is None:
                did = units.factory_unit(fname, "assault")
            if did is not None:
                self.log(f"queue strider {units.name_of(did)}")
            return did
        if fname in ("factoryplane", "factorygunship"):
            # Air is for hunting and scouting; do not spend its queue on
            # builders while the ground economy already has constructors.
            n = self.air_counter
            self.air_counter += 1
            if self.hunt_commander:
                ec = (
                    self.enemies.get(self.enemy_commander_id)
                    if self.enemy_commander_id is not None
                    else None
                )
                low = ec is not None and ec.alive_frac < 0.5
                if fname == "factoryplane":
                    if low:
                        roles = ["assault", "heavy", "bomb", "assault", "heavy", "bomb"]
                    else:
                        roles = [
                            "scout",
                            "bomb",
                            "heavyfighter",
                            "heavy",
                            "bomb",
                            "assault",
                            "bomb",
                        ]
                else:
                    roles = ["emp", "assault", "skirm", "emp", "raid"]
            else:
                if fname == "factoryplane":
                    roles = ["fighter", "scout", "bomb", "heavyfighter", "bomb"]
                else:
                    roles = ["raid", "skirm", "aa"]
            role = roles[n % len(roles)]
            did = units.factory_unit(fname, role)
            if did is None:
                did = units.factory_unit(fname, "fighter" if fname == "factoryplane" else "raid")
            return did
        force_combat = bool(
            self.threat_pos
            and self.frame - self.threat_frame < 300
            and len(self.army) < 6
        )
        if force_combat:
            enemy_air_here = any(u.is_air for u in self.enemies.values())
            if enemy_air_here:
                role = "aa"
            else:
                role = "riot" if self.production_counter % 2 == 0 else "raid"
            self.production_counter += 1
        elif builders < self.desired_builder_count() and (
            len(self.army) >= 2 or builders < 2
        ):
            role = "builder"
        elif self.hunt_commander:
            n = self.production_counter
            self.production_counter += 1
            hunt_roles = ["assault", "riot", "bomb", "skirm", "assault", "snipe", "riot", "skirm"]
            role = hunt_roles[n % len(hunt_roles)]
        else:
            enemy_air = any(u.is_air for u in self.enemies.values())
            enemy_raiders = sum(
                1 for u in self.enemies.values() if units.role_of(u.data) in ("raid", "scout")
            )
            enemy_heavy = sum(
                1
                for u in self.enemies.values()
                if units.role_of(u.data) in ("riot", "assault", "heavy")
            )
            our_aa = sum(1 for uid in self.army if units.is_aa(self.units[uid].data))
            our_riot = sum(
                1 for uid in self.army if units.role_of(self.units[uid].data) == "riot"
            )
            our_skirm = sum(
                1 for uid in self.army if units.role_of(self.units[uid].data) == "skirm"
            )
            our_assault = sum(
                1 for uid in self.army if units.role_of(self.units[uid].data) == "assault"
            )
            our_arty = sum(
                1 for uid in self.army if units.role_of(self.units[uid].data) == "arty"
            )
            if enemy_air and our_aa < max(1, len(self.army) // 6):
                role = "aa"
            elif enemy_raiders >= 4 and our_riot < 2:
                role = "riot"
            elif enemy_heavy >= 3 and our_skirm < 2:
                role = "skirm"
            elif len(self.army) > 10 and our_assault < max(1, len(self.army) // 10):
                role = "assault"
            elif len(self.army) > 12 and our_arty < max(1, len(self.army) // 12):
                role = "arty"
            else:
                n = self.production_counter
                self.production_counter += 1
                m = n % 10
                if m <= 2:
                    role = "raid"
                elif m == 3:
                    role = "bomb"
                elif m <= 6:
                    role = "skirm"
                elif m <= 8:
                    role = "riot"
                elif n % 20 == 9:
                    role = "assault"
                else:
                    role = "arty"
        did = units.factory_unit(fname, role)
        if did is None:
            did = units.factory_unit(fname, "raid")
        return did

    def production_step(self) -> None:
        factories = self.factory_list()
        if not factories:
            return
        for f in factories:
            q = self.factory_queue.get(f.uid, 0)
            if q >= 2:
                continue
            did = self.next_unit_for_factory(f)
            if did is None:
                continue
            unit_name = units.name_of(did)
            if f.name in ("factoryplane", "factorygunship"):
                self.log(f"queue air unit {unit_name} at {f.name}")
            elif unit_name == "cloakbomb":
                self.log(f"queue cloak bomb at {f.name}")
            self.queue(
                {
                    "type": "build",
                    "unitId": int(f.uid),
                    "defId": int(did),
                    "pos": [float(f.x), 0.0, float(f.z)],
                }
            )
            self.factory_queue[f.uid] = q + 1

    # ---------------------------------------------------------------- queries

    def query_step(self) -> None:
        if self.frame - self.last_od_query_frame >= 300 and self.frame > 60:
            self.last_od_query_frame = self.frame
            try:
                res = self.client.query(
                    {
                        "type": "team_rules_params",
                        "keys": [
                            "OD_team_metalOverdrive",
                            "OD_team_energyOverdrive",
                            "OD_team_energyWaste",
                            "OD_metalBase",
                            "OD_metalOverdrive",
                            "OD_energyOverdrive",
                        ],
                    },
                    timeout=5.0,
                )
                self.debug_queries += 1
                values = (res or {}).get("values") or {}

                def f(key: str) -> float:
                    entry = values.get(key) or {}
                    try:
                        return float(entry.get("floatValue") or 0.0)
                    except Exception:
                        return 0.0

                self.od_metal = f("OD_team_metalOverdrive")
                self.od_energy = f("OD_team_energyOverdrive")
                self.od_waste = f("OD_team_energyWaste")
            except QueryError:
                pass

        for f in self.factory_list():
            last = self.factory_query_frame.get(f.uid, -10**9)
            if self.frame - last < 60:
                continue
            try:
                res = self.client.query(
                    {"type": "unit_queue_state", "unitId": int(f.uid)}, timeout=5.0
                )
                self.debug_queries += 1
                self.factory_query_frame[f.uid] = self.frame
                if res:
                    self.factory_queue[f.uid] = int(res.get("currentCommandCount") or 0)
            except QueryError:
                self.factory_query_frame[f.uid] = self.frame

        # Detect builders that are stuck without commands.  Do not query a
        # builder in the frame it was assigned: the command has not been sent
        # yet and the engine would correctly report it as idle.
        for uid, key in list(self.builder_task.items()):
            u = self.units.get(uid)
            if not u or not u.is_builder:
                continue
            job_frame = self.builder_job_frame.get(uid, self.frame)
            if self.frame - job_frame < 180:
                continue
            if key not in self.tasks:
                continue
            try:
                res = self.client.query(
                    {"type": "unit_queue_state", "unitId": int(uid)}, timeout=5.0
                )
                self.debug_queries += 1
                if res and (not res.get("hasCommands") or res.get("isIdle")):
                    t = self.tasks.get(key)
                    if t is not None:
                        building = None
                        for b in self.units.values():
                            if b.def_id != t.def_id:
                                continue
                            if self.dist(b.pos, (t.x, t.z)) < 170:
                                building = b
                                break
                        if building is not None and building.is_finished:
                            if building.first_seen >= t.created_frame:
                                self.finish_task(key)
                            else:
                                self.drop_task(key, "occupied")
                        elif building is not None:
                            # The builder has no queued command but the target
                            # still exists.  Re-issue the build order and keep
                            # the assignment: releasing it here caused large
                            # factories to be abandoned half-built.
                            t.x, t.z = building.pos
                            t.placed = True
                            # Assist the existing unfinished building.  Issuing
                            # another "build" at its center is rejected by the
                            # engine because the footprint is occupied.
                            self.queue(
                                {
                                    "type": "guard",
                                    "unitId": int(uid),
                                    "targetId": int(building.uid),
                                }
                            )
                            self.builder_job_frame[uid] = self.frame
                            self.builder_idle[uid] = False
                            self.log(f"assist build {u.name}#{uid} -> {key}")
                            continue
                        else:
                            # The command was rejected or the site is unusable.
                            # Re-query next time instead of reusing a bad spot.
                            t.placed = False
                            if t.anchor:
                                t.x, t.z = t.anchor
                        t.assignees.discard(uid)
                    self.builder_task.pop(uid, None)
                    self.builder_idle[uid] = True
                    self.builder_job_frame.pop(uid, None)
                    self.stuck_releases += 1
                    self.log(f"release idle builder {u.name}#{uid} from {key}")
            except QueryError:
                pass

    # ---------------------------------------------------------------- military
