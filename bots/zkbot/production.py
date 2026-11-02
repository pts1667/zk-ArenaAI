"""ProductionMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
import os
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo


class ProductionMixin:
    def desired_builder_count(self) -> int:
        # CircuitAI's buildpower ratio is about 1.2 buildpower per metal
        # income, not a fixed number of constructors.  Extra builders are a
        # liability early: they eat the factory queue and the raiders eat
        # them.  The commander plus one or two cloakcons is enough while
        # expanding; add more only once the economy can pay for them.
        # Mobile build power follows metal income (CircuitAI Brutal uses
        # 1.25 BP per metal/s).  The count includes the commander (12 BP);
        # each cloakcon adds 5.  Builders are cheap compared with the
        # extractors they claim, and dead ones must be replaced quickly.
        t = self.frame_time()
        minc = self.metal_income()
        bp_con = 5.0
        want_cons = math.ceil(max(0.0, 1.25 * minc - 12.0) / bp_con)
        lo = 2 if t < 60 else 3
        if len(self.army) < 3 and t > 90:
            lo = 2
        cons = max(lo, min(9, want_cons))
        return 1 + cons

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
            return self.next_air_unit(factory)
        if self.should_build_gremlin_scout():
            did = units.id_of("cloakaa")
            if did is not None:
                self.gremlin_pending = int(getattr(self, "gremlin_pending", 0)) + 1
                self.log("queue gremlin scout for artillery LOS")
                return did
        # Emergency combat units must not starve expansion completely: an
        # early raider poke used to stop constructor production for minutes
        # (3 builders and 5 extractors from 1:00 to 5:30).
        flags = os.environ.get("ZKBOT_FLAGS", "")
        # Lost constructors come back first: after a raid killed both cons
        # the bot ran on the commander alone with 1-3 extractors for five
        # minutes while it built combat units.
        min_builders = 1 if "staticmix" in flags else min(3, self.desired_builder_count())
        force_combat = bool(
            self.threat_pos
            and self.frame - self.threat_frame < 300
            and len(self.army) < 6
            and (
                "buildfirst" not in flags
                or builders >= min(4, self.desired_builder_count())
            )
            and not (builders < min_builders and len(self.army) >= 2)
        )
        if force_combat:
            ec = self.enemy_composition()
            if ec.get("air", 0) > 0 and self.aircraft_seen():
                role = "aa"
            elif "staticmix" in flags:
                role = "riot" if self.production_counter % 2 == 0 else "raid"
            else:
                # Raiders answer raiders while the numbers are small; a riot
                # only once several enemy raiders are around and we have
                # fewer than one riot per three of them.
                our_riot = sum(
                    1 for uid in self.army
                    if units.role_of(self.units[uid].data) == "riot"
                )
                many = ec.get("raider", 0) >= 4 and our_riot * 3 < ec.get("raider", 0)
                role = "riot" if many else "raid"
            self.production_counter += 1
        elif (
            builders < self.desired_builder_count()
            and (len(self.army) >= 2 or builders < 2)
            and not (
                builders < 2
                and self.factory_queue.get(factory.uid, 0) > 0
            )
        ):
            role = "builder"
        elif self.hunt_commander:
            n = self.production_counter
            self.production_counter += 1
            hunt_roles = ["assault", "riot", "bomb", "skirm", "assault", "snipe", "riot", "skirm"]
            role = hunt_roles[n % len(hunt_roles)]
        elif (counter := self.counter_role()) is not None:
            role = counter
            if self.frame - self.last_composition_log_frame >= 300:
                self.last_composition_log_frame = self.frame
                ec = self.enemy_composition()
                self.log(
                    "counter "
                    f"{role} enemy air={ec['air']} raid={ec['raider']} "
                    f"skirm={ec['skirm']} riot={ec['riot']} "
                    f"assault={ec['assault']} arty={ec['arty']} "
                    f"static={ec['static']}"
                )
        else:
            enemy_air = self.aircraft_seen()
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
            if enemy_air and our_aa < self.desired_aa_count():
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
                # CircuitAI's tier1 army is mostly skirmishers with a solid
                # riot/assault core.  Bombs (50 HP) and pure raider spam melt
                # against that, so keep only a small raider screen.
                m = n % 12
                if m <= 1:
                    role = "raid"
                elif m == 2:
                    role = "riot"
                elif m == 3:
                    role = "skirm"
                elif m == 4:
                    role = "assault"
                elif m == 5:
                    role = "skirm"
                elif m == 6:
                    role = "riot"
                elif m == 7:
                    role = "skirm"
                elif m == 8:
                    role = "assault"
                elif m == 9:
                    role = "heavyraid"
                elif m == 10:
                    role = "riot"
                else:
                    role = "skirm"
        did = units.factory_unit(fname, role)
        if did is None:
            did = units.factory_unit(fname, "raid")
        return did

    def production_step(self) -> None:
        factories = self.factory_list()
        if not factories:
            return
        # Opening metal priority: while metal is empty and constructors are
        # walking to extractors, do not queue more combat units beyond a
        # small screen; an 85-metal extractor pays back in ~45 s.
        hold_army = (
            'nohold' not in os.environ.get('ZKBOT_FLAGS', '')
            and self.frame_time() < 300
            and self.metal_current() < 40.0
            and len(self.army) >= 4
            and not (self.threat_pos is not None and self.frame - self.threat_frame < 300)
            and any(t.kind == "mex" and t.assignees for t in self.tasks.values())
            and self.builder_count() >= self.desired_builder_count()
        )
        for f in factories:
            q = self.factory_queue.get(f.uid, 0)
            if q >= 2 or (hold_army and q >= 1):
                continue
            if hold_army and f.name not in ("factoryplane", "factorygunship") and q == 0:
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
            # Once a second per builder is enough to notice an idle one; this
            # check used to run on every observation for every builder.
            bq = self.__dict__.setdefault("builder_query_frame", {})
            if self.frame - bq.get(uid, -10**9) < 30:
                continue
            bq[uid] = self.frame
            try:
                res = self.client.query(
                    {"type": "unit_queue_state", "unitId": int(uid)}, timeout=5.0
                )
                self.debug_queries += 1
                if res and (not res.get("hasCommands") or res.get("isIdle")):
                    t = self.tasks.get(key)
                    if t is not None:
                        building = None
                        # Energy tasks are created close together around a
                        # base; a 170-elmo match radius found a generator two
                        # slots away and declared the new job "occupied".
                        match_radius = 50.0 if t.kind in ("energy", "fusion") else 170.0
                        for b in self.units.values():
                            if b.def_id != t.def_id:
                                continue
                            if self.dist(b.pos, (t.x, t.z)) < match_radius:
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
                            t.placement_attempts += 1
                            # The cached "site is free" answer is now known
                            # to be wrong; re-probe for real.
                            self.client._cache.clear()
                            if t.anchor:
                                t.x, t.z = t.anchor
                            if t.placement_attempts >= 3:
                                if t.kind == "mex" and t.spot_index is not None:
                                    self.blocked_spot_until[int(t.spot_index)] = (
                                        self.frame + 1800
                                    )
                                self.drop_task(key, "blocked")
                                self.builder_task.pop(uid, None)
                                self.builder_idle[uid] = True
                                self.builder_job_frame.pop(uid, None)
                                self.stuck_releases += 1
                                continue
                        t.assignees.discard(uid)
                    self.builder_task.pop(uid, None)
                    self.builder_idle[uid] = True
                    self.builder_job_frame.pop(uid, None)
                    self.stuck_releases += 1
                    self.log(f"release idle builder {u.name}#{uid} from {key}")
            except QueryError:
                pass

    # ---------------------------------------------------------------- military
