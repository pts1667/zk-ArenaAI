"""ConstructionMixin methods for the zkbot Brain."""
from __future__ import annotations

import math
import os
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo

FLAGS = os.environ.get("ZKBOT_FLAGS", "")


class ConstructionMixin:
    def idle_builders(self) -> list[UnitInfo]:
        out = []
        factories = self.factory_list()
        for u in self.units.values():
            if not (u.is_builder and u.is_finished):
                continue
            if not u.is_mobile:
                if u.name != "staticmissilesilo":
                    continue
                if not any(t.kind == "seismic" for t in self.tasks.values()):
                    continue
            if (
                u.uid == self.commander_id
                and self.factory_plop_sent
                and not factories
            ):
                continue
            key = self.builder_task.get(u.uid)
            if key:
                if key in self.tasks:
                    continue
                self.builder_task.pop(u.uid, None)
            if self.frame < self.builder_fallback_until.get(u.uid, -1):
                continue
            # Invariant: a builder without a live task is available once its
            # fallback/flee timer has expired.  (The old builder_idle flag was
            # set False by retreat_builder and never reset, which orphaned
            # every builder that had fled once.)
            self.builder_idle[u.uid] = True
            out.append(u)
        return out

    def retreat_builder(self, u: UnitInfo, danger_pos: tuple[float, float] | None = None) -> None:
        target = self.builder_safe_point(u, danger_pos)
        self.queue(
            {"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]]}
        )
        key = self.builder_task.pop(u.uid, None)
        if key and key in self.tasks:
            self.tasks[key].assignees.discard(u.uid)
        self.builder_idle[u.uid] = True
        self.builder_job_frame.pop(u.uid, None)
        self.builder_fallback_until[u.uid] = self.frame + 150
        self.builder_flee_count = getattr(self, "builder_flee_count", 0) + 1
        self.log(f"builder {u.name}#{u.uid} fleeing to {tuple(round(v) for v in target)}")

    def site_safe_for_builder(self, pos: tuple[float, float], kind: str = "") -> bool:
        """A build site is usable if enemy DPS there is covered by our own."""
        threat = self.threat_at(pos)
        if threat <= 0.0:
            return True
        own = self.own_power_at(pos)
        # Turrets are exactly what a contested site needs, so tolerate a bit
        # more risk for them when our army is present.
        factor = 1.0 if kind in ("defense", "mexdef", "frontdef") else 1.4
        return own >= threat * factor and threat < 400.0

    def builder_safety_step(self) -> None:
        """Pull constructors out of enemy weapon reach before they die.

        Runs for every mobile constructor, including ones walking to or
        working on a job.  A builder whose position (or job site) is under
        uncovered enemy fire drops the job and moves to the safest nearby
        point toward home; the job stays open for later.
        """
        for u in list(self.units.values()):
            if not (u.is_builder and u.is_mobile and u.is_finished):
                continue
            if u.uid == self.commander_id:
                continue
            if self.frame < self.builder_fallback_until.get(u.uid, -1):
                continue
            here = self.threat_at(u.pos)
            own = self.own_power_at(u.pos)
            key = self.builder_task.get(u.uid)
            t = self.tasks.get(key) if key else None
            site_bad = False
            if t is not None and self.dist(u.pos, (t.x, t.z)) < 900.0:
                site_bad = not self.site_safe_for_builder((t.x, t.z), t.kind)
            if (here > 20.0 and own < here * 1.2) or site_bad:
                danger = self.enemy_near(u.pos, 1000.0)
                self.retreat_builder(u, danger.pos if danger is not None else None)
                continue
            # Raiders on radar heading here: step aside before they arrive.
            w = self.warning_near(u.pos, 10.0)
            if w is not None and self.own_power_at(w["pos"]) < w["power"] * 1.2:
                self.log(f"builder {u.name}#{u.uid} early raid warning ({w['n']} raiders, eta {w['eta']:.0f}s)")
                self.retreat_builder(u, w["from"])

    def builder_safe_point(
        self, u: UnitInfo, danger_pos: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Run away from the danger toward home, not across the enemy."""
        if hasattr(self, "safest_point_near"):
            p = self.safest_point_near(u.pos, toward=self.base_pos, radius=700.0)
            # A nearby point is only useful if it is genuinely quiet;
            # otherwise go all the way home instead of oscillating mid-map.
            if p is not None and self.threat_in_radius(p, 250.0) <= 0.0:
                return p
            home = self.safest_point_near(self.base_pos, toward=self.base_pos, radius=300.0)
            if home is not None:
                return home
        return self.retreat_point(u)

    def assign_builders(self) -> None:
        if self.frame - self.last_assign_frame < 5:
            return
        self.last_assign_frame = self.frame
        builders = self.idle_builders()
        if not builders:
            return
        tasks = [
            t
            for t in self.tasks.values()
            if not t.unplaceable and len(t.assignees) < t.max_assignees
        ]
        tasks.sort(key=lambda t: t.priority, reverse=True)
        for u in builders:
            if not self.builder_idle.get(u.uid, True) and u.uid in self.builder_task:
                continue
            if u.uid == self.commander_id:
                # The commander is the economy engine, not a forward
                # constructor.  Losing it to a raider at a frontier turret
                # collapses the whole game.
                if self.dist(u.pos, self.base_pos) > self.commander_radius() + 150.0:
                    self.queue(
                        {
                            "type": "move",
                            "unitId": int(u.uid),
                            "pos": [self.base_pos[0], 0.0, self.base_pos[1]],
                        }
                    )
                    self.log(f"commander returning home from {tuple(round(v) for v in u.pos)}")
                    continue
                # Only an enemy close to the commander itself stops work; a
                # raider lingering at the base edge used to idle it for minutes.
                danger = self.enemy_near(u.pos, 600)
                if danger is not None and danger.is_finished:
                    continue
            danger = self.enemy_near(u.pos, 350)
            if danger is not None and u.uid != self.commander_id:
                self.retreat_builder(u, danger.pos)
                continue
            chosen = None
            candidates: list[tuple[float, Task]] = []
            for t in tasks:
                if u.uid in t.assignees or len(t.assignees) >= t.max_assignees:
                    continue
                d = self.dist(u.pos, (t.x, t.z))
                if u.name == "staticmissilesilo":
                    if t.kind != "seismic":
                        continue
                    limit = 180.0
                elif t.kind == "seismic":
                    continue
                elif u.uid == self.commander_id:
                    # Any job inside the home perimeter.  The commander is the
                    # strongest constructor (12 BP); keeping it off the
                    # nearby extractors made the opening far too slow.
                    if t.kind in ("frontdef", "silo"):
                        continue
                    reach = self.commander_radius()
                    # Expansion first: extractors anywhere in its working
                    # radius, everything else only near home.
                    if self.dist((t.x, t.z), self.base_pos) > (reach if t.kind == "mex" else 850.0):
                        continue
                    limit = reach + 300.0
                elif t.kind in ("energy", "fusion", "pylon"):
                    limit = 2600.0
                else:
                    limit = 7000.0
                if d > limit:
                    continue
                if not self.site_safe_for_builder((t.x, t.z), t.kind):
                    continue
                path_risk = self.path_threat(u.pos, (t.x, t.z))
                if path_risk > 60.0 and u.uid != self.commander_id:
                    continue
                score = t.priority - d * (0.20 if u.uid == self.commander_id else 0.08)
                if (
                    u.uid == self.commander_id
                    and t.kind == "mex"
                    and "staticcomm" not in FLAGS
                ):
                    score += 250.0
                score -= path_risk * 4.0
                candidates.append((score, t))
            candidates.sort(key=lambda p: p[0], reverse=True)
            for _score, t in candidates:
                if self.ensure_placement(t):
                    chosen = t
                    break
            if chosen is None:
                self.builder_fallback(u)
                continue
            existing = None
            # For tightly packed generators this must be the footprint radius,
            # not 220: base solar rings are 135 apart, so every new energy job
            # used to be "completed" by the previous solar the moment it was
            # assigned.  That is what kept energy income flat for minutes.
            existing_radius = 50.0 if chosen.kind in ("energy", "fusion") else 220.0
            for b in self.units.values():
                if b.def_id != chosen.def_id:
                    continue
                if self.dist(b.pos, (chosen.x, chosen.z)) < existing_radius:
                    existing = b
                    break
            if existing is not None:
                if existing.is_finished:
                    self.finish_task(chosen.key)
                    continue
                # Another constructor already started this footprint.  Help
                # it finish instead of spamming a build order the engine will
                # reject as occupied.
                self.queue(
                    {
                        "type": "guard",
                        "unitId": int(u.uid),
                        "targetId": int(existing.uid),
                    }
                )
                chosen.assignees.add(u.uid)
                self.builder_task[u.uid] = chosen.key
                self.builder_idle[u.uid] = False
                self.builder_job_frame[u.uid] = self.frame
                self.log(
                    f"assist existing {u.name}#{u.uid} -> {chosen.key}"
                )
                continue
            if not (chosen.kind == "mex" and self.try_burrow_mex(u, chosen)):
                self.queue(
                    {
                        "type": "build",
                        "unitId": int(u.uid),
                        "defId": int(chosen.def_id),
                        "pos": [float(chosen.x), 0.0, float(chosen.z)],
                        "facing": int(chosen.facing),
                    }
                )
            chosen.assignees.add(u.uid)
            self.builder_task[u.uid] = chosen.key
            self.builder_idle[u.uid] = False
            self.builder_job_frame[u.uid] = self.frame
            self.log(
                f"assign {u.name}#{u.uid} -> {chosen.kind} {chosen.key} "
                f"at ({chosen.x:.0f},{chosen.z:.0f}) "
                f"d={self.dist(u.pos,(chosen.x,chosen.z)):.0f}"
            )

    def generator_range(self, name: str) -> float:
        return {
            "energysolar": 100.0,
            "energywind": 63.0,
            "energyfusion": 150.0,
            "energysingu": 150.0,
        }.get(name, 0.0)

    def site_ok(self, t: Task, x: float, z: float) -> bool:
        margin = 320.0 if t.kind in ("factory", "strider") else 45.0
        if (
            x < margin
            or z < margin
            or x > self.map.width - margin
            or z > self.map.height - margin
        ):
            return False
        # Energy generators must stay linked to a metal extractor so that
        # Zero-K's overdrive grid can use their output.  Generators link to
        # each other too, so allow a small cluster around every mex.
        if t.kind in ("energy", "fusion"):
            link_range = 200.0 if t.kind == "fusion" else 150.0
            mexes = [
                u
                for u in self.units.values()
                if u.name == "staticmex" and (u.is_finished or u.is_being_built)
            ]
            if mexes:
                direct = any(
                    math.hypot(x - u.x, z - u.z) <= link_range for u in mexes
                )
                if not direct:
                    my_range = self.generator_range(units.name_of(t.def_id))
                    linked = False
                    for u in self.units.values():
                        if not (u.is_finished or u.is_being_built):
                            continue
                        if u.name not in (
                            "energysolar",
                            "energywind",
                            "energyfusion",
                            "energysingu",
                        ):
                            continue
                        if math.hypot(x - u.x, z - u.z) > my_range + self.generator_range(u.name):
                            continue
                        if any(
                            math.hypot(u.x - m.x, u.z - m.z) <= link_range
                            for m in mexes
                        ):
                            linked = True
                            break
                    if not linked:
                        for other in self.tasks.values():
                            if other is t or not other.placed:
                                continue
                            if other.kind not in ("energy", "fusion"):
                                continue
                            if math.hypot(x - other.x, z - other.z) > my_range + self.generator_range(
                                units.name_of(other.def_id)
                            ):
                                continue
                            if any(
                                math.hypot(other.x - m.x, other.z - m.z) <= link_range
                                for m in mexes
                            ):
                                linked = True
                                break
                    if not linked:
                        return False
        if t.kind in ("energy", "fusion"):
            # The same solar/fusion already standing on this anchor is not a
            # build site.  Previously a task placed on top of an existing
            # generator was "assisted" forever and the energy income never
            # actually grew.
            for u in self.units.values():
                if u.def_id != t.def_id:
                    continue
                if math.hypot(x - u.x, z - u.z) < 82.0:
                    return False
        # Do not place multiple jobs on the same footprint.
        for other in self.tasks.values():
            if other is t or not other.placed:
                continue
            if other.kind == t.kind and math.hypot(x - other.x, z - other.z) < 85.0:
                return False
        cutoff = self.frame - 900
        for px, pz, pf, _key in self.recent_placements:
            if pf < cutoff:
                continue
            if math.hypot(x - px, z - pz) < 70.0:
                return False
        return True

    def record_placement(self, t: Task) -> None:
        self.recent_placements.append((t.x, t.z, self.frame, t.key))
        cutoff = self.frame - 1800
        self.recent_placements = [p for p in self.recent_placements if p[2] >= cutoff]

    def ensure_placement(self, t: Task) -> bool:
        if t.placed:
            return True
        if self.frame - t.last_placement_frame < 10:
            return False
        t.last_placement_frame = self.frame
        if t.kind == "mex" and t.spot_index is not None:
            s = self.map.spots[t.spot_index]
            try:
                res = self.client.query(
                    {
                        "type": "can_build_at",
                        "unitDefId": int(t.def_id),
                        "pos": [s.x, 0.0, s.z],
                        "facing": 0,
                    }
                )
                self.debug_queries += 1
                if res and res.get("possible"):
                    p = res.get("pos") or [s.x, 0, s.z]
                    t.x = float(p[0])
                    t.z = float(p[2])
                    t.facing = int(res.get("facing") or 0)
                    t.placed = True
                    self.record_placement(t)
                    return True
            except QueryError:
                pass
            try:
                res = self.client.query(
                    {
                        "type": "closest_build_site",
                        "unitDefId": int(t.def_id),
                        "searchRadius": 90,
                        "minDist": 0,
                        "pos": [s.x, 0.0, s.z],
                        "facing": 0,
                    }
                )
                self.debug_queries += 1
                if res and res.get("found"):
                    p = res["pos"]
                    if math.hypot(float(p[0]) - s.x, float(p[2]) - s.z) <= 90:
                        t.x = float(p[0])
                        t.z = float(p[2])
                        t.facing = int(res.get("facing") or 0)
                        t.placed = True
                        self.record_placement(t)
                        return True
            except QueryError:
                pass
            t.placement_attempts += 1
            if t.placement_attempts >= 20:
                t.unplaceable = True
            return False

        anchor = t.anchor or (t.x, t.z)
        if t.kind == "mex":
            # A metal extractor has exactly one legal position: its metal
            # spot.  Letting closest_build_site search a 900-elmo radius
            # placed extractors on unrelated spots, after which the task
            # never completed and the builder re-issued it forever.
            try:
                res = self.client.query(
                    {
                        "type": "can_build_at",
                        "unitDefId": int(t.def_id),
                        "pos": [anchor[0], 0.0, anchor[1]],
                        "facing": t.facing,
                    }
                )
                self.debug_queries += 1
                if res and res.get("possible"):
                    p = res.get("pos") or [anchor[0], 0, anchor[1]]
                    t.x = float(p[0])
                    t.z = float(p[2])
                    t.facing = int(res.get("facing") or 0)
                    t.placed = True
                    self.record_placement(t)
                    return True
            except QueryError:
                pass
            t.placement_attempts += 1
            if t.placement_attempts >= 5:
                t.unplaceable = True
            return False
        if t.kind == "caretaker":
            # A caretaker only helps a factory within its ~500-elmo build
            # range.  closest_build_site used to drift 600-1100 elmos away in
            # crowded bases, so the caretakers never assisted production and
            # metal sat at the cap.  Try a ring of explicit spots instead and
            # keep the factory's exit lane (south, facing 0) free.
            fx, fz = anchor
            for radius in (130, 180, 240, 300, 360, 420):
                for k in range(16):
                    a = 2.0 * math.pi * k / 16.0
                    x = fx + math.cos(a) * radius
                    z = fz + math.sin(a) * radius
                    if z > fz and abs(x - fx) < 110.0:
                        continue
                    if not self.site_ok(t, x, z):
                        continue
                    try:
                        res = self.client.query(
                            {
                                "type": "can_build_at",
                                "unitDefId": int(t.def_id),
                                "pos": [x, 0.0, z],
                                "facing": 0,
                            }
                        )
                        self.debug_queries += 1
                    except QueryError:
                        continue
                    if res and res.get("possible"):
                        q = res.get("pos") or [x, 0, z]
                        t.x = float(q[0])
                        t.z = float(q[2])
                        t.facing = 0
                        t.placed = True
                        self.record_placement(t)
                        return True
            t.placement_attempts += 1
            if t.placement_attempts >= 5:
                t.unplaceable = True
            return False
        if t.kind == "strider":
            radii = (180, 260, 340)
        elif t.kind in ("defense", "frontdef", "aa"):
            radii = (80, 130, 190)
        elif t.kind == "seismic":
            radii = (70, 100, 130, 160)
        elif t.kind == "energy":
            radii = (110, 170, 230)
        elif t.kind == "fusion":
            radii = (170, 250, 330)
        else:
            radii = (240, 420, 650, 900)
        for radius in radii:
            try:
                res = self.client.query(
                    {
                        "type": "closest_build_site",
                        "unitDefId": int(t.def_id),
                        "searchRadius": radius,
                        "minDist": 30,
                        "pos": [anchor[0], 0.0, anchor[1]],
                        "facing": t.facing,
                    }
                )
                self.debug_queries += 1
                if res and res.get("found"):
                    p = res["pos"]
                    if self.site_ok(t, float(p[0]), float(p[2])):
                        t.x = float(p[0])
                        t.z = float(p[2])
                        t.facing = int(res.get("facing") or 0)
                        t.placed = True
                        self.record_placement(t)
                        return True
            except QueryError:
                continue
        # Last resort: try to build directly on the anchor.
        try:
            res = self.client.query(
                {
                    "type": "can_build_at",
                    "unitDefId": int(t.def_id),
                    "pos": [anchor[0], 0.0, anchor[1]],
                    "facing": t.facing,
                }
            )
            self.debug_queries += 1
            if res and res.get("possible"):
                p = res.get("pos") or [anchor[0], 0, anchor[1]]
                if self.site_ok(t, float(p[0]), float(p[2])):
                    t.x = float(p[0])
                    t.z = float(p[2])
                    t.facing = int(res.get("facing") or 0)
                    t.placed = True
                    self.record_placement(t)
                    return True
        except QueryError:
            pass
        # The closest-build-site query always returns the single closest
        # valid spot.  If that spot is already taken by another pending job,
        # probing a small ring gives us distinct footprints around the anchor.
        for radius in (90, 130, 170, 210):
            for angle_deg in range(0, 360, 45):
                a = math.radians(angle_deg)
                cand = [
                    anchor[0] + math.cos(a) * radius,
                    0.0,
                    anchor[1] + math.sin(a) * radius,
                ]
                try:
                    res = self.client.query(
                        {
                            "type": "can_build_at",
                            "unitDefId": int(t.def_id),
                            "pos": cand,
                            "facing": t.facing,
                        }
                    )
                    self.debug_queries += 1
                    if res and res.get("possible"):
                        p = res.get("pos") or cand
                        if self.site_ok(t, float(p[0]), float(p[2])):
                            t.x = float(p[0])
                            t.z = float(p[2])
                            t.facing = int(res.get("facing") or 0)
                            t.placed = True
                            self.record_placement(t)
                            return True
                except QueryError:
                    continue
        t.placement_attempts += 1
        if t.placement_attempts >= 20:
            t.unplaceable = True
        return False

    def builder_fallback(self, u: UnitInfo) -> None:
        if u.name == "staticmissilesilo":
            return
        if u.uid == self.commander_id:
            # Never send the commander wandering after distant wreckage.
            if self.dist(u.pos, self.base_pos) > 800.0:
                self.queue(
                    {
                        "type": "move",
                        "unitId": int(u.uid),
                        "pos": [self.base_pos[0], 0.0, self.base_pos[1]],
                    }
                )
                return
            factories = self.factory_list()
            if factories:
                f = min(factories, key=lambda x: self.dist(u.pos, x.pos))
                if self.dist(u.pos, f.pos) < 800.0:
                    self.queue({"type": "guard", "unitId": int(u.uid), "targetId": int(f.uid)})
            return
        # Constructors without a job reclaim nearby wreckage first; debris is a
        # real income stream in Zero-K and CircuitAI reclaims aggressively.
        last = self.builder_fallback_frame.get(u.uid, -10**9)
        if self.frame - last < 90:
            return
        best_fid = None
        best_metal = 0.0
        best_d = 1e18
        for fid, feat in self._feature_cache.items():
            try:
                pos = feat.get("pos") or [0, 0, 0]
                definition = feat.get("definition") or {}
                contained = definition.get("containedResources") or {}
                metal = float(contained.get("Metal") or contained.get("metal") or 0.0)
                if metal < 10.0:
                    continue
                d = self.dist(u.pos, (float(pos[0]), float(pos[2])))
                if d > 1600.0:
                    continue
                if self.threat_in_radius((float(pos[0]), float(pos[2])), 200.0) > 0.0:
                    continue
                if metal > best_metal * 1.3 or d < best_d:
                    if metal >= best_metal * 0.7:
                        best_fid = int(fid)
                        best_metal = metal
                        best_d = d
            except Exception:
                continue
        if best_fid is not None:
            self.builder_fallback_frame[u.uid] = self.frame
            self.builder_fallback_until[u.uid] = self.frame + 360
            self.queue({"type": "reclaim", "unitId": int(u.uid), "featureId": best_fid})
            self.log(
                f"reclaim {best_metal:.0f}M with {u.name}#{u.uid} d={best_d:.0f}"
            )
            return
        # Otherwise assist a factory.
        factories = self.factory_list()
        if not factories:
            return
        f = min(factories, key=lambda x: self.dist(u.pos, x.pos))
        if self.dist(u.pos, f.pos) > 2500:
            return
        self.builder_fallback_frame[u.uid] = self.frame
        self.builder_fallback_until[u.uid] = self.frame + 120
        self.queue({"type": "guard", "unitId": int(u.uid), "targetId": int(f.uid)})

    # -------------------------------------------------------------- production

    def ensure_caretakers_guarding(self) -> None:
        factories = self.factory_list()
        if not factories:
            return
        if self.frame - self.last_economy_frame < 15:
            return
        for u in self.units.values():
            if u.name != "staticcon" or not u.is_finished:
                continue
            if u.uid in self.caretaker_guarded:
                continue
            if self.frame < getattr(self, "caretaker_repair_until", {}).get(u.uid, -1):
                continue
            f = min(factories, key=lambda x: self.dist(u.pos, x.pos))
            if self.dist(u.pos, f.pos) < 700:
                self.queue({"type": "guard", "unitId": int(u.uid), "targetId": int(f.uid)})
                self.caretaker_guarded.add(u.uid)

    def repair_step(self) -> None:
        """Caretakers repair damaged combat units and turrets in range.

        Retreating units otherwise only heal through slow idle regeneration
        and come back to the fight at partial health.  A caretaker on a
        repair job is un-guarded; ensure_caretakers_guarding re-guards it
        once the repair window has passed.
        """
        until = getattr(self, "caretaker_repair_until", {})
        self.caretaker_repair_until = until
        damaged = [
            u
            for u in self.units.values()
            if u.is_finished
            and u.alive_frac < 0.75
            and (u.is_combat or units.is_defense(u.data))
        ]
        if not damaged:
            return
        taken: set[int] = set()
        caretakers = [c for c in self.units.values() if c.name == "staticcon" and c.is_finished]
        # Production comes first: at most half the caretakers repair.
        budget = len(caretakers) // 2
        busy = sum(1 for c in caretakers if self.frame < until.get(c.uid, -1))
        for c in caretakers:
            if busy >= budget:
                break
            if self.frame < until.get(c.uid, -1):
                continue
            best = None
            best_score = 1e18
            for d in damaged:
                if d.uid in taken:
                    continue
                dist = self.dist(c.pos, d.pos)
                if dist > 480.0:
                    continue
                score = d.alive_frac * 1000.0 + dist * 0.2
                if score < best_score:
                    best_score = score
                    best = d
            if best is None:
                continue
            taken.add(best.uid)
            busy += 1
            self.queue({"type": "repair", "unitId": int(c.uid), "targetId": int(best.uid)})
            until[c.uid] = self.frame + 120
            self.caretaker_guarded.discard(c.uid)
