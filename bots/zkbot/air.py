"""Air plant production and control: one dedicated scout plus fire support.

The second factory is an air plant.  Its first unit is a scout.  From the
airplane plant that is a Swift: cheap, and its 1-second speed boost lets it
dash through anti-air cover to see the enemy base; losing one now and then is
fine, losing many is not.  From the gunship plant it is a Locust, which tours
enemy territory while avoiding anti-air.  Either way the threat map, structure
memory and enemy-power estimate get real sightings.  An Owl (planescout) is
not a scout: it is a mid/late-game radar and sight platform for artillery
and is built only once we field artillery.  The rest
of the production is fire support that hovers with the main ground group and
shoots what it fights; it falls back over the group when exposed.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo

AIR_FACTORIES = ("factorygunship", "factoryplane")
SCOUT_UNITS = {"factorygunship": "gunshipraid", "factoryplane": "planefighter"}
CMD_ONECLICK_WEAPON = 35000  # Swift "Speed Boost": 5x speed for 1 s
BOOST_COOLDOWN = 870  # frames (gadget reload 850)
MAX_SCOUT_LOSSES = 5


class AirMixin:
    # ------------------------------------------------------------ production

    def next_air_unit(self, factory: UnitInfo) -> int | None:
        fname = factory.name
        scout_name = SCOUT_UNITS.get(fname)
        scout_alive = self.air_scout_id in self.units if getattr(self, "air_scout_id", None) else False
        losses = getattr(self, "air_scout_losses", 0)
        # Swifts are cheap but not free: wait between replacements, longer
        # once several have died.
        since = self.frame - getattr(self, "air_scout_lost_frame", -10**9)
        if losses >= MAX_SCOUT_LOSSES:
            respawn_ok = since > 30 * 240
        elif losses >= 2:
            respawn_ok = since > 30 * 120
        else:
            respawn_ok = since > 30 * 45
        if (
            scout_name
            and not scout_alive
            and not getattr(self, "air_scout_pending", False)
            and respawn_ok
        ):
            self.air_scout_pending = True
            self.log(f"queue air scout {scout_name} (losses {losses})")
            return units.id_of(scout_name)
        if fname == "factoryplane" and self.want_owl():
            self.owl_pending = True
            self.owl_queued_frame = self.frame
            self.owl_count = getattr(self, "owl_count", 0) + 1
            self.log("queue owl for artillery radar/LOS")
            return units.id_of("planescout")
        if not self.combat_air_allowed():
            return None
        n = self.air_counter
        self.air_counter += 1
        minc = self.metal_income()
        if fname == "factorygunship":
            if minc >= 55:
                names = ["gunshipheavyskirm", "gunshipskirm", "gunshipheavyskirm", "gunshipassault"]
            else:
                names = ["gunshipskirm", "gunshipskirm", "gunshipheavyskirm"]
        else:
            comp = self.enemy_composition()
            if comp.get("air", 0) >= 2:
                names = ["planefighter", "planefighter", "planeheavyfighter"]
            else:
                names = ["bomberprec", "bomberprec", "planefighter", "bomberriot"]
        return units.id_of(names[n % len(names)])

    def combat_air_allowed(self) -> bool:
        """Air is support: keep it a bounded share of the army, and stop
        feeding it when (often cloaked) anti-air is shooting it down."""
        if self.frame < getattr(self, "air_pause_until", -1):
            return False
        air_value = 0.0
        ground_value = 0.0
        for uid in self.army:
            u = self.units.get(uid)
            if u is None or uid == getattr(self, "air_scout_id", None):
                continue
            if u.is_air:
                air_value += units.cost(u.data)
            else:
                ground_value += units.cost(u.data)
        if air_value > 300.0 and air_value > 0.35 * ground_value:
            return False
        spent = getattr(self, "spend_by_name", {})
        lost = getattr(self, "lost_by_name", {})
        names = [n for n in spent if (units.by_name(n) or {}).get("isAir")]
        base_s, base_l = getattr(self, "air_ledger_base", (0.0, 0.0))
        air_spent = sum(spent.get(n, 0.0) for n in names) - base_s
        air_lost = sum(lost.get(n, 0.0) for n in names) - base_l
        if air_spent >= 1200.0 and air_lost >= 0.6 * air_spent:
            self.air_pause_until = self.frame + 30 * 240
            self.air_ledger_base = (air_spent + base_s, air_lost + base_l)
            self.log(f"combat air paused 4 min: lost {air_lost:.0f} of {air_spent:.0f}")
            return False
        return True

    def want_owl(self) -> bool:
        if getattr(self, "owl_pending", False):
            return False
        if any(u.name == "planescout" for u in self.units.values()):
            return False
        if getattr(self, "owl_count", 0) >= 3:
            return False
        # At most one replacement Owl per 4 minutes.
        if self.frame - getattr(self, "owl_queued_frame", -10**9) < 30 * 240:
            return False
        arty = len(self.artillery_units()) if hasattr(self, "artillery_units") else 0
        return self.frame_time() >= 900 and arty >= 4

    def note_new_air_unit(self, u: UnitInfo) -> None:
        if u.name == "planescout":
            self.owl_pending = False
        if getattr(self, "air_scout_pending", False) and u.name in SCOUT_UNITS.values():
            self.air_scout_pending = False
            self.air_scout_id = u.uid
            self.log(f"air scout {u.name} id={u.uid}")

    # --------------------------------------------------------------- control

    def control_air(self, u: UnitInfo, group_center, group_target) -> None:
        if u.uid == getattr(self, "air_scout_id", None):
            if u.name == "planefighter":
                self.control_swift_scout(u)
            else:
                self.control_air_scout(u)
            return
        if u.name == "planescout":
            self.control_owl(u, group_center, group_target)
            return
        role = units.role_of(u.data)
        if u.name.startswith("bomber"):
            self.control_bomber(u, group_center)
            return
        anchor = group_target or group_center or self.rally_point()
        exposed = self.air_threat_at(u.pos)
        own = self.own_power_at(u.pos)
        if exposed > 0 and exposed > own * 1.2 and group_center is not None:
            # Under fire with no ground support: back over the army.
            self.order_move(u.uid, group_center)
            return
        # Hover just behind the group's front so ground units tank.
        if group_center is not None and group_target is not None:
            dx, dz = group_target[0] - group_center[0], group_target[1] - group_center[1]
            d = math.hypot(dx, dz)
            if d > 200:
                anchor = (group_center[0] + dx / d * 150.0, group_center[1] + dz / d * 150.0)
            else:
                anchor = group_target
        self.order_fight(u.uid, anchor)

    def control_bomber(self, u: UnitInfo, group_center) -> None:
        """Strike the most valuable visible target near our army that is
        not under heavy anti-air; otherwise wait over the base/pads."""
        ref = group_center or self.rally_point()
        best = None
        best_score = -1e18
        for e in self.enemies.values():
            if e.is_air or not e.is_finished:
                continue
            d = self.dist(e.pos, ref)
            if d > 1600.0:
                continue
            aa = self.air_threat_in_radius(e.pos, 250.0)
            value = units.cost(e.data)
            if units.is_defense(e.data):
                value *= 1.3
            if e.name == "staticmex" or e.is_builder:
                value += 150.0
            score = value - aa * 6.0 - d * 0.1
            if score > best_score:
                best_score = score
                best = e
        if best is not None and best_score > 0:
            self.order_attack(u.uid, best.uid)
            return
        pads = [p for p in self.units.values() if p.name == "staticrearm" and p.is_finished]
        home = pads[0].pos if pads else self.base_pos
        self.order_move(u.uid, home)

    def control_owl(self, u: UnitInfo, group_center, group_target) -> None:
        """Hold behind the front line over our artillery, outside AA reach."""
        arty = self.artillery_units() if hasattr(self, "artillery_units") else []
        if arty:
            cx = sum(a.x for a in arty) / len(arty)
            cz = sum(a.z for a in arty) / len(arty)
            ref = (cx, cz)
        else:
            ref = group_center or self.rally_point()
        front = group_target or self.enemy_base or self.map.center
        dx, dz = front[0] - ref[0], front[1] - ref[1]
        d = math.hypot(dx, dz) or 1.0
        p = (ref[0] + dx / d * 250.0, ref[1] + dz / d * 250.0)
        if self.air_threat_in_radius(p, 250.0) > 0.0:
            p = ref
        if self.air_threat_at(u.pos) > 0.0:
            p = self.safest_air_point(u.pos)
        self.order_move(u.uid, p)

    def control_swift_scout(self, u: UnitInfo) -> None:
        """Dash runs into the enemy base, boosting through anti-air."""
        state = getattr(self, "swift_state", None)
        if state is None or state.get("uid") != u.uid:
            state = {"uid": u.uid, "phase": "wait", "next_run": self.frame, "boost": -10**9, "target": None}
            self.swift_state = state
        home = self.base_pos
        if state["phase"] == "wait":
            if self.frame >= state["next_run"]:
                target = self.enemy_base or self.map.center
                # Prefer the enemy structure we know least recently about.
                mem = getattr(self, "enemy_structures", {})
                if mem and (self.frame // 30) % 2 == 0:
                    e, _f = min(mem.values(), key=lambda v: v[1])
                    target = e.pos
                state.update(phase="run", target=target)
                self.log(f"swift {u.uid} scouting run -> {tuple(round(v) for v in target)}")
            else:
                self.order_move(u.uid, home)
                return
        target = state["target"]
        if state["phase"] == "run":
            if self.dist(u.pos, target) < 300.0:
                state["phase"] = "return"
            else:
                # Boost as soon as anti-air is just ahead (or already on us).
                dx, dz = target[0] - u.x, target[1] - u.z
                d = math.hypot(dx, dz) or 1.0
                ahead = (u.x + dx / d * 450.0, u.z + dz / d * 450.0)
                aa = max(self.air_threat_at(u.pos), self.air_threat_at(ahead))
                # Unknown anti-air shows up as damage: boost out of it.
                if u.health < u.max_health * 0.85:
                    aa = max(aa, 1.0)
                if aa > 0.0 and self.frame - state["boost"] >= BOOST_COOLDOWN:
                    state["boost"] = self.frame
                    self.queue({"type": "custom", "unitId": int(u.uid), "cmdId": CMD_ONECLICK_WEAPON, "params": [1.0]})
                    self.queue({"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]], "options": 32})
                    self.unit_order[u.uid] = ("move", target[0], target[1], None, self.frame)
                    return
                self.order_move(u.uid, target)
                return
        if state["phase"] == "return":
            if self.dist(u.pos, home) < 500.0:
                state.update(phase="wait", next_run=self.frame + 30 * 90)
            aa = self.air_threat_at(u.pos)
            if u.health < u.max_health * 0.85:
                aa = max(aa, 1.0)
            if aa > 0.0 and self.frame - state["boost"] >= BOOST_COOLDOWN:
                state["boost"] = self.frame
                self.queue({"type": "custom", "unitId": int(u.uid), "cmdId": CMD_ONECLICK_WEAPON, "params": [1.0]})
                self.queue({"type": "move", "unitId": int(u.uid), "pos": [home[0], 0.0, home[1]], "options": 32})
                self.unit_order[u.uid] = ("move", home[0], home[1], None, self.frame)
                return
            self.order_move(u.uid, home)

    def note_air_scout_lost(self) -> None:
        self.air_scout_losses = getattr(self, "air_scout_losses", 0) + 1
        self.air_scout_lost_frame = self.frame
        self.air_scout_id = None
        self.log(f"air scout lost ({self.air_scout_losses})")

    def control_air_scout(self, u: UnitInfo) -> None:
        seen = getattr(self, "scout_seen_frame", None)
        if seen is None:
            seen = {}
            self.scout_seen_frame = seen
        los = float(u.data.get("losRadius") or 500.0)
        for s in self.map.spots:
            if self.dist((s.x, s.z), u.pos) < los * 0.8:
                seen[s.index] = self.frame
        if self.air_threat_at(u.pos) > 0.0:
            p = self.safest_air_point(u.pos)
            self.order_move(u.uid, p)
            return
        # Next waypoint: the spot not scouted for longest that is safe to
        # reach, weighted toward the enemy half of the map.
        ex, ez = self.enemy_base or self.map.center
        best = None
        best_score = -1e18
        for s in self.map.spots:
            p = (s.x, s.z)
            if self.air_threat_in_radius(p, 200.0) > 0.0:
                continue
            age = self.frame - seen.get(s.index, -10**6)
            score = min(age, 9000) * 1.0
            score -= self.dist(u.pos, p) * 0.8
            score -= self.dist(p, (ex, ez)) * 0.3
            if self.air_path_threat(u.pos, p) > 0.0:
                score -= 4000.0
            if score > best_score:
                best_score = score
                best = p
        target = best or self.rally_point()
        last = getattr(self, "_scout_target", None)
        if last != target:
            self._scout_target = target
        self.order_move(u.uid, target)

    # ------------------------------------------------------------ air threat

    def air_threat_at(self, pos) -> float:
        grid = getattr(self, "air_grid", None)
        c = self._cell(pos)
        if grid is None or c is None:
            return 0.0
        return float(grid[c])

    def air_threat_in_radius(self, pos, radius: float) -> float:
        grid = getattr(self, "air_grid", None)
        if grid is None:
            return 0.0
        from .threat import CELL

        h, w = grid.shape
        cx, cz = pos[0] / CELL, pos[1] / CELL
        r = radius / CELL
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
        z0, z1 = max(0, int(cz - r)), min(h, int(cz + r) + 1)
        if x1 <= x0 or z1 <= z0:
            return 0.0
        return float(grid[z0:z1, x0:x1].max())

    def air_path_threat(self, a, b) -> float:
        from .threat import CELL

        n = max(2, int(self.dist(a, b) / CELL) + 1)
        best = 0.0
        for i in range(n + 1):
            f = i / n
            best = max(best, self.air_threat_at((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)))
        return best

    def safest_air_point(self, pos):
        best = self.base_pos
        best_score = 1e18
        for ring in (400.0, 800.0):
            for k in range(12):
                a = k * math.pi / 6.0
                p = (pos[0] + math.cos(a) * ring, pos[1] + math.sin(a) * ring)
                if not (40 < p[0] < self.map.width - 40 and 40 < p[1] < self.map.height - 40):
                    continue
                score = self.air_threat_at(p) * 50.0 + self.dist(p, self.base_pos) * 0.1
                if score < best_score:
                    best_score = score
                    best = p
        return best

    def note_own_death(self, u: UnitInfo, event: dict) -> None:
        if u.uid == getattr(self, "air_scout_id", None):
            self.note_air_scout_lost()
