"""PhantomMixin: Phantom (cloaksnipe) play with Gremlin spotters.

User guidance: Phantoms are cloaked, fragile anti-heavy snipers that stay
cloaked while firing (~1500 damage per shot; two together kill an
unupgraded commander).  They need line of sight, not just radar, to hit
mobile units, so Gremlins (cloaked, fast, 760 sight, decloak radius 125)
spot for them from in front, outside decloak range, and keep moving.
Fire-at-radar is turned off for Phantoms (CMD_DONT_FIRE_AT_RADAR 38372,
param 1 = don't fire at radar; the unit default is to fire at radar).
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo

from .params import FLAGS, P

CMD_DONT_FIRE_AT_RADAR = 38372
PHANTOM = "cloaksnipe"
PHANTOM_DECLOAK = 155.0
PHANTOM_RANGE = 700.0
GREMLIN_DECLOAK = 125.0
SPOTTER_STANDOFF = P("phantom.spotter_standoff", 330.0, 150.0, 600.0, "Gremlin spotter distance")
SPOTTER_EVADE = P("phantom.spotter_evade", 230.0, 100.0, 500.0, "Gremlin spotter evade distance")


class PhantomMixin:
    # ------------------------------------------------------------ production

    def phantom_ids(self) -> list[int]:
        return [
            uid
            for uid in self.army
            if uid in self.units and self.units[uid].name == PHANTOM and self.units[uid].is_finished
        ]

    def phantom_quota(self) -> int:
        """How many Phantoms the army should hold (cloak factory only)."""
        if "nophantom" in FLAGS:
            return 0
        if (self.factory_name or "").lower() != "factorycloak":
            return 0
        if self.frame_time() < 600.0 or self.metal_income() < 25.0:
            return 0
        seen = self.seen_enemy_roles()
        heavies = seen["riot"] + seen["assault"]
        ball = self.enemy_slow_ball() if hasattr(self, "enemy_slow_ball") else None
        if heavies < 3 and ball is None:
            return 0
        cap = 2 + int(self.metal_income() / 20.0)
        return max(2, min(cap, heavies // 3))

    def phantom_spotters_wanted(self) -> int:
        """ceil(phantoms / 2), at least one once any Phantom exists."""
        n = len(self.phantom_ids())
        return math.ceil(n / 2) if n > 0 else 0

    def note_new_phantom(self, u: UnitInfo) -> None:
        done = self.__dict__.setdefault("phantom_radar_off", set())
        if u.uid in done:
            return
        done.add(u.uid)
        self.queue(
            {
                "type": "custom",
                "unitId": int(u.uid),
                "cmdId": CMD_DONT_FIRE_AT_RADAR,
                "params": [1.0],
            }
        )
        self.log(f"phantom {u.uid} ready; fire-at-radar off")

    # ------------------------------------------------------------ targeting

    def _phantom_volley_target(self, phantoms: list[UnitInfo]) -> UnitInfo | None:
        """One target for every Phantom in range, so the shots land together.
        Only units in LOS: radar-only contacts are not shot at."""
        if not phantoms:
            return None
        cache = getattr(self, "_volley_cache", None)
        if cache is not None and self.frame - cache[0] < 15:
            return self.enemies.get(cache[1]) if cache[1] is not None else None
        best = None
        best_score = 0.0
        for e in self.enemies.values():
            if e.is_air or not e.is_finished:
                continue
            in_range = sum(
                1 for p in phantoms if self.dist(p.pos, e.pos) <= PHANTOM_RANGE + 60.0
            )
            near = sum(
                1 for p in phantoms if self.dist(p.pos, e.pos) <= PHANTOM_RANGE + 400.0
            )
            if near == 0:
                continue
            name = e.name.lower()
            is_comm = "dyntrainer" in name or "commander" in name
            role = units.role_of(e.data)
            speed = float(e.data.get("speed") or 0.0)
            if is_comm:
                # Two shots kill an unupgraded commander: only worth it with
                # two Phantoms able to fire.
                score = 5000.0 if near >= 2 else 600.0
            elif e.is_mobile and role in ("riot", "assault", "heavy", "antiheavy") and speed < 75.0:
                score = 1000.0 + units.cost(e.data)
            elif e.is_mobile and units.cost(e.data) >= 250.0:
                score = 300.0 + units.cost(e.data) * 0.5
            elif not e.is_mobile and units.is_defense(e.data):
                score = 150.0 + units.cost(e.data) * 0.3
            else:
                continue
            score += 400.0 * in_range
            if score > best_score:
                best, best_score = e, score
        self._volley_cache = (self.frame, best.uid if best is not None else None)
        return best

    # ------------------------------------------------------------ movement

    def phantom_order(self, u: UnitInfo, anchor: tuple[float, float]) -> str:
        """Drive one Phantom.  Returns the branch name for the counters."""
        # Too close to enemies means decloaking: step away first.
        threat = None
        td = 1e18
        for e in self.enemies.values():
            if e.is_air or not self.is_dangerous(e):
                continue
            d = self.dist(u.pos, e.pos)
            if d < td:
                threat, td = e, d
        if threat is not None and td < PHANTOM_DECLOAK + 180.0:
            dx, dz = u.x - threat.x, u.z - threat.z
            n = math.hypot(dx, dz) or 1.0
            step = PHANTOM_DECLOAK + 320.0 - td
            x = min(max(40.0, u.x + dx / n * step), self.map.width - 40.0)
            z = min(max(40.0, u.z + dz / n * step), self.map.height - 40.0)
            self.order_move(u.uid, (x, z))
            return "snipe_evade"
        target = self._phantom_volley_target(
            [self.units[p] for p in self.phantom_ids() if p in self.units]
        )
        if target is not None:
            d = self.dist(u.pos, target.pos)
            if d <= PHANTOM_RANGE:
                self.order_attack(u.uid, target.uid)
                return "snipe_fire"
            # Approach to just inside range, from our side.
            dx, dz = u.x - target.x, u.z - target.z
            n = math.hypot(dx, dz) or 1.0
            p = (target.x + dx / n * (PHANTOM_RANGE - 60.0), target.z + dz / n * (PHANTOM_RANGE - 60.0))
            if self.enemy_near(p, PHANTOM_DECLOAK + 150.0) is None:
                self.order_move(u.uid, p)
                return "snipe_approach"
        # Hold behind the army, at about firing distance from its front.
        self.order_move(u.uid, anchor)
        return "snipe_hold"

    def phantom_anchor(self, group_center: tuple[float, float]) -> tuple[float, float]:
        ref = self.enemy_base or self.map.center
        dx, dz = group_center[0] - ref[0], group_center[1] - ref[1]
        n = math.hypot(dx, dz) or 1.0
        return group_center[0] + dx / n * 250.0, group_center[1] + dz / n * 250.0

    # ------------------------------------------------------------ spotters

    def spotter_point(self, uid: int) -> tuple[float, float] | None:
        """A point on a ring around the enemy we want Phantoms to see,
        outside decloak range, changing every few seconds so the Gremlin
        never parks."""
        phantoms = [self.units[p] for p in self.phantom_ids() if p in self.units]
        if not phantoms:
            return None
        ball = self.enemy_slow_ball() if hasattr(self, "enemy_slow_ball") else None
        focus = None
        if ball is not None:
            focus = ball["pos"]
        else:
            pc = self.median_pos([p.uid for p in phantoms])
            best = None
            bd = 1e18
            for e in list(self.enemies.values()) + list(self.radar.values()):
                if e.is_air or not e.is_mobile:
                    continue
                d = self.dist(pc, e.pos) if pc else 1e18
                if d < bd:
                    best, bd = e, d
            if best is not None and bd < PHANTOM_RANGE + 900.0:
                focus = best.pos
        if focus is None:
            return None
        pc = self.median_pos([p.uid for p in phantoms]) or self.base_pos
        base_ang = math.atan2(pc[1] - focus[1], pc[0] - focus[0])
        # Sweep +-60 degrees on our side of the target, a new point every 4 s.
        slot = (self.frame // 120 + uid) % 5
        ang = base_ang + math.radians((-60, -30, 0, 30, 60)[slot])
        x = focus[0] + math.cos(ang) * SPOTTER_STANDOFF
        z = focus[1] + math.sin(ang) * SPOTTER_STANDOFF
        return (
            min(max(60.0, x), self.map.width - 60.0),
            min(max(60.0, z), self.map.height - 60.0),
        )
