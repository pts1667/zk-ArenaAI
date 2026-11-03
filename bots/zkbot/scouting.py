"""Cloakbot artillery spotters.

The Gremlin (``cloakaa``) is nominally an anti-air unit, but its 760-elmo LOS
and cloak make it a useful forward observer for Sling artillery.  These
Gremlins are deliberately *not* added to the normal army: they move alone,
keep their fire state on hold-fire, and stay outside decloak range of enemy
ground units.  A radar tower supplies blips; the Gremlin supplies true LOS
for the guns.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo

from .params import FLAGS


class ScoutingMixin:
    # Being this far from an enemy keeps cloak from being broken by proximity.
    # The exact decloak radius is not exposed in the extracted unit data; 420
    # is a deliberately conservative value for the Gremlin's small chassis.
    GREMLIN_DECLOAK_SAFE = 420.0

    def artillery_units(self) -> list[UnitInfo]:
        out = []
        for uid in self.army:
            u = self.units.get(uid)
            if u is not None and units.role_of(u.data) in ("arty", "heavyarty"):
                out.append(u)
        # A Gremlin is excluded from self.army, but never treat it as
        # artillery itself.
        return out

    def should_build_gremlin_scout(self) -> bool:
        if (self.factory_name or "").lower() != "factorycloak":
            return False
        spot = self.phantom_spotters_wanted() if hasattr(self, "phantom_spotters_wanted") else 0
        if not self.artillery_units() and spot == 0:
            return False
        if self.frame_time() < 180.0 or self.metal_income() < 8.0:
            return False
        # Spotters that keep dying are wasted metal: cap the total.
        built = getattr(self, "gremlin_built", 0)
        if built >= 2 + int(self.frame_time() / 600) + 2 * spot:
            return False
        target = 0
        if self.artillery_units():
            target = 1 if len(self.artillery_units()) < 4 else 2
        target = max(target, spot)
        active = len(getattr(self, "gremlin_scouts", set())) + int(
            getattr(self, "gremlin_pending", 0)
        )
        return active < target

    def note_new_gremlin(self, u: UnitInfo) -> None:
        self.gremlin_pending = max(0, int(getattr(self, "gremlin_pending", 0)) - 1)
        self.gremlin_built = getattr(self, "gremlin_built", 0) + 1
        scouts = getattr(self, "gremlin_scouts", set())
        scouts.add(u.uid)
        self.gremlin_scouts = scouts
        self.gremlin_fire_state = getattr(self, "gremlin_fire_state", set())
        self.log(f"gremlin scout complete id={u.uid}; hold fire, forward LOS")

    def _gremlin_target(self) -> tuple[float, float]:
        # Prefer the enemy economy/front sector, but do not send a lone
        # observer deep into a fortified base before the army can support it.
        sector = self.best_attack_sector()
        if sector is not None:
            p = self.best_defense_pos(sector) or (sector.x, sector.z)
            if self.dist(p, self.base_pos) < 3600.0 or len(self.army) >= 12:
                if self.enemy_near(p, 650.0) is None:
                    return p
        if self.enemy_base:
            bx, bz = self.base_pos
            ex, ez = self.enemy_base
            p = (bx + (ex - bx) * 0.62, bz + (ez - bz) * 0.62)
            if self.enemy_near(p, 600.0) is None:
                return p
        return self.rally_point()

    def gremlin_step(self) -> None:
        scouts = getattr(self, "gremlin_scouts", set())
        if not scouts:
            return
        alive = {uid for uid in scouts if uid in self.units}
        if alive != scouts:
            self.gremlin_scouts = alive
            scouts = alive
        if not scouts:
            return
        fire_state = getattr(self, "gremlin_fire_state", set())
        self.gremlin_fire_state = fire_state
        last_orders = getattr(self, "gremlin_last_order", {})
        self.gremlin_last_order = last_orders
        danger_logs = getattr(self, "gremlin_last_danger_log", {})
        self.gremlin_last_danger_log = danger_logs
        target = self._gremlin_target()
        n_spot = self.phantom_spotters_wanted() if hasattr(self, "phantom_spotters_wanted") else 0
        spotters = set(sorted(scouts)[:n_spot])
        for uid in scouts:
            u = self.units.get(uid)
            if u is None or not u.is_finished:
                continue
            if uid in spotters and "nophantom" not in FLAGS:
                if uid not in fire_state:
                    self.queue({"type": "custom", "unitId": int(uid), "cmdId": 45, "params": [0.0]})
                    fire_state.add(uid)
                # Phantom spotter: close in front (Gremlin decloak radius is
                # 125, sight 760), always moving between ring points.
                danger = self.enemy_near(u.pos, 230.0)
                if danger is not None:
                    dx, dz = u.x - danger.x, u.z - danger.z
                    n = math.hypot(dx, dz) or 1.0
                    self.order_move(uid, (u.x + dx / n * 260.0, u.z + dz / n * 260.0))
                    continue
                p = self.spotter_point(uid)
                if p is not None:
                    self.order_move(uid, p)
                    continue
            # Hold fire permanently.  A Gremlin that shoots a scout or a
            # passing raider immediately loses its cloak and its value as a
            # forward LOS provider.
            if uid not in fire_state:
                self.queue(
                    {
                        "type": "custom",
                        "unitId": int(uid),
                        "cmdId": 45,  # CMD_FIRE_STATE
                        "params": [0.0],  # FIRESTATE_HOLDFIRE
                    }
                )
                fire_state.add(uid)
            danger = self.enemy_near(u.pos, self.GREMLIN_DECLOAK_SAFE + 120.0)
            if danger is not None:
                dest = self.safe_retreat_point(u)
                if self.frame - last_orders.get(uid, -10**9) > 90:
                    self.order_move(uid, dest)
                    last_orders[uid] = self.frame
                    if self.frame - danger_logs.get(uid, -10**9) > 300:
                        danger_logs[uid] = self.frame
                        self.log(
                            f"gremlin {uid} withdrawing; {danger.name} "
                            f"at {self.dist(u.pos, danger.pos):.0f}"
                        )
                continue
            last = self.gremlin_last_order.get(uid, -10**9)
            if self.frame - last > 300 or self.dist(u.pos, target) > 350.0:
                self.order_move(uid, target)
                last_orders[uid] = self.frame
                if last < 0:
                    self.log(
                        f"gremlin {uid} scouting {tuple(round(v) for v in target)} "
                        "with fire-at-will off"
                    )
