"""Raider-vs-raider micro (RaiderMicroMixin), from the user's second review.

- A fleeing raider shoots its chasers, so our raiders never chase an enemy
  raider that is moving away and out of our range, and back off (move
  orders still fire at will) from raider groups that outrange them or that
  they cannot beat locally.
- On offence a squad paths around enemy raider groups (Glaives are faster
  than Kodachis) instead of running into them.
- On defence raiders intercept: they wait at the asset the enemy raiders are
  heading for (early-warning trajectory) and collapse on the group once it
  commits to the asset and we have the numbers.
- Pincer: with enough raider power, defending raiders split into two halves
  that stage on opposite flanks of an enemy raider group just outside its
  range, then close in together (close range favours Glaives).
- At least one raider squad stays on offence after two minutes, whatever
  the army mode.

Flags: `oldmicro` (all of it off), `nopincer`, `nooffense`.
"""
from __future__ import annotations

import math
import os
from typing import Any

from . import units
from .models import UnitInfo
from .threat import combat_power, weapon_range

FLAGS = os.environ.get("ZKBOT_FLAGS", "")

GROUP_RADIUS = 450.0
PINCER_ANGLE = math.radians(70.0)
PINCER_RATIO = 1.4
OFFENSE_SHARE = 0.35
OFFENSE_START = 120.0


def _unit(dx: float, dz: float) -> tuple[float, float]:
    n = math.hypot(dx, dz) or 1.0
    return dx / n, dz / n


class RaiderMicroMixin:
    # ------------------------------------------------------- enemy groups

    def enemy_raider_groups(self) -> list[dict[str, Any]]:
        """Clusters of enemy ground raiders seen in the last 5 s."""
        cache = getattr(self, "_rgroup_cache", None)
        if cache is not None and self.frame - cache[0] < 15:
            return cache[1]
        pool: dict[int, UnitInfo] = {}
        for uid, (e, f) in self.enemy_memory.items():
            if self.frame - f <= 150:
                pool[uid] = e
        for e in self.contacts().values():
            pool[e.uid] = e
        raiders = [
            e for e in pool.values()
            if e.data and not e.is_air and e.is_mobile and units.role_of(e.data) == "raid"
        ]
        groups: list[dict[str, Any]] = []
        left = {e.uid: e for e in raiders}
        while left:
            seed = next(iter(left.values()))
            members = [seed]
            del left[seed.uid]
            grew = True
            while grew:
                grew = False
                for uid, e in list(left.items()):
                    if any(self.dist(e.pos, m.pos) < GROUP_RADIUS for m in members):
                        members.append(e)
                        del left[uid]
                        grew = True
            n = len(members)
            vel = getattr(self, "enemy_vel", {})
            vx = sum(vel.get(m.uid, (0.0, 0.0))[0] for m in members) / n
            vz = sum(vel.get(m.uid, (0.0, 0.0))[1] for m in members) / n
            groups.append({
                "pos": (sum(m.x for m in members) / n, sum(m.z for m in members) / n),
                "vel": (vx, vz),
                "n": n,
                "uids": [m.uid for m in members],
                "power": sum(combat_power(m) for m in members),
                "range": max(weapon_range(m) for m in members),
                "speed": min(float(m.data.get("speed") or 0.0) for m in members),
            })
        self._rgroup_cache = (self.frame, groups)
        return groups

    def _group_near(self, pos: tuple[float, float], radius: float) -> dict[str, Any] | None:
        best, bd = None, radius
        for g in self.enemy_raider_groups():
            d = self.dist(g["pos"], pos)
            if d < bd:
                best, bd = g, d
        return best

    def _moving_away(self, g: dict[str, Any], pos: tuple[float, float]) -> bool:
        vx, vz = g["vel"]
        spd = math.hypot(vx, vz)
        if spd < 30.0:
            return False
        dx, dz = g["pos"][0] - pos[0], g["pos"][1] - pos[1]
        d = math.hypot(dx, dz) or 1.0
        return (vx * dx + vz * dz) / (spd * d) > 0.5

    def _our_raider_power(self, pos: tuple[float, float], radius: float) -> float:
        total = 0.0
        for uid in self.raiders:
            u = self.units.get(uid)
            if u is not None and self.dist(u.pos, pos) < radius and uid not in self.retreating:
                total += combat_power(u)
        return total

    def _turret_power(self, pos: tuple[float, float], radius: float) -> float:
        return sum(
            combat_power(t)
            for t in self.units.values()
            if units.is_defense(t.data) and not t.is_mobile and t.is_finished
            and self.dist(t.pos, pos) < radius
        )

    def _clamp(self, p: tuple[float, float]) -> tuple[float, float]:
        w = float(getattr(self.map, "width", 0) or 0) or 1e9
        h = float(getattr(self.map, "height", 0) or 0) or 1e9
        return (min(w - 64.0, max(64.0, p[0])), min(h - 64.0, max(64.0, p[1])))

    def _away_point(self, u: UnitInfo, g: dict[str, Any], toward: tuple[float, float] | None) -> tuple[float, float]:
        ax, az = _unit(u.x - g["pos"][0], u.z - g["pos"][1])
        if toward is not None:
            tx, tz = _unit(toward[0] - u.x, toward[1] - u.z)
            # Sideways toward the destination, but never back into the group.
            if ax * tx + az * tz > -0.2:
                ax, az = _unit(ax * 0.6 + tx * 0.4, az * 0.6 + tz * 0.4)
        return self._clamp((u.x + ax * 320.0, u.z + az * 320.0))

    def detour_waypoint(self, start: tuple[float, float], dest: tuple[float, float]) -> tuple[float, float] | None:
        """A waypoint around the first enemy raider group near the path."""
        sx, sz = start
        dx, dz = dest[0] - sx, dest[1] - sz
        L2 = dx * dx + dz * dz
        if L2 < 1.0:
            return None
        best = None
        for g in self.enemy_raider_groups():
            gx, gz = g["pos"]
            t = ((gx - sx) * dx + (gz - sz) * dz) / L2
            if t <= 0.05 or t >= 1.0:
                continue
            px, pz = sx + dx * t, sz + dz * t
            off = math.hypot(gx - px, gz - pz)
            clear = g["range"] + 350.0
            if off >= clear:
                continue
            if best is None or t < best[0]:
                best = (t, g, (px, pz), off)
        if best is None:
            return None
        _t, g, (px, pz), off = best
        gx, gz = g["pos"]
        if off > 30.0:
            nx, nz = _unit(px - gx, pz - gz)
            sides = [(nx, nz)]
        else:
            nx, nz = _unit(-dz, dx)
            sides = [(nx, nz), (-nx, -nz)]
        cands = [self._clamp((gx + sx_ * (g["range"] + 520.0), gz + sz_ * (g["range"] + 520.0))) for sx_, sz_ in sides]
        return min(cands, key=lambda p: self.threat_at(p))

    # ------------------------------------------------------------ offence

    def offensive_raiders(self, raider_uids: list[int], threat: tuple[float, float] | None) -> list[int]:
        """Raiders that keep raiding whatever the army mode (after 2 min)."""
        if "oldmicro" in FLAGS or "nooffense" in FLAGS or self.frame_time() < OFFENSE_START:
            return []
        n = len(raider_uids)
        need = 2 if self.frame_time() < 300 else 3
        if n < need + 1:
            return []
        want = max(need, int(math.ceil(OFFENSE_SHARE * n)))
        keep = [uid for uid in getattr(self, "offense_raiders", []) if uid in raider_uids]
        if len(keep) < want:
            ref = threat or self.base_pos
            rest = sorted(
                (uid for uid in raider_uids if uid not in keep and uid not in self.retreating),
                key=lambda uid: -self.dist(self.units[uid].pos, ref),
            )
            keep += rest[: want - len(keep)]
        self.offense_raiders = keep[:want]
        return self.offense_raiders

    def raider_offense_order(self, u: UnitInfo, dest: tuple[float, float]) -> str | None:
        """Raider on its way to a raid: dodge or path around enemy raiders.
        None = carry on with the normal order."""
        if "oldmicro" in FLAGS:
            return None
        g = self._group_near(u.pos, 1100.0)
        if g is None:
            return None
        d = self.dist(u.pos, g["pos"])
        ours = self._our_raider_power(u.pos, 550.0) + self._turret_power(u.pos, 500.0)
        rng = weapon_range(u)
        stronger = ours >= g["power"] * 1.3 and g["range"] <= rng + 40.0
        if stronger and not self._moving_away(g, u.pos):
            return None
        if d < g["range"] + 200.0:
            if stronger:
                # Their raiders are leaving: shoot what is in range, no chase.
                self.order_move(u.uid, dest)
                return "raider_nochase"
            self.order_move(u.uid, self._away_point(u, g, dest))
            return "raider_evade"
        wp = self.detour_waypoint(u.pos, dest)
        if wp is not None and self.dist(u.pos, wp) > 200.0:
            self.order_move(u.uid, wp)
            return "raider_detour"
        return None

    # ------------------------------------------------------------ defence

    def raider_defense_order(self, u: UnitInfo, anchor: tuple[float, float] | None) -> str | None:
        """Raider defending: pincer, intercept at the threatened asset, or
        evade; None when no enemy raider group is involved."""
        if "oldmicro" in FLAGS:
            return None
        pin = getattr(self, "pincer_orders", {}).get(u.uid)
        if pin is not None:
            kind, pos = pin
            if kind == "stage":
                g = self._group_near(u.pos, 900.0)
                if g is not None and self.dist(u.pos, g["pos"]) < g["range"] + 120.0:
                    self.order_move(u.uid, self._away_point(u, g, pos))
                else:
                    self.order_move(u.uid, pos)
                return "pincer_stage"
            self.order_fight(u.uid, pos)
            return "pincer_strike"
        ref = anchor or u.pos
        g = self._group_near(ref, 1600.0) or self._group_near(u.pos, 900.0)
        if g is None:
            return None
        asset = None
        for w in getattr(self, "raid_warnings", []) or []:
            if any(self.dist(s, g["pos"]) < 500.0 for s in w.get("src", [])):
                asset = w["pos"]
                break
        if asset is None and anchor is None:
            # Our mex / constructor nearest the group is its likely target.
            best_d = 1500.0
            for o in self.units.values():
                if (o.name == "staticmex" and o.is_finished) or (
                    o.is_builder and o.is_mobile and o.uid != self.commander_id
                ):
                    dd = self.dist(o.pos, g["pos"])
                    if dd < best_d:
                        asset, best_d = o.pos, dd
        if asset is None:
            asset = anchor or self.base_pos
        bx, bz = _unit(self.base_pos[0] - asset[0], self.base_pos[1] - asset[1])
        point = (asset[0] + bx * 120.0, asset[1] + bz * 120.0)
        ours_local = self._our_raider_power(point, 550.0) + self._turret_power(point, 600.0)
        # Our own army near the asset counts too.
        o_army, _t = self.local_balance(point, 550.0)
        ours_local = max(ours_local, o_army)
        d = self.dist(u.pos, g["pos"])
        committed = self.dist(g["pos"], asset) < 380.0
        if committed and ours_local >= 0.9 * g["power"]:
            self.order_fight(u.uid, g["pos"])
            return "raider_collapse"
        if d < g["range"] + 220.0 and ours_local < 0.6 * g["power"]:
            self.order_move(u.uid, self._away_point(u, g, self.base_pos))
            return "raider_evade"
        rng = weapon_range(u)
        if d < rng + 20.0 and not self._moving_away(g, u.pos):
            self.order_fight(u.uid, g["pos"])
            return "raider_shoot"
        if self.dist(u.pos, point) > 120.0:
            wp = self.detour_waypoint(u.pos, point)
            self.order_move(u.uid, wp if wp is not None else point)
            return "raider_intercept"
        self.order_move(u.uid, point)
        return "raider_hold"

    # ------------------------------------------------------------- pincer

    def update_pincer(self, raider_uids: list[int]) -> dict[int, tuple[str, tuple[float, float]]]:
        """Split defending raiders around an enemy raider group they
        outnumber; strike together once both halves are in place."""
        out: dict[int, tuple[str, tuple[float, float]]] = {}
        self.pincer_orders = out
        if "oldmicro" in FLAGS or "nopincer" in FLAGS:
            return out
        avail = [
            uid for uid in raider_uids
            if uid in self.units and uid not in self.retreating
        ]
        st = getattr(self, "pincer", None)
        groups = self.enemy_raider_groups()
        if not avail or not groups:
            self.pincer = None
            return out
        # Keep the current target if it still exists.
        g = None
        if st is not None:
            g = min(groups, key=lambda x: self.dist(x["pos"], st["pos"]))
            if self.dist(g["pos"], st["pos"]) > 700.0:
                g = None
        if g is None:
            st = None
            assets = [
                o.pos for o in self.units.values()
                if (o.name == "staticmex" and o.is_finished)
                or (o.is_builder and o.is_mobile)
            ]
            cands = [
                x for x in groups
                if x["n"] >= 2 and (
                    self.dist(x["pos"], self.base_pos) < 1800.0
                    or any(self.dist(x["pos"], a) < 900.0 for a in assets)
                )
            ]
            if not cands:
                self.pincer = None
                return out
            g = min(cands, key=lambda x: self.dist(x["pos"], self.base_pos))
        near = [uid for uid in avail if self.dist(self.units[uid].pos, g["pos"]) < 2200.0]
        power = sum(combat_power(self.units[uid]) for uid in near)
        need = PINCER_RATIO * g["power"] if st is None else 0.9 * g["power"]
        if len(near) < (4 if st is None else 3) or power < need:
            if st is not None:
                self.log(f"pincer abort: {len(near)} raiders {power:.0f} vs {g['power']:.0f}")
            self.pincer = None
            return out
        gx, gz = g["pos"]
        cx = sum(self.units[uid].x for uid in near) / len(near)
        cz = sum(self.units[uid].z for uid in near) / len(near)
        bx, bz = _unit(cx - gx, cz - gz)
        r = g["range"] + 260.0
        pts = []
        for sgn in (1.0, -1.0):
            a = PINCER_ANGLE * sgn
            px = bx * math.cos(a) - bz * math.sin(a)
            pz = bx * math.sin(a) + bz * math.cos(a)
            pts.append(self._clamp((gx + px * r, gz + pz * r)))
        if st is None:
            # Split by side: left / right of the line from the group to us.
            def side(uid: int) -> float:
                u = self.units[uid]
                return bx * (u.z - gz) - bz * (u.x - gx)
            ordered = sorted(near, key=side)
            half = len(ordered) // 2
            st = {"halves": [set(ordered[half:]), set(ordered[:half])], "start": self.frame,
                  "strike": None, "pos": g["pos"]}
            self.log(
                f"pincer on {g['n']} raiders power {g['power']:.0f} at "
                f"({gx:.0f},{gz:.0f}) with {len(near)} ({power:.0f})"
            )
        st["pos"] = g["pos"]
        halves = [h & set(near) for h in st["halves"]]
        for uid in near:
            if not any(uid in h for h in halves):
                min(halves, key=len).add(uid)
        st["halves"] = halves
        if st["strike"] is None:
            ready = True
            for h, p in zip(halves, pts):
                if not h:
                    continue
                c = (sum(self.units[u].x for u in h) / len(h), sum(self.units[u].z for u in h) / len(h))
                if self.dist(c, p) > 220.0:
                    ready = False
            if ready or self.frame - st["start"] > 450:
                st["strike"] = self.frame
                self.log(f"pincer strike ({'in place' if ready else 'timeout'})")
        elif self.frame - st["strike"] > 450:
            st["start"], st["strike"] = self.frame, None
        for h, p in zip(halves, pts):
            for uid in h:
                out[uid] = ("strike", g["pos"]) if st["strike"] is not None else ("stage", p)
        self.pincer = st
        self.pincer_orders = out
        return out
