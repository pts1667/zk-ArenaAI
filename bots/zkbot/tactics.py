"""Composition, retreat and target-selection helpers for the zkbot Brain.

These are deliberately coarse: ControllerAI cannot promise full unit-vs-unit
weapon data, but role, cost, range, health and air/ground are enough to stop
the worst counter mistakes (no AA versus air, riot tanks versus skirmishers,
raiders trickling in one at a time).
"""
from __future__ import annotations

import math
from typing import Any

from . import units
from .defense import _is_real_aircraft
from .models import UnitInfo


class TacticsMixin:
    # ------------------------------------------------------- composition

    def contacts(self) -> dict[int, UnitInfo]:
        """Visible enemies plus radar contacts, de-duplicated by unit id."""
        out = dict(self.enemies)
        for uid, u in self.radar.items():
            out.setdefault(uid, u)
        return out

    def enemy_composition(self) -> dict[str, int]:
        """Coarse enemy composition, including radar contacts.

        Radar blips carry a def id, so an incoming air or raider wave can be
        answered before it reaches line of sight.
        """
        comp = {
            "air": 0,
            "raider": 0,
            "skirm": 0,
            "riot": 0,
            "assault": 0,
            "arty": 0,
            "aa": 0,
            "builder": 0,
            "static": 0,
            "other": 0,
        }
        for e in self.contacts().values():
            if e.is_builder:
                comp["builder"] += 1
            if e.is_factory or units.is_defense(e.data):
                comp["static"] += 1
            if not self.is_dangerous(e):
                continue
            if e.is_air:
                if _is_real_aircraft(e):
                    comp["air"] += 1
                else:
                    # Commander drones / radar-morph scouts are not a reason
                    # to build anti-air.
                    comp["other"] += 1
                continue
            role = units.role_of(e.data)
            if role in ("raid", "scout"):
                comp["raider"] += 1
            elif role == "skirm":
                comp["skirm"] += 1
            elif role == "riot":
                comp["riot"] += 1
            elif role in ("assault", "heavy", "antiheavy", "snipe"):
                comp["assault"] += 1
            elif role in ("arty", "heavyarty"):
                comp["arty"] += 1
            elif role == "aa":
                comp["aa"] += 1
            elif e.is_combat:
                comp["other"] += 1
        return comp

    def our_role_counts(self) -> dict[str, int]:
        counts = {
            "air": 0,
            "raider": 0,
            "skirm": 0,
            "riot": 0,
            "assault": 0,
            "arty": 0,
            "aa": 0,
            "other": 0,
        }
        for uid in self.army:
            u = self.units.get(uid)
            if not u:
                continue
            if u.is_air:
                counts["air"] += 1
                continue
            role = units.role_of(u.data)
            if units.is_aa(u.data) or role == "aa":
                counts["aa"] += 1
            elif role in ("raid", "scout"):
                counts["raider"] += 1
            elif role == "skirm":
                counts["skirm"] += 1
            elif role == "riot":
                counts["riot"] += 1
            elif role in ("assault", "heavy", "antiheavy", "snipe"):
                counts["assault"] += 1
            elif role in ("arty", "heavyarty"):
                counts["arty"] += 1
            else:
                counts["other"] += 1
        return counts

    def counter_role(self) -> str | None:
        """Choose the role with the largest deficit against a balanced quota.

        The old version returned one hard counter for the whole army: e.g.
        "enemy has assault, build only skirmishers".  That produced armies
        with no front line and no siege.  Quotas keep a general-purpose army
        and merely tilt it toward the current enemy mix.
        """
        ec = self.enemy_composition()
        oc = self.our_role_counts()
        combat = max(1, len(self.army))
        desired = {
            "raid": 0.24,
            "skirm": 0.24,
            "riot": 0.20,
            "assault": 0.20,
            "arty": 0.08,
        }
        ground = ec["raider"] + ec["skirm"] + ec["riot"] + ec["assault"] + ec["arty"] + ec["other"]
        air_active = ec["air"] >= 1 or self.aircraft_seen()
        if air_active:
            desired["aa"] = 0.12
        if ec["raider"] >= max(3, int(ground * 0.30)):
            desired["riot"] += 0.10
            desired["raid"] -= 0.03
        if ec["skirm"] >= max(3, int(ground * 0.30)):
            desired["raid"] += 0.12
            desired["riot"] -= 0.04
        if ec["assault"] >= max(3, int(ground * 0.22)):
            desired["skirm"] += 0.10
            desired["arty"] += 0.04
        if ec["arty"] >= 2:
            desired["raid"] += 0.08
        if ec["static"] >= 3:
            desired["arty"] += 0.08
            desired["assault"] += 0.03
        if ec["riot"] >= max(3, int(ground * 0.25)):
            desired["skirm"] += 0.08
        for key in list(desired):
            desired[key] = max(0.04, min(0.42, desired[key]))
        # Explicit AA minimum: no ground army can shoot down an air wave.
        # The historical aircraft-seen flag, rather than a commander drone,
        # is what makes this active.
        if air_active and oc["aa"] < max(1, combat // 6):
            return "aa"
        current = {
            "raid": oc["raider"],
            "skirm": oc["skirm"],
            "riot": oc["riot"],
            "assault": oc["assault"],
            "arty": oc["arty"],
        }
        best_role = None
        best_deficit = 0.0
        for role, want in desired.items():
            deficit = want * combat - current.get(role, 0)
            # A small preference for the role that counters the largest
            # visible category avoids ties producing a random composition.
            if role == "riot" and ec["raider"] > 0:
                deficit += min(1.0, ec["raider"] * 0.05)
            if role == "skirm" and ec["assault"] > 0:
                deficit += min(1.0, ec["assault"] * 0.04)
            if role == "raid" and ec["skirm"] > 0:
                deficit += min(1.0, ec["skirm"] * 0.04)
            if role == "arty" and ec["static"] > 0:
                deficit += min(1.5, ec["static"] * 0.10)
            if deficit > best_deficit:
                best_deficit = deficit
                best_role = role
        return best_role

    # ------------------------------------------------------- retreat

    def path_conflicted(
        self, start: tuple[float, float], end: tuple[float, float], radius: float = 300.0
    ) -> bool:
        """True when a straight retreat path passes close to a dangerous unit."""
        length = self.dist(start, end)
        if length < 80.0:
            return False
        contacts = [e for e in self.contacts().values() if self.is_dangerous(e)]
        if not contacts:
            return False
        steps = max(1, int(length / 180.0))
        for i in range(steps + 1):
            f = i / steps
            x = start[0] + (end[0] - start[0]) * f
            z = start[1] + (end[1] - start[1]) * f
            for e in contacts:
                if self.dist((x, z), e.pos) < radius:
                    return True
        return False

    def safe_retreat_point(self, u: UnitInfo) -> tuple[float, float]:
        """Pick the safest reachable fallback instead of always running home."""
        bx, bz = self.base_pos
        candidates: list[tuple[float, float]] = [self.retreat_point(u)]
        for own in self.units.values():
            if own.uid == u.uid:
                continue
            if units.is_defense(own.data) and own.is_finished:
                if self.dist(own.pos, self.base_pos) < 2400.0:
                    candidates.append(own.pos)
            elif own.name == "staticmex" and own.is_finished:
                if self.dist(own.pos, self.base_pos) < 1800.0:
                    candidates.append(own.pos)
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        dx, dz = bx - ex, bz - ez
        length = math.hypot(dx, dz) or 1.0
        px, pz = -dz / length, dx / length
        for side in (-1.0, 1.0):
            candidates.append((bx + px * side * 500.0, bz + pz * side * 500.0))
            candidates.append((bx + px * side * 300.0, bz + pz * side * 300.0))
        best = None
        best_score = 1e18
        for c in candidates:
            cx = min(max(40.0, c[0]), max(41.0, self.map.width - 40.0))
            cz = min(max(40.0, c[1]), max(41.0, self.map.height - 40.0))
            if self.dist(u.pos, (cx, cz)) < 40.0:
                continue
            enemy_at_dest = self.enemy_near((cx, cz), 320.0)
            conflict = self.path_conflicted(u.pos, (cx, cz))
            score = self.dist(u.pos, (cx, cz))
            if conflict:
                score += 1800.0
            if enemy_at_dest is not None:
                score += 2500.0
            if score < best_score:
                best_score = score
                best = (cx, cz)
        return best or self.retreat_point(u)

    def retreat_decision(
        self, u: UnitInfo
    ) -> tuple[bool, tuple[float, float] | None]:
        """Return (retreat, destination), weighing survival against sacrifice.

        Cheap raiders may trade themselves for economy or keep firing when
        every path home crosses the enemy.  Expensive units fall back along the
        least dangerous route instead.
        """
        role = units.role_of(u.data)
        cheap = units.cost(u.data) <= 120.0 or role in ("raid", "scout", "bomb")
        threshold = 0.30 if cheap else 0.45
        if u.alive_frac >= threshold:
            return False, None
        recent_threat = self.frame - self.threat_frame < 600
        nearby_enemy = any(
            self.dist(u.pos, e.pos) < 900.0
            for e in self.contacts().values()
            if self.is_dangerous(e)
        )
        if not recent_threat and not nearby_enemy:
            return False, None
        # If the local fight is clearly won, a damaged unit keeps firing from
        # the back instead of walking away from free damage.
        own_near = sum(
            1
            for uid in self.army
            if uid in self.units and self.dist(self.units[uid].pos, u.pos) < 700.0
        )
        enemy_near = sum(
            1
            for e in self.contacts().values()
            if self.is_dangerous(e) and self.dist(e.pos, u.pos) < 700.0
        )
        if own_near >= max(2, int(enemy_near * 1.5)) and u.alive_frac > 0.18:
            return False, None
        destination = self.safe_retreat_point(u)
        conflict = self.path_conflicted(u.pos, destination)
        dangerous_dest = self.enemy_near(destination, 260.0) is not None
        if cheap and (conflict or dangerous_dest):
            # A cheap raider with no clean way out is worth more as a trade.
            # Only burn it if there is actually something to hit nearby.
            has_target = any(
                self.dist(u.pos, e.pos) < 750.0
                and (not e.is_combat or e.is_builder)
                for e in self.contacts().values()
            )
            if has_target or dangerous_dest:
                return False, None
        return True, destination

    # ------------------------------------------------------- targeting

    def best_local_target(self, u: UnitInfo) -> UnitInfo | None:
        """Counter-aware target choice among currently visible enemies.

        Artillery may also fire at radar-only blips; that is why the bot keeps
        radar coverage near the guns.  Other units still need LOS for a
        normal attack order.
        """
        role = units.role_of(u.data)
        use_radar = role in ("arty", "heavyarty")
        if not self.enemies and not (use_radar and self.radar):
            return None
        is_aa = units.is_aa(u.data) or role == "aa"
        threat = (
            self.threat_pos
            if self.threat_pos is not None and self.frame - self.threat_frame < 600
            else None
        )
        reach = max(900.0, float(u.data.get("maxWeaponRange") or 0.0) + 600.0)
        best = None
        best_score = -1e18
        sources = self.contacts().values() if use_radar else self.enemies.values()
        for e in sources:
            # Radar-only contacts are only useful as artillery target points.
            if e.uid in self.radar and not use_radar:
                continue
            if not e.is_finished and e.uid not in self.radar:
                continue
            d = self.dist(u.pos, e.pos)
            if d > reach:
                continue
            score = -d * 0.25
            erole = units.role_of(e.data)
            if is_aa:
                score += 1200.0 if e.is_air else -600.0
            elif e.is_air:
                # Ground units cannot shoot air; avoid chasing it.
                score -= 700.0
            if role in ("skirm", "arty") and erole in ("assault", "heavy", "riot", "antiheavy"):
                score += 260.0
            if role == "arty" and units.is_defense(e.data):
                score += 340.0
            if role == "arty" and e.is_factory:
                score += 280.0
            if role == "riot" and erole in ("raid", "scout"):
                score += 220.0
            if role in ("assault", "riot") and e.is_builder:
                score += 180.0
            if role == "skirm" and erole == "skirm":
                score += 60.0
            if e.is_factory:
                score += 220.0
            elif e.is_builder:
                score += 140.0
            elif e.name in ("staticmex", "staticcon"):
                score += 100.0
            if threat is not None:
                td = self.dist(e.pos, threat)
                score += max(0.0, 350.0 - td * 0.15)
            if score > best_score:
                best_score = score
                best = e
        return best

    def choose_raid_group_target(self, ready: list[int]) -> tuple[float, float] | None:
        """One target for the whole raider squad, not one per raider."""
        rep = self.units.get(ready[0]) if ready else None
        if rep is None:
            return self.enemy_base
        # Prefer visible economy near the squad.
        best = None
        best_score = 1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            value = 0.0
            if e.name in ("staticmex", "staticcon"):
                value = 420.0
            elif e.is_factory:
                value = 650.0
            elif units.is_energy(e.data):
                value = 300.0
            elif e.is_builder:
                value = 300.0
            else:
                continue
            score = self.dist(rep.pos, e.pos) - value
            if score < best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        # Nothing visible: head for the enemy-side extractors/base.
        return self.pick_raid_target(rep) or self.enemy_base
