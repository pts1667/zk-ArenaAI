"""Composition, retreat and target-selection helpers for the zkbot Brain.

These are deliberately coarse: ControllerAI cannot promise full unit-vs-unit
weapon data, but role, cost, range, health and air/ground are enough to stop
the worst counter mistakes (no AA versus air, riot tanks versus skirmishers,
raiders trickling in one at a time).
"""
from __future__ import annotations

import collections
import math

from . import catalog, units
from .defense import _is_real_aircraft
from .heavycounter import is_armored_heavy
from .models import UnitInfo
from .threat import ground_dps, weapon_range

from .params import FLAGS, P
from .raidprobe import ACCEPT_SHARE, RAID_SHARE, RIOT_MEMORY, RIOT_VETO, is_ground_turret

RETREAT_HP_CHEAP = P("retreat.hp_cheap", 0.30, 0.1, 0.6, "base retreat HP fraction for cheap units")
RETREAT_HP = P("retreat.hp", 0.45, 0.2, 0.7, "base retreat HP fraction for other units")
REGEN_RETREAT_HP = P("retreat.regen_hp", 0.55, 0.3, 0.6, "retreat HP for idle-regenerating units (flag regenretreat)")


class TacticsMixin:
    # ------------------------------------------------------- composition

    def contacts(self) -> dict[int, UnitInfo]:
        """Visible enemies plus radar contacts, de-duplicated by unit id."""
        out = dict(self.enemies)
        for uid, u in self.radar.items():
            out.setdefault(uid, u)
        return out

    def seen_enemy_roles(self) -> "collections.Counter[str]":
        """Distinct enemy units ever seen, by coarse role."""
        out: collections.Counter[str] = collections.Counter()
        for name, ids in getattr(self, "enemy_seen_ids", {}).items():
            role = units.role_of(units.by_name(name))
            if role in ("raid", "scout"):
                out["raider"] += len(ids)
            elif role in ("assault", "heavy", "antiheavy"):
                out["assault"] += len(ids)
            else:
                out[role] += len(ids)
        # Moving radar blips identified by speed but never seen in LOS.
        for role, n in self.blip_guessed_roles().items():
            out["raider" if role == "raid" else role] += n
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
        late = {
            "raid": 0.24,
            "skirm": 0.24,
            "riot": 0.20,
            "assault": 0.20,
            "arty": 0.08,
        }
        spendmix = "spendmix" in FLAGS
        if spendmix:
            # CircuitAI's army metal after minute 12 (o2 observer games):
            # assault 12%, skirm 8%, raid 7%, arty 7%, riot 4% of all spend.
            late = {"raid": 0.18, "skirm": 0.24, "riot": 0.12, "assault": 0.30, "arty": 0.16}
        # Early game is about map control: lots of raiders, a little riot
        # and assault, no artillery.  (A human review: too few Glaives, and
        # the bot lost the map to a handful of Kodachis.)  Blend toward the
        # mid-game mix between 5 and 15 minutes.
        early = {"raid": 0.62, "skirm": 0.08, "riot": 0.14, "assault": 0.12, "arty": 0.0}
        phase = min(1.0, max(0.0, (self.frame_time() - 300.0) / 600.0))
        if "staticmix" in FLAGS:
            phase = 1.0
        desired = {k: early[k] * (1.0 - phase) + late[k] * phase for k in late}
        ground = ec["raider"] + ec["skirm"] + ec["riot"] + ec["assault"] + ec["arty"] + ec["other"]
        air_active = ec["air"] >= 1 or self.aircraft_seen()
        want_aa = self.desired_aa_count()
        if air_active and want_aa > 0:
            desired["aa"] = min(0.12, want_aa / combat)
        seen = self.seen_enemy_roles()
        if "staticmix" in FLAGS:
            if ec["raider"] >= max(3, int(ground * 0.30)):
                desired["riot"] += 0.10
                desired["raid"] -= 0.03
        else:
            # Riots scale with the raider threat instead of jumping: a few
            # raiders killing a constructor used to flip production to riots
            # and hand over the map.
            desired["riot"] += min(0.14, 0.015 * max(ec["raider"], seen["raider"] // 3))
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
        ball = self.enemy_slow_ball() if hasattr(self, "enemy_slow_ball") else None
        if ball is not None and "staticmix" not in FLAGS and "noball" not in FLAGS:
            # Large raider + Ronin groups beat Buoy/Grizzly balls; our own
            # riots are too slow and outranged there.
            desired.update(raid=0.40, skirm=0.35, riot=0.07, assault=0.10, arty=0.08)
        for key in list(desired):
            desired[key] = max(0.04, min(0.62, desired[key]))
        ramp = self.raid_ramp()
        if ramp < 1.0:
            desired["riot"] *= ramp
            desired["assault"] *= ramp
        if "staticmix" not in FLAGS:
            desired["arty"] = desired["arty"] if phase > 0.2 or ec["static"] >= 3 else 0.0
            # Skirmishers exist to kite slow units (riots, assaults).  Until
            # some have been seen they are dead weight.
            slow = seen["riot"] + seen["assault"]
            if ball is not None:
                pass
            elif slow == 0:
                desired["skirm"] = 0.0
            elif phase < 1.0:
                desired["skirm"] = max(desired["skirm"], min(0.24, 0.04 * slow))
        desired = self.armored_counter_mix(desired)
        # Explicit AA minimum: no ground army can shoot down an air wave.
        # The historical aircraft-seen flag, rather than a commander drone,
        # is what makes this active.
        if air_active and oc["aa"] < self.desired_aa_count():
            return "aa"
        current = {
            "raid": oc["raider"],
            "skirm": oc["skirm"],
            "riot": oc["riot"],
            "assault": oc["assault"],
            "arty": oc["arty"],
            # Missing before: the AA deficit never shrank, so AA won every
            # tie once aircraft had been seen.
            "aa": oc["aa"],
        }
        if spendmix:
            # Shares of metal started in the last 4 minutes, not of the
            # standing army: raiders are cheap and die first, so a head-count
            # quota refilled them without end (~150 dead Glaives a game,
            # 20% of all metal vs CircuitAI's 5-7%).
            spend = self.recent_army_spend()
            total = sum(spend.values())
            if total >= 300.0:
                current = {k: spend.get(k, 0.0) / total * combat for k in current}
        best_role = None
        best_deficit = 0.0
        for role, want in desired.items():
            deficit = want * combat - current.get(role, 0)
            # A small preference for the role that counters the largest
            # visible category avoids ties producing a random composition.
            if role == "riot" and ec["raider"] > 0 and ramp >= 1.0:
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
        # Caretakers sit next to factories and repair what comes home.
        repair_spots = [
            own.pos
            for own in self.units.values()
            if own.name == "staticcon" and own.is_finished
        ]
        candidates.extend(repair_spots)
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
            if c in repair_spots:
                score -= 600.0
            score += self.threat_at((cx, cz)) * 20.0
            if conflict:
                score += 1800.0
            if enemy_at_dest is not None:
                score += 2500.0
            if score < best_score:
                best_score = score
                best = (cx, cz)
        return best or self.retreat_point(u)

    def retreat_threshold(self, u: UnitInfo, cheap: bool) -> float:
        """HP fraction below which a unit pulls back, from the game state.

        A fixed 30%/45% either fed damaged units into lost fights or walked
        them out of won ones (lowering it everywhere lost 5 of 7 games): stay
        while the local fight is clearly won, leave earlier when it is lost,
        when the unit is expensive, or when our whole army is behind.
        """
        base = RETREAT_HP_CHEAP if cheap else RETREAT_HP
        if "staticretreat" in FLAGS:
            return base
        ours, theirs = self.local_balance(u.pos, 700.0)
        ratio = ours / max(1.0, theirs)
        ev = self.fight_eval(u.pos, commit=True) if "fightnet" in FLAGS else None
        if ev is not None:
            # Learned trade instead of the power ratio (docs/ML.md stage 1).
            from .fightnet import RETREAT_LOSE_EXCH, RETREAT_WIN_EXCH
            x = ev["exchange"]
            ratio = x
            if x >= RETREAT_WIN_EXCH:
                f = 0.55
            elif x >= 0.5 * (1.0 + RETREAT_WIN_EXCH):
                f = 0.75
            elif x < RETREAT_LOSE_EXCH:
                f = 1.25
            else:
                f = 1.0
        elif ratio >= 3.0:
            f = 0.55
        elif ratio >= 1.8:
            f = 0.75
        elif ratio < 0.6:
            f = 1.25
        else:
            f = 1.0
        # Deep in enemy territory the walk home rarely survives; while the
        # fight there is even or better, keep shooting a little longer.
        span = self.dist(self.base_pos, self.enemy_base or self.map.center)
        if span > 0 and ratio >= 1.0 and self.dist(u.pos, self.base_pos) > 0.6 * span:
            f *= 0.85
        if units.cost(u.data) >= 600.0:
            f *= 1.1
        # Against a slow ball a fast unit can leave whenever it likes; stay
        # in the fight longer.
        ball = self.enemy_slow_ball() if hasattr(self, "enemy_slow_ball") else None
        if (
            ball is not None
            and float(u.data.get("speed") or 0.0) >= 1.5 * ball["speed"]
            and self.dist(u.pos, ball["pos"]) < 1200.0
        ):
            f *= 0.75
        cache = getattr(self, "_power_cache", (0, 0.0, 0.0))
        if cache[2] > 0 and cache[1] < 0.7 * cache[2]:
            f *= 1.1
        return min(0.60, max(0.12, base * f))

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
        if u.alive_frac >= 0.60:
            return False, None
        threshold = self.retreat_threshold(u, cheap)
        if "regenretreat" in FLAGS and "regen" in catalog.tags(u.name):
            # Glaives and Reavers heal 20 HP/s after 5 s out of combat (full
            # in ~12 s).  At the cheap-unit threshold (~0.2-0.35) they died
            # within a second of turning back: 40% of losses in the first
            # 10 minutes were on retreat branches (h121-x3).
            threshold = max(threshold, REGEN_RETREAT_HP)
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
        # Running from something faster that already has us in range only
        # stops our own fire; stand and shoot instead.
        my_speed = float(u.data.get("speed") or 0.0)
        for e in self.contacts().values():
            if not self.is_dangerous(e) or not e.is_mobile:
                continue
            e_speed = float(e.data.get("speed") or 0.0)
            if e_speed > my_speed * 1.15 and self.dist(e.pos, u.pos) < weapon_range(e) + 120.0:
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
        my_range = weapon_range(u)
        my_dps = ground_dps(u) if not is_aa else 0.0
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
            # Never trade an army for a reactor: only shoot one from
            # outside its death blast.
            blast = units.BLAST_RADIUS.get(e.name)
            if blast is not None and weapon_range(u) < blast:
                continue
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
            if "focus" in FLAGS and (not e.is_air or is_aa):
                score += self._focus_score(u, e, d, my_range, my_dps)
            if score > best_score:
                best_score = score
                best = e
        if best is not None:
            focus = self._focus_assigned()
            focus[best.uid] = focus.get(best.uid, 0.0) + my_dps
        return best

    def _focus_assigned(self) -> dict[int, float]:
        """DPS already assigned to each enemy during this decision step."""
        if getattr(self, "_focus_frame", -1) != self.frame:
            self._focus_frame = self.frame
            self._focus = {}
        return self._focus

    def _focus_score(self, u: UnitInfo, e: UnitInfo, d: float, my_range: float, my_dps: float) -> float:
        """Shoot what is in range, what dies fastest per damage dealt, and
        what the rest of the army is already shooting (until overkill)."""
        s = 0.0
        if d <= my_range + 20.0:
            s += 220.0
        hp = max(30.0, e.health if e.health > 0 else e.max_health)
        e_dps = ground_dps(e)
        s += min(260.0, 1800.0 * e_dps / hp)
        if e.max_health > 0:
            s += (1.0 - min(1.0, hp / e.max_health)) * 200.0
        cur = getattr(self, "unit_order", {}).get(u.uid)
        if cur and cur[0] == "attack" and cur[3] == e.uid:
            s += 150.0  # keep shooting the current target; retargeting costs time
        assigned = self._focus_assigned().get(e.uid, 0.0)
        if assigned > 0.0:
            # About two seconds of assigned fire kills it: pile on; more
            # than that is overkill, spread out.
            if assigned < hp / 2.0:
                s += 180.0
            elif assigned > hp:
                s -= 160.0
        return s

    # ------------------------------------------------------- raid squads

    def raid_defense_value(self, pos: tuple[float, float], radius: float = 520.0) -> float:
        """Estimated metal value of what protects a raid target.  Riots and
        turrets are what raiders lose to, so they count extra."""
        r2 = radius * radius
        total = 0.0
        seen: set[int] = set()
        probe = "raidprobe" in FLAGS
        see_turrets = probe or "seeturrets" in FLAGS
        armored = "armorcounter" in FLAGS
        hitrun = "hitrun" in FLAGS
        for e in self.contacts().values():
            seen.add(e.uid)
            turret = see_turrets and is_ground_turret(e)
            if e.is_air or not (turret or self.is_dangerous(e)):
                continue
            if turret and probe:
                if not self.turret_covers(e, pos):
                    continue
            elif (e.x - pos[0]) ** 2 + (e.z - pos[1]) ** 2 > r2:
                continue
            c = units.cost(e.data)
            role = units.role_of(e.data)
            if "dyntrainer" in e.name or "commander" in e.name:
                c = 900.0
            elif hitrun and (role == "riot" or (armored and is_armored_heavy(e.name))):
                c = RIOT_VETO
            elif role == "riot" or (not e.is_mobile and units.is_defense(e.data)):
                c *= 1.5
            total += c
        if hitrun:
            for uid, (e, f) in getattr(self, "enemy_memory", {}).items():
                if (uid not in seen and self.frame - f <= RIOT_MEMORY
                        and (units.role_of(e.data) == "riot" or (armored and is_armored_heavy(e.name)))
                        and (e.x - pos[0]) ** 2 + (e.z - pos[1]) ** 2 <= r2):
                    total += RIOT_VETO
        for uid, (e, _f) in getattr(self, "enemy_structures", {}).items():
            if uid in seen or not units.is_defense(e.data):
                continue
            if probe:
                if is_ground_turret(e) and self.turret_covers(e, pos):
                    total += units.cost(e.data) * 1.5
            elif (e.x - pos[0]) ** 2 + (e.z - pos[1]) ** 2 <= r2:
                total += units.cost(e.data) * 1.5
        return total

    def raid_accept_value(self, squad: list[int]) -> float:
        """Defence a squad will raid through: half its own metal value, so a
        big group ignores a lone turret or riot while a pair of raiders
        does not."""
        value = sum(units.cost(self.units[uid].data) for uid in squad if uid in self.units)
        if "hitrun" in FLAGS:
            return RAID_SHARE * value
        return (ACCEPT_SHARE if "raidprobe" in FLAGS else 0.5) * value

    def update_raid_squads(
        self,
        raider_uids: list[int],
        stage: tuple[float, float],
        group_needed: int,
        out: dict[int, tuple[tuple[float, float], bool]],
    ) -> None:
        t = self.frame_time()
        size_cap = 4 if t < 300 else (6 if t < 900 else 8)
        uids = sorted(raider_uids)
        n = min(3, max(1, math.ceil(len(uids) / size_cap)))
        while n > 1 and len(uids) // n < group_needed:
            n -= 1
        per = math.ceil(len(uids) / n)
        squads = [uids[i * per:(i + 1) * per] for i in range(n)]
        state = self.__dict__.setdefault("raid_squad_state", {})
        black = self.__dict__.setdefault("raid_blacklist", {})
        for k, until in list(black.items()):
            if until < self.frame:
                black.pop(k, None)
        taken: list[tuple[float, float]] = []
        for i, squad in enumerate(squads):
            if len(squad) < group_needed:
                continue
            tgt, tframe = state.get(i, (None, -10**9))
            if tgt is not None:
                defended = self.raid_defense_value(tgt)
                accept = self.raid_accept_value(squad)
                hot = defended > accept
                if hot:
                    black[(int(tgt[0] // 250), int(tgt[1] // 250))] = self.frame + 2400
                    self.log(
                        f"raid squad {i} target {tuple(round(v) for v in tgt)} too well "
                        f"defended ({defended:.0f} vs {accept:.0f}); retargeting"
                    )
                if (
                    hot
                    or self.frame - tframe > 600
                    or self.dist(tgt, self.base_pos) < 450.0
                ):
                    tgt = None
            if tgt is None:
                old = state.get(i, (None, 0))[0]
                tgt = self.choose_raid_group_target(squad, exclude=taken)
                state[i] = (tgt, self.frame)
                if tgt is not None and (old is None or self.dist(old, tgt) > 200.0):
                    self.log(f"raid squad {i} ({len(squad)}) -> {tuple(round(v) for v in tgt)}")
            if tgt is None:
                continue
            taken.append(tgt)
            for uid in squad:
                u = self.units.get(uid)
                if u is None:
                    continue
                at = self.dist(u.pos, stage) < 900.0 or self.dist(u.pos, tgt) < 1000.0
                # Members already on the way keep going: waiting at the stage
                # for stragglers stalled whole squads.
                out[uid] = (tgt, at or self.dist(u.pos, tgt) < self.dist(stage, tgt))
        for i in list(state):
            if i >= len(squads):
                state.pop(i, None)

    def choose_raid_group_target(
        self, ready: list[int], exclude: list[tuple[float, float]] | None = None
    ) -> tuple[float, float] | None:
        """One target for the whole raider squad, not one per raider."""
        rep = self.units.get(ready[0]) if ready else None
        if rep is None:
            return self.enemy_base
        # Prefer visible economy near the squad.
        best = None
        best_score = 1e18
        squad_dps = 0.0
        for uid in ready:
            su = self.units.get(uid)
            if su is not None:
                squad_dps += ground_dps(su)
        remembered = [e for e, _f in getattr(self, "enemy_structures", {}).values()]
        seen_ids = set(self.enemies)
        pool = list(self.enemies.values()) + [e for e in remembered if e.uid not in seen_ids]
        new = "oldraid" not in FLAGS
        exclude = exclude or []
        black = getattr(self, "raid_blacklist", {})
        accept = self.raid_accept_value(ready)
        for e in pool:
            if not e.is_finished and e.uid in seen_ids:
                continue
            if new:
                if (int(e.x // 250), int(e.z // 250)) in black:
                    continue
                if any(self.dist(e.pos, q) < 600.0 for q in exclude):
                    continue
                # Compare the metal value guarding the target with the
                # squad's own (big squads may ignore some defence).
                defended = self.raid_defense_value(e.pos)
                if defended > accept:
                    continue
                guard = defended / 40.0
            else:
                # Raiders win by hitting what is not defended: the enemy DPS
                # covering the target must be small compared with the squad.
                guard = self.threat_at(e.pos)
                if guard > squad_dps * 0.6:
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
            score = self.dist(rep.pos, e.pos) - value + guard * 8.0
            if score < best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        if "raidprobe" in FLAGS:
            return self.probe_target(ready, self._squad_center(ready) or rep.pos, exclude)
        # Nothing visible: head for the enemy-side extractors/base.
        if new:
            return self.pick_raid_target(rep, exclude=exclude) or self.enemy_base
        return self.pick_raid_target(rep) or self.enemy_base
