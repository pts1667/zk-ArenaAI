"""MilitaryMixin methods for the zkbot Brain."""
from __future__ import annotations

import collections
import math

from . import units
from .models import UnitInfo
from .threat import combat_power, weapon_range


from .params import FLAGS


class _BranchCounter(collections.Counter):
    """Counter of military branches that also remembers, per unit, the last
    branch it was given, so deaths can be attributed to a behaviour."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.uid = None
        if not hasattr(owner, "unit_branch"):
            owner.unit_branch = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.uid is not None:
            self.owner.unit_branch[self.uid] = key


class MilitaryMixin:
    def is_raider(self, u: UnitInfo) -> bool:
        return units.role_of(u.data) in ("raid", "scout") and not u.is_air

    def army_mode(self) -> str:
        enemy_count = (
            sum(1 for e in self.enemies.values() if self.is_dangerous(e))
            + len(self.radar)
        )
        combat = self.combat_count()
        if enemy_count > self.enemy_peak or self.frame - self.enemy_peak_frame > 1800:
            self.enemy_peak = enemy_count
            self.enemy_peak_frame = self.frame
        # Overwhelming strength attacks and leaves raids to the responders
        # and turrets: with 17x the enemy's power the whole army kept
        # answering hover raids in defend mode until the game timed out.
        if "nooverwhelm" not in FLAGS and combat >= 40:
            peak = getattr(self, "enemy_power_peak", 0.0)
            cache = getattr(self, "_power_cache", (0, 0.0, 0.0))
            ours_now = self.army_power()
            theirs_now = max(self.enemy_army_power(), peak, cache[2])
            near_threat = 0.0
            if self.threat_pos is not None and self.frame - self.threat_frame < 900:
                _o, near_threat = self.local_balance(self.threat_pos, 900.0)
            if ours_now >= 4.0 * max(1.0, theirs_now) and near_threat < 0.25 * ours_now:
                self._last_mode = "attack"
                return "attack"
        # Sector threat is the strategic view: a wave moving through a
        # frontier sector is defended as a group, not chased unit by unit.
        threatened = self.most_threatened_sector()
        if (
            threatened is not None
            and threatened.threat > 0
            and threatened.distance_base < 1800.0
            and threatened.owner == "ours"
        ):
            return "defend"
        # Defend only a real attack, or enemy units close enough to the base
        # / turret line.  Previously any distant contact put the whole army in
        # defend mode forever while CircuitAI freely rebuilt its economy.
        if self.threat_pos and self.frame - self.threat_frame < 900:
            near_base = self.dist(self.threat_pos, self.base_pos) < 1800.0
            if near_base or enemy_count >= max(4, int(combat * 0.60)):
                return "defend"
        # Strength, not head count: compare the total combat power of our
        # army with every enemy combat unit seen in the last minute.
        ours = self.army_power()
        seen = self.enemy_army_power()
        # We only ever see part of the enemy army.  Keep a peak estimate that
        # decays over ~4 minutes, so units walking out of LOS are not
        # treated as dead.
        peak = getattr(self, "enemy_power_peak", 0.0)
        last = getattr(self, "enemy_power_peak_frame", self.frame)
        peak *= math.exp(-max(0, self.frame - last) / (30.0 * 240.0))
        peak = max(peak, seen)
        self.enemy_power_peak = peak
        self.enemy_power_peak_frame = self.frame
        theirs = max(seen, peak)
        self._power_cache = (self.frame, ours, theirs)
        was_attacking = getattr(self, "_last_mode", "gather") == "attack"
        min_units = max(12, 8 + int(self.frame_time() / 120))
        if was_attacking:
            ok = combat >= min_units * 0.6 and ours >= theirs * 1.0
        else:
            ok = combat >= min_units and ours >= max(1.0, theirs) * 1.4
        mode = "attack" if ok or combat >= 45 else "gather"
        self._last_mode = mode
        return mode

    def army_power(self) -> float:
        return sum(
            combat_power(self.units[uid]) for uid in self.army if uid in self.units
        )

    def enemy_army_power(self) -> float:
        total = 0.0
        for uid, (e, f) in self.enemy_memory.items():
            if not e.is_mobile or not self.is_dangerous(e):
                continue
            if e.is_air and e.name.startswith("drone"):
                continue
            if self.frame - f > 1800:
                continue
            total += combat_power(e)
        return total

    def assign_home_guard(self) -> None:
        """Keep a small defensive force at home while the main army attacks.

        Sector threats are useful, but if every combat unit marches to the
        enemy base, CircuitAI simply walks into the solar field.  The guard is
        made of AA / artillery / skirmishers first, then nearby units.
        """
        self.home_guard = {uid for uid in self.home_guard if uid in self.army}
        target = max(2, min(10, len(self.army) // 4))
        if len(self.home_guard) >= target:
            return
        candidates = [
            UID
            for UID in self.army
            if UID not in self.home_guard and UID in self.units and not self.units[UID].is_air
        ]
        if not candidates:
            return

        def priority(uid: int) -> tuple[int, float]:
            u = self.units[uid]
            role = units.role_of(u.data)
            if units.is_aa(u.data) or role == "aa":
                p = 0
            elif role == "skirm":
                p = 1
            elif role == "riot":
                p = 2
            elif role == "assault":
                p = 3
            elif role in ("arty", "heavyarty"):
                # Artillery belongs with the attacking force unless the base
                # is actually under threat; leaving all of it at home makes
                # every attack into a turret line fail.
                p = 4
            else:
                p = 6
            return p, self.dist(u.pos, self.base_pos)

        candidates.sort(key=priority)
        for uid in candidates[: target - len(self.home_guard)]:
            self.home_guard.add(uid)
        if self.home_guard:
            self.say(
                "home_guard",
                f"home guard {len(self.home_guard)}/{len(self.army)}",
            )

    def military_step(self) -> None:
        if not self.army:
            return
        self.assign_home_guard()
        mode = self.army_mode()
        # What the army does this step (armyplan.py), then each unit's order
        # from the behaviour list (behaviours.py).
        plan = self.plan_army(mode)
        self.army_plan = plan
        # Every unit goes down the behaviour list (behaviours.UNIT_BEHAVIOURS)
        # until one gives it an order.
        br = _BranchCounter(self)
        chain = self.unit_behaviours()
        for uid in list(self.army):
            u = self.units.get(uid)
            if not u:
                continue
            br.uid = uid
            for behaviour in chain:
                branch = behaviour(u, plan)
                if branch is not None:
                    br[branch] += 1
                    break
        self._branch_counts = br

    # ------------------------------------------------------------ group logic

    def assign_responders(self, mode: str) -> dict[int, tuple[float, float]]:
        """Detach squads to owned sectors under attack anywhere on the map.

        Each squad is the nearest available units until its combat power is
        1.5x the local enemy power; a sector we cannot answer is skipped
        rather than fed.  Near-base attacks are already handled by defend
        mode, which pulls everyone.
        """
        out: dict[int, tuple[float, float]] = {}
        if mode == "defend" or not getattr(self, "sectors", None):
            return out
        self.update_sector_state()
        cands = []
        for sec in self.sectors:
            if sec.owner != "ours" or sec.enemy_strength <= 0:
                continue
            pos = (sec.x, sec.z)
            _o, theirs = self.local_balance(pos, 750.0)
            if theirs <= 0:
                continue
            cands.append((sec.income, theirs, pos))
        # Raiders on radar heading for our mexes/constructors: meet them.
        for w in self.update_raid_warnings():
            if any(self.dist(w["pos"], c[2]) < 600.0 for c in cands):
                continue
            cands.append((3.0, w["power"], w["pos"]))
        if not cands:
            return out
        cands.sort(key=lambda c: -c[0])
        free = [
            uid
            for uid in self.army
            if uid in self.units
            and uid not in self.retreating
            and uid not in self.home_guard
            and units.role_of(self.units[uid].data) != "bomb"
            and not self.units[uid].is_air
        ]
        used: set[int] = set()
        for _inc, theirs, pos in cands[:3]:
            need = theirs * 1.5
            pool = sorted(
                (uid for uid in free if uid not in used),
                key=lambda uid: self.dist(self.units[uid].pos, pos),
            )
            got = 0.0
            squad = []
            for uid in pool:
                if self.dist(self.units[uid].pos, pos) > 3000.0:
                    break
                squad.append(uid)
                got += combat_power(self.units[uid])
                if got >= need and len(squad) >= 2:
                    break
            if got < need:
                continue
            for uid in squad:
                used.add(uid)
                out[uid] = pos
        if out and self.frame - getattr(self, "_last_resp_log", -10**9) >= 300:
            self._last_resp_log = self.frame
            self.log(f"responders {len(out)} -> {sorted({tuple(round(v) for v in p) for p in out.values()})}")
        return out

    def attack_target_stale(self, target: tuple[float, float], center: tuple[float, float]) -> bool:
        """True once the group has stood on a target for 20 s with nothing to
        shoot: forget remembered structures there and blacklist the spot."""
        black = getattr(self, "attack_blacklist", {})
        self.attack_blacklist = black
        for k, until in list(black.items()):
            if until < self.frame:
                black.pop(k, None)
        key = (int(target[0] // 200), int(target[1] // 200))
        if key in black:
            return True
        if self.dist(center, target) > 450.0:
            self._target_arrive = None
            return False
        busy = any(self.dist(e.pos, target) < 600.0 for e in self.enemies.values())
        arr = getattr(self, "_target_arrive", None)
        if busy or arr is None or arr[0] != key:
            self._target_arrive = (key, self.frame)
            return False
        if self.frame - arr[1] < 600:
            return False
        black[key] = self.frame + 3600
        for uid, (e, _f) in list(getattr(self, "enemy_structures", {}).items()):
            if self.dist(e.pos, target) < 300.0:
                self.enemy_structures.pop(uid, None)
        self.log(f"attack target {tuple(round(v) for v in target)} empty; blacklisted")
        return True


    def median_pos(self, uids: list[int]) -> tuple[float, float] | None:
        xs = sorted(self.units[u].x for u in uids if u in self.units)
        zs = sorted(self.units[u].z for u in uids if u in self.units)
        if not xs:
            return None
        return xs[len(xs) // 2], zs[len(zs) // 2]

    def fallback_point(self, anchor: tuple[float, float] | None) -> tuple[float, float]:
        """Where a defending group that cannot win holds: our turret cluster
        between the attack and the base, else the base itself."""
        ref = anchor or self.base_pos
        best = None
        best_d = 1e18
        for t in self.units.values():
            if not (units.is_defense(t.data) and not t.is_mobile and t.is_finished):
                continue
            if self.dist(t.pos, self.base_pos) > 1800.0:
                continue
            d = self.dist(t.pos, ref) + self.dist(t.pos, self.base_pos) * 0.5
            if d < best_d:
                best_d = d
                best = t.pos
        return best or self.base_pos

    def choose_attack_target(
        self, center: tuple[float, float], power: float
    ) -> tuple[float, float] | None:
        """Most valuable enemy position the group can actually beat."""
        cands: list[tuple[float, tuple[float, float]]] = []
        for e, _f in getattr(self, "enemy_structures", {}).values():
            v = 0.0
            if e.name == "staticmex":
                v = 420.0
            elif e.is_factory:
                v = 700.0
            elif units.is_energy(e.data):
                v = 260.0
            elif units.is_defense(e.data):
                v = 120.0
            elif e.name in ("staticcon", "staticradar"):
                v = 200.0
            if v > 0:
                cands.append((v, e.pos))
        for e in self.enemies.values():
            if e.is_mobile and e.is_builder:
                cands.append((300.0, e.pos))
        s = self.best_attack_sector()
        if s is not None:
            p = self.best_defense_pos(s) or (s.x, s.z)
            cands.append((350.0 + s.income * 20.0, p))
        if self.enemy_base:
            cands.append((250.0, self.enemy_base))
        best = None
        best_score = -1e18
        cache: dict[tuple[int, int], float] = {}
        black = getattr(self, "attack_blacklist", {})
        prev = getattr(self, "_attack_target_prev", None)
        reactors = [
            (e.pos, units.BLAST_RADIUS[e.name])
            for e, _f in getattr(self, "enemy_structures", {}).values()
            if e.name in units.BLAST_RADIUS
        ]
        for value, pos in cands:
            if (int(pos[0] // 200), int(pos[1] // 200)) in black:
                continue
            if any(self.dist(pos, rp) < r + 150.0 for rp, r in reactors):
                continue
            key = (int(pos[0] // 300), int(pos[1] // 300))
            theirs = cache.get(key)
            if theirs is None:
                _o, theirs = self.local_balance(pos, 900.0)
                cache[key] = theirs
            if power < theirs * 1.3:
                continue
            score = value - self.dist(center, pos) * 0.12 - theirs * 0.5
            # Stick to the current objective: re-scoring every step flipped
            # the army between targets and it marched back and forth.
            if "sticky" in FLAGS and prev is not None and self.dist(pos, prev) < 350.0:
                score += 250.0
            if score > best_score:
                best_score = score
                best = pos
        self._attack_target_prev = best
        return best

    def outranged_escape(self, u: UnitInfo) -> tuple[float, float] | None:
        """Step out of reach of a stronger enemy that outranges us.

        Reavers walked into an Ogre (a riot that outranges them) one at a
        time.  Unless the local fight is clearly won, back off to just
        outside its range on the side of our own forces and let skirmishers
        or numbers deal with it.
        """
        role = units.role_of(u.data)
        if role in ("arty", "heavyarty", "aa", "bomb") or u.is_air:
            return None
        my_r = weapon_range(u)
        if my_r <= 0.0:
            return None
        my_p = combat_power(u)
        worst = None
        worst_d = 1e18
        for e in self.enemies.values():
            if e.is_air or not e.is_mobile or not self.is_dangerous(e):
                continue
            # Only units that win the brawl once we close in: riots and
            # assaults.  Skirmishers (e.g. Buoy) outrange everything we have
            # and are beaten by closing the distance, not by backing off
            # (the broad rule lost ~10k metal per game against amphibious).
            if units.role_of(e.data) not in ("riot", "assault", "heavy", "antiheavy"):
                continue
            er = weapon_range(e)
            if er < my_r + 25.0:
                continue
            d = self.dist(u.pos, e.pos)
            if d > er + 120.0:
                continue
            if combat_power(e) < 1.3 * my_p:
                continue
            if d < worst_d:
                worst, worst_d = e, d
        if worst is None:
            return None
        ours, theirs = self.local_balance(u.pos, 650.0)
        if ours >= 1.5 * theirs:
            return None
        er = weapon_range(worst)
        hx, hz = self.rally_point() if self.dist(u.pos, self.base_pos) > 1200.0 else self.base_pos
        dx, dz = u.x - worst.x, u.z - worst.z
        n = math.hypot(dx, dz) or 1.0
        # Away from the enemy, bent toward home/rally.
        tx, tz = hx - u.x, hz - u.z
        tn = math.hypot(tx, tz) or 1.0
        vx = dx / n + 0.6 * tx / tn
        vz = dz / n + 0.6 * tz / tn
        vn = math.hypot(vx, vz) or 1.0
        step = er + 160.0 - worst_d + 60.0
        if step <= 0.0:
            return None
        x = min(max(40.0, u.x + vx / vn * step), self.map.width - 40.0)
        z = min(max(40.0, u.z + vz / vn * step), self.map.height - 40.0)
        return x, z

    def kite(self, u: UnitInfo) -> bool:
        """Long-ranged units step back from shorter-ranged attackers."""
        role = units.role_of(u.data)
        if role not in ("skirm", "arty", "heavyarty", "snipe", "support"):
            return False
        my_r = weapon_range(u)
        if my_r < 380.0:
            return False
        best = None
        bd = 1e18
        for e in self.enemies.values():
            if e.is_air or not self.is_dangerous(e) or not e.is_mobile:
                continue
            d = self.dist(u.pos, e.pos)
            if d < bd:
                best, bd = e, d
        if best is None or bd > my_r:
            return False
        er = weapon_range(best)
        if er >= my_r * 0.88:
            return False
        if bd > er + 90.0:
            return False
        dx, dz = u.x - best.x, u.z - best.z
        length = math.hypot(dx, dz) or 1.0
        back = my_r * 0.9 - bd + 40.0
        p = (
            min(max(40.0, u.x + dx / length * back), self.map.width - 40.0),
            min(max(40.0, u.z + dz / length * back), self.map.height - 40.0),
        )
        last = getattr(self, "_kite_frame", {})
        self._kite_frame = last
        if self.frame - last.get(u.uid, -10**9) < 20:
            return True
        last[u.uid] = self.frame
        self.queue({"type": "move", "unitId": int(u.uid), "pos": [p[0], 0.0, p[1]]})
        self.unit_order[u.uid] = ("move", p[0], p[1], None, self.frame)
        return True

    def army_center(self) -> tuple[float, float] | None:
        if not self.army:
            return None
        xs = [self.units[uid].x for uid in self.army if uid in self.units]
        zs = [self.units[uid].z for uid in self.army if uid in self.units]
        if not xs:
            return None
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def retreat_point(self, u: UnitInfo) -> tuple[float, float]:
        bx, bz = self.base_pos
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        dx, dz = bx - ex, bz - ez
        length = math.hypot(dx, dz) or 1.0
        return bx + dx / length * 140.0, bz + dz / length * 140.0

    def rally_point(self) -> tuple[float, float]:
        if getattr(self, "sectors", None):
            return self.frontier_rally()
        bx, bz = self.base_pos
        ex, ez = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        return bx + (ex - bx) * 0.22, bz + (ez - bz) * 0.22

    def hunt_target(
        self, center: tuple[float, float], power: float
    ) -> tuple[float, float] | None:
        """Attack mode with nothing worth attacking: the army used to stand
        at the rally point for 25+ minutes at 20x the enemy's power while a
        few hidden structures kept the game alive.  Go for any known ground
        enemy, else sweep map points we have not visited recently."""
        if center is None:
            return None
        cache = getattr(self, "_power_cache", (0, 0.0, 0.0))
        if power < 2.0 * cache[2] + 1500.0:
            return None
        black = getattr(self, "attack_blacklist", {})
        best, bd = None, 1e18
        for e in self.enemies.values():
            if e.is_air:
                continue
            if (int(e.x // 200), int(e.z // 200)) in black:
                continue
            d = self.dist(center, e.pos)
            if d < bd:
                best, bd = e.pos, d
        if best is not None:
            return best
        for e, _f in getattr(self, "enemy_structures", {}).values():
            if (int(e.x // 200), int(e.z // 200)) in black:
                continue
            d = self.dist(center, e.pos)
            if d < bd:
                best, bd = e.pos, d
        if best is not None:
            return best
        visited = getattr(self, "_hunt_visited", None)
        if visited is None:
            visited = self._hunt_visited = {}
        pts = self.search_points()
        if not pts:
            return None
        for q in pts:
            if self.dist(q, center) < 500.0:
                visited[(int(q[0]), int(q[1]))] = self.frame
        ref = self.enemy_base or (self.map.width * 0.5, self.map.height * 0.5)
        for q in pts:
            seen = visited.get((int(q[0]), int(q[1])), -10**9)
            if self.frame - seen < 9000:
                continue
            if (int(q[0] // 200), int(q[1] // 200)) in black:
                continue
            d = self.dist(center, q) + 0.5 * self.dist(ref, q)
            if d < bd:
                best, bd = q, d
        return best

    def search_points(self) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        w, h = self.map.width, self.map.height
        if w <= 0 or h <= 0:
            return pts
        # Dense enough that a cloaked commander is physically bumped into:
        # cloak is only revealed by proximity in Zero-K, so sparse waypoints
        # can miss a hiding commander forever.
        step = 400.0
        gx = step * 0.5
        while gx < w:
            gz = step * 0.5
            while gz < h:
                if self.dist((gx, gz), self.base_pos) > 1600.0:
                    pts.append((gx, gz))
                gz += step
            gx += step
        if self.enemy_base:
            ex, ez = self.enemy_base
            pts.insert(0, (ex, ez))
            for dx, dz in ((-220, 0), (220, 0), (0, -220), (0, 220)):
                pts.insert(1, (ex + dx, ez + dz))
        for s in self.map.spots:
            if self.dist((s.x, s.z), self.base_pos) > 1800:
                pts.append((s.x, s.z))
        return pts

    def group_attack_target(
        self, center: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if center is None:
            center = self.rally_point()
        # Prefer the enemy commander if it is close to our army or our base.
        ec = self.enemies.get(self.enemy_commander_id) if self.enemy_commander_id else None
        if ec is not None and ec.is_finished and not self.enemy_commander_dead:
            if self.dist(ec.pos, center) < 2200.0 or self.dist(ec.pos, self.base_pos) < 1800.0:
                return ec.pos
        if self.enemy_commander_id is not None and not self.enemy_commander_dead:
            remembered = self.enemy_memory.get(self.enemy_commander_id)
            if remembered and self.frame - remembered[1] < 1800:
                return remembered[0].pos
        # Otherwise hit the most valuable enemy thing near the army.
        best = None
        best_score = -1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            score = 0.0
            if not self.enemy_commander_dead and (
                "dyntrainer" in e.name or "commander" in e.name
            ):
                score += 1500.0
            if e.is_factory:
                score += 1200.0
            if e.is_builder:
                score += 800.0
            if e.name in ("staticmex", "staticcon"):
                score += 300.0
            if units.is_energy(e.data):
                score += 220.0
            if e.is_combat:
                score += 100.0
            d = self.dist(center, e.pos)
            score -= d * 0.20
            if score > best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        if self.enemy_army_pos and self.frame - self.enemy_army_frame < 600:
            return self.enemy_army_pos
        return self.enemy_base

    def pick_attack_target(self, u: UnitInfo) -> tuple[float, float] | None:
        best = None
        best_score = -1e18
        for e in self.enemies.values():
            if not e.is_finished:
                continue
            score = 0.0
            if e.is_factory:
                score += 500.0
            if e.name in ("staticmex", "staticcon"):
                score += 250.0
            if units.is_energy(e.data):
                score += 180.0
            if e.is_combat:
                score += 80.0
            score -= self.dist(u.pos, e.pos) * 0.15
            if score > best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        if self.enemy_army_pos and self.frame - self.enemy_army_frame < 600:
            return self.enemy_army_pos
        return self.enemy_base

    def pick_raid_target(
        self, u: UnitInfo, exclude: list[tuple[float, float]] | None = None
    ) -> tuple[float, float] | None:
        best = None
        best_score = 1e18
        black = getattr(self, "raid_blacklist", {}) if exclude is not None else {}
        exclude = exclude or []

        def skip(pos: tuple[float, float]) -> bool:
            if (int(pos[0] // 250), int(pos[1] // 250)) in black:
                return True
            return any(self.dist(pos, q) < 600.0 for q in exclude)

        for e in self.enemies.values():
            if not e.is_finished or skip(e.pos):
                continue
            d = self.dist(u.pos, e.pos)
            value = 0.0
            if e.name in ("staticmex", "staticcon"):
                value = 350.0
            elif e.is_factory:
                value = 500.0
            elif units.is_energy(e.data):
                value = 250.0
            elif e.is_builder:
                value = 250.0
            else:
                continue
            score = d - value
            if score < best_score:
                best_score = score
                best = e.pos
        if best is not None:
            return best
        # Nothing is visible yet.  Send the raiders toward the enemy base so
        # they actually find the expansion/army instead of idling at rally.
        # This is how CircuitAI uses its opening raiders, and Loss-of-Sight
        # contacts are exactly what our own FOV/radar need.
        if self.enemy_base:
            ex, ez = self.enemy_base
            # Prefer the enemy-side metal spots.  This sends raiders into the
            # enemy economy even when no extractor is currently in LOS.
            side = [
                s
                for s in self.map.spots
                if s.status != "ours"
                and math.hypot(s.x - ex, s.z - ez) < 1900.0
                and not skip((s.x, s.z))
            ]
            if side:
                side.sort(key=lambda s: math.hypot(s.x - u.x, s.z - u.z))
                return side[0].x, side[0].z
            return self.enemy_base
        return None

    def order_fight(self, uid: int, pos: tuple[float, float]) -> None:
        self._order(uid, "fight", pos)

    def order_move(self, uid: int, pos: tuple[float, float]) -> None:
        self._order(uid, "move", pos)

    def order_attack(self, uid: int, target_id: int) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == "attack" and old[3] == target_id and self.frame - old[4] < 120:
            return
        self.queue({"type": "attack", "unitId": int(uid), "targetId": int(target_id)})
        self.unit_order[uid] = ("attack", 0.0, 0.0, int(target_id), self.frame)

    def order_attack_ground(self, uid: int, pos: tuple[float, float]) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == "attack_ground":
            if math.hypot(old[1] - pos[0], old[2] - pos[1]) < 150 and self.frame - old[4] < 120:
                return
        self.queue(
            {
                "type": "custom",
                "unitId": int(uid),
                "cmdId": 20,
                "params": [float(pos[0]), 0.0, float(pos[1])],
            }
        )
        self.unit_order[uid] = (
            "attack_ground",
            float(pos[0]),
            float(pos[1]),
            None,
            self.frame,
        )

    def _order(self, uid: int, typ: str, pos: tuple[float, float]) -> None:
        old = self.unit_order.get(uid)
        if old and old[0] == typ:
            ox, oz = old[1], old[2]
            if math.hypot(ox - pos[0], oz - pos[1]) < 150 and self.frame - old[4] < 90:
                return
        self.queue({"type": typ, "unitId": int(uid), "pos": [float(pos[0]), 0.0, float(pos[1])]})
        self.unit_order[uid] = (typ, float(pos[0]), float(pos[1]), None, self.frame)

    # ------------------------------------------------------------- commander

    def commander_radius(self) -> float:
        """How far from home the commander may work.  In the opening it is
        the main expander (and beats small raider groups); later it stays
        closer to the base."""
        if "staticcomm" in FLAGS:
            return 850.0
        span = self.dist(self.base_pos, self.enemy_base or self.map.center)
        reach = 1600.0 if self.frame_time() < 600 else 1150.0
        return max(850.0, min(reach, 0.30 * span))

    def ensure_commander_safe(self) -> None:
        if not self.commander:
            return
        u = self.commander
        # Hard-code the safe radius around home, not around "wherever the
        # commander happens to be".  CircuitAI regularly sends fast shield
        # raiders around the front to kill constructors and the commander.
        if self.dist(u.pos, self.base_pos) > self.commander_radius() + 150.0:
            if self.frame - self.last_commander_retreat_frame < 30:
                return
            self.last_commander_retreat_frame = self.frame
            target = self.retreat_point(u)
            self.queue(
                {"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]]}
            )
            self.say(
                "commander_return",
                f"commander returning home from {tuple(round(v) for v in u.pos)}",
            )
            return
        danger = None
        best = 1e18
        for e in self.enemies.values():
            if not self.is_dangerous(e):
                continue
            d = self.dist(u.pos, e.pos)
            if d < 1000 and d < best:
                best = d
                danger = e
        if danger is not None and best < 900:
            nearby = [
                e
                for e in self.enemies.values()
                if self.is_dangerous(e) and self.dist(u.pos, e.pos) < 900
            ]
            is_enemy_commander = "dyntrainer" in danger.name or "commander" in danger.name
            # The commander (420 range, ~4k HP, repairs itself) beats a small
            # raider group outright; fleeing from two Kodachis gave up the
            # map and the expansion.  Fight while the raiders' combined
            # power, net of nearby friends, is within its own.
            small_group = False
            if "staticcomm" not in FLAGS and not is_enemy_commander:
                heavy = any(
                    units.role_of(e.data) not in ("raid", "scout", "skirm")
                    or "dyntrainer" in e.name
                    or "commander" in e.name
                    for e in nearby
                )
                theirs = sum(combat_power(e) for e in nearby)
                ours_near, _t = self.local_balance(u.pos, 600.0)
                mine = combat_power(u)
                small_group = (
                    not heavy
                    and u.alive_frac > 0.40
                    and theirs <= 1.4 * mine + 0.8 * ours_near
                )
            if best < (650 if small_group else 450) and (
                small_group
                or (not is_enemy_commander and len(nearby) <= 1 and u.alive_frac > 0.75)
                or (
                    is_enemy_commander
                    and len(nearby) <= 1
                    and u.alive_frac > 0.7
                    and u.health > danger.health * 1.3
                )
            ):
                if self.frame - self.last_commander_retreat_frame < 45:
                    return
                self.last_commander_retreat_frame = self.frame
                if danger.is_finished:
                    self.order_attack(u.uid, danger.uid)
                else:
                    self.order_fight(u.uid, danger.pos)
                self.say(
                    "commander_defend",
                    f"commander defending against {danger.name} at "
                    f"{tuple(round(v) for v in danger.pos)}",
                )
                return
            if self.frame - self.last_commander_retreat_frame < 60:
                return
            self.last_commander_retreat_frame = self.frame
            target = self.retreat_point(u)
            self.queue(
                {"type": "move", "unitId": int(u.uid), "pos": [target[0], 0.0, target[1]]}
            )
            self.say(
                "commander_danger",
                f"commander retreating from enemy at {tuple(round(v) for v in danger.pos)}",
            )

    # ----------------------------------------------------------------- summary
