"""RaidMixin: slow-ball engagement and persistent raid squads.

Slow ball (user guidance): a Buoy + Grizzly style army of slow units is
beaten by one large group of raiders and Ronins that commits together, not
by small squads trickling in.  A Grizzly (1900 M, 8700 HP) needs ~30 raiders
to die before it retreats, so the commit test compares metal value; the
sqrt(dps*hp) power score would send ~13.

Raid squads: persistent membership, new raiders join the nearest squad,
nearby squads merge, several squads combine for a prize (commander,
factory) that is too well defended for one, and squads route through our
fresh expansions on the way so they are close when an enemy raid arrives.
"""
from __future__ import annotations

import math
import os

from . import units
from .models import UnitInfo
from .threat import weapon_range

FLAGS = os.environ.get("ZKBOT_FLAGS", "")

SLOW_SPEED = 60.0          # Buoy 51, Grizzly 45, Ogre 69 is borderline-fast
BALL_MIN_VALUE = 800.0
BALL_RADIUS = 600.0

_BALL_WEIGHT = {
    "skirm": 1.2,   # Ronins keep firing while the ball retreats
    "raid": 1.0,
    "assault": 1.0,
    "riot": 0.7,    # slow and outranged by Buoys
    "arty": 0.8,
    "aa": 0.3,
}


class RaidMixin:
    # ------------------------------------------------------------ slow ball

    def enemy_slow_ball(self) -> dict | None:
        """The densest cluster of slow enemy combat units, if it dominates
        the enemy army around it.  Cached for 30 frames."""
        cache = getattr(self, "_ball_cache", None)
        if cache is not None and self.frame - cache[0] < 30:
            return cache[1]
        res = None
        if "noball" not in FLAGS:
            pool: dict[int, UnitInfo] = {}
            for uid, (e, f) in getattr(self, "enemy_memory", {}).items():
                if self.frame - f <= 1800:
                    pool[uid] = e
            for e in self.contacts().values():
                pool[e.uid] = e
            army = [
                e
                for e in pool.values()
                if e.is_mobile
                and not e.is_air
                and not e.is_builder
                and e.data
                and self.is_dangerous(e)
            ]
            slow = [e for e in army if 0.0 < float(e.data.get("speed") or 0.0) < SLOW_SPEED]
            best = None
            best_v = 0.0
            for s in slow:
                v = sum(
                    units.cost(o.data)
                    for o in slow
                    if self.dist(o.pos, s.pos) < BALL_RADIUS
                )
                if v > best_v:
                    best, best_v = s, v
            members = (
                [o for o in slow if self.dist(o.pos, best.pos) < BALL_RADIUS]
                if best is not None
                else []
            )
            # Only balls anchored by an expensive slow heavy (Grizzly,
            # Cyclops, Sumo...).  Counting any cluster of cheap slow units
            # switched shield/spider/cloak games to the ball mix too and
            # lost more than it won (8/12 vs 10/12).
            heavy = any(units.cost(o.data) >= 600.0 for o in members)
            if best is not None and best_v >= BALL_MIN_VALUE and (heavy or "anyball" in FLAGS):
                around = sum(
                    units.cost(o.data) for o in army if self.dist(o.pos, best.pos) < 1000.0
                )
                if best_v >= 0.5 * around:
                    cx = sum(o.x * units.cost(o.data) for o in members) / best_v
                    cz = sum(o.z * units.cost(o.data) for o in members) / best_v
                    res = {
                        "pos": (cx, cz),
                        "value": around,
                        "range": max(weapon_range(o) for o in members),
                        "count": len(members),
                        "speed": max(float(o.data.get("speed") or 0.0) for o in members),
                    }
        self._ball_cache = (self.frame, res)
        return res

    def ball_value_of(self, uids) -> float:
        total = 0.0
        for uid in uids:
            u = self.units.get(uid)
            if u is None:
                continue
            w = _BALL_WEIGHT.get(units.role_of(u.data), 0.8)
            total += units.cost(u.data) * w * max(0.3, u.alive_frac)
        return total

    def ball_joiners(self, raider_uids: list[int], ball: dict | None) -> set[int]:
        """Raiders that fight the slow ball with the main group: all of them
        up to 8, else the 60% closest to the ball.  The rest keep raiding."""
        if ball is None or not raider_uids:
            return set()
        if len(raider_uids) <= 8:
            return set(raider_uids)
        n = max(8, math.ceil(0.6 * len(raider_uids)))
        ordered = sorted(
            raider_uids, key=lambda uid: self.dist(self.units[uid].pos, ball["pos"])
        )
        return set(ordered[:n])

    def ball_engagement(
        self, ball: dict, group: list[int], group_center: tuple[float, float]
    ) -> tuple[str, tuple[float, float]]:
        """("commit", ball position) or ("gather", gather point)."""
        ours = self.ball_value_of(group)
        committed = getattr(self, "_ball_committed", False)
        need = 0.7 if committed else 1.0
        near = sum(
            1
            for uid in group
            if uid in self.units and self.dist(self.units[uid].pos, group_center) < 800.0
        )
        cohesive = not group or near >= 0.6 * len(group)
        commit = ours >= need * ball["value"] and (cohesive or committed)
        if commit != committed:
            self.log(
                f"slow ball {'COMMIT' if commit else 'gather'}: ours={ours:.0f} vs "
                f"ball={ball['value']:.0f} ({ball['count']} slow units) at "
                f"{tuple(round(v) for v in ball['pos'])}"
            )
        self._ball_committed = commit
        if commit:
            return "commit", ball["pos"]
        bx, bz = ball["pos"]
        dx, dz = group_center[0] - bx, group_center[1] - bz
        n = math.hypot(dx, dz) or 1.0
        stand = ball["range"] + 300.0
        x = min(max(60.0, bx + dx / n * stand), self.map.width - 60.0)
        z = min(max(60.0, bz + dz / n * stand), self.map.height - 60.0)
        return "gather", (x, z)

    # ------------------------------------------------------------ raid squads

    def _squad_center(self, members) -> tuple[float, float] | None:
        return self.median_pos([uid for uid in members if uid in self.units])

    def raid_route_via(
        self, start: tuple[float, float], target: tuple[float, float]
    ) -> tuple[float, float] | None:
        """One of our fresh extractors or a working constructor that lies
        nearly on the way: pass through it so the squad is close when an
        enemy raid arrives there."""
        if "noroute" in FLAGS:
            return None
        direct = self.dist(start, target)
        if direct < 900.0:
            return None
        ref = self.enemy_base or self.map.center
        best = None
        best_score = 1e18
        for u in self.units.values():
            fresh_mex = (
                u.name == "staticmex"
                and u.is_finished
                and u.finished_frame >= 0
                and self.frame - u.finished_frame < 3600
            )
            con = u.is_builder and u.is_mobile and u.uid != self.commander_id
            if not (fresh_mex or con):
                continue
            via = self.dist(start, u.pos) + self.dist(u.pos, target)
            if via > min(direct * 1.25, direct + 600.0):
                continue
            if self.dist(start, u.pos) < 350.0 or self.dist(u.pos, target) < 600.0:
                continue
            # The most exposed asset (closest to the enemy) first.
            score = self.dist(u.pos, ref)
            if score < best_score:
                best, best_score = u.pos, score
        return best

    def _raid_prize(self) -> tuple[tuple[float, float], float, str] | None:
        """High-value target worth combining squads for."""
        ec = None
        if getattr(self, "enemy_commander_id", None) and not getattr(
            self, "enemy_commander_dead", False
        ):
            ec = self.enemies.get(self.enemy_commander_id)
            if ec is None:
                mem = self.enemy_memory.get(self.enemy_commander_id)
                if mem and self.frame - mem[1] < 450:
                    ec = mem[0]
        if ec is not None:
            return ec.pos, 1500.0, "commander"
        best = None
        for e, _f in getattr(self, "enemy_structures", {}).values():
            if e.is_factory:
                d = self.dist(e.pos, self.base_pos)
                if best is None or d < best[1]:
                    best = (e.pos, d)
        if best is not None:
            return best[0], 1000.0, "factory"
        return None

    def update_raid_squads_v2(
        self,
        raider_uids: list[int],
        stage: tuple[float, float],
        group_needed: int,
        out: dict[int, tuple[tuple[float, float], bool]],
    ) -> None:
        t = self.frame_time()
        cap = 4 if t < 300 else (6 if t < 900 else 8)
        squads: list[dict] = self.__dict__.setdefault("raid_squads", [])
        black = self.__dict__.setdefault("raid_blacklist", {})
        for k, until in list(black.items()):
            if until < self.frame:
                black.pop(k, None)
        alive = set(raider_uids)
        for sq in squads:
            sq["members"] &= alive
        squads[:] = [sq for sq in squads if sq["members"]]
        assigned = set().union(*(sq["members"] for sq in squads)) if squads else set()
        # New raiders join the nearest squad with room, else found one.
        for uid in sorted(alive - assigned):
            u = self.units[uid]
            best = None
            bd = 1e18
            for sq in squads:
                c = self._squad_center(sq["members"])
                if c is None or len(sq["members"]) >= cap:
                    continue
                d = self.dist(c, u.pos)
                if d < bd:
                    best, bd = sq, d
            if best is not None and (bd < 1400.0 or len(squads) >= 3):
                best["members"].add(uid)
            elif len(squads) < 3:
                squads.append({"members": {uid}, "target": None, "tframe": -10**9,
                               "via": None, "gather": None, "gframe": 0})
            else:
                min(squads, key=lambda s: len(s["members"]))["members"].add(uid)
        # Merge squads that are close (keep the larger squad's target).
        merged = True
        while merged and len(squads) > 1:
            merged = False
            for i in range(len(squads)):
                for j in range(i + 1, len(squads)):
                    a, b = squads[i], squads[j]
                    ca, cb = self._squad_center(a["members"]), self._squad_center(b["members"])
                    if ca is None or cb is None:
                        continue
                    if self.dist(ca, cb) < 700.0 and len(a["members"]) + len(b["members"]) <= cap * 1.5:
                        keep, drop = (a, b) if len(a["members"]) >= len(b["members"]) else (b, a)
                        keep["members"] |= drop["members"]
                        squads.remove(drop)
                        self.log(
                            f"raid squads merged -> {len(keep['members'])} at "
                            f"{tuple(round(v) for v in ca)}"
                        )
                        merged = True
                        break
                if merged:
                    break
        # Combine for a prize no single squad may take.
        prize = self._raid_prize() if "nocombine" not in FLAGS else None
        combine_ids: set[int] = set()
        if prize is not None and len(squads) > 1:
            ppos, pworth, pkind = prize
            defended = self.raid_defense_value(ppos)
            singles = [self.raid_accept_value(list(sq["members"])) for sq in squads]
            near = [
                (sq, acc)
                for sq, acc in zip(squads, singles)
                if (c := self._squad_center(sq["members"])) is not None
                and self.dist(c, ppos) < 3000.0
            ]
            total = sum(acc for _sq, acc in near)
            # The prize's own worth buys some extra risk.
            if defended > max(singles) and total + 0.25 * pworth >= defended and len(near) > 1:
                for sq, _acc in near:
                    combine_ids.add(id(sq))
                    if sq["target"] is None or self.dist(sq["target"], ppos) > 300.0:
                        sq["target"] = ppos
                        sq["tframe"] = self.frame
                        sq["via"] = None
                        sq["gather"] = None
                if self.frame - getattr(self, "_combine_log", -10**9) > 600:
                    self._combine_log = self.frame
                    self.log(
                        f"raid squads combine x{len(near)} on {pkind} at "
                        f"{tuple(round(v) for v in ppos)} (defended {defended:.0f}, "
                        f"combined {total:.0f})"
                    )
        taken: list[tuple[float, float]] = []
        for i, sq in enumerate(squads):
            members = sorted(sq["members"])
            if len(members) < group_needed and id(sq) not in combine_ids:
                continue
            center = self._squad_center(members) or stage
            tgt = sq["target"]
            if tgt is not None and id(sq) not in combine_ids:
                defended = self.raid_defense_value(tgt)
                accept = self.raid_accept_value(members)
                hot = defended > accept
                if hot:
                    black[(int(tgt[0] // 250), int(tgt[1] // 250))] = self.frame + 2400
                    self.log(
                        f"raid squad {i} target {tuple(round(v) for v in tgt)} too well "
                        f"defended ({defended:.0f} vs {accept:.0f}); retargeting"
                    )
                if hot or self.frame - sq["tframe"] > 600 or self.dist(tgt, self.base_pos) < 450.0:
                    tgt = None
            if tgt is None:
                old = sq["target"]
                tgt = self.choose_raid_group_target(members, exclude=taken)
                sq["target"], sq["tframe"] = tgt, self.frame
                sq["via"] = self.raid_route_via(center, tgt) if tgt is not None else None
                if tgt is not None and (old is None or self.dist(old, tgt) > 200.0):
                    via = sq["via"]
                    self.log(
                        f"raid squad {i} ({len(members)}) -> {tuple(round(v) for v in tgt)}"
                        + (f" via {tuple(round(v) for v in via)}" if via else "")
                    )
            if tgt is None:
                continue
            taken.append(tgt)
            dest = tgt
            if id(sq) in combine_ids:
                # Gather outside the prize's defence, strike together.
                if sq["gather"] is None:
                    dx, dz = center[0] - tgt[0], center[1] - tgt[1]
                    n = math.hypot(dx, dz) or 1.0
                    sq["gather"] = (tgt[0] + dx / n * 750.0, tgt[1] + dz / n * 750.0)
                    sq["gframe"] = self.frame
                ready = all(
                    (c := self._squad_center(o["members"])) is not None
                    and self.dist(c, o["gather"] or c) < 450.0
                    for o in squads
                    if id(o) in combine_ids
                )
                if not ready and self.frame - sq["gframe"] < 900:
                    dest = sq["gather"]
            elif sq["via"] is not None:
                if self.dist(center, sq["via"]) < 350.0:
                    sq["via"] = None
                else:
                    dest = sq["via"]
            for uid in members:
                u = self.units.get(uid)
                if u is None:
                    continue
                at = self.dist(u.pos, stage) < 900.0 or self.dist(u.pos, dest) < 1000.0
                out[uid] = (dest, at or self.dist(u.pos, dest) < self.dist(stage, dest))
