"""Raiders first, and raiders that go looking (user review, spectated game).

Flag `raidfirst`: the opening is raiders and constructors.  The fixed mix used to
queue a Reaver once a few Glaives existed, whatever the enemy had, and counted
Fleas as raiders to answer.  Riots and assaults phase in after RAID_ONLY_UNTIL,
or at once if the enemy shows something raiders can't handle.

Flag `raidprobe`: a squad that reaches an empty target moves on at once, working
through unchecked points toward the enemy spawn.  Only a turret whose range
reaches a target counts against it, so targets in a defence's dead zone stay
open, and routes bend around turret coverage on the way there.

Flag `hitrun`, used with `raidprobe`: raids move rather than fight on the way in,
then shoot what can't shoot back and leave as soon as the local defence is more
than half the squad's metal.  A riot rules out a target whatever else is there.
Hurt raiders in a large squad go back the way it came and wander until healed.

Flag `spearhead`: the first raiders built never come back to defend.

Flag `seeturrets`: count turrets in view as defence.  They were missed, because
they aren't `is_dangerous`, while being marked seen so memory skipped them too.
`raidprobe` includes this.
"""
from __future__ import annotations

import math

from . import units
from .models import UnitInfo
from .params import FLAGS, P
from .threat import weapon_range

RAID_ONLY_UNTIL = P("raidfirst.until", 300.0, 120.0, 600.0, "seconds of raiders-only opening")
RAMP_SECONDS = P("raidfirst.ramp", 300.0, 60.0, 600.0, "seconds for riots and assaults to phase in")
RIOT_TRIGGER = P("raidfirst.riot_trigger", 6, 3, 15, "real enemy raiders that end the opening early")
SLOW_TRIGGER = P("raidfirst.slow_trigger", 3, 1, 10, "enemy riots and assaults seen that end the opening early")

ACCEPT_SHARE = P("raidprobe.accept", 1.0, 0.5, 2.0, "defence value a squad accepts, as a share of its own metal")
RANGE_MARGIN = P("raidprobe.margin", 40.0, 0.0, 200.0, "elmos added to a turret's range")
ARRIVED = P("raidprobe.arrived", 250.0, 100.0, 500.0, "squad this close has reached its target")
EMPTY_RADIUS = P("raidprobe.empty_radius", 400.0, 200.0, 800.0, "nothing to shoot within this of a target means empty")
CHECKED = P("raidprobe.checked", 350.0, 150.0, 700.0, "a squad this close has looked at a point")
RECHECK_FRAMES = P("raidprobe.recheck", 1800, 300, 5400, "frames before a checked point is worth another look")
INWARD = P("raidprobe.inward", 0.7, 0.0, 2.0, "pull toward the enemy spawn when choosing the next point")
STAGE_SHARE = P("raidprobe.stage_share", 0.55, 0.2, 0.8, "where raiders rally, as a share of the way to the enemy")
BLACKLIST_FRAMES = P("raidprobe.blacklist", 900, 150, 2400, "frames a defended target is left alone")
STAND_OFF = P("raidprobe.stand_off", 120.0, 40.0, 300.0, "elmos outside a defence's range when poking at it")
ROUTE_CLEARANCE = P("raidprobe.route_clearance", 150.0, 50.0, 400.0, "elmos outside coverage when routing around it")

RAID_SHARE = P("hitrun.accept", 0.5, 0.25, 1.0, "defence value a squad accepts, as a share of its metal: 0.5 means 2x")
RIOT_VETO = 1e6               # defence value of a riot: no squad accepts it
RIOT_MEMORY = P("hitrun.riot_memory", 300, 0, 1800, "frames a riot out of sight still rules out a target")
CAP_EARLY = P("hitrun.cap_early", 6, 2, 12, "squad size before minute 5")
CAP_MID = P("hitrun.cap_mid", 8, 2, 16, "squad size before minute 15")
CAP_LATE = P("hitrun.cap_late", 10, 2, 20, "squad size after minute 15")
STRIKE_RADIUS = P("hitrun.strike_radius", 450.0, 200.0, 800.0, "squad this close to its target starts shooting")
ARMED_RADIUS = P("hitrun.armed_radius", 350.0, 100.0, 700.0, "armed enemy this close: fight it, not the economy")
BREAK_RADIUS = P("hitrun.break_radius", 650.0, 300.0, 1200.0, "defence counted around the squad itself")
BREAK_FRAMES = P("hitrun.break_frames", 150, 30, 600, "frames a squad runs after breaking off")
REGEN_GROUP = P("hitrun.regen_group", 6, 2, 20, "squad size from which hurt raiders peel off")
REGEN_HP = P("hitrun.regen_hp", 0.5, 0.2, 0.8, "health share below which a raider peels off")
REGEN_FULL = P("hitrun.regen_full", 0.95, 0.6, 1.0, "health share at which it rejoins")
REGEN_TIMEOUT = P("hitrun.regen_timeout", 1800, 300, 5400, "frames before it rejoins anyway, for slow healers")
WAYPOINT_REACHED = P("hitrun.waypoint", 200.0, 80.0, 400.0, "elmos that count as having reached a point")
TRAIL_STEP = P("hitrun.trail_step", 200.0, 80.0, 500.0, "elmos between remembered points on a squad's route")
TRAIL_POINTS = 40
SPEARHEAD = P("spearhead.size", 3, 0, 8, "first raiders built, which never defend")
REGEN_DANGER = P("hitrun.regen_danger", 450.0, 200.0, 900.0, "enemy this close: a healing raider runs for safety")
WANDER_CLEAR = P("hitrun.wander_clear", 600.0, 300.0, 1200.0, "wander only to points with no enemy this close")
WANDER_FROM_SPAWN = P("hitrun.wander_from_spawn", 1200.0, 500.0, 2500.0, "wander no closer than this to the enemy spawn")


def _unit(dx: float, dz: float) -> tuple[float, float]:
    n = math.hypot(dx, dz) or 1.0
    return dx / n, dz / n


def is_ground_turret(e: UnitInfo) -> bool:
    """A turret that can shoot a raider; anti-air ones can't."""
    return not e.is_mobile and units.is_defense(e.data) and not units.is_aa(e.data)


class RaidProbeMixin:
    spearhead: set[int] = frozenset()

    # ------------------------------------------------------------ raidfirst

    def enemy_real_raiders(self) -> int:
        """Enemy raiders in contact, without scouts: Fleas don't call for riots."""
        return sum(1 for e in self.contacts().values() if units.role_of(e.data) == "raid")

    def raid_ramp(self) -> float:
        """Share of the normal riot and assault quota allowed: 0 in the opening, up to 1."""
        if "raidfirst" not in FLAGS or self.armored_heavies():
            return 1.0
        real = self.enemy_real_raiders()
        if real >= RIOT_TRIGGER and real > self.our_role_counts()["raider"]:
            return 1.0
        seen = self.seen_enemy_roles()
        if seen["riot"] + seen["assault"] >= SLOW_TRIGGER:
            return 1.0
        return min(1.0, max(0.0, (self.frame_time() - RAID_ONLY_UNTIL) / RAMP_SECONDS))

    # ----------------------------------------------------------- raidprobe

    def known_defenses(self) -> list[UnitInfo]:
        """Enemy turrets in contact or remembered."""
        out = {e.uid: e for e, _f in getattr(self, "enemy_structures", {}).values()
               if is_ground_turret(e)}
        for e in self.contacts().values():
            if is_ground_turret(e):
                out[e.uid] = e
        return list(out.values())

    def turret_reach(self, e: UnitInfo) -> float:
        return weapon_range(e) + RANGE_MARGIN

    def turret_covers(self, e: UnitInfo, pos: tuple[float, float]) -> bool:
        return self.dist(e.pos, pos) <= self.turret_reach(e)

    def coverage_waypoint(
        self, start: tuple[float, float], dest: tuple[float, float]
    ) -> tuple[float, float] | None:
        """A point around the first turret whose coverage the straight path enters.
        Turrets covering either end are ignored: the squad has accepted those."""
        sx, sz = start
        dx, dz = dest[0] - sx, dest[1] - sz
        length2 = dx * dx + dz * dz
        if length2 < 1.0:
            return None
        first = None
        for e in self.known_defenses():
            reach = self.turret_reach(e)
            if self.dist(e.pos, dest) <= reach or self.dist(e.pos, start) <= reach:
                continue
            t = ((e.x - sx) * dx + (e.z - sz) * dz) / length2
            if not 0.0 < t < 1.0:
                continue
            near = (sx + dx * t, sz + dz * t)
            if self.dist(near, e.pos) < reach and (first is None or t < first[0]):
                first = (t, e, near, reach)
        if first is None:
            return None
        _t, e, near, reach = first
        off = self.dist(near, e.pos)
        if off > 1.0:
            sides = [_unit(near[0] - e.x, near[1] - e.z)]
        else:
            nx, nz = _unit(-dz, dx)
            sides = [(nx, nz), (-nx, -nz)]
        stand = reach + ROUTE_CLEARANCE
        cands = [self._clamp((e.x + a * stand, e.z + b * stand)) for a, b in sides]
        return min(cands, key=self.threat_at)

    def probe_points(self) -> list[tuple[float, float]]:
        """Where to look: metal spots on the enemy's side of the map, and the spawn."""
        if not self.enemy_base:
            return []
        pts = [(s.x, s.z) for s in self.map.spots
               if s.status != "ours"
               and self.dist((s.x, s.z), self.enemy_base) < self.dist((s.x, s.z), self.base_pos)]
        pts.append(self.enemy_base)
        return pts

    def note_points_checked(self, center: tuple[float, float]) -> None:
        checked = self.__dict__.setdefault("raid_checked", {})
        for q in self.probe_points():
            if self.dist(q, center) < CHECKED:
                checked[q] = self.frame

    def probe_target(
        self, squad: list[int], center: tuple[float, float], exclude: list[tuple[float, float]]
    ) -> tuple[float, float] | None:
        """The next point to look at, working inward.  With every point guarded,
        the edge of the least guarded one: a poke shows what's really there."""
        checked = self.__dict__.setdefault("raid_checked", {})
        black = getattr(self, "raid_blacklist", {})
        accept = self.raid_accept_value(squad)
        best, best_score = None, 1e18
        oldest, oldest_frame = self.enemy_base, 10**12
        guarded = []
        for q in self.probe_points():
            if any(self.dist(q, o) < 600.0 for o in exclude):
                continue
            defended = self.raid_defense_value(q)
            if defended > accept or (int(q[0] // 250), int(q[1] // 250)) in black:
                guarded.append((defended, q))
                continue
            looked = checked.get(q, -10**9)
            if looked < oldest_frame:
                oldest, oldest_frame = q, looked
            if self.frame - looked < RECHECK_FRAMES:
                continue
            score = self.dist(center, q) + INWARD * self.dist(q, self.enemy_base)
            if score < best_score:
                best, best_score = q, score
        if best is not None:
            return best
        if oldest_frame < 10**12:
            return oldest
        pokeable = [g for g in guarded if g[0] < RIOT_VETO]
        if pokeable:
            return self.poke_point(min(pokeable)[1], center)
        if guarded:
            return None
        return self.enemy_base

    def poke_point(self, q: tuple[float, float], center: tuple[float, float]) -> tuple[float, float]:
        """Just outside this point's defences, on our side of them.  Stepping out of
        one turret's range can land in another's, so repeat a few times."""
        turrets = self.known_defenses()
        point = q
        for _ in range(6):
            covering = [e for e in turrets if self.turret_covers(e, point)]
            if not covering:
                break
            e = min(covering, key=lambda d: self.dist(d.pos, center))
            a, b = _unit(center[0] - e.x, center[1] - e.z)
            stand = self.turret_reach(e) + STAND_OFF
            point = self._clamp((e.x + a * stand, e.z + b * stand))
        return point

    def target_is_empty(self, tgt: tuple[float, float], center: tuple[float, float]) -> bool:
        """The squad is there and there's nothing to shoot."""
        if self.dist(center, tgt) > ARRIVED:
            return False
        if "hitrun" in FLAGS:
            # Armed units don't hold a raid at a finished target: it fires as it leaves.
            return not self.raid_work_near(tgt, EMPTY_RADIUS)
        if any(not e.is_air and self.dist(e.pos, tgt) < EMPTY_RADIUS for e in self.contacts().values()):
            return False
        return not any(self.dist(e.pos, tgt) < EMPTY_RADIUS
                       for e, _f in getattr(self, "enemy_structures", {}).values())

    # ----------------------------------------------------------- spearhead

    def update_spearhead(self, raider_uids: list[int]) -> set[int]:
        """The first raiders built that are still alive.  The dead aren't replaced:
        the offence share takes over from there."""
        if "spearhead" not in FLAGS:
            return set()
        first = self.__dict__.setdefault("spearhead_first", [])
        for uid in raider_uids:
            if len(first) >= SPEARHEAD:
                break
            if uid not in first:
                first.append(uid)
        self.spearhead = {uid for uid in first if uid in self.units}
        return self.spearhead

    # -------------------------------------------------------------- hitrun

    def is_raid_work(self, e: UnitInfo) -> bool:
        """Something a raid is there for: it can't shoot back."""
        return not e.is_air and not self.is_dangerous(e) and not is_ground_turret(e) and not units.is_aa(e.data)

    def raid_work_near(self, pos: tuple[float, float], radius: float) -> list[UnitInfo]:
        found = {e.uid: e for e in self.enemies.values()
                 if self.is_raid_work(e) and self.dist(e.pos, pos) < radius}
        for e, _f in getattr(self, "enemy_structures", {}).values():
            if self.is_raid_work(e) and self.dist(e.pos, pos) < radius:
                found.setdefault(e.uid, e)
        return list(found.values())

    def strike_target(self, center: tuple[float, float]) -> int | None:
        """What the squad shoots: constructors, then the nearest in view."""
        visible = [e for e in self.enemies.values()
                   if self.is_raid_work(e) and self.dist(e.pos, center) < STRIKE_RADIUS]
        if not visible:
            return None
        return min(visible, key=lambda e: (not e.is_builder, self.dist(e.pos, center))).uid

    def raid_mode(self, center: tuple[float, float], tgt: tuple[float, float]) -> int | str | None:
        """Run (None) until at the target, then "fight" any armed unit there, else
        the uid to shoot.  Moving units still fire at what they pass."""
        if self.dist(center, tgt) > STRIKE_RADIUS:
            return None
        for e in self.contacts().values():
            if e.is_mobile and not e.is_air and self.is_dangerous(e) and self.dist(e.pos, center) < ARMED_RADIUS:
                return "fight"
        return self.strike_target(center)

    def hitrun_order(self, u: UnitInfo, dest: tuple[float, float], mode: int | str | None) -> str:
        if mode == "fight":
            self.order_fight(u.uid, dest)
            return "raid_group"
        if mode is not None and mode in self.enemies:
            self.order_attack(u.uid, mode)
            return "raid_strike"
        self.order_move(u.uid, dest)
        return "raid_run"

    def squad_breaks_off(self, sq: dict, members: list[int], center: tuple[float, float]) -> bool:
        """The defence around the squad is more than it would have accepted."""
        if self.frame - sq.get("broke", -10**9) < BREAK_FRAMES:
            return False
        local = self.raid_defense_value(center, BREAK_RADIUS)
        if local <= self.raid_accept_value(members):
            return False
        sq["broke"] = self.frame
        self.log(f"raid squad ({len(members)}) breaks off at {tuple(round(v) for v in center)}: defence {local:.0f}")
        return True

    def squad_running(self, sq: dict) -> bool:
        return self.frame - sq.get("broke", -10**9) < BREAK_FRAMES

    def note_trail(self, sq: dict, center: tuple[float, float]) -> None:
        trail = sq.setdefault("trail", [])
        if not trail or self.dist(trail[-1], center) >= TRAIL_STEP:
            trail.append(center)
            del trail[:-TRAIL_POINTS]

    def peel_hurt(self, sq: dict, members: list[int]) -> list[int]:
        """Hurt members of a large squad leave to heal, back along its route."""
        if len(members) < REGEN_GROUP:
            return members
        regen = self.__dict__.setdefault("raid_regen", {})
        keep = []
        for uid in members:
            u = self.units.get(uid)
            if u is None or u.alive_frac >= REGEN_HP:
                keep.append(uid)
                continue
            regen[uid] = {"path": list(reversed(sq.get("trail", []))), "since": self.frame, "wander": None}
            sq["members"].discard(uid)
            self.log(f"raider {uid} at {u.alive_frac:.2f} peels off to heal")
        return keep

    def beh_raider_regen(self, u: UnitInfo, p) -> str | None:
        state = self.__dict__.get("raid_regen", {}).get(u.uid)
        if state is None:
            return None
        if u.alive_frac >= REGEN_FULL or self.frame - state["since"] > REGEN_TIMEOUT:
            del self.raid_regen[u.uid]
            self.log(f"raider {u.uid} back at {u.alive_frac:.2f} after {(self.frame - state['since']) / 30:.0f}s")
            return None
        self.retreating.discard(u.uid)
        self.order_move(u.uid, self.regen_destination(u, state))
        return "raider_regen"

    def regen_destination(self, u: UnitInfo, state: dict) -> tuple[float, float]:
        if self.enemy_near(u.pos, REGEN_DANGER) is not None:
            state["wander"] = None
            return self.safe_retreat_point(u)
        path = state["path"]
        while path and (self.dist(u.pos, path[0]) < WAYPOINT_REACHED or self.raid_defense_value(path[0]) > 0.0):
            path.pop(0)
        if path:
            return path[0]
        target = state["wander"]
        if target is None or not self.wander_ok(target) or self.dist(u.pos, target) < WAYPOINT_REACHED:
            if target is not None:
                self.__dict__.setdefault("raid_checked", {})[target] = self.frame
            target = state["wander"] = self.wander_point(u.pos)
        return target or self.rally_point()

    def wander_ok(self, q: tuple[float, float]) -> bool:
        return self.raid_defense_value(q) <= 0.0 and self.enemy_near(q, WANDER_CLEAR) is None

    def wander_point(self, pos: tuple[float, float]) -> tuple[float, float] | None:
        """The nearest metal spot not looked at lately, away from the enemy: a healing
        raider might as well be scouting."""
        checked = self.__dict__.setdefault("raid_checked", {})
        best, best_d = None, 1e18
        for s in self.map.spots:
            q = (s.x, s.z)
            if s.status == "ours" or self.frame - checked.get(q, -10**9) < RECHECK_FRAMES:
                continue
            if self.enemy_base and self.dist(q, self.enemy_base) < WANDER_FROM_SPAWN:
                continue
            d = self.dist(pos, q)
            if d < best_d and self.wander_ok(q):
                best, best_d = q, d
        return best

    def forward_stage(self, stage: tuple[float, float]) -> tuple[float, float]:
        """Rally part way to the enemy, unless the enemy is already there."""
        if not self.enemy_base:
            return stage
        bx, bz = self.base_pos
        ex, ez = self.enemy_base
        out = (bx + (ex - bx) * STAGE_SHARE, bz + (ez - bz) * STAGE_SHARE)
        if (self.dist(stage, self.enemy_base) < self.dist(out, self.enemy_base)
                or self.enemy_near(out, 650.0) is not None):
            return stage
        return out
