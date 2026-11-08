"""How a side's fighters fight, when the policy doesn't control them: the army script.

It plays every script-controlled side ("army" in `SimEnv`).  It replaced a raider-only fight script, which is now only the frame
data's (frame_world.py `FightScript`).  Each fighter fights by its catalog role.

  raid     in squads of three, which go for the enemy's extractors first and then its other buildings, one target at a time.  A
           target is taken only when the squad's metal is at least the metal of what covers it.  Defences and units that outrange the
           squad count four times over, and a riot heavy near a target rules it out.  A squad with no target probes towards the
           enemy's start, as far as the ground it reaches is clear, and holds where it is when none of it is.  Squads that probe
           close together become one.  Past the first three raiders, a quarter patrol our extractors, and from 1:30 at least two do.
           They answer any threat to an extractor, and no one else.
  assault  leads the army, which waits at the rally for a group worth enough metal, as the old script did.
  riot     and any other role hold 150 elmos behind the assaults, or behind the goal when there are none.
  skirm    the same, but backs off at once from an enemy armed unit that's close to its range.
  arty     stands at three quarters of its range from the enemy building nearest the lead, and shells it.
  aa       follows the army's centre.
  snipe    goes for the nearest slow heavy: the sniper's job.
  scout    moves from waypoint to waypoint between the two starts.

Orders are only given when they're needed.  A unit still moving to a goal that hasn't moved far isn't given it again, since units
visibly stop each time they're re-ordered.  It's given one when it's idle away from its goal, when the goal moves, or when its job changes.

Production is in policy_mix.py.  Each change of what a factory builds is a PRODUCE order on its uid, in the orders returned and in
`productions`, so that a recorder can read every choice per decision.  A silo makes the missile that hurts most
(`ScriptData.best_missile`).  Missiles and the Trinity are aimed by `policy_eco.Aimer`, when the script is given one: they're the
policy's units, so the script's aims are labels.  Constructors and buildings are the economy script's (policy_eco.py).
"""
from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot.policy import featurize as fz  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402

from policy_mix import Mix  # noqa: E402
from policy_script import (  # noqa: E402
    ARRIVED, GROUPS_FROM, LONGEST_WAIT, NEW, RECOVERED, RETREAT_BELOW, RETREATING, SENT, WAITING, ScriptData, group_metal,
)

FIRST_STRAIGHT = 3          # the first raiders go straight for the enemy: only later ones patrol
PATROL_SHARE = 0.25         # of the raiders
PATROL_MIN = 2              # and at least this many from PATROL_MIN_FROM, if there are that many to spare
PATROL_MIN_FROM = 90.0      # seconds: 1:30
PATROL_REACH = 900.0        # elmos from an extractor: a threat this near is answered
PATROL_OFFSET = 150.0       # elmos towards the enemy from an extractor, where a patroller waits
SQUAD_SIZE = 3
COMMIT = 1.0                # a squad goes at a target whose cover is no more than this share of its own metal
DEFENCE_MARGIN = 100.0      # elmos beyond a defence's range that it still covers a target
THREAT_REACH = 500.0        # elmos: an enemy armed unit this near a target covers it
OUTRANGE_MARGIN = 50.0      # elmos: a defence or unit reaching this much past the squad's range outranges it
OUTRANGE_WEIGHT = 4.0       # what an outranging cover counts for, against one that doesn't
HEAVY_RIOT_REACH = 800.0    # elmos: raiders never go this near a riot heavy
PROBE = 300.0               # elmos towards the enemy's start that a probing squad goes at most, in steps of PROBE_STEP
PROBE_STEP = 100.0
TRAVEL_SPEED = 20.0             # elmos a second a squad is allowed to cover on its way to a target, which it can take a while over: it stops to fight
GIVE_UP_SLACK = 60.0            # seconds a squad may take over and above that
PATIENCE_AT_TARGET = 300.0      # seconds it may take to kill a target it has reached
GIVE_UP_FOR = 300.0             # seconds a target given up is left alone by every squad
NEAR_TARGET = 300.0             # elmos: a squad this near its target has reached it
MERGE_REACH = 900.0         # elmos: probing squads this close become one
REORDER = 200.0             # elmos a goal must move before a unit is given it again
AT_GOAL = 150.0             # elmos from an idle unit's goal, where it's left be
BEHIND = 150.0              # elmos behind the lead that riots and the others hold
KITE_RANGE = 0.6            # of a skirmisher's range: an enemy armed unit this near makes it back off
KITE_STEP = 250.0           # elmos a skirmisher backs off at once
ARTY_STAND = 0.75           # of an artillery unit's range from the building it shells
SCOUT_LEG = 250.0           # elmos from a waypoint at which a scout turns to the next
FAR = 1e5                   # sorts extractors ahead of other buildings when choosing a target

Order = tuple[int, float, float, int]       # (kind, x, z, target uid or -1)


def _along(frm: tuple[float, float], to: tuple[float, float], dist: float) -> tuple[float, float]:
    """The point `dist` from `frm` towards `to`, which is `frm` itself when they're the same point."""
    d = math.hypot(to[0] - frm[0], to[1] - frm[1])
    if d < 1e-6:
        return float(frm[0]), float(frm[1])
    return float(frm[0] + (to[0] - frm[0]) / d * dist), float(frm[1] + (to[1] - frm[1]) / d * dist)


@dataclass
class View:
    """The enemy as the script sees it at this decision, one entry each: what's a building, what's a threat, and so on."""
    uid: np.ndarray
    x: np.ndarray
    z: np.ndarray
    cost: np.ndarray
    reach: np.ndarray           # its range
    building: np.ndarray        # standing and not a defence: what raiders go for
    defence: np.ndarray
    threat: np.ndarray          # a mobile unit that can shoot
    heavy_riot: np.ndarray
    slow_heavy: np.ndarray
    extractor: np.ndarray

    def __post_init__(self) -> None:
        self.index = {int(u): i for i, u in enumerate(self.uid)}


def view_of(data: ScriptData, enemies: np.ndarray) -> View:
    t = data.table
    rows = t.rows(enemies["def_id"])
    known = rows != t.unknown
    mobile = t.mobile[rows]
    defence = data.defence[rows]
    return View(
        uid=enemies["uid"].astype(np.int64), x=enemies["x"].astype(float), z=enemies["z"].astype(float),
        cost=t.cost[rows].astype(float), reach=t.range[rows].astype(float),
        building=known & ~mobile & ~defence, defence=known & defence, threat=known & mobile & t.armed[rows],
        heavy_riot=data.heavy_riot[rows], slow_heavy=data.slow_heavy[rows], extractor=data.extractor[rows],
    )


def _cover(view: View, x: np.ndarray, z: np.ndarray, reach: float, armies: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """For each of these points: the cost of the defences and armed units covering it, those that outrange a squad of this reach
    counting for more, and whether a riot heavy is near it.  Without `armies`, mobile units don't count: a squad probing for
    defences goes on past an army, which the fight itself deals with, and stops only at what doesn't move."""
    dist = np.hypot(view.x[None, :] - x[:, None], view.z[None, :] - z[:, None])
    weight = view.cost * np.where(view.reach > reach + OUTRANGE_MARGIN, OUTRANGE_WEIGHT, 1.0)
    covered = view.defence[None, :] & (dist <= view.reach[None, :] + DEFENCE_MARGIN)
    if armies:
        covered = covered | (view.threat[None, :] & (dist <= THREAT_REACH))
    cover = (covered * weight[None, :]).sum(axis=1)
    heavy = (view.heavy_riot[None, :] & (dist <= HEAVY_RIOT_REACH)).any(axis=1)
    return cover, heavy


def _nearest(mask: np.ndarray, view: View, p, keep=lambda d: True) -> int | None:
    """The index of the masked enemy nearest a point, if `keep` takes its distance."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    d = np.hypot(view.x[idx] - p[0], view.z[idx] - p[1])
    j = int(np.argmin(d))
    return int(idx[j]) if keep(float(d[j])) else None


def _waypoints(rally: tuple[float, float], enemy_start: tuple[float, float]) -> list[tuple[float, float]]:
    """Where scouts go: along the way between the starts, to either side of it and down the middle."""
    a, b = np.array(rally, float), np.array(enemy_start, float)
    v = b - a
    side = np.array([-v[1], v[0]]) * 0.3
    return [tuple(a + v * 0.3 + side), tuple(a + v * 0.55 - side), tuple(a + v * 0.8 + side), tuple(a + v * 0.55)]


class ArmyScript:
    """One side of one game.  `orders` takes the side's own records and the enemy's, as `FightScript` does."""

    def __init__(self, data: ScriptData, rally: tuple[float, float], enemy_start: tuple[float, float], opening_row: int | None = None,
                 aimer=None):
        self.data = data
        self.aimer = aimer                      # policy_eco.Aimer: where missiles and the Trinity go
        self.rally = rally
        self.enemy_start = enemy_start
        self.opening_row = opening_row          # the type the factory is on when the opening hands over, if it has one
        self.mix = Mix(data)
        self.productions: list[tuple[int, int]] = []    # (factory uid, table row) this decision: what a recorder reads
        self.state: dict[int, int] = {}
        self.ordered: dict[int, tuple[int, float, float]] = {}      # uid -> (kind, x, z) of its last order
        self.seen: set[int] = set()
        self.squad_of: dict[int, int] = {}
        self.squads: dict[int, int | None] = {}     # squad -> the enemy building it's going for, None when it's probing
        self.since: dict[int, tuple[float, float]] = {}     # squad -> when it took its target, and how far off that was
        self.given_up: dict[int, float] = {}        # enemy building uid -> the time until which no squad goes at it
        self.next_squad = 0
        self.patrol: set[int] = set()
        self.way: dict[int, int] = {}
        self.current: dict[int, int] = {}           # factory uid -> the table row it's on
        self.waypoints = _waypoints(rally, enemy_start)
        self.waiting_since: float | None = None
        self.reached_start = False

    # Orders.

    def orders(self, own: np.ndarray, enemies: np.ndarray, seconds: float, metal_income: float = 0.0,
               seen: np.ndarray | None = None) -> np.ndarray:
        """Orders for this side, given all of its own records and all of the enemy's.  Production orders come first.  metal_income
        sets how many constructors the side wants, as well as the time.  seen: the enemy uids in sight or on radar, for aiming
        missiles and the Trinity."""
        data = self.data
        rows = data.table.rows(own["def_id"])
        built = (own["flags"] & rf.BEING_BUILT) == 0
        fights = data.fights[rows]
        fighting = built & fights
        extracting = built & data.extractor[rows]
        self.mix.observe(enemies)
        self.productions = []
        self._births(own[fights], rows[fights])
        self._forget(set(own["uid"].tolist()))
        out = self._produce(own, rows, built, seconds, metal_income)
        extractors = list(zip(own["x"][extracting].astype(float).tolist(), own["z"][extracting].astype(float).tolist()))
        if fighting.any():
            out += self._fight(own[fighting], rows[fighting], view_of(data, enemies), extractors, seconds)
        if self.aimer is not None and seen is not None:
            out += self.aimer.orders(own, enemies, seen, seconds)
        return np.array(out, zksim.ORDER_DTYPE) if out else np.zeros(0, zksim.ORDER_DTYPE)

    def _births(self, own: np.ndarray, rows: np.ndarray) -> None:
        """Each fighter that first shows up has been started, and the mix counts it."""
        for uid, row in zip(own["uid"].tolist(), rows.tolist()):
            if uid not in self.seen:
                self.seen.add(uid)
                self.mix.record(self.data.role_of[row])

    def _forget(self, alive: set[int]) -> None:
        for table in (self.state, self.ordered, self.way, self.squad_of, self.current):
            for uid in [u for u in table if u not in alive]:
                del table[uid]
        self.patrol &= alive
        used = set(self.squad_of.values())
        for sid in [s for s in self.squads if s not in used]:
            del self.squads[sid]
            self.since.pop(sid, None)

    def _produce(self, own: np.ndarray, rows: np.ndarray, built: np.ndarray, seconds: float, metal_income: float = 0.0) -> list[tuple]:
        """A PRODUCE on each finished factory whose next unit is due to change.  Nothing is ordered for a factory that's on what it should be."""
        out = []
        t = self.data.table
        is_factory = built & t.factory[rows]
        constructors = sum(1 for r in rows.tolist() if self.data.role_of[r] == "builder" and t.mobile[r])
        # What a factory is on, if we haven't told it: what the frame says it's making (the opening's choice, for the first), or
        # nothing, for one just built, which then has to be told.
        making = np.where(own["ord_kind"] == rf.PRODUCE, t.rows(own["ord_target"].astype(np.int64)), -1)
        # A PRODUCE given in the last few decisions holds, as it does for the policy (featurize's commit lock), so that every change
        # the script makes is one the policy could have made.
        held = ((own["ord_kind"] == rf.PRODUCE) & (own["ord_age"] < fz.COMMIT_DECISIONS * fz.DECISION_FRAMES)
                & ((own["flags"] & (rf.DAMAGED | rf.IDLE)) == 0)).tolist()
        for k in np.flatnonzero(is_factory).tolist():
            uid, row, frame_on = int(own["uid"][k]), int(rows[k]), int(making[k])
            on = self.current.get(uid, frame_on if frame_on >= 0 and frame_on != t.unknown else None)
            due = self.mix.choose(row, seconds, constructors, metal_income)
            if due is None or due == on or held[k]:
                if on is not None:
                    self.current[uid] = on
                continue
            self.current[uid] = due
            self.productions.append((uid, due))
            out.append((uid, rf.PRODUCE, 0.0, 0.0, int(self.data.catalog_row[due])))
        for i in np.flatnonzero(built & ~t.factory[rows]):
            due = self.data.best_missile.get(int(rows[i]))
            uid = int(own["uid"][i])
            if due is None or self.current.get(uid) == due or held[i]:
                continue
            self.current[uid] = due
            self.productions.append((uid, due))
            out.append((uid, rf.PRODUCE, 0.0, 0.0, int(self.data.catalog_row[due])))
        return out

    def _fight(self, own: np.ndarray, rows: np.ndarray, view: View, extractors: list[tuple[float, float]], seconds: float) -> list[tuple]:
        data, t = self.data, self.data.table
        uids = own["uid"].tolist()
        roles = [data.role_of[r] for r in rows.tolist()]
        hurt = own["hp_frac"].astype(float)
        pos = np.stack([own["x"], own["z"]], axis=1).astype(float)
        idle = ((own["flags"] & rf.IDLE) != 0).tolist()
        cost = dict(zip(uids, t.cost[rows].astype(float).tolist()))
        reach = dict(zip(uids, t.range[rows].astype(float).tolist()))
        where = {u: pos[k] for k, u in enumerate(uids)}
        retreat: dict[int, bool] = {}
        for k, uid in enumerate(uids):
            # Raiders and scouts never wait for a group.  The army waits from GROUPS_FROM on, as the old script's fighters did.
            back = SENT if roles[k] in ("raid", "scout") or seconds < GROUPS_FROM else WAITING
            s = self.state.get(uid, NEW)
            if s == NEW:
                s = back
            if data.regenerates[rows[k]] and hurt[k] < RETREAT_BELOW:
                s = RETREATING
            elif s == RETREATING and hurt[k] >= RECOVERED:
                s = back
            self.state[uid] = s
            retreat[uid] = s == RETREATING
        self._release([u for k, u in enumerate(uids) if self.state[u] == WAITING and roles[k] not in ("raid", "scout")], cost, seconds)

        sent = [k for k, u in enumerate(uids) if self.state[u] == SENT]
        centre = pos[sent].mean(axis=0) if sent else np.array(self.rally, float)
        if not self.reached_start and sent:
            ex, ez = self.enemy_start
            self.reached_start = bool((np.hypot(pos[sent, 0] - ex, pos[sent, 1] - ez) < ARRIVED).any())
        goal = self._goal(view, centre)
        army_sent = [k for k in sent if roles[k] not in ("raid", "scout")]
        # The others hold behind where the army is heading.  That's the assaults' middle when there are assaults, and the goal when
        # there aren't: holding behind the middle of the others themselves would walk the whole army back to the rally.
        lead = goal
        assault = [k for k in army_sent if roles[k] == "assault"]
        if assault:
            lead = (float(pos[assault, 0].mean()), float(pos[assault, 1].mean()))

        orders: dict[int, Order] = {}
        for k, uid in enumerate(uids):
            if roles[k] not in ("raid", "scout"):
                orders[uid] = self._army_order(roles[k], self.state[uid], pos[k], reach[uid], view, goal, lead, centre)
            elif roles[k] == "scout":
                orders[uid] = self._scout(uid, pos[k])
        raiders = [uids[k] for k in range(len(uids)) if roles[k] == "raid"]
        orders.update(self._raids(raiders, where, cost, reach, retreat, view, extractors, seconds))

        out: list[tuple] = []
        for k, uid in enumerate(uids):
            if uid in orders:
                self._emit(uid, orders[uid], pos[k], idle[k], out)
        return out

    def _release(self, waiting: list[int], cost: dict[int, float], seconds: float) -> None:
        """The army's group goes once the waiting are worth enough, or have waited too long.  As in policy_script.py."""
        if not waiting:
            self.waiting_since = None
            return
        if self.waiting_since is None:
            self.waiting_since = seconds
        if sum(cost[u] for u in waiting) >= group_metal(seconds) or seconds - self.waiting_since >= LONGEST_WAIT:
            for u in waiting:
                self.state[u] = SENT
            self.waiting_since = None

    def _goal(self, view: View, centre: np.ndarray) -> tuple[float, float]:
        """Where the army is sent: the enemy's start until it has got there, then the enemy building nearest it, then any enemy."""
        if self.reached_start:
            for mask in (view.building, view.threat):
                i = _nearest(mask, view, centre)
                if i is not None:
                    return float(view.x[i]), float(view.z[i])
        return self.enemy_start

    def _army_order(self, role: str, state: int, p: np.ndarray, reach: float, view: View, goal, lead, centre) -> Order:
        """What one army fighter should do."""
        rally = (float(self.rally[0]), float(self.rally[1]))
        if state == RETREATING:
            return rf.MOVE, rally[0], rally[1], -1
        if state != SENT:
            return rf.FIGHT, rally[0], rally[1], -1
        behind = _along(lead, rally, min(BEHIND, math.hypot(rally[0] - lead[0], rally[1] - lead[1])))
        if role == "assault":
            return rf.FIGHT, float(goal[0]), float(goal[1]), -1
        if role == "aa":
            return rf.FIGHT, float(centre[0]), float(centre[1]), -1
        if role == "skirm":
            near = _nearest(view.threat, view, p, lambda d: d < KITE_RANGE * reach)
            if near is not None:
                away = _along((float(p[0]), float(p[1])), (view.x[near], view.z[near]), -KITE_STEP)
                return rf.MOVE, away[0], away[1], -1
        if role == "arty":
            building = _nearest(view.building, view, lead)
            if building is not None:
                stand = _along((view.x[building], view.z[building]), rally, ARTY_STAND * reach)
                return rf.FIGHT, stand[0], stand[1], -1
        if role == "snipe":
            heavy = _nearest(view.slow_heavy, view, p)
            if heavy is not None:
                return rf.ATTACK_UNIT, float(view.x[heavy]), float(view.z[heavy]), int(view.uid[heavy])
        return rf.FIGHT, behind[0], behind[1], -1

    def _scout(self, uid: int, p: np.ndarray) -> Order:
        i = self.way.get(uid, uid % len(self.waypoints))
        if math.hypot(p[0] - self.waypoints[i][0], p[1] - self.waypoints[i][1]) < SCOUT_LEG:
            i = (i + 1) % len(self.waypoints)
        self.way[uid] = i
        return rf.MOVE, float(self.waypoints[i][0]), float(self.waypoints[i][1]), -1

    def _emit(self, uid: int, order: Order, p: np.ndarray, idle: bool, out: list[tuple]) -> None:
        """Gives the order if it's new, or the unit is idle away from its goal, or its goal has moved far."""
        kind, x, z, target = order
        last = self.ordered.get(uid)
        if last is not None and last[0] == kind and math.hypot(last[1] - x, last[2] - z) <= REORDER:
            if not idle or math.hypot(p[0] - x, p[1] - z) <= AT_GOAL:
                return
        self.ordered[uid] = (kind, x, z)
        out.append((uid, kind, float(x), float(z), int(target)))

    # Raiders.

    def _raids(self, raiders: list[int], where: dict, cost: dict, reach: dict, retreat: dict, view: View,
               extractors: list[tuple[float, float]], seconds: float) -> dict[int, Order]:
        """Each raider's order.  The squads, their targets and the patrol are kept between decisions."""
        alive = sorted(raiders)
        want = self._patrol_count(len(alive), seconds, len(extractors))
        self.patrol &= set(alive)
        for u in sorted(self.patrol)[want:]:
            self.patrol.discard(u)
        for u in alive[FIRST_STRAIGHT:]:
            if len(self.patrol) >= want:
                break
            self.patrol.add(u)
        for u in [u for u in self.squad_of if u in self.patrol or u not in alive]:
            del self.squad_of[u]
        for u in alive:
            if u not in self.patrol and not retreat[u] and u not in self.squad_of:
                self.squad_of[u] = self._squad_with_room()

        members: dict[int, list[int]] = {}
        for u, sid in sorted(self.squad_of.items()):
            members.setdefault(sid, []).append(u)
        centre: dict[int, np.ndarray] = {}
        metal: dict[int, float] = {}
        strength: dict[int, float] = {}
        for sid, mem in members.items():
            metal[sid] = sum(cost[u] for u in mem)
            strength[sid] = max(reach[u] for u in mem)
            active = [u for u in mem if not retreat[u]]
            if active:
                centre[sid] = np.mean([where[u] for u in active], axis=0)

        # A squad keeps its target while it's still worth going at and it's getting there, and the others pick from what's left: no two
        # squads on one building, and none that a squad has given up on lately.
        for uid in [u for u, until in self.given_up.items() if until <= seconds]:
            del self.given_up[uid]
        targets: dict[int, int | None] = {}
        for sid in sorted(centre):
            kept = self.squads.get(sid)
            if kept is None or kept in targets.values() or not self._acceptable(kept, metal[sid], strength[sid], view):
                continue
            if self._patient(sid, kept, centre[sid], seconds, view):
                targets[sid] = kept
            else:
                self.given_up[kept] = seconds + GIVE_UP_FOR
        for sid in sorted(centre):
            if sid not in targets:
                targets[sid] = self._choose(centre[sid], metal[sid], strength[sid], view, set(targets.values()) | set(self.given_up))
                if targets[sid] is not None and targets[sid] != self.squads.get(sid):
                    i = view.index[targets[sid]]
                    self.since[sid] = (seconds, math.hypot(view.x[i] - centre[sid][0], view.z[i] - centre[sid][1]))
        for sid in centre:
            self.squads[sid] = targets[sid]

        # Probing squads close together become one, and go on together: no more than two squads' worth, or they'd be one blob.
        probing = sorted(sid for sid in centre if self.squads[sid] is None)
        for i, a in enumerate(probing):
            for b in probing[i + 1:]:
                if a in members and b in members and len(members[a]) + len(members[b]) <= 2 * SQUAD_SIZE and math.hypot(*(centre[a] - centre[b])) <= MERGE_REACH:
                    for u in members.pop(b):
                        self.squad_of[u] = a
                        members[a].append(u)
                    self.squads.pop(b, None)
                    centre.pop(b, None)
                    metal[a] += metal.pop(b)
                    strength[a] = max(strength[a], strength.pop(b))

        orders: dict[int, Order] = {}
        for sid, mem in members.items():
            if sid not in centre:
                continue
            target = self.squads[sid]
            if target is not None:
                i = view.index[target]
                goal = (float(view.x[i]), float(view.z[i]))
            else:
                goal = self._probe(centre[sid], metal[sid], strength[sid], view)
            for u in mem:
                if not retreat[u]:
                    orders[u] = (rf.FIGHT, goal[0], goal[1], -1)
        orders.update(self._patrol_orders(view, sorted(self.patrol), extractors, retreat))
        for u in alive:
            if retreat[u]:
                orders[u] = (rf.MOVE, float(self.rally[0]), float(self.rally[1]), -1)
        return orders

    def _patrol_count(self, raiders: int, seconds: float, extractors: int) -> int:
        if not extractors:
            return 0
        want = math.ceil(PATROL_SHARE * raiders)
        if seconds >= PATROL_MIN_FROM:
            want = max(want, PATROL_MIN)
        return min(want, max(0, raiders - FIRST_STRAIGHT))

    def _squad_with_room(self) -> int:
        counts: dict[int, int] = {}
        for sid in self.squad_of.values():
            counts[sid] = counts.get(sid, 0) + 1
        for sid in sorted(counts):
            if counts[sid] < SQUAD_SIZE:
                return sid
        self.next_squad += 1
        return self.next_squad

    def _acceptable(self, uid: int, metal: float, reach: float, view: View) -> bool:
        """Whether a squad this strong, with this reach, should still go at this building."""
        i = view.index.get(uid)
        if i is None or not view.building[i]:
            return False
        cover, heavy = _cover(view, view.x[[i]], view.z[[i]], reach)
        return bool(not heavy[0] and cover[0] <= metal / COMMIT)

    def _patient(self, sid: int, uid: int, centre: np.ndarray, seconds: float, view: View) -> bool:
        """Whether a squad has had long enough to get to its target, given how far it was, or to kill it once there."""
        i = view.index[uid]
        near = math.hypot(view.x[i] - centre[0], view.z[i] - centre[1]) <= NEAR_TARGET
        started, distance = self.since.get(sid, (seconds, 0.0))
        allowed = GIVE_UP_SLACK + distance / TRAVEL_SPEED + (PATIENCE_AT_TARGET if near else 0.0)
        return seconds - started <= allowed

    def _choose(self, origin: np.ndarray, metal: float, reach: float, view: View, taken: set) -> int | None:
        """The enemy building a squad should go for: extractors first, then the nearest it can take, not one another squad has."""
        cand = np.array([i for i in np.flatnonzero(view.building) if int(view.uid[i]) not in taken], dtype=int)
        if not len(cand):
            return None
        cover, heavy = _cover(view, view.x[cand], view.z[cand], reach)
        ok = ~heavy & (cover <= metal / COMMIT)
        if not ok.any():
            return None
        dist = np.hypot(view.x[cand] - origin[0], view.z[cand] - origin[1])
        key = np.where(ok, dist + np.where(view.extractor[cand], 0.0, FAR), np.inf)
        return int(view.uid[cand[int(np.argmin(key))]])

    def _probe(self, centre: np.ndarray, metal: float, reach: float, view: View) -> tuple[float, float]:
        """Where a squad with no target goes: towards the enemy's start, as far as the ground there is covered no more than its
        metal.  Where every step of it is covered, it holds, and prods from there."""
        frm = (float(centre[0]), float(centre[1]))
        dist = math.hypot(self.enemy_start[0] - frm[0], self.enemy_start[1] - frm[1])
        steps = np.arange(min(PROBE, dist), 0.0, -PROBE_STEP)
        if not len(steps):
            return frm
        points = [_along(frm, self.enemy_start, float(s)) for s in steps]
        px = np.array([p[0] for p in points])
        pz = np.array([p[1] for p in points])
        cover, heavy = _cover(view, px, pz, reach, armies=False)
        clear = ~heavy & (cover <= metal / COMMIT)
        return points[int(np.argmax(clear))] if clear.any() else frm

    def _patrol_orders(self, view: View, patrollers: list[int], extractors: list[tuple[float, float]], retreat: dict[int, bool]) -> dict[int, Order]:
        """Each patroller waits part way from its extractor to the enemy, and goes to any threat near that extractor."""
        out: dict[int, Order] = {}
        threats = np.stack([view.x[view.threat], view.z[view.threat]], axis=1) if view.threat.any() else np.zeros((0, 2))
        for j, u in enumerate(patrollers):
            if retreat[u] or not extractors:
                continue
            ex, ez = extractors[j % len(extractors)]
            point = _along((ex, ez), self.enemy_start, min(PATROL_OFFSET, math.hypot(self.enemy_start[0] - ex, self.enemy_start[1] - ez)))
            if len(threats):
                d = np.hypot(threats[:, 0] - ex, threats[:, 1] - ez)
                k = int(np.argmin(d))
                if d[k] <= PATROL_REACH:
                    point = (float(threats[k, 0]), float(threats[k, 1]))
            out[u] = (rf.FIGHT, point[0], point[1], -1)
        return out


def make_script(kind: str, data: ScriptData, rally: tuple[float, float], enemy_start: tuple[float, float], opening_row: int | None = None,
                aimer=None):
    """The script of a kind: "army", this module's, the only one now."""
    if kind == "army":
        return ArmyScript(data, rally, enemy_start, opening_row, aimer)
    raise ValueError(f"no script called {kind!r}")
