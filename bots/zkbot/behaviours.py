"""Per-unit army behaviours (UnitBehaviourMixin).

MilitaryMixin.military_step first builds an ArmyPlan (armyplan.py: posture,
commander focus, search, slow ball, raids, main group, responders), then gives every army unit to the behaviours in
UNIT_BEHAVIOURS, in order.  Each behaviour either issues the unit's order
and returns a branch name (logged in "branches=" and used to attribute
deaths), or returns None to pass the unit on.  So:

  * a new behaviour is one method plus one entry in the list;
  * priorities are visible in one place instead of a 200-line if-chain;
  * an A/B flag can drop or reorder a behaviour without touching the rest.

The order and the logic are the former military_step loop, unchanged,
plus `manual_fire` (fire D-gun type weapons such as the Dante's napalm
salvo; opt out with flag `nomanualfire`).
"""
from __future__ import annotations


from . import catalog, units
from .armyplan import ArmyPlan
from .models import UnitInfo
from .threat import ground_dps, weapon_range

from .params import FLAGS

CMD_MANUALFIRE = 105


# Priority order.  The first behaviour that returns a branch name wins.
UNIT_BEHAVIOURS = (
    "beh_bomb",
    "beh_retreat",
    "beh_air",
    "beh_aa_escort",
    "beh_home_guard",
    "beh_raider_micro",
    "beh_respond",
    "beh_focus_commander",
    "beh_search",
    "beh_raider_squad",
    "beh_phantom",
    "beh_manual_fire",
    "beh_kite",
    "beh_ball_gather",
    "beh_outranged",
    "beh_local_target",
    "beh_regroup",
    "beh_defend",
    "beh_attack",
    "beh_rally",
)


class UnitBehaviourMixin:
    def unit_behaviours(self):
        return [getattr(self, name) for name in UNIT_BEHAVIOURS]

    # Each behaviour: (unit, plan) -> branch name or None.

    def beh_bomb(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        # Cloak bombs are expendable: drive them straight at the most
        # valuable target instead of retreating or forming a firing line.
        if units.role_of(u.data) != "bomb" or u.is_air:
            return None
        target = (
            p.focus.pos
            or p.enemy_army_pos
            or (self.threat_pos if p.posture.mode == "defend" else None)
            or self.group_attack_target(p.army_center)
            or self.enemy_base
            or self.rally_point()
        )
        if target:
            self.order_move(u.uid, target)
        return "bomb"

    def beh_retreat(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        # Retreat only when it is actually worth preserving the unit.
        # Retreat destinations are chosen to avoid enemy contact; a cheap
        # raider with no clean path home is allowed to trade itself.
        uid = u.uid
        if uid in self.retreating and u.alive_frac > (0.70 if "oldretreat" in FLAGS else 0.85):
            self.retreating.discard(uid)
        if uid in self.retreating:
            self.order_move(uid, self.safe_retreat_point(u))
            return "retreating"
        should_retreat, retreat_dest = self.retreat_decision(u)
        if should_retreat and retreat_dest is not None:
            self.retreating.add(uid)
            self.order_move(uid, retreat_dest)
            return "retreat_new"
        self.retreating.discard(uid)
        return None

    def beh_air(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not u.is_air:
            return None
        air_target = p.posture.attack_target if p.posture.mode_eff == "attack" else (
            p.posture.defense_anchor if p.posture.mode_eff == "defend" else None
        )
        self.control_air(u, p.group.center, air_target)
        return "air"

    def beh_aa_escort(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not (ground_dps(u) < 5.0 and (units.is_aa(u.data) or units.role_of(u.data) == "aa")):
            return None
        # Anti-air that cannot shoot ground (Gremlin) walked at the front and
        # died in every ground fight.  Shadow the group and only engage
        # aircraft.
        air_t = None
        for e in self.enemies.values():
            if e.is_air and self.dist(u.pos, e.pos) < weapon_range(u) + 150.0:
                air_t = e
                break
        if air_t is not None:
            self.order_attack(u.uid, air_t.uid)
        else:
            self.order_move(u.uid, p.group.center)
        return "aa_escort"

    def beh_home_guard(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        # The guard holds the base / threatened owned sector while the main
        # force is away.  In defend mode everyone converges, so this branch
        # is intentionally limited to attack.
        if u.uid in self.home_guard and p.posture.mode == "attack":
            self.order_fight(u.uid, p.posture.home_anchor)
            return "guard"
        return None

    def beh_raider_micro(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if (
            "oldmicro" in FLAGS
            or not self.is_raider(u)
            or u.uid in p.ball.joiners
            or p.focus.active
            or p.search.active
            or p.posture.mode == "attack"
        ):
            return None
        if u.uid in p.raids.orders:
            tgt, at_stage = p.raids.orders[u.uid]
            return self.raider_offense_order(u, tgt if at_stage else p.raids.stage)
        return self.raider_defense_order(
            u, p.responders.get(u.uid) or (p.posture.defense_anchor if p.posture.mode == "defend" else None)
        )

    def beh_respond(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if u.uid not in p.responders or p.focus.active:
            return None
        local = self.best_local_target(u)
        if local is not None and self.dist(u.pos, local.pos) < weapon_range(u) + 80.0:
            self.order_attack(u.uid, local.uid)
        else:
            self.order_fight(u.uid, p.responders[u.uid])
        return "respond"

    def beh_focus_commander(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not (p.focus.active and p.focus.pos is not None):
            return None
        ec = p.focus.unit
        if ec is not None and ec.is_finished:
            self.order_attack(u.uid, ec.uid)
        elif u.is_air or units.role_of(u.data) in ("arty", "assault", "riot", "skirm", "heavy"):
            # The commander went out of LOS (often cloaked).  Fire at the
            # last known position; area damage can still finish it.
            self.order_attack_ground(u.uid, p.focus.pos)
        else:
            self.order_fight(u.uid, p.focus.pos)
        return "focus_comm"

    def beh_search(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not (p.search.active and p.search.points):
            return None
        should_search = not p.search.unknown_hunt or u.is_air or self.is_raider(u) or u.uid % 6 == 0
        if not should_search:
            return None
        idx = (u.uid + self.frame // 900) % len(p.search.points)
        self.order_fight(u.uid, p.search.points[idx])
        return "search"

    def beh_raider_squad(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not self.is_raider(u) or u.uid in p.ball.joiners:
            return None
        uid = u.uid
        if uid in p.raids.orders:
            tgt, at_stage = p.raids.orders[uid]
            self.order_fight(uid, tgt if at_stage else p.raids.stage)
            return "raid_group"
        if p.raids.group_active and p.raids.target is not None and "oldraid" in FLAGS:
            self.order_fight(uid, p.raids.target if uid in p.raids.ready else p.raids.stage)
            return "raid_group"
        if p.posture.mode != "attack" and not (
            p.posture.mode == "defend"
            and self.threat_pos is not None
            and self.dist(self.threat_pos, self.base_pos) < 2200.0
        ):
            self.order_fight(uid, p.raids.stage)
            return "raid_stage"
        return None

    def beh_phantom(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if u.name != "cloaksnipe" or "nophantom" in FLAGS or not u.is_finished:
            return None
        if u.uid not in getattr(self, "phantom_radar_off", set()):
            self.note_new_phantom(u)
        return self.phantom_order(u, self.phantom_anchor(p.group.center))

    def beh_manual_fire(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        """Fire a manual-fire weapon (Dante napalm salvo, Scorpion lightning,
        Bantha EMP missile, Detriment blaster, ...) at the best cluster in
        range.  The unit keeps its other order; this only adds the shot, so
        it never claims the unit (returns None) unless it fired."""
        if "nomanualfire" in FLAGS:
            return None
        mw = catalog.manual_weapon(u.name)
        if not mw or not u.is_finished:
            return None
        last = self.__dict__.setdefault("_manual_fire_frame", {})
        reload_f = max(30.0, float(mw.get("reload") or 1.0) * 30.0)
        if self.frame - last.get(u.uid, -10**9) < reload_f:
            return None
        rng = float(mw.get("range") or 0.0)
        if rng <= 0 or rng > 1500:
            return None  # nukes and map-wide weapons are not army weapons
        effects = set(mw.get("effects") or ())
        radius = max(120.0, float(mw.get("aoe") or 0.0))
        best, best_val = None, 0.0
        cands = [e for e in self.enemies.values() if not e.is_air and self.dist(u.pos, e.pos) < rng]
        for e in cands:
            if "disarm" in effects or "emp" in effects:
                # Disable the most valuable unit (and what stands next to it).
                val = units.cost(e.data) + 0.3 * sum(
                    units.cost(o.data) for o in cands if o is not e and self.dist(o.pos, e.pos) < radius
                )
            else:
                val = sum(units.cost(o.data) for o in cands if self.dist(o.pos, e.pos) < radius)
            if val > best_val:
                best, best_val = e, val
        if best is None or best_val < 400.0:
            return None
        last[u.uid] = self.frame
        # The shot replaces the unit's queue: forget the cached order so the
        # next step re-issues it instead of leaving the unit idle.
        self.unit_order.pop(u.uid, None)
        x, z = best.pos
        y = self.map.ground_height(x, z) if hasattr(self.map, "ground_height") else 0.0
        self.queue({"type": "custom", "unitId": int(u.uid), "cmdId": CMD_MANUALFIRE,
                    "params": [float(x), float(y), float(z)]})
        self.log(f"manual fire {u.name}#{u.uid} {mw.get('name')} at {best.name} value={best_val:.0f}")
        return "manual_fire"

    def beh_kite(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if "nokite" not in FLAGS and self.kite(u):
            return "kite"
        return None

    def beh_ball_gather(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if not (p.posture.mode_eff == "ballgather" and p.ball.gather is not None and u.uid in p.group.member_set):
            return None
        # Wait out of the ball's reach until the group is big enough.
        if self.dist(u.pos, p.ball.ball["pos"]) < p.ball.ball["range"] + 350.0:
            self.order_move(u.uid, p.ball.gather)
        else:
            self.order_fight(u.uid, p.ball.gather)
        return "ball_gather"

    def beh_outranged(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if "nooutrange" in FLAGS:
            return None
        away = self.outranged_escape(u)
        if away is None:
            return None
        self.order_move(u.uid, away)
        return "outranged"

    def beh_local_target(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        local = self.best_local_target(u)
        if local is None:
            return None
        role = units.role_of(u.data)
        reach = max(220.0, weapon_range(u) + 80.0)
        if role in ("arty", "heavyarty"):
            reach = weapon_range(u) * 0.95
        near = self.dist(u.pos, local.pos) < reach
        aa_air = (units.is_aa(u.data) or role == "aa") and local.is_air
        # Do not let a unit chase into a fight the group would lose.
        if near and not aa_air and p.posture.mode_eff != "defend":
            ours_l, theirs_l = self.local_balance(local.pos, 600.0)
            if theirs_l > ours_l * 1.3 and self.dist(u.pos, local.pos) > weapon_range(u):
                near = False
        if not (near or aa_air):
            return None
        # Radar blips have no valid attack target id for most weapons;
        # artillery can still shell the location.
        if local.uid in self.radar and role in ("arty", "heavyarty"):
            self.order_attack_ground(u.uid, local.pos)
        else:
            self.order_attack(u.uid, local.uid)
        return "local"

    def beh_regroup(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if (
            p.group.regroup
            and "noregroup" not in FLAGS
            and u.uid in p.group.member_set
            and self.dist(u.pos, p.group.center) > 600.0
        ):
            self.order_move(u.uid, p.group.center)
            return "regroup"
        return None

    def beh_defend(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if p.posture.mode_eff != "defend" or p.posture.defense_anchor is None:
            return None
        # Meet the attack as one group when we can hold it; otherwise fall
        # back onto the turrets instead of feeding units in.
        self.order_fight(u.uid, p.posture.defense_anchor if p.posture.defend_engage else p.posture.hold_point)
        return "defend" if p.posture.defend_engage else "hold"

    def beh_attack(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        if p.posture.mode_eff == "attack" and p.posture.attack_target is not None:
            self.order_fight(u.uid, p.posture.attack_target)
            return "attack"
        return None

    def beh_rally(self, u: UnitInfo, p: ArmyPlan) -> str | None:
        self.order_fight(u.uid, self.rally_point())
        return "rally"
