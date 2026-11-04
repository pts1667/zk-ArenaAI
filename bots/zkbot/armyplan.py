"""Army planning (ArmyPlannerMixin): what the army does this step.

military_step used to compute ~30 local variables in a 300-line preamble
and then branch on them per unit.  The plan is now a composition of small
parts, each built by one planner method in a fixed order:

  CommanderFocus  plan_commander_focus   go for the enemy commander?
  SearchPlan      plan_search            sweep the map for a hidden enemy?
  SlowBallPlan    plan_slow_ball         the enemy's slow death ball, and
                                         which raiders join the main group
  RaidPlan        plan_raids             raid squads, their stage and orders
  MainGroup       plan_main_group        the main group, its centre, cohesion
  Posture         plan_posture           effective mode, anchors, attack
                                         target, whether a defence engages

ArmyPlan bundles them with the responder assignments.  Behaviours
(behaviours.py) read the plan; nothing in it is written after planning,
except the stale-target refresh at the end of plan_army.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import units
from .models import UnitInfo
from .threat import combat_power

from .params import FLAGS

Pos = tuple[float, float]


@dataclass
class CommanderFocus:
    active: bool = False             # the army goes for the enemy commander
    pos: Pos | None = None           # its visible or recently seen position
    unit: UnitInfo | None = None     # the commander, when visible


@dataclass
class SearchPlan:
    active: bool = False             # nothing seen for a while: sweep
    points: list[Pos] = field(default_factory=list)
    unknown_hunt: bool = False       # winning outright, commander hidden


@dataclass
class SlowBallPlan:
    ball: dict[str, Any] | None = None       # enemy_slow_ball() cluster
    joiners: set[int] = field(default_factory=set)  # raiders fighting it
    gather: Pos | None = None                # where to wait until strong enough


@dataclass
class RaidPlan:
    stage: Pos                                # raider rally/stage point
    orders: dict[int, tuple[Pos, bool]] = field(default_factory=dict)  # uid -> (target, at stage)
    modes: dict[int, int | str | None] = field(default_factory=dict)   # hitrun: uid -> run, fight or a target
    group_active: bool = False
    target: Pos | None = None                 # shared target (oldraid only)
    ready: list[int] = field(default_factory=list)


@dataclass
class MainGroup:
    members: list[int]
    member_set: set[int]
    center: Pos
    power: float
    cohesion: int                             # members within 700 of the centre
    regroup: bool


@dataclass
class Posture:
    mode: str                                 # army_mode(): attack/gather/defend
    mode_eff: str                             # after targets/ball: + ballgather
    attack_target: Pos | None
    defense_anchor: Pos | None
    home_anchor: Pos
    hold_point: Pos
    defend_engage: bool


@dataclass
class ArmyPlan:
    posture: Posture
    focus: CommanderFocus
    search: SearchPlan
    ball: SlowBallPlan
    raids: RaidPlan
    group: MainGroup
    responders: dict[int, Pos]
    army_center: Pos | None
    enemy_army_pos: Pos | None


class ArmyPlannerMixin:
    def plan_army(self, mode: str) -> ArmyPlan:
        center = self.army_center()
        focus = self.plan_commander_focus()
        enemy_army_pos = self.nearest_enemy_contact()
        defense_anchor = self.defense_anchor()
        # Keeps the sector state fresh (its anchor is not used here).
        self.best_attack_sector()
        home_anchor = defense_anchor or self.base_pos
        raider_uids = [
            uid for uid in self.army if uid in self.units and self.is_raider(self.units[uid])
        ]
        stage = self.rally_point()
        if self.enemy_near(stage, 650.0) is not None:
            stage = self.base_pos
        if "raidprobe" in FLAGS:
            stage = self.forward_stage(stage)
        search = self.plan_search(mode, focus, enemy_army_pos)
        ball = self.plan_slow_ball(raider_uids)
        raids = self.plan_raids(mode, focus, search, ball, raider_uids, stage)
        group = self.plan_main_group(mode, ball, defense_anchor)
        posture = self.plan_posture(mode, group, ball, defense_anchor, home_anchor)
        self.log_army_plan(posture, group)
        responders = self.assign_responders(mode) if "noresponse" not in FLAGS else {}
        self.update_pincer([
            uid for uid in raider_uids
            if uid not in raids.orders and uid not in ball.joiners
        ] if mode != "attack" and not focus.active else [])
        if (
            posture.attack_target is not None
            and not (ball.ball is not None and posture.attack_target == ball.ball["pos"])
            and self.attack_target_stale(posture.attack_target, group.center)
        ):
            self.choose_main_attack(posture, group)
        return ArmyPlan(
            posture=posture, focus=focus, search=search, ball=ball, raids=raids,
            group=group, responders=responders, army_center=center,
            enemy_army_pos=enemy_army_pos,
        )

    # ------------------------------------------------------------- parts

    def plan_commander_focus(self) -> CommanderFocus:
        enemy_commander = (
            self.enemies.get(self.enemy_commander_id) if self.enemy_commander_id else None
        )
        commander_memory = (
            self.enemy_memory.get(self.enemy_commander_id)
            if self.enemy_commander_id is not None and not self.enemy_commander_dead
            else None
        )
        focus = CommanderFocus()
        if self.enemy_commander_dead:
            enemy_commander = None
        focus.unit = enemy_commander
        if enemy_commander is not None and enemy_commander.is_finished:
            focus.pos = enemy_commander.pos
        elif commander_memory and self.frame - commander_memory[1] < 900:
            focus.pos = commander_memory[0].pos
        if focus.pos is None:
            self.hunt_commander = False
            return focus
        enemy_count = sum(1 for e in self.enemies.values() if self.is_dangerous(e)) + len(self.radar)
        close_to_base = self.dist(focus.pos, self.base_pos) < 2000.0
        weak_commander = enemy_commander is not None and enemy_commander.alive_frac < 0.5
        if self.combat_count() >= (7 if close_to_base else 12) and close_to_base and (
            enemy_count <= self.combat_count() * 1.5 or enemy_count <= 8 or weak_commander
        ):
            focus.active = True
            self.hunt_commander = True
            if self.frame - self.last_attack_frame > 300:
                self.last_attack_frame = self.frame
                hp = ""
                if enemy_commander is not None:
                    hp = f" hp={enemy_commander.health:.0f}/{enemy_commander.max_health:.0f}"
                near = sum(
                    1 for uid in self.army
                    if uid in self.units and self.dist(self.units[uid].pos, focus.pos) < 1200.0
                )
                self.log(
                    f"focus enemy commander at {tuple(round(v) for v in focus.pos)}"
                    f"{hp} army_near={near}"
                )
        return focus

    def nearest_enemy_contact(self) -> Pos | None:
        """The enemy contact (not a builder, except commanders) closest to
        our base."""
        contacts = list(self.enemies.values()) + list(self.radar.values())
        e = min(
            (
                e for e in contacts
                if not e.is_builder or "dyntrainer" in e.name or "commander" in e.name
            ),
            key=lambda e: self.dist(e.pos, self.base_pos),
            default=None,
        )
        return e.pos if e is not None else None

    def defense_anchor(self) -> Pos | None:
        threatened = self.most_threatened_sector()
        if threatened is not None and (
            threatened.distance_base >= 2000.0 or threatened.owner != "ours"
        ):
            threatened = None
        anchor = self.best_defense_pos(threatened)
        return anchor if anchor is not None else self.threat_pos

    def plan_search(self, mode: str, focus: CommanderFocus, enemy_army_pos: Pos | None) -> SearchPlan:
        # If nothing has been visible for a while, the enemy commander may be
        # cloaked or hiding.  Only sweep for it when we are winning outright
        # and nothing has been seen for a full minute (an older rule that
        # searched whenever the commander was unknown after 8 minutes
        # scattered raiders and disabled local engagement).
        unknown_hunt = (
            focus.pos is None
            and mode == "attack"
            and self.combat_count() >= 20
            and self.frame - self.last_enemy_seen_frame > 1800
        )
        active = unknown_hunt or (
            mode != "defend"
            and self.combat_count() >= 8
            and self.frame - self.last_enemy_seen_frame > 600
            and enemy_army_pos is None
        )
        return SearchPlan(
            active=active,
            points=self.search_points() if active else [],
            unknown_hunt=unknown_hunt,
        )

    def plan_slow_ball(self, raider_uids: list[int]) -> SlowBallPlan:
        ball = self.enemy_slow_ball()
        return SlowBallPlan(ball=ball, joiners=self.ball_joiners(raider_uids, ball))

    def plan_raids(
        self,
        mode: str,
        focus: CommanderFocus,
        search: SearchPlan,
        ball: SlowBallPlan,
        raider_uids: list[int],
        stage: Pos,
    ) -> RaidPlan:
        self.raid_modes = {}
        raids = RaidPlan(stage=stage, modes=self.raid_modes)
        limit = 3 if self.frame_time() < 420 else 5
        squad = raider_uids[:limit]
        group_needed = 2 if self.frame_time() < 300 or "fastgather" in FLAGS else 3
        free_to_raid = not focus.active and not search.active and (mode != "attack" or "raidprobe" in FLAGS)
        if "oldraid" in FLAGS:
            raids.group_active = (
                len(squad) >= group_needed
                and free_to_raid
                and not (
                    mode == "defend"
                    and self.threat_pos is not None
                    and self.dist(self.threat_pos, self.base_pos) < 2200.0
                )
            )
            if raids.group_active:
                self.plan_shared_raid(raids, squad)
            return raids
        # Every raider raids, in up to three squads with separate targets.
        # (Only the first 3-5 raided before; the rest idled at the rally
        # point while a human held the map with a few Kodachis.)
        squad_raiders = [uid for uid in raider_uids if uid not in ball.joiners]
        spear = self.update_spearhead(squad_raiders)
        patrol = self.update_mex_patrol(squad_raiders, spear)
        squad_raiders = [uid for uid in squad_raiders if uid not in patrol]
        update = self.update_raid_squads if "oldsquads" in FLAGS else self.update_raid_squads_v2
        raids.group_active = (
            len(squad_raiders) >= group_needed
            and free_to_raid
            and not (
                mode == "defend"
                and self.threat_pos is not None
                and self.dist(self.threat_pos, self.base_pos) < 1400.0
            )
        )
        if raids.group_active:
            update(squad_raiders, stage, group_needed, raids.orders)
            raids.group_active = bool(raids.orders)
        elif free_to_raid or (("raidprobe" in FLAGS or spear) and not focus.active and not search.active):
            # At least one squad keeps raiding while the rest defend, or attack.
            offense = self.offensive_raiders(squad_raiders, self.threat_pos)
            if spear:
                offense = sorted(set(offense) | spear)
            needed = 1 if spear else group_needed
            if len(offense) >= needed:
                update(offense, stage, needed, raids.orders)
        return raids

    def plan_shared_raid(self, raids: RaidPlan, squad: list[int]) -> None:
        """The old single shared raid target (flag oldraid)."""
        if (
            self.raid_group_target is None
            or self.frame - self.raid_group_frame > 600
            or self.dist(self.raid_group_target, self.base_pos) < 450.0
        ):
            self.raid_group_target = self.choose_raid_group_target(squad)
            self.raid_group_frame = self.frame
            if self.raid_group_target is not None:
                self.log(
                    f"raid group {len(squad)} -> {tuple(round(v) for v in self.raid_group_target)}"
                )
        raids.target = self.raid_group_target
        if raids.target is None:
            raids.group_active = False
            return
        raids.ready = [
            uid for uid in squad
            if uid in self.units
            and (
                self.dist(self.units[uid].pos, raids.stage) < 900.0
                or self.dist(self.units[uid].pos, raids.target) < 1000.0
            )
        ]
        self.raid_group_size = len(raids.ready)

    def plan_main_group(self, mode: str, ball: SlowBallPlan, defense_anchor: Pos | None) -> MainGroup:
        members = [
            uid for uid in self.army
            if uid in self.units
            and (not self.is_raider(self.units[uid]) or uid in ball.joiners)
            and units.role_of(self.units[uid].data) != "bomb"
            and not self.units[uid].is_air
            and not (uid in self.home_guard and mode == "attack")
            and uid not in self.retreating
        ]
        center = self.median_pos(members) or self.rally_point()
        gather_radius, gather_share = self.gather_terms()
        cohesion = sum(1 for uid in members if self.dist(self.units[uid].pos, center) < gather_radius)
        # Away from the base a defence is fought as one group too; feeding
        # units in one by one lost even fights.  At the base, everyone goes.
        far_defence = (
            "defregroup" in FLAGS
            and mode == "defend"
            and defense_anchor is not None
            and self.dist(defense_anchor, self.base_pos) > 1200.0
        )
        regroup = self.regroup_allowed(
            (mode in ("attack", "gather") or far_defence)
            and len(members) >= 4
            and cohesion < gather_share * len(members)
        )
        return MainGroup(
            members=members,
            member_set=set(members),
            center=center,
            power=sum(combat_power(self.units[uid]) for uid in members),
            cohesion=cohesion,
            regroup=regroup,
        )

    def plan_posture(
        self,
        mode: str,
        group: MainGroup,
        ball: SlowBallPlan,
        defense_anchor: Pos | None,
        home_anchor: Pos,
    ) -> Posture:
        posture = Posture(
            mode=mode,
            mode_eff=mode,
            attack_target=None,
            defense_anchor=defense_anchor,
            home_anchor=home_anchor,
            hold_point=self.fallback_point(defense_anchor),
            defend_engage=True,
        )
        if mode == "defend" and defense_anchor is not None:
            posture.defend_engage = self.can_hold_defence(group, ball, defense_anchor)
        if mode == "attack":
            self.choose_main_attack(posture, group)
        b = ball.ball
        if b is not None and mode in ("attack", "gather") and (
            self.dist(b["pos"], group.center) < 2500.0
            or (posture.attack_target is not None and self.dist(b["pos"], posture.attack_target) < 1300.0)
        ):
            how, where = self.ball_engagement(b, group.members, group.center)
            if how == "commit":
                posture.attack_target = where
                posture.mode_eff = "attack"
            else:
                ball.gather = where
                posture.mode_eff = "ballgather"
        elif b is None:
            self._ball_committed = False
        return posture

    def can_hold_defence(self, group: MainGroup, ball: SlowBallPlan, anchor: Pos) -> bool:
        """Meet an attack only with group + turrets >= 0.8x the local enemy
        (or, against the slow ball, more value than the ball)."""
        _o, theirs = self.local_balance(anchor, 900.0)
        turrets = [
            t for t in self.units.values()
            if units.is_defense(t.data)
            and not t.is_mobile
            and t.is_finished
            and self.dist(t.pos, anchor) < 900.0
        ]
        engage = group.power + sum(combat_power(t) for t in turrets) >= 0.8 * theirs
        if "fightnet" in FLAGS and theirs > 0:
            from .fightnet import HOLD_EXCH
            ev = self.fight_eval_moved(anchor, group.members)
            if ev is not None:
                engage = ev["exchange"] >= HOLD_EXCH
        b = ball.ball
        if b is not None and self.dist(b["pos"], anchor) < 1100.0:
            turret_value = sum(units.cost(t.data) for t in turrets)
            engage = self.ball_value_of(group.members) + turret_value >= b["value"]
        return engage

    def choose_main_attack(self, posture: Posture, group: MainGroup) -> None:
        target = self.kept_attack_target()
        if target is None:
            target = self.choose_attack_target(group.center, group.power, group.members)
        if target is None and "nohunt" not in FLAGS:
            target = self.hunt_target(group.center, group.power)
        self.note_attack_target(target)
        posture.attack_target = target
        posture.mode_eff = "attack" if target is not None else "gather"

    def log_army_plan(self, posture: Posture, group: MainGroup) -> None:
        if self.frame - getattr(self, "_last_group_log", -10**9) < 600:
            return
        self._last_group_log = self.frame
        cache = getattr(self, "_power_cache", (0, 0.0, 0.0))
        tgt = tuple(round(v) for v in posture.attack_target) if posture.attack_target else None
        self.log(
            f"army mode={posture.mode}/{posture.mode_eff} group={len(group.members)} "
            f"cohesion={group.cohesion}/{len(group.members)} "
            f"power={cache[1]:.0f} vs enemy={cache[2]:.0f} regroup={group.regroup} "
            f"defend_engage={posture.defend_engage} target={tgt} "
            f"branches={dict(getattr(self, '_branch_counts', {}))}"
            f" comp={self.our_role_counts()}"
            f" orders={dict(self.__dict__.get('order_counts', {}))}"
            f" order_sources={self.top_order_sources()}"
        )
