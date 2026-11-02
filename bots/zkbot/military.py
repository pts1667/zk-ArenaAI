"""MilitaryMixin methods for the zkbot Brain."""
from __future__ import annotations

import collections
import math
import os
from typing import Any

from . import units
from .client import QueryError
from .models import Task, UnitInfo
from .threat import combat_power, ground_dps, weapon_range


FLAGS = set(os.environ.get('ZKBOT_FLAGS', '').split(','))


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
        center = self.army_center()
        enemy_commander = (
            self.enemies.get(self.enemy_commander_id) if self.enemy_commander_id else None
        )
        commander_memory = (
            self.enemy_memory.get(self.enemy_commander_id)
            if self.enemy_commander_id is not None and not self.enemy_commander_dead
            else None
        )
        focus_commander = False
        focus_pos: tuple[float, float] | None = None
        if self.enemy_commander_dead:
            enemy_commander = None
        if enemy_commander is not None and enemy_commander.is_finished:
            focus_pos = enemy_commander.pos
        elif commander_memory and self.frame - commander_memory[1] < 900:
            focus_pos = commander_memory[0].pos
        enemy_count = (
            sum(1 for e in self.enemies.values() if self.is_dangerous(e))
            + len(self.radar)
        )
        close_to_base = (
            focus_pos is not None and self.dist(focus_pos, self.base_pos) < 2000.0
        )
        weak_commander = (
            enemy_commander is not None and enemy_commander.alive_frac < 0.5
        )
        if (
            focus_pos is not None
            and self.combat_count() >= (7 if close_to_base else 12)
            and (
                (
                    close_to_base
                    and (
                        enemy_count <= self.combat_count() * 1.5
                        or enemy_count <= 8
                    )
                )
                or (weak_commander and close_to_base)
            )
        ):
            focus_commander = True
            self.hunt_commander = True
            if self.frame - self.last_attack_frame > 300:
                self.last_attack_frame = self.frame
                hp = ""
                if enemy_commander is not None:
                    hp = (
                        f" hp={enemy_commander.health:.0f}/"
                        f"{enemy_commander.max_health:.0f}"
                    )
                near = sum(
                    1
                    for uid in self.army
                    if uid in self.units
                    and self.dist(self.units[uid].pos, focus_pos) < 1200.0
                )
                self.log(
                    f"focus enemy commander at {tuple(round(v) for v in focus_pos)}"
                    f"{hp} army_near={near}"
                )

        if focus_pos is None:
            self.hunt_commander = False
        enemy_army_pos = None
        contacts = list(self.enemies.values()) + list(self.radar.values())
        if contacts:
            e = min(
                (
                    e
                    for e in contacts
                    if not e.is_builder or "dyntrainer" in e.name or "commander" in e.name
                ),
                key=lambda e: self.dist(e.pos, self.base_pos),
                default=None,
            )
            if e is not None:
                enemy_army_pos = e.pos

        threatened = self.most_threatened_sector()
        if threatened is not None and (
            threatened.distance_base >= 2000.0 or threatened.owner != "ours"
        ):
            threatened = None
        defense_anchor = self.best_defense_pos(threatened)
        if defense_anchor is None:
            defense_anchor = self.threat_pos
        attack_sector = self.best_attack_sector()
        attack_anchor = self.best_defense_pos(attack_sector)
        home_anchor = defense_anchor or self.base_pos
        raider_uids = [
            uid for uid in self.army if uid in self.units and self.is_raider(self.units[uid])
        ]
        raider_stage = self.rally_point()
        if self.enemy_near(raider_stage, 650.0) is not None:
            raider_stage = self.base_pos
        raider_group_limit = 3 if self.frame_time() < 420 else 5
        squad = raider_uids[:raider_group_limit]
        group_needed = 2 if self.frame_time() < 300 else 3
        # If nothing has been visible for a while, the enemy commander may be
        # cloaked or hiding.  Sweep the enemy half of the map with small groups
        # instead of letting a large army sit on an empty base.
        # Only sweep for a hiding commander when we are winning outright and
        # nothing has been seen for a full minute.  (The old rule switched
        # search mode on whenever the commander position was unknown after
        # 8 minutes, which scattered raiders and disabled local engagement.)
        unknown_hunt = (
            focus_pos is None
            and mode == "attack"
            and self.combat_count() >= 20
            and self.frame - self.last_enemy_seen_frame > 1800
        )
        search_mode = unknown_hunt or (
            mode != "defend"
            and self.combat_count() >= 8
            and self.frame - self.last_enemy_seen_frame > 600
            and enemy_army_pos is None
        )
        search_pts = self.search_points() if search_mode else []
        group_active = (
            len(squad) >= group_needed
            and not focus_commander
            and not search_mode
            and mode != "attack"
            and not (
                mode == "defend"
                and self.threat_pos is not None
                and self.dist(self.threat_pos, self.base_pos) < 2200.0
            )
        )
        ball = self.enemy_slow_ball()
        ball_join = self.ball_joiners(raider_uids, ball)
        squad_raiders = [uid for uid in raider_uids if uid not in ball_join]
        raid_target = None
        ready: list[int] = []
        # uid -> (squad target, member is at the stage/target)
        raid_orders: dict[int, tuple[tuple[float, float], bool]] = {}
        if "oldraid" not in FLAGS:
            # Every raider raids, in up to three squads with separate targets.
            # (Only the first 3-5 raided before; the rest idled at the rally
            # point while a human held the map with a few Kodachis.)
            group_active = (
                len(squad_raiders) >= group_needed
                and not focus_commander
                and not search_mode
                and mode != "attack"
                and not (
                    mode == "defend"
                    and self.threat_pos is not None
                    and self.dist(self.threat_pos, self.base_pos) < 1400.0
                )
            )
            if group_active:
                if "oldsquads" in FLAGS:
                    self.update_raid_squads(squad_raiders, raider_stage, group_needed, raid_orders)
                else:
                    self.update_raid_squads_v2(squad_raiders, raider_stage, group_needed, raid_orders)
                group_active = bool(raid_orders)
        elif group_active:
            if (
                self.raid_group_target is None
                or self.frame - self.raid_group_frame > 600
                or self.dist(self.raid_group_target, self.base_pos) < 450.0
            ):
                self.raid_group_target = self.choose_raid_group_target(squad)
                self.raid_group_frame = self.frame
                if self.raid_group_target is not None:
                    self.log(
                        "raid group "
                        f"{len(squad)} -> {tuple(round(v) for v in self.raid_group_target)}"
                    )
            raid_target = self.raid_group_target
            if raid_target is None:
                group_active = False
            else:
                ready = [
                    uid
                    for uid in squad
                    if uid in self.units
                    and (
                        self.dist(self.units[uid].pos, raider_stage) < 900.0
                        or self.dist(self.units[uid].pos, raid_target) < 1000.0
                    )
                ]
                self.raid_group_size = len(ready)
        # ---- main group: cohesion and strength-based engagement ----------
        group = [
            uid
            for uid in self.army
            if uid in self.units
            and (not self.is_raider(self.units[uid]) or uid in ball_join)
            and units.role_of(self.units[uid].data) != "bomb"
            and not self.units[uid].is_air
            and not (uid in self.home_guard and mode == "attack")
            and uid not in self.retreating
        ]
        group_set = set(group)
        group_center = self.median_pos(group) or self.rally_point()
        group_power = sum(combat_power(self.units[uid]) for uid in group)
        near_center = sum(
            1 for uid in group if self.dist(self.units[uid].pos, group_center) < 700.0
        )
        # Away from the base a defence is fought as one group too; feeding
        # units in one by one lost even fights.  At the base, everyone goes.
        far_defence = (
            "defregroup" in FLAGS
            and mode == "defend"
            and defense_anchor is not None
            and self.dist(defense_anchor, self.base_pos) > 1200.0
        )
        regroup = (
            (mode in ("attack", "gather") or far_defence)
            and len(group) >= 4
            and near_center < 0.6 * len(group)
        )
        defend_engage = True
        hold_point = self.fallback_point(defense_anchor)
        if mode == "defend" and defense_anchor is not None:
            _o, theirs = self.local_balance(defense_anchor, 900.0)
            turret_power = sum(
                combat_power(t)
                for t in self.units.values()
                if units.is_defense(t.data)
                and not t.is_mobile
                and t.is_finished
                and self.dist(t.pos, defense_anchor) < 900.0
            )
            defend_engage = group_power + turret_power >= 0.8 * theirs
            if ball is not None and self.dist(ball["pos"], defense_anchor) < 1100.0:
                turret_value = sum(
                    units.cost(t.data)
                    for t in self.units.values()
                    if units.is_defense(t.data)
                    and not t.is_mobile
                    and t.is_finished
                    and self.dist(t.pos, defense_anchor) < 900.0
                )
                defend_engage = self.ball_value_of(group) + turret_value >= ball["value"]
        attack_target = None
        mode_eff = mode
        ball_gather = None
        if mode == "attack":
            attack_target = self.choose_attack_target(group_center, group_power)
            if attack_target is None and "nohunt" not in FLAGS:
                attack_target = self.hunt_target(group_center, group_power)
            mode_eff = "attack" if attack_target is not None else "gather"
        if ball is not None and mode in ("attack", "gather") and (
            self.dist(ball["pos"], group_center) < 2500.0
            or (attack_target is not None and self.dist(ball["pos"], attack_target) < 1300.0)
        ):
            how, where = self.ball_engagement(ball, group, group_center)
            if how == "commit":
                attack_target = where
                mode_eff = "attack"
            else:
                ball_gather = where
                mode_eff = "ballgather"
        elif ball is None:
            self._ball_committed = False
        if self.frame - getattr(self, "_last_group_log", -10**9) >= 600:
            self._last_group_log = self.frame
            cache = getattr(self, "_power_cache", (0, 0.0, 0.0))
            tgt = tuple(round(v) for v in attack_target) if attack_target else None
            self.log(
                f"army mode={mode}/{mode_eff} group={len(group)} cohesion={near_center}/{len(group)} "
                f"power={cache[1]:.0f} vs enemy={cache[2]:.0f} regroup={regroup} "
                f"defend_engage={defend_engage} target={tgt} "
                f"branches={dict(getattr(self, '_branch_counts', {}))}"
                f" comp={self.our_role_counts()}"
            )
        responders = self.assign_responders(mode) if 'noresponse' not in FLAGS else {}
        if (
            attack_target is not None
            and not (ball is not None and attack_target == ball["pos"])
            and self.attack_target_stale(attack_target, group_center)
        ):
            attack_target = self.choose_attack_target(group_center, group_power)
            if attack_target is None and "nohunt" not in FLAGS:
                attack_target = self.hunt_target(group_center, group_power)
            mode_eff = "attack" if attack_target is not None else "gather"
        br = _BranchCounter(self)
        for uid in list(self.army):
            u = self.units.get(uid)
            if not u:
                continue
            br.uid = uid
            # Cloak bombs are expendable: drive them straight at the most
            # valuable target instead of retreating or forming a firing line.
            if units.role_of(u.data) == "bomb" and not u.is_air:
                target = (
                    focus_pos
                    or enemy_army_pos
                    or (self.threat_pos if mode == "defend" else None)
                    or self.group_attack_target(center)
                    or self.enemy_base
                    or self.rally_point()
                )
                if target:
                    self.order_move(uid, target)
                br['bomb'] += 1
                continue
            # Retreat only when it is actually worth preserving the unit.
            # Retreat destinations are chosen to avoid enemy contact; a cheap
            # raider with no clean path home is allowed to trade itself.
            if uid in self.retreating and u.alive_frac > (0.70 if 'oldretreat' in FLAGS else 0.85):
                self.retreating.discard(uid)
            if uid in self.retreating:
                self.order_move(uid, self.safe_retreat_point(u))
                br['retreating'] += 1
                continue
            should_retreat, retreat_dest = self.retreat_decision(u)
            if should_retreat and retreat_dest is not None:
                self.retreating.add(uid)
                self.order_move(uid, retreat_dest)
                br['retreat_new'] += 1
                continue
            self.retreating.discard(uid)
            if u.is_air:
                air_target = attack_target if mode_eff == "attack" else (
                    defense_anchor if mode_eff == "defend" else None
                )
                self.control_air(u, group_center, air_target)
                br['air'] += 1
                continue
            if ground_dps(u) < 5.0 and (units.is_aa(u.data) or units.role_of(u.data) == "aa"):
                # Anti-air that cannot shoot ground (Gremlin) walked at the
                # front and died in every ground fight.  Shadow the group and
                # only engage aircraft.
                air_t = None
                for e in self.enemies.values():
                    if e.is_air and self.dist(u.pos, e.pos) < weapon_range(u) + 150.0:
                        air_t = e
                        break
                if air_t is not None:
                    self.order_attack(uid, air_t.uid)
                else:
                    self.order_move(uid, group_center)
                br['aa_escort'] += 1
                continue
            if uid in self.home_guard and mode == "attack":
                # The guard holds the base / threatened owned sector while
                # the main force is away.  In defend mode everyone converges,
                # so this branch is intentionally limited to attack/gather.
                self.order_fight(uid, home_anchor)
                br['guard'] += 1
                continue
            if uid in responders and not focus_commander:
                local = self.best_local_target(u)
                if local is not None and self.dist(u.pos, local.pos) < weapon_range(u) + 80.0:
                    self.order_attack(uid, local.uid)
                else:
                    self.order_fight(uid, responders[uid])
                br['respond'] += 1
                continue
            if focus_commander and focus_pos is not None:
                if enemy_commander is not None and enemy_commander.is_finished:
                    self.order_attack(uid, enemy_commander.uid)
                elif u.is_air or units.role_of(u.data) in (
                    "arty",
                    "assault",
                    "riot",
                    "skirm",
                    "heavy",
                ):
                    # The commander went out of LOS (often cloaked).  Fire at
                    # the last known position; area damage can still finish it.
                    self.order_attack_ground(uid, focus_pos)
                else:
                    self.order_fight(uid, focus_pos)
                br['focus_comm'] += 1
                continue
            if search_mode and search_pts:
                should_search = (
                    not unknown_hunt
                    or u.is_air
                    or self.is_raider(u)
                    or uid % 6 == 0
                )
                if should_search:
                    idx = (uid + self.frame // 900) % len(search_pts)
                    self.order_fight(uid, search_pts[idx])
                    br['search'] += 1
                    continue
            if self.is_raider(u) and uid not in ball_join:
                if uid in raid_orders:
                    tgt, at_stage = raid_orders[uid]
                    self.order_fight(uid, tgt if at_stage else raider_stage)
                    br['raid_group'] += 1
                    continue
                if group_active and raid_target is not None and "oldraid" in FLAGS:
                    if uid in ready:
                        self.order_fight(uid, raid_target)
                    else:
                        self.order_fight(uid, raider_stage)
                    br['raid_group'] += 1
                    continue
                if mode != "attack" and not (
                    mode == "defend"
                    and self.threat_pos is not None
                    and self.dist(self.threat_pos, self.base_pos) < 2200.0
                ):
                    self.order_fight(uid, raider_stage)
                    br['raid_stage'] += 1
                    continue
            if u.name == "cloaksnipe" and "nophantom" not in FLAGS and u.is_finished:
                if uid not in getattr(self, "phantom_radar_off", set()):
                    self.note_new_phantom(u)
                br[self.phantom_order(u, self.phantom_anchor(group_center))] += 1
                continue
            if 'nokite' not in FLAGS and self.kite(u):
                br['kite'] += 1
                continue
            if mode_eff == "ballgather" and ball_gather is not None and uid in group_set:
                # Wait out of the ball's reach until the group is big enough.
                if self.dist(u.pos, ball["pos"]) < ball["range"] + 350.0:
                    self.order_move(uid, ball_gather)
                else:
                    self.order_fight(uid, ball_gather)
                br['ball_gather'] += 1
                continue
            if 'nooutrange' not in FLAGS:
                away = self.outranged_escape(u)
                if away is not None:
                    self.order_move(uid, away)
                    br['outranged'] += 1
                    continue
            local = self.best_local_target(u)
            if local is not None:
                role = units.role_of(u.data)
                reach = max(220.0, weapon_range(u) + 80.0)
                if role in ("arty", "heavyarty"):
                    reach = weapon_range(u) * 0.95
                near = self.dist(u.pos, local.pos) < reach
                aa_air = (units.is_aa(u.data) or role == "aa") and local.is_air
                # Do not let a unit chase into a fight the group would lose.
                if near and not aa_air and mode_eff != "defend":
                    ours_l, theirs_l = self.local_balance(local.pos, 600.0)
                    if theirs_l > ours_l * 1.3 and self.dist(u.pos, local.pos) > weapon_range(u):
                        near = False
                if near or aa_air:
                    # Radar blips have no valid attack target id for most
                    # weapons; artillery can still shell the location.
                    if local.uid in self.radar and role in ("arty", "heavyarty"):
                        self.order_attack_ground(uid, local.pos)
                    else:
                        self.order_attack(uid, local.uid)
                    br['local'] += 1
                    continue
            if regroup and 'noregroup' not in FLAGS and uid in group_set and self.dist(u.pos, group_center) > 600.0:
                self.order_move(uid, group_center)
                br['regroup'] += 1
                continue
            if mode_eff == "defend" and defense_anchor is not None:
                # Meet the attack as one group when we can hold it; otherwise
                # fall back onto the turrets instead of feeding units in.
                self.order_fight(uid, defense_anchor if defend_engage else hold_point)
                br['defend' if defend_engage else 'hold'] += 1
                continue
            if mode_eff == "attack" and attack_target is not None:
                self.order_fight(uid, attack_target)
                br['attack'] += 1
                continue
            self.order_fight(uid, self.rally_point())
            br['rally'] += 1
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
