"""Defence intelligence: aircraft, artillery pressure and turret budgeting.

The old bot treated every contact as equally threatening and spent metal on
whatever turret happened to be hard-coded in the builder.  This module keeps
the information needed to make a deliberate static-defence decision:

* which aircraft are real aircraft (commander drones and the radar-morph
  scout do not justify AA),
* which enemy units have recently damaged us and at what range,
* how often metal-extractor sectors are attacked and how strained their
  defenders are,
* how much enemy artillery is known directly or inferred from turret damage
  that could not be attributed to a normal attacker.

It deliberately stores raw observations and lets the economy query them.  No
per-cell simulation is necessary for the few dozen sectors involved.
"""
from __future__ import annotations

import math
import os
from typing import Any

from . import units
from .models import UnitInfo

# Excluded from the "real aircraft" test: all of these are spawned by a
# commander or by the radar-tower morph and would otherwise make every bot
# build AA at minute one.
_IGNORED_AIRCRAFT = {
    "dronecarry",
    "dronefighter",
    "droneheavyslow",
    "dronelight",
    "planelightscout",
    "fakeunit_aatarget",
    "fakeunit_los",
}


def _is_real_aircraft(u: UnitInfo) -> bool:
    if not u.is_air:
        return False
    name = u.name.lower()
    if name in _IGNORED_AIRCRAFT or name.startswith("drone"):
        return False
    return True


FLAGS = os.environ.get("ZKBOT_FLAGS", "")

class DefenseMixin:
    # ------------------------------------------------------------ air intel

    def aircraft_seen(self) -> bool:
        return getattr(self, "air_seen_frame", -10**9) > -10**8

    def recent_air_damage(self, window: int = 5400) -> float:
        """Damage real aircraft dealt to us in the last `window` frames."""
        log = getattr(self, "air_damage_log", [])
        cutoff = self.frame - window
        if log and log[0][0] < cutoff - 1800:
            self.air_damage_log = log = [e for e in log if e[0] >= cutoff]
        return sum(d for f, d, _p in log if f >= cutoff)

    def air_damage_hotspot(self, window: int = 5400) -> tuple[float, float] | None:
        """Where aircraft hurt us most recently (300-elmo cells)."""
        cells: dict[tuple[int, int], float] = {}
        cutoff = self.frame - window
        for f, d, p in getattr(self, "air_damage_log", []):
            if f >= cutoff:
                k = (int(p[0] // 300), int(p[1] // 300))
                cells[k] = cells.get(k, 0.0) + d
        if not cells:
            return None
        k = max(cells, key=cells.get)
        return k[0] * 300.0 + 150.0, k[1] * 300.0 + 150.0

    def desired_static_aa(self) -> int:
        """Static AA follows the air threat.  One base turret (the old cap)
        let heavy bombers deal 200k damage in a game we led 3:1 on metal."""
        if not self.aircraft_seen():
            return 0
        if "oldaa" in FLAGS:
            return 1
        contacts = getattr(self, "air_contacts", {})
        recent = sum(1 for f in contacts.values() if self.frame - f <= 5400)
        want = 1 + recent // 4 + int(self.recent_air_damage() // 2500)
        return min(6, want)

    def desired_aa_count(self) -> int:
        """Mobile AA scaled to enemy aircraft seen in the last 3 minutes.

        The old rule (one AA per six units whenever any aircraft had ever
        been seen) spent 34k metal on Gremlins against a few bombers."""
        contacts = getattr(self, "air_contacts", {})
        recent = sum(1 for f in contacts.values() if self.frame - f <= 5400)
        if recent == 0:
            return 0
        share = 6
        if "oldaa" not in FLAGS and self.recent_air_damage() > 5000:
            share = 4
        return min(max(1, len(self.army) // share), math.ceil(recent * 0.8))

    def aircraft_seen_recent(self, window: int = 2700) -> bool:
        return self.aircraft_seen() and self.frame - self.air_seen_frame <= window

    def note_air_contact(self, u: UnitInfo) -> None:
        if _is_real_aircraft(u):
            contacts = getattr(self, "air_contacts", None)
            if contacts is None:
                contacts = self.air_contacts = {}
            contacts[u.uid] = self.frame
            self.air_seen_frame = self.frame
            self.air_seen_defs = getattr(self, "air_seen_defs", set())
            self.air_seen_defs.add(u.name)

    # ------------------------------------------------------ damage / artillery

    def _attacker(self, attacker_id: int) -> UnitInfo | None:
        if attacker_id <= 0 or attacker_id in self.units:
            return None
        u = self.enemies.get(attacker_id) or self.radar.get(attacker_id)
        if u is not None:
            return u
        remembered = self.enemy_memory.get(attacker_id)
        if remembered is not None:
            return remembered[0]
        return None

    def _battle_sector_id(self, pos: tuple[float, float]) -> int | None:
        s = self.sector_for_pos(pos)
        return s.sid if s is not None else None

    def _note_attack_pressure(self, sid: int | None, amount: float = 1.0) -> None:
        if sid is None:
            return
        pressure = getattr(self, "sector_attack_pressure", {})
        pressure[sid] = min(40.0, pressure.get(sid, 0.0) + max(0.1, amount))
        self.sector_attack_pressure = pressure
        self.sector_last_attack[sid] = self.frame

    def _note_artillery_damage(self, sid: int | None, damage: float) -> None:
        if damage <= 0.0:
            return
        self.artillery_damage_log.append((self.frame, float(damage)))
        if sid is not None:
            self.sector_artillery_damage[sid] = (
                self.sector_artillery_damage.get(sid, 0.0) + float(damage)
            )

    def _record_attacker(self, a: UnitInfo, sid: int | None) -> None:
        rng = float(a.data.get("maxWeaponRange") or 0.0)
        if rng <= 0.0:
            return
        self.recent_attacker_range = max(self.recent_attacker_range, rng)
        self.recent_attacker_frame = self.frame
        self.recent_attacker_name = a.name
        if sid is not None:
            old = self.sector_attacker_range.get(sid)
            if old is None or self.frame - old[1] > 1800 or rng > old[0]:
                self.sector_attacker_range[sid] = (rng, self.frame)

    def read_battle_events(self, obs: dict[str, Any]) -> None:
        """Consume unit-damaged/destroyed events from this observation.

        ControllerAI exposes the engine's SUnitDamagedEvent, including the
        attacker id, damage, paralyzer flag and impact direction.  Unknown
        attacker ids are normal for artillery firing from outside our LOS.
        """
        events = obs.get("events") or []
        if not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            try:
                topic = int(event.get("topic") or 0)
            except Exception:
                continue
            if topic == 9:  # EVENT_UNIT_DAMAGED
                try:
                    uid = int(event.get("unitId") or -1)
                except Exception:
                    continue
                u = self.units.get(uid)
                if u is None:
                    continue
                try:
                    damage = max(0.0, float(event.get("damage") or 0.0))
                except Exception:
                    damage = 0.0
                paralyzer = bool(event.get("paralyzer") or False)
                attacker_id = int(event.get("attackerId") or -1)
                a = self._attacker(attacker_id)
                if damage > 0.0 and not paralyzer:
                    led = getattr(self, "dmg_from", None)
                    if led is None:
                        led = self.dmg_from = {}
                    if a is not None:
                        an = a.name
                    elif attacker_id <= 0:
                        an = "?noid"
                    elif attacker_id in self.units:
                        an = "?own"
                    else:
                        an = "?unseen"
                    led[an] = led.get(an, 0.0) + damage
                    if a is not None and a.is_air and _is_real_aircraft(a):
                        log = self.__dict__.setdefault("air_damage_log", [])
                        log.append((self.frame, damage, u.pos))
                sid = self._battle_sector_id(u.pos)
                turret = units.is_defense(u.data) and not u.is_mobile
                mex = u.name == "staticmex"
                # Attacks on extractors and static defences matter most, but
                # damage to any unit in an extractor sector is evidence that
                # the sector is under pressure.
                if mex or turret:
                    self._note_attack_pressure(sid, 1.0 + min(8.0, damage / 120.0))
                elif damage > 0:
                    self._note_attack_pressure(sid, 0.25 + min(2.0, damage / 400.0))
                if a is not None:
                    self._record_attacker(a, sid)
                # Artillery can hit from outside LOS.  A known artillery
                # attacker, or unattributed damage to a turret, is enough to
                # raise the inferred artillery count.  Disarm/paralyzation
                # damage is explicitly excluded.
                if not paralyzer:
                    role = units.role_of(a.data) if a is not None else ""
                    artillery = role in ("arty", "heavyarty")
                    if not artillery and turret:
                        dir_pos = event.get("dirPos")
                        steep = False
                        if isinstance(dir_pos, (list, tuple)) and len(dir_pos) >= 3:
                            try:
                                steep = abs(float(dir_pos[1])) > 0.30
                            except Exception:
                                steep = False
                        artillery = a is None or steep
                    if artillery and damage > 0.0:
                        self._note_artillery_damage(sid, damage)
            elif topic == 10:  # EVENT_UNIT_DESTROYED
                try:
                    uid = int(event.get("unitId") or -1)
                except Exception:
                    uid = -1
                u = self.units.get(uid)
                if u is None or u.name == "terraunit":
                    continue
                lost = getattr(self, "lost_by_name", None)
                if lost is None:
                    lost = self.lost_by_name = {}
                dead = getattr(self, "_dead_counted", None)
                if dead is None:
                    dead = self._dead_counted = set()
                if (uid, u.name) in dead:
                    self.duplicate_death_events = getattr(self, "duplicate_death_events", 0) + 1
                    continue
                dead.add((uid, u.name))
                self.own_lost_value = getattr(self, "own_lost_value", 0.0) + (
                    units.cost(u.data) * max(0.05, min(1.0, u.build_progress or 1.0))
                )
                lost[u.name] = lost.get(u.name, 0.0) + units.cost(u.data)
                if u.is_combat and u.is_mobile:
                    lb = getattr(self, "lost_by_branch", None)
                    if lb is None:
                        lb = self.lost_by_branch = {}
                    bname = getattr(self, "unit_branch", {}).get(uid, "none")
                    lb[bname] = lb.get(bname, 0.0) + units.cost(u.data)
                if u.name == "staticmex" and hasattr(self, "note_mex_lost"):
                    self.note_mex_lost(u)
                if u.is_builder and u.is_mobile:
                    self.builder_death_count = getattr(self, "builder_death_count", 0) + 1
                if hasattr(self, "note_own_death"):
                    self.note_own_death(u, event)
                sid = self._battle_sector_id(u.pos)
                if u.name == "staticmex" or units.is_defense(u.data):
                    self._note_attack_pressure(sid, 2.0)

    def _artillery_damage_recent(self, window: int = 3600) -> float:
        cutoff = self.frame - window
        self.artillery_damage_log = [
            (f, d) for f, d in self.artillery_damage_log if f >= cutoff
        ]
        return sum(d for _f, d in self.artillery_damage_log)

    def enemy_artillery_estimate(self) -> float:
        """Scouted artillery plus artillery inferred from unexplained damage."""
        self.update_sector_state()
        scouted = 0
        seen: set[int] = set()
        for e in self.contacts().values():
            if not self.is_dangerous(e):
                continue
            if units.role_of(e.data) in ("arty", "heavyarty"):
                scouted += 1
                seen.add(e.uid)
        # Remembered but currently out-of-LOS artillery still counts.
        for uid, (e, frame) in self.enemy_memory.items():
            if uid in seen or self.frame - frame > 2400:
                continue
            if units.role_of(e.data) in ("arty", "heavyarty"):
                scouted += 1
        # A normal artillery piece does a few hundred damage per hit.  Scale
        # recent unattributed damage into a comparable unit count.
        inferred = min(6.0, self._artillery_damage_recent() / 180.0)
        return float(scouted) + inferred

    def sector_attack_pressure_value(self, s) -> float:
        raw = self.sector_attack_pressure.get(s.sid, 0.0)
        if raw <= 0.0:
            return 0.0
        last = self.sector_last_attack.get(s.sid, self.frame)
        age = max(0.0, (self.frame - last) / 30.0)
        return raw * math.exp(-age / 110.0)

    def sector_strain(self, s) -> float:
        """Low-HP/retreating defenders and damaged turrets in a sector."""
        return max(0.0, float(getattr(s, "strain", 0.0)))

    def recent_attacker_range_for(self, pos: tuple[float, float]) -> float:
        """Maximum range of enemies that recently attacked near ``pos``."""
        self.update_sector_state()
        sid = self._battle_sector_id(pos)
        best = 0.0
        cutoff = self.frame - 1800
        for key, (rng, frame) in self.sector_attacker_range.items():
            if frame < cutoff:
                continue
            if sid is not None and key != sid:
                # Also accept adjacent sectors: an enemy skirmisher attacking
                # an extractor on the edge of a cluster is relevant to the
                # neighbouring turret.
                ss = self.sectors[key] if 0 <= key < len(self.sectors) else None
                if ss is None or self.dist((ss.x, ss.z), pos) > 750.0:
                    continue
            best = max(best, rng)
        if best <= 0.0 and self.frame - self.recent_attacker_frame < 1800:
            if sid is None or self.dist(pos, self.base_pos) < 1300.0:
                best = self.recent_attacker_range
        # A visible skirmisher/artillery piece near a position is a reason to
        # choose a longer-ranged turret even before it damages anything.
        for e in self.contacts().values():
            if not self.is_dangerous(e):
                continue
            if self.dist(pos, e.pos) <= 1100.0:
                best = max(best, float(e.data.get("maxWeaponRange") or 0.0))
        if self.enemy_artillery_estimate() >= 0.75:
            best = max(best, 860.0)
        return best

    # -------------------------------------------------------- spending model

    def _spend_records(self) -> list[tuple[int, float, str]]:
        cutoff = self.frame - 18000  # about ten minutes
        records = [
            (f, c, kind)
            for f, c, kind in getattr(self, "_defense_spend_records", [])
            if f >= cutoff
        ]
        self._defense_spend_records = records
        return records

    def defense_spend_share(self) -> float:
        records = self._spend_records()
        total = sum(c for _f, c, _kind in records)
        if total <= 0.0:
            return 0.0
        defense = sum(c for _f, c, kind in records if kind == "defense")
        return defense / max(1.0, total)

    def _record_spending(self, kind: str, cost: float) -> None:
        if cost <= 0.0:
            return
        records = getattr(self, "_defense_spend_records", [])
        records.append((self.frame, cost, kind))
        self._defense_spend_records = records

    def desired_defense_share(self) -> float:
        """Fraction of mobile/static metal that should go to turrets.

        Attack frequency dominates; low-HP/retreating defenders and enemy
        artillery raise it further.  The result is intentionally bounded so a
        single raid cannot turn the whole bot into a turtle.
        """
        attack = 0.0
        strain = 0.0
        for s in getattr(self, "sectors", []):
            if s.owner == "ours" or s.distance_base < 1300.0:
                attack += self.sector_attack_pressure_value(s)
            if s.owner == "ours":
                strain = max(strain, self.sector_strain(s))
        arty = self.enemy_artillery_estimate()
        share = 0.07
        share += min(0.18, attack * 0.018)
        share += min(0.20, strain * 0.28)
        share += min(0.16, arty * 0.035)
        if self.threat_pos is not None and self.frame - self.threat_frame < 600:
            if self.dist(self.threat_pos, self.base_pos) < 1800.0:
                share += 0.05
        return max(0.04, min(0.52, share))

    def defense_pending_cost(self) -> float:
        total = 0.0
        for t in self.tasks.values():
            if t.kind not in ("defense", "mexdef", "frontdef", "aa"):
                continue
            # A task whose builder died two minutes ago must not block a
            # replacement turret near the base.  It still gets cleaned up by
            # the normal task timeout, but it no longer holds the budget.
            if not t.assignees and self.frame - t.created_frame > 1800:
                continue
            total += units.cost(units.by_id(t.def_id))
        return total

    def defense_budget_ok(self, extra_cost: float = 0.0, urgent: bool = False) -> bool:
        desired = self.desired_defense_share()
        records = self._spend_records()
        recent_total = sum(c for _f, c, _kind in records)
        recent_def = sum(c for _f, c, kind in records if kind == "defense")
        pending = self.defense_pending_cost()
        current = (recent_def + pending * 0.65 + extra_cost) / max(
            1.0, recent_total + pending * 0.65 + extra_cost
        )
        allowance = 0.10 if urgent else 0.015
        return current < desired + allowance

    def defense_unit_for(self, pos: tuple[float, float], aa: bool = False) -> int | None:
        """Pick the cheapest turret that outranges recent attackers.

        For example, a 530-range Rogue makes a Lotus (460) the wrong answer;
        a Picket (610) is the right one.  If enemy artillery has been seen or
        inferred, only a turret that can reach it is considered.
        """
        if aa:
            if not self.aircraft_seen():
                return None
            req = 700.0
            for e in self.contacts().values():
                if e.is_air and _is_real_aircraft(e):
                    req = max(req, float(e.data.get("maxWeaponRange") or 0.0))
            options = [
                ("turretaalaser", 1000.0),
                ("turretaaflak", 1000.0),
                ("turretaafar", 1800.0),
                ("turretaaheavy", 2400.0),
            ]
        else:
            req = self.recent_attacker_range_for(pos)
            if req <= 0.0:
                # No attack range evidence yet: mostly Lotus (LLT), the
                # solid answer to early raiders, with an occasional Picket
                # for reach.  (Picket-only defence was a human reviewer's
                # complaint; it loses to the raiders it is meant to stop.)
                n = int(getattr(self, "_default_turret_n", 0))
                self._default_turret_n = n + 1
                if "oldturret" in os.environ.get("ZKBOT_FLAGS", ""):
                    req = 500.0
                else:
                    req = 500.0 if n % 3 == 2 else 400.0
            options = [
                ("turretlaser", 460.0),
                ("turretgauss", 560.0),
                ("turretmissile", 610.0),
                ("turretheavylaser", 620.0),
                ("turretsunlance", 750.0),
                ("turretantiheavy", 1200.0),
            ]
        for name, rng in options:
            d = units.by_name(name)
            if d is None or rng < req:
                continue
            # Skip dominated choices: Picket is cheaper and longer-ranged
            # than Gauss, so Gauss is only a fallback if Picket is missing.
            if name == "turretgauss" and units.by_name("turretmissile") is not None:
                continue
            if name == "turretaaflak" and units.by_name("turretaalaser") is not None:
                continue
            # Out-ranged by artillery: a 2200-metal Desolator is no answer
            # to it early on; mobile units are.
            if name == "turretantiheavy" and (
                self.metal_income() < 60.0 or self.frame_time() < 1200
            ):
                continue
            return int(d["id"])
        return None

    def defense_step(self) -> None:
        """Periodic diagnostic; the actual decisions happen in economy.py."""
        if self.frame - getattr(self, "last_defense_log_frame", -10**9) < 600:
            return
        self.last_defense_log_frame = self.frame
        turrets = sum(
            1
            for u in self.units.values()
            if units.is_defense(u.data) and not u.is_mobile and u.is_finished
        )
        attack = 0.0
        strain = 0.0
        for s in getattr(self, "sectors", []):
            if s.owner == "ours" or s.distance_base < 1300.0:
                attack += self.sector_attack_pressure_value(s)
            if s.owner == "ours":
                strain = max(strain, self.sector_strain(s))
        air = "yes" if self.aircraft_seen() else "no"
        self.log(
            "defense turrets={} spend={:.2f} desired={:.2f} attack={:.1f} "
            "strain={:.2f} arty={:.2f} air={} gremlins={}".format(
                turrets,
                self.defense_spend_share(),
                self.desired_defense_share(),
                attack,
                strain,
                self.enemy_artillery_estimate(),
                air,
                len(getattr(self, "gremlin_scouts", set())),
            )
        )

    # --------------------------------------------------------- new-unit hooks

    def note_new_units(self, new_units: dict[int, UnitInfo], old_ids: set[int]) -> None:
        for uid, u in new_units.items():
            if uid in old_ids:
                continue
            if u.is_air and hasattr(self, "note_new_air_unit"):
                self.note_new_air_unit(u)
            if u.name == "cloakaa" and getattr(self, "gremlin_pending", 0) > 0:
                # A Gremlin explicitly queued for artillery spotting.
                if hasattr(self, "note_new_gremlin"):
                    self.note_new_gremlin(u)
            # Do not count the very first commander/factory observation as
            # army spending.  Only started combat/defence units are part of
            # the turret-versus-mobile budget.
            if self.frame < 10:
                continue
            # Count every unit once: some (bombers) drop out of the
            # observation and reappear, which inflated the ledger ~14x.
            counted = getattr(self, "_spend_counted", None)
            if counted is None:
                counted = self._spend_counted = set()
            if (uid, u.name) in counted:
                continue
            counted.add((uid, u.name))
            cost = units.cost(u.data)
            if cost <= 0.0 or u.name == "terraunit":
                continue
            ledger = getattr(self, "spend_by_name", None)
            if ledger is None:
                ledger = {}
                self.spend_by_name = ledger
            ledger[u.name] = ledger.get(u.name, 0.0) + cost
            if units.is_defense(u.data):
                self._record_spending("defense", cost)
            elif u.is_mobile and not u.is_factory and (u.is_combat or u.is_builder):
                self._record_spending("mobile", cost)
