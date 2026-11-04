"""Engagement logging (EngagementLogMixin): training data for a combat model.

Passive: issues no orders.  Every 2 s it looks for fights (a cluster of
visible enemy combat units with our combat units or turrets nearby), takes a
snapshot of both sides, and 20 s later records what each side lost.  One
JSON line per engagement goes to engagements-team<N>.jsonl next to the log:

  t, pos, our/their: count, cost, hp, power (sqrt(dps*hp)), dps, mean range
  and speed (cost weighted), role counts, turret cost, mean ground height,
  outcome: our_lost / their_lost (cost of dead units + cost x HP fraction
  lost by survivors), their_unknown (enemy cost whose fate we did not see)

The aim (docs/ARCHITECTURE.md, P4) is a fitted engage/retreat estimator to
replace the hand-tuned power ratios.  Enemy deaths come from
EVENT_ENEMY_DESTROYED (topic 18); the topic histogram is logged so that can
be checked.  Opt out with flag `noenglog`.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from . import units
from .params import FLAGS
from .threat import combat_power, ground_dps, weapon_range

SNAP_EVERY = 60          # frames
OUTCOME_AFTER = 600      # frames (20 s)
CLUSTER_RADIUS = 500.0
CONTACT_RADIUS = 800.0
MAX_OPEN = 6
EVENT_ENEMY_DESTROYED = 18
# Negative weapon def ids are engine damage sources (CSolidObject::DamageType).
ENV_DAMAGE = {
    1: "debris", 2: "ground_collision", 3: "object_collision", 4: "fire",
    5: "water", 6: "killed", 7: "crushed", 8: "aircraft_crash",
    9: "negative_health", 10: "selfd_expired", 11: "killed_cheat",
    12: "reclaimed", 13: "out_of_bounds",
}
DEF_QUERIES_PER_STEP = 4


class EngagementLogMixin:
    def note_event_topics(self, obs: dict) -> None:
        """Count event topics and remember enemy deaths (topic 18)."""
        if "noenglog" in FLAGS:
            return
        hist = self.__dict__.setdefault("_topic_hist", Counter())
        for ev in obs.get("events") or []:
            if not isinstance(ev, dict):
                continue
            try:
                topic = int(ev.get("topic") or 0)
            except (TypeError, ValueError):
                continue
            hist[topic] += 1
            if topic == EVENT_ENEMY_DESTROYED:
                # ControllerAI forwards only the topic, no unit id: the
                # deaths are matched to vanished enemies after read_enemies.
                self._pending_kills = getattr(self, "_pending_kills", 0) + 1

    def resolve_enemy_deaths(self) -> None:
        """Attribute this observation's EVENT_ENEMY_DESTROYED count to enemies
        that were visible last observation and are now neither visible nor a
        radar blip, nearest our units first."""
        if "noenglog" in FLAGS:
            return
        dead = self.__dict__.setdefault("enemy_dead", {})
        prev = getattr(self, "_prev_visible", {})
        n = getattr(self, "_pending_kills", 0)
        self._pending_kills = 0
        if n and prev:
            gone = [e for uid, e in prev.items() if uid not in self.enemies and uid not in self.radar]
            mine = [u.pos for u in self.units.values() if u.is_combat or not u.is_mobile]

            def near(e):
                return min((self.dist(e.pos, p) for p in mine), default=1e9)

            for e in sorted(gone, key=near)[:n]:
                if e.uid not in dead:
                    dead[e.uid] = self.frame
                    self.enemy_killed_value = getattr(self, "enemy_killed_value", 0.0) + units.cost(e.data)
        self._prev_visible = dict(self.enemies)

    def engagement_step(self) -> None:
        if "noenglog" in FLAGS:
            return
        if self.frame - getattr(self, "_eng_frame", -10**9) < SNAP_EVERY:
            return
        self._eng_frame = self.frame
        open_ = self.__dict__.setdefault("_eng_open", [])
        # Close finished engagements.
        still = []
        for eng in open_:
            if self.frame - eng["frame"] >= OUTCOME_AFTER:
                self._write_engagement(eng)
            else:
                still.append(eng)
        open_[:] = still
        if self.frame % 9000 < SNAP_EVERY and getattr(self, "_topic_hist", None):
            self.log("event topics " + " ".join(f"{k}={v}" for k, v in sorted(self._topic_hist.items())))
        if len(open_) >= MAX_OPEN:
            return
        enemies = [
            e for e in self.enemies.values()
            if e.is_mobile and not e.is_builder and self.is_dangerous(e) and not e.is_air
        ]
        if not enemies:
            return
        mine = [
            u for u in self.units.values()
            if u.is_finished and not u.is_air and (
                (u.is_mobile and u.is_combat) or (units.is_defense(u.data) and not u.is_mobile)
            )
        ]
        if not mine:
            return
        used: set[int] = set()
        for seed in sorted(enemies, key=lambda e: -units.cost(e.data)):
            if seed.uid in used or len(open_) >= MAX_OPEN:
                continue
            cluster = [e for e in enemies if e.uid not in used and self.dist(e.pos, seed.pos) < CLUSTER_RADIUS]
            used.update(e.uid for e in cluster)
            cx = sum(e.pos[0] for e in cluster) / len(cluster)
            cz = sum(e.pos[1] for e in cluster) / len(cluster)
            c = (cx, cz)
            ours = [u for u in mine if self.dist(u.pos, c) < CONTACT_RADIUS]
            if not ours:
                continue
            if any(self.dist(c, eng["pos"]) < 600.0 for eng in open_):
                continue
            theirs = [e for e in enemies if self.dist(e.pos, c) < CONTACT_RADIUS]
            open_.append({
                "frame": self.frame,
                "pos": c,
                "ours": {u.uid: (units.cost(u.data), u.alive_frac) for u in ours},
                "theirs": {e.uid: (units.cost(e.data), e.alive_frac) for e in theirs},
                "features": {
                    "t": round(self.frame_time()),
                    "our": self._side_features(ours),
                    "their": self._side_features(theirs),
                    "branch": dict(Counter(getattr(self, "unit_branch", {}).get(u.uid, "none") for u in ours)),
                },
            })

    # ------------------------------------------------- weapon def index

    def learn_weapon_defs(self) -> None:
        """Map weapon def ids to what fired them, a few unit defs per step.

        Damage events carry a weaponDefId even when the attacker is hidden
        or already dead.  A def's weapon mounts name the unit; its death and
        self-destruct explosions are shared weapons (a fusion's atomic_blast,
        a pylon's estor_buildingex) that killed ~10% of our losses in h124
        while being logged as "?noid".  Def types we have seen go first, then
        the rest of the metadata table, so the index is built from the game
        itself and follows balance changes."""
        if "noenglog" in FLAGS:
            return
        done = self.__dict__.setdefault("_wd_defs_done", set())
        queue = self.__dict__.setdefault("_wd_def_queue", [])
        if not queue:
            seen = {u.def_id for u in self.units.values()}
            seen.update(e.def_id for e in self.enemies.values())
            seen.update(e.def_id for e in self.radar.values())
            queue.extend(d for d in seen if d > 0 and d not in done)
            if not queue and self.frame > 900 and not getattr(self, "_wd_meta_queued", False):
                self._wd_meta_queued = True
                for k in getattr(self, "metadata", {}) or {}:
                    try:
                        d = int(k)
                    except (TypeError, ValueError):
                        continue
                    if d > 0 and d not in done:
                        queue.append(d)
        index = self.__dict__.setdefault("wd_index", {})
        for _ in range(DEF_QUERIES_PER_STEP):
            if not queue:
                break
            did = queue.pop(0)
            if did in done:
                continue
            done.add(did)
            try:
                res = self.client.query({"type": "unit_def_by_id", "unitDefId": int(did)}, timeout=3.0)
            except Exception:
                continue
            if not isinstance(res, dict):
                continue
            name = res.get("name") or f"def{did}"
            for m in res.get("weaponMounts") or []:
                w = (m or {}).get("weaponDef") or {}
                if isinstance(w.get("id"), int) and w["id"] not in index:
                    index[w["id"]] = name
            for key in ("deathExplosion", "selfDExplosion"):
                w = res.get(key) or {}
                if isinstance(w, dict) and isinstance(w.get("id"), int):
                    index.setdefault(w["id"], "boom:" + str(w.get("name") or w["id"]))

    def weapon_label(self, weapon_def: int | None) -> str | None:
        """What a weapon def id is: a unit name, "boom:<explosion>" or
        "env:<damage source>"; None when unknown."""
        if weapon_def is None:
            return None
        if weapon_def < 0:
            return "env:" + ENV_DAMAGE.get(-weapon_def, str(-weapon_def))
        owner = getattr(self, "wd_owner", {}).get(weapon_def)
        if owner:
            return owner
        return getattr(self, "wd_index", {}).get(weapon_def)

    def resolve_attacker(self, attacker: str, weapon_def: int | None) -> str:
        """Name a hidden or unidentified attacker from its weapon: "~name"
        for an inferred enemy unit, "own:name" for damage from our own side
        (death explosions included), "boom:..."/"env:..." as is."""
        if not attacker.startswith("?"):
            return attacker
        label = self.weapon_label(weapon_def)
        if not label:
            return attacker
        if attacker == "?own":
            return "own:" + label
        if label.startswith(("boom:", "env:")):
            return label
        return "~" + label

    def note_hit(self, uid: int, attacker: str, damage: float, weapon_def: int | None) -> str:
        """Remember who last damaged each of our units, for log_own_death.
        Returns the attacker name, resolved from the weapon when hidden."""
        if "noenglog" in FLAGS:
            return attacker
        # Learn weapon def -> unit name from visible attackers too: it
        # covers weapons whose def has not been queried yet.
        owners = self.__dict__.setdefault("wd_owner", {})
        if attacker.startswith("def"):
            # A radar blip of unknown type ("def0"): name it from the weapon.
            attacker = "?blip"
        if weapon_def is not None and weapon_def >= 0 and not attacker.startswith("?"):
            owners[weapon_def] = attacker
        attacker = self.resolve_attacker(attacker, weapon_def)
        hits = self.__dict__.setdefault("_hits", {})
        rec = hits.get(uid)
        if rec is None or self.frame - rec["f"] > 300:
            rec = hits[uid] = {"f0": self.frame, "by": Counter()}
        rec["f"] = self.frame
        rec["last"] = attacker
        rec["wd"] = weapon_def
        rec["by"][attacker] += damage
        return attacker

    def log_own_death(self, u, event: dict) -> None:
        """One JSON line per lost unit (deaths-team<N>.jsonl): what killed it
        (attacker name, ?unseen / ?noid when hidden, plus the weapon def id),
        where, in which behaviour branch, and the local balance it died in.
        Answers "what kills our units" directly, unlike the engagement
        snapshots, which see only the visible mobile enemies."""
        if "noenglog" in FLAGS:
            return
        hits = self.__dict__.setdefault("_hits", {}).pop(u.uid, None) or {}
        killer_id = int(event.get("attackerId") or -1)
        k = self._attacker(killer_id) if hasattr(self, "_attacker") else None
        near_e = [e for e in self.contacts().values()
                  if not e.is_air and self.dist(e.pos, u.pos) < 700.0 and (e.is_combat or not e.is_mobile)]
        near_o = [o for uid, o in self.units.items()
                  if uid != u.uid and o.is_combat and o.is_mobile and self.dist(o.pos, u.pos) < 700.0]
        span = self.dist(self.base_pos, self.enemy_base or self.map.center) or 1.0
        if k is not None and k.name.startswith("def"):
            k = None
        if k is not None:
            killer = k.name
        else:
            raw = "?noid" if killer_id <= 0 else ("?own" if killer_id in self.units else "?unseen")
            # Late labels: the def index may have learned the weapon since.
            killer = self.resolve_attacker(raw, hits.get("wd"))
            if killer == raw and hits.get("last") and not str(hits["last"]).startswith("?"):
                killer = hits["last"]
        rec = {
            "t": round(self.frame_time()),
            "name": u.name,
            "cost": round(units.cost(u.data)),
            "branch": getattr(self, "unit_branch", {}).get(u.uid, "none"),
            "pos": [round(v) for v in u.pos],
            "depth": round(self.dist(self.base_pos, u.pos) / span, 2),
            "killer": killer,
            "built": u.is_finished,
            "last_hit": hits.get("last"),
            "weapon_def": hits.get("wd"),
            "hit_for": round((self.frame - hits["f0"]) / 30.0, 1) if hits else None,
            "dmg_by": {a: round(d) for a, d in hits.get("by", Counter()).most_common(4)},
            "near_enemy": dict(Counter(e.name for e in near_e).most_common(6)),
            "near_enemy_power": round(sum(combat_power(e) for e in near_e if e.is_mobile)),
            "near_enemy_turrets": sum(1 for e in near_e if not e.is_mobile and units.is_defense(e.data)),
            "near_own_power": round(sum(combat_power(o) for o in near_o)),
        }
        f = getattr(self, "_death_file", None)
        if f is None:
            try:
                path = os.path.join(os.environ.get("ZK_ARENA_OUT", "."), f"deaths-team{self.team_id}.jsonl")
                f = self._death_file = open(path, "a", encoding="utf8", buffering=1)
            except OSError:
                self._death_file = False
                return
        if f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def _side_features(self, us: list) -> dict:
        cost = sum(units.cost(u.data) for u in us) or 1.0

        def wavg(f):
            return sum(f(u) * units.cost(u.data) for u in us) / cost

        hmap = getattr(self.map, "ground_height", None)
        return {
            "n": len(us),
            "cost": round(cost),
            "hp": round(sum(u.health for u in us)),
            "power": round(sum(combat_power(u) for u in us), 1),
            "dps": round(sum(ground_dps(u) for u in us), 1),
            "range": round(wavg(weapon_range)),
            "speed": round(wavg(lambda u: float(u.data.get("speed") or 0.0) if u.data else 0.0), 1),
            "turret_cost": round(sum(units.cost(u.data) for u in us if not u.is_mobile)),
            "roles": dict(Counter(units.role_of(u.data) for u in us)),
            "height": round(sum(hmap(*u.pos) for u in us) / len(us), 1) if hmap and us else 0.0,
        }

    def _write_engagement(self, eng: dict) -> None:
        dead = getattr(self, "enemy_dead", {})
        our_lost = 0.0
        for uid, (cost, frac0) in eng["ours"].items():
            u = self.units.get(uid)
            our_lost += cost * (frac0 if u is None else max(0.0, frac0 - u.alive_frac))
        their_lost = their_unknown = 0.0
        for uid, (cost, frac0) in eng["theirs"].items():
            if uid in dead:
                their_lost += cost * frac0
                continue
            e = self.enemies.get(uid)
            if e is None:
                their_unknown += cost
            else:
                their_lost += cost * max(0.0, frac0 - e.alive_frac)
        rec = dict(eng["features"])
        rec["pos"] = [round(v) for v in eng["pos"]]
        rec["our_lost"] = round(our_lost)
        rec["their_lost"] = round(their_lost)
        rec["their_unknown"] = round(their_unknown)
        f = getattr(self, "_eng_file", None)
        if f is None:
            try:
                path = os.path.join(os.environ.get("ZK_ARENA_OUT", "."), f"engagements-team{self.team_id}.jsonl")
                f = self._eng_file = open(path, "a", encoding="utf8", buffering=1)
            except OSError:
                self._eng_file = False
                return
        if f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.engagements_logged = getattr(self, "engagements_logged", 0) + 1
