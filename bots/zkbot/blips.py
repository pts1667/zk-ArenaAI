"""Radar blip identification and raid early warning (BlipMixin).

Radar blips carry an exact velocity (`vel`, elmos/frame) but no `defId`
unless the engine already knows the unit's type from earlier LOS.  An
unidentified blip used to be a `def0` contact that nothing counted.

Identification: the blip's sustained top speed (minimum speed over a ~2 s
window, maximum over its life) is a lower bound on its unit type's speed.
Candidates are the mobile build options of the enemy's known factories
(factories seen, or inferred from the units seen); air/ground comes from the
blip's height above the ground.  Candidates are weighted by how often the
enemy has fielded them, by closeness of their top speed to the observed one
and mildly by cheapness.  When one role dominates, the blip takes the most
likely type of that role, so the threat map, local balance, responders and
the slow-ball detector treat it like a seen unit.  Guesses are checked when
the unit enters LOS and the accuracy is logged.

Early warning: enemy raiders (seen or guessed, or any ground blip faster
than a raider threshold) whose straight-line track passes our mexes or
constructors within ~12 s raise a warning at that asset: constructors there
step aside and responders are sent to meet the raid.

Flags: `noblipid` (no identification), `noblipwarn` (no early warning),
`noblipcount` (guesses do not feed the unit-mix counters).
"""
from __future__ import annotations

import math
import os
from collections import deque
from typing import Any

from . import units
from .models import UnitInfo
from .threat import combat_power

FLAGS = os.environ.get("ZKBOT_FLAGS", "")

MOVING_SPEED = 20.0     # sustained elmos/s before speed says anything
RAIDER_SPEED = 85.0     # ground blips faster than this are raiders
SPEED_WINDOW = 60       # frames: sustained speed = min over this window
AIR_ALTITUDE = 60.0
GUESS_ROLE_SHARE = 0.45
WARN_HORIZON = (4.0, 8.0, 12.0)   # seconds of straight-line projection
WARN_RADIUS = 450.0

_FACTORY_OPTIONS: dict[str, list[dict[str, Any]]] | None = None


def factory_options() -> dict[str, list[dict[str, Any]]]:
    """factory name -> mobile unit defs it builds (from `fromFactory`)."""
    global _FACTORY_OPTIONS
    if _FACTORY_OPTIONS is None:
        out: dict[str, list[dict[str, Any]]] = {}
        for d in units.BY_NAME.values():
            fac = d.get("fromFactory") or (d.get("customParams") or {}).get("from_factory")
            if fac and d.get("isMobile") and float(d.get("speed") or 0.0) > 0.0:
                out.setdefault(str(fac), []).append(d)
        _FACTORY_OPTIONS = out
    return _FACTORY_OPTIONS


def _coarse_role(d: dict[str, Any]) -> str:
    r = units.role_of(d)
    if r in ("raid", "scout"):
        return "raid"
    if r in ("assault", "heavy", "antiheavy"):
        return "assault"
    return r


class BlipMixin:
    # ------------------------------------------------------------ tracking

    def blip_state(self) -> None:
        if not hasattr(self, "blip_tracks"):
            self.blip_tracks: dict[int, dict[str, Any]] = {}
            self.blip_guess: dict[int, str] = {}
            self.blip_hits = 0
            self.blip_role_hits = 0
            self.blip_checked = 0
            self.enemy_vel: dict[int, tuple[float, float]] = {}
            self.raid_warnings: list[dict[str, Any]] = []
            self._warn_frame = -10**9
            self._blip_fac_cache = (-10**9, set())

    def note_enemy_velocity(self, uid: int, raw: dict[str, Any]) -> None:
        v = raw.get("vel")
        if v:
            self.enemy_vel[uid] = (float(v[0]) * 30.0, float(v[2]) * 30.0)

    def track_blip(self, uid: int, raw: dict[str, Any], u: UnitInfo) -> None:
        """Update the speed/altitude track of a radar blip."""
        now = self.frame
        tr = self.blip_tracks.get(uid)
        if tr is None:
            tr = self.blip_tracks[uid] = {
                "speeds": deque(), "smax": 0.0, "air": 0, "ground": 0,
                "first": now, "gframe": -10**9, "guess": None,
            }
        vx, vz = self.enemy_vel.get(uid, (0.0, 0.0))
        spd = math.hypot(vx, vz)
        sp = tr["speeds"]
        sp.append((now, spd))
        while sp and now - sp[0][0] > SPEED_WINDOW:
            sp.popleft()
        if len(sp) >= 2 and sp[-1][0] - sp[0][0] >= SPEED_WINDOW * 0.6:
            tr["smax"] = max(tr["smax"], min(s for _f, s in sp))
        tr["last"] = now
        try:
            ground = max(0.0, self.map.ground_height(u.x, u.z))
            if u.y - ground > AIR_ALTITUDE:
                tr["air"] += 1
            else:
                tr["ground"] += 1
        except Exception:
            tr["ground"] += 1

    def prune_blip_tracks(self) -> None:
        if self.frame % 300 >= 12:
            return
        for uid in [k for k, tr in self.blip_tracks.items() if self.frame - tr.get("last", 0) > 1800]:
            self.blip_tracks.pop(uid, None)

    # ------------------------------------------------------ identification

    def enemy_factory_names(self) -> set[str]:
        f, cached = self._blip_fac_cache
        if self.frame - f < 150:
            return cached
        opts = factory_options()
        facs: set[str] = set()
        for name in getattr(self, "enemy_seen_ids", {}):
            d = units.by_name(name)
            if not d:
                continue
            if d.get("isFactory"):
                if name in opts:
                    facs.add(name)
            elif d.get("isMobile"):
                fac = d.get("fromFactory") or (d.get("customParams") or {}).get("from_factory")
                if fac in opts:
                    facs.add(str(fac))
        for _uid, (e, _fr) in getattr(self, "enemy_structures", {}).items():
            if e.is_factory and e.name in opts:
                facs.add(e.name)
        self._blip_fac_cache = (self.frame, facs)
        return facs

    def blip_candidates(self, tr: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
        """(probability, unit def) for a tracked blip, most likely first."""
        opts = factory_options()
        facs = self.enemy_factory_names() or set(opts)
        air = tr["air"] > tr["ground"]
        smax = tr["smax"]
        seen = getattr(self, "enemy_seen_ids", {})
        pool: dict[str, dict[str, Any]] = {}
        for f in facs:
            for d in opts.get(f, ()):
                pool[d["name"]] = d
        out = []
        for name, d in pool.items():
            if bool(d.get("isAir")) != air:
                continue
            if "commander" in name or "dyntrainer" in name:
                continue
            top = float(d.get("speed") or 0.0)
            if top <= 0.0 or top < smax * 0.93:
                continue
            w = 1.0 + 2.0 * len(seen.get(name, ()))
            w *= math.sqrt(150.0 / (units.cost(d) + 150.0))
            if smax >= MOVING_SPEED:
                w *= math.exp(-max(0.0, top - smax) / max(25.0, 0.35 * smax))
            out.append((w, d))
        total = sum(w for w, _d in out)
        if total <= 0.0:
            return []
        out.sort(key=lambda t: -t[0])
        return [(w / total, d) for w, d in out]

    def guess_blip(self, uid: int) -> dict[str, Any] | None:
        """Most likely unit def for an unidentified moving blip, or None."""
        tr = self.blip_tracks.get(uid)
        if tr is None or "noblipid" in FLAGS:
            return None
        if self.frame - tr["gframe"] < 30:
            g = tr["guess"]
            return units.by_name(g) if g else None
        tr["gframe"] = self.frame
        if tr["smax"] < MOVING_SPEED:
            tr["guess"] = None
            return None
        cands = self.blip_candidates(tr)
        if not cands:
            tr["guess"] = None
            return None
        shares: dict[str, float] = {}
        for p, d in cands:
            r = _coarse_role(d)
            shares[r] = shares.get(r, 0.0) + p
        role, share = max(shares.items(), key=lambda t: t[1])
        if share < GUESS_ROLE_SHARE:
            tr["guess"] = None
            return None
        best = next(d for p, d in cands if _coarse_role(d) == role)
        if tr["guess"] != best["name"]:
            self.log(
                f"blip {uid} speed {tr['smax']:.0f}{' air' if tr['air'] > tr['ground'] else ''}"
                f" -> {best['name']} ({role} {share:.2f}; top "
                + ", ".join(f"{d['name']}:{p:.2f}" for p, d in cands[:3])
                + ")"
            )
        tr["guess"] = best["name"]
        self.blip_guess[uid] = best["name"]
        return best

    def check_blip_guess(self, uid: int, actual: UnitInfo) -> None:
        """Called when a unit enters LOS: score the guess made for it."""
        g = self.blip_guess.pop(uid, None)
        if g is None:
            return
        self.blip_checked += 1
        if g == actual.name:
            self.blip_hits += 1
        gd = units.by_name(g) or {}
        if actual.data and _coarse_role(gd) == _coarse_role(actual.data):
            self.blip_role_hits += 1
        if self.blip_checked <= 20 or self.blip_checked % 10 == 0:
            self.log(
                f"blip check {uid}: guess {g} actual {actual.name}; "
                f"acc {self.blip_hits}/{self.blip_checked} role {self.blip_role_hits}/{self.blip_checked}"
            )

    def blip_guessed_roles(self) -> dict[str, int]:
        """Roles of guessed units never seen in LOS (for the mix counters)."""
        out: dict[str, int] = {}
        if "noblipcount" in FLAGS or "noblipid" in FLAGS:
            return out
        for _uid, name in getattr(self, "blip_guess", {}).items():
            d = units.by_name(name)
            if d:
                r = _coarse_role(d)
                out[r] = out.get(r, 0) + 1
        return out

    # ------------------------------------------------------- early warning

    def update_raid_warnings(self) -> list[dict[str, Any]]:
        """Group enemy raiders whose track passes one of our mexes or
        constructors within WARN_HORIZON seconds.  Cached for 15 frames."""
        if "noblipwarn" in FLAGS:
            return []
        if self.frame - self._warn_frame < 15:
            return self.raid_warnings
        self._warn_frame = self.frame
        assets: list[tuple[float, float]] = []
        for u in self.units.values():
            if u.name == "staticmex" and u.is_finished:
                assets.append(u.pos)
            elif u.is_builder and u.is_mobile and u.uid != self.commander_id:
                assets.append(u.pos)
        warns: list[dict[str, Any]] = []
        if assets:
            contacts = dict(self.radar)
            contacts.update(self.enemies)
            for uid, e in contacts.items():
                if e.is_air:
                    continue
                vx, vz = self.enemy_vel.get(uid, (0.0, 0.0))
                spd = math.hypot(vx, vz)
                if spd < 40.0:
                    continue
                if e.data:
                    if units.role_of(e.data) not in ("raid",) and "commander" not in e.name:
                        continue
                else:
                    tr = self.blip_tracks.get(uid)
                    if tr is None or tr["smax"] < RAIDER_SPEED or tr["air"] > tr["ground"]:
                        continue
                hit = None
                for t in WARN_HORIZON:
                    p = (e.x + vx * t, e.z + vz * t)
                    for a in assets:
                        if (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2 < WARN_RADIUS ** 2:
                            if (e.x - a[0]) ** 2 + (e.z - a[1]) ** 2 > 350.0 ** 2:
                                hit = (a, t)
                            break
                    if hit is not None:
                        break
                if hit is None:
                    continue
                a, t = hit
                power = combat_power(e) if e.data else 60.0
                for w in warns:
                    if self.dist(w["pos"], a) < 500.0:
                        w["power"] += power
                        w["n"] += 1
                        w["eta"] = min(w["eta"], t)
                        w["src"].append(e.pos)
                        break
                else:
                    warns.append({"pos": a, "power": power, "n": 1, "eta": t, "src": [e.pos]})
        for w in warns:
            sx = sum(p[0] for p in w["src"]) / len(w["src"])
            sz = sum(p[1] for p in w["src"]) / len(w["src"])
            w["from"] = (sx, sz)
        if warns and self.frame - getattr(self, "_warn_log", -10**9) >= 300:
            self._warn_log = self.frame
            self.log(
                "raid warning "
                + "; ".join(
                    f"{w['n']} raiders power {w['power']:.0f} eta {w['eta']:.0f}s -> "
                    f"({w['pos'][0]:.0f},{w['pos'][1]:.0f})"
                    for w in warns
                )
            )
        self.raid_warnings = warns
        return warns

    def warning_near(self, pos: tuple[float, float], eta: float = 10.0) -> dict[str, Any] | None:
        for w in self.update_raid_warnings():
            if w["eta"] <= eta and self.dist(w["pos"], pos) < WARN_RADIUS:
                return w
        return None
