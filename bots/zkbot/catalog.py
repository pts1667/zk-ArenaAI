"""Unit catalog: roles and factory rosters derived from unit stats.

The bot used to know units only by name (units.FACTORY_UNITS, ROLE_OVERRIDES
and per-name checks).  A balance patch or a new unit then silently falls
back to "raid" or "other".  This module derives what the bot needs from the
dumped UnitDefs (data/units.json) and weapon stats (data/weapons.json):

  role(name)                 stat-inferred role (raid/scout/riot/skirm/assault/
                             arty/snipe/aa/builder/bomb/other)
  roster(factory)            mobile units a factory builds
  pick(factory, role)        the cheapest roster unit with that role

The rules are relative and stat based (speed, range, HP per metal, DPS per
metal, area of effect), so they also hold for factories the bot has never
been tuned on.  tools/catalog_check.py compares them with the hand-written
roles; disagreements are listed there rather than patched here by name.
"""
from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Any

from . import units

_HERE = pathlib.Path(__file__).resolve().parent
try:
    with (_HERE / "data" / "weapons.json").open("r", encoding="utf8") as _f:
        WEAPONS: dict[str, dict[str, Any]] = json.load(_f)
except OSError:
    WEAPONS = {}
try:
    # Description, abilities and manual-fire weapons from the unit Lua files
    # (tools/extract_abilities.py): things stats can't show, like the
    # Dante's napalm salvo or the Reef's disarm missiles.
    with (_HERE / "data" / "abilities.json").open("r", encoding="utf8") as _f:
        ABILITIES: dict[str, dict[str, Any]] = json.load(_f)
except OSError:
    ABILITIES = {}

COMBAT_ROLES = ("raid", "scout", "riot", "skirm", "assault", "snipe", "arty", "aa", "antiheavy", "bomb")


def abilities(name: str) -> dict[str, Any]:
    return ABILITIES.get(name) or {}


def tags(name: str) -> frozenset[str]:
    """Ability tags: manualfire, cloak, area_cloak, shield, jump, emp,
    disarm, slow, fire, gravity, capture, torpedo, stockpile, kamikaze,
    transport, jammer, radar, drones, morph, ... plus script markers."""
    a = abilities(name)
    return frozenset(a.get("tags") or ()) | frozenset(a.get("scripts") or ())


def manual_weapon(name: str) -> dict[str, Any] | None:
    """The unit's manual-fire ("D-gun") weapon: range, damage, effects."""
    return abilities(name).get("manual_weapon")


def desc_roles(name: str) -> list[str]:
    """Roles named in the game's own description ("Assault/Riot Strider")."""
    return list(abilities(name).get("desc_roles") or [])

AIR_CATS = ("GUNSHIP", "FIXEDWING")


def stats(name: str) -> dict[str, float] | None:
    d = units.by_name(name)
    if not d:
        return None
    w = WEAPONS.get(name, {})
    cost = max(1.0, float(d.get("costM") or 0.0))
    hp = float(d.get("health") or 0.0)
    cat = str(d.get("category") or "")
    return {
        "cost": cost,
        "hp": hp,
        "speed": float(d.get("speed") or 0.0),
        "range": float(d.get("maxWeaponRange") or 0.0),
        "dps_ground": float(w.get("dps_ground") or 0.0),
        "dps_air": float(w.get("dps_air") or 0.0),
        "aoe": float(w.get("aoe") or 0.0),
        "hp_per_m": hp / cost,
        "dps_per_m": float(w.get("dps_ground") or 0.0) / cost,
        "air": 1.0 if any(c in cat for c in AIR_CATS) else 0.0,
        "unarmed": 1.0 if "UNARMED" in cat else 0.0,
        "toofast": 1.0 if "TOOFAST" in cat else 0.0,
    }


@lru_cache(maxsize=None)
def infer_role(name: str) -> str:
    """Role from the unit's description when it names a combat role, else
    from stats.  Use roles(name) for every role a unit can fill."""
    st = stat_role(name)
    named = [r for r in desc_roles(name) if r in COMBAT_ROLES]
    # Dual-role descriptions ("Raider/Riot Corvette"): the stats decide
    # between the named roles; otherwise the first named role wins.
    if st in named or not named:
        return st
    return named[0]


@lru_cache(maxsize=None)
def stat_role(name: str) -> str:
    """Role from stats alone.  Order matters: the first rule that fits wins."""
    d = units.by_name(name)
    s = stats(name)
    if not d or s is None:
        return "other"
    if d.get("isFactory") or not d.get("isMobile"):
        return "static"
    if units.is_builder(d):
        return "builder"
    if s["unarmed"] or s["range"] <= 0:
        # Bombs have no weapon but explode; the dump marks them.
        if d.get("role") == "bomb":
            return "bomb"
        return "scout" if s["speed"] >= 150 else "other"
    # Some weapons (torpedoes, launchers, gravity guns) have no usable DPS in
    # the weapon dump; classify those by range/speed/HP only.
    if s["dps_ground"] <= 0.5 < s["dps_air"]:
        return "aa"
    if s["cost"] <= 50:
        return "scout"
    if 650 <= s["range"] < 800 and s["cost"] >= 600 and s["hp_per_m"] < 1.0:
        return "snipe"
    if s["range"] >= 700 and s["hp_per_m"] < 4.0:
        return "arty"
    if s["speed"] >= 90 and s["cost"] <= 300 and s["range"] < 350 and s["hp_per_m"] < 5.0:
        return "raid"
    if s["range"] < 400 and (s["aoe"] >= 64 or s["dps_per_m"] >= 0.8):
        return "riot"
    if s["hp_per_m"] >= 5.0 or (s["cost"] >= 1500 and s["hp_per_m"] >= 3.0):
        return "assault"
    if s["range"] >= 350:
        return "skirm"
    return "assault"


def roles(name: str) -> list[str]:
    """Every role the unit can fill: the primary one first, then the other
    roles its description names (Dante: riot, assault)."""
    out = [role(name)]
    for r in desc_roles(name):
        if r not in out:
            out.append(r)
    return out


@lru_cache(maxsize=None)
def roster(factory: str) -> tuple[str, ...]:
    d = units.by_name(factory) or {}
    out = []
    for o in d.get("buildOptions") or []:
        n = o.get("name") if isinstance(o, dict) else units.name_of(o)
        u = units.by_name(n)
        if u and u.get("isMobile"):
            out.append(n)
    return tuple(out)


def role(name: str) -> str:
    """The hand-written role when there is a specific one, else inferred."""
    d = units.by_name(name)
    r = units.role_of(d)
    if r in ("combat", "other", "", None):
        return infer_role(name)
    return r


def pick(factory: str, want: str) -> str | None:
    """Cheapest unit of `factory` whose role is `want`."""
    cand = [n for n in roster(factory) if role(n) == want]
    if not cand:
        return None
    return min(cand, key=lambda n: float((units.by_name(n) or {}).get("costM") or 0.0))
