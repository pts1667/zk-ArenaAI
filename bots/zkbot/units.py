"""Static unit data helpers.

The data file is generated from the live game's UnitDefs.  Unit IDs are stable
for this Zero-K version; the bot also receives the metadata table at startup and
can fall back to names if an id is unfamiliar.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from .params import FLAGS as _FLAGS


_HERE = pathlib.Path(__file__).resolve().parent
_DATA_PATH = _HERE / "data" / "units.json"

with _DATA_PATH.open("r", encoding="utf8") as f:
    _RAW: dict[str, dict[str, Any]] = json.load(f)

BY_ID: dict[int, dict[str, Any]] = {int(k): v for k, v in _RAW.items()}
BY_NAME: dict[str, dict[str, Any]] = {v["name"]: v for v in _RAW.values()}


def by_name(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    return BY_NAME.get(name)


def by_id(def_id: int | None) -> dict[str, Any] | None:
    if def_id is None:
        return None
    return BY_ID.get(int(def_id))


def name_of(def_id: int | None) -> str:
    d = by_id(def_id)
    return d["name"] if d else f"def{def_id}"


def id_of(name: str) -> int | None:
    d = BY_NAME.get(name)
    return int(d["id"]) if d else None


def is_combat(d: dict[str, Any] | None) -> bool:
    if not d:
        return False
    if d.get("isBuilder") or d.get("isFactory"):
        return False
    if (d.get("role") or "") == "bomb":
        # Cloak bombs are expendable mobile explosives, not builders.
        return bool(d.get("isMobile"))
    return d.get("isMobile", False) and (
        (d.get("maxWeaponRange") or 0) > 0 or (d.get("power") or 0) > 0
    )


def is_builder(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isBuilder") and not d.get("isFactory"))


def is_factory(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isFactory"))


def is_mex(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("name") == "staticmex")


def is_defense(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isDefense"))


def is_air(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isAir"))


def cost(d: dict[str, Any] | None) -> float:
    return float((d or {}).get("costM") or 0)


# units.json leaves several mainline units as a generic "combat" role, which
# the composition counters ignore.  Buoy alone dealt most of the damage in
# our amphibious losses while counting as nothing.
ROLE_OVERRIDES = {
    "amphfloater": "skirm",
    "amphimpulse": "riot",
    "amphlaunch": "arty",
    "shieldfelon": "skirm",
    "spideremp": "raid",
    "jumpsumo": "assault",
    "spidercrabe": "assault",
    "hoverdepthcharge": "riot",
    "gunshipkrow": "assault",
}


def role_of(d: dict[str, Any] | None) -> str:
    if not d:
        return "other"
    r = d.get("role") or "other"
    if r == "combat":
        name = d.get("name") or ""
        if name in ROLE_OVERRIDES:
            return ROLE_OVERRIDES[name]
        # Generic ground units (striders, new or rebalanced units) get a
        # role from their stats instead of counting as nothing.  Air keeps
        # "combat": the air code handles aircraft by name.
        if d.get("isMobile") and not d.get("isAir") and "noinfer" not in _FLAGS:
            from . import catalog

            inferred = catalog.infer_role(name)
            if inferred not in ("other", "static", "builder"):
                return inferred
    return r
