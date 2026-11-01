"""Static unit data helpers.

The data file is generated from the live game's UnitDefs.  Unit IDs are stable
for this Zero-K version; the bot also receives the metadata table at startup and
can fall back to names if an id is unfamiliar.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

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


def is_energy(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isEnergy"))


def is_defense(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isDefense"))


def is_aa(d: dict[str, Any] | None) -> bool:
    if not d:
        return False
    n = d.get("name", "")
    return n.endswith("aa") or "aa" in n or n.startswith("turretaa") or d.get("role") == "aa"


def is_air(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("isAir"))


def cost(d: dict[str, Any] | None) -> float:
    return float((d or {}).get("costM") or 0)


def value(d: dict[str, Any] | None) -> float:
    """Rough combat value of a unit, used only for high-level comparisons."""
    if not d:
        return 0.0
    return max(float(d.get("costM") or 0.0), 1.0)


def role_of(d: dict[str, Any] | None) -> str:
    return (d or {}).get("role") or "other"


# Factory unit preferences.  These names are from the standard Zero-K factories.
FACTORY_UNITS = {
    "factorycloak": {
        "builder": "cloakcon",
        "raid": "cloakraid",
        "heavyraid": "cloakheavyraid",
        "riot": "cloakriot",
        "skirm": "cloakskirm",
        "assault": "cloakassault",
        "arty": "cloakarty",
        "aa": "cloakaa",
        "snipe": "cloaksnipe",
        "bomb": "cloakbomb",
        "bomb": "cloakbomb",
        "jammer": "cloakjammer",
    },
    "factoryveh": {
        "builder": "vehcon",
        "raid": "vehraid",
        "scout": "vehscout",
        "skirm": "vehsupport",
        "riot": "vehriot",
        "assault": "vehassault",
        "heavyarty": "vehheavyarty",
        "aa": "vehaa",
    },
    "factoryhover": {
        "builder": "hovercon",
        "raid": "hoverraid",
        "heavyraid": "hoverheavyraid",
        "skirm": "hoverskirm",
        "riot": "hoverriot",
        "assault": "hoverassault",
        "arty": "hoverarty",
        "aa": "hoveraa",
    },
    "factoryplane": {
        "builder": "planecon",
        "scout": "planescout",
        "fighter": "planefighter",
        "heavyfighter": "planeheavyfighter",
        "bomb": "bomberprec",
        "riot": "bomberriot",
        "heavy": "bomberheavy",
        "assault": "bomberassault",
        "aa": "planefighter",
    },
    "factorygunship": {
        "builder": "gunshipcon",
        "scout": "gunshipcon",
        "emp": "gunshipemp",
        "raid": "gunshipraid",
        "skirm": "gunshipskirm",
        "assault": "gunshipassault",
        "aa": "gunshipaa",
    },
    "striderhub": {
        "assault": "striderdante",
        "heavy": "striderbantha",
        "riot": "striderscorpion",
        "arty": "striderarty",
        "aa": "striderantiheavy",
        "antiheavy": "striderantiheavy",
    },
    "factoryspider": {
        "builder": "spidercon",
        "scout": "spiderscout",
        "assault": "spiderassault",
        "riot": "spiderriot",
        "skirm": "spiderskirm",
        "aa": "spideraa",
        "antiheavy": "spiderantiheavy",
    },
    "factoryjump": {
        "builder": "jumpcon",
        "scout": "jumpscout",
        "raid": "jumpraid",
        "skirm": "jumpskirm",
        "assault": "jumpassault",
        "arty": "jumparty",
        "aa": "jumpaa",
    },
    "factorytank": {
        "builder": "tankcon",
        "raid": "tankraid",
        "heavyraid": "tankheavyraid",
        "riot": "tankriot",
        "assault": "tankassault",
        "heavyassault": "tankheavyassault",
        "arty": "tankarty",
        "aa": "tankaa",
    },
    "factoryshield": {
        "builder": "shieldcon",
        "scout": "shieldscout",
        "raid": "shieldraid",
        "skirm": "shieldskirm",
        "riot": "shieldriot",
        "assault": "shieldassault",
    },
    "factoryamph": {
        "builder": "amphcon",
        "raid": "amphraid",
        "riot": "amphriot",
        "assault": "amphassault",
        "aa": "amphaa",
    },
}


def factory_unit(factory_name: str, role: str) -> int | None:
    table = FACTORY_UNITS.get(factory_name) or {}
    return id_of(table.get(role, ""))
