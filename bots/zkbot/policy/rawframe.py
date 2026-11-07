"""The raw frame: one decision's view of the game, as a player has it.

The sim and the live bot both produce this, and the featurizer reads nothing else, so this file is the only definition
of the schema.  sim/include/zksim/rawframe.hpp is generated from it (`python tools/policy_schema.py`) and a test
compares the two.  It holds only what the live bot can really supply: no facing, weapon reload or pending shots.

Three additions to docs/learned-policy/02-policy.md section 2.2, all because a stateless featurizer needs them: the
flags DAMAGED (for the commit lock) and RESERVED (units the rules keep), and `ord_age` on a remembered structure, which
says how long ago it was seen.  Later, the flag JUMP_READY: an own unit's jump is charged, as its rules param says.

Whose a unit is and what is known of it are separate: `player` and `ally_team` are the game's own numbers, -1 if unknown, and
`knowledge` is how the viewer knows of it.  `player` is the engine's team, the one that holds the unit.  The viewer's own two
numbers are in the globals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import catalog, units

SCHEMA_VERSION = 4

KNOW_LOS, KNOW_RADAR, KNOW_MEMORY, KNOW_HIDDEN = 0, 1, 2, 3
UNKNOWN_ID = -1

BEING_BUILT, CLOAKED, PARALYZED, IN_LOS, IDLE, TYPE_KNOWN, DAMAGED, RESERVED, JUMP_READY, PULLING = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512

SITE_FREE, SITE_OURS, SITE_ENEMY, SITE_BUILDING = 0, 1, 2, 3
_SITE_STATUS = {"free": SITE_FREE, "ours": SITE_OURS, "enemy": SITE_ENEMY, "building": SITE_BUILDING}

# Order kinds: 02-policy.md section 4.1.
KEEP, MOVE, FIGHT, ATTACK_UNIT, ATTACK_GROUND, STOP, MANUAL_FIRE, JUMP, CLOAK_ON, CLOAK_OFF = range(10)
BUILD, ASSIST, REPAIR, RECLAIM, PRODUCE, RESERVED_KIND = range(10, 16)
ORDER_SLOTS = 16        # of a token's order memory, in which an order's kind is one of those above: the frames and encoders made so far have these
THROW, PUSH, PULL, PLACE_BEACON, WAIT_AT_BEACON = range(16, 21)
KIND_COUNT = 21
METAL_RESOURCE = "Metal"
_ORDER_KIND = {"move": MOVE, "fight": FIGHT, "attack": ATTACK_UNIT, "attack_ground": ATTACK_GROUND, "stop": STOP}

# Largest fields first, so a record has no padding and the C++ struct needs no pragma.
UNIT_FIELDS = (
    ("uid", "i8"), ("ord_target", "i8"), ("def_id", "i4"),
    ("x", "f4"), ("y", "f4"), ("z", "f4"), ("vx", "f4"), ("vz", "f4"),
    ("hp_frac", "f4"), ("build_frac", "f4"), ("experience", "f4"), ("blip_top_speed", "f4"),
    ("ord_age", "f4"), ("ord_x", "f4"), ("ord_z", "f4"),
    ("flags", "u2"), ("knowledge", "u1"), ("ord_kind", "i1"), ("player", "i2"), ("ally_team", "i2"), ("pad", "i4"),
)
SITE_FIELDS = (("site_id", "i4"), ("x", "f4"), ("z", "f4"), ("income", "f4"), ("region_id", "i2"), ("status", "u1"), ("pad", "u1"))
FEATURE_FIELDS = (("feat_id", "i8"), ("x", "f4"), ("z", "f4"), ("metal", "f4"), ("def_id", "i4"))
UNIT_DTYPE = np.dtype(list(UNIT_FIELDS), align=True)
SITE_DTYPE = np.dtype(list(SITE_FIELDS), align=True)
FEATURE_DTYPE = np.dtype(list(FEATURE_FIELDS), align=True)

GLOBAL_FIELDS = (
    "frame", "metal", "metal_income", "metal_storage", "energy", "energy_income", "energy_storage",
    "commander_hp", "commander_uid", "base_x", "base_z", "map_w", "map_h",
    "frames_since_contact", "unfinished_structures", "idle_builders", "player", "ally_team",
)
G = {name: i for i, name in enumerate(GLOBAL_FIELDS)}


@dataclass
class Frame:
    units: np.ndarray
    sites: np.ndarray
    glob: np.ndarray
    features: np.ndarray = field(default_factory=lambda: np.zeros(0, FEATURE_DTYPE))

    @property
    def idle_uids(self) -> np.ndarray:
        own = self.units[self.units["player"] == int(self.glob[G["player"]])]
        return own["uid"][(own["flags"] & IDLE) != 0]


class LiveMemory:
    """What the live producer carries between decisions."""

    def __init__(self) -> None:
        self.idle: dict[int, int] = {}          # uid -> frame of its last UNIT_IDLE
        self.health: dict[int, float] = {}      # uid -> health at the last decision
        self.owners: dict[int, tuple[int, int]] = {}     # uid -> player and ally team, when last seen
        self.last_owner: tuple[int, int] = (UNKNOWN_ID, UNKNOWN_ID)     # for a structure seen before the frames began

    def note_events(self, obs: dict[str, Any]) -> None:
        frame = int(obs.get("frame") or 0)
        for ev in obs.get("events") or []:
            if isinstance(ev, dict) and ev.get("topic") == 7 and ev.get("unitId") is not None:
                self.idle[int(ev["unitId"])] = frame

    @staticmethod
    def reserved(brain: Any) -> set[int]:
        """Units the rules keep even in policy mode: Gremlin scouts and the volley-firing Phantoms."""
        keep = set(getattr(brain, "gremlin_scouts", None) or ())
        keep.update(uid for uid, u in brain.units.items() if u.name == "cloaksnipe")
        return keep


def _hostile_owner(uid: int, raw: dict[str, Any], memory: LiveMemory) -> tuple[int, int]:
    """Every unit in the observation has its ally team.  ControllerAI doesn't send the player yet, so the alliance stands for it:
    right in a 1v1, and only coarse with several players on a side."""
    ally = int(raw.get("allyTeam", UNKNOWN_ID))
    owner = (int(raw.get("team", ally)), ally)
    memory.owners[uid] = memory.last_owner = owner
    return owner


def _record(rec: np.void, uid: int, knowledge: int, owner: tuple[int, int], raw: dict[str, Any], flags: int) -> None:
    pos = raw.get("pos") or (0.0, 0.0, 0.0)
    vel = raw.get("vel") or (0.0, 0.0, 0.0)
    def_id = int(raw.get("defId") or 0) or -1
    rec["uid"], rec["knowledge"], rec["def_id"] = uid, knowledge, def_id
    rec["player"], rec["ally_team"] = owner
    rec["x"], rec["y"], rec["z"] = float(pos[0]), float(pos[1]), float(pos[2])
    rec["vx"], rec["vz"] = float(vel[0]), float(vel[2])
    rec["ord_kind"], rec["ord_target"] = -1, -1
    rec["hp_frac"] = float(raw.get("health") or 0.0) / float(raw["maxHealth"]) if raw.get("maxHealth") else np.nan
    rec["build_frac"] = float(raw.get("buildProgress") or 0.0)
    rec["experience"] = float(raw.get("experience") or 0.0)
    if raw.get("isBeingBuilt"):
        flags |= BEING_BUILT
    if raw.get("isCloaked"):
        flags |= CLOAKED
    if raw.get("isParalyzed"):
        flags |= PARALYZED
    if def_id > 0:
        flags |= TYPE_KNOWN
    rec["flags"] = flags


def _jump_charged(brain: Any, uid: int, raw: dict[str, Any]) -> bool:
    """unit_jumpjets.lua's reload, in a rules param only its team can read.  A brain with no game behind it has never jumped.
    A thrower's reload is a weapon's, which ControllerAI doesn't show, so it counts as ready: the game makes it wait to fire."""
    d = units.BY_ID.get(int(raw.get("defId") or 0))
    if d and d["name"] in catalog.SPECIALS["throwers"]:
        return True
    if not d or not d.get("isMobile") or "jump" not in catalog.tags(d["name"]):
        return False
    client = getattr(brain, "client", None)
    if client is None:
        return True
    answer = client.query({"type": "unit_rules_param", "unitId": uid, "key": "jumpReload", "defaultFloat": 1.0})
    return float((answer or {}).get("floatValue", 1.0)) >= 1.0


def from_brain(brain: Any, obs: dict[str, Any], memory: LiveMemory, region_of: Any = None) -> Frame:
    """Reads the observation and the Brain, and changes neither.  `memory` holds what one frame can't: which units went
    idle, and each unit's health at the last decision."""
    frame = int(obs.get("frame") or 0)
    own, enemies, blips = obs.get("units") or {}, obs.get("enemies") or {}, obs.get("radarBlips") or {}
    me = int(obs.get("teamId", brain.team_id)), int(obs.get("allyTeamId", brain.ally_team_id))
    ghosts = {uid: entry for uid, entry in (getattr(brain, "enemy_structures", None) or {}).items()
              if str(uid) not in enemies and str(uid) not in blips}
    units = np.zeros(len(own) + len(enemies) + len(blips) + len(ghosts), UNIT_DTYPE)
    i = 0
    reserved = memory.reserved(brain)
    orders = brain.unit_order
    healths: dict[int, float] = {}
    idle_builders = 0
    for uid_s, raw in own.items():
        uid = int(uid_s)
        rec = units[i]
        order = orders.get(uid)
        flags = IN_LOS
        if uid in memory.idle and (order is None or order[4] <= memory.idle[uid]):
            flags |= IDLE
            known = brain.units.get(uid)
            idle_builders += 1 if known is not None and known.is_builder else 0
        if uid in reserved:
            flags |= RESERVED
        health = float(raw.get("health") or 0.0)
        healths[uid] = health
        if health < memory.health.get(uid, health):
            flags |= DAMAGED
        if _jump_charged(brain, uid, raw):
            flags |= JUMP_READY
        _record(rec, uid, KNOW_LOS, (int(raw.get("team", me[0])), int(raw.get("allyTeam", me[1]))), raw, flags)
        if order is not None:
            rec["ord_kind"] = _ORDER_KIND.get(order[0], -1)
            rec["ord_x"], rec["ord_z"] = float(order[1]), float(order[2])
            rec["ord_target"] = -1 if order[3] is None else int(order[3])
            rec["ord_age"] = frame - int(order[4])
        i += 1
    memory.health = healths
    for uid_s, raw in enemies.items():
        _record(units[i], int(uid_s), KNOW_LOS, _hostile_owner(int(uid_s), raw, memory), raw, IN_LOS)
        i += 1
    tracks = getattr(brain, "blip_tracks", None) or {}
    for uid_s, raw in blips.items():
        uid = int(uid_s)
        _record(units[i], uid, KNOW_RADAR, _hostile_owner(uid, raw, memory), raw, 0)
        units[i]["hp_frac"] = np.nan
        units[i]["build_frac"] = 0.0
        units[i]["blip_top_speed"] = float((tracks.get(uid) or {}).get("smax") or 0.0)
        i += 1
    for uid, (u, seen) in ghosts.items():
        rec = units[i]
        rec["uid"], rec["knowledge"], rec["def_id"] = uid, KNOW_MEMORY, u.def_id or -1
        rec["player"], rec["ally_team"] = memory.owners.get(uid, memory.last_owner)
        rec["x"], rec["y"], rec["z"] = u.x, u.y, u.z
        rec["hp_frac"] = u.health / max(1.0, u.max_health)
        rec["build_frac"] = u.build_progress
        rec["flags"] = TYPE_KNOWN | (BEING_BUILT if u.is_being_built else 0)
        rec["ord_kind"], rec["ord_target"] = -1, -1
        rec["ord_age"] = frame - int(seen)
        i += 1
    features = _features(obs)
    spots = brain.map.spots
    sites = np.zeros(len(spots), SITE_DTYPE)
    for k, s in enumerate(spots):
        sites[k]["site_id"], sites[k]["x"], sites[k]["z"], sites[k]["income"] = k, s.x, s.z, s.income
        sites[k]["status"] = _SITE_STATUS.get(getattr(s, "status", "free"), SITE_FREE)
        sites[k]["region_id"] = -1 if region_of is None else int(region_of(s.x, s.z))
    commander = brain.units.get(brain.commander_id) if brain.commander_id else None
    glob = np.zeros(len(GLOBAL_FIELDS), np.float32)
    values = {
        "frame": frame,
        "metal": brain.metal_current(), "metal_income": brain.metal_income(), "metal_storage": brain.effective_storage(brain.metal()),
        "energy": brain.energy_current(), "energy_income": brain.energy_income(), "energy_storage": brain.effective_storage(brain.energy()),
        "commander_hp": commander.alive_frac if commander else 0.0, "commander_uid": commander.uid if commander else -1,
        "base_x": brain.base_pos[0], "base_z": brain.base_pos[1], "map_w": brain.map.width, "map_h": brain.map.height,
        "frames_since_contact": max(0, frame - int(getattr(brain, "last_enemy_seen_frame", 0) or 0)),
        "unfinished_structures": sum(1 for u in brain.units.values() if not u.is_mobile and not u.is_finished),
        "idle_builders": idle_builders, "player": me[0], "ally_team": me[1],
    }
    for name, value in values.items():
        glob[G[name]] = float(value)
    return Frame(units, sites, glob, features)


def _features(obs: dict[str, Any]) -> np.ndarray:
    """ControllerAI sends what's reclaimable and in sight, with its def's metal and the share of that which is left."""
    seen = obs.get("features") or {}
    out = np.zeros(len(seen), FEATURE_DTYPE)
    for i, (key, raw) in enumerate(seen.items()):
        pos = raw.get("pos") or (0.0, 0.0, 0.0)
        whole = float(((raw.get("definition") or {}).get("containedResources") or {}).get(METAL_RESOURCE) or 0.0)
        out[i]["feat_id"], out[i]["def_id"] = int(key), int(raw.get("defId", UNKNOWN_ID))
        out[i]["x"], out[i]["z"] = float(pos[0]), float(pos[2])
        out[i]["metal"] = whole * float(raw.get("reclaimLeft", 1.0))
    return out


def save_frames(path: Any, frames: list[Frame], **extra: Any) -> None:
    """A list of frames as one compressed file."""
    np.savez_compressed(
        path, schema_version=SCHEMA_VERSION,
        units=np.concatenate([f.units for f in frames]) if frames else np.zeros(0, UNIT_DTYPE),
        sites=np.concatenate([f.sites for f in frames]) if frames else np.zeros(0, SITE_DTYPE),
        features=np.concatenate([f.features for f in frames]) if frames else np.zeros(0, FEATURE_DTYPE),
        feature_counts=np.array([len(f.features) for f in frames], np.int32),
        unit_counts=np.array([len(f.units) for f in frames], np.int32),
        site_counts=np.array([len(f.sites) for f in frames], np.int32),
        globs=np.stack([f.glob for f in frames]) if frames else np.zeros((0, len(GLOBAL_FIELDS)), np.float32),
        **extra,
    )


def load_frames(path: Any) -> list[Frame]:
    with np.load(path) as d:
        # Version 2 is this without features.
        if int(d["schema_version"]) not in (2, 3, SCHEMA_VERSION):
            raise ValueError(f"{path}: schema {int(d['schema_version'])}, this code reads {SCHEMA_VERSION}")
        if not len(d["unit_counts"]):
            return []
        units = np.split(d["units"], np.cumsum(d["unit_counts"])[:-1])
        sites = np.split(d["sites"], np.cumsum(d["site_counts"])[:-1])
        if "features" in d:
            features = np.split(d["features"], np.cumsum(d["feature_counts"])[:-1])
        else:
            features = [np.zeros(0, FEATURE_DTYPE)] * len(units)
        return [Frame(u, s, g, f) for u, s, g, f in zip(units, sites, d["globs"], features)]


_C_TYPES = {"i8": "int64_t", "i4": "int32_t", "i2": "int16_t", "i1": "int8_t", "u2": "uint16_t", "u1": "uint8_t", "f4": "float"}


def _struct(name: str, fields: tuple[tuple[str, str], ...], dtype: np.dtype) -> list[str]:
    lines = [f"struct {name} {{"]
    lines += [f"    {_C_TYPES[t]} {n};" for n, t in fields]
    lines += ["};", f'static_assert(sizeof({name}) == {dtype.itemsize}, "{name}: rawframe.py disagrees");', ""]
    return lines


def cpp_header() -> str:
    """sim/include/zksim/rawframe.hpp."""
    knowledge = (("LOS", KNOW_LOS), ("RADAR", KNOW_RADAR), ("MEMORY", KNOW_MEMORY), ("HIDDEN", KNOW_HIDDEN))
    flags = (("BEING_BUILT", BEING_BUILT), ("CLOAKED", CLOAKED), ("PARALYZED", PARALYZED), ("IN_LOS", IN_LOS),
             ("IDLE", IDLE), ("TYPE_KNOWN", TYPE_KNOWN), ("DAMAGED", DAMAGED), ("RESERVED", RESERVED), ("JUMP_READY", JUMP_READY), ("PULLING", PULLING))
    lines = ["// Generated by tools/policy_schema.py from bots/zkbot/policy/rawframe.py.  Don't edit.", "#pragma once", "",
             "#include <cstdint>", "", "namespace zk {", "", f"constexpr int RAWFRAME_SCHEMA = {SCHEMA_VERSION};", ""]
    lines += [f"constexpr int KNOW_{n} = {v};" for n, v in knowledge] + [""]
    lines += [f"constexpr int UF_{n} = {v};" for n, v in flags] + [""]
    lines += _struct("UnitRecord", UNIT_FIELDS, UNIT_DTYPE) + _struct("SiteRecord", SITE_FIELDS, SITE_DTYPE) + _struct("FeatureRecord", FEATURE_FIELDS, FEATURE_DTYPE)
    lines += ["enum GlobalField {"] + [f"    G_{n.upper()} = {i}," for i, n in enumerate(GLOBAL_FIELDS)]
    lines += [f"    G_COUNT = {len(GLOBAL_FIELDS)}", "};", "", "}  // namespace zk", ""]
    return "\n".join(lines)
