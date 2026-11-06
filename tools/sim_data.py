"""Builds the simulator's data from the engine dump and the map archives.

  python tools/sim_engine.py dump           # writes results/simtools/simdump (engine, 20-45 s)
  python tools/sim_data.py catalog          # sim/data/catalog.json, priority.npz; cross-checks
  python tools/sim_data.py maps "TitanDuel 2.2" "Carabanossi 1.0"     # sim/data/maps/*.npz: the gate's
  python tools/sim_data.py maps "Agalta v1.0" "Faultline v1.0" "FrostyCove v1.13" "Red Comet Remake 1.7"     # and training's: policy_sim_env.py
  python tools/sim_data.py splits           # tools/splits.json: game folds for the stage 1 gate

Runs in Windows Python: map reading needs 7-Zip.  Everything is converted here to frames and
elmos, so the C++ never sees seconds or heading units.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import random
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
sys.path.insert(0, str(ROOT / "tools"))

DUMP = ROOT / "results/simtools/simdump"
DATA = ROOT / "sim/data"
FRAMES = 30.0
HEADING_CIRCLE = 65536.0
SLOW_UPDATE_FRAMES = 16.0
MAX_MOUNTS = 8
MOVE_CLASS_NAMES = ("kbot", "tank", "spider", "amph", "hover", "ship")
MIN_SEGMENT_CELLS = 64               # smaller pieces join a neighbour
MIN_BORDER_CELLS = 8                 # 128 elmos; a narrower gap keeps two segments apart
# Merging neighbours by height and slope is off: it gives fewer segments, but worse routes and arrival
# (docs/learned-policy/01-simulator.md 8.6).  The flags are there to repeat that sweep.
HEIGHT_BAND = 0.0                    # elmos; the most a merged segment's pieces differ in mean height
SLOPE_BAND = 0.0                     # engine slope; the steepest mean border that merges
SPIDER_MIN_SLOPE = 0.6               # maxSlope in the moveDef; kbots are 0.41, all-terrain 0.97
AMPHIBIOUS_DEPTH = 1000.0
SM_TANK, SM_KBOT, SM_HOVER, SM_SHIP = 0, 1, 2, 3

KIND_INSTANT, KIND_PROJECTILE, KIND_GUIDED, KIND_SHIELD = 0, 1, 2, 3
INSTANT_TYPES = {"BeamLaser", "LightningCannon", "Melee"}
MISSILE_TYPES = {"MissileLauncher", "StarburstLauncher", "TorpedoLauncher"}
W_DISARM_ONLY, W_MANUAL, W_STOCKPILE, W_UNMODELLED = 256, 512, 1024, 2048
W_IMPACT_ONLY, W_TURRET, W_PARALYZER, W_NO_SELF_DAMAGE, W_NO_AUTO_TARGET, W_NO_FRIENDLY_FIRE, W_SLOW_ONLY, W_AREA_SLOWS = 1, 2, 4, 8, 16, 32, 64, 128
U_MOBILE, U_AIR, U_ARMED, U_FIREPROOF = 1, 2, 4, 8
DEFAULT_BURN_FRAMES = 450.0          # fire_defs.lua
SLOW_DECLINE_PER_FRAME = 0.04 / FRAMES   # timeslow_defs.lua DEGRADE_FACTOR
# Weapons that would rather hit something not yet disarmed: the penalty, and the status above which
# it counts (target_priority_defs.lua disarmPenaltyDefs, disarmWeaponTimeDefs after its conversion).
DISARM_TARGETING = {"shieldarty_emp_rocket": (200.0, 0.4), "shipscout_missile": (10.0, 0.33)}
DEFAULT_BURN_RANDOM = 0.3
DEFAULT_BURN_DAMAGE = 0.5
SHIELD_ARMOR = "4"

# A trainer commander's def lists every weapon its modules could add.  Level 1 has only its
# first one (gamedata/modularcomms/dyncomms_predefined.lua), and morphing renames the unit.
COMMANDER_WEAPONS = {
    "dyntrainer_strike_base": "commweapon_heavymachinegun",
    "dyntrainer_recon_base": "commweapon_heavymachinegun",
    "dyntrainer_assault_base": "commweapon_heavymachinegun",
    "dyntrainer_support_base": "commweapon_lparticlebeam",
}


def _custom(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float((d.get("customParams") or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def _frames(seconds: float) -> float:
    # The engine truncates to whole frames; the small margin absorbs values such as 0.30001.
    return max(1.0, math.floor(seconds * FRAMES + 1e-3))


def _weapon_kind(w: dict) -> int:
    if w.get("isShield"):
        return KIND_SHIELD
    if w["type"] in INSTANT_TYPES:
        return KIND_INSTANT
    if w["type"] in MISSILE_TYPES and w.get("tracks"):
        return KIND_GUIDED
    return KIND_PROJECTILE


def _weapon_record(w: dict, velocity: tuple[float, float, float], alpha: float) -> dict:
    custom = w.get("customParams") or {}
    damages = w["damages"]
    # Burst weapons fire from their script: the def's reload is the time between shots.
    reload_s, salvo, salvo_delay_s = w["reload"], w["salvoSize"], w["salvoDelay"]
    if "script_reload" in custom:
        salvo = _custom(w, "script_burst", 1.0)
        salvo_delay_s = reload_s
        reload_s = _custom(w, "script_reload")
    # Starburst missiles leave straight up, so the launcher needn't face the target.
    starburst = w["type"] == "StarburstLauncher"
    flags = W_TURRET if w.get("turret") or starburst else 0
    if w.get("impactOnly"):
        flags |= W_IMPACT_ONLY
    if w.get("paralyzer"):
        flags |= W_PARALYZER
    # A disarm weapon has the multiplier; "only" can be there and still be 0 (shipscout_missile).
    disarm_mult = _custom(w, "disarmdamagemult")
    if disarm_mult > 0.0 and _custom(w, "disarmdamageonly") > 0.0:
        flags |= W_DISARM_ONLY
    disarm_penalty, disarm_time = DISARM_TARGETING.get(w["name"], (0.0, 0.0))
    if w.get("noSelfDamage"):
        flags |= W_NO_SELF_DAMAGE
    if w.get("noAutoTarget") or w.get("manualFire"):
        flags |= W_NO_AUTO_TARGET
    if w.get("manualFire"):
        flags |= W_MANUAL
    if w.get("stockpile"):
        flags |= W_STOCKPILE
    # A script's AimWeapon returns false for these, so they never fire (amphfloater.lua).  The harmless
    # ones stay: commanders aim with them.
    if "fake_weapon" in custom and max(float(damages[str(i)]) for i in range(5)) >= 1.0:
        flags |= W_NO_AUTO_TARGET
    if "nofriendlyfire" in custom:
        flags |= W_NO_FRIENDLY_FIRE
    if "timeslow_onlyslow" in custom:
        flags |= W_SLOW_ONLY
    if custom.get("area_damage_is_slow") == "1":
        flags |= W_AREA_SLOWS
    # Impulse areas push rather than hurt, so they count as no area here.
    has_area = "area_damage_dps" in custom and custom.get("area_damage_is_impulse") != "1"
    on_fire = "setunitsonfire" in custom
    fire_starter = float(w.get("fireStarter") or 0.0)
    damage = [float(damages[str(i)]) for i in range(5)]
    # What these are for isn't in the sim, so no army may rely on them: capture (a capture weapon does
    # no health damage, unit_capture.lua), spawned units, pushing and pulling, throwing.
    if {"is_capture", "spawns_name", "impulse", "thower_weapon", "area_damage_is_impulse"} & custom.keys():
        flags |= W_UNMODELLED
    if "is_capture" in custom:
        damage = [0.0] * 5
    # Slow: timeslow_defs.lua.  Slow-only weapons default to 1; an absolute amount scales with the damage.
    raw_damage = _custom(w, "raw_damage")
    slow_factor = _custom(w, "timeslow_damagefactor", 1.0 if "timeslow_onlyslow" in custom else 0.0)
    if "timeslow_damage" in custom and raw_damage > 0.0:
        slow_factor = _custom(w, "timeslow_damage") / raw_damage
    # The Felon's gun fires from its shield's charge (shieldfelon.lua).  On shields this is an energy cost.
    shield_cost = _custom(w, "shield_drain") if not w.get("isShield") else 0.0
    if "shield_damage" in custom:
        damage[int(SHIELD_ARMOR)] = _custom(w, "shield_damage")
    para_seconds = _custom(w, "emp_paratime", _custom(w, "disarmtimer", float(damages.get("paralyzeDamageTime") or 0.0)))
    return {
        "name": w["name"], "damage": damage,
        "range": w["range"], "reload": _frames(reload_s), "salvo": max(1.0, float(salvo)),
        "salvo_delay": _frames(salvo_delay_s) if salvo > 1 else 0.0,
        "projectiles": max(1.0, float(w["projectiles"])),
        "speed": w["projectilespeed"], "accel": w.get("weaponAcceleration") or 0.0,
        "start_speed": w.get("startvelocity") or 0.0,
        "blast_radius": w["damageAreaOfEffect"],     # already halved by the engine's scaleValue
        "edge": w["edgeEffectiveness"], "height_mod": w["heightMod"],
        "spray": w["sprayAngle"], "accuracy": w["accuracy"], "moving_accuracy": w.get("movingAccuracy") or 0.0,
        "para_seconds": para_seconds, "extra_para": _custom(w, "extra_damage"),
        "slow_factor": slow_factor,
        "disarm_mult": disarm_mult, "disarm_penalty": disarm_penalty, "disarm_time": disarm_time,
        "overstun_seconds": _custom(w, "overstun_time"),
        "overslow": _custom(w, "timeslow_overslow_frames") * SLOW_DECLINE_PER_FRAME,
        "shield_cost": shield_cost,
        "climb_frames": float(w.get("uptime") or 0.0) * FRAMES if starburst else 0.0,
        "dyn_exp": float(w.get("dynDamageExp") or 0.0), "dyn_range": float(w.get("dynDamageRange") or 0.0),
        "shield_power": w["shieldPower"], "shield_regen": _custom(w, "shield_rate") / FRAMES,
        "shield_radius": w["shieldRadius"],
        "vel_threshold": velocity[0], "vel_base": velocity[1], "vel_scale": velocity[2], "alpha_damage": alpha,
        "area_dps": _custom(w, "area_damage_dps") if has_area else 0.0,
        "area_radius": _custom(w, "area_damage_radius") if has_area else 0.0,
        "area_plateau": _custom(w, "area_damage_plateau_radius"),
        "area_range_fall": _custom(w, "area_damage_range_falloff"),
        "area_plateau_fall": _custom(w, "area_damage_plateau_fall"),
        "area_time_fall": _custom(w, "area_damage_time_falloff"),
        "area_frames": _custom(w, "area_damage_duration") * FRAMES if has_area else 0.0,
        "burn_frames": _custom(w, "burntime", fire_starter * DEFAULT_BURN_FRAMES) if on_fire else 0.0,
        "burn_random": _custom(w, "burntimerand", DEFAULT_BURN_RANDOM),
        "burn_chance": _custom(w, "burnchance", fire_starter / 10.0) if on_fire else 0.0,
        "burn_damage": _custom(w, "burndamage", DEFAULT_BURN_DAMAGE),
        "kind": _weapon_kind(w), "flags": flags,
        "shield_intercept": int(w["shieldInterceptType"]), "intercepted_by": int(w["interceptedByShieldType"]),
    }


def _move_class(u: dict) -> int:
    if u.get("canFly") or not u.get("moveDef"):
        return -1
    md = u["moveDef"]
    sm = int(md["smClass"])
    if sm == SM_TANK:
        name = "tank"
    elif sm == SM_HOVER:
        name = "hover"
    elif sm == SM_SHIP:
        name = "ship"
    elif md["depth"] >= AMPHIBIOUS_DEPTH:
        name = "amph"
    elif md["maxSlope"] > SPIDER_MIN_SLOPE:
        name = "spider"
    else:
        name = "kbot"
    return MOVE_CLASS_NAMES.index(name)


def _mask(names: dict | None, categories: dict[str, float]) -> int:
    mask = 0
    for n in names or {}:
        if n in categories:
            mask |= 1 << int(categories[n])
    return mask


def _unit_record(u: dict, weapon_index: dict[str, int], weapons_by_id: dict[int, dict], records: list[dict],
                 categories: dict[str, float]) -> tuple[dict, bool]:
    mounts = sorted((u.get("weapons") or {}).items(), key=lambda kv: int(kv[0]))
    only_name = COMMANDER_WEAPONS.get(u["name"])
    picked, seen = [], set()
    for _, m in mounts:
        name = weapons_by_id[int(m["weaponDef"])]["name"]
        if only_name and (name != only_name or name in seen):
            continue
        seen.add(name)
        picked.append((name, m))
    truncated = len(picked) > MAX_MOUNTS
    picked = picked[:MAX_MOUNTS]
    armed = False
    max_range = 0.0
    for name, _ in picked:
        rec = records[weapon_index[name]]
        attacks = rec["kind"] != KIND_SHIELD and not rec["flags"] & W_NO_AUTO_TARGET and max(rec["damage"]) > 0.0
        if attacks:
            armed = True
            max_range = max(max_range, rec["range"])
    move_def = u.get("moveDef") or {}
    speed = u["speed"] / FRAMES
    air = bool(u.get("canFly"))
    mobile = bool(u.get("canMove")) and speed > 0.0
    volume = u.get("collisionVolume") or {}
    scale_x, scale_z = volume.get("scaleX", 1.0), volume.get("scaleZ", 1.0)
    # A scale of 1 means the volume comes from the model: use the def's radius.
    hit_radius = (scale_x + scale_z) * 0.25 if scale_x > 1.0 and scale_z > 1.0 else u["radius"]
    footprint = move_def.get("xsize", u.get("xsize", 2.0)) * 4.0
    turn = u["turnRate"] * 2.0 * math.pi / HEADING_CIRCLE
    if air and turn <= 0.0:
        turn = math.pi / 6.0          # planes don't use turnRate; they come around in about 2 s at most
    slow_turn = u.get("turnInPlaceSpeedLimit") or speed
    explosion = u.get("deathExplosion") or ""
    flags = (U_MOBILE if mobile else 0) | (U_AIR if air else 0) | (U_ARMED if armed else 0)
    custom = u.get("customParams") or {}
    if custom.get("fireproof") == "1":
        flags |= U_FIREPROOF
    return {
        "name": u["name"], "cost": u["cost"], "hp": u["health"], "speed": speed,
        "accel": u["maxAcc"], "brake": u["maxDec"], "turn_rate": turn, "slow_turn_speed": slow_turn,
        "footprint_radius": footprint, "hit_radius": hit_radius,
        "armored_mult": u.get("armoredMultiple", 1.0), "auto_heal": (u.get("autoHeal") or 0.0) / SLOW_UPDATE_FRAMES,
        "max_range": max_range, "armor_class": int(u["armorType"]), "move_class": _move_class(u),
        "def_id": int(u["id"]), "los_radius": u.get("losRadius", 0.0), "air_los_radius": u.get("airLosRadius", 0.0),
        "radar_radius": u.get("radarRadius", 0.0), "los_height": u.get("losHeight", 0.0),
        # Costs nothing until the sim has an economy (unit_stockpile.lua: seconds, then frames).
        "stockpile_frames": float(custom.get("stockpiletime") or 0.0) * FRAMES,
        "reload_move_penalty": float(custom.get("reload_move_penalty") or 1.0),
        "category": _mask(u.get("modCategories"), categories), "flags": flags,
        "explode_weapon": weapon_index.get(explosion.lower(), -1),
        "mount_weapon": [weapon_index[n] for n, _ in picked],
        "mount_only": [_mask(m.get("onlyTargets"), categories) for _, m in picked],
        "mount_bad": [_mask(m.get("badTargets"), categories) for _, m in picked],
    }, truncated


def _read_priority() -> tuple[list[str], dict[str, tuple[tuple[float, float, float], float, np.ndarray]]]:
    rows: dict[str, tuple[tuple[float, float, float], float, np.ndarray]] = {}
    with open(DUMP / "priority.txt", encoding="utf8") as f:
        names = f.readline().split()[1:]
        for line in f:
            parts = line.split()
            values = np.array(parts[5:], np.float32)
            if len(values) != len(names):
                raise SystemExit(f"priority row {parts[0]}: {len(values)} values for {len(names)} units")
            rows[parts[0]] = ((float(parts[1]), float(parts[2]), float(parts[3])), float(parts[4]), values)
    return names, rows


def _cross_check(units: list[dict], weapons: list[dict]) -> int:
    """The bot's own data was exported another way: the two must agree."""
    known = json.loads((ROOT / "bots/zkbot/data/units.json").read_text("utf8"))
    abilities = json.loads((ROOT / "bots/zkbot/data/weapons.json").read_text("utf8"))
    by_weapon = {i: w for i, w in enumerate(weapons)}
    bad = 0
    for u in units:
        k = known.get(u["name"])
        if not k:
            continue
        pairs = (("cost", u["cost"], k["costM"]), ("hp", u["hp"], k["health"]), ("speed", u["speed"] * FRAMES, k["speed"]))
        for label, ours, theirs in pairs:
            if abs(ours - theirs) > 0.01 * max(abs(theirs), 1.0):
                print(f"  mismatch {u['name']} {label}: dump {ours:g}, bot data {theirs:g}")
                bad += 1
        info = abilities.get(u["name"])
        if info and u["name"] not in COMMANDER_WEAPONS and u["mount_weapon"]:
            top = max((by_weapon[i]["range"] for i in u["mount_weapon"]), default=0.0)
            listed = max((w["range"] for w in info["weapons"]), default=0.0)
            if abs(top - listed) > 0.01 * max(listed, 1.0) and len(info["weapons"]) <= MAX_MOUNTS:
                print(f"  mismatch {u['name']} range: dump {top:g}, bot data {listed:g}")
                bad += 1
    return bad


def _roster_check(units: list[dict]) -> None:
    from zkbot import hazards
    from zkbot import units as bot_units
    bot_units.load() if hasattr(bot_units, "load") else None
    differ = {}
    for u in units:
        if not u["flags"] & U_MOBILE or u["name"].startswith(("chicken", "dyn", "comm", "c")):
            continue
        try:
            expected = hazards.move_class(u["name"])
        except Exception:     # names the bot never sees
            continue
        ours = MOVE_CLASS_NAMES[u["move_class"]] if u["move_class"] >= 0 else None
        if ours != expected:
            differ[u["name"]] = (ours, expected)
    print(f"move class vs the bot's unit families: {len(differ)} differ {differ}")


def catalog() -> None:
    raw_units = json.loads((DUMP / "units.json").read_text("utf8", "replace"))
    raw_weapons = json.loads((DUMP / "weapons.json").read_text("utf8", "replace"))
    game = json.loads((DUMP / "game.json").read_text("utf8", "replace"))
    categories = game["springCategories"]
    unit_order, priority = _read_priority()
    weapon_order = sorted(raw_weapons)
    weapon_index = {n: i for i, n in enumerate(weapon_order)}
    weapons = []
    for n in weapon_order:
        velocity, alpha, _ = priority.get(n, ((0.0, 0.0, 0.0), 0.0, None))
        weapons.append(_weapon_record(raw_weapons[n], velocity, alpha))
    by_id = {int(w["id"]): w for w in raw_weapons.values()}
    units, truncated = [], []
    for n in unit_order:
        rec, cut = _unit_record(raw_units[n], weapon_index, by_id, weapons, categories)
        units.append(rec)
        if cut:
            truncated.append(n)
    table = np.full((len(weapons), len(units)), 100.0, np.float32)
    for n, (_, _, row) in priority.items():
        if n in weapon_index:
            table[weapon_index[n]] = row
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "catalog.json").write_text(json.dumps({"units": units, "weapons": weapons}), "utf8")
    np.savez_compressed(DATA / "priority.npz", priority=table)
    moves = {u["name"]: MOVE_CLASS_NAMES[u["move_class"]] for u in units if u["move_class"] >= 0}
    (ROOT / "bots/zkbot/data/moveclass.json").write_text(json.dumps(moves, indent=0, sort_keys=True), "utf8")
    print(f"{len(units)} units, {len(weapons)} weapons, priority {table.shape}; more than {MAX_MOUNTS} mounts: {truncated}")
    bad = _cross_check(units, weapons)
    print(f"cross-check against bots/zkbot/data: {bad} mismatches")
    _roster_check(units)
    print("missing death explosion:", sum(1 for n in unit_order if raw_units[n].get("deathExplosion", "").lower() not in weapon_index
                                          and raw_units[n].get("deathExplosion")))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _borders(seg: np.ndarray, slope: np.ndarray) -> dict[tuple[int, int], tuple[int, float]]:
    """(cells of shared border, mean slope across it) of each touching pair of segments."""
    total: dict[tuple[int, int], list[float]] = {}
    for a, b, sa, sb in ((seg[:, :-1], seg[:, 1:], slope[:, :-1], slope[:, 1:]), (seg[:-1, :], seg[1:, :], slope[:-1, :], slope[1:, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for p, q, s in zip(a[m].tolist(), b[m].tolist(), ((sa[m] + sb[m]) * 0.5).tolist()):
            entry = total.setdefault((min(p, q), max(p, q)), [0, 0.0])
            entry[0] += 1
            entry[1] += s
    return {k: (int(n), s / n) for k, (n, s) in total.items()}


def merge_segments(seg: np.ndarray, height: np.ndarray, slope: np.ndarray, height_band: float, slope_band: float) -> np.ndarray:
    """Fewer segments for a class.  Pockets join the neighbour they share most border with, or lose their segment
    if nothing touches them.  With a height band, neighbours join too if the border is gentle and the result stays
    in the band.  Merged segments stay connected, since only touching ones join."""
    if seg.max() < 0:
        return seg
    count = int(seg.max()) + 1
    walked = seg >= 0
    sizes = np.bincount(seg[walked], minlength=count)
    means = np.bincount(seg[walked], height[walked], minlength=count) / np.maximum(sizes, 1)
    pocket = sizes < MIN_SEGMENT_CELLS
    # Pockets don't widen a band: they're small, and their height says little about the ground around them.
    low = [math.inf if p else m for p, m in zip(pocket.tolist(), means.tolist())]
    high = [-math.inf if p else m for p, m in zip(pocket.tolist(), means.tolist())]
    total = sizes.tolist()
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a: int, b: int) -> None:
        parent[a] = b
        low[b], high[b] = min(low[a], low[b]), max(high[a], high[b])
        total[b] += total[a]

    borders = _borders(seg, slope)
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for (a, b), (cells, _) in borders.items():
        neighbours.setdefault(a, []).append((cells, b))
        neighbours.setdefault(b, []).append((cells, a))
    for s in np.argsort(sizes).tolist():
        if not pocket[s]:
            break
        touching = [(cells, find(o)) for cells, o in neighbours.get(s, []) if find(o) != find(s)]
        if touching:
            join(find(s), max(touching)[1])
    for (a, b), (cells, border_slope) in sorted(borders.items(), key=lambda kv: -kv[1][0]) if height_band > 0.0 else ():
        ra, rb = find(a), find(b)
        if ra != rb and cells >= MIN_BORDER_CELLS and border_slope <= slope_band and max(high[ra], high[rb]) - min(low[ra], low[rb]) <= height_band:
            join(ra, rb)
    # What's still a pocket touches nothing else, so no goal elsewhere could use it.
    roots = sorted({find(s) for s in range(count) if total[find(s)] >= MIN_SEGMENT_CELLS})
    label = {r: i for i, r in enumerate(roots)}
    table = np.array([label.get(find(s), -1) for s in range(count)] + [-1], np.int16)
    return table[seg]        # -1 indexes the table's last entry, which is -1


def _segments(fine: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """A class's connected pieces of each terrain region.  A region can hold parts the class can't walk
    between, and a straight line inside a segment must have a way through."""
    from zkbot import terrain as tr
    rows, cols = shape
    out = np.full(shape, -1, np.int16)
    if fine is None:
        return out
    fine = fine[:rows, :cols]
    count = 0
    for region in np.unique(fine[fine >= 0]):
        pieces, sizes = tr.label_runs(fine == region)
        out[pieces >= 0] = pieces[pieces >= 0] + count
        count += len(sizes)
    return out


def maps(names: list[str], height_band: float = HEIGHT_BAND, slope_band: float = SLOPE_BAND, max_parts: int | None = None,
         seen_share: float | None = None) -> None:
    sys.path.insert(0, str(ROOT / "sim" / "py"))
    import zknav
    max_parts = zknav.MAX_PARTS if max_parts is None else max_parts
    seen_share = zknav.SEEN_SHARE if seen_share is None else seen_share
    import terrain_map as tm
    from zkbot import terrain as tr
    out = DATA / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for name, path in tm.find_maps(names):
        mf = tm.load_map(path, name)
        slope = np.nan_to_num(np.asarray(tr.slope_from_corners(mf.corners), np.float32))
        h16 = tr._block_mean(mf.center, 2)
        rows, cols = min(h16.shape[0], slope.shape[0]), min(h16.shape[1], slope.shape[1])
        h16, slope = h16[:rows, :cols], slope[:rows, :cols]
        water = mf.water_damage * tm.DAMAGE_PER_SLOW_UPDATE
        speed = np.stack([tr.speed_field(tr.MOVE_CLASSES[c], h16, slope, water, mf.hover_safe) for c in MOVE_CLASS_NAMES])
        slope_mod = np.array([0.0 if tr.MOVE_CLASSES[c].kind == "ship" else tr.MOVE_CLASSES[c].slope_mod_value for c in MOVE_CLASS_NAMES], np.float32)
        terrain = tm.build(mf)
        h16 = h16.astype(np.float32)
        segments = np.stack([merge_segments(_segments(terrain.fine_label.get(c), (rows, cols)), h16, slope, height_band, slope_band)
                             for c in MOVE_CLASS_NAMES])
        built = [zknav.navigation(s, v, max_parts, seen_share) for s, v in zip(segments, speed)]
        segments = np.stack([labels for labels, _ in built])
        routes = {f"class{c}_{k}": v for c, (_, r) in enumerate(built) for k, v in r.items()}
        np.savez_compressed(out / f"{_slug(name)}.npz", height=h16.astype(np.float32), speed=speed.astype(np.float32),
                            slope=slope, slope_mod=slope_mod, segments=segments, cell=16.0, name=name, **routes)
        from zkbot.policy import bundle as policy_bundle
        policy_bundle.from_terrain(terrain).save(out / f"{_slug(name)}.bundle.npz")
        print(f"  {len(terrain.regions)} regions; segments per class " + " ".join(f"{c} {len(r['centres'])}" for c, (_, r) in zip(MOVE_CLASS_NAMES, built)))
        print(f"{name}: {cols}x{rows}, classes {MOVE_CLASS_NAMES}, passable " +
              " ".join(f"{(s > 0).mean():.2f}" for s in speed))


def splits(folds: int = 5, seed: int = 0) -> None:
    games = sorted({str(pathlib.Path(f).parent.relative_to(ROOT).as_posix())
                    for f in glob.glob(str(ROOT / "results/*/game-*/fights-team0.jsonl"))})
    random.Random(seed).shuffle(games)
    table = {g: i % folds for i, g in enumerate(games)}
    (ROOT / "tools/splits.json").write_text(json.dumps({"seed": seed, "folds": folds, "games": dict(sorted(table.items()))}, indent=0), "utf8")
    print(f"{len(games)} games in {folds} folds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("catalog", "maps", "splits"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--height-band", type=float, default=HEIGHT_BAND)
    ap.add_argument("--slope-band", type=float, default=SLOPE_BAND)
    ap.add_argument("--max-parts", type=int, help="most parts a segment splits into")
    ap.add_argument("--seen-share", type=float, help="stop splitting once its centres see this share of a segment")
    a = ap.parse_args()
    if a.what == "catalog":
        catalog()
    elif a.what == "maps":
        maps(a.names, a.height_band, a.slope_band, a.max_parts, a.seen_share)
    else:
        splits()


if __name__ == "__main__":
    main()
