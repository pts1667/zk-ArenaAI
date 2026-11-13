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
MAX_MOUNTS = 8
MOVE_CLASS_NAMES = ("kbot", "tank", "spider", "amph", "hover", "ship")
MIN_SEGMENT_CELLS = 64               # smaller pieces join a neighbour
MIN_BORDER_CELLS = 8                 # 128 elmos; a narrower gap keeps two segments apart
# Merging neighbours by height and slope is off: a sweep gave fewer segments, but worse routes and arrival.
# The flags are there to repeat that sweep.
HEIGHT_BAND = 0.0                    # elmos; the most a merged segment's pieces differ in mean height
SLOPE_BAND = 0.0                     # engine slope; the steepest mean border that merges
SPIDER_MIN_SLOPE = 0.6               # maxSlope in the moveDef; kbots are 0.41, all-terrain 0.97
AMPHIBIOUS_DEPTH = 1000.0
SM_TANK, SM_KBOT, SM_HOVER, SM_SHIP = 0, 1, 2, 3

KIND_INSTANT, KIND_PROJECTILE, KIND_GUIDED, KIND_SHIELD = 0, 1, 2, 3
INSTANT_TYPES = {"BeamLaser", "LightningCannon", "Melee"}
MISSILE_TYPES = {"MissileLauncher", "StarburstLauncher", "TorpedoLauncher"}
W_DISARM_ONLY, W_MANUAL, W_STOCKPILE, W_UNMODELLED, W_UNLINKED, W_HIGH_ARC, W_BEAM = 256, 512, 1024, 2048, 4096, 8192, 16384
W_IMPACT_ONLY, W_TURRET, W_PARALYZER, W_NO_SELF_DAMAGE, W_NO_AUTO_TARGET, W_NO_FRIENDLY_FIRE, W_SLOW_ONLY, W_AREA_SLOWS = 1, 2, 4, 8, 16, 32, 64, 128
W_OKP_STATIC_ONLY, W_OKP_FLIGHT, W_OKP_TARGET_SPEED, W_SCRIPT_BURST = 32768, 65536, 131072, 262144
W_PIERCES, W_ONE_HIT_EACH, W_THROWS, W_BOMB, W_DROPPED = 524288, 1048576, 2097152, 4194304, 8388608
W_ROCKET, W_BURNBLOW, W_LEAD_ADDED = 16777216, 33554432, 67108864
ROCKET_TYPES = {"MissileLauncher", "TorpedoLauncher"}       # fly under their own power, as far as their fuel goes
# Metal and energy per second from the module all new commanders have (dynamic_comm_defs.lua, "econ"); it's not in the unit def.
# Logged games agree: income less 2 per extractor and solar is 6 and 8, this and the team's innate 2 and 2.
COMMANDER_ECON = (4.0, 6.0)
U_MOBILE, U_AIR, U_ARMED, U_FIREPROOF, U_FACTORY, U_PLOPPABLE, U_COMMANDER = 1, 2, 4, 8, 16, 32, 64
U_FLOATS, U_MISSILE, U_NEUTRAL, U_PLANE, U_LANDS, U_STRAFES = 8192, 16384, 32768, 65536, 131072, 262144
# Planes: frames a half turn takes, and the share of top speed held through it.  `tools/sim_engine.py air`, a move order
# straight behind (01-simulator.md, "Aircraft").  The engine's flight model is 3D with rudder, elevator and bank; the sim's is a
# turn at a fixed rate, so these are what it's fitted to.  A plane not here turns as the median one.
PLANE_TURNS = {"planefighter": (72, 0.59), "planeheavyfighter": (56, 0.84), "planesupport": (50, 0.85), "bomberprec": (54, 0.63),
               "bomberriot": (64, 0.71), "bomberdisarm": (102, 0.85), "bomberassault": (176, 0.89), "bomberheavy": (90, 0.78),
               "planescout": (46, 0.60), "planelightscout": (64, 0.73)}
PLANE_TURN_DEFAULT = (64, 0.75)
STRAFE_SHARE = 0.8         # of its combat range, where a strafing gunship circles: the same run, five gunships within 10 elmos
STRAFE_GRIP = 0.37         # its speed round that circle is as if it could turn with this much of its acceleration, at most top speed
REAMMO_DRAIN = 10.0        # unit_refuel_pad_handler.lua DEFAULT_REAMMO_DRAIN, energy a second
PAD_BP = 2.5               # ... DEFAULT_REPAIR_BP
RUN_SHOT_FRAMES = 1.0      # bomberdisarm.lua: a bolt each Sleep(35), which is a frame (the engine run; 01-simulator.md, "Aircraft")
THERMITE_START_SHARE = 0.05   # of its end dps: an area here can't start at nothing, so it ramps from this to keep the same total
# weapon_impulse.lua: GRAVITY_BASELINE, and what a unit gets of an impulse by how it moves.
IMPULSE_BASELINE, IMPULSE_PLANE, IMPULSE_GUNSHIP, IMPULSE_GROUND = 120.0, 0.022, 0.004, 0.0036
BEACON_SECONDS_PER_SPAWN_TIME = 9.0 / 8.0  # amphtele.lua: 90 steps of 0.1 s at a speed of 8 over its def's time
U_CAN_CLOAK, U_STARTS_CLOAKED, U_FIRES_CLOAKED, U_AREA_CLOAK_ON, U_LEAVES_GHOST, U_KAMIKAZE_BLIND = 128, 256, 512, 1024, 2048, 4096
# bait_prevention_defs.lua: a target whose build time is up to the n-th of these is at bait level n.
BAIT_COST_BANDS = (10.0, 89.9, 239.9, 419.9)
# capi.h: how a unit type regards a type of enemy, and how it acts.  From unit_tactical_ai.lua's tables.
R_SKIRM, R_FLEE, R_IDLE_FLEE, R_HUG, R_FIGHT_ONLY, R_BONUS_RANGE, R_AVOID_HEIGHT, R_SWARM = 1, 2, 4, 8, 16, 32, 64, 128
T_HANDLED, T_IDLE_ONLY, T_SKIRM_RADAR, T_SKIRM_EVERYTHING, T_KEEP_ORDER, T_FLEE_EVERYTHING, T_FLEE_RADAR, T_FLEE_COMBAT, T_IDLE_FLEE_COMBAT = 1, 2, 4, 8, 16, 32, 64, 128, 256
T_LOCAL_JINK, T_CIRCLE_STRAFE = 512, 1024
T_SELF_VELOCITY, T_WARD, T_STARTS_OFF, T_ALWAYS_JINK = 2048, 4096, 8192, 16384
WARD_TARGETS = 4                # capi.h
WARD_MOST_BAIT = 2              # unit_ward_fire.lua: a unit kept from baits above this level doesn't ward
DEFAULT_BURN_FRAMES = 450.0          # fire_defs.lua
SLOW_DECLINE_PER_FRAME = 0.04 / FRAMES   # timeslow_defs.lua DEGRADE_FACTOR
# Weapons that would rather hit something not yet disarmed: the penalty, and the status above which
# it counts (target_priority_defs.lua disarmPenaltyDefs, disarmWeaponTimeDefs after its conversion).
DISARM_TARGETING = {"shieldarty_emp_rocket": (200.0, 0.4), "shipscout_missile": (10.0, 0.33)}
DEFAULT_BURN_RANDOM = 0.3
# A shot off up or down in the game misses, where the sim's scatter is sideways only.  It matters for a Stardust, with its 9 degrees of it: at the rate
# its heat allows the game gave 0.97 of the sim's damage at 150 elmos on a Knight and 0.54 at 400 on a Glaive, and 0.81 over those eight cases
# (01-simulator.md, "Known gaps").  This is a mean over ranges and sizes, not the falloff.
SCATTER_DAMAGE = {"turretriot_turretriot_weapon": 0.8}
LANDING_SUFFIX = "_landing"
BURROW_SPEED, BURROW_SPEED_AIR = 0.05, 3.5
DETONATE_REACH = 160.0      # gunshipbomb.lua AimFromWeapon: its weapon's range at top speed
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


def _arc(w: dict, game_gravity: float) -> float:
    """Gravity over speed squared for what falls as it flies: Cannon.cpp, with the map's gravity unless it has its own."""
    speed = w["projectilespeed"]
    if w["type"] != "Cannon" or speed <= 0.0:
        return 0.0
    return (w.get("myGravity") or game_gravity) / (speed * speed)


def _weapon_record(w: dict, velocity: tuple[float, float, float], alpha: float, game_gravity: float) -> dict:
    custom = w.get("customParams") or {}
    damages = w["damages"]
    # Burst weapons fire from their script: the def's reload is the time between shots.
    reload_s, salvo, salvo_delay_s = w["reload"], w["salvoSize"], w["salvoDelay"]
    script_burst = "script_reload" in custom
    if script_burst:
        salvo = _custom(w, "script_burst", 1.0)
        salvo_delay_s = reload_s
        reload_s = _custom(w, "script_reload")
    # Starburst missiles leave straight up, so the launcher needn't face the target.
    starburst = w["type"] == "StarburstLauncher"
    flags = W_TURRET if w.get("turret") or starburst else 0
    if script_burst:
        flags |= W_SCRIPT_BURST
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
    if w.get("noAutoTarget") or w.get("manualFire") or w["name"].endswith(LANDING_SUFFIX):
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
    if "unlinked" in custom:
        flags |= W_UNLINKED
    if w["type"] == "BeamLaser":
        flags |= W_BEAM
    # Thermite is noExplode too, but it falls and then sits there burning (weapon_noexplode_speed_damage.lua).
    if w.get("noExplode") and "thermite_frames" not in custom:
        flags |= W_PIERCES
        if {"single_hit", "single_hit_multi"} & custom.keys():
            flags |= W_ONE_HIT_EACH
    if "timeslow_onlyslow" in custom:
        flags |= W_SLOW_ONLY
    if custom.get("area_damage_is_slow") == "1":
        flags |= W_AREA_SLOWS
    # Impulse areas push rather than hurt, so they count as no area here.
    has_area = "area_damage_dps" in custom and custom.get("area_damage_is_impulse") != "1"
    on_fire = "setunitsonfire" in custom
    fire_starter = float(w.get("fireStarter") or 0.0)
    damage = [float(damages[str(i)]) * SCATTER_DAMAGE.get(w["name"], 1.0) for i in range(5)]
    # What these are for isn't in the sim, so no army may rely on them: capture (a capture weapon does
    # no health damage, unit_capture.lua), spawned units, an area that pulls.
    if {"is_capture", "spawns_name"} & custom.keys() or custom.get("area_damage_is_impulse", "0") != "0":
        flags |= W_UNMODELLED
    if "is_capture" in custom:
        damage = [0.0] * 5
    # weapon_impulse.lua: all it does is push, unless it says otherwise.
    if "impulse" in custom and "normaldamage" not in custom:
        damage = [0.0] * 5
    # A Lobster's guns fire only when told to (ATTACK_BLOCK_DEFAULT) and do nothing but throw (unit_thrower.lua).
    if "thower_weapon" in custom:
        flags |= W_THROWS | W_NO_AUTO_TARGET
    if w["type"] == "AircraftBomb":
        flags |= W_BOMB
    if w["type"] in ROCKET_TYPES:
        flags |= W_ROCKET
    if w.get("selfExplode"):
        flags |= W_BURNBLOW
    # A bomber's fake bomb only marks where it's going: Odin's script blocks it, and Thunderbird's starts its run when it would
    # drop (bomberassault.lua, bomberdisarm.lua).  Thermite falls from the plane, and then burns where it lands.
    if flags & W_BOMB and "fake_weapon" in custom:
        damage = [0.0] * 5
    thermite = "thermite_frames" in custom
    if thermite:
        flags |= W_BOMB
    # Slow: timeslow_defs.lua.  Slow-only weapons default to 1; an absolute amount scales with the damage.
    raw_damage = _custom(w, "raw_damage")
    slow_factor = _custom(w, "timeslow_damagefactor", 1.0 if "timeslow_onlyslow" in custom else 0.0)
    if "timeslow_damage" in custom and raw_damage > 0.0:
        slow_factor = _custom(w, "timeslow_damage") / raw_damage
    # The Felon's gun fires from its shield's charge (shieldfelon.lua).  On shields this is an energy cost.
    shield_cost = _custom(w, "shield_drain") if not w.get("isShield") else 0.0
    # armordefs.lua: shield_damage is per frame of a beam, and what a whole shot takes is the stats figure.
    if "shield_damage" in custom:
        damage[int(SHIELD_ARMOR)] = _custom(w, "stats_shield_damage", _custom(w, "shield_damage"))
    para_seconds = _custom(w, "emp_paratime", _custom(w, "disarmtimer", float(damages.get("paralyzeDamageTime") or 0.0)))
    rec = {
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
        "shield_energy": _custom(w, "shield_drain") / FRAMES if w.get("isShield") else 0.0,
        "arc": _arc(w, game_gravity),
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
        "impulse": _custom(w, "impulse"),
        "gravity": float(w.get("myGravity") or 0.0) or game_gravity,
        # Weapon.cpp GetLeadVec and MissileProjectile.cpp.  weapondefs_post.lua has set predictBoost where a def didn't.
        "lead_limit": float(w.get("leadLimit", -1.0)), "predict_boost": float(w.get("predictBoost", 1.0)),
        "move_error": float(w.get("targetMoveError") or 0.0), "flight_frames": float(w.get("flightTime") or 0.0),
        "turn_rate": float(w.get("turnRate") or 0.0), "rise": float(w.get("trajectoryHeight") or 0.0), "accurate_leading": 0,
        "kind": _weapon_kind(w), "flags": flags,
        "shield_intercept": int(w["shieldInterceptType"]), "intercepted_by": int(w["interceptedByShieldType"]),
    }
    if thermite:
        # weapon_noexplode_speed_damage.lua: it burns what it's in for thermite_frames, its dps going from start to end.  Here
        # an area, from the start share up to what keeps the total, as wide as the bomb and what it's in together.
        start, end = _custom(w, "thermite_dps_start"), _custom(w, "thermite_dps_end")
        first = max(start, THERMITE_START_SHARE * end)
        last = start + end - first
        rec.update(range=_custom(w, "truerange", w["range"]), area_dps=first, area_time_fall=1.0 - last / first,
                   area_frames=_custom(w, "thermite_frames"), area_radius=2.0 * w["damageAreaOfEffect"],
                   area_plateau=0.0, area_range_fall=0.0, area_plateau_fall=0.0)
    return rec


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


def _remains(u: dict, features: dict[str, dict], features_by_id: dict[int, dict]) -> dict:
    """What a unit leaves when it dies: its corpse, and what the corpse leaves in turn (featuredefs_post.lua)."""
    wreck = features.get(u.get("wreckName") or "") or {}
    heap = features_by_id.get(int(wreck.get("deathFeatureID") or -1)) or {}
    out = {}
    for prefix, f in (("wreck", wreck), ("heap", heap)):
        out[f"{prefix}_metal"] = f.get("metal", 0.0) if f.get("reclaimable") else 0.0
        out[f"{prefix}_hp"] = f.get("maxHealth", 0.0)
        out[f"{prefix}_feature"] = int(f.get("id", -1))
    return out


def _cloak(u: dict, custom: dict) -> tuple[dict, int]:
    """A commander's def says it can cloak whether or not it has the module, and a new one hasn't."""
    can = bool(u.get("canCloak")) and not custom.get("dynamic_comm") and not custom.get("cannotcloak")
    radius = float(custom.get("area_cloak_radius") or 0.0)
    flags = ((U_CAN_CLOAK if can else 0) | (U_STARTS_CLOAKED if can and custom.get("initcloaked") else 0)
             | (U_FIRES_CLOAKED if not u.get("decloakOnFire", True) else 0)
             | (U_AREA_CLOAK_ON if radius > 0.0 and str(custom.get("area_cloak_init", "1")) != "0" else 0))
    return {
        "cloak_cost": u.get("cloakCost", 0.0) if can else 0.0, "cloak_cost_moving": u.get("cloakCostMoving", 0.0) if can else 0.0,
        "decloak_distance": u.get("decloakDistance", 0.0), "bestowed_decloak": float(custom.get("cloaker_bestowed_radius") or 0.0),
        "area_cloak_radius": radius, "area_cloak_upkeep": float(custom.get("area_cloak_upkeep") or 0.0),
    }, flags


def _jump(custom: dict, mobile: bool) -> dict:
    """jump_defs.lua.  A factory's def says it can as well, to give its orders to what it builds."""
    can = mobile and custom.get("canjump") == "1"
    return {"jump_range": float(custom["jump_range"]) if can else 0.0, "jump_speed": float(custom["jump_speed"]) if can else 0.0,
            "jump_height": float(custom["jump_height"]) if can else 0.0, "jump_reload": float(custom["jump_reload"]) * FRAMES if can else 0.0,
            "jump_delay": float(custom.get("jump_delay") or 0.0) if can else 0.0}


def _push_scale(u: dict, air: bool) -> float:
    """weapon_impulse.lua's AddGadgetImpulse: the speed a unit of impulse gives it."""
    mass = float(u.get("mass") or 0.0)
    if mass <= 0.0:
        return 0.0
    if not air:
        return IMPULSE_BASELINE * IMPULSE_GROUND / mass
    return IMPULSE_BASELINE * (IMPULSE_GUNSHIP if u.get("isHoveringAirUnit") else IMPULSE_PLANE) / mass


def _burrow(custom: dict, air: bool) -> float:
    """The speed under which what has idle_cloak digs in: GG.StartStopMovingControl's, or what gunshipbomb.lua polls for."""
    if not custom.get("idle_cloak"):
        return 0.0
    return BURROW_SPEED_AIR if air else BURROW_SPEED


def _bait(u: dict, custom: dict) -> dict:
    """bait_prevention_defs.lua, for what would rather not shoot at cheap things and what's cheap.  0 is no level."""
    target = int(custom.get("bait_level_target") or 0)
    if not target:
        target = next((i + 1 for i, top in enumerate(BAIT_COST_BANDS) if u["buildTime"] <= top), 0)
    return {"bait_level": int(custom.get("bait_level_default") or 0), "bait_target": target,
            "bait_target_armor": int(custom.get("bait_level_target_armor") or 0), "bait_cost": u["buildTime"]}


def _heat(custom: dict) -> dict:
    """turretriot.lua: firing heats the unit, which reloads this much slower when fully hot."""
    return {"heat_per_shot": float(custom.get("heat_per_shot") or 0.0), "heat_decay": float(custom.get("heat_decay") or 0.0) / FRAMES,
            "heat_max_slow": float(custom.get("heat_max_slow") or 0.0), "heat_initial": float(custom.get("heat_initial") or 0.0)}


def _transport(u: dict, custom: dict, air: bool, mobile: bool) -> dict:
    """As an air transport: its speed with a load of each class, as a share of its own (unit_transport_speed.lua), and nothing for
    a class it can't lift (unit_transport_load_unload.lua: a light transport lifts nothing heavy).  As a load: its class, which
    unitdefs_post.lua sets by cost, its footprint against the transport's transportSize (CUnit::CanTransport), and how far under
    the transport it hangs."""
    lifts = air and bool(u.get("isTransport")) and float(u.get("transportCapacity") or 0.0) > 0.0

    def share(key: str) -> float:
        return float(custom.get(key) or 1.0) if lifts else 0.0
    carried = mobile and not air and not u.get("cantBeTransported", True)
    return {"lift_light": share("transport_speed_light"), "lift_medium": share("transport_speed_medium"),
            "lift_heavy": 0.0 if custom.get("islighttransport") else share("transport_speed_heavy"),
            "lift_size": float(u.get("transportSize") or 0.0) * 8.0 if lifts else 0.0,
            "load_size": u.get("xsize", 2.0) * 4.0, "load_height": float(u.get("height") or 0.0),
            "load_class": 0 if not carried else 3 if custom.get("requireheavytrans") else 2 if custom.get("requiremediumtrans") else 1}


def _unit_record(u: dict, weapon_index: dict[str, int], weapons_by_id: dict[int, dict], records: list[dict],
                 categories: dict[str, float], features: dict[str, dict], features_by_id: dict[int, dict]) -> tuple[dict, bool]:
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
    # bomberdisarm.lua: where the fake bomb would drop, the script fires what targets nothing, a shot a frame straight down.
    marks = any(records[weapon_index[n]]["flags"] & W_BOMB and max(records[weapon_index[n]]["damage"]) <= 0.0 for n, _ in picked)
    for n, m in picked:
        if marks and set(m.get("onlyTargets") or {}) == {"none"}:
            records[weapon_index[n]]["flags"] |= W_DROPPED
            records[weapon_index[n]]["salvo_delay"] = RUN_SHOT_FRAMES
    armed = False
    max_range = 0.0
    for name, _ in picked:
        rec = records[weapon_index[name]]
        attacks = rec["kind"] != KIND_SHIELD and not rec["flags"] & W_NO_AUTO_TARGET and (max(rec["damage"]) > 0.0 or rec["impulse"] != 0.0)
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
    plane = air and not u.get("isHoveringAirUnit")
    if plane:
        flags |= U_PLANE
        half_turn, held = PLANE_TURNS.get(u["name"], PLANE_TURN_DEFAULT)
        turn = math.pi / half_turn
    # A gunship that strafes circles inside its combat range (unit_control_gunship_stafe_range.lua sets it).
    strafe_radius = strafe_speed = 0.0
    if air and not plane and u.get("airStrafe") and armed:
        reach = max(float((weapons_by_id[int(m["weaponDef"])].get("customParams") or {}).get("combatrange") or records[weapon_index[n]]["range"])
                    for n, m in picked if records[weapon_index[n]]["kind"] != KIND_SHIELD and max(records[weapon_index[n]]["damage"]) > 0.0)
        strafe_radius = STRAFE_SHARE * reach
        strafe_speed = min(speed, math.sqrt(STRAFE_GRIP * u["maxAcc"] * strafe_radius))
        flags |= U_STRAFES
    if custom.get("fireproof") == "1":
        flags |= U_FIREPROOF
    # The strider hub isn't a factory to the engine, but builds as one: what it's told to, near it, over and over.
    if u.get("isFactory") or (custom.get("isfakefactory") == "1" and not mobile):
        flags |= U_FACTORY
    if custom.get("ploppable"):
        flags |= U_PLOPPABLE
    if custom.get("ai_start_unit"):
        flags |= U_COMMANDER
    bomb = custom.get("stats_detonate_weapon")
    cloak, cloak_flags = _cloak(u, custom)
    flags |= cloak_flags
    if u.get("floatOnWater"):
        flags |= U_FLOATS
    if u.get("leavesGhost"):
        flags |= U_LEAVES_GHOST
    kamikaze = float(u.get("kamikazeDist") or 0.0) if u.get("canKamikaze") else 0.0
    if kamikaze > 0.0 and not u.get("kamikazeUseLOS"):
        flags |= U_KAMIKAZE_BLIND
    return {
        **_remains(u, features, features_by_id), **cloak, **_bait(u, custom), **_heat(custom),
        "kamikaze_distance": kamikaze, **_jump(custom, mobile), "burrow_speed": _burrow(custom, air),
        # The Lua's *_landing weapon goes off when it lands, from its script.  Nothing aims it.
        "landing_weapon": next((weapon_index[n] for n, _ in picked if n.endswith(LANDING_SUFFIX)), -1),
        # Only Blastwing has one.
        "drop_weapon": weapon_index[bomb] if bomb else -1, "detonate_reach": DETONATE_REACH if bomb else 0.0,
        "drop_height": float(u.get("wantedHeight") or 0.0) if bomb else 0.0,
        "drop_gravity": float(next(w for w in weapons_by_id.values() if w["name"] == bomb)["customParams"]["mygravity"]) if bomb else 0.0,
        # The engine's defaults for a def that doesn't say: fire at will, maneuver.
        "move_state": int(u["moveState"]) if u.get("moveState", -1) >= 0 else 1,
        "fire_state": int(u["fireState"]) if u.get("fireState", -1) >= 0 else 2,
        "no_chase": _mask(u.get("noChaseCategories"), categories),
        "perceived_range": float(custom.get("percieved_range") or max_range),
        "reclaim_speed": u.get("reclaimSpeed", 0.0), "energy_upkeep": float(custom.get("upkeep_energy") or 0.0),
        "stockpile_cost": float(custom.get("stockpilecost") or 0.0), "close_frames": float(custom.get("auto_close_time") or 0.0) * FRAMES,
        "mass": float(u.get("mass") or 0.0), "push_scale": _push_scale(u, air), "needed_link": float(custom.get("neededlink") or 0.0),
        "throw_radius": float(custom.get("thrower_gather") or 0.0),
        "teleport_rate": float(custom.get("teleporter_throughput") or 0.0) / FRAMES,
        "beacon_frames": float(custom.get("teleporter_beacon_spawn_time") or 0.0) * BEACON_SECONDS_PER_SPAWN_TIME * FRAMES,
        "missile_pads": int(custom.get("missile_silo_capacity") or 0), "beacon_type": 0, "plate_parent": -1,
        "cruise_height": float(u.get("wantedHeight") or 0.0) if air else 0.0,
        "vertical_speed": float(u.get("verticalSpeed") or 0.0) if air else 0.0,
        "turn_speed": speed * held if plane else 0.0,
        "strafe_radius": strafe_radius, "strafe_speed": strafe_speed,
        # scripts/bombers.lua: a salvo empties it, or for planesupport.lua so many shots.
        "ammo": int(custom.get("shots_per_refuel") or 1) if custom.get("reammoseconds") else 0,
        "reammo_frames": float(custom.get("reammoseconds") or 0.0) * FRAMES,
        "reammo_energy": float(custom.get("reammodrain") or REAMMO_DRAIN) / FRAMES if custom.get("reammoseconds") else 0.0,
        "pad_count": int(custom.get("pad_count") or 1) if custom.get("ispad") else 0,
        "pad_bp": float(custom.get("pad_bp") or PAD_BP) if custom.get("ispad") else 0.0,
        **_transport(u, custom, air, mobile),
        "name": u["name"], "cost": u["cost"], "hp": u["health"], "speed": speed,
        "accel": u["maxAcc"], "brake": u["maxDec"], "turn_rate": turn, "slow_turn_speed": slow_turn,
        "footprint_radius": footprint, "hit_radius": hit_radius,
        # What the game tests a build against (CGameHelper::TestUnitBuildSquare): the footprint's rectangle, here its half sizes
        # in x and z (xsize and zsize are in 8-elmo squares), with every square within max_height_dif of the height at its middle.
        "foot_x": u.get("xsize", 2.0) * 4.0, "foot_z": u.get("zsize", u.get("xsize", 2.0)) * 4.0,
        "max_height_dif": float(u.get("maxHeightDif", 0.0)),
        "armored_mult": u.get("armoredMultiple", 1.0), "idle_regen": float(custom.get("idle_regen") or 0.0) / FRAMES, "idle_time": u.get("idleTime") or 0.0,
        "max_range": max_range, "armor_class": int(u["armorType"]), "move_class": _move_class(u),
        "def_id": int(u["id"]), "los_radius": u.get("losRadius", 0.0), "air_los_radius": u.get("airLosRadius", 0.0),
        "radar_radius": u.get("radarRadius", 0.0), "los_height": u.get("losHeight", 0.0),
        # unit_stockpile.lua: seconds, then frames.
        "stockpile_frames": float(custom.get("stockpiletime") or 0.0) * FRAMES,
        "reload_move_penalty": float(custom.get("reload_move_penalty") or 1.0),
        "metal_cost": u["metalCost"], "energy_cost": u["energyCost"], "build_time": u["buildTime"],
        "build_speed": u["buildSpeed"], "build_range": u["buildDistance"],
        "metal_make": u["metalMake"] + float(custom.get("income_metal") or 0.0) + (COMMANDER_ECON[0] if custom.get("ai_start_unit") else 0.0),
        "energy_make": u["energyMake"] + float(custom.get("income_energy") or 0.0) + (COMMANDER_ECON[1] if custom.get("ai_start_unit") else 0.0),
        "extract_mult": float(custom.get("metal_extractor_mult") or 0.0),
        "pylon_range": float(custom.get("pylonrange") or 0.0),
        # Where a structure can go.  Nothing in the sim reads these: tools/frame_data.py places structures by them.
        "min_water_depth": float(u.get("minWaterDepth", 0.0)), "max_water_depth": float(u.get("maxWaterDepth", 0.0)),
        "category": _mask(u.get("modCategories"), categories), "flags": flags,
        "explode_weapon": weapon_index.get(explosion.lower(), -1),
        "mount_weapon": [weapon_index[n] for n, _ in picked],
        "mount_only": [_mask(m.get("onlyTargets"), categories) for _, m in picked],
        "mount_bad": [_mask(m.get("badTargets"), categories) for _, m in picked],
    }, truncated


def _air_idle(units: list[dict], raw_units: dict[str, dict]) -> None:
    """unit_air_plants.lua: what a plant builds lands when idle or not by the plant's factory_land_state; anything else as the
    engine would, which lands it."""
    by_id = {int(u["id"]): u["name"] for u in raw_units.values()}
    states: dict[str, set[str]] = {}
    for u in raw_units.values():
        state = (u.get("customParams") or {}).get("factory_land_state")
        if state is not None:
            for o in (u.get("buildOptions") or {}).values():
                states.setdefault(by_id[int(o)], set()).add(state)
    # Planetwars' own plants fly what the ordinary plane plant lands: the ordinary plant's state goes.
    hovers = {name for name, s in states.items() if s == {"0"}}
    for rec in units:
        if rec["flags"] & U_AIR and rec["name"] not in hovers:
            rec["flags"] |= U_LANDS


def _link_abilities(units: list[dict], raw_units: dict[str, dict]) -> None:
    """What one unit's def says of another: a silo of its missiles, a teleporter of its beacon, and a factory plate of its factory."""
    index = {u["name"]: i for i, u in enumerate(units)}
    by_id = {int(u["id"]): u["name"] for u in raw_units.values()}
    for u in units:
        raw = raw_units[u["name"]]
        custom = raw.get("customParams") or {}
        if u["missile_pads"] > 0:
            for option in (raw.get("buildOptions") or {}).values():
                units[index[by_id[int(option)]]]["flags"] |= U_MISSILE
        beacon = custom.get("teleporter_beacon_unit")
        if beacon:
            u["beacon_type"] = index[beacon]
            units[index[beacon]]["flags"] |= U_NEUTRAL
        parent = custom.get("child_of_factory")
        if parent in index:
            u["plate_parent"] = index[parent]


def _specials(units: list[dict], weapons: list[dict], raw_units: dict[str, dict]) -> dict:
    """What the policy's masks need to know of the few units that do something other than walk and shoot.  The bot can't read the sim's catalog."""
    by_id = {int(u["id"]): u["name"] for u in raw_units.values()}
    out: dict = {"throwers": {}, "switches": [], "teleporters": {}, "silos": {}, "plates": {}, "transports": {}, "loads": {}, "wide_loads": {}}
    for u in units:
        mounts = [weapons[m] for m in u["mount_weapon"]]
        impulses = [w["impulse"] for w in mounts]
        if u["throw_radius"] > 0.0:
            out["throwers"][u["name"]] = {"radius": u["throw_radius"], "reach": max(w["range"] for w in mounts if w["flags"] & W_THROWS)}
        if impulses and min(impulses) < 0.0 < max(impulses):
            out["switches"].append(u["name"])
        if u["teleport_rate"] > 0.0:
            out["teleporters"][u["name"]] = units[u["beacon_type"]]["name"]
        if u["plate_parent"] >= 0:
            out["plates"][u["name"]] = units[u["plate_parent"]]["name"]
        # An air transport: its speed with each class of load, and the widest it takes.  A load: its class.
        if u["lift_light"] > 0.0 or u["lift_medium"] > 0.0 or u["lift_heavy"] > 0.0:
            out["transports"][u["name"]] = {"speed": [u["lift_light"], u["lift_medium"], u["lift_heavy"]], "size": u["lift_size"]}
        if u["load_class"] > 0:
            out["loads"][u["name"]] = u["load_class"]
        if u["missile_pads"] > 0:
            out["silos"][u["name"]] = sorted(by_id[int(o)] for o in (raw_units[u["name"]].get("buildOptions") or {}).values())
    # A load's width is said only where it's too much for some transport.
    narrowest = min((t["size"] for t in out["transports"].values()), default=0.0)
    out["wide_loads"] = {u["name"]: u["load_size"] for u in units if u["load_class"] > 0 and u["load_size"] > narrowest}
    return out


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


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _call_args(line: str, name: str) -> list[str] | None:
    """The arguments after unitID and targetID of a call in a line of script, if none is itself a call."""
    at = line.find(name + "(")
    if at < 0:
        return None
    inside = line[at + len(name) + 1:].split(")")[0]
    return None if "(" in inside else [a.strip() for a in inside.split(",")[2:]]


def _overkill(lines: list[str], okp_damage: float) -> tuple[int | None, dict] | None:
    """What a script's first ordinary overkill check asks: unit_overkill_prevention.lua, called from its BlockShot.  Returns the one mount
    it's for, if it says, and the weapon's fields.  One whose damage is worked out as it fires isn't read: Merlin's, and the bombers'."""
    for line in lines:
        mount = re.search(r"num\s*==\s*(\d)", line)
        which = int(mount.group(1)) - 1 if mount else None
        args = _call_args(line, "GG.Script.OverkillPreventionCheck")
        out = {}
        if args is not None:
            # (damage, range, fullTime, hitTimeMod, useTargetSpeed, minRange, fastMult, radarMult, staticOnly), constants.lua
            args += ["false"] * (9 - len(args))
            hit_mod, min_range = _number(args[3]) or 0.0, _number(args[5]) or 0.0
            out = {"okp_full_frames": _number(args[2]) or 0.0, "okp_hit_mod": hit_mod, "okp_min_range": min_range,
                   "okp_fast_mult": _number(args[6]) or 0.0, "okp_radar_mult": _number(args[7]) or 0.0}
            flags = W_OKP_TARGET_SPEED if args[4] == "true" else 0
            flags |= W_OKP_STATIC_ONLY if args[8] == "true" else 0
        else:
            args = _call_args(line, "GG.OverkillPrevention_CheckBlock")
            if args is None or len(args) < 2:
                continue
            args += ["false"] * (5 - len(args))
            time = _number(args[1])
            times = re.fullmatch(r"([\d.]+)\s*\*\s*distMult", args[1])
            flags = W_OKP_STATIC_ONLY if args[4] == "true" else 0
            out = {"okp_fast_mult": _number(args[2]) or 0.0, "okp_radar_mult": _number(args[3]) or 0.0}
            if time is not None:
                out["okp_frames"] = time
            elif times:
                out["okp_full_frames"] = float(times.group(1))
            else:          # framesETA and the like: how long the shot takes
                flags |= W_OKP_FLIGHT
        damage = _number(args[0])
        if damage is None and args[0] == "OKP_DAMAGE":
            damage = okp_damage
        if damage:
            return which, {**out, "okp_damage": damage, "okp_flags": flags}
    return None


def _apply_overkill(units: list[dict], weapons: list[dict], raw_units: dict, lines: dict) -> int:
    done = 0
    for u in units:
        script = _lists(lines.get(u["name"]) or {})
        found = _overkill(script, float((raw_units[u["name"]].get("customParams") or {}).get("okp_damage") or 0.0)) if script else None
        if not found:
            continue
        mount, fields = found
        done += 1
        for i, w in enumerate(u["mount_weapon"]):
            if (mount is None or mount == i) and weapons[w]["kind"] != KIND_SHIELD:
                weapons[w].update({k: v for k, v in fields.items() if k != "okp_flags"})
                weapons[w]["flags"] |= fields["okp_flags"]
    return done


def _lists(v):
    """The dump writes a Lua array as an object keyed "1", "2"..."""
    return [v[k] for k in sorted(v, key=int)] if isinstance(v, dict) else v


def _tactics_record(b: dict, unit: dict, index: dict, speed_buffer: float) -> dict:
    """One table of unit_tactical_ai.lua's, after its defaults, with what unit_ward_fire.lua reads from it.  The sim has no order
    for ward fire's button, so it's as the tables start it: on for what fires at shields, and never for the three that fire
    towards anything once it's turned on (Badger, Firewalker, Kodachi), whose parameters aren't read.  speed_buffer: the unit
    def's set_target_speed_buffer."""
    # A negative stands for what the table leaves out.
    def num(key: str, default: float = 0.0) -> float:
        return float(b.get(key, default))
    flags = T_HANDLED
    if b.get("onlyIdleHandling"):
        flags |= T_IDLE_ONLY
    if not b.get("defaultAIState", 1):
        flags |= T_STARTS_OFF
    for key, flag in (("skirmRadar", T_SKIRM_RADAR), ("skirmEverything", T_SKIRM_EVERYTHING), ("skirmKeepOrder", T_KEEP_ORDER),
                      ("fleeEverything", T_FLEE_EVERYTHING), ("fleeRadar", T_FLEE_RADAR), ("fleeCombat", T_FLEE_COMBAT),
                      ("idleFleeCombat", T_IDLE_FLEE_COMBAT), ("localJinkOrder", T_LOCAL_JINK), ("circleStrafe", T_CIRCLE_STRAFE),
                      ("selfVelocityPrediction", T_SELF_VELOCITY), ("alwaysJinkFight", T_ALWAYS_JINK)):
        if b.get(key):
            flags |= flag
    # With no enemy looked for, such a unit does nothing but jink: the sim's code for it stops there.
    assert not (flags & T_ALWAYS_JINK) or not (flags & (T_SKIRM_RADAR | T_SKIRM_EVERYTHING | T_FLEE_EVERYTHING | T_FLEE_RADAR)), unit["name"]
    ward: list[tuple[int, float]] = []
    if b.get("hasWardFire") and b.get("wardFireDefault"):
        # Each type it fires towards, with how far past the weapon's range one may be.
        ward = [(index[n], float(v)) for n, v in (item.split(":") for item in b["wardFireTargetsValues"].split()) if n in index]
        shield = b.get("wardFireShield")
        assert isinstance(shield, (int, float)) and not isinstance(shield, bool), unit["name"]
        assert not any(b.get(k) for k in ("wardFireEverything", "wardFireUnboundedRange", "wardFireRangeOverride", "wardFireTravelVectorAvoid")), unit["name"]
        assert 0 < len(ward) <= WARD_TARGETS and unit["bait_level"] <= WARD_MOST_BAIT, unit["name"]
        flags |= T_WARD
    return {
        "skirm_jink": num("skirmJinkLength"), "skirm_heading_block": num("skirmBlockApproachHeadingBlock", 2.0),
        "ward_leeway": num("wardFireLeeway", 10.0), "ward_enable_leeway": num("wardFireEnableLeeway", 3.0), "ward_predict": num("wardFirePredict"),
        "ward_shield": float(b["wardFireShield"]) if ward else -1.0, "ward_height": num("wardFireHeight"), "ward_speed_buffer": speed_buffer,
        "ward_target": [t for t, _ in ward] + [-1] * (WARD_TARGETS - len(ward)),
        "ward_range": [r for _, r in ward] + [0.0] * (WARD_TARGETS - len(ward)),
        "sea": -1, "fight_only": -1, "waterline": 0.0,
        "search_range": num("searchRange", max(unit["max_range"] + 100.0, 800.0)), "idle_push_dist_sq": num("idlePushAggressDistSq"),
        "idle_chase_leeway": num("idleChaseEnemyLeeway"), "idle_commit_dist": num("idleCommitDist"),
        "idle_commit_mult": num("idleCommitDistMult"), "idle_enemy_mult": num("idleEnemyDistMult"),
        "flee_order_dis": num("fleeOrderDis", 120.0), "flee_leeway": num("fleeLeeway", 100.0), "flee_distance": num("fleeDistance", 120.0),
        "min_flee_range": num("minFleeRange"), "flee_vel_prediction": num("fleeVelPrediction", -1.0),
        "weapon_range": num("skirmRange", unit["max_range"]), "stopping_distance": num("stoppingDistance"),
        "velocity_prediction": num("velocityPrediction", 30.0), "skirm_order_dis": num("skirmOrderDis", 120.0),
        "skirm_order_dis_min": num("skirmOrderDisMin", -1.0), "hug_range": num("hugRange", 50.0), "skirm_leeway": num("skirmLeeway"),
        "skirm_range_override": num("skirmRangeOverride", -1.0), "reload_skirm_leeway": num("reloadSkirmLeeway"),
        "skirm_blocked_frames": num("skirmBlockedApproachFrames", -1.0), "vel_pred_chase": num("velPredChaseFactor", -1.0),
        "skirm_keep_leeway": num("skirmKeepOrderLeeway", -1e9), "skirm_only_near": num("skirmOnlyNearEnemyRange", -1.0),
        "my_speed": num("mySpeed", unit["speed"]), "flags": flags, "weapon_num": max(0, int(num("weaponNum", 1.0)) - 1),
        "max_swarm_range": num("maxSwarmRange"), "min_swarm_range": num("minSwarmRange"), "swarm_leeway": num("swarmLeeway", 50.0),
        "jink_tangent": num("jinkTangentLength"), "jink_parallel": num("jinkParallelLength"),
        "jink_away_parallel": num("jinkAwayParallelLength"), "strafe_order_length": num("strafeOrderLength"),
        "min_circle_strafe": num("minCircleStrafeDistance"), "jink_period": num("jinkPeriod"),
        "blocked_on_fight": 1 if b.get("skirmBlockedApproachOnFight") else 0,
    }


def _tactics(units: list[dict], table: dict, raw_units: dict) -> tuple[list[dict], np.ndarray]:
    """A record for each unit type, and after those the other tables some types have: for the sea (GetUnitBehavior), and against
    what they engage only on a fight order (fightOnlyOverride).  relations: a row for each record, a column for each type."""
    index = {u["name"]: i for i, u in enumerate(units)}
    records: list[dict] = [{} for _ in units]
    behaviours: dict[int, dict] = {}

    def add(row: int, b: dict, u: dict, state) -> None:
        b = {**b, "defaultAIState": state}
        # unit_target_on_the_move.lua reads the speed buffer only beside a range buffer.
        params = raw_units[u["name"]].get("customParams") or {}
        speed_buffer = float(params.get("set_target_speed_buffer") or 0.0) if params.get("set_target_range_buffer") else 0.0
        records[row] = _tactics_record(b, u, index, speed_buffer)
        behaviours[row] = b
        for key, field in (("sea", "sea"), ("fightOnlyOverride", "fight_only")):
            other = b.get(key)
            if isinstance(other, dict) and other:
                records.append({})
                records[row][field] = len(records) - 1
                add(len(records) - 1, other, u, state)
        if records[row]["sea"] >= 0:
            # By the unit's height, which for these is the ground's: they walk the sea bed, or go by the ground under them.
            assert b.get("floatWaterline") or MOVE_CLASS_NAMES[u["move_class"]] == "amph", u["name"]
            records[row]["waterline"] = float(b.get("waterline", 0.0))

    for i, u in enumerate(units):
        b = table.get(u["name"])
        if b and not b.get("externallyHandled"):
            add(i, b, u, b.get("defaultAIState", 1))
    relations = np.zeros((len(records), len(units)), np.uint8)
    for row, b in behaviours.items():
        for key, bit in (("skirms", R_SKIRM), ("flees", R_FLEE), ("idleFlee", R_IDLE_FLEE), ("hugs", R_HUG), ("fightOnlyUnits", R_FIGHT_ONLY),
                         ("bonusRangeUnits", R_BONUS_RANGE), ("avoidHeightDiff", R_AVOID_HEIGHT), ("swarms", R_SWARM)):
            for name in (b.get(key) or "").split():
                if name in index:
                    relations[row, index[name]] |= bit
    return records, relations


def catalog() -> None:
    raw_units = json.loads((DUMP / "units.json").read_text("utf8", "replace"))
    raw_weapons = json.loads((DUMP / "weapons.json").read_text("utf8", "replace"))
    raw_features = json.loads((DUMP / "features.json").read_text("utf8", "replace"))
    features_by_id = {int(f["id"]): f for f in raw_features.values()}
    game = json.loads((DUMP / "game.json").read_text("utf8", "replace"))
    categories = game["springCategories"]
    unit_order, priority = _read_priority()
    weapon_order = sorted(raw_weapons)
    weapon_index = {n: i for i, n in enumerate(weapon_order)}
    game_gravity = game["gravity"] / (FRAMES * FRAMES)
    weapons = []
    for n in weapon_order:
        velocity, alpha, _ = priority.get(n, ((0.0, 0.0, 0.0), 0.0, None))
        weapons.append(_weapon_record(raw_weapons[n], velocity, alpha, game_gravity))
    by_id = {int(w["id"]): w for w in raw_weapons.values()}
    units, truncated = [], []
    for n in unit_order:
        rec, cut = _unit_record(raw_units[n], weapon_index, by_id, weapons, categories, raw_features, features_by_id)
        units.append(rec)
        if cut:
            truncated.append(n)
        # The unit's def says which of a cannon's two paths it takes.
        if raw_units[n].get("highTrajectoryType") == 1:
            for mount in rec["mount_weapon"]:
                weapons[mount]["flags"] |= W_HIGH_ARC
    _link_abilities(units, raw_units)
    _air_idle(units, raw_units)
    extras = json.loads((DUMP / "tactics.json").read_text("utf8", "replace"))
    print("scripts read for overkill prevention:", _apply_overkill(units, weapons, raw_units, extras["overkill"]))
    if "accurate_leading" not in extras:
        raise SystemExit("the dump has no accurate_leading: run tools/sim_engine.py dump again")
    for w in weapons:
        radar = extras["radar"].get(w["name"]) or {}
        w["radar_wobble_penalty"], w["radar_dot_penalty"] = float(radar.get("wobble", 0.0)), float(radar.get("dot", 0.0))
        w["accurate_leading"] = int(extras["accurate_leading"].get(w["name"], 0))
        # weapon_projectile_retarget.lua's ApplyProjectileLead: by distance over speed, whatever the def's limit.
        if w["name"] in extras["gadget_lead"]:
            w["lead_limit"] = -1.0
            w["flags"] |= W_LEAD_ADDED
    if not any("wardFireTargetsValues" in b for b in extras["tactics"].values()):
        raise SystemExit("the dump has no wardFireTargetsValues: run tools/sim_engine.py dump again")
    tactics, relations = _tactics(units, extras["tactics"], raw_units)
    table = np.full((len(weapons), len(units)), 100.0, np.float32)
    for n, (_, _, row) in priority.items():
        if n in weapon_index:
            table[weapon_index[n]] = row
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "catalog.json").write_text(json.dumps({"units": units, "weapons": weapons, "tactics": tactics}), "utf8")
    np.savez_compressed(DATA / "priority.npz", priority=table, relations=relations)
    moves = {u["name"]: MOVE_CLASS_NAMES[u["move_class"]] for u in units if u["move_class"] >= 0}
    (ROOT / "bots/zkbot/data/moveclass.json").write_text(json.dumps(moves, indent=0, sort_keys=True), "utf8")
    (ROOT / "bots/zkbot/data/jumprange.json").write_text(json.dumps({u["name"]: u["jump_range"] for u in units if u["jump_range"] > 0.0}, indent=0, sort_keys=True), "utf8")
    (ROOT / "bots/zkbot/data/specials.json").write_text(json.dumps(_specials(units, weapons, raw_units), indent=0, sort_keys=True), "utf8")
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


def export_map(mf, name: str, out: pathlib.Path, height_band: float = HEIGHT_BAND, slope_band: float = SLOPE_BAND, max_parts: int | None = None, seen_share: float | None = None):
    """A map's heights and spots to what the sim and the policy load: its name's slug, .npz and .bundle.npz.  Returns the terrain."""
    sys.path.insert(0, str(ROOT / "sim" / "py"))
    import terrain_map as tm
    import zknav
    from zkbot import terrain as tr
    from zkbot.policy import bundle as policy_bundle
    max_parts = zknav.MAX_PARTS if max_parts is None else max_parts
    seen_share = zknav.SEEN_SHARE if seen_share is None else seen_share
    out.mkdir(parents=True, exist_ok=True)
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
    policy_bundle.from_terrain(terrain).save(out / f"{_slug(name)}.bundle.npz")
    print(f"  {len(terrain.regions)} regions; segments per class " + " ".join(f"{c} {len(r['centres'])}" for c, (_, r) in zip(MOVE_CLASS_NAMES, built)))
    print(f"{name}: {cols}x{rows}, classes {MOVE_CLASS_NAMES}, passable " +
          " ".join(f"{(s > 0).mean():.2f}" for s in speed), flush=True)
    return terrain


def maps(names: list[str], height_band: float = HEIGHT_BAND, slope_band: float = SLOPE_BAND, max_parts: int | None = None,
         seen_share: float | None = None) -> None:
    import terrain_map as tm
    for name, path in tm.find_maps(names):
        export_map(tm.load_map(path, name), name, DATA / "maps", height_band, slope_band, max_parts, seen_share)


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
