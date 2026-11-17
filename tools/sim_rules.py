"""Builds sim/data/rules.json: the game rules the simulator takes from Zero-K's gadgets rather than from its units.

  python tools/sim_rules.py [--zk ../Zero-K] [--dump results/simtools/simdump]

* `dont_count`: units that don't keep a team in the game (LuaRules/Gadgets/game_over.lua: its list of names, and any unit
  whose customParams say `dontcount`, from the engine dump's units.json).
* `reveal`: game_over.lua's overwhelming-advantage rule: how many times the other's value, and how often it's looked at.
* `vital`: what keeps an alliance of AIs in the game when finished (game_over.lua's vitalAlive): builders, and factories that
  aren't `notreallyafactory`, by the engine dump's isBuilder and isFactory.  Every team in the sim stands for an AI.
* `dont_count` leaves out the gadget's automatic list, `(not ud.canAttack) and (not ud.speed) and (not ud.isFactory)`: in Lua
  0 is true, so `not ud.speed` is false for every unit, and the list adds none.
* `chickens`: chicken defense (LuaRules/Gadgets/unit_spawner.lua with LuaRules/Configs/spawn_defs.lua): for each difficulty,
  Beginner to Suicidal, the settings the gadget ends up with (spawn_defs' defaults with the difficulty's own over them), its
  kinds of chicken by role, and the eggs chickens leave (features/chicken_eggs.lua, by the engine dump's feature ids).
* `floats`: the amphibious units that float up to fire (LuaRules/Gadgets/unit_impulsefloat_toggle.lua with
  LuaRules/Configs/float_defs.lua), each with its settings, whether it sinks when stunned (its customParams' sink_on_emp, from
  the engine dump), and what its script adds: FLOAT_SCRIPTS.

spawn_defs.lua is run, in Lua 5.1 (`pip install lupa`), with no mod options: what it computes (times in seconds, the
difficulties' time modifiers) comes out as the game has it.  Times stay in seconds, as the gadget's formulas use them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import lupa.lua51 as lupa

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "sim" / "data" / "rules.json"
DIFFICULTIES = ("Chicken: Beginner", "Chicken: Very Easy", "Chicken: Easy", "Chicken: Normal", "Chicken: Hard", "Chicken: Suicidal")
NEVER = 1e9     # seconds: an obsolete time for a kind that never goes out of date

# What spawn_defs.lua reads of the game.  Mod options are empty, as a default chicken game's are.
PRELUDE = """
UnitDefNames = setmetatable({}, {__index = function(t, name) return {id = name} end})
Spring = {GetModOptions = function() return {} end,
          Utilities = {CustomKeyToUsefulTable = function() return nil end, tobool = function(v) return v and v ~= "0" end}}
VFS = {Include = function() end}
"""

# The settings a difficulty may override (unit_spawner.lua's SetGlobals copies every key of it over the defaults).
SETTINGS = ("spawnSquare", "spawnSquareIncrement", "maxBurrows", "minBaseDistance", "maxBaseDistance", "maxAge", "eggDecayTime",
            "burrowEggs", "playerMalus", "burrowSpawnRate", "chickenSpawnRate", "waveRatio", "baseWaveSize", "waveSizeMult",
            "queenSpawnMult", "defensePerWave", "defensePerBurrowKill", "gracePeriod", "gracePenalty", "gracePeriodMin", "rampUpTime",
            "queenTime", "queenHealthMod", "endMiniQueenWaves", "burrowQueenTime", "burrowWaveSize", "burrowRespawnChance",
            "burrowRegressTime", "humanAggroPerBurrow", "humanAggroDecay", "humanAggroMin", "humanAggroMax", "humanAggroWaveFactor",
            "humanAggroWaveMax", "humanAggroDefenseFactor", "humanAggroTechTimeProgress", "humanAggroTechTimeRegress",
            "humanAggroQueenTimeFactor", "humanAggroQueenTimeMin", "humanAggroQueenTimeMax", "techAccelPerPlayer",
            "techTimeFloorFactor", "techTimeMax", "timeSpawnBonus")


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def table(t) -> dict:
    return {k: (table(v) if lupa.lua_type(v) == "table" else v) for k, v in t.items()}


def kinds(d: dict, *fields: str) -> list[dict]:
    """A role's kinds, by unit name (the gadget walks them in Lua's hash order, which no table here depends on)."""
    out = []
    for unit, k in sorted(d.items()):
        rec = {"unit": unit, "time": float(k["time"]), "obsolete": float(k.get("obsolete", NEVER)), "squad_size": float(k["squadSize"])}
        for f in fields:
            rec[f] = float(k.get(f, 1.0))
        out.append(rec)
    return out


def chickens(zk: pathlib.Path, dump: pathlib.Path) -> dict:
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(PRELUDE)
    lua.execute((zk / "LuaRules" / "Configs" / "spawn_defs.lua").read_text("utf8"))
    g = lua.globals()
    gadget = (zk / "LuaRules" / "Gadgets" / "unit_spawner.lua").read_text("utf8")
    out = {
        "burrow": g.burrowName,
        "no_target": sorted(table(g.noTarget)),
        # The ground a burrow and a queen are placed on is tested as these would be (TestBuildOrder).
        "test_building": "energypylon", "test_building_queen": "chicken_dragon",
        "max_tries": int(re.search(r"local maxTries\s*=\s*(\d+)", gadget).group(1)),
        "max_tries_small": int(re.search(r"local maxTriesSmall\s*=\s*(\d+)", gadget).group(1)),
        "difficulties": [],
    }
    difficulties = table(g.difficulties)
    for level, name in enumerate(DIFFICULTIES, start=1):
        d = difficulties[name]
        rec = {"name": name.split(": ")[1], "level": level}
        for key in SETTINGS:
            value = d[key] if key in d else g[key]
            rec[snake(key)] = float(value)
        rec["queen"] = d.get("queenName", g.queenName)
        morph = d.get("queenMorphName", g.queenMorphName)
        rec["queen_morph"] = morph or None
        rec["mini_queen"] = d.get("miniQueenName", g.miniQueenName)
        rec["mini_queen_time"] = [float(v) for _, v in sorted((d.get("miniQueenTime") or table(g.miniQueenTime) or {}).items())]
        morph_time = d.get("queenMorphTime") or table(g.queenMorphTime)
        rec["queen_morph_frames"] = [float(morph_time[1]), float(morph_time[2])]
        rec["offense"] = kinds(d["chickenTypes"])
        rec["defense"] = kinds(d["defenders"], "cost")
        rec["support"] = kinds(d["supporters"])
        # A special power's fields that it doesn't set are null: the gadget has its own fallbacks for them.
        rec["special"] = [{"name": s["name"], "unit": s["unit"], "time": float(s["time"]), "obsolete": float(s.get("obsolete", NEVER)),
                           "max_aggro": float(s["maxAggro"]), "count": s.get("count"), "burrow_ratio": s.get("burrowRatio"),
                           "min_dist": s.get("minDist"), "max_dist": s.get("maxDist"), "cooldown": float(s.get("cooldown", 1.0)),
                           "target_human": bool(s.get("targetHuman", False)), "tie_to_burrow": bool(s.get("tieToBurrow", False))}
                          for _, s in sorted(d["specialPowers"].items())]
        out["difficulties"].append(rec)
    # The eggs: what's in the game's feature defs, by the unit they're named for (unit_spawner.lua makes `name.."_egg"`).
    features = json.loads((dump / "features.json").read_text("utf8", "replace"))
    out["eggs"] = {name[:-len("_egg")]: {"feature": int(f["id"]), "metal": float(f["metal"]), "hp": float(f["maxHealth"]), "radius": float(f["radius"])}
                   for name, f in sorted(features.items()) if name.endswith("_egg") and not name.startswith("commander")}
    return out


# float_defs.lua's names, as the sim takes them (capi.h zk_float_rec).  A missing one is the gadget's nil: no surface wait.
FLOAT_FIELDS = {"initialRiseSpeed": "init_speed", "riseAccel": "rise_accel", "riseUpDrag": "rise_up_drag", "riseDownDrag": "rise_down_drag",
                "sinkAccel": "sink_accel", "sinkUpDrag": "sink_up_drag", "sinkDownDrag": "sink_down_drag", "airAccel": "air_accel",
                "airDrag": "air_drag", "waterHitDrag": "water_hit_drag", "floatPoint": "float_point", "surfaceDampen": "surface_dampen",
                "depthRequirement": "depth", "surfaceWaitPos": "wait_y", "surfaceMinWait": "wait_min", "surfaceMaxWait": "wait_max"}
# What each unit's script does about floating (scripts/amph*.lua), where it's more than floating whenever a weapon aims
# (GG.Floating_AimWeapon).  reload_frames: it calls that only once its first weapon's reload has less than this left
# (amphassault.lua's AimWeapon(2), amphaa.lua's AimWeapon).  block_below: its BlockShot holds every shot while its base is
# below this (amphimpulse.lua's UNDERWATER_DEPTH), its weapon being a water weapon the engine would let fire.  The fake weapons
# of the Buoy, Grizzly and Bulkhead (amphfloater.lua, amphassault.lua, amphsupport.lua), which aim with the real one but as
# water weapons, so from under water too, are what float those: the sim has the real one aim as from the surface instead.
FLOAT_SCRIPTS = {
    "amphassault": {"reload_frames": 90.0},
    "amphaa": {"reload_frames": 70.0},
    "amphimpulse": {"block_below": -32.0},
}


def floats(zk: pathlib.Path, dump: pathlib.Path) -> dict:
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(PRELUDE + "UnitDefs = setmetatable({}, {__index = function() return {customParams = {}} end})")
    defs = table(lua.execute((zk / "LuaRules" / "Configs" / "float_defs.lua").read_text("utf8")))
    units = json.loads((dump / "units.json").read_text("utf8", "replace"))
    assert set(FLOAT_SCRIPTS) <= set(defs), "a script's floating for a unit float_defs.lua doesn't have"
    out = {}
    for name, d in sorted(defs.items()):
        rec = {ours: float(d[theirs]) for theirs, ours in FLOAT_FIELDS.items() if theirs in d}
        rec.setdefault("wait_max", 0.0)
        rec["sink_on_para"] = int((units[name].get("customParams") or {}).get("sink_on_emp") != "0")
        rec.update({"reload_frames": 0.0, "block_below": -1.0e30, **FLOAT_SCRIPTS.get(name, {})})
        out[name] = rec
    return out


def game_over(zk: pathlib.Path, dump: pathlib.Path) -> tuple[list[str], dict, list[str]]:
    text = (zk / "LuaRules" / "Gadgets" / "game_over.lua").read_text("utf8")
    # The list for ordinary games is the second (the first is a campaign's).
    blocks = re.findall(r"doesNotCountList = \{(.*?)\}", text, re.S)
    names = set(re.findall(r'GetUnitDefIdByName\("(\w+)"\)', blocks[1]))
    units = json.loads((dump / "units.json").read_text("utf8", "replace"))
    names |= {n for n, u in units.items() if (u.get("customParams") or {}).get("dontcount") or u.get("isFeature")}
    reveal = {"value_ratio": float(re.search(r"ECON_SUPREMACY_MULT = (\d+)", text).group(1)),
              "every_frames": int(re.search(r"if \(n % (\d+) == 0\) then\s*\n\s*if not gameIsOver", text).group(1))}
    assert "if ud.isBuilder or (ud.isFactory and not ud.customParams.notreallyafactory) then" in text, "game_over.lua's vital units changed"
    vital = {n for n, u in units.items() if n not in names and (u.get("isBuilder") or (
             u.get("isFactory") and not (u.get("customParams") or {}).get("notreallyafactory")))}
    return sorted(names), reveal, sorted(vital)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zk", type=pathlib.Path, default=ROOT.parent / "Zero-K")
    ap.add_argument("--dump", type=pathlib.Path, default=ROOT / "results" / "simtools" / "simdump")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    dont_count, reveal, vital = game_over(args.zk, args.dump)
    rules = {"dont_count": dont_count, "reveal": reveal, "vital": vital, "chickens": chickens(args.zk, args.dump),
             "floats": floats(args.zk, args.dump)}
    args.out.write_text(json.dumps(rules, indent=1) + "\n", "utf8")
    print(f"{args.out}: {len(dont_count)} that don't count, {len(vital)} vital, {len(rules['chickens']['difficulties'])} difficulties, "
          f"{len(rules['chickens']['eggs'])} eggs, {len(rules['floats'])} that float")


if __name__ == "__main__":
    main()
