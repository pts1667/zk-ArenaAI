"""Builds sim/data/rules.json: the game rules the simulator takes from Zero-K's gadgets rather than from its units.

  python tools/sim_rules.py [--zk ../Zero-K] [--dump results/simtools/simdump]

* `dont_count`: units that don't keep a team in the game (LuaRules/Gadgets/game_over.lua: its list of names, and any unit
  whose customParams say `dontcount`, from the engine dump's units.json).
* `reveal`: game_over.lua's overwhelming-advantage rule: how many times the other's value, and how often it's looked at.
* `chickens`: chicken defense (LuaRules/Gadgets/unit_spawner.lua with LuaRules/Configs/spawn_defs.lua): for each difficulty,
  Beginner to Suicidal, the settings the gadget ends up with (spawn_defs' defaults with the difficulty's own over them), its
  kinds of chicken by role, and the eggs chickens leave (features/chicken_eggs.lua, by the engine dump's feature ids).

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


def game_over(zk: pathlib.Path, dump: pathlib.Path) -> tuple[list[str], dict]:
    text = (zk / "LuaRules" / "Gadgets" / "game_over.lua").read_text("utf8")
    # The list for ordinary games is the second (the first is a campaign's).
    blocks = re.findall(r"doesNotCountList = \{(.*?)\}", text, re.S)
    names = set(re.findall(r'GetUnitDefIdByName\("(\w+)"\)', blocks[1]))
    units = json.loads((dump / "units.json").read_text("utf8", "replace"))
    names |= {n for n, u in units.items() if (u.get("customParams") or {}).get("dontcount") or u.get("isFeature")}
    reveal = {"value_ratio": float(re.search(r"ECON_SUPREMACY_MULT = (\d+)", text).group(1)),
              "every_frames": int(re.search(r"if \(n % (\d+) == 0\) then\s*\n\s*if not gameIsOver", text).group(1))}
    return sorted(names), reveal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zk", type=pathlib.Path, default=ROOT.parent / "Zero-K")
    ap.add_argument("--dump", type=pathlib.Path, default=ROOT / "results" / "simtools" / "simdump")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    dont_count, reveal = game_over(args.zk, args.dump)
    rules = {"dont_count": dont_count, "reveal": reveal, "chickens": chickens(args.zk, args.dump)}
    args.out.write_text(json.dumps(rules, indent=1) + "\n", "utf8")
    print(f"{args.out}: {len(dont_count)} that don't count, {len(rules['chickens']['difficulties'])} difficulties, "
          f"{len(rules['chickens']['eggs'])} eggs")


if __name__ == "__main__":
    main()
