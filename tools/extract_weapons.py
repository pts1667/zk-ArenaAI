"""Build bots/zkbot/data/weapons.json from the Zero-K unit Lua files.

ControllerAI's unit_def_by_id only names the weapon defs, so DPS, area of
effect and air/ground targeting come from the game source.  Requires
``pip install lupa`` (only for this offline tool; the bot reads the JSON).

  python tools/extract_weapons.py [C:/path/to/Zero-K]
"""
from __future__ import annotations

import json
import pathlib
import sys

import lupa.lua51 as lupa

ZK = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Thoma\Documents\sources\Zero-K")
OUT = pathlib.Path(__file__).resolve().parent.parent / "bots" / "zkbot" / "data" / "weapons.json"

PRELUDE = r"""
local magic
local mt = {}
mt.__index = function() return magic end
mt.__call = function() return magic end
for _, k in ipairs({"__add","__sub","__mul","__div","__mod","__pow","__unm","__concat"}) do
  mt[k] = function() return 0 end
end
magic = setmetatable({}, mt)
setmetatable(_G, {__index = function(t, k) return magic end})
function lowerkeys(t)
  local out = {}
  for k, v in pairs(t) do
    if type(k) == "string" then k = string.lower(k) end
    if type(v) == "table" and getmetatable(v) == nil then v = lowerkeys(v) end
    out[k] = v
  end
  return out
end
Shared = setmetatable({BURST_RELIABLE = 1, BURST_UNRELIABLE = 1}, mt)
VFS = setmetatable({Include = function() return magic end}, mt)
include = function() return magic end
"""

AIR_CATS = ("FIXEDWING", "GUNSHIP")
GROUND_CATS = ("LAND", "SINK", "SWIM", "TURRET", "FLOAT", "HOVER", "SHIP", "SUB")


def lower(t):
    if lupa.lua_type(t) != "table":
        return t
    out = {}
    for k, v in t.items():
        if isinstance(k, str):
            k = k.lower()
        out[k] = lower(v)
    return out


def num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main() -> int:
    result = {}
    failures = []
    for path in sorted((ZK / "units").glob("*.lua")):
        lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        lua.execute(PRELUDE)
        try:
            ret = lua.execute(path.read_text(encoding="utf8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {str(exc)[:80]}")
            continue
        if lupa.lua_type(ret) != "table":
            continue
        for uname, udef in lower(ret).items():
            if not isinstance(udef, dict):
                continue
            wdefs = udef.get("weapondefs") or {}
            mounts = udef.get("weapons") or {}
            dps_ground = dps_air = 0.0
            max_range = 0.0
            aoe = 0.0
            paralyze = False
            weapons = []
            for _, m in sorted(mounts.items()) if isinstance(mounts, dict) else []:
                if not isinstance(m, dict):
                    continue
                wname = str(m.get("def") or "").lower()
                w = wdefs.get(wname)
                if not isinstance(w, dict):
                    continue
                cp = w.get("customparams") or {}
                if cp.get("bogus") or "fake" in wname:
                    continue
                dmg = w.get("damage") or {}
                d_default = num(dmg.get("default"))
                d_planes = num(dmg.get("planes"), d_default)
                reload = max(0.033, num(w.get("reloadtime"), 1.0))
                shots = max(1.0, num(w.get("projectiles"), 1.0)) * max(1.0, num(w.get("burst"), 1.0))
                rng = num(w.get("range"))
                cats = str(m.get("onlytargetcategory") or "").upper()
                hits_air = any(c in cats for c in AIR_CATS)
                hits_ground = (not cats) or any(c in cats for c in GROUND_CATS)
                if w.get("canattackground") is False:
                    hits_ground = False
                para = bool(w.get("paralyzer"))
                factor = 0.5 if para else 1.0
                # Slow / disarm damage is real but does not kill.
                if cp.get("disarmdamagemult") or cp.get("disarmdamageonly"):
                    factor *= 0.5
                if cp.get("timeslow_onlyslow"):
                    factor *= 0.3
                if num(cp.get("stats_damage")) > 0:
                    d_default = num(cp.get("stats_damage"))
                dps = d_default * shots / reload * factor
                dps_a = d_planes * shots / reload * factor
                if hits_ground:
                    dps_ground += dps
                if hits_air:
                    dps_air += dps_a
                if rng > 0 and (hits_ground or hits_air):
                    max_range = max(max_range, rng)
                aoe = max(aoe, num(w.get("areaofeffect")))
                paralyze = paralyze or para
                weapons.append(
                    {
                        "name": wname,
                        "dps": round(dps, 1),
                        "range": rng,
                        "air": hits_air,
                        "ground": hits_ground,
                        "aoe": num(w.get("areaofeffect")),
                        "paralyzer": para,
                        "type": str(w.get("weapontype") or ""),
                    }
                )
            result[uname] = {
                "dps_ground": round(dps_ground, 1),
                "dps_air": round(dps_air, 1),
                "range": max_range,
                "aoe": aoe,
                "paralyzer": paralyze,
                "weapons": weapons,
            }
    OUT.write_text(json.dumps(result, indent=0, sort_keys=True))
    print(f"wrote {len(result)} units to {OUT}")
    for f in failures:
        print("  fail", f)
    for n in ("cloakraid", "cloakskirm", "cloakriot", "cloakassault", "cloakarty", "turretlaser",
              "turretmissile", "turretriot", "shieldraid", "vehassault", "cloakaa", "dyntrainer_support_base",
              "tankassault", "gunshipskirm", "turretheavylaser"):
        r = result.get(n)
        if r:
            print(f"  {n:24s} g={r['dps_ground']:7.1f} a={r['dps_air']:6.1f} r={r['range']:5.0f} aoe={r['aoe']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
