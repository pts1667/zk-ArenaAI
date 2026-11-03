"""Build bots/zkbot/data/abilities.json from the Zero-K unit Lua files.

Stats (cost, HP, speed, DPS) don't show what makes several units special:
the Dante's manual-fire napalm salvo, the Reef's disarm missiles, EMP/disarm/
slow weapons, shields, cloak, jumping, gravity guns, capture rays, drones.
This reads each unit's description, its manual-fire ("D-gun") weapon and the
weapon/unit custom params, and writes per unit:

  description      the game's own one-line role description
  desc_roles       role words found in it (raid, riot, skirm, assault, ...)
  tags             abilities: manualfire, cloak, area_cloak, shield, jump,
                   emp, disarm, slow, fire, gravity, capture, torpedo,
                   stockpile, kamikaze, transport, jammer, radar, drones,
                   morph, stealth, sonar_stealth, regen, teleport
  manual_weapon    {name, range, damage, aoe, reload, effects, desc} or null
  sustained_dps    ground DPS without manual-fire weapons (they fire rarely)
  scripts          notable calls found in the unit's animation script

Requires ``pip install lupa`` (offline tool only; the bot reads the JSON).

  python tools/extract_abilities.py [C:/path/to/Zero-K]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import lupa.lua51 as lupa

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_weapons import PRELUDE, GROUND_CATS, lower, num  # noqa: E402

ZK = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Thoma\Documents\sources\Zero-K")
OUT = pathlib.Path(__file__).resolve().parent.parent / "bots" / "zkbot" / "data" / "abilities.json"

# Description words -> role.  Order: the first match is the primary role.
DESC_ROLES = [
    (r"\braid", "raid"),
    (r"\bscout", "scout"),
    (r"\briot", "riot"),
    (r"\bskirmish", "skirm"),
    (r"\bassault", "assault"),
    (r"\bsnip", "snipe"),
    (r"artillery|bombard", "arty"),
    (r"anti-?air|\baa\b|fighter", "aa"),
    (r"anti-?heavy|anti-?strider", "antiheavy"),
    (r"bomb\b|\bbomber|crawling bomb|kamikaze", "bomb"),
    (r"\bshield", "shield"),
    (r"jammer|cloaker|area cloak", "support"),
    (r"construct|builder|engineer", "builder"),
    (r"transport", "transport"),
    (r"\bcapture|mind control", "capture"),
]

SCRIPT_MARKERS = {
    "GG.SetWantedCloaked": "cloak",
    "GG.Floating_": "floating",
    "armored": "armor",
    "GG.TakeOffFuncs": "takeoff",
    "GG.DontFireRadar_": "radar_fire_toggle",
    "GG.QueueUnitDescruction": "self_destructs",
    "GG.att_ReloadChange": "reload_change",
}


def weapon_effects(w: dict) -> list[str]:
    cp = w.get("customparams") or {}
    out = []
    if w.get("paralyzer") or num(cp.get("extra_damage")) > 0:
        out.append("emp")
    if cp.get("disarmdamagemult") or cp.get("disarmdamageonly"):
        out.append("disarm")
    if cp.get("timeslow_damagefactor") or cp.get("timeslow_onlyslow"):
        out.append("slow")
    if cp.get("setunitsonfire"):
        out.append("fire")
    if (cp.get("gravity_pos") or cp.get("gravity_neg") or cp.get("impulse")
            or cp.get("area_damage_is_impulse") or num(w.get("impulsefactor")) < 0):
        out.append("gravity")
    if cp.get("captureray") or cp.get("capture_scaling"):
        out.append("capture")
    if str(w.get("weapontype") or "").lower() == "torpedolauncher":
        out.append("torpedo")
    elif w.get("waterweapon"):
        out.append("underwater")
    if w.get("stockpile"):
        out.append("stockpile")
    return out


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
        for uname, u in lower(ret).items():
            if not isinstance(u, dict):
                continue
            cp = u.get("customparams") or {}
            desc = str(u.get("description") or "")
            tags: set[str] = set()
            if u.get("cancloak") or num(u.get("cloakcost")) > 0:
                tags.add("cloak")
            if cp.get("area_cloak"):
                tags.add("area_cloak")
            if cp.get("canjump"):
                tags.add("jump")
            if num(u.get("transportcapacity")) > 0:
                tags.add("transport")
            if num(u.get("radardistancejam")) > 0:
                tags.add("jammer")
            if num(u.get("radardistance")) > 0:
                tags.add("radar")
            if u.get("stealth"):
                tags.add("stealth")
            if u.get("sonarstealth"):
                tags.add("sonar_stealth")
            if u.get("kamikaze"):
                tags.add("kamikaze")
            if cp.get("morphto") or any(str(k).startswith("morphto") for k in cp):
                tags.add("morph")
            if cp.get("teleporter_throughput") or cp.get("teleporter_beacon_unit"):
                tags.add("teleport")
            if any(str(k).startswith("drone") or "drone" in str(k) for k in cp):
                tags.add("drones")
            if num(u.get("idleautoheal")) >= 20 or cp.get("amph_regen"):
                tags.add("regen")
            wdefs = u.get("weapondefs") or {}
            mounts = u.get("weapons") or {}
            manual = None
            sustained = 0.0
            for _, m in sorted(mounts.items()) if isinstance(mounts, dict) else []:
                if not isinstance(m, dict):
                    continue
                wname = str(m.get("def") or "").lower()
                w = wdefs.get(wname)
                if not isinstance(w, dict):
                    continue
                wcp = w.get("customparams") or {}
                if wcp.get("bogus") or "fake" in wname or wname.startswith("bogus"):
                    continue
                if str(w.get("weapontype") or "").lower() == "shield":
                    tags.add("shield")
                    continue
                eff = weapon_effects(w)
                dmg = w.get("damage") or {}
                d = num(wcp.get("stats_damage")) or num(dmg.get("default"))
                reload = max(0.033, num(w.get("reloadtime"), 1.0))
                shots = max(1.0, num(w.get("projectiles"), 1.0)) * max(1.0, num(w.get("burst"), 1.0))
                is_manual = bool(w.get("commandfire")) or str(w.get("weapontype") or "").lower() == "dgun"
                if is_manual:
                    tags.add("manualfire")
                    manual = {
                        "name": wname,
                        "range": num(w.get("range")),
                        "damage": round(d * shots, 1),
                        "aoe": num(w.get("areaofeffect")),
                        "reload": reload,
                        "effects": eff,
                        "desc": str(cp.get("manualfire_desc") or ""),
                    }
                    continue
                tags.update(eff)
                cats = str(m.get("onlytargetcategory") or "").upper()
                hits_ground = (not cats) or any(c in cats for c in GROUND_CATS)
                if w.get("canattackground") is False:
                    hits_ground = False
                factor = 0.5 if ("emp" in eff or "disarm" in eff) else 1.0
                if wcp.get("timeslow_onlyslow"):
                    factor *= 0.3
                if hits_ground:
                    sustained += d * shots / reload * factor
            low = desc.lower()
            # In the order the description names them ("Raider/Riot" -> raid, riot).
            found = {}
            for pat, role in DESC_ROLES:
                m = re.search(pat, low)
                if m and role not in found:
                    found[role] = m.start()
            desc_roles = sorted(found, key=found.get)
            scripts = []
            sname = str(u.get("script") or "")
            if sname:
                sp = ZK / "scripts" / sname
                if sp.exists():
                    text = sp.read_text(encoding="utf8", errors="replace")
                    scripts = sorted({v for k, v in SCRIPT_MARKERS.items() if k in text})
            result[uname] = {
                "description": desc,
                "desc_roles": desc_roles,
                "tags": sorted(tags),
                "manual_weapon": manual,
                "sustained_dps": round(sustained, 1),
                "scripts": scripts,
            }
    OUT.write_text(json.dumps(result, indent=0, sort_keys=True))
    print(f"wrote {len(result)} units to {OUT}")
    for f in failures:
        print("  fail", f)
    for n in ("striderdante", "shipcarrier", "cloakraid", "shieldfelon", "jumpblackhole", "vehcapture",
              "spideremp", "cloakjammer", "shieldshield", "amphlaunch", "jumpsumo", "spidercrabe"):
        r = result.get(n)
        if r:
            print(f"  {n:16s} {r['description'][:40]:40s} roles={r['desc_roles']} tags={r['tags']} "
                  f"manual={(r['manual_weapon'] or {}).get('effects')} sdps={r['sustained_dps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
