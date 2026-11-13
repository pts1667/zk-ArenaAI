"""Equal-cost fights between two kinds of unit, the same in the engine and in the simulator: who wins, with what left, how much
of what was fired landed, and how straight the attackers walked.  And one slow shooter at a time against a Glaive that can't be
killed and doesn't shoot back, standing, walking across its front and walking to and from it: how much of each shot lands.  And
one unit at a time with a Thug standing just past its range, whose shield it wears down by firing at the ground short of it
(unit_ward_fire.lua).  And a Mace sent to fight onto a fusion reactor, which the tactical AI has it do from the edge of its
range (its `fightOnlyOverride`).  And mirror fights, the same units on both sides, with each team in the west and made first
in turn: neither a team's number, nor its end of the lane, nor being made first should win them.

  python tools/sim_fights.py engine [--kinds K...]       # Windows: the engine headless, about 2 min, and 4 more for the mirrors
  ~/zkml/bin/python tools/sim_fights.py sim [--tag T]    # WSL: the same fights in the sim, into results/simfights/sim-T.json
  python tools/sim_fights.py compare [--tags T...]       # the table: the engine's against each sim run's

--kinds: only these of fight, aim, ward, keep and mirror.  The engine's are added to what it has from its earlier runs.

Both sides get a fight order onto where the other starts, from outside either's range, on sim_battery.py's levelled lane and
through its gadget (tools/simgame/LuaRules/Gadgets/zz_sim_battery.lua), which logs each shot and each hit for these.  The last
result is in 01-simulator.md, "Validation".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import sim_battery  # noqa: E402

# ZKSIM_PY: another tree's binding and catalog, to run a library built from that tree.
sys.path.insert(0, os.environ.get("ZKSIM_PY") or str(ROOT / "sim/py"))

OUT = ROOT / "results/simfights"
# Slow shooters against raiders, each pair at the same metal: (name, the shooters, how many, the raiders, how many).
FIGHTS = (("ronin-glaive", "cloakskirm", 13, "cloakraid", 18),
          ("crab-glaive", "spidercrabe", 1, "cloakraid", 25),
          ("rogue-scorcher", "shieldskirm", 26, "vehraid", 25))
REPEATS = 5
GAP = 950.0              # between the two lines: past every range here, and past sight
SPACING = 40.0
ROW = 9                  # units abreast
FRAMES = 2700            # 90 s
ORDER_FRAME = 5
LOG_EVERY = 5
# The stretch of a raider's approach its path is measured over: from where the tactical AI may begin to swarm what it's
# going for, to where its own gun reaches.
APPROACH_FAR, APPROACH_NEAR = 450.0, 200.0
# One shooter, holding its position, and a target on a plain move order, which the tactical AI leaves alone.
AIM_SHOOTERS = ("cloakskirm", "spidercrabe", "shieldskirm", "hoverskirm", "cloakarty")
AIM_PATHS = ("still", "across", "along")
AIM_TARGET, AIM_TARGET_HP = "cloakraid", 1000000.0
AIM_FRAMES, AIM_LEG_FRAMES = 1800, 150
AIM_ACROSS_SHARE, AIM_HALF_WIDTH = 0.75, 230.0     # of the shooter's range: how far off the target crosses, and how far to each side
AIM_ALONG_SHARES = (0.4, 0.9)                      # ... and between which distances it walks to and fro
# Ward fire: a rocket, a stream of bolts and a beam, each holding its position, and a Thug that far past its range with its shield full.
WARD_SHOOTERS = ("cloakskirm", "cloakraid", "hoverriot")
WARD_TARGET, WARD_PAST, WARD_FRAMES = "shieldassault", 45.0, 900
WARD_AT = (150, 300, 600, 890)                     # the frames the shield's charge is read at
# What a unit does to something its tactical AI deals with only on a fight order, by another table then: (the unit, the thing).
KEEP = (("hoverriot", "energyfusion"),)
KEEP_GAP, KEEP_FRAMES = 700.0, 2700
# Mirror fights: (the unit, how many a side).  Each is fought MIRROR_REPEATS times with each team in the west, made first and not.
MIRRORS = (("cloakraid", 12), ("cloakskirm", 8), ("shieldassault", 6), ("vehraid", 8), ("hoverriot", 4), ("shieldskirm", 8),
           ("vehraid", 1), ("cloakraid", 1))
MIRROR_REPEATS = 3
# Duels whose two units aren't made one after the other: ten more are made between them and parked in the map's corners, so the
# tactical AI's turns for the two fall 11 frames apart in a game, where a mirror duel's fall on neighbouring frames.
APART = ("cloakraid", "vehraid")
APART_REPEATS, APART_FRAMES, APART_PARKED = 4, 1500, 5


def _line(unit: str, count: int, team: int, x: float, z: float, back: float, facing: int, prefix: str) -> list[dict]:
    rows = []
    for i in range(count):
        row, col = divmod(i, ROW)
        wide = min(ROW, count - row * ROW)
        rows.append({"key": f"{prefix}{i}", "unit": unit, "team": team, "x": x + back * row * SPACING,
                     "z": z + (col - (wide - 1) / 2.0) * SPACING, "facing": facing})
    return rows


def build_scenarios(kinds: list[str] | None = None) -> list[dict]:
    out = _all_scenarios()
    return [s for s in out if not kinds or s["kind"] in kinds]


def _all_scenarios() -> list[dict]:
    _, _, patch = sim_battery._patched_map()
    x0, z0 = patch["x1"] + sim_battery.LANE_PAD, (patch["z1"] + patch["z2"]) * 0.5
    ax, bx = x0 + 250.0, x0 + 250.0 + GAP
    out = []
    for name, a, na, b, nb in FIGHTS:
        for r in range(REPEATS):
            units = _line(a, na, 0, ax, z0, -1.0, 1, "a") + _line(b, nb, 1, bx, z0, 1.0, 3, "b")
            orders = [{"f": ORDER_FRAME, "u": u["key"], "c": "fight", "x": bx if u["team"] == 0 else ax, "z": z0} for u in units]
            out.append({"name": f"{name}/{r}", "kind": "fight", "duration": FRAMES, "log": LOG_EVERY, "fight": True, "units": units, "orders": orders})
    catalog = json.loads((ROOT / "sim/data/catalog.json").read_text("utf8"))
    reach = {u["name"]: u["max_range"] for u in catalog["units"]}
    for shooter in AIM_SHOOTERS:
        for path in AIM_PATHS:
            r = reach[shooter]
            ends = {"still": ((ax + AIM_ACROSS_SHARE * r, z0),) * 2,
                    "across": ((ax + AIM_ACROSS_SHARE * r, z0 - AIM_HALF_WIDTH), (ax + AIM_ACROSS_SHARE * r, z0 + AIM_HALF_WIDTH)),
                    "along": ((ax + AIM_ALONG_SHARES[0] * r, z0), (ax + AIM_ALONG_SHARES[1] * r, z0))}[path]
            units = [{"key": "a0", "unit": shooter, "team": 0, "x": ax, "z": z0, "facing": 1, "hold": True},
                     {"key": "b0", "unit": AIM_TARGET, "team": 1, "x": ends[0][0], "z": ends[0][1], "facing": 3, "hold": True, "fire": 0, "hp": AIM_TARGET_HP}]
            orders = [] if path == "still" else [{"f": f, "u": "b0", "c": "move", "x": ends[(i + 1) % 2][0], "z": ends[(i + 1) % 2][1]}
                                                  for i, f in enumerate(range(ORDER_FRAME, AIM_FRAMES, AIM_LEG_FRAMES))]
            out.append({"name": f"aim/{shooter}/{path}", "kind": "aim", "duration": AIM_FRAMES, "log": LOG_EVERY, "fight": True, "units": units, "orders": orders})
    target = next(u for u in catalog["units"] if u["name"] == WARD_TARGET)
    charge = max(catalog["weapons"][m]["shield_power"] for m in target["mount_weapon"] if m >= 0)
    for shooter in WARD_SHOOTERS:
        units = [{"key": "a0", "unit": shooter, "team": 0, "x": ax, "z": z0, "facing": 1, "hold": True},
                 {"key": "b0", "unit": WARD_TARGET, "team": 1, "x": ax + reach[shooter] + WARD_PAST, "z": z0, "facing": 3, "hold": True, "fire": 0,
                  "shield": charge}]
        out.append({"name": f"ward/{shooter}", "kind": "ward", "duration": WARD_FRAMES, "log": LOG_EVERY, "fight": True, "units": units, "orders": []})
    for unit, thing in KEEP:
        units = [{"key": "a0", "unit": unit, "team": 0, "x": ax, "z": z0, "facing": 1},
                 {"key": "b0", "unit": thing, "team": 1, "x": ax + KEEP_GAP, "z": z0, "facing": 3}]
        orders = [{"f": ORDER_FRAME, "u": "a0", "c": "fight", "x": ax + KEEP_GAP, "z": z0}]
        out.append({"name": f"keep/{unit}", "kind": "keep", "duration": KEEP_FRAMES, "log": LOG_EVERY, "fight": True, "units": units, "orders": orders})
    for unit, count in MIRRORS:
        for west in (0, 1):
            for first in (0, 1):
                for r in range(MIRROR_REPEATS):
                    lines = {west: _line(unit, count, west, ax, z0, -1.0, 1, "ab"[west]), 1 - west: _line(unit, count, 1 - west, bx, z0, 1.0, 3, "ab"[1 - west])}
                    units = lines[first] + lines[1 - first]
                    orders = [{"f": ORDER_FRAME, "u": u["key"], "c": "fight", "x": bx if u["team"] == west else ax, "z": z0} for u in units]
                    out.append({"name": f"mirror/{unit}x{count}/w{west}f{first}/{r}", "kind": "mirror", "duration": FRAMES, "log": LOG_EVERY, "fight": True,
                                "units": units, "orders": orders, "west": west, "first": first})
    height = sim_battery._patched_map()[0]
    depth, width = (n * sim_battery.CELL for n in height.shape)
    for unit in APART:
        for r in range(APART_REPEATS):
            parked = [{"key": f"p{team}{k}", "unit": unit, "team": team, "x": (400.0 + 60.0 * k) if team == 0 else width - 400.0 - 60.0 * k,
                       "z": 400.0 if team == 0 else depth - 400.0, "facing": 1, "hold": True, "fire": 0} for team in (0, 1) for k in range(APART_PARKED)]
            units = [{"key": "a0", "unit": unit, "team": 0, "x": ax, "z": z0, "facing": 1}] + parked + [{"key": "b0", "unit": unit, "team": 1, "x": bx, "z": z0, "facing": 3}]
            orders = [{"f": ORDER_FRAME, "u": "a0", "c": "fight", "x": bx, "z": z0}, {"f": ORDER_FRAME, "u": "b0", "c": "fight", "x": ax, "z": z0}]
            out.append({"name": f"apart/{unit}/{r}", "kind": "apart", "duration": APART_FRAMES, "log": LOG_EVERY, "fight": True, "units": units, "orders": orders})
    return out


def scenarios_lua(scenarios: list[dict]) -> str:
    keys = sim_battery.GAME_KEYS + ("fight",)
    rows = ",\n".join(sim_battery._lua_value({k: s[k] for k in keys if k in s}) for s in scenarios)
    return f"return {{patch={sim_battery._lua_value(sim_battery._patched_map()[2])},\nlist={{\n{rows}\n}}}}\n"


def parse_log(path: pathlib.Path, scenarios: list[dict]) -> dict:
    """{name: {'rows': {key: [[frame, x, z, hp, shield], ...]}, 'shots': {key: n}, 'taken': {key: damage}, 'dead': [key, ...],
    'events': [[frame, 'shot', key] or [frame, 'hit', key, damage, by] or [frame, 'dead', key], ...]}}"""
    out: dict = {s["name"]: {"rows": {}, "shots": {}, "taken": {}, "dead": [], "events": []} for s in scenarios}
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            m = re.search(r"(SIMLOG|SIMSHOT|SIMHIT|SIMDEAD|SIMERROR),(.*)", line)
            if not m:
                continue
            kind, rest = m.group(1), m.group(2).strip().split(",")
            if kind == "SIMERROR":
                print("engine:", rest)
                continue
            cell = out[scenarios[int(rest[0]) - 1]["name"]]
            if kind == "SIMLOG":
                cell["rows"].setdefault(rest[2], []).append([int(rest[1]), float(rest[3]), float(rest[4]), float(rest[7]), float(rest[10]) if len(rest) > 10 else 0.0])
            elif kind == "SIMDEAD":
                cell["dead"].append(rest[2])
                cell["events"].append([int(rest[1]), "dead", rest[2]])
            elif kind == "SIMSHOT":
                cell["shots"][rest[2]] = cell["shots"].get(rest[2], 0) + 1
                cell["events"].append([int(rest[1]), "shot", rest[2]])
            else:
                cell["taken"][rest[2]] = cell["taken"].get(rest[2], 0.0) + float(rest[3])
                cell["events"].append([int(rest[1]), "hit", rest[2], float(rest[3]), rest[4]])
    return out


def run_engine(kinds: list[str] | None = None) -> None:
    import sim_engine
    scenarios = build_scenarios(kinds)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenarios.json").write_text(json.dumps(scenarios), "utf8")
    wd = sim_engine.run("battery", sim_battery.MAP_NAME, timeout=900.0, port=8792,
                        extra_files={"LuaRules/Configs/sim_scenarios.lua": scenarios_lua(scenarios)})
    # Kept beside the result: the next thing the tools run in the engine writes over the log.
    (OUT / "engine-infolog.txt").write_bytes((wd / "infolog.txt").read_bytes())
    parsed = parse_log(OUT / "engine-infolog.txt", scenarios)
    print(f"{sum(1 for v in parsed.values() if v['rows'])} of {len(scenarios)} fights logged; written to {OUT / 'engine.json'}")
    if kinds and (OUT / "engine.json").exists():
        parsed = {**json.loads((OUT / "engine.json").read_text("utf8")), **parsed}
    (OUT / "engine.json").write_text(json.dumps(parsed), "utf8")


def run_sim(tag: str, kinds: list[str] | None = None) -> None:
    import zksim
    scenarios = build_scenarios(kinds)
    catalog = zksim.Catalog()
    height, speed, _ = sim_battery._patched_map()
    data = np.load(sim_battery.MAP_FILE)
    world_map = zksim.Map(height, speed, sim_battery.CELL, sim_battery.MAP_NAME, sim_battery._patched_slope(), data["slope_mod"], data["segments"],
                          zksim.load_routes(data))
    heading_of = {0: 0.0, 1: math.pi / 2, 2: math.pi, 3: -math.pi / 2}
    out: dict = {}
    for i, s in enumerate(scenarios):
        w = zksim.World(catalog, world_map, seed=1000 + i, behaviour=zksim.ORDERS)
        full = {u["unit"]: catalog.units[catalog.index[u["unit"]]]["hp"] for u in s["units"]}
        ids = {u["key"]: w.add(u["unit"], u["team"], u["x"], u["z"], heading_of[u["facing"]], hp=u.get("hp", full[u["unit"]]) / full[u["unit"]])
               for u in s["units"]}
        team = {u["key"]: u["team"] for u in s["units"]}
        for u in s["units"]:
            if u.get("hold"):
                w.set_move_state(ids[u["key"]], 0)
            if "fire" in u:
                w.set_fire_state(ids[u["key"]], u["fire"])
            if "shield" in u:
                w.set_shield(ids[u["key"]], u["shield"])
        w.ready_weapons()
        rows: dict = {k: [] for k in ids}
        shots: dict = {}
        taken: dict = {}
        events: list = []
        was = {k: (0, 0.0, True) for k in ids}
        for tick in range(s["duration"] // zksim.TICK_FRAMES + 1):
            frame = tick * zksim.TICK_FRAMES
            living = [0, 0]
            for k, u in ids.items():
                st = w.get(u)
                shots[k], taken[k] = st.shots_fired, st.damage_taken
                # As the engine's log has them, to the tick.
                events += [[frame, "shot", k]] * (st.shots_fired - was[k][0])
                if st.damage_taken > was[k][1]:
                    events.append([frame, "hit", k, st.damage_taken - was[k][1], "-"])
                if was[k][2] and not st.alive:
                    events.append([frame, "dead", k])
                was[k] = (st.shots_fired, st.damage_taken, bool(st.alive))
                if st.alive:
                    rows[k].append([frame, st.x, st.z, st.hp, st.shield])
                    living[team[k]] += 1
            if not all(living):
                break
            for o in s["orders"]:
                if frame <= o["f"] < frame + zksim.TICK_FRAMES:
                    (w.fight if o["c"] == "fight" else w.move)(ids[o["u"]], o["x"], o["z"])
            w.step(1)
        out[s["name"]] = {"rows": rows, "shots": shots, "taken": taken, "dead": [k for k, u in ids.items() if not w.get(u).alive], "events": events}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sim-{tag}.json").write_text(json.dumps(out), "utf8")
    print(f"written to {OUT / f'sim-{tag}.json'}")


def measure(scenario: dict, cell: dict, catalog: dict) -> dict:
    """One fight's numbers.  Side 0 is the slow shooters, side 1 the raiders."""
    units = {u["name"]: u for u in catalog["units"]}
    weapons = catalog["weapons"]
    spec = {u["key"]: u for u in scenario["units"]}
    last = max((rows[-1][0] for rows in cell["rows"].values() if rows), default=0)
    left, shots, dealt, full = [0.0, 0.0], [0, 0], [0.0, 0.0], [0.0, 0.0]
    for key, u in spec.items():
        d = units[u["unit"]]
        if cell["rows"].get(key) and key not in cell["dead"]:
            left[u["team"]] += d["metal_cost"]
        shots[u["team"]] += cell["shots"].get(key, 0)
        dealt[1 - u["team"]] += cell["taken"].get(key, 0.0)
        full[u["team"]] = max(weapons[m]["damage"][0] for m in d["mount_weapon"])
    # How far a raider walked for each elmo it came closer over its approach, and how far off its straight line it got.
    at = {}
    for key, rows in cell["rows"].items():
        for f, x, z, *_ in rows:
            at.setdefault(f, [[], []])[spec[key]["team"]].append((x, z))
    walked = closed = 0.0
    off = []
    for key, rows in cell["rows"].items():
        if spec[key]["team"] != 1:
            continue
        stretch = []
        for f, x, z, *_ in rows:
            foes = at[f][0]
            near = min(math.hypot(x - ex, z - ez) for ex, ez in foes) if foes else 0.0
            if near < APPROACH_NEAR:
                break
            if near < APPROACH_FAR:
                stretch.append((x, z))
        if len(stretch) < 3:
            continue
        (x1, z1), (x2, z2) = stretch[0], stretch[-1]
        chord = math.hypot(x2 - x1, z2 - z1)
        if chord < 1.0:
            continue
        walked += sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(stretch, stretch[1:]))
        closed += chord
        off += [abs((x - x1) * (z2 - z1) - (z - z1) * (x2 - x1)) / chord for x, z in stretch]
    return {"left": left, "frames": last, "hit_share": [dealt[s] / (shots[s] * full[s]) if shots[s] else 0.0 for s in (0, 1)],
            "shots": shots, "walked_per_closed": walked / closed if closed else float("nan"),
            "off_line": float(np.sqrt(np.mean(np.square(off)))) if off else float("nan")}


def compare(tags: list[str]) -> None:
    scenarios = build_scenarios()
    catalog = json.loads((ROOT / "sim/data/catalog.json").read_text("utf8"))
    sources = {}
    if (OUT / "engine.json").exists():
        sources["engine"] = json.loads((OUT / "engine.json").read_text("utf8"))
    for tag in tags:
        sources[f"sim {tag}"] = json.loads((OUT / f"sim-{tag}.json").read_text("utf8"))
    print(f"{'fight':16s} {'':12s} {'raiders win':>11s} {'shooters left':>13s} {'raiders left':>12s} {'seconds':>8s} {'shooters hit':>12s} {'raiders hit':>11s} "
          f"{'walked/closed':>13s} {'off line':>8s}")
    for name, a, na, b, nb in FIGHTS:
        for label, data in sources.items():
            cells = [measure(s, data[s["name"]], catalog) for s in scenarios if s["kind"] == "fight" and s["name"].startswith(name + "/") and data.get(s["name"], {}).get("rows")]
            if not cells:
                continue
            mean = lambda f: float(np.nanmean([f(c) for c in cells]))  # noqa: E731
            wins = sum(c["left"][1] > c["left"][0] for c in cells)
            print(f"{name:16s} {label:12s} {wins:>7d} of {len(cells)} {mean(lambda c: c['left'][0]):13.0f} {mean(lambda c: c['left'][1]):12.0f} "
                  f"{mean(lambda c: c['frames']) / 30.0:8.1f} {mean(lambda c: c['hit_share'][0]):12.2f} {mean(lambda c: c['hit_share'][1]):11.2f} "
                  f"{mean(lambda c: c['walked_per_closed']):13.3f} {mean(lambda c: c['off_line']):8.1f}")
    # Of each shot at the lone Glaive, the share of a full hit that landed, and how many were fired.
    print(f"\n{'shooter':12s} {'target':7s} " + " ".join(f"{label:>16s}" for label in sources))
    for s in scenarios:
        if s["kind"] != "aim":
            continue
        _, shooter, path = s["name"].split("/")
        cells = []
        for data in sources.values():
            cell = data.get(s["name"])
            if not cell or not cell["rows"]:
                cells.append(f"{'':>16s}")
                continue
            m = measure(s, cell, catalog)
            cells.append(f"{m['hit_share'][0]:9.2f} of {m['shots'][0]:3d}")
        print(f"{shooter:12s} {path:7s} " + " ".join(cells))
    # Ward fire: what the shooter fired, what's in the Thug's shield at each of WARD_AT, and what the Thug itself took.
    print(f"\n{'ward fire':12s} {'':12s} {'shots':>6s} " + " ".join(f"{'at ' + str(f // 30) + ' s':>8s}" for f in WARD_AT) + f" {'damage':>8s}")
    for s in scenarios:
        if s["kind"] != "ward":
            continue
        for label, data in sources.items():
            cell = data.get(s["name"])
            if not cell or not cell["rows"].get("b0") or len(cell["rows"]["b0"][0]) < 5:
                continue
            charge = {row[0]: row[4] for row in cell["rows"]["b0"]}
            print(f"{s['name'].split('/')[1]:12s} {label:12s} {cell['shots'].get('a0', 0):6d} " + " ".join(f"{charge.get(f, float('nan')):8.0f}" for f in WARD_AT) +
                  f" {cell['taken'].get('b0', 0.0):8.0f}")
    # The fight-only table: how near the unit came while the thing stood, against its range, and when the thing died.
    reach = {u["name"]: u["max_range"] for u in catalog["units"]}
    print(f"\n{'onto a building':16s} {'':12s} {'range':>6s} {'nearest':>8s} {'dies at':>8s}")
    for s in scenarios:
        if s["kind"] != "keep":
            continue
        for label, data in sources.items():
            cell = data.get(s["name"])
            if not cell or not cell["rows"].get("b0"):
                continue
            there = {row[0]: (row[1], row[2]) for row in cell["rows"]["b0"]}
            near = min(math.hypot(x - there[f][0], z - there[f][1]) for f, x, z, *_ in cell["rows"]["a0"] if f in there)
            died = next((f"{e[0] / 30.0:6.1f} s" for e in cell["events"] if e[1] == "dead" and e[2] == "b0"), "never")
            unit = s["name"].split("/")[1]
            print(f"{unit:16s} {label:12s} {reach[unit]:6.0f} {near:8.0f} {died:>8s}")
    # Mirror fights: whose side had more left, by the team's number, by its end of the lane and by which was made first.
    print(f"\n{'mirror':16s} {'':12s} {'fights':>6s} {'team 0, 1, even':>16s} {'west, east':>11s} {'made first, second':>19s} {'seconds':>8s}")
    for unit, count in MIRRORS:
        for label, data in sources.items():
            tally = {"team": [0, 0], "west": [0, 0], "first": [0, 0]}
            even, seconds, fights = 0, [], 0
            for s in scenarios:
                cell = data.get(s["name"])
                if s["kind"] != "mirror" or not s["name"].startswith(f"mirror/{unit}x{count}/") or not cell or not cell["rows"]:
                    continue
                m = measure(s, cell, catalog)
                fights += 1
                seconds.append(m["frames"] / 30.0)
                if m["left"][0] == m["left"][1]:
                    even += 1
                    continue
                winner = int(m["left"][1] > m["left"][0])
                tally["team"][winner] += 1
                tally["west"][int(winner != s["west"])] += 1
                tally["first"][int(winner != s["first"])] += 1
            if fights:
                print(f"{unit + ' x' + str(count):16s} {label:12s} {fights:6d} {tally['team'][0]:8d}, {tally['team'][1]:2d}, {even:2d} {tally['west'][0]:6d}, {tally['west'][1]:2d} "
                      f"{tally['first'][0]:14d}, {tally['first'][1]:2d} {float(np.mean(seconds)):8.1f}")
    # Duels with the two units' turns apart: when the first of them died in each, or that neither did.
    print(f"\n{'duel, turns apart':18s} {'':12s} first death in each")
    for unit in APART:
        for label, data in sources.items():
            ends = []
            for s in scenarios:
                cell = data.get(s["name"])
                if s["kind"] != "apart" or not s["name"].startswith(f"apart/{unit}/") or not cell or not cell["rows"]:
                    continue
                died = [e[0] for e in cell["events"] if e[1] == "dead" and e[2] in ("a0", "b0")]
                ends.append(f"{min(died) / 30.0:.1f} s" if died else "none")
            if ends:
                print(f"{unit:18s} {label:12s} " + ", ".join(ends))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("engine", "sim", "compare", "reparse"))
    ap.add_argument("--tag", default="now", help="sim: what to call this run")
    ap.add_argument("--tags", nargs="*", default=["now"], help="compare: the sim runs to show")
    ap.add_argument("--kinds", nargs="*", help="engine, sim: only these kinds of scenario")
    a = ap.parse_args()
    if a.what == "engine":
        run_engine(a.kinds)
    elif a.what == "reparse":
        scenarios = build_scenarios(a.kinds)
        (OUT / "engine.json").write_text(json.dumps(parse_log(OUT / "engine-infolog.txt", scenarios)), "utf8")
    elif a.what == "sim":
        run_sim(a.tag, a.kinds)
    else:
        compare(a.tags)


if __name__ == "__main__":
    main()
