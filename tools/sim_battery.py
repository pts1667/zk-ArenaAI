"""Stage 1 gate (b): the same scripted scenarios in the engine and in the simulator.

  python tools/sim_battery.py engine                 # Windows: runs the engine headless, 5-10 min
  ~/zkml/bin/python tools/sim_battery.py compare     # WSL: runs the sim, compares, writes the report

Thresholds are those of docs/learned-policy/01-simulator.md section 5.2.  The scenarios are built
from the map file alone, so both sides get the same ones.  Output goes to results/simbattery/.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "sim/py"))

OUT = ROOT / "results/simbattery"
MAP_NAME = "TitanDuel 2.2"
MAP_FILE = ROOT / "sim/data/maps/titanduel_2_2.npz"
CATALOG = ROOT / "sim/data/catalog.json"
CLASSES = ("kbot", "tank", "spider", "amph", "hover", "ship")
CELL = 16.0
LANE_LENGTH = 1500.0
LANE_MARGIN = 96.0
LANE_PAD = 256.0
MOVERS = {"kbot": "cloakraid", "tank": "tankraid", "spider": "spiderassault", "hover": "hoverraid"}
DUELS = (("cloakraid", "cloakraid", 150.0), ("cloakassault", "vehassault", 250.0), ("shieldassault", "spiderassault", 260.0),
         ("hoverraid", "tankraid", 190.0), ("spiderscout", "cloakraid", 140.0), ("cloakskirm", "cloakskirm", 400.0),
         ("tankriot", "tankassault", 300.0), ("jumpassault", "vehassault", 115.0))
DUEL_REPEATS = 5
DUEL_FRAMES = 1200
BLAST_BOMBS = ("jumpbomb", "shieldbomb", "energyfusion")
BLAST_TARGET = "staticstorage"
BLAST_TARGET_HP = 1000000.0   # enough to measure a blast without killing what it hits
BLAST_RINGS = (0.25, 0.5, 0.75)
REORDER_SECONDS = (0.5, 1.0, 2.0, 4.0)
REORDER_FRAMES = 330
REORDER_SHIFT = 60.0
ORDER_FRAME = 5           # on the sim's tick grid, so both sides order at the same frame
ORDER_DELAY_UNITS = ("cloakraid", "cloakcon", "cloakheavyraid", "cloakassault", "shieldraid", "shieldassault", "jumpraid", "jumpassault", "amphraid",
                     "amphfloater", "tankraid", "tankassault", "vehraid", "vehassault", "spiderscout", "spiderassault", "hoverraid", "hoverassault")
ORDER_DELAY_FRAMES = 300
ROUTES_PER_CLASS = 2
SLOPES_PER_CLASS = 2
ROUTE_MIN_DETOUR = 1.1
ROUTE_MAX_CELLS = 110     # 1760 elmos

# Thresholds to pass.
ARRIVAL_FRAMES = 2.0
TURN_FRAMES = 1.0
TTK_SHARE = 0.10
BLAST_SHARE = 0.10
ROUTE_SHARE = 0.05
SLOPE_SHARE = 0.02
ARRIVAL_REACH = 100.0     # units stop short of a goal by up to 50 elmos, depending on the class
LEFT_START = 10.0
REORDER_SHARE = 0.10


def _lane_patch(height: np.ndarray) -> dict:
    """The gentlest rectangle above water; the battery levels it in both engine and sim, because
    the map has no lane long enough that is level for every class."""
    rows, cols = height.shape
    wide, tall = int((LANE_LENGTH + 2 * LANE_PAD) / CELL), int(2 * LANE_PAD / CELL)
    best, where = 1e30, None
    for r in range(4, rows - tall - 4, 4):
        for c in range(4, cols - wide - 4, 4):
            box = height[r:r + tall, c:c + wide]
            if box.min() > 5.0:
                spread = float(box.max() - box.min())
                if spread < best:
                    best, where = spread, (r, c, float(box.mean()))
    if where is None:
        raise SystemExit("no lane on this map")
    r, c, level = where
    return {"x1": c * CELL, "z1": r * CELL, "x2": (c + wide) * CELL, "z2": (r + tall) * CELL, "h": round(level, 1)}


def _patched_map() -> tuple[np.ndarray, np.ndarray, dict]:
    d = np.load(MAP_FILE)
    height, speed = d["height"], d["speed"]
    patch = _lane_patch(height)
    x1, z1, x2, z2 = (int(patch[k] / CELL) for k in ("x1", "z1", "x2", "z2"))
    height = height.copy()
    speed = speed.copy()
    height[z1:z2, x1:x2] = patch["h"]
    speed[:, z1:z2, x1:x2] = 1.0
    return height, speed, patch


def _patched_slope() -> np.ndarray:
    slope = np.load(MAP_FILE)["slope"].copy()
    patch = _patched_map()[2]
    x1, z1, x2, z2 = (int(patch[k] / CELL) for k in ("x1", "z1", "x2", "z2"))
    slope[z1:z2, x1:x2] = 0.0
    return slope


def _line_blocked(grid: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> tuple[bool, float]:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(1, int(dist / CELL))
    total, blocked = 0.0, False
    for i in range(1, steps + 1):
        t = i / steps
        v = grid[int((a[1] + (b[1] - a[1]) * t) / CELL), int((a[0] + (b[0] - a[0]) * t) / CELL)]
        blocked = blocked or v <= 0.0
        total += v
    return blocked, total / steps


def _route_length(grid: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """Walking length of the best route; None if outside the window or cut off."""
    rows, cols = grid.shape
    start, goal = (int(a[1] / CELL), int(a[0] / CELL)), (int(b[1] / CELL), int(b[0] / CELL))
    done: dict[tuple[int, int], float] = {start: 0.0}
    open_ = [(0.0, start, 0.0)]
    while open_:
        t, cur, length = heapq.heappop(open_)
        if cur == goal:
            return length * CELL
        if t > done.get(cur, 1e30):
            continue
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dx and not dz:
                    continue
                n = (cur[0] + dz, cur[1] + dx)
                if not (0 <= n[0] < rows and 0 <= n[1] < cols) or abs(n[0] - goal[0]) > ROUTE_MAX_CELLS or abs(n[1] - goal[1]) > ROUTE_MAX_CELLS:
                    continue
                v = grid[n]
                if v <= 0.0:
                    continue
                step = math.hypot(dx, dz)
                nt = t + step / v
                if nt < done.get(n, 1e30):
                    done[n] = nt
                    heapq.heappush(open_, (nt, n, length + step))
    return None


def _pick_routes(speed: np.ndarray, cls: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Moves blocked on the straight line, and clear moves over slopes."""
    grid = speed[CLASSES.index(cls)]
    rng = np.random.default_rng(seed)
    rows, cols = grid.shape
    blocked, slopes = [], []
    for _ in range(4000):
        if len(blocked) >= ROUTES_PER_CLASS and len(slopes) >= SLOPES_PER_CLASS:
            break
        a = (float(rng.uniform(200, cols * CELL - 200)), float(rng.uniform(200, rows * CELL - 200)))
        angle, dist = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(700, 1400))
        b = (a[0] + math.cos(angle) * dist, a[1] + math.sin(angle) * dist)
        if not (100 < b[0] < cols * CELL - 100 and 100 < b[1] < rows * CELL - 100):
            continue
        if grid[int(a[1] / CELL), int(a[0] / CELL)] <= 0.8 or grid[int(b[1] / CELL), int(b[0] / CELL)] <= 0.8:
            continue
        cut, mean = _line_blocked(grid, a, b)
        if cut and len(blocked) < ROUTES_PER_CLASS:
            length = _route_length(grid, a, b)
            if length and length > dist * ROUTE_MIN_DETOUR:
                blocked.append({"a": a, "b": b, "straight": dist, "route": length})
        elif not cut and 0.6 < mean < 0.9 and len(slopes) < SLOPES_PER_CLASS:
            slopes.append({"a": a, "b": b, "straight": dist, "mean_speed": mean})
    return blocked, slopes


def _facing_toward(a: tuple[float, float], b: tuple[float, float]) -> int:
    dx, dz = b[0] - a[0], b[1] - a[1]
    if abs(dx) > abs(dz):
        return 1 if dx > 0 else 3
    return 0 if dz > 0 else 2


def build_scenarios() -> list[dict]:
    _, speed, patch = _patched_map()
    catalog = json.loads(CATALOG.read_text("utf8"))
    units = {u["name"]: u for u in catalog["units"]}
    weapons = catalog["weapons"]
    x0, z0 = patch["x1"] + LANE_PAD, (patch["z1"] + patch["z2"]) * 0.5
    out: list[dict] = []
    for cls, name in MOVERS.items():
        goal = (x0 + LANE_LENGTH - 100, z0)
        v = units[name]["speed"]
        out.append({"name": f"straight/{cls}", "kind": "straight", "duration": int((LANE_LENGTH - 100) / v) + 60, "log": 1,
                    "units": [{"key": "a", "unit": name, "team": 0, "x": x0, "z": z0, "facing": 1}],
                    "orders": [{"f": ORDER_FRAME, "u": "a", "c": "move", "x": goal[0], "z": goal[1]}], "goal": goal})
        out.append({"name": f"turn/{cls}", "kind": "turn", "duration": 120, "log": 1,
                    "units": [{"key": "a", "unit": name, "team": 0, "x": x0 + 900, "z": z0, "facing": 1}],
                    "orders": [{"f": ORDER_FRAME, "u": "a", "c": "move", "x": x0 + 300, "z": z0}]})
    for a, b, gap in DUELS:
        for r in range(DUEL_REPEATS):
            out.append({"name": f"duel/{a}-{b}/{r}", "kind": "duel", "duration": DUEL_FRAMES, "log": 5, "stop_on_death": True,
                        "units": [{"key": "a", "unit": a, "team": 0, "x": x0 + 500, "z": z0, "facing": 1, "hold": True},
                                  {"key": "b", "unit": b, "team": 1, "x": x0 + 500 + gap, "z": z0, "facing": 3, "hold": True}]})
    target_hit = units[BLAST_TARGET]["hit_radius"]
    for bomb in BLAST_BOMBS:
        w = weapons[units[bomb]["explode_weapon"]]
        radius = w["blast_radius"]
        rows = [{"key": "bomb", "unit": bomb, "team": 0, "x": x0 + 700, "z": z0, "facing": 0, "fire": 0}]
        targets = {}
        for i, share in enumerate(BLAST_RINGS):
            dist = target_hit + radius * share
            angle = 2.0 * math.pi * i / len(BLAST_RINGS)
            key = f"t{i}"
            targets[key] = dist
            rows.append({"key": key, "unit": BLAST_TARGET, "team": 1, "x": x0 + 700 + math.cos(angle) * dist,
                         "z": z0 + math.sin(angle) * dist, "facing": 0, "fire": 0, "hp": BLAST_TARGET_HP})
        out.append({"name": f"blast/{bomb}", "kind": "blast", "duration": 60, "log": 5, "units": rows,
                    "orders": [{"f": 15, "u": "bomb", "c": "kill"}], "weapon": w["name"], "radius": radius, "targets": targets})
    for cls, name in MOVERS.items():
        blocked, slopes = _pick_routes(speed, cls, 7 + CLASSES.index(cls))
        v = units[name]["speed"]
        for kind, rows in (("route", blocked), ("slope", slopes)):
            for i, r in enumerate(rows):
                reach = r.get("route", r["straight"]) / max(0.4, r.get("mean_speed", 0.8))
                out.append({"name": f"{kind}/{cls}/{i}", "kind": kind, "duration": int(reach / v * 1.5) + 90, "log": 5,
                            "units": [{"key": "a", "unit": name, "team": 0, "x": r["a"][0], "z": r["a"][1], "facing": _facing_toward(r["a"], r["b"])}],
                            "orders": [{"f": ORDER_FRAME, "u": "a", "c": "move", "x": r["b"][0], "z": r["b"][1]}], "goal": r["b"],
                            "planned_route": r.get("route"), "straight": r["straight"]})
        # The same goal again, and a goal that shifts each time, as a bot following a target does.
        for command in ("move", "fight"):
            for shifting in (False, True):
                label = f"{command}-shifting" if shifting else command
                for seconds in (0.0,) + REORDER_SECONDS:
                    every = int(seconds * 30)
                    frames = range(ORDER_FRAME, REORDER_FRAMES, every) if every else (ORDER_FRAME,)
                    goal_x = x0 + LANE_LENGTH - 50
                    orders = [{"f": f, "u": "a", "c": command, "x": goal_x,
                               "z": z0 + (REORDER_SHIFT if shifting and i % 2 else 0.0)} for i, f in enumerate(frames)]
                    out.append({"name": f"reorder/{cls}/{label}/{seconds}", "kind": "reorder", "duration": REORDER_FRAMES, "log": 5,
                                "units": [{"key": "a", "unit": name, "team": 0, "x": x0, "z": z0, "facing": 1}],
                                "orders": orders, "group": f"{cls}/{label}", "seconds": seconds})
    for name in ORDER_DELAY_UNITS:
        if name in units:
            out.append({"name": f"orderdelay/{name}", "kind": "orderdelay", "duration": ORDER_DELAY_FRAMES, "log": 1,
                        "units": [{"key": "a", "unit": name, "team": 0, "x": x0, "z": z0, "facing": 1}],
                        "orders": [{"f": f, "u": "a", "c": "move", "x": x0 + LANE_LENGTH - 50, "z": z0} for f in range(ORDER_FRAME, ORDER_DELAY_FRAMES, 35)]})
    return out


def _lua_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(round(v, 3)) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(_lua_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}={_lua_value(x)}" for k, x in v.items() if isinstance(k, str) and x is not None) + "}"
    raise TypeError(type(v))


GAME_KEYS = ("name", "duration", "log", "stop_on_death", "units", "orders")


def scenarios_lua(scenarios: list[dict]) -> str:
    rows = ",\n".join(_lua_value({k: s[k] for k in GAME_KEYS if k in s}) for s in scenarios)
    return f"return {{patch={_lua_value(_patched_map()[2])},\nlist={{\n{rows}\n}}}}\n"


def parse_log(path: pathlib.Path, scenarios: list[dict]) -> dict:
    """{scenario name: {'rows': {key: [[frame, x, z, heading, speed, hp], ...]}, 'dead': {key: frame}}}"""
    out: dict = {}
    begun = 0
    errors = []
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            m = re.search(r"(SIMLOG|SIMDEAD|SIMBEGIN|SIMERROR|SIMDONE),?(.*)", line)
            if not m:
                continue
            kind, rest = m.group(1), m.group(2).strip().split(",")
            if kind == "SIMERROR":
                errors.append(rest)
            elif kind == "SIMBEGIN":
                begun = max(begun, int(rest[0]))
                out.setdefault(scenarios[int(rest[0]) - 1]["name"], {"rows": {}, "dead": {}})["start"] = int(rest[2])
            elif kind == "SIMLOG":
                name = scenarios[int(rest[0]) - 1]["name"]
                cell = out.setdefault(name, {"rows": {}, "dead": {}})
                cell["rows"].setdefault(rest[2], []).append([int(rest[1])] + [float(v) for v in rest[3:]])
            elif kind == "SIMDEAD":
                name = scenarios[int(rest[0]) - 1]["name"]
                out.setdefault(name, {"rows": {}, "dead": {}})["dead"][rest[2]] = int(rest[1])
    print(f"parsed {len(out)} of {len(scenarios)} scenarios, {begun} begun; errors: {errors[:5]}")
    return out


def run_engine() -> None:
    import sim_engine
    scenarios = build_scenarios()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenarios.json").write_text(json.dumps(scenarios), "utf8")
    wd = sim_engine.run("battery", MAP_NAME, timeout=420.0, port=8791,
                        extra_files={"LuaRules/Configs/sim_scenarios.lua": scenarios_lua(scenarios)})
    parsed = parse_log(wd / "infolog.txt", scenarios)
    (OUT / "engine.json").write_text(json.dumps(parsed), "utf8")
    print(f"written to {OUT / 'engine.json'}")


def run_sim(scenarios: list[dict], engine_starts: dict[str, int]) -> dict:
    import zksim
    catalog = zksim.Catalog()
    height, speed, _ = _patched_map()
    data = np.load(MAP_FILE)
    world_map = zksim.Map(height, speed, CELL, MAP_NAME, _patched_slope(), data["slope_mod"], data["segments"], zksim.load_routes(data))
    out: dict = {}
    heading_of = {0: 0.0, 1: math.pi / 2, 2: math.pi, 3: -math.pi / 2}
    for i, s in enumerate(scenarios):
        w = zksim.World(catalog, world_map, seed=1000 + i, behaviour=zksim.ORDERS)
        w.set_frame(engine_starts.get(s["name"], 0))
        ids = {}
        for spec in s["units"]:
            full = catalog.units[catalog.index[spec["unit"]]]["hp"]
            ids[spec["key"]] = w.add(spec["unit"], spec["team"], spec["x"], spec["z"], heading_of[spec.get("facing", 0)],
                                     hp=spec.get("hp", full) / full)
        w.ready_weapons()
        rows: dict = {k: [] for k in ids}
        dead: dict = {}
        ticks = s["duration"] // zksim.TICK_FRAMES + 1
        for tick in range(ticks + 1):
            frame = tick * zksim.TICK_FRAMES        # relative to the scenario, as the orders are
            for k, u in ids.items():
                st = w.get(u)
                if st.alive:
                    rows[k].append([frame, st.x, st.z, st.heading, st.speed, st.hp])
                elif k not in dead:
                    dead[k] = frame
            if dead and s.get("stop_on_death"):
                break
            for o in s.get("orders", []):
                # Orders given within this tick are seen at its start.
                if frame <= o["f"] < frame + zksim.TICK_FRAMES:
                    if o["c"] == "kill":
                        w.kill(ids[o["u"]])
                    elif o["c"] == "fight":
                        w.fight(ids[o["u"]], o["x"], o["z"])
                    else:
                        w.move(ids[o["u"]], o["x"], o["z"])
            w.step(1)
        out[s["name"]] = {"rows": rows, "dead": dead}
    return out


def _crossing(rows: list[list[float]], test, column_value=None) -> float | None:
    """First frame, interpolated, where test(row) holds."""
    prev = None
    for r in rows:
        if test(r):
            if prev is None:
                return float(r[0])
            a, b = column_value(prev), column_value(r)
            share = (a / (a - b)) if a != b else 1.0
            return prev[0] + (r[0] - prev[0]) * min(1.0, max(0.0, share))
        prev = r
    return None


def _arrival(rows: list[list[float]], goal, reach: float = ARRIVAL_REACH) -> float | None:
    distance = lambda r: math.hypot(r[1] - goal[0], r[2] - goal[1]) - reach  # noqa: E731
    return _crossing(rows, lambda r: distance(r) <= 0.0, distance)


def _left_start(rows: list[list[float]], reach: float = LEFT_START) -> float | None:
    x0, z0 = rows[0][1], rows[0][2]
    distance = lambda r: reach - math.hypot(r[1] - x0, r[2] - z0)  # noqa: E731
    return _crossing(rows, lambda r: distance(r) <= 0.0, distance)


def _walk_frames(rows: list[list[float]], goal) -> float | None:
    """From leaving the start to reaching the goal: free of the wait for the unit's slow update,
    whose phase differs per unit."""
    start, end = _left_start(rows), _arrival(rows, goal)
    return None if start is None or end is None else end - start


def _half_turn_frames(rows: list[list[float]]) -> float | None:
    """Frames a half turn takes at the fastest rate seen, free of the same wait."""
    best = 0.0
    for a, b in zip(rows, rows[1:]):
        if b[0] > a[0]:
            turned = abs(math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3])))
            if turned < 2.0:         # a jump this size is sampling across the wrap
                best = max(best, turned / (b[0] - a[0]))
    return math.pi / best if best > 0.0 else None


def _path_length(rows: list[list[float]]) -> float:
    return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(rows, rows[1:]))


def _damage_taken(rows: list[list[float]]) -> float:
    return rows[0][5] - rows[-1][5] if rows else 0.0


def _ttk(cell: dict) -> float | None:
    return min(cell["dead"].values()) if cell["dead"] else None


def zksim_tick_frames() -> int:
    import zksim
    return zksim.TICK_FRAMES


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, check: str, name: str, engine, sim, limit: float, share: bool) -> None:
        if engine is None or sim is None:
            ok, gap = False, None
        else:
            gap = abs(sim - engine) / max(abs(engine), 1e-6) if share else abs(sim - engine)
            ok = gap <= limit
        self.rows.append({"check": check, "name": name, "engine": engine, "sim": sim, "gap": gap, "limit": limit, "share": share, "ok": bool(ok)})

    def print(self) -> None:
        last = None
        for r in self.rows:
            if r["check"] != last:
                last = r["check"]
                passed = [x["ok"] for x in self.rows if x["check"] == last]
                print(f"\n{last}: {sum(passed)}/{len(passed)} within {r['limit']:g}{' share' if r['share'] else ' frames'}")
            fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"  {'ok ' if r['ok'] else 'BAD'} {r['name']:44s} engine {fmt(r['engine']):>9s}  sim {fmt(r['sim']):>9s}  gap {fmt(r['gap']):>7s}")


def compare() -> None:
    scenarios = json.loads((OUT / "scenarios.json").read_text("utf8"))
    engine = json.loads((OUT / "engine.json").read_text("utf8"))
    sim = run_sim(scenarios, {k: v.get("start", 0) for k, v in engine.items()})
    report = Report()
    duels: dict[str, dict[str, list]] = {}
    reorder: dict[str, dict] = {}
    for s in scenarios:
        name, kind = s["name"], s["kind"]
        e = engine.get(name)
        m = sim[name]
        if e is None:
            report.add(kind, name, None, None, 0.0, False)
            continue
        if kind == "straight":
            report.add("straight-line walk (frames)", name, _walk_frames(e["rows"]["a"], s["goal"]), _walk_frames(m["rows"]["a"], s["goal"]), ARRIVAL_FRAMES, False)
        elif kind == "turn":
            report.add("turn-around (frames)", name, _half_turn_frames(e["rows"]["a"]), _half_turn_frames(m["rows"]["a"]), TURN_FRAMES, False)
        elif kind == "duel":
            group = duels.setdefault(name.rsplit("/", 1)[0], {"engine": [], "sim": []})
            group["engine"].append(_ttk(e))
            group["sim"].append(_ttk(m))
        elif kind == "blast":
            for key in s["targets"]:
                report.add("blast damage", f"{name}/{key}", _damage_taken(e["rows"].get(key, [])), _damage_taken(m["rows"].get(key, [])), BLAST_SHARE, True)
        elif kind == "route":
            report.add("path around blocked terrain (length)", name, _path_length(e["rows"]["a"]), _path_length(m["rows"]["a"]), ROUTE_SHARE, True)
            report.add("blocked move (walk frames)", name, _walk_frames(e["rows"]["a"], s["goal"]), _walk_frames(m["rows"]["a"], s["goal"]), ROUTE_SHARE, True)
        elif kind == "slope":
            report.add("clear move over slopes (walk frames)", name, _walk_frames(e["rows"]["a"], s["goal"]), _walk_frames(m["rows"]["a"], s["goal"]), SLOPE_SHARE, True)
        elif kind == "orderdelay":
            continue
        elif kind == "reorder":
            group = reorder.setdefault(s["group"], {})
            group[s["seconds"]] = {"engine": _path_length(e["rows"]["a"]), "sim": _path_length(m["rows"]["a"])}
    for name, group in duels.items():
        e = [v for v in group["engine"] if v is not None]
        m = [v for v in group["sim"] if v is not None]
        engine_ttk = float(np.mean(e)) if e else None
        # The sim sees a death only at the end of a tick, so short duels can't get closer than that.
        limit = max(TTK_SHARE, zksim_tick_frames() / engine_ttk) if engine_ttk else TTK_SHARE
        report.add("duel time to kill (frames, mean of repeats)", name, engine_ttk, float(np.mean(m)) if m else None, limit, True)
    for name, group in reorder.items():
        base = group[0.0]
        for seconds, cell in sorted(group.items()):
            if seconds:
                report.add("re-order mean speed (share of no re-orders)", f"{name}/{seconds}s", cell["engine"] / max(base["engine"], 1.0),
                           cell["sim"] / max(base["sim"], 1.0), REORDER_SHARE, True)
    report.print()
    (OUT / "report.json").write_text(json.dumps(report.rows, indent=1), "utf8")
    total = sum(r["ok"] for r in report.rows)
    print(f"\n{total}/{len(report.rows)} checks within thresholds; written to {OUT / 'report.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("engine", "compare", "reparse"))
    a = ap.parse_args()
    if a.what == "engine":
        run_engine()
    elif a.what == "reparse":
        scenarios = json.loads((OUT / "scenarios.json").read_text("utf8"))
        parsed = parse_log(ROOT / "results/simtools/infolog.txt", scenarios)
        (OUT / "engine.json").write_text(json.dumps(parsed), "utf8")
    else:
        compare()


if __name__ == "__main__":
    main()
