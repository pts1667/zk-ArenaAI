"""The dashboard's server (tools/policy_dashboard.py), the replays it shows (tools/policy_replay.py) and the viewer's reading of them
(tools/dashboard/static/replay.js, run in node where there is one)."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import shutil
import struct
import subprocess
import zlib

import numpy as np
import pytest

import policy_dashboard as pd
import policy_replay as pr
from dashboard import render

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(__file__).resolve().parent / "data"


def unpng(data: bytes) -> np.ndarray:
    """The dashboard's own PNGs back: every row Up-filtered."""
    w, h, _, kind = struct.unpack(">IIBB", data[16:26])
    at, idat = 8, b""
    while at < len(data):
        n = struct.unpack(">I", data[at:at + 4])[0]
        if data[at + 4:at + 8] == b"IDAT":
            idat += data[at + 8:at + 8 + n]
        at += 12 + n
    c = 4 if kind == 6 else 3
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * c + 1)
    assert (rows[:, 0] == 2).all()
    return np.cumsum(rows[:, 1:].astype(np.int64), axis=0).astype(np.uint8).reshape(h, w, c)


def test_png_round_trip():
    pixels = np.random.default_rng(1).integers(0, 256, (7, 5, 4), dtype=np.uint8)
    assert (unpng(pd.png(pixels)) == pixels).all()


def test_dxt1_block():
    # Colour 0 pure red, colour 1 pure blue (c0 > c1: four colours), the first row's pixels 0, 1, 2, 3 and the rest 0.
    block = struct.pack("<HHI", 0xF800, 0x001F, 0b11100100)
    pixels = pd.decode_dxt(block, 4, 4, "DXT1")
    assert pixels[0, 0].tolist() == [255, 0, 0, 255] and pixels[0, 1].tolist() == [0, 0, 255, 255]
    assert pixels[0, 2].tolist() == [170, 0, 85, 255] and pixels[0, 3].tolist() == [85, 0, 170, 255]
    assert (pixels[1:] == pixels[0, 0]).all()


def test_stages_status_and_results(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "record-1.json").write_text(json.dumps({"win_rate": 0.25, "records": [{}] * 4}))
    (tmp_path / "pipeline.json").write_text(json.dumps({
        "plan": ["clone", "record-1", "dagger-1", "eval"], "done": ["clone", "record-1"],
        "runs": [{"stage": "clone", "exit": 0}, {"stage": "record-1", "exit": 0}, {"stage": "dagger-1", "exit": 1}],
        "current": {"stage": "eval", "began": 0.0, "pid": os.getpid()}}))
    run = pd.Run(tmp_path)
    status = {s["name"]: s["status"] for s in run.stages()["stages"]}
    assert status == {"clone": "done", "record-1": "done", "dagger-1": "failed", "eval": "running"}
    assert run.summary("record-1")["eval"] == {"win_rate": 0.25}
    data = json.loads((tmp_path / "pipeline.json").read_text())
    data["current"]["pid"] = 2**22 + 1         # past Linux's largest pid: the pipeline was killed in its stage
    (tmp_path / "pipeline.json").write_text(json.dumps(data))
    assert run.stages()["stages"][-1]["status"] == "interrupted"


def test_metrics_thinned_keep_the_last(tmp_path):
    (tmp_path / "ppo").mkdir()
    (tmp_path / "ppo" / "metrics.jsonl").write_text("".join(json.dumps({"iteration": i, "kl": i / 10}) + "\n" for i in range(1, 11)) + '{"iter')
    rows = pd.Run(tmp_path).metrics("ppo", ["iteration"], 4)["rows"]
    assert [r["iteration"] for r in rows] == [1, 4, 7, 10]


def test_log_lines_wait_for_their_end(tmp_path):
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "clone.out"
    hub = pd.Hub()
    watcher = pd.Watcher(pd.Run(tmp_path), hub)
    log.write_text("one\ntw")
    watcher._logs()
    with log.open("a") as f:
        f.write("o\nstep 1\rstep 2\n")
    watcher._logs()
    assert [(e["stage"], e["text"]) for e in hub.lines] == [("clone", "one"), ("clone", "two"), ("clone", "step 1"), ("clone", "step 2")]


def _read(path: pathlib.Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def _index(directory: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]


def _play(env, capture) -> int:
    env.capture = capture
    steps = 0
    while not capture.done:
        env.step(0.0)
        env.finished_seats()
        steps += 1
    return steps


def test_capture_keeps_each_slots_first_game(tmp_path):
    import policy_sim_env as se

    env = se.SimEnv(3, 7, ["scripts"], threads=2)
    env.time_limit = 30         # 15 s games
    env.begin()
    steps = _play(env, pr.Capture(env, tmp_path, 2, "eval", "a test", ("army script", "army script")))
    assert steps == 30          # both kept games end at the limit together, and the third slot's isn't kept
    index = _index(tmp_path)
    assert sorted(r["slot"] for r in index) == [0, 1]
    replay = _read(tmp_path / index[0]["file"])
    assert replay["format"] == 3 and replay["kind"] == "1v1" and "pages" not in replay
    assert replay["result"] == {"winner": None, "how": "limit", "seconds": 15.0}
    # Today's env has no teams of its own: two, each its own alliance, played by the scripts in this mode, named as asked.
    assert [{k: t[k] for k in ("name", "ally", "plays", "outcome")} for t in replay["teams"]] == [
        {"name": "army script", "ally": 0, "plays": "script", "outcome": 0.0}, {"name": "army script", "ally": 1, "plays": "script", "outcome": 0.0}]
    assert index[0]["teams"] == replay["teams"] and index[0]["kind"] == "1v1" and index[0]["winner"] is None
    assert [f["t"] for f in replay["frames"]] == [float(t) for t in range(16)]      # a frame a second, from the start to the end
    assert all(len(f["teams"]) == 2 and len(f["teams"][0]) == len(pr.TEAM_FIELDS) for f in replay["frames"])
    n = len(pr.UNIT_FIELDS)
    units = np.array(replay["frames"][-1]["units"]).reshape(-1, n)
    assert set(units[:, 2]) == {0, 1}           # both teams' units
    assert {str(d) for d in units[:, 1]} <= set(replay["defs"])
    assert "features" in replay["frames"][0] and "features" in replay["frames"][-1]
    from zkbot import units as zu
    seen = {int(d) for f in replay["frames"] for d in np.array(f["units"]).reshape(-1, n)[:, 1]}
    assert {d: replay["defs"][str(d)]["name"] for d in seen} == {d: zu.BY_ID[d]["name"] for d in seen}     # the engine's own ids


class Both:
    """Two captures of the same games."""

    def __init__(self, *captures):
        self.captures = captures

    @property
    def done(self) -> bool:
        return all(c.done for c in self.captures)

    def after_step(self, env) -> None:
        for c in self.captures:
            c.after_step(env)


def test_a_long_games_units_go_to_pages_of_their_own_and_read_as_one_file_does(tmp_path):
    import policy_sim_env as se

    env = se.SimEnv(1, 15, ["scripts"], units=6, threads=2)     # won by side 0 at 4:08
    env.time_limit = 600
    env.begin()
    _play(env, Both(pr.Capture(env, tmp_path / "whole", 1, "eval", "a test"), pr.Capture(env, tmp_path / "paged", 1, "eval", "a test", page_rows=200)))
    whole, paged = _read(tmp_path / "whole" / "game0.json.gz"), _read(tmp_path / "paged" / "game0.json.gz")
    assert "pages" not in whole and len(paged["pages"]) >= 4
    assert whole["result"] == paged["result"] == {"winner": 0, "how": "eliminated", "seconds": 248.5}
    assert [t["outcome"] for t in paged["teams"]] == [1.0, -1.0]
    pages = [_read(tmp_path / "paged" / p["file"]) for p in paged["pages"]]
    assert [(p["page"], p["first"], len(p["frames"])) for p in pages] == [(k, p["first"], p["count"]) for k, p in enumerate(paged["pages"])]
    rows = [f for p in pages for f in p["frames"]]
    assert all(len(p["frames"][0]["units"]) // len(pr.UNIT_FIELDS) <= 200 + 30 for p in pages)
    assert [f["units"] for f in rows] == [f["units"] for f in whole["frames"]]
    assert [{k: f[k] for k in ("t", "teams")} for f in paged["frames"]] == [{k: f[k] for k in ("t", "teams")} for f in whole["frames"]]
    assert all("features" in p["frames"][0] for p in pages)        # a page's wrecks from its own first frame
    assert all(r["features"] == w["features"] for r, w in zip(rows, whole["frames"]) if "features" in w)
    line = _index(tmp_path / "paged")[0]
    assert line["pages"] == len(pages) and line["bytes"] == sum((tmp_path / "paged" / p).stat().st_size for p in [line["file"]] + [p["file"] for p in paged["pages"]])


def test_repair_names_format_ones_units_by_their_engine_ids(tmp_path):
    from zkbot import units as zu
    mex, wind = (next(int(k) for k, v in zu.BY_ID.items() if v["name"] == name) for name in ("staticmex", "energywind"))
    old = {"format": 1, "unit_fields": ["uid", "def", "side"] + list(pr.UNIT_FIELDS[3:]), "frames": [{"t": 0.0, "units": [7, mex, 0] + [0] * 9 + [8, wind, 1] + [0] * 9}],
           "defs": {str(mex): {"name": "staticmissilesilo", "cost": 1220.0}, str(wind): {"name": "factoryamph", "cost": 711.7}}}
    path = tmp_path / "game0.json.gz"
    pr.save(path, old)
    assert pr.repair([path]) == [path] and pr.repair([path]) == []          # once: then it's format 2, as its shape is
    replay = _read(path)
    assert replay["format"] == 2 and {k: v["name"] for k, v in replay["defs"].items()} == {str(mex): "staticmex", str(wind): "energywind"}


# Games of several teams, before the env plays them: a small stand-in for it ------------------------------------------------------

class TeamGames:
    """One game of several teams for `policy_replay.Capture`: the sim (zksim.Batch) and the C++ scripts (policy_scripts_cpp.Scripts)
    with a slot a team, every team played by the scripts, and what the env offers of its teams (`n_teams`, `ally`, `kind`, `plays`,
    `outcome`, `_game_facts`) as the spec has it.  armies: each team starts as that many fighters at its start; 0 for SimEnv's
    start instead, a commander that plays the opening, then the economy script.  An alliance with no unit left is out, and the game
    ends when one is left or at the time limit."""

    def __init__(self, world, starts: list[tuple[float, float]], ally_of: list[int], seed: int = 1, armies: int = 6, time_limit: int = 9600,
                 threads: int = 2, kind: str | None = None):
        import policy_scripts_cpp as psc
        import policy_sim_env as se
        import zksim
        self.se, self.psc, self.zksim = se, psc, zksim
        world_map, bundle, facts = world
        n = len(ally_of)
        self.maps, self.count, self.slots, self.ticks, self.time_limit = [world], 1, n, se.TICKS_PER_DECISION, time_limit
        self.n_teams, self.ally = np.array([n]), np.array([ally_of], np.int32)
        self.kind = [kind or ("ffa" if len(set(ally_of)) == n else "teams")]
        self.bundle, self.decisions, self.result = [bundle], np.zeros(1, np.int32), np.zeros(1)
        self.outcome, self.out, self._replace, self.how, self.capture = np.zeros((1, n)), np.zeros(n, bool), [], "limit", None
        rng = np.random.default_rng(seed)
        table = se.tables.table()
        self.batch = zksim.Batch(se.catalog(), 1, threads=threads, cap=se.UNIT_RECORDS, teams=n)
        self.cpp = psc.Scripts(psc.ScriptsData(se.ScriptData(table, se.catalog()), se.EcoData(table, se.catalog())), 1, 1, teams=n)
        ground = se.Ground(bundle, facts)
        flat = [v for p in starts for v in p]
        self.seats = None
        if armies:
            types = se.army_types(table)
            pick = rng.integers(0, len(types["index"]), armies)
            placed = [se.place(bundle, rng, types["classes"][pick], x, z) for x, z in starts]
            self.batch.reset(0, world_map, int(rng.integers(1 << 62)), np.tile(types["index"][pick], n), np.repeat(np.arange(n), armies),
                             np.concatenate([x for x, _ in placed]), np.concatenate([z for _, z in placed]), base=flat, sites=bundle.sites(), ally_of=ally_of)
            self.cpp.reset(0, ground, flat, flat, economy=False, aimer=False, ally_of=ally_of)
        else:
            opening = se.Opening(se.catalog(), 1)
            types = [opening.commanders[int(rng.integers(len(opening.commanders)))] for _ in range(n)]
            self.batch.reset(0, world_map, int(rng.integers(1 << 62)), np.array(types), np.arange(n), np.array([x for x, _ in starts]),
                             np.array([z for _, z in starts]), commander=np.arange(n), base=flat, sites=bundle.sites(), ally_of=ally_of)
            self.batch.economy(0, bundle.spot_list)

            def enemy(t: int) -> tuple[float, float]:
                return min((starts[u] for u in range(n) if ally_of[u] != ally_of[t]), key=lambda p: np.hypot(p[0] - starts[t][0], p[1] - starts[t][1]))
            self.seats = [opening.seat(rng, t, ground, starts[t], enemy(t)) for t in range(n)]
            self.cpp.reset(0, ground, flat, [v for s in self.seats for v in s.plan.rally], economy=True, aimer=True,
                           seeds=[int(rng.integers(1 << 62)) for _ in range(n)], ally_of=ally_of)
        self.batch.add_features(0, facts.features)
        self.batch.step(ticks=0)

    def plays(self, g: int, team: int) -> str:
        return "script"

    def finished_seats(self) -> list:
        return []

    def _opening(self, team: int):
        """The commander's next step, as SimEnv's `_opening_orders`."""
        seat = self.seats[team] if self.seats else None
        if seat is None or seat.done:
            return None
        n = int(self.batch.counts[0, team])
        units = self.batch.records[0, team, :n]
        if not n or units["uid"][0] != team:
            seat.done = True
            return None
        if not units["flags"][0] & self.se.rf.IDLE and self.decisions[0] < seat.deadline:
            return None
        orders = seat.advance(units, int(self.decisions[0]))
        return np.array(orders, self.zksim.ORDER_DTYPE) if orders else None

    def step(self, progress: float = 0.0) -> None:
        n, zksim = self.slots, self.zksim
        busy, post, counts = np.full(n, -1, np.int32), [], np.zeros(n, np.int32)
        for t in range(n):
            orders = self._opening(t)
            if orders is not None:
                post.append(orders)
                counts[t] = len(orders)
            if self.seats and not self.seats[t].done:
                busy[t] = t
        joined = (np.concatenate(post), np.concatenate(([0], np.cumsum(counts))).astype(np.int32)) if post else None
        orders, offsets, *_ = self.cpp.step(self.batch, self.decisions, np.full(n, self.psc.SEAT_ARMY, np.int32), busy, self.ticks,
                                            self.se.SCRIPT_EVERY, self.se.ECO_EVERY, post=joined, group_orders=False)
        self.batch.step(orders, offsets, ticks=self.ticks)
        self.decisions += 1
        self._replace = []
        allies = sorted(set(self.ally[0].tolist()))
        alive = self.batch.stats[0, :n, 1] > 0
        living = [a for a in allies if alive[self.ally[0] == a].any()]
        for t in range(n):
            if self.ally[0, t] not in living and not self.out[t]:
                self.outcome[0, t], self.out[t] = -1.0 / (len(allies) - 1), True
        if len(living) <= 1 or self.decisions[0] >= self.time_limit:
            self.how = "eliminated" if len(living) <= 1 else "limit"
            for t in np.flatnonzero(~self.out):
                self.outcome[0, t] = (len(allies) - len(living)) / ((len(allies) - 1) * len(living))
            self.result[0] = self.outcome[0, 0]
            self._replace = [0]
        if self.capture is not None:
            self.capture.after_step(self)

    def _game_facts(self, g: int, team: int, won: bool, lost: bool) -> dict:
        stats, names = self.batch.stats[g], self.se.catalog().units
        enemies = self.ally[g] != self.ally[g, team]
        return {"map": self.maps[0][2].name, "seconds": float(self.decisions[g]) * self.ticks * self.zksim.TICK_FRAMES / self.zksim.FRAMES_PER_SECOND,
                "result": "won" if won else ("lost" if lost else "drew"), "how": self.how, "own_value": float(stats[team, 0]),
                "enemy_value": float(stats[:self.slots, 0][enemies].sum()), "kind": self.kind[g], "variant": "standard",
                "own_factory": names[self.seats[team].factory_type]["name"] if self.seats else ""}


def _starts(bundle, n: int) -> list[tuple[float, float]]:
    """n starts round the middle of a map, in whole elmos."""
    turn = 2 * np.pi * np.arange(n) / n
    return [(float(round(bundle.map_w * (0.5 + 0.36 * np.cos(a)))), float(round(bundle.map_h * (0.5 + 0.36 * np.sin(a))))) for a in turn]


@pytest.fixture(scope="module")
def world():
    import policy_scripts_cpp as psc
    import policy_sim_env as se
    try:
        psc.lib()
    except OSError:
        pytest.skip("needs the built libzkscripts")
    return next(m for m in se.pool() if m[2].name.startswith("titanduel"))      # 5120 x 5120: room for eight starts


@pytest.mark.parametrize("ally_of, kind", [([0, 0, 1, 1], "teams"), ([0, 1, 2, 3, 4, 5, 6, 7], "ffa")])
def test_capture_records_every_team_of_a_game_of_several_and_its_alliance(tmp_path, world, ally_of, kind):
    env = TeamGames(world, _starts(world[1], len(ally_of)), ally_of, seed=3, time_limit=40)
    _play(env, pr.Capture(env, tmp_path, 1, "eval", "teams", ("policy", "army script")))
    replay = _read(tmp_path / "game0.json.gz")
    n = len(ally_of)
    assert replay["kind"] == kind and len(replay["teams"]) == n
    assert [(t["name"], t["ally"], t["plays"]) for t in replay["teams"]] == [("army script", a, "script") for a in ally_of]
    assert all(len(f["teams"]) == n for f in replay["frames"])
    units = np.array(replay["frames"][0]["units"]).reshape(-1, len(pr.UNIT_FIELDS))
    assert sorted(set(units[:, 2].tolist())) == list(range(n)) and (np.bincount(units[:, 2]) == 6).all()      # each team's own, once
    outcome = [t["outcome"] for t in replay["teams"]]
    if replay["result"]["winner"] is None:       # at the limit, every alliance still in: each gets nothing
        assert replay["result"]["how"] == "limit" and outcome == [0.0] * n
    line = _index(tmp_path)[0]
    assert line["kind"] == kind and line["teams"] == replay["teams"]


# The viewer's reading (static/replay.js), in node: what it makes of each replay -----------------------------------------------------

READER = r"""
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const Replay = require(process.argv[2]);
const read = (p) => JSON.parse(zlib.gunzipSync(fs.readFileSync(p)).toString());
(async () => {
  const out = [];
  for (const file of process.argv.slice(3)) {
    const json = read(file), format = json.format;
    let fetched = 0;
    const R = Replay.prepare(json, {}, async (name) => { fetched++; return read(path.join(path.dirname(file), name)); });
    const rows = Array(R.n).fill(0);
    let deaths = 0, most = 0, featureFrames = 0;
    for (let i = 0; i < R.N; i++) {
      if (!R.loaded(i)) await R.ensure(i);
      const f = R.frame(i);
      for (let o = 0; o < f.u.length; o += R.NF) rows[f.u[o + R.F.team]]++;
      deaths += (R.deaths(i) || []).length / 4;
      if (R.features(i)) featureFrames++;
      most = Math.max(most, R.pages.filter((p) => p.frames).length);
    }
    out.push({ format, kind: R.kindName, n: R.n, frames: R.N, end: R.end, pages: R.pages.length, fetched, most, rows, deaths, featureFrames,
      alliances: R.alliances.map((a) => [a.letter, a.ally, a.teams]), slot: R.slot, marks: R.marks, winner: json.result.winner,
      teams: json.teams.map((t) => [t.name, t.ally, t.plays]), series: R.value.map((v) => v.length), unitFields: json.unit_fields });
  }
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _viewer_reads(tmp_path: pathlib.Path, *files: pathlib.Path) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node, to run the viewer's reading")
    script = tmp_path / "read.js"
    script.write_text(READER)
    out = subprocess.run([node, str(script), str(ROOT / "tools" / "dashboard" / "static" / "replay.js"), *map(str, files)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _rows_by_team(replay: dict, directory: pathlib.Path) -> list[int]:
    frames = replay["frames"] if "pages" not in replay else [f for p in replay["pages"] for f in _read(directory / p["file"])["frames"]]
    teams = np.concatenate([np.array(f["units"], np.int64).reshape(-1, len(replay["unit_fields"]))[:, 2] for f in frames])
    return np.bincount(teams, minlength=len(replay.get("teams") or replay.get("sides"))).tolist()


def test_the_viewer_reads_format_2_and_format_3_of_two_four_and_eight_teams_paged_or_not(tmp_path, world):
    import policy_sim_env as se

    env = se.SimEnv(1, 15, ["scripts"], units=6, threads=2)     # today's env: two teams, won by team 0 at 4:08
    env.time_limit = 600
    env.begin()
    _play(env, Both(pr.Capture(env, tmp_path / "two", 1, "eval", "a test"), pr.Capture(env, tmp_path / "paged", 1, "eval", "a test", page_rows=200)))
    for n, ally_of in ((4, [0, 0, 1, 1]), (8, list(range(8)))):
        env = TeamGames(world, _starts(world[1], n), ally_of, seed=3, time_limit=40)
        _play(env, pr.Capture(env, tmp_path / f"teams{n}", 1, "eval", "teams"))
    old = DATA / "replay_format2.json.gz"         # written by format 2's writer (2026-11-10), from such a game as "two" of the sim's then
    files = [old, *(tmp_path / d / "game0.json.gz" for d in ("two", "paged", "teams4", "teams8"))]
    two, paged, old_seen, four, eight = (_viewer_reads(tmp_path, *files)[k] for k in (1, 2, 0, 3, 4))
    assert old_seen["format"] == 2 and two["format"] == 3
    for k in ("kind", "n", "frames", "end", "rows", "deaths", "featureFrames", "alliances", "slot", "marks", "winner", "series", "unitFields"):
        assert two[k] == paged[k], k                     # the same game, read the same paged or not
    for k in ("kind", "n", "alliances", "slot", "marks", "winner", "unitFields"):
        assert old_seen[k] == two[k], k                  # the same line-up and winner, read the same from either format
    assert old_seen["frames"] == 137 and old_seen["series"] == [137, 137] and old_seen["deaths"] > 0 and min(old_seen["rows"]) > 0
    assert two["kind"] == "1v1" and two["alliances"] == [["A", 0, [0]], ["B", 1, [1]]] and two["winner"] == 0 and not two["marks"]
    assert old_seen["teams"] == [["policy", 0, "policy"], ["army script", 1, "script"]]
    assert two["rows"] == _rows_by_team(_read(files[1]), files[1].parent) and two["deaths"] > 0
    assert two["pages"] == 1 and paged["pages"] >= 4 and paged["fetched"] == paged["pages"] and paged["most"] <= 4
    assert four["kind"] == "2v2" and four["alliances"] == [["A", 0, [0, 1]], ["B", 1, [2, 3]]] and four["marks"]
    assert eight["kind"] == "8-way free-for-all" and [a[2] for a in eight["alliances"]] == [[t] for t in range(8)]
    assert eight["slot"] == list(range(8)) and eight["series"] == [eight["frames"]] * 8
    for r, f in ((four, files[3]), (eight, files[4])):
        assert r["rows"] == _rows_by_team(_read(f), f.parent) and min(r["rows"]) > 0


# The dashboard's fragments -----------------------------------------------------------------------------------------------------

def test_a_replay_card_says_the_kind_of_game_and_who_won_from_either_formats_index_line():
    two = {"url": "/replay/d/game0.json.gz", "map": "agalta", "winner": 1, "how": "eliminated", "seconds": 61,
           "sides": [{"name": "policy", "variant": "standard", "factory": "factorycloak", "value": 10}, {"name": "army script", "value": 20}]}
    html = render._replay_item(two)
    assert "army script won" in html and "1v1" in html and 'class="sw t1"' in html
    teams = [{"name": "policy", "ally": 0}, {"name": "army script", "ally": 0}, {"name": "army script", "ally": 1}, {"name": "army script", "ally": 1}]
    html = render._replay_item({**two, "format": 3, "kind": "teams", "teams": teams, "winner": 1})
    assert "alliance B won" in html and "2v2" in html and html.count('class="sw t') == 4 + 2       # the line-up, and the winners in the badge
    ffa = [{"name": "army script", "ally": a} for a in range(8)]
    html = render._replay_item({**two, "format": 3, "kind": "ffa", "teams": ffa, "winner": None})
    assert "no sole winner" in html and "8-way free-for-all" in html and 'class="sw t7"' in html


def test_a_ppo_tests_numbers_are_its_1v1s_with_each_kind_tallied_when_the_games_say_theirs():
    games = [{"result": "won"}, {"result": "lost"}, {"result": "drew"}]
    assert pd.test_summary({"iteration": 3, "games": games}) == {"iteration": 3, "decisions": None, "opponent": None, "games": 3, "won": 1, "drew": 1}
    kinds = games + [{"result": "won", "kind": "ffa"}, {"result": "lost", "kind": "teams"}, {"result": "won", "kind": "1v1"}]
    t = pd.test_summary({"iteration": 4, "games": kinds})
    assert (t["games"], t["won"], t["drew"]) == (4, 2, 1)
    assert t["kinds"] == {"1v1": {"games": 4, "won": 2, "drew": 1}, "ffa": {"games": 1, "won": 1, "drew": 0}, "teams": {"games": 1, "won": 0, "drew": 0}}
    html = render._ppo_body({"last": {"iteration": 4, "lead_share": 0.5}, "tests": [t]})
    assert "1v1 2/4 · ffa 1/1 · teams 0/1" in html and "lead rule share" in html
