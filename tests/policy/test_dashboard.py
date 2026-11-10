"""The dashboard's server (tools/policy_dashboard.py) and the replays it shows (tools/policy_replay.py)."""
from __future__ import annotations

import gzip
import json
import os
import struct
import zlib

import numpy as np

import policy_dashboard as pd
import policy_replay as pr


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


def test_capture_keeps_each_slots_first_game(tmp_path):
    import policy_sim_env as se

    env = se.SimEnv(3, 7, ["scripts"], threads=2)
    env.time_limit = 30         # 15 s games
    env.begin()
    env.capture = capture = pr.Capture(env, tmp_path, 2, "eval", "a test", ("army script", "army script"))
    steps = 0
    while not capture.done:
        env.step(0.0)
        env.finished_seats()
        steps += 1
    assert steps == 30          # both kept games end at the limit together, and the third slot's isn't kept
    index = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert sorted(r["slot"] for r in index) == [0, 1]
    replay = json.loads(gzip.decompress((tmp_path / index[0]["file"]).read_bytes()))
    assert replay["result"] == {"winner": None, "how": "limit", "seconds": 15.0}
    assert [f["t"] for f in replay["frames"]] == [float(t) for t in range(16)]      # a frame a second, from the start to the end
    n = len(pr.UNIT_FIELDS)
    units = np.array(replay["frames"][-1]["units"]).reshape(-1, n)
    assert set(units[:, 2]) == {0, 1}           # both sides' units
    assert {str(d) for d in units[:, 1]} <= set(replay["defs"])
    assert "features" in replay["frames"][0] and "features" in replay["frames"][-1]
    from zkbot import units as zu
    seen = {int(d) for f in replay["frames"] for d in np.array(f["units"]).reshape(-1, n)[:, 1]}
    assert {d: replay["defs"][str(d)]["name"] for d in seen} == {d: zu.BY_ID[d]["name"] for d in seen}     # the engine's own ids


def test_repair_names_format_ones_units_by_their_engine_ids(tmp_path):
    from zkbot import units as zu
    mex, wind = (next(int(k) for k, v in zu.BY_ID.items() if v["name"] == name) for name in ("staticmex", "energywind"))
    old = {"format": 1, "unit_fields": list(pr.UNIT_FIELDS), "frames": [{"t": 0.0, "units": [7, mex, 0] + [0] * 9 + [8, wind, 1] + [0] * 9}],
           "defs": {str(mex): {"name": "staticmissilesilo", "cost": 1220.0}, str(wind): {"name": "factoryamph", "cost": 711.7}}}
    path = tmp_path / "game0.json.gz"
    pr.save(path, old)
    assert pr.repair([path]) == [path] and pr.repair([path]) == []          # once: then it's at FORMAT
    replay = json.loads(gzip.decompress(path.read_bytes()))
    assert replay["format"] == pr.FORMAT and {k: v["name"] for k, v in replay["defs"].items()} == {str(mex): "staticmex", str(wind): "energywind"}
