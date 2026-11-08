"""The behaviour-cloning recorder: whole games only, both sides, frames and orders in step, and labels read back from them."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import policy_clone_data as cd  # noqa: E402
import policy_sim_env as se  # noqa: E402
from zkbot.policy import rawframe as rf, tables  # noqa: E402

LIMIT = 30


def _record(out: Path, budget: int) -> cd.Recorder:
    env = se.SimEnv(3, 7, ["scripts"], threads=2)
    env.time_limit = LIMIT      # every game ends at the limit, so that each is the same length
    env.begin()
    rec = cd.Recorder(env, out)
    while rec.decisions < budget:
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    return rec


def test_running_games_are_played_out_and_none_is_started_after_closing(tmp_path):
    rec = _record(tmp_path, 100)
    files = sorted(tmp_path.glob("game-*.npz"))
    assert len(files) == 2 * rec.games
    assert rec.games == 6       # the first 3 reach the budget as they end; the 3 started then are played out, and none after
    assert rec.decisions == rec.games * 2 * LIMIT
    assert dict(rec.ended) == {"limit": rec.games}
    for path in files:
        frames, orders, info = cd.load(path)
        assert len(frames) == len(orders) == LIMIT
        assert info["script"] == "army" and info["result"] == 0.0
    # A file is one game, from its start, in order.
    frames, _, _ = cd.load(files[-1])
    times = [int(f.glob[rf.G["frame"]]) for f in frames]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_the_scripts_orders_read_back_as_labels(tmp_path):
    rec = _record(tmp_path, 1)
    maps = {f.name: b for _, b, f in se.pool()}
    table = tables.table()
    orders_seen = missed_all = 0
    for path in sorted(tmp_path.glob("game-*.npz")):
        frames, given, info = cd.load(path)
        for frame, o in zip(frames, given):
            t, decision, missed = cd.labels(frame, o, maps[info["map"]], table, enabled_kinds=se.KINDS)
            assert len(decision.kind) == t.controllable
            assert 0 <= missed <= len(o)
            orders_seen += len(o)
            missed_all += missed
    assert rec.games == 6 and orders_seen > 0
    assert missed_all <= orders_seen // 4
