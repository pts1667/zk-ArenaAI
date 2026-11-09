"""The behaviour-cloning recorder: whole games only, both sides, frames and orders in step, read back as they were (within the narrowing
zkrecord.h describes), and labels read back from them."""
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import policy_clone_data as cd  # noqa: E402
import policy_sim_env as se  # noqa: E402
import zkrecord  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf, tables  # noqa: E402

LIMIT = 30


class Capturing(cd.Recorder):
    """The recorder, keeping a copy of every frame and order it's given, by the file each goes to."""

    def __init__(self, env, out):
        super().__init__(env, out, threads=2)
        self.runs = {seat: ([], []) for seat in range(env.seat_count)}
        self.written: list[list[tuple[list, list]]] = []
        inner, empty = env.step, np.zeros(0, zksim.ORDER_DTYPE)

        def step(progress):
            seats = [g * 2 + side for g in sorted(self.kept) for side in (0, 1)]
            for seat in seats:
                f = env.frame(seat)
                self.runs[seat][0].append(rf.Frame(f.units.copy(), f.sites.copy(), f.glob.copy(), f.features.copy()))
            inner(progress)
            for seat in seats:
                self.runs[seat][1].append(env.script_record.get(seat, empty).copy())
        env.step = step

    def _write(self, g):
        self.written.append([self.runs[g * 2], self.runs[g * 2 + 1]])
        self.runs[g * 2], self.runs[g * 2 + 1] = ([], []), ([], [])
        super()._write(g)


def _record(out: Path, budget: int) -> Capturing:
    env = se.SimEnv(3, 7, ["scripts"], threads=2)
    env.time_limit = LIMIT      # every game ends at the limit, so that each is the same length
    env.begin()
    rec = Capturing(env, out)
    while rec.decisions < budget:
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    return rec


def test_running_games_are_played_out_and_none_is_started_after_closing(tmp_path):
    rec = _record(tmp_path, 100)
    files = sorted(tmp_path.glob("game-*.zkr"))
    assert len(files) == rec.games
    assert rec.games == 6       # the first 3 reach the budget as they end; the 3 started then are played out, and none after
    assert rec.decisions == rec.games * 2 * LIMIT
    assert dict(rec.ended) == {"limit": rec.games}
    assert [n for _, _, n in cd.sides(tmp_path)] == [LIMIT] * 2 * rec.games
    for path in files:
        for frames, orders, info in cd.read_game(path):
            assert len(frames) == len(orders) == LIMIT
            assert info["script"] == "army" and info["result"] == 0.0
    # A file is one game, from its start, in order.
    frames, _, _ = cd.load(files[-1], side=1)
    times = [int(f.glob[rf.G["frame"]]) for f in frames]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_a_game_reads_back_as_it_was_given_within_its_rounding(tmp_path):
    rec = _record(tmp_path, 1)
    assert rec.writer.clamps() == dict.fromkeys(zkrecord.CLAMP_NAMES, 0)
    files = sorted(tmp_path.glob("game-*.zkr"))
    assert len(files) == len(rec.written) == rec.games
    exact = ("uid", "ord_target", "def_id", "flags", "knowledge", "ord_kind", "player", "ally_team", "ord_age")
    units_seen = 0
    for path, given in zip(files, rec.written):
        meta, sides = zkrecord.read(path)
        map_w = float(given[0][0][0].glob[rf.G["map_w"]])
        map_h = float(given[0][0][0].glob[rf.G["map_h"]])
        assert meta["data_version"] == cd.DATA_VERSION
        for (frames, orders), (want_frames, want_orders) in zip(sides, given):
            assert len(frames) == len(want_frames) == LIMIT
            for got, want, o, wo in zip(frames, want_frames, orders, want_orders):
                assert np.array_equal(got.glob.view(np.uint32), want.glob.view(np.uint32))
                assert np.array_equal(got.sites, want.sites)
                assert np.array_equal(o, wo)
                u, w = got.units, want.units
                assert len(u) == len(w)
                units_seen += len(u)
                for name in exact:
                    assert np.array_equal(u[name], w[name]), name
                for name, step in (("x", map_w), ("z", map_h), ("ord_x", 1.5 * map_w), ("ord_z", 1.5 * map_h), ("build_frac", 1.0)):
                    assert np.abs(u[name] - w[name]).max(initial=0.0) <= step / 65535 * 0.5 + 1e-3, name
                assert np.abs(u["y"] - w["y"]).max(initial=0.0) <= 0.05
                for name in ("vx", "vz", "hp_frac", "experience", "blip_top_speed"):
                    assert np.array_equal(u[name], w[name].astype(np.float16).astype(np.float32), equal_nan=True), name
                f, wf = got.features, want.features
                assert np.array_equal(f["feat_id"], wf["feat_id"]) and np.array_equal(f["def_id"], wf["def_id"])
                assert np.array_equal(f["metal"], wf["metal"])
                assert np.abs(f["x"] - wf["x"]).max(initial=0.0) <= map_w / 65535
    assert units_seen > 0


def test_one_side_as_far_as_a_frame_reads_as_the_start_of_the_whole(tmp_path, monkeypatch):
    """What a cloning sample reads (zkrecord.read's side and until), over chunks of 8 frames, so that it stops in the middle."""
    monkeypatch.setattr(zkrecord, "Writer", functools.partial(zkrecord.Writer, chunk_frames=8))
    _record(tmp_path, 1)
    for path in sorted(tmp_path.glob("game-*.zkr"))[:2]:
        _, whole = zkrecord.read(path)
        for side in (0, 1):
            frames, orders = whole[side]
            for until in (1, 8, 9, 17, LIMIT):
                _, part = zkrecord.read(path, side=side, until=until)
                got, got_orders = part[side]
                assert part[1 - side] == ([], [])
                assert len(got) == min(LIMIT, -(-until // 8) * 8) == len(got_orders)
                for a, b, o, p in zip(got, frames, got_orders, orders):
                    assert a.units.tobytes() == b.units.tobytes() and a.features.tobytes() == b.features.tobytes()
                    assert a.glob.tobytes() == b.glob.tobytes() and o.tobytes() == p.tobytes()
            got, got_orders, info = cd.read_side(path, side, until=9)
            assert len(got) == 16 and info["script"] == "army"


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return _same(vars(a), vars(b))
    if hasattr(a, "shape"):
        return a.shape == b.shape and bool((np.asarray(a.cpu() if hasattr(a, "cpu") else a) ==
                                            np.asarray(b.cpu() if hasattr(b, "cpu") else b)).all())
    return a == b


def test_stretches_cut_from_one_read_are_those_read_one_at_a_time(tmp_path):
    """A worker cuts a run of stretches from one decoding, each frame featurized once and shared with the next stretch's history:
    each must be what reading that stretch alone gives, its own names dropped and nothing of another's."""
    import policy_clone_train as ct
    _record(tmp_path, 1)
    maps = {f.name: b for _, b, f in se.pool()}
    ct._setup(maps, max(b.regions for b in maps.values()), len(tables.table().names) + 1, False)
    path = str(sorted(tmp_path.glob("game-*.zkr"))[0])
    length, reach, seed = 4, 3, 11
    run = ct._samples((path, 1, 2, 3, length, reach, seed, 1))
    assert len(run) == 3
    for i, sample in enumerate(run):
        alone = ct._sample((path, 1, 2 + i * length, 1, length, reach, seed + i, 1))
        assert sample["history"] == min(reach, 2 + i * length) and len(sample["counted"]) == length
        assert _same(sample, alone), i


def test_the_scripts_orders_read_back_as_labels(tmp_path):
    rec = _record(tmp_path, 1)
    maps = {f.name: b for _, b, f in se.pool()}
    table = tables.table()
    orders_seen = missed_all = 0
    for path in sorted(tmp_path.glob("game-*.zkr")):
        for frames, given, info in cd.read_game(path):
            for frame, o in zip(frames, given):
                t, decision, missed = cd.labels(frame, o, maps[info["map"]], table, enabled_kinds=se.KINDS)
                assert len(decision.kind) == t.controllable
                assert 0 <= missed <= len(o)
                orders_seen += len(o)
                missed_all += missed
    assert rec.games == 6 and orders_seen > 0
    assert missed_all <= orders_seen // 4
