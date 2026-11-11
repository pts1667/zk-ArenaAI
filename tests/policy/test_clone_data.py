"""The behaviour-cloning recorder: whole games only, both sides, frames and orders in step, read back as they were (within the narrowing
zkrecord.h describes), and labels read back from them."""
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def _as_given(got: rf.Frame, want: rf.Frame, map_w: float, map_h: float, y_step: float = 0.05) -> int:
    """A frame read back is the one given, within the narrowing zkrecord.h describes: its units, how many."""
    exact = ("uid", "ord_target", "def_id", "flags", "knowledge", "ord_kind", "player", "ally_team", "ord_age")
    assert np.array_equal(got.glob.view(np.uint32), want.glob.view(np.uint32))
    assert np.array_equal(got.sites, want.sites)
    u, w = got.units, want.units
    assert len(u) == len(w)
    for name in exact:
        assert np.array_equal(u[name], w[name]), name
    for name, step in (("x", map_w), ("z", map_h), ("ord_x", 1.5 * map_w), ("ord_z", 1.5 * map_h), ("build_frac", 1.0)):
        assert np.abs(u[name] - w[name]).max(initial=0.0) <= step / 65535 * 0.5 + 1e-3, name
    assert np.abs(u["y"] - w["y"]).max(initial=0.0) <= y_step
    for name in ("vx", "vz", "hp_frac", "experience", "blip_top_speed"):
        assert np.array_equal(u[name], w[name].astype(np.float16).astype(np.float32), equal_nan=True), name
    f, wf = got.features, want.features
    assert np.array_equal(f["feat_id"], wf["feat_id"]) and np.array_equal(f["def_id"], wf["def_id"])
    assert np.array_equal(f["metal"], wf["metal"])
    assert np.abs(f["x"] - wf["x"]).max(initial=0.0) <= map_w / 65535
    return len(u)


def test_a_game_reads_back_as_it_was_given_within_its_rounding(tmp_path):
    rec = _record(tmp_path, 1)
    assert rec.writer.clamps() == dict.fromkeys(zkrecord.CLAMP_NAMES, 0)
    files = sorted(tmp_path.glob("game-*.zkr"))
    assert len(files) == len(rec.written) == rec.games
    units_seen = 0
    for path, given in zip(files, rec.written):
        meta, sides = zkrecord.read(path)
        map_w = float(given[0][0][0].glob[rf.G["map_w"]])
        map_h = float(given[0][0][0].glob[rf.G["map_h"]])
        assert meta["data_version"] == cd.DATA_VERSION
        assert meta["teams"] == meta["allies"] == [0, 1] and meta["kind"] == "1v1"
        assert meta["how"] == ["limit", "limit"] and meta["until"] == [LIMIT, LIMIT]
        assert zkrecord.header(path) == (LIMIT, (0, 1))
        for (frames, orders), (want_frames, want_orders) in zip(sides, given):
            assert len(frames) == len(want_frames) == LIMIT
            for got, want, o, wo in zip(frames, want_frames, orders, want_orders):
                assert np.array_equal(o, wo)
                units_seen += _as_given(got, want, map_w, map_h)
    assert units_seen > 0


def _hand_made_frame(rng: np.random.Generator, team: int, ally: int, frame: int, map_w: float, map_h: float) -> rf.Frame:
    """A team's frame of random units, half of them carried on from uids an earlier frame may have had, and some features."""
    n = int(rng.integers(0, 24))
    u = np.zeros(n, rf.UNIT_DTYPE)
    u["uid"] = rng.permutation(40)[:n] + 1000 * team
    u["ord_target"] = rng.integers(-1, 500, n)
    u["def_id"] = rng.integers(0, 400, n)
    u["x"], u["z"] = rng.uniform(0.0, map_w, n), rng.uniform(0.0, map_h, n)
    u["y"] = rng.uniform(-200.0, 900.0, n)
    for name in ("vx", "vz", "experience", "blip_top_speed"):
        u[name] = rng.normal(0.0, 3.0, n)
    u["hp_frac"], u["build_frac"] = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    u["ord_age"] = rng.integers(0, 3000, n)
    u["ord_x"], u["ord_z"] = rng.uniform(-0.25 * map_w, 1.25 * map_w, n), rng.uniform(-0.25 * map_h, 1.25 * map_h, n)
    u["flags"], u["knowledge"], u["ord_kind"] = rng.integers(0, 1 << 16, n), rng.integers(0, 5, n), rng.integers(-1, 20, n)
    u["player"], u["ally_team"] = team, ally
    k = int(rng.integers(0, 5))
    f = np.zeros(k, rf.FEATURE_DTYPE)
    f["feat_id"], f["def_id"] = rng.permutation(10)[:k] + 50 * team, rng.integers(0, 90, k)
    f["x"], f["z"], f["metal"] = rng.uniform(0.0, map_w, k), rng.uniform(0.0, map_h, k), rng.uniform(0.0, 900.0, k).astype(np.float32)
    glob = rng.normal(0.0, 1000.0, len(rf.GLOBAL_FIELDS)).astype(np.float32)
    glob[rf.G["frame"]], glob[rf.G["player"]], glob[rf.G["ally_team"]] = frame, team, ally
    return rf.Frame(u, np.zeros(0, rf.SITE_DTYPE), glob, f)


def test_of_four_teams_the_three_recorded_read_back_in_their_order_each_as_it_was_given(tmp_path):
    """The recorder reads only the arrays it's given, so a batch of two games of 4 teams made by hand stands in for a free-for-all:
    game 1 keeps teams 3, 0 and 2, in that order, and game 0 teams 1 and 2.  Each side's frames, orders and meta come back, over two
    chunks, and one side alone as far as a frame reads as the start of the whole."""
    rng = np.random.default_rng(5)
    games, teams, cap, feature_cap, frames = 2, 4, 32, 8, 70
    map_w, map_h, y_range = 3072.0, 5120.0, (-256.0, 1024.0)
    sites = np.zeros(3, rf.SITE_DTYPE)
    sites["site_id"], sites["x"], sites["z"] = (0, 1, 2), (100.0, 1500.0, 2900.0), (200.0, 2500.0, 4900.0)
    batch = SimpleNamespace(records=np.zeros((games, teams, cap), rf.UNIT_DTYPE), cap=cap, counts=np.zeros((games, teams), np.int32),
                            globs=np.zeros((games, teams, len(rf.GLOBAL_FIELDS)), np.float32),
                            features=np.zeros((games, teams, feature_cap), rf.FEATURE_DTYPE), feature_counts=np.zeros((games, teams), np.int32))
    recorded = {0: [1, 2], 1: [3, 0, 2]}
    writer = zkrecord.Writer(games, threads=2, teams=teams)
    for g, kept in recorded.items():
        writer.begin(g, map_w, map_h, y_range, sites, teams=kept)
    given = {(g, t): ([], []) for g in range(games) for t in range(teams)}
    for k in range(frames):
        for g in range(games):
            for t in range(teams):
                f = _hand_made_frame(rng, t, t % 2, k, map_w, map_h)
                batch.records[g, t, :len(f.units)], batch.counts[g, t] = f.units, len(f.units)
                batch.features[g, t, :len(f.features)], batch.feature_counts[g, t] = f.features, len(f.features)
                batch.globs[g, t] = f.glob
                given[(g, t)][0].append(rf.Frame(f.units, sites, f.glob, f.features))
        writer.frames(batch, np.arange(games))
        parts = []
        for g in range(games):
            for t in range(teams):
                o = np.zeros(int(rng.integers(0, 4)), zksim.ORDER_DTYPE)
                o["unit"], o["kind"], o["target"] = rng.integers(0, 40, len(o)), rng.integers(0, 20, len(o)), rng.integers(-1, 99, len(o))
                o["x"], o["z"] = rng.uniform(0.0, map_w, len(o)), rng.uniform(0.0, map_h, len(o))
                given[(g, t)][1].append(o)
                parts.append(o)
        writer.orders(np.arange(games), np.concatenate(parts), np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int32))
    for g, kept in recorded.items():
        writer.finish(g, tmp_path / f"game-{g}.zkr", {"teams": kept})
    writer.wait()
    assert writer.clamps() == dict.fromkeys(zkrecord.CLAMP_NAMES, 0)
    y_step = (y_range[1] - y_range[0]) / 65535 * 0.5 + 1e-3
    for g, kept in recorded.items():
        path = tmp_path / f"game-{g}.zkr"
        assert zkrecord.header(path) == (frames, tuple(kept))
        meta, sides = zkrecord.read(path)
        assert meta == {"teams": kept} and len(sides) == len(kept)
        for (got, got_orders), t in zip(sides, kept):
            want, want_orders = given[(g, t)]
            assert len(got) == len(got_orders) == frames
            for a, b, o, p in zip(got, want, got_orders, want_orders):
                _as_given(a, b, map_w, map_h, y_step)
                assert np.array_equal(o, p)
        last = len(kept) - 1
        _, part = zkrecord.read(path, side=last, until=65)
        assert all(p == ([], []) for p in part[:last]) and len(part[last][0]) == frames       # past the first chunk: both
        _, part = zkrecord.read(path, side=last, until=64)
        assert len(part[last][0]) == 64 and all(a.units.tobytes() == b.units.tobytes() for a, b in zip(part[last][0], sides[last][0]))
    with pytest.raises(ValueError):
        writer.begin(0, map_w, map_h, y_range, sites, teams=[4])


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


def test_a_dataset_indexed_before_any_number_of_sides_still_lists_two_sides_a_file_and_their_z(tmp_path):
    """sides.npz and z.npz as data version 6 wrote them (army-6's): no count of sides, and z a row a file of its two."""
    import argparse
    _record(tmp_path, 1)
    files = sorted(tmp_path.glob("game-*.zkr"))
    np.savez(tmp_path / "sides.npz", names=np.array([p.name for p in files]), frames=np.full(len(files), LIMIT, np.int64))
    cache = (tmp_path / "sides.npz").read_bytes()
    assert [(Path(p).name, s, n) for p, s, n in cd.sides(tmp_path)] == [(p.name, s, LIMIT) for p in files for s in (0, 1)]
    assert (tmp_path / "sides.npz").read_bytes() == cache, "the same files: the cache stays as it was"
    cd.zindex(argparse.Namespace(data=str(tmp_path), workers=1))
    z = cd.load_z(tmp_path)
    assert sorted(z) == sorted((p.name, s) for p in files for s in (0, 1))
    np.savez(tmp_path / "z.npz", names=np.array([p.name for p in files]), z=np.stack([[z[(p.name, s)][0] for s in (0, 1)] for p in files]),
             result=np.array([[z[(p.name, s)][1] for s in (0, 1)] for p in files], np.float32))
    old = cd.load_z(tmp_path)
    assert old.keys() == z.keys() and all(np.array_equal(old[k][0], z[k][0]) and old[k][1] == z[k][1] for k in z)


def test_z_is_the_knobs_and_the_counts_each_with_its_flag():
    import policy_variants as pv
    assert cd.Z_COUNTS.stop == cd.Z_DIM
    standard, rush = cd.variant_z(pv.STANDARD), cd.variant_z(pv.PRESETS[0])
    assert standard[cd.Z_KNOWN] == 1.0 and not standard[cd.Z_KNOBS].any() and not standard[cd.Z_PRESET].any()
    assert rush[cd.Z_PRESET].tolist() == [1.0, 0.0, 0.0] and rush[cd.Z_KNOBS].any()
    assert not cd.variant_z(None).any()
    assert np.allclose(cd.variant_z(pv.PRESETS[1]), cd.variant_z(pv.PRESETS[1].as_dict()))


def test_z_dropout_forgets_all_of_z_or_only_its_counts():
    import torch
    import policy_clone_train as ct
    z = np.ones(cd.Z_DIM, np.float32)
    out = ct.z_tensor([{"z": z} for _ in range(4000)] + [{"z": None}], "cpu", np.random.default_rng(0), 0.2).numpy()
    assert not out[-1].any()
    out = out[:-1]
    gone = ~out.any(1)
    counts_gone = out[:, cd.Z_KNOWN].astype(bool) & ~out[:, cd.Z_COUNTED].astype(bool)
    assert abs(gone.mean() - 0.2) < 0.03 and abs(counts_gone.mean() - 0.2) < 0.03
    assert not out[counts_gone][:, cd.Z_COUNTS].any() and out[counts_gone][:, cd.Z_KNOBS].all()
    assert torch.equal(ct.z_tensor([{"z": z}], "cpu")[0], torch.as_tensor(z))


def test_economy_actions_weigh_more_and_done_does_not():
    import torch
    import policy_clone_train as ct
    import policy_model as pm
    kind = torch.tensor([[rf.BUILD, rf.FIGHT, rf.ASSIST, 0]])
    anchor = torch.tensor([[0, 1, 2, -1]])
    a = pm.Actions(anchor, kind, *(torch.full_like(anchor, -1) for _ in range(8)), logp=torch.zeros(1, 4))
    assert ct.action_weights(a, 2.5).tolist() == [[2.5, 1.0, 2.5, 1.0]]
    assert ct.action_weights(a, 1.0).tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_hl_gauss_targets_centre_on_the_return_and_a_new_critic_values_zero():
    import torch
    import policy_model as pm
    critic = pm.Critic(tables.table().names.__len__() + 1, d=16, layers=1, heads=2, bins=51)
    target = torch.tensor([-1.0, 0.0, 0.7])
    loss = critic.hl_gauss(torch.zeros(3, 51), target)
    assert torch.allclose(loss, loss[0].expand(3), atol=1e-4)         # uniform logits: the same for any target inside
    best = torch.log(torch.special.ndtr((torch.linspace(-3, 3, 51) + 0.06 - 0.7) / 0.09) - torch.special.ndtr((torch.linspace(-3, 3, 51) - 0.06 - 0.7) / 0.09) + 1e-12)
    assert critic.hl_gauss(best[None], target[2:]) < critic.hl_gauss(best[None], target[:1])
    mean = (best.softmax(-1) * critic.centres).sum()
    assert abs(float(mean) - 0.7) < 0.02


def test_a_free_for_all_records_every_team_and_lists_a_side_only_as_far_as_its_alliance_played(tmp_path):
    """An alliance out while the game goes on: its side is in the file to the end, as the others, but its meta says how far it
    played (`until`), which `sides` lists and z counts up to; its result is its loss, and the two in at the limit share it."""
    from test_sim_env import _ffa_maps, _held
    env = se.SimEnv(2, 3, ["scripts"], threads=2, maps=_ffa_maps(3), kinds={"ffa": 1.0}, teams=3, variants=True)
    keep = np.ones(3, np.float32)
    _held(env, np.full(3, 5000.0, np.float32), keep)
    env.time_limit = 20
    env.begin()
    rec = cd.Recorder(env, tmp_path)
    for step in range(1, 21):
        if step == 8:
            keep[2] = 0.0       # game 0's third alliance is out after its 8th frame
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    files = sorted(tmp_path.glob("game-*.zkr"))
    metas = [cd._meta(p) for p in files]
    out = next(i for i, m in enumerate(metas) if m["until"] == [20, 20, 8])
    meta = metas[out]
    assert meta["kind"] == "ffa" and meta["teams"] == meta["allies"] == [0, 1, 2] and meta["result"] == [0.25, 0.25, -0.5]
    assert meta["how"] == ["limit", "limit", "eliminated"] and zkrecord.header(files[out]) == (20, (0, 1, 2))
    assert [(s, n) for p, s, n in cd.sides(tmp_path) if p == str(files[out])] == [(0, 20), (1, 20), (2, 8)]
    assert len(cd.read_game(files[out])) == 3 and cd.read_game(files[out])[2][2]["until"] == 8
    assert rec.decisions == sum(sum(m["until"]) for m in metas)
    again = cd.sides(tmp_path)          # from sides.npz now
    assert again == cd.sides(tmp_path) and (str(files[out]), 2, 8) in again


def test_a_chicken_game_records_its_players_sides_alone_with_its_kind_and_difficulty(tmp_path):
    env = se.SimEnv(2, 5, ["scripts"], threads=2, kinds={"chicken": 1.0}, difficulties={"Normal": 1.0}, variants=True)
    env.time_limit = 10
    env.begin()
    rec = cd.Recorder(env, tmp_path)
    for _ in range(9):
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    files = sorted(tmp_path.glob("game-*.zkr"))
    assert len(files) == 2
    for path in files:
        meta = cd._meta(path)
        n = len(meta["teams"])
        assert meta["kind"] == "chicken" and meta["difficulty"] == "Normal" and meta["teams"] == list(range(n)) and meta["allies"] == [0] * n
        assert meta["result"] == [0.0] * n and meta["how"] == ["limit"] * n and len(meta["variants"]) == n
        assert zkrecord.header(path) == (10, tuple(range(n))) and cd.side_info(meta, n - 1)["difficulty"] == "Normal"
