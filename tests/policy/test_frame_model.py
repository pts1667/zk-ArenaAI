"""The frame encoder (tools/frame_model.py): its guarantees, its probes, and whether the frames it learns from are ones that could happen."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import frame_data as fd
    import frame_model as fm
    import policy_model as pm
    import policy_sim_env as se
    from zkbot.policy import featurize as fz, rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

SMALL = fm.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=16)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    fd.collect(out, script_frames=300, scenario_frames=300, games=24, seed=5, threads=4, script_every=6, frames_per_shard=120, army_frames=300)
    return out


def _frames(path: pathlib.Path, count: int) -> list[dict]:
    s = fm.Shards._read(path)
    return [fm.Shards.frame(s, i) for i in range(count)]


def _encoder(seed: int = 1) -> fm.FrameEncoder:
    torch.manual_seed(seed)
    return fm.FrameEncoder(SMALL, 700).eval()


# The encoder -------------------------------------------------------------------------------------------------------------------------------------------


def test_order_of_tokens_does_not_matter(data):
    encoder = _encoder()
    frame = next(f for f in _frames(sorted(data.glob("scenario-*.npz"))[0], 100) if len(f["unit_x"]) > 3)
    order = np.random.default_rng(1).permutation(len(frame["unit_x"]))
    shuffled = dict(frame)
    for k in ("unit_x", "unit_name", "unit_ally", "unit_player", "tok_kind"):
        shuffled[k] = frame[k][order]
    with torch.no_grad():
        a, b = encoder(fm.collate_frames([frame])), encoder(fm.collate_frames([shuffled]))
    assert torch.allclose(a, b, atol=1e-5)


def test_padding_does_not_matter(data):
    encoder = _encoder()
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 8)
    longest = max(frames, key=lambda f: len(f["unit_x"]))
    shortest = min(frames, key=lambda f: len(f["unit_x"]))
    assert len(longest["unit_x"]) > len(shortest["unit_x"])
    with torch.no_grad():
        alone, together = encoder(fm.collate_frames([shortest])), encoder(fm.collate_frames([shortest, longest]))[0]
    assert torch.allclose(alone[0], together, atol=1e-5)


def test_token_self_attention_keeps_the_same_guarantees(data):
    torch.manual_seed(1)
    encoder = fm.FrameEncoder(fm.EncoderConfig(d=32, heads=2, queries=4, rounds=1, token_layers=1, z=16), 700).eval()
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 8)
    shortest = min(frames, key=lambda f: len(f["unit_x"]))
    longest = max(frames, key=lambda f: len(f["unit_x"]))
    with torch.no_grad():
        assert torch.allclose(encoder(fm.collate_frames([shortest]))[0], encoder(fm.collate_frames([shortest, longest]))[0], atol=1e-5)


def test_it_reads_the_main_models_batches_with_the_same_answer():
    """So the main model can call it on what it already has."""
    env = se.SimEnv(2, 3, ["scripts"], threads=2)
    env.begin()
    env.step(0.0)
    tensors = [fz.featurize(env.frame(s), env.bundle_of(s), env.table, economy=True) for s in (0, 1)]
    frames = [{"unit_x": t.unit_x, "unit_name": t.unit_name, "unit_ally": t.unit_ally, "unit_player": t.unit_player, "tok_kind": t.tok_kind[:len(t.unit_x)], "reg_x": t.reg_x, "site_x": t.site_x,
               "feat_x": t.feat_x, "glob_x": t.glob_x} for t in tensors]
    encoder = _encoder()
    with torch.no_grad():
        ours = encoder(fm.collate_frames(frames))
        theirs = encoder(pm.collate(tensors).to("cpu"))
    assert torch.allclose(ours, theirs, atol=1e-5)


# Probes -------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_probe_finds_what_is_in_a_latent_and_not_what_isnt():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(1200, 20))
    y = np.stack((z[:, 3] - 2.0 * z[:, 7] + 0.1 * rng.normal(size=1200), rng.normal(size=1200)), 1)
    r2 = fm.ridge_r2(z[:800], y[:800], z[800:], y[800:])
    assert r2[0] > 0.95
    assert r2[1] < 0.1


def test_a_probe_says_nothing_of_a_column_that_does_not_vary():
    """Say, ships on maps with no water.  The score would be huge and mean nothing."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(1000, 4))
    y = np.column_stack((z[:, 0], rng.normal(size=1000)))
    y[800:, 1] = 0.0
    r2 = fm.ridge_r2(z[:800], y[:800], z[800:], y[800:])
    assert r2[0] > 0.95 and np.isnan(r2[1])
    sample = {"z": z, "probes": y, "map": np.where(np.arange(1000) < 800, 0, 1), "game": np.arange(1000), "seat": np.zeros(1000, np.int32), "source": np.zeros(1000, np.int32)}
    report = fm.probe_report(sample, ("a", "b"), None, [1], {"both": [0, 1]})
    assert "r2_a" in report and "r2_b" not in report and "r2_both_b" not in report


def test_collapse_report_tells_a_collapse_from_a_spread():
    rng = torch.Generator().manual_seed(1)
    spread = fm.collapse_report(torch.randn(500, 32, generator=rng))
    one_way = fm.collapse_report(torch.randn(500, 1, generator=rng).repeat(1, 32))
    same = fm.collapse_report(torch.ones(500, 32))
    assert spread["effective_rank"] > 25.0
    assert one_way["effective_rank"] < 2.0
    assert same["std_mean"] == 0.0


def test_probes_are_taken_on_a_map_they_did_not_fit():
    rng = np.random.default_rng(1)
    n = 600
    sample = {"z": rng.normal(size=(n, 8)), "map": np.repeat(np.arange(3), n // 3), "source": np.zeros(n, np.int32), "seat": np.tile([0, 1], n // 2), "game": np.arange(n, dtype=np.int64)}
    sample["probes"] = sample["z"][:, :2].astype(np.float32)
    report = fm.probe_report(sample, ("a", "b"), None, held_out=2)
    assert report["r2_a"] > 0.99 and report["r2_b"] > 0.99
    assert fm.probe_report(sample, ("a", "b"), None, held_out=9) == {}


def test_the_outcome_probe_takes_the_seat_into_account():
    """A frame of side 1 says the opposite of the game's result, and an unfinished game or a scenario says nothing."""
    rng = np.random.default_rng(2)
    n = 1200
    games = np.repeat(np.arange(n // 2, dtype=np.int64), 2)
    seat = np.tile([0, 1], n // 2)
    result = rng.choice([-1.0, 1.0], n // 2).astype(np.float32)
    from_seat = result[games] * np.where(seat == 0, 1.0, -1.0)
    z = rng.normal(size=(n, 4))
    z[:, 0] = from_seat
    sample = {"z": z, "map": (games % 3).astype(np.int32), "source": np.zeros(n, np.int32), "seat": seat, "game": games, "probes": z[:, :1].astype(np.float32)}
    report = fm.probe_report(sample, ("a",), (np.arange(n // 2, dtype=np.int64), result), held_out=2)
    assert report["r2_outcome"] > 0.99
    sample["source"] = np.ones(n, np.int32)
    assert "r2_outcome" not in fm.probe_report(sample, ("a",), (np.arange(n // 2, dtype=np.int64), result), held_out=2)


# Scenarios: could this have happened? ------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    return fd.ScenarioEnv(2, 11, threads=2)


@pytest.mark.parametrize("slot", range(len(se.TRAINING_MAPS)))
def test_everything_in_a_scenario_stands_where_it_can(scenarios, slot):
    world_map, bundle, facts = se.pool()[slot]
    ground = fd.Ground(bundle, facts)
    table, roster = scenarios.table, scenarios.roster
    rng = np.random.default_rng(slot)
    box = facts.layouts[0]
    own, enemy = box[0].point, box[1].point
    placed = 0
    for _ in range(25):
        side = fd.draw_side(rng, scenarios.rules, roster, ground, own, enemy)
        placed += len(side.rows)
        assert len(side.rows) <= scenarios.rules.most_units
        mobile = table.mobile[side.rows]
        walkers = mobile & ~table.air[side.rows]
        assert fd._on_passable(bundle, side.x[walkers], side.z[walkers], table.move_class[side.rows[walkers]].astype(np.int32)).all()
        spots = bundle.spot_list[:, :2]
        for r, x, z in zip(side.rows[~mobile], side.x[~mobile], side.z[~mobile]):
            if roster.is_extractor[r]:
                assert np.hypot(spots[:, 0] - x, spots[:, 1] - z).min() < 1.0
            else:
                assert ground.fits(float(x), float(z), roster.footing[int(r)])
        assert ((side.hp > 0.0) & (side.hp <= 1.0)).all()
        assert (side.build[mobile] == 1.0).all()
        assert (0.0 <= side.x).all() and (side.x <= bundle.map_w).all() and (0.0 <= side.z).all() and (side.z <= bundle.map_h).all()
    assert placed > 25


def test_structures_never_stand_on_each_other(scenarios):
    _, bundle, facts = se.pool()[0]
    roster, table = scenarios.roster, scenarios.table
    rows = np.tile(roster.structures[table.cost[roster.structures] < 300.0][:6], 6)
    rows = np.concatenate((rows, np.full(5, roster.extractor)))
    start = tuple(float(v) for v in bundle.spot_list[0, :2])
    built, x, z = fd._place_structures(np.random.default_rng(1), fd.Ground(bundle, facts), roster, rows, start, (start[0] + 2000.0, start[1]))
    # Footprints are rectangles: apart along x or along z.
    half = roster.half[built]
    along_x = np.abs(x[:, None] - x[None]) - half[:, None, 0] - half[None, :, 0]
    along_z = np.abs(z[:, None] - z[None]) - half[:, None, 1] - half[None, :, 1]
    d = np.maximum(along_x, along_z)
    np.fill_diagonal(d, 1.0)
    assert (d >= -1e-3).all()
    assert len(built) >= 5
    # Nor on what a team drawn before has built.
    placed = fd.footprints(x, z, half[:, 0], half[:, 1])
    more, mx, mz = fd._place_structures(np.random.default_rng(2), fd.Ground(bundle, facts), roster, rows, start, (start[0] + 2000.0, start[1]), placed)
    for r, px, pz in zip(more, mx, mz):
        assert fd.apart(placed, float(px), float(pz), *roster.half[r])
    assert len(more) >= 5


def test_a_scenarios_budget_holds(scenarios):
    _, bundle, facts = se.pool()[1]
    rules = fd.ScenarioRules(side_value=(500.0, 500.0), structure_share=(0.0, 0.0), extractors=(0, 0))
    box = facts.layouts[0]
    rng = np.random.default_rng(3)
    for _ in range(20):
        side = fd.draw_side(rng, rules, scenarios.roster, fd.Ground(bundle, facts), box[0].point, box[1].point)
        assert scenarios.table.cost[side.rows].sum() <= 500.0 + 1.0


def test_only_what_the_catalog_offers_is_on_the_roster(scenarios):
    roster, table = scenarios.roster, scenarios.table
    names = [table.names[r] for r in roster.mobile]
    assert not [n for n in names if n.startswith("chicken")]
    assert all(table.mobile[r] for r in roster.mobile)
    assert not any(table.mobile[r] for r in roster.structures)
    assert "staticmex" in [table.names[r] for r in roster.structures]
    for name in ("cloakraid", "tankassault", "shieldcon"):
        assert name in names


def test_scenarios_play_on_every_map_and_change_as_they_go():
    env = fd.ScenarioEnv(32, 21, threads=4)
    env.begin()
    maps = {id(b) for b in env.bundle}
    assert len(maps) == len(se.pool())
    seats = [g * env.slots + t for g in range(4) for t in range(2)]
    before = [env.frame(s).units[["uid", "x"]].copy() for s in seats]
    for _ in range(8):
        env.step(0.0)
        env.finished_seats()
    # By uid, as what's on the board changes too.
    moved = []
    for s, was in zip(seats, before):
        now = env.frame(s).units
        moved += [not np.array_equal(now["x"][now["uid"] == uid], was["x"][was["uid"] == uid]) for uid in np.intersect1d(was["uid"], now["uid"])]
    assert any(moved)
    assert (env.batch.counts >= 0).all()
    assert env.frame(0).glob[rf.G["frame"]] >= env.batch.frame(0, 0).glob[rf.G["frame"]]


def _own(env, g: int) -> list[np.ndarray]:
    return [units for units, _ in env._team_units(g)]


def test_scenarios_are_1v1s_team_games_and_chicken_games_within_what_a_frame_holds():
    env = fd.ScenarioEnv(48, 21, threads=4)
    env.begin()
    assert set(env.kind) == {"1v1", "teams", "chicken"} and env.slots == 8        # the training maps have two boxes: no free-for-all
    table, burrow = env.table, se.catalog().rules["chickens"]["burrow"]
    is_chicken = np.array([name.startswith("chicken") or name == burrow for name in table.names] + [False])
    seen = {"burrows": 0, "armies": 0, "allies": 0}
    for g in range(env.count):
        n, c, own = int(env.n_teams[g]), int(env.chickens[g]), _own(env, g)
        assert sum(len(u) for u in own) <= fz.CAP_UNITS and all(len(u) <= fz.CAP_CONTROLLABLE for u in own)
        assert len(env.scripts[g]) == len(env.bases[g]) == len(env.scouted[g]) == n
        for t in range(n):
            assert is_chicken[table.rows(own[t]["def_id"])].all() == (t == c) or not len(own[t])
            assert (env.scripts[g][t] is None) == (env.bases[g][t] is None) == (t == c)
            assert env.ally[g, env.facing[g, t]] != env.ally[g, t]
        if env.kind[g] == "teams":
            assert n in (4, 6, 8) and env.ally[g, :n].tolist() == [0] * (n // 2) + [1] * (n // 2) and c < 0
            seen["allies"] += 1
        elif env.kind[g] == "chicken":
            assert c == n - 1 and 2 <= n <= 5 and env.ally[g, :n].tolist() == [0] * c + [1]
            assert env.batch.chicken_state(g)["team"] == c and [env.plays(g, t) for t in range(n)] == ["script"] * c + ["chickens"]
            names = [table.names[r] for r in table.rows(own[c]["def_id"])]
            seen["burrows"] += burrow in names
            seen["armies"] += any(table.mobile[r] for r in table.rows(own[c]["def_id"]))
            # A burrow is no nearer a player's start than the spawner would put one.
            near = se.catalog().rules["chickens"]["difficulties"][int(env.difficulty[g])]["min_base_distance"]
            starts = env.starts[g, :2 * c].reshape(c, 2)
            for u in own[c][[name == burrow for name in names]]:
                assert np.hypot(starts[:, 0] - u["x"], starts[:, 1] - u["z"]).min() >= near - 1.0
        else:
            assert n == 2 and c < 0
    assert seen["allies"] and seen["burrows"] >= 3 and seen["armies"] >= 3, seen
    # They play: the sim sends the chickens at the players, as in a chicken game, and every player's scripts give orders.
    games = [g for g in range(env.count) if env.kind[g] == "chicken"]
    before = {g: _own(env, g)[int(env.chickens[g])][["uid", "x", "z"]].copy() for g in games}
    ordered = set()
    for _ in range(12):
        env.step(0.0)
        ordered |= {seat for seat, orders in env.script_record.items() if len(orders)}
    moved = 0
    for g in games:
        now = _own(env, g)[int(env.chickens[g])]
        both, a, b = np.intersect1d(before[g]["uid"], now["uid"], return_indices=True)
        moved += int((np.hypot(before[g]["x"][a] - now["x"][b], before[g]["z"][a] - now["z"][b]) > 16.0).sum())
    assert moved > 10
    assert {s % env.slots for s in ordered} >= {0, 1, 2, 3}, "allies past the first two have scripts too"


def test_a_free_for_all_scenario_has_an_alliance_a_box_and_every_seat_makes_a_frame():
    from test_sim_env import _ffa_maps
    env = fd.ScenarioEnv(6, 4, fd.ScenarioRules(shapes=(("ffa", 1.0),)), threads=2, maps=_ffa_maps(4))
    env.begin()
    is_extractor = fd.extractors(env.table)
    for _ in range(6):
        env.step(0.0)
    for g in range(env.count):
        n = int(env.n_teams[g])
        assert env.kind[g] == "ffa" and n in (3, 4) and env.ally[g, :n].tolist() == list(range(n)) and env.n_allies[g] == n
        own = _own(env, g)
        assert sum(len(u) > 0 for u in own) >= n - 1
        for t in range(n):
            seat = g * env.slots + t
            tensors = fz.featurize(fd.with_site_status(env.frame(seat), env.table, is_extractor), env.bundle_of(seat), env.table, economy=True)
            assert len(set(tensors.unit_ally.tolist())) <= n and np.isfinite(tensors.unit_x).all()
    # Only what the maps hold and the catalog has: no free-for-all on the training maps, which have two boxes.
    assert set(fd.scenario_kinds(fd.ScenarioRules(), se.pool(), se.catalog())) == {"1v1", "teams", "chicken"}
    assert fd.scenario_kinds(fd.ScenarioRules(shapes=(("ffa", 1.0),)), se.pool(), se.catalog()) == {"1v1": 1.0}


def test_no_game_frames_are_collected_from_ends_on_a_lead(tmp_path, monkeypatch):
    """The late game is what the encoder would otherwise hardly see (the user, 2026-11-10)."""
    envs = []
    monkeypatch.setattr(fd.Collector, "run", lambda self, frames: envs.append(self.env))
    for source in fd.SOURCES:
        fd._collect_part(fd.Part(tmp_path, source, 1, 2, 3, 2, 6, fd.ScenarioRules(), 60), se.pool())
    assert [env.lead_share for env in envs] == [0.0, 0.0, 0.0]
    for env in envs:
        env.begin()
        assert not env.lead_rule.any() or env.early_end > env.time_limit
    assert se.SimEnv(1, 3, ["scripts"], threads=1).lead_share == 1.0, "everywhere else it's on"


def test_extractors_placed_at_the_start_earn():
    """Or frames would show a base and the income of having none."""
    def income(extractors: tuple[int, int]) -> float:
        env = fd.ScenarioEnv(32, 3, fd.ScenarioRules(extractors=extractors, side_value=(2000.0, 2000.0), structure_share=(0.0, 0.0), decisions=200), threads=2)
        env.begin()
        for _ in range(4):
            env.step(0.0)
            env.finished_seats()
        return float(np.mean([env.batch.globs[g, s, rf.G["metal_income"]] for g in range(32) for s in (0, 1)]))

    assert income((8, 8)) > income((0, 0)) + 4.0


def test_a_scenario_ends_after_its_time(scenarios):
    env = fd.ScenarioEnv(4, 9, fd.ScenarioRules(decisions=6), threads=2)
    env.begin()
    for _ in range(6):
        env.step(0.0)
        if env._replace:
            assert (env.decisions[env._replace] <= 6).all()
        env.finished_seats()
    assert (env.decisions < 6).all()    # all have started afresh by now


# Site status and probes ------------------------------------------------------------------------------------------------------------------------------------


def test_site_status_says_who_holds_a_spot_as_far_as_a_seat_knows():
    env = se.SimEnv(1, 4, ["scripts"], threads=1)
    env.begin()
    frame = env.frame(0)
    is_extractor = fd.extractors(env.table)
    row = int(np.flatnonzero(is_extractor)[0])
    mex_def = next(int(k) for k in range(len(env.table.row_of_def) - 1) if env.table.row_of_def[k] == row)
    units = frame.units[:1].repeat(3)
    sx, sz = frame.sites["x"], frame.sites["z"]
    ally = int(frame.glob[rf.G["ally_team"]])
    for i, (site, team, flags, knowledge) in enumerate(((0, ally, 0, rf.KNOW_LOS), (1, ally + 1, 0, rf.KNOW_LOS), (2, ally, rf.BEING_BUILT, rf.KNOW_LOS))):
        units[i]["def_id"], units[i]["x"], units[i]["z"], units[i]["ally_team"], units[i]["flags"], units[i]["knowledge"] = mex_def, sx[site], sz[site], team, flags, knowledge
    status = fd.with_site_status(rf.Frame(units, frame.sites, frame.glob, frame.features), env.table, is_extractor).sites["status"]
    assert status[0] == rf.SITE_OURS and status[1] == rf.SITE_ENEMY and status[2] == rf.SITE_BUILDING
    assert frame.sites["status"].max() == rf.SITE_FREE      # the batch's own is left alone
    units[1]["knowledge"] = rf.KNOW_HIDDEN
    status = fd.with_site_status(rf.Frame(units, frame.sites, frame.glob, frame.features), env.table, is_extractor).sites["status"]
    assert status[1] == rf.SITE_FREE                        # what a seat can't see, it doesn't know


def test_probes_count_what_is_on_the_board():
    env = se.SimEnv(1, 6, ["scripts"], threads=1)
    env.begin()
    frame, bundle = env.frame(0), env.bundle_of(0)
    probes = dict(zip(fd.PROBES, fd.probe_targets(frame, bundle, env.table)))
    assert probes["commander_hp"] == pytest.approx(1.0)
    assert probes["own_value"] > 0.0
    x = frame.units["x"][frame.units["player"] == 0].mean() / bundle.map_w * 2.0 - 1.0
    assert probes["own_x"] == pytest.approx(x, abs=1e-5)


# Shards -------------------------------------------------------------------------------------------------------------------------------------------------------


def test_collected_data_has_every_source_and_good_links(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    assert manifest["maps"] == list(se.TRAINING_MAPS)
    assert sorted(manifest["frames_per_second"]) == ["army", "scenario", "script"]
    for source, name in enumerate(fd.SOURCES):
        paths = sorted(data.glob(f"{name}-*.npz"))
        assert len(paths) >= 2
        for path in paths:
            s = fm.Shards._read(path)
            assert (s["source"] == source).all()
            for group, dim in fd.GROUPS:
                assert s[f"{group}_x"].shape[1] == dim
                assert s[f"{group}_at"][-1] == len(s[f"{group}_x"])
                assert np.isfinite(s[f"{group}_x"]).all()
            n = len(s["map"])
            assert s["glob_x"].shape == (n, fz.GLOBAL_DIM) and s["probes"].shape == (n, len(fd.PROBES))
            linked = np.flatnonzero(s["later"] >= 0)
            assert len(linked) or n < 120      # the last shard can be too short to have any
            at = s["later"][linked]
            assert (s["game"][at] == s["game"][linked]).all() and (s["seat"][at] == s["seat"][linked]).all()
            assert (s["decision"][at] > s["decision"][linked]).all()
    # Scenarios are of several kinds of game, and a frame is a player's: never the chickens', who are a chicken game's last team.
    scenario = [fm.Shards._read(p) for p in sorted(data.glob("scenario-*.npz"))]
    kind, seat, teams = (np.concatenate([s[k] for s in scenario]) for k in ("kind", "seat", "teams"))
    chicken = kind == fd.GAME_KINDS.index("chicken")
    assert {fd.GAME_KINDS[k] for k in set(kind.tolist())} == {"1v1", "teams", "chicken"}     # the training maps have two boxes: no free-for-all
    assert (seat < teams - chicken).all() and {0, 1} <= set(seat.tolist()) and seat.max() >= 2


def test_a_later_frame_cant_coincide_with_the_next_one():
    with pytest.raises(ValueError):
        fd.Capture(every=4, later=4)


def test_script_and_scenario_games_are_numbered_apart(data):
    script = np.concatenate([fm.Shards._read(p)["game"] for p in data.glob("script-*.npz")])
    scenario = np.concatenate([fm.Shards._read(p)["game"] for p in data.glob("scenario-*.npz")])
    assert not set(script.tolist()) & set(scenario.tolist())


def test_16_bit_floats_lose_little_of_a_frame():
    """What the shards store against what the featurizer gave."""
    env = se.SimEnv(4, 8, ["scripts"], threads=2)
    env.begin()
    for _ in range(10):
        env.step(0.0)
    t = fz.featurize(env.frame(0), env.bundle_of(0), env.table, economy=True)
    for x in (t.unit_x, t.reg_x, t.site_x, t.glob_x):
        assert np.abs(x.astype(np.float16).astype(np.float32) - x).max() < 1e-2


def test_sampling_stays_off_the_held_out_map(data):
    rng = np.random.default_rng(1)
    shards = fm.Shards(sorted(data.glob("*-*.npz")), rng, in_memory=4, maps=(0, 1, 2, 3))
    for s in shards.loaded:
        assert (s["map"][s["anchors"]] != 4).all()
    now, later = shards.sample(8)
    assert now.unit_x.shape[0] == 8 and later.glob_x.shape[0] == 8


# Games of more than two -----------------------------------------------------------------------------------------------------------------------------------


def test_kinds_parse_as_policy_train_spells_them_and_army_games_leave_out_what_simenv_lacks(capsys):
    kinds = fd.parse_kinds(fd.KINDS)
    assert kinds == {"1v1": 0.5, "teams": 0.2, "ffa": 0.2, "chicken": 0.1}
    assert fd.parse_kinds("teams ffa:0") == {"teams": 1.0}
    with pytest.raises(ValueError):
        fd.parse_kinds(["duel:1"])
    played = fd.army_kinds(kinds)
    assert set(played) == set(kinds) & set(se.GAME_KINDS) and fd.army_kinds(None) is None
    if "chicken" not in se.GAME_KINDS:
        assert "chicken" in capsys.readouterr().err


def test_army_team_games_give_frames_of_every_seat_with_each_teams_own_outcome(tmp_path):
    facts = fd.collect(tmp_path, 0, 0, games=4, seed=3, threads=2, script_every=6, frames_per_shard=400, army_frames=400, kinds={"teams": 1.0})
    assert facts["army_kinds"] == {"teams": 1.0} and facts["game_kinds"][0] == "1v1"
    s = fm.Shards._read(sorted(tmp_path.glob("army-*.npz"))[0])
    teams = int(s["teams"][0])
    assert teams >= 4 and (s["kind"] == fd.GAME_KINDS.index("teams")).all() and (s["alliances"] == 2).all()
    assert (s["seat"] < s["teams"]).all() and len(set(s["seat"].tolist())) > 2
    fd.write_manifest(tmp_path, {7: np.array([1.0, -1.0], np.float32), 3: np.array([-0.5, 1.0, -0.5], np.float32)}, {})
    with np.load(tmp_path / "outcomes.npz") as ends:
        assert ends["game"].tolist() == [3, 7] and ends["result"].tolist() == [-0.5, 1.0]
        assert ends["results"][0].tolist() == [-0.5, 1.0, -0.5] and ends["results"][1, :2].tolist() == [1.0, -1.0] and np.isnan(ends["results"][1, 2])


def test_the_outcome_probe_reads_each_teams_own_result():
    """With several teams a seat's result isn't team 0's turned round: a free-for-all of three gives 1, -1/2, -1/2."""
    rng = np.random.default_rng(3)
    games = np.repeat(np.arange(400, dtype=np.int64), 3)
    seat = np.tile([0, 1, 2], 400)
    results = np.full((400, 3), -0.5, np.float32)
    results[np.arange(400), rng.integers(0, 3, 400)] = 1.0
    z = rng.normal(size=(len(games), 4))
    z[:, 0] = results[games, seat]
    sample = {"z": z, "map": (games % 3).astype(np.int32), "source": np.zeros(len(games), np.int32), "seat": seat, "game": games, "probes": z[:, :1].astype(np.float32)}
    assert fm.probe_report(sample, ("a",), (np.arange(400, dtype=np.int64), results), held_out=2)["r2_outcome"] > 0.99


def test_a_scenarios_clock_can_claim_up_to_a_whole_game():
    assert fd.ScenarioRules().longest_clock == se.TIME_LIMIT_DECISIONS / 2.0 == 4800.0
