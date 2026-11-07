"""The frame model: its loss, its encoder, and whether the frames it learns from are ones that could happen."""
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
    import frame_jepa as fj
    import policy_model as pm
    import policy_sim_env as se
    from zkbot.policy import featurize as fz, rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

SMALL = fj.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=16, projector=32, projected=8)
LOSS = fj.LossConfig(slices=64)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    fd.collect(out, script_frames=300, scenario_frames=300, games=24, seed=5, threads=4, script_every=6, frames_per_shard=120)
    return out


def _frames(path: pathlib.Path, count: int) -> list[dict]:
    s = fj.Shards._read(path)
    return [fj.Shards.frame(s, i) for i in range(count)]


def _encoder(seed: int = 1) -> fj.FrameEncoder:
    torch.manual_seed(seed)
    return fj.FrameEncoder(SMALL, 700).eval()


# SIGReg --------------------------------------------------------------------------------------------------------------------------------------------


def _reg(x: torch.Tensor) -> float:
    return float(fj.sigreg(x, LOSS, torch.Generator().manual_seed(1)))


def test_sigreg_is_small_for_a_standard_normal_and_large_for_anything_else():
    g = torch.Generator().manual_seed(2)
    normal = torch.randn(2000, 32, generator=g)
    good = _reg(normal)
    assert 0.5 < good < 2.0                                     # its expectation for a true normal: the integral of (1 - phi^2) phi, about 1.05
    assert _reg(torch.zeros(2000, 32)) > 50.0 * good            # all the same: the collapse it's there to stop
    assert _reg(normal * 3.0) > 50.0 * good                     # too spread out
    assert _reg(normal * 0.2) > 50.0 * good                     # too close together
    assert _reg(normal + 2.0) > 50.0 * good                     # off centre
    low_rank = normal[:, :1].repeat(1, 32)                      # looks fine along any one axis, and all axes say one thing
    assert _reg(low_rank) > 10.0 * good


def test_epps_pulley_is_the_formula_for_a_known_case():
    # For all-zero samples, cos = 1 and sin = 0, so the statistic is N times the integral of (1 - phi)^2 phi by the trapezoid rule.
    t = np.linspace(0.0, 3.0, 17)
    phi = np.exp(-0.5 * t * t)
    w = np.full(17, 2.0 * (t[1] - t[0]))
    w[0] = w[-1] = t[1] - t[0]
    expected = 100.0 * ((1.0 - phi) ** 2 * w * phi).sum()
    assert float(fj.epps_pulley(torch.zeros(100, 1, dtype=torch.float64))[0]) == pytest.approx(expected)


def test_sigreg_passes_gradients_and_they_spread_a_collapsed_batch():
    x = (torch.randn(512, 16) * 0.01).requires_grad_()
    optimiser = torch.optim.Adam([x], lr=0.05)
    before = x.detach().std(0).mean()
    for _ in range(100):
        optimiser.zero_grad()
        fj.sigreg(x, LOSS).backward()
        optimiser.step()
    assert x.detach().std(0).mean() > 20.0 * before


# Invariance --------------------------------------------------------------------------------------------------------------------------------------------


def test_invariance_is_zero_when_views_agree_and_measures_against_the_global_views_only():
    same = torch.randn(1, 8, 4).repeat(5, 1, 1)
    assert float(fj.invariance(same, 2)) == 0.0
    views = torch.zeros(3, 1, 1)
    views[2] = 3.0
    assert float(fj.invariance(views, 2)) == pytest.approx(3.0)     # (0 + 0 + 9) / 3: the third is no part of the centre


def test_loss_weighs_its_two_parts():
    x = torch.randn(4, 32, 8)
    loss, parts = fj.lejepa_loss(x, 2, fj.LossConfig(weight=0.25, slices=16), torch.Generator().manual_seed(1))
    assert float(loss) == pytest.approx(0.75 * parts["invariance"] + 0.25 * parts["sigreg"], rel=1e-5)


# The encoder -------------------------------------------------------------------------------------------------------------------------------------------


def test_order_of_tokens_does_not_matter(data):
    encoder = _encoder()
    frame = next(f for f in _frames(sorted(data.glob("scenario-*.npz"))[0], 100) if len(f["unit_x"]) > 3)
    order = np.random.default_rng(1).permutation(len(frame["unit_x"]))
    shuffled = dict(frame)
    for k in ("unit_x", "unit_name", "unit_ally", "unit_player", "tok_kind"):
        shuffled[k] = frame[k][order]
    with torch.no_grad():
        a, b = encoder(fj.collate_frames([frame])), encoder(fj.collate_frames([shuffled]))
    assert torch.allclose(a, b, atol=1e-5)


def test_padding_does_not_matter(data):
    encoder = _encoder()
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 8)
    longest = max(frames, key=lambda f: len(f["unit_x"]))
    shortest = min(frames, key=lambda f: len(f["unit_x"]))
    assert len(longest["unit_x"]) > len(shortest["unit_x"])
    with torch.no_grad():
        alone, together = encoder(fj.collate_frames([shortest])), encoder(fj.collate_frames([shortest, longest]))[0]
    assert torch.allclose(alone[0], together, atol=1e-5)


def test_token_self_attention_keeps_the_same_guarantees(data):
    torch.manual_seed(1)
    encoder = fj.FrameEncoder(fj.EncoderConfig(d=32, heads=2, queries=4, rounds=1, token_layers=1, z=16), 700).eval()
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 8)
    shortest = min(frames, key=lambda f: len(f["unit_x"]))
    longest = max(frames, key=lambda f: len(f["unit_x"]))
    with torch.no_grad():
        assert torch.allclose(encoder(fj.collate_frames([shortest]))[0], encoder(fj.collate_frames([shortest, longest]))[0], atol=1e-5)


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
        ours = encoder(fj.collate_frames(frames))
        theirs = encoder(pm.collate(tensors).to("cpu"))
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_a_view_only_ever_takes_tokens_away(data):
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 16)
    b = fj.collate_frames(frames)
    view = fj.View(0.5, 0.5, 0.5, 0.5, 0.3, 0.01)
    seen = fj.apply_view(b, view, 699, torch.Generator().manual_seed(1))
    for mask, base in ((seen.unit_mask, b.unit_mask), (seen.reg_mask, b.reg_mask), (seen.site_mask, b.site_mask), (seen.feat_mask, b.feat_mask)):
        assert not (mask & ~base).any()
    assert 0.3 < seen.unit_mask.sum() / b.unit_mask.sum() < 0.7
    assert ((seen.unit_name == 699) & b.unit_mask).any()
    assert not torch.equal(seen.unit_x, b.unit_x)
    # The original is left alone, since other views read it.
    assert torch.equal(b.unit_x, fj.collate_frames(frames).unit_x)
    assert fj.apply_view(b, fj.View(), 699).unit_mask is b.unit_mask


def test_an_empty_view_of_units_still_gives_a_number(data):
    encoder = _encoder()
    b = fj.collate_frames(_frames(sorted(data.glob("scenario-*.npz"))[0], 4))
    gone = fj.apply_view(b, fj.View(keep_units=0.0, keep_regions=0.0, keep_sites=0.0, keep_features=0.0), 699)
    with torch.no_grad():
        assert torch.isfinite(encoder(gone)).all()


def test_later_views_read_the_later_frame(data):
    torch.manual_seed(1)
    model = fj.FrameJepa(SMALL, 700).eval()
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 4)
    now, other = fj.collate_frames(frames[:2]), fj.collate_frames(frames[2:])
    with torch.no_grad():
        z, projected = model.embed_views(now, other, (fj.View(), fj.View(later=True)))
    assert z.shape == (2, 2, SMALL.z) and projected.shape == (2, 2, SMALL.projected)
    assert torch.allclose(z[1], model.encoder(other), atol=1e-5)
    assert not torch.allclose(z[0], z[1], atol=1e-3)


# Probes -------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_probe_finds_what_is_in_a_latent_and_not_what_isnt():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(1200, 20))
    y = np.stack((z[:, 3] - 2.0 * z[:, 7] + 0.1 * rng.normal(size=1200), rng.normal(size=1200)), 1)
    r2 = fj.ridge_r2(z[:800], y[:800], z[800:], y[800:])
    assert r2[0] > 0.95
    assert r2[1] < 0.1


def test_a_probe_says_nothing_of_a_column_that_does_not_vary():
    """Say, ships on maps with no water.  The score would be huge and mean nothing."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(1000, 4))
    y = np.column_stack((z[:, 0], rng.normal(size=1000)))
    y[800:, 1] = 0.0
    r2 = fj.ridge_r2(z[:800], y[:800], z[800:], y[800:])
    assert r2[0] > 0.95 and np.isnan(r2[1])
    sample = {"z": z, "probes": y, "map": np.where(np.arange(1000) < 800, 0, 1), "game": np.arange(1000), "seat": np.zeros(1000, np.int32), "source": np.zeros(1000, np.int32)}
    report = fj.probe_report(sample, ("a", "b"), None, [1], {"both": [0, 1]})
    assert "r2_a" in report and "r2_b" not in report and "r2_both_b" not in report


def test_collapse_report_tells_a_collapse_from_a_spread():
    rng = torch.Generator().manual_seed(1)
    spread = fj.collapse_report(torch.randn(500, 32, generator=rng))
    one_way = fj.collapse_report(torch.randn(500, 1, generator=rng).repeat(1, 32))
    same = fj.collapse_report(torch.ones(500, 32))
    assert spread["effective_rank"] > 25.0
    assert one_way["effective_rank"] < 2.0
    assert same["std_mean"] == 0.0


def test_probes_are_taken_on_a_map_they_did_not_fit():
    rng = np.random.default_rng(1)
    n = 600
    sample = {"z": rng.normal(size=(n, 8)), "map": np.repeat(np.arange(3), n // 3), "source": np.zeros(n, np.int32), "seat": np.tile([0, 1], n // 2), "game": np.arange(n, dtype=np.int64)}
    sample["probes"] = sample["z"][:, :2].astype(np.float32)
    report = fj.probe_report(sample, ("a", "b"), None, held_out=2)
    assert report["r2_a"] > 0.99 and report["r2_b"] > 0.99
    assert fj.probe_report(sample, ("a", "b"), None, held_out=9) == {}


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
    report = fj.probe_report(sample, ("a",), (np.arange(n // 2, dtype=np.int64), result), held_out=2)
    assert report["r2_outcome"] > 0.99
    sample["source"] = np.ones(n, np.int32)
    assert "r2_outcome" not in fj.probe_report(sample, ("a",), (np.arange(n // 2, dtype=np.int64), result), held_out=2)


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
    roster, cat, table = scenarios.roster, se.catalog(), scenarios.table
    rows = np.tile(roster.structures[table.cost[roster.structures] < 300.0][:6], 6)
    rows = np.concatenate((rows, np.full(5, roster.extractor)))
    start = tuple(float(v) for v in bundle.spot_list[0, :2])
    built, x, z = fd._place_structures(np.random.default_rng(1), fd.Ground(bundle, facts), roster, rows, start, (start[0] + 2000.0, start[1]))
    radius = np.array([cat.units[cat.index[table.names[r]]]["footprint_radius"] for r in built])
    d = np.hypot(x[:, None] - x[None], z[:, None] - z[None]) - radius[:, None] - radius[None]
    np.fill_diagonal(d, 1.0)
    assert (d >= -1e-3).all()
    assert len(built) >= 5


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
    before = [env.frame(s).units[["uid", "x"]].copy() for s in range(0, 8)]
    for _ in range(8):
        env.step(0.0)
        env.finished_seats()
    # By uid, as what's on the board changes too.
    moved = []
    for s, was in enumerate(before):
        now = env.frame(s).units
        moved += [not np.array_equal(now["x"][now["uid"] == uid], was["x"][was["uid"] == uid]) for uid in np.intersect1d(was["uid"], now["uid"])]
    assert any(moved)
    assert (env.batch.counts >= 0).all()
    assert env.frame(0).glob[rf.G["frame"]] >= env.batch.frame(0, 0).glob[rf.G["frame"]]


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
    bundle = env.bundle_of(0)
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


def test_collected_data_has_both_sources_and_good_links(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    assert manifest["maps"] == list(se.TRAINING_MAPS)
    assert sorted(manifest["frames_per_second"]) == ["scenario", "script"]
    for source, name in enumerate(fd.SOURCES):
        paths = sorted(data.glob(f"{name}-*.npz"))
        assert len(paths) >= 2
        for path in paths:
            s = fj.Shards._read(path)
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
    assert set(fj.Shards._read(sorted(data.glob("scenario-*.npz"))[0])["seat"].tolist()) == {0, 1}


def test_a_later_frame_cant_coincide_with_the_next_one():
    with pytest.raises(ValueError):
        fd.Capture(every=4, later=4)


def test_script_and_scenario_games_are_numbered_apart(data):
    script = np.concatenate([fj.Shards._read(p)["game"] for p in data.glob("script-*.npz")])
    scenario = np.concatenate([fj.Shards._read(p)["game"] for p in data.glob("scenario-*.npz")])
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
    shards = fj.Shards(sorted(data.glob("*-*.npz")), rng, in_memory=4, maps=(0, 1, 2, 3))
    for s in shards.loaded:
        assert (s["map"][s["anchors"]] != 4).all()
    now, later = shards.sample(8)
    assert now.unit_x.shape[0] == 8 and later.glob_x.shape[0] == 8


# Training, a very little -----------------------------------------------------------------------------------------------------------------------------------


def test_a_short_training_run_leaves_a_loadable_encoder_and_logs(data, tmp_path):
    run = tmp_path / "run"
    cfg = fj.TrainConfig(batch=8, steps=12, warmup=2, report_every=4, probe_every=8, probe_frames=64, checkpoint_every=1000, seed=3)
    model = fj.train(data, run, cfg, SMALL, LOSS, device="cpu")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(np.isfinite(r["loss"]) and np.isfinite(r["sigreg"]) for r in rows)
    assert {"invariance", "effective_rank", "grad_norm"} <= set(rows[0])
    encoder = fj.load_encoder(run / "encoder-last.pt")
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 3)
    with torch.no_grad():
        assert torch.allclose(encoder(fj.collate_frames(frames)), model.encoder.cpu().eval()(fj.collate_frames(frames)))


def test_the_ceiling_trains_on_the_probes_and_reports_as_a_run_does(data, tmp_path):
    out = tmp_path / "ceiling.json"
    cfg = fj.TrainConfig(batch=8, steps=6, warmup=0, probe_frames=64, seed=3)
    report = fj.ceiling(data, out, cfg, SMALL, device="cpu")
    assert json.loads(out.read_text()) == report
    assert report == {} or all(k.startswith("r2_") and np.isfinite(v) for k, v in report.items())


def test_a_stopped_run_goes_on_from_where_it_was(data, tmp_path):
    run = tmp_path / "run"
    (run).mkdir()
    (run / "stop").touch()
    cfg = fj.TrainConfig(batch=8, steps=20, warmup=2, report_every=4, probe_every=1000, probe_frames=32, checkpoint_every=1000, seed=3)
    fj.train(data, run, cfg, SMALL, LOSS, device="cpu")
    assert torch.load(run / "state.pt", weights_only=True)["step"] == 0
    (run / "stop").unlink()
    fj.train(data, run, cfg, SMALL, LOSS, device="cpu")
    steps = [json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert steps == [0, 4, 8, 12, 16, 19]
    assert torch.load(run / "state.pt", weights_only=True)["step"] == 19


def test_invariance_alone_collapses_and_sigreg_prevents_it(data):
    """The reason for the second term, on real frames."""
    frames = _frames(sorted(data.glob("scenario-*.npz"))[0], 32)
    batch = fj.collate_frames(frames)
    views = (fj.View(), fj.View(0.6, 0.6, 0.6, 0.6))

    def run(weight: float) -> dict:
        torch.manual_seed(1)
        model = fj.FrameJepa(SMALL, 700)
        optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
        generator = torch.Generator().manual_seed(1)
        for _ in range(60):
            _, projected = model.embed_views(batch, batch, views, generator)
            loss, _ = fj.lejepa_loss(projected, 1, fj.LossConfig(weight=weight, slices=64), generator)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            _, projected = model.embed_views(batch, batch, (fj.View(),))
        return fj.collapse_report(projected[0])

    alone, with_sigreg = run(0.0), run(0.2)
    assert with_sigreg["std_mean"] > 5.0 * alone["std_mean"]


def test_refuses_an_encoder_from_other_features(data, tmp_path):
    model = fj.FrameJepa(SMALL, 700)
    path = tmp_path / "e.pt"
    fj.save_encoder(path, model, 1)
    saved = torch.load(path, weights_only=True)
    saved["feature_version"] = -1
    torch.save(saved, path)
    with pytest.raises(ValueError):
        fj.load_encoder(path)
