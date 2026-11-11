"""The frame VAE: what it's asked to give back, that its latents are well behaved, and that it can learn."""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import frame_data as fd
    import frame_model as fm
    import frame_vae as fv
    from zkbot.policy import featurize as fz, tables
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

ENCODER = fm.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=0)
VAE = fv.VaeConfig(latent=8, decoder_d=32, decoder_heads=2, decoder_rounds=1, beta_warmup=0)
TABLE = tables.table()


def _model(seed: int = 1, cfg: fv.VaeConfig = VAE) -> fv.FrameVae:
    torch.manual_seed(seed)
    return fv.FrameVae(cfg, ENCODER, TABLE).eval()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    fd.collect(out, script_frames=300, scenario_frames=300, games=24, seed=7, threads=4, script_every=6, frames_per_shard=120)
    return out


@pytest.fixture(scope="module")
def frames(data):
    s = fm.Shards._read(sorted(data.glob("*-*.npz"))[0])
    return [fm.Shards.frame(s, i) for i in range(32)]


def _hand_made(units: list[tuple[str, float, float, int, float]]) -> fm.FrameBatch:
    """Two regions, at x = -0.5 and 0.5.  units: name, x, z, alliance from the viewer, health."""
    reg = np.zeros((2, fz.REGION_DIM), np.float32)
    reg[:, 0] = (-0.5, 0.5)
    reg[0, fv.REGION_STATIC] = 0.25
    x = np.zeros((len(units), fz.UNIT_DIM), np.float32)
    for i, (_, ux, uz, _, hp) in enumerate(units):
        x[i, fz._I:fz._I + 2] = ux, uz
        x[i, fz._I + 6] = hp
    names = np.array([TABLE.names.index(u[0]) for u in units], np.int32)
    return fm.collate_frames([{"unit_x": x, "unit_name": names, "unit_ally": np.array([u[3] for u in units], np.int32), "unit_player": np.zeros(len(units), np.int32),
                               "tok_kind": np.zeros(len(units), np.int32), "reg_x": reg, "site_x": np.zeros((0, fz.SITE_DIM), np.float32), "feat_x": np.zeros((0, fz.FEATURE_DIM), np.float32),
                               "glob_x": np.zeros(fz.GLOBAL_DIM, np.float32)}])


def _cell(side: str, group: str) -> int:
    return fv.REGION_TARGETS.index(f"{side}_{group}")


def _value(name: str, hp: float = 1.0) -> float:
    return float(np.log1p(TABLE.cost[TABLE.names.index(name)] * hp) / fv.VALUE_SCALE)


# What it's asked for ------------------------------------------------------------------------------------------------------------------------------------


def test_targets_count_a_unit_for_its_side_class_and_nearest_region():
    b = _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0), ("gunshipraid", 0.6, 0.0, 1, 0.5), ("turretlaser", 0.4, 0.1, 1, 1.0), ("cloakcon", -0.6, 0.0, 0, 1.0), ("shipraider" if "shipraider" in TABLE.names else "shipscout", 0.5, 0.0, 0, 1.0)])
    t = _model().region_targets(b)[0]
    ship = "shipraider" if "shipraider" in TABLE.names else "shipscout"
    assert float(t[0, _cell("own", "ground")]) == pytest.approx(_value("cloakraid"), rel=1e-4)
    assert float(t[0, _cell("own", "builder")]) == pytest.approx(_value("cloakcon"), rel=1e-4)
    assert float(t[1, _cell("enemy", "air")]) == pytest.approx(_value("gunshipraid", 0.5), rel=1e-4)
    assert float(t[1, _cell("enemy", "structure")]) == pytest.approx(_value("turretlaser"), rel=1e-4)
    assert float(t[1, _cell("own", "ship")]) == pytest.approx(_value(ship), rel=1e-4)
    assert float(t[1, _cell("own", "ground")]) == 0.0 and float(t[0, _cell("enemy", "ground")]) == 0.0
    assert int((t[:, len(fv.REGION_NOW):] != 0.0).sum()) == 5


def test_targets_include_the_columns_of_the_region_token_that_change():
    t = _model().region_targets(_hand_made([("cloakraid", -0.4, 0.0, 0, 1.0)]))[0]
    assert float(t[0, 0]) == 0.25 and float(t[1, 0]) == 0.0


def test_summaries_and_unknown_names_add_nothing_to_the_class_columns():
    assert float(TABLE.cost[TABLE.unknown]) == 0.0


def test_what_a_region_is_asked_does_not_carry_what_it_has_to_answer():
    """The query is the map's columns only: change what's there and the query stays."""
    assert fv.REGION_STATIC + len(fv.REGION_NOW) == fz.REGION_DIM
    model = _model()
    b = _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0)])
    latents = torch.zeros(1, ENCODER.queries, VAE.latent)
    before = model.decode(latents, b)[0].clone()
    b.reg_x[..., fv.REGION_STATIC:] = 9.0
    b.unit_x[..., fz._I + 6] = 0.1
    assert torch.equal(model.decode(latents, b)[0], before)


# Its latents ----------------------------------------------------------------------------------------------------------------------------------------


def test_latents_come_as_tokens_of_the_configured_size(frames):
    assert _model().latent_tokens(fm.collate_frames(frames[:3])).shape == (3, ENCODER.queries, VAE.latent)


def test_latents_do_not_depend_on_token_order_or_padding(frames):
    model = _model()
    alone = model.latent_tokens(fm.collate_frames(frames[:1]))
    padded = model.latent_tokens(fm.collate_frames(frames[:4]))[:1]
    assert torch.allclose(alone, padded, atol=1e-4)
    f = dict(frames[0])
    order = np.random.default_rng(1).permutation(len(f["unit_x"]))
    for name in ("unit_x", "unit_name", "unit_ally", "unit_player", "tok_kind"):
        f[name] = f[name][order]
    assert torch.allclose(alone, model.latent_tokens(fm.collate_frames([f])), atol=1e-4)


def test_noise_only_when_sampling(frames):
    model = _model()
    b = fm.collate_frames(frames[:4])
    a, c = model(b, sample=False)[0], model(b, sample=False)[0]
    assert torch.equal(a, c)
    assert not torch.equal(model(b, sample=True)[0], a)


def test_padded_regions_change_no_answer(frames):
    model = _model()
    alone = model(fm.collate_frames(frames[:1]), sample=False)
    together = model(fm.collate_frames(frames[:4]), sample=False)
    count = len(frames[0]["reg_x"])
    assert torch.allclose(alone[0][0, :count], together[0][0, :count], atol=1e-3)
    assert torch.allclose(alone[1][0], together[1][0], atol=1e-3)


def test_kl_is_the_formula():
    assert float(fv.kl_per_token_number(torch.zeros(1), torch.zeros(1))) == 0.0
    assert float(fv.kl_per_token_number(torch.tensor(2.0), torch.tensor(0.0))) == pytest.approx(2.0)
    var = torch.tensor(0.25)
    assert float(fv.kl_per_token_number(torch.tensor(0.0), var.log())) == pytest.approx(0.5 * (0.25 - 1.0 - float(var.log())))


def test_loss_does_not_grow_with_padding(frames):
    model = _model()
    b = fm.collate_frames(frames[:8])
    model.fit_scales([b])
    torch.manual_seed(3)
    plain = fv.vae_loss(model, b, 0.0)[1]["recon"]
    wider = fm.FrameBatch(**{**b.__dict__, "reg_x": torch.cat((b.reg_x, torch.zeros(8, 5, fz.REGION_DIM)), 1), "reg_mask": torch.cat((b.reg_mask, torch.zeros(8, 5, dtype=torch.bool)), 1)})
    torch.manual_seed(3)
    assert fv.vae_loss(model, wider, 0.0)[1]["recon"] == pytest.approx(plain, rel=2e-2)


def test_beta_rises_from_nothing_over_its_warmup():
    cfg = fv.VaeConfig(beta=0.2, beta_warmup=100)
    assert fv.beta_at(cfg, 0) == 0.0 and fv.beta_at(cfg, 50) == pytest.approx(0.1) and fv.beta_at(cfg, 500) == 0.2


# A run ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_run_trains_reports_saves_and_goes_on(data, tmp_path):
    run = tmp_path / "run"
    cfg = fm.TrainConfig(batch=8, steps=4, report_every=1, probe_every=3, probe_frames=64, warmup=1, checkpoint_every=100)
    model = fv.train(data, run, cfg, VAE, ENCODER, device="cpu")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert all(r["kl_nats"] >= 0.0 for r in rows)
    assert (model.region_std != 1.0).any()
    loaded = fv.load_vae(run / "vae-last.pt")
    frames = fm.collate_frames([fm.Shards.frame(fm.Shards._read(sorted(data.glob("*-*.npz"))[0]), 0)])
    assert torch.allclose(loaded.latent_tokens(frames), model.cpu().eval().latent_tokens(frames))
    assert (loaded.region_std == model.region_std).all()
    fv.train(data, run, replace(cfg, steps=6), VAE, ENCODER, device="cpu")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2, 3, 4, 5]


def test_it_refuses_a_checkpoint_of_other_features(tmp_path):
    model = _model()
    fv.save_vae(tmp_path / "a.pt", model, 0)
    saved = torch.load(tmp_path / "a.pt", weights_only=True)
    saved["feature_version"] = -1
    torch.save(saved, tmp_path / "a.pt")
    with pytest.raises(ValueError):
        fv.load_vae(tmp_path / "a.pt")


# The plain autoencoder ----------------------------------------------------------------------------------------------------------------------


AUTO = replace(VAE, variational=False)


def test_an_autoencoder_has_no_noise_and_no_kl(frames):
    model = _model(cfg=AUTO)
    b = fm.collate_frames(frames[:8])
    assert not hasattr(model, "to_log_var")
    assert torch.equal(model(b, sample=True)[0], model(b, sample=False)[0])
    model.fit_scales([b])
    _, parts, _ = fv.vae_loss(model, b, 0.5)
    assert parts["kl_nats"] == 0.0
    assert float(fv.vae_loss(model, b, 0.5)[0]) == pytest.approx(float(fv.vae_loss(model, b, 0.0)[0]))


def test_an_autoencoders_latents_keep_to_one_scale(frames):
    model = _model(cfg=AUTO)
    z = model.latent_tokens(fm.collate_frames(frames[:8]))
    assert z.shape == (8, ENCODER.queries, AUTO.latent)
    assert float(z.mean(-1).abs().max()) < 1e-4 and float((z.std(-1, unbiased=False) - 1.0).abs().max()) < 1e-2


def test_an_autoencoder_comes_back_from_a_checkpoint(frames, tmp_path):
    model = _model(cfg=AUTO)
    b = fm.collate_frames(frames[:16])
    model.fit_scales([b])
    fv.save_vae(tmp_path / "a.pt", model, 40)
    again = fv.load_vae(tmp_path / "a.pt")
    assert not again.cfg.variational
    assert torch.allclose(again.latent_tokens(b), model.latent_tokens(b))


# Summing units into regions -----------------------------------------------------------------------------------------------------------------------


POOLED = fm.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=0, region_pool=6)


def _pooled_model(seed: int = 1, cfg: fv.VaeConfig = VAE) -> fv.FrameVae:
    torch.manual_seed(seed)
    return fv.FrameVae(cfg, POOLED, TABLE).eval()


def _sums(model: fv.FrameVae, b) -> torch.Tensor:
    return model.encoder.pooled(model.encoder.tokens(b)[0][:, :b.unit_x.shape[1]], b).sinh()


def test_pooling_adds_up_each_regions_units_and_leaves_the_others_alone():
    model = _pooled_model()
    one = _sums(model, _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0)]))[0]
    three = _sums(model, _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0)] * 3))[0]
    assert torch.allclose(three[0], 3.0 * one[0], rtol=1e-3, atol=1e-4)
    assert float(one[0].abs().max()) > 0.0
    assert float(one[1].abs().max()) == 0.0 and float(three[1].abs().max()) == 0.0


def test_pooling_sends_a_unit_to_the_region_its_targets_count_it_for():
    b = _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0), ("cloakraid", 0.6, 0.0, 1, 1.0)])
    assert fm.nearest_region(b).tolist() == [[0, 1]]


def test_pooling_ignores_padded_units_regions_and_order(frames):
    model = _pooled_model()
    alone = model.latent_tokens(fm.collate_frames(frames[:1]))
    assert torch.allclose(alone, model.latent_tokens(fm.collate_frames(frames[:4]))[:1], atol=1e-3)
    f = dict(frames[0])
    order = np.random.default_rng(1).permutation(len(f["unit_x"]))
    for name in ("unit_x", "unit_name", "unit_ally", "unit_player", "tok_kind"):
        f[name] = f[name][order]
    assert torch.allclose(alone, model.latent_tokens(fm.collate_frames([f])), atol=1e-3)


def test_pooling_passes_gradients_and_a_checkpoint_keeps_it(frames, tmp_path):
    model = _pooled_model()
    b = fm.collate_frames(frames[:4])
    model.fit_scales([b])
    fv.vae_loss(model, b, 0.0)[0].backward()
    assert float(model.encoder.pool_in[0].weight.grad.abs().sum()) > 0.0
    fv.save_vae(tmp_path / "a.pt", model, 1)
    again = fv.load_vae(tmp_path / "a.pt")
    assert again.encoder_cfg.region_pool == POOLED.region_pool
    assert torch.allclose(again.latent_tokens(b), model.eval().latent_tokens(b))


def test_an_encoder_without_pooling_has_no_new_weights():
    assert not hasattr(_model().encoder, "pool_in")
    assert _model().encoder.reg_in.in_features == fz.REGION_DIM + 4 * ENCODER.position_bands


# Two answers for each cell ------------------------------------------------------------------------------------------------------------------------


TWO = replace(VAE, two_part=True)


def _scaled_model(frames, cfg: fv.VaeConfig = TWO) -> tuple[fv.FrameVae, fm.FrameBatch]:
    model = _model(cfg=cfg)
    b = fm.collate_frames(frames[:8])
    model.fit_scales([b])
    return model, b


def test_two_part_answers_twice_for_each_cell_and_one_value_for_each(frames):
    model, b = _scaled_model(frames)
    region = model(b, sample=False)[0]
    assert region.shape == (*b.reg_mask.shape, 2 * len(fv.REGION_TARGETS))
    assert model.region_answer(region).shape == (*b.reg_mask.shape, len(fv.REGION_TARGETS))


def test_two_part_expects_its_size_if_sure_and_an_empty_cell_if_sure_not(frames):
    model, _ = _scaled_model(frames)
    count = len(fv.REGION_TARGETS)
    size = torch.full((1, 1, count), 3.0)
    sure = model.region_answer(torch.cat((torch.full((1, 1, count), 50.0), size), -1))
    sure_not = model.region_answer(torch.cat((torch.full((1, 1, count), -50.0), size), -1))
    assert torch.allclose(sure, size)
    assert torch.allclose(sure_not[0, 0], model.empty_cell())


def test_filled_cells_are_those_not_at_a_columns_zero(frames):
    model, b = _scaled_model(frames)
    raw = model.region_targets(b)
    scaled = (raw - model.region_mean) / model.region_std
    assert (model.filled(scaled)[b.reg_mask] == (raw[b.reg_mask].abs() * (1.0 / model.region_std) > fv.FILLED_ABOVE)).all()


def test_two_part_does_not_pay_for_size_on_empty_cells(frames):
    model, b = _scaled_model(frames)
    want = model.scaled_targets(b)[0]
    filled = model.filled(want)
    assert filled[b.reg_mask].any() and not filled[b.reg_mask].all()

    def costs(on_filled: torch.Tensor, on_empty: float) -> float:
        size = torch.where(filled, on_filled, torch.full_like(want, on_empty))
        model.decode = lambda latents, batch: (torch.cat((torch.zeros_like(want), size), -1), torch.zeros(len(want), fz.GLOBAL_DIM))
        return float(fv.vae_loss(model, b, 0.0)[0])

    assert costs(want, 100.0) == pytest.approx(costs(want, -100.0))
    assert costs(want + 5.0, 0.0) > costs(want, 0.0)


# Questions about units and wrecks -------------------------------------------------------------------------------------------------------------------


ASKING = replace(VAE, variational=False, two_part=True, questions=True)


def _asking_model(seed: int = 1) -> fv.FrameVae:
    torch.manual_seed(seed)
    return fv.FrameVae(ASKING, ENCODER, TABLE).eval()


def test_a_description_gives_no_place_away():
    first, last = fv.UNIT_STATE
    assert first == fz._I + 3 and last == fz._O     # past place and altitude, and short of orders and rays
    assert fv.WRECK_METAL == 2 and fz.FEATURE_DIM == 5  # after it, how far it is from the base and the army


def test_asking_where_does_not_depend_on_the_columns_that_would_tell(frames):
    model = _asking_model()
    b = fm.collate_frames(frames[:4])
    torch.manual_seed(5)
    q = fv.draw_questions(b)
    moved = fm.FrameBatch(**b.__dict__)
    moved.unit_x = b.unit_x.clone()
    moved.unit_x[..., fz._I:fz._I + 3] += 0.25
    moved.unit_x[..., fz._O:] += 0.25
    moved.feat_x = b.feat_x.clone()
    moved.feat_x[..., [0, 1, 3, 4]] += 0.25
    torch.manual_seed(5)
    again = fv.draw_questions(moved)
    assert torch.equal(q.name, again.name) and torch.equal(q.state, again.state) and torch.equal(q.wreck_metal, again.wreck_metal)
    assert not torch.equal(q.place, again.place)
    assert model is not None


def test_questions_skip_summaries_and_mark_what_a_small_frame_lacks():
    b = _hand_made([("cloakraid", -0.4, 0.0, 0, 1.0), ("cloakraid", 0.4, 0.0, 1, 1.0), ("cloakraid", 0.0, 0.5, 0, 1.0)])
    b.tok_kind[0, 2] = fz.TK_SUMMARY
    q = fv.draw_questions(b)
    assert int(q.unit_ok.sum()) == 2
    wanted = {tuple(round(v, 3) for v in p) for p in q.place[q.unit_ok].tolist()}
    assert wanted == {(-0.4, 0.0), (0.4, 0.0)}
    assert int(q.wreck_ok.sum()) == 0


def test_what_is_at_a_place_comes_from_what_is_nearest_it():
    b = _hand_made([("cloakraid", -0.4, 0.0, 0, 0.5), ("gunshipraid", 0.4, 0.0, 1, 1.0)])
    q = fv.draw_questions(b)
    count = int(q.unit_ok.sum())
    assert count == 2
    at_units = q.at[:, :2]
    assert q.near_unit[:, :2].all()
    for i in range(2):
        expected = "cloakraid" if float(at_units[0, i, 0]) < 0 else "gunshipraid"
        assert int(q.at_name[0, i]) == TABLE.names.index(expected)
        assert int(q.at_side[0, i]) == (0 if expected == "cloakraid" else 1)


def test_places_beside_a_unit_start_just_outside_its_radius(frames):
    b = fm.collate_frames(frames[:16])
    q = fv.draw_questions(b)
    start = fv.ASK_AT_UNITS + fv.ASK_AT_WRECKS
    beside = q.at[:, start:start + fv.ASK_BESIDE]
    ok = q.at_ok[:, start:start + fv.ASK_BESIDE]
    assert ok.any()
    # Each was drawn outside one unit's radius, so most hold nothing, and a few have another unit.
    assert 0.0 < float(q.near_unit[:, start:start + fv.ASK_BESIDE][ok].float().mean()) < 0.9
    assert float(beside.abs().max()) <= 1.0
    assert q.at_ok[:, -fv.ASK_ANYWHERE:].all()


def _answers(q: fv.Questions, weights, means, spread) -> fv.Answers:
    b, w = q.name.shape[0], q.name.shape[1] + q.wreck_metal.shape[1]
    a = q.at.shape[1]
    places = len(means)
    return fv.Answers(torch.tensor(weights, dtype=torch.float32).expand(b, w, places), torch.tensor(means, dtype=torch.float32).expand(b, w, places, 2),
                      torch.full((b, w, places), spread), torch.zeros(b, a), torch.zeros(b, a), torch.zeros(b, a, len(TABLE.names) + 1), torch.zeros(b, a, 2),
                      torch.zeros(b, a, q.at_state.shape[-1]), torch.zeros(b, a, 1))


def test_a_where_answer_with_a_place_for_each_look_alike_beats_their_average():
    """Two units described alike: one place at their mean is near neither."""
    b = _hand_made([("cloakraid", -0.5, 0.0, 0, 1.0), ("cloakraid", 0.5, 0.0, 0, 1.0)])
    q = fv.draw_questions(b)
    both, _ = fv.question_cost(q, _answers(q, [0.0, 0.0], [[-0.5, 0.0], [0.5, 0.0]], 0.02))
    mean, _ = fv.question_cost(q, _answers(q, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], 0.02))
    assert float(both.sum()) < float(mean.sum())


def test_the_question_cost_ignores_what_was_not_asked():
    b = _hand_made([("cloakraid", -0.5, 0.0, 0, 1.0)])
    q = fv.draw_questions(b)
    a = _answers(q, [0.0, 0.0], [[-0.5, 0.0], [0.5, 0.0]], 0.1)
    where, what = fv.question_cost(q, a)
    gone = q._replace(unit_ok=torch.zeros_like(q.unit_ok), wreck_ok=torch.zeros_like(q.wreck_ok), at_ok=torch.zeros_like(q.at_ok))
    none_where, none_what = fv.question_cost(gone, a)
    assert float(where.abs().sum()) > 0.0 and float(what.abs().sum()) > 0.0
    assert float(none_where.sum()) == 0.0 and float(none_what.sum()) == 0.0


def test_a_right_what_answer_costs_less_than_a_wrong_one():
    b = _hand_made([("cloakraid", -0.5, 0.0, 0, 1.0)])
    q = fv.draw_questions(b)
    right = _answers(q, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], 1.0)
    picks = lambda t, cols: torch.zeros_like(t).scatter(-1, cols[..., None], 8.0)  # noqa: E731
    right = right._replace(unit_there=(q.near_unit.float() * 2.0 - 1.0) * 8.0, wreck_there=torch.full_like(right.wreck_there, -8.0), name=picks(right.name, q.at_name), side=picks(right.side, q.at_side), state=q.at_state.clone())
    wrong = right._replace(unit_there=-right.unit_there, name=picks(right.name, (q.at_name + 1) % right.name.shape[-1]), side=picks(right.side, 1 - q.at_side), state=q.at_state + 1.0)
    assert float(fv.question_cost(q, right)[1].sum()) < float(fv.question_cost(q, wrong)[1].sum())


def test_answers_have_the_shapes_and_spreads_they_should(frames):
    model = _asking_model()
    b = fm.collate_frames(frames[:4])
    q = fv.draw_questions(b)
    a = model.answer_questions(model.latent_tokens(b), q)
    asked = fv.ASK_WHERE_UNITS + fv.ASK_WHERE_WRECKS
    assert a.place_means.shape[1:] == (q.name.shape[1] + q.wreck_metal.shape[1], ASKING.places, 2) and q.name.shape[1] + q.wreck_metal.shape[1] <= asked
    assert a.name.shape == (4, q.at.shape[1], model.encoder.vocabulary)
    assert float(a.place_spreads.min()) >= fv.PLACE_SPREAD[0] * 0.999 and float(a.place_spreads.max()) <= fv.PLACE_SPREAD[1] * 1.001


def test_questions_reach_the_encoder_and_recon_leaves_them_out(frames):
    model = _asking_model()
    b = fm.collate_frames(frames[:8])
    model.fit_scales([b])
    torch.manual_seed(2)
    loss, parts, _ = fv.vae_loss(model, b, 0.0)
    loss.backward()
    assert float(model.encoder.queries.grad.abs().sum()) > 0.0 and float(model.at_in.weight.grad.abs().sum()) > 0.0
    assert "where_nats" in parts and "what_nats" in parts
    torch.manual_seed(1)
    plain = fv.FrameVae(replace(ASKING, questions=False), ENCODER, TABLE).eval()
    assert not hasattr(plain, "where_out")


def test_the_report_marks_questions_on_held_out_maps(data):
    manifest, paths, held_out, _, _ = fm.collection(data)
    model = _asking_model()
    model.fit_scales([fm.collate_frames([fm.Shards.frame(fm.Shards._read(paths[0]), i) for i in range(8)])])
    out = fv.question_report(model, paths, held_out, "cpu", 64)
    assert "where_unit_nats" in out and "where_any_0.05" in out and "what_unit_found" in out
    assert 0.0 <= out["where_top_0.05"] <= out["where_any_0.05"] <= 1.0
    assert 0.0 <= out["what_type"] <= 1.0


# The optimiser -------------------------------------------------------------------------------------------------------------------------------------


MUON = fm.TrainConfig(batch=8, steps=3, report_every=1, probe_every=2, probe_frames=64, warmup=1, checkpoint_every=100, muon_lr=1e-3)


def test_muon_takes_only_hidden_layers_weights():
    model = _asking_model()
    both = fv.Optimisers(model, MUON)
    muon = {id(p) for group in both.parts["muon"].param_groups for p in group["params"]}
    adam = {id(p) for group in both.parts["adam"].param_groups for p in group["params"]}
    assert muon and not muon & adam and muon | adam == {id(p) for p in model.parameters()}
    assert id(model.encoder.cross[0].q.weight) in muon
    for kept in (model.region_out.weight, model.where_out.weight, model.at_out.weight, model.global_out.weight, model.encoder.queries, model.encoder.name_embed.weight, model.encoder.cross[0].q.bias):
        assert id(kept) in adam


def test_without_a_muon_rate_it_is_plain_adamw():
    both = fv.Optimisers(_asking_model(), fm.TrainConfig())
    assert list(both.parts) == ["adam"]


def test_the_schedule_moves_both_rates_together():
    both = fv.Optimisers(_asking_model(), MUON)
    both.set_rate(MUON.lr / 2.0)
    assert both.parts["adam"].param_groups[0]["lr"] == pytest.approx(MUON.lr / 2.0)
    assert both.parts["muon"].param_groups[0]["lr"] == pytest.approx(MUON.muon_lr / 2.0)


def test_a_run_as_the_pipeline_trains_it_reports_goes_on_and_keeps_what_it_is(data, tmp_path):
    """An autoencoder with two answers a cell, questions, summed regions and Muon: policy_pipeline's ENCODER_FLAGS."""
    run = tmp_path / "run"
    trained = fv.train(data, run, MUON, ASKING, POOLED, device="cpu")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert "where_unit_nats" in rows[-1] and "recon_r2" in rows[-1]
    assert trained.encoder_cfg.region_pool == POOLED.region_pool
    fv.train(data, run, replace(MUON, steps=5), ASKING, POOLED, device="cpu")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2, 3, 4]
    kept = fv.load_vae(run / "vae-last.pt")
    assert kept.cfg.questions and kept.cfg.two_part and not kept.cfg.variational and kept.encoder_cfg.region_pool == POOLED.region_pool


def test_it_learns_and_its_answers_get_better(frames):
    """The one check that the loss can be learned at all, by the optimisers a run uses."""
    torch.manual_seed(1)
    model = fv.FrameVae(ASKING, ENCODER, TABLE)
    b = fm.collate_frames(frames[:16])
    model.fit_scales([b])
    both = fv.Optimisers(model, fm.TrainConfig(muon_lr=2e-3, lr=2e-3))
    first = None
    for _ in range(30):
        loss, parts, _ = fv.vae_loss(model, b, 0.0)
        first = first or (float(loss), parts["recon"], parts["where_nats"], parts["what_nats"])
        both.zero_grad()
        loss.backward()
        both.step()
    assert float(loss) < first[0] and parts["recon"] < first[1] and parts["where_nats"] < first[2] and parts["what_nats"] < first[3]

