"""The trainer: minibatches within their budget, playing many games in pieces, the move to self-play, the card's memory shared
between runs, and a short run of the actor on latents (tests/policy/test_latent_actor.py has the actor itself)."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import policy_latent as pl
    import policy_model as pm
    import policy_train as pt
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)
from zkbot.policy import rawframe as rf, tables  # noqa: E402

from conftest import sim_frames  # noqa: E402

VOCABULARY = len(tables.table().names) + 1


@pytest.fixture(scope="module")
def learner(tmp_path_factory, encoder_path):
    cfg = pt.Config(data_workers=0, latent_encoder=str(encoder_path))
    torch.manual_seed(3)
    return pt.Learner(cfg, tmp_path_factory.mktemp("run"), pt.DataWorkers(0, [sim_frames(1)[1], sim_frames(40)[1]], cfg.enabled_kinds), device="cpu")


def test_a_minibatch_budgets_its_padded_size(learner):
    """A few frames from a big map can't ride along with many small ones: padded, they would all cost as much as the big map's."""
    def segment(n_units, regions, length):
        frame = rf.Frame(np.zeros(n_units, rf.UNIT_DTYPE), np.zeros(0, rf.SITE_DTYPE), np.zeros(len(rf.GLOBAL_FIELDS), np.float32))
        return pt.Segment([pt.Step(frame, *(np.zeros(0) for _ in range(11)))] * length, np.zeros(1), np.zeros((regions, 1)), SimpleNamespace(regions=regions), np.zeros(1))
    segments = [segment(40, 24, 32) for _ in range(40)] + [segment(20, 76, 32) for _ in range(40)]
    budget = learner.token_budget
    learner.token_budget = 200_000
    try:
        batches = learner._minibatches(segments, np.random.default_rng(1))
    finally:
        learner.token_budget = budget
    assert sorted(id(s) for b in batches for s in b) == sorted(id(s) for s in segments)
    for b in batches:
        decisions = sum(len(s.steps) for s in b)
        padded = max(len(s.steps[0].frame.units) for s in b) + max(s.bundle.regions for s in b) + 2
        assert decisions * (padded + pt.DECISION_TOKENS) <= 200_000 or len(b) == 1


def test_a_minibatch_that_does_not_fit_is_done_in_runs_of_its_segments(learner, monkeypatch):
    """Each run's loss weighed by its share of the decisions, the gradient is the whole minibatch's; and the budget shrinks."""
    weight = learner.trained[0]
    segments = [pt.Segment([None] * n, None, None, None, np.full(1, v)) for n, v in ((3, 1.0), (5, 2.0), (8, -1.0))]

    def loss(run, eps, note):
        if len(run) > 1:
            raise torch.OutOfMemoryError("too big")
        note("tried", 1.0)
        return weight.sum() * sum(len(s.steps) * float(s.names[0]) for s in run) / sum(len(s.steps) for s in run)
    monkeypatch.setattr(learner, "_loss", loss)
    budget, notes = learner.token_budget, []
    try:
        learner._backward(segments, 0.2, lambda *a: notes.append(a))
        assert learner.token_budget < budget
    finally:
        learner.token_budget = budget
    assert torch.allclose(weight.grad, torch.full_like(weight, (3 * 1.0 + 5 * 2.0 - 8 * 1.0) / 16))
    assert notes == [("tried", 1.0)] * 3 + [("minibatch_decisions", 16)]


def test_playing_in_pieces_decides_the_same_as_in_one(learner, monkeypatch):
    frames, bundle = sim_frames(40)
    names = np.arange(learner.vocabulary, dtype=np.int32)
    picked = [(frames[k], bundle, names) for k in (0, 10, 20, 30, 39)]     # armies of different sizes, so widths differ between pieces
    history, seen = learner.actor.initial_state(len(picked), 0, "cpu")
    torch.manual_seed(4)
    items, out, value, new_history, new_seen = learner.act(picked, history, seen)
    tokens = max(len(t.unit_x) + len(t.reg_x) + len(t.site_x) + len(t.feat_x) for t in items) + 2
    monkeypatch.setattr(pt, "ACT_PAIRS", 2 * tokens ** 2)
    assert len(pt._act_slices(items)) > 1
    torch.manual_seed(4)
    cut_items, cut, cut_value, cut_history, cut_seen = learner.act(picked, history, seen)
    assert np.allclose(cut_value, value, atol=1e-5)
    assert torch.allclose(cut_seen, new_seen)
    for k, t in enumerate(cut_items):
        picked = cut["anchor"][k] >= 0
        assert t.kind_mask[cut["anchor"][k][picked], cut["kind"][k][picked]].all()


def _self_play():
    return pt.SelfPlay(start_at=0.8, hold_below=0.7, per_iteration=0.01)


def test_self_play_waits_for_two_tests_that_beat_the_script():
    plan = _self_play()
    assert plan.advance() == 0.0
    plan.note_test(0.95, 10)
    assert plan.advance() == 0.0, "one good test is mostly luck"
    plan.note_test(0.5, 20)
    assert plan.advance() == 0.0
    plan.note_test(0.9, 30)
    assert plan.advance() == 0.0, "0.9 and 0.5"
    plan.note_test(0.8, 40)
    assert plan.started_at == 40 and plan.advance() == 0.01


def test_self_play_grows_a_little_at_a_time_up_to_all_of_it():
    plan = _self_play()
    plan.note_test(0.8, 10)
    plan.note_test(0.8, 20)
    shares = [plan.advance() for _ in range(150)]
    assert shares[0] == 0.01 and shares[49] == pytest.approx(0.5) and shares[-1] == 1.0
    assert all(b >= a and b - a <= 0.0101 for a, b in zip(shares, shares[1:]))


def test_self_play_pauses_if_the_script_starts_winning_and_never_goes_back():
    plan = _self_play()
    plan.note_test(0.8, 10)
    plan.note_test(0.8, 20)
    for _ in range(30):
        plan.advance()
    held = plan.share
    plan.note_test(0.6, 30)
    plan.note_test(0.6, 40)
    assert plan.advance() == held and plan.started_at == 20
    plan.note_test(0.9, 50)
    assert plan.advance() > held, "0.6 and 0.9"


def test_a_run_remembers_how_far_self_play_got(tmp_path, encoder_path):
    cfg = pt.Config(decisions=0, games=2, iteration=4, latent_encoder=str(encoder_path))
    workers = pt.DataWorkers(0, [], cfg.enabled_kinds)
    first = pt.Learner(cfg, tmp_path, workers, "cpu")
    first.self_play.note_test(0.8, 10)
    first.self_play.note_test(0.9, 20)
    first.self_play.share = 0.25
    first.save()
    second = pt.Learner(cfg, tmp_path, workers, "cpu")
    assert second.resume() and second.self_play.share == 0.25 and second.self_play.started_at == 20 and second.self_play.tests == [0.8, 0.9]


def test_runs_sharing_the_card_stay_inside_their_share():
    card = 12 * 2**30
    assert pt.gpu_fraction(0.85, 1, card) == 0.85
    for runs in (2, 4, 7):
        each = pt.gpu_fraction(0.85, runs, card)
        assert runs * each * card + (runs - 1) * pt.PROCESS_OVERHEAD == pytest.approx(0.85 * card)


def test_too_many_runs_for_the_card_is_an_error():
    with pytest.raises(ValueError):
        pt.gpu_fraction(0.85, 40, 12 * 2**30)


def test_spo_is_greatest_at_the_edge_of_the_trust_region_and_pulls_back_from_past_it():
    """Where PPO's clip gives a ratio past 1 + eps no gradient at all."""
    eps = 0.2
    for adv, best in ((1.0, 1.2), (-1.0, 0.8)):
        r = torch.tensor([best, 1.0, 1.0 + 3 * (best - 1.0)], requires_grad=True)
        pt.spo_objective(r, torch.tensor(adv), eps).sum().backward()
        at_best, inside, past = r.grad.tolist()
        assert at_best == pytest.approx(0.0, abs=1e-6)
        assert inside * (best - 1.0) > 0 and past * (best - 1.0) < 0      # toward the edge, from either side


def test_the_kl_to_the_clone_and_the_cloning_loss_fade_to_a_fifth_by_iteration_100_and_to_nothing_by_500():
    for fade in (pt.Config.kl_fade, pt.Config.clone_fade):
        assert [pt.fade_share(i, fade) for i in (0, 50, 100, 300, 500, 2000)] == pytest.approx([1.0, 0.6, 0.2, 0.1, 0.0, 0.0])


def test_a_lead_can_end_nine_in_ten_training_games_at_first_and_none_from_iteration_100(learner):
    assert [pt.fade_share(i, pt.Config.lead_fade) for i in (0, 50, 100, 400)] == pytest.approx([0.9, 0.45, 0.0, 0.0])
    held = learner.iteration
    try:
        learner.iteration = 25
        assert learner.lead_share == pytest.approx(0.675)
    finally:
        learner.iteration = held


# A run of the actor on latents ---------------------------------------------------------------------------------------------------------------------------------------


def _config(encoder_path, **kw) -> pt.Config:
    base = dict(decisions=120, iteration=60, games=2, passes=1, segment=4, token_budget=25_000, data_workers=0, eval_every=10**6, checkpoint_every=10**6, latent_encoder=str(encoder_path))
    return pt.Config(**{**base, **kw})


def test_a_short_run_trains_leaves_the_encoder_alone_and_goes_on(tmp_path, encoder_path):
    # Not concurrent, and Muon on the critic: as policy_pipeline runs it.  (Muon on the actor's matrices takes minutes on the CPU.)
    cfg = _config(encoder_path, concurrent=False, decisions=60, iteration=30, critic_muon_lr=0.004)
    pt.train(cfg, tmp_path, pt.make_sim, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and all(r["masked_out"] == 0 for r in rows)
    marker = json.loads((tmp_path / "latest.json").read_text())
    saved = torch.load(tmp_path / marker["checkpoint"], weights_only=True)
    original = torch.load(encoder_path, weights_only=True)["model"]
    assert torch.equal(saved["actor"]["encoder.encoder.queries"], original["encoder.queries"])
    assert not torch.equal(saved["actor"]["latent_in.weight"], pl.make(VOCABULARY, encoder_path).state_dict()["latent_in.weight"])
    pt.train(replace(cfg, decisions=90), tmp_path, pt.make_sim, device="cpu")
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) > len(rows)


def test_the_optimisers_hold_only_what_trains_each_once_with_the_hidden_matrices_in_muon(tmp_path, encoder_path):
    cfg = _config(encoder_path, critic_muon_lr=0.004, actor_muon_lr=1.3e-4)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [sim_frames(1)[1]], cfg.enabled_kinds), "cpu")
    adam = [id(p) for group in learner.optimizer.param_groups for p in group["params"]]
    muon = [id(p) for group in learner.muon.param_groups for p in group["params"]]
    assert sorted(adam + muon) == sorted(id(p) for p in learner.trained)
    assert not set(adam + muon) & {id(p) for p in learner.actor.encoder.parameters()}
    assert id(learner.critic.value.weight) in adam and id(learner.critic.queries) in adam and id(learner.actor.head_kind.weight) in adam
    assert id(learner.critic.reads[0].kv.weight) in muon and id(learner.critic.unit_in.weight) in muon and id(learner.actor.latent_in.weight) in muon


def test_the_critic_learns_at_its_own_rate_falling_as_the_actors_does(tmp_path, encoder_path):
    cfg = replace(_config(encoder_path), lr_start=1e-5, lr_end=3e-6, critic_lr=3e-4, critic_muon_lr=0.004, actor_muon_lr=1.3e-4,
                  critic_warmup=2, decisions=1000)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [sim_frames(1)[1]], cfg.enabled_kinds), "cpu")
    actor, critic = learner.optimizer.param_groups
    muon = {g["role"]: g for g in learner.muon.param_groups}
    learner.set_rates()      # warming up: the critic alone
    assert actor["lr"] == 0.0 == muon["actor"]["lr"] and critic["lr"] == pytest.approx(3e-4) and muon["critic"]["lr"] == pytest.approx(0.004)
    learner.iteration, learner.decisions = 2, 500
    learner.set_rates()
    assert actor["lr"] == pytest.approx(6.5e-6) and critic["lr"] == pytest.approx(3e-4 * 0.65)
    assert muon["actor"]["lr"] == pytest.approx(1.3e-4 * 0.65) and muon["critic"]["lr"] == pytest.approx(0.004 * 0.65)
    plain = pt.Learner(replace(cfg, critic_lr=0.0, critic_muon_lr=0.0, actor_muon_lr=0.0, critic_warmup=0), tmp_path, learner.workers, "cpu")
    plain.set_rates()
    assert [g["lr"] for g in plain.optimizer.param_groups] == pytest.approx([1e-5, 1e-5])


def test_a_frozen_player_copies_the_actor_and_its_encoder(tmp_path, encoder_path):
    cfg = _config(encoder_path)
    workers = pt.DataWorkers(0, [sim_frames(1)[1]], cfg.enabled_kinds)
    learner = pt.Learner(cfg, tmp_path, workers, "cpu")
    player = learner.frozen_player(workers)
    assert player.actor is not learner.actor
    with torch.no_grad():
        learner.actor.latent_in.weight.add_(1.0)
    assert not torch.equal(player.actor.latent_in.weight, learner.actor.latent_in.weight)
    learner.copy_weights_to(player)
    assert torch.equal(player.actor.latent_in.weight, learner.actor.latent_in.weight)


def test_segments_score_in_training_as_their_decisions_did_in_play(tmp_path, encoder_path):
    """All of it: played through the trainer's own path and cut into segments, mid-game ones starting from the frames the seat had kept, then
    featurized again and read, every segment in one forward pass."""
    cfg = _config(encoder_path)
    actor = pl.make(VOCABULARY, encoder_path)
    frames, bundle = sim_frames(actor.reach + 12)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [bundle], cfg.enabled_kinds), "cpu")
    names = np.arange(VOCABULARY, dtype=np.int32)
    seats = pt.Seats(1, "cpu", learner.actor)
    tr = pt.Trajectory(bundle, names)
    tr.history, tr.seen = pt._kept(seats.history[0], seats.seen[0])
    segments = []
    for t, frame in enumerate(frames):
        items, acts, value, seats.history, seats.seen = learner.act([(frame, bundle, names)], seats.history, seats.seen)
        c = items[0].controllable
        tr.steps.append(pt.Step(frame, *(acts[h][0] for h in pt.act.SLOT_FIELDS + ("logp",)), float(value[0]), latents=acts["latents"][0]))
        if t in (6, learner.actor.reach + 4):
            pt._cut(tr, segments, seats.history[0], seats.seen[0])
    pt._close(tr, segments)
    assert [len(s.steps) for s in segments] == [7, learner.actor.reach - 2, 7]
    assert float(segments[2].seen.sum()) == learner.actor.reach and float(segments[0].seen.sum()) == 0.0
    batch = segments[::-1]
    steps = [s for seg in batch for s in seg.steps]
    chunks = [[(s.frame, bundle, names) for s in steps]]
    whole = pm.join(list(learner.workers.batches(chunks, bundle.regions))).to("cpu")
    lengths = [len(seg.steps) for seg in batch]
    latents = torch.zeros(len(batch), max(lengths), *steps[0].latents.shape)
    for i, seg in enumerate(batch):
        latents[i, :len(seg.steps)] = torch.as_tensor(np.stack([s.latents for s in seg.steps]))
    history = torch.as_tensor(np.stack([seg.history for seg in batch]))
    seen = torch.as_tensor(np.stack([seg.seen for seg in batch]))
    with torch.no_grad():
        a = pt._stored_actions(steps, "cpu")
        acted = learner.actor.segment_actions(whole, a, lengths)
        ctx = learner.actor.context(whole, learner.actor.segments(history, seen, latents, lengths, acted))
        logp, _ = pm.evaluate_actions(learner.actor, whole, ctx, a)
    for k, s in enumerate(steps):
        assert np.allclose(logp[k, :len(s.logp)].numpy(), s.logp, atol=2e-3), k


def test_a_game_that_ends_forgets_its_frames(encoder_path):
    actor = pl.make(VOCABULARY, encoder_path)
    seats = pt.Seats(2, "cpu", actor)
    seats.history += 1.0
    seats.seen += 1.0
    seats.clear(0)
    assert float(seats.history[0].abs().sum()) == 0.0 and float(seats.seen[0].sum()) == 0.0 and float(seats.seen[1].sum()) > 0.0
