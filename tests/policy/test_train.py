"""The trainer: minibatches within their budget, playing many games in pieces, what its games are, the card's memory shared
between runs, and a short run of the actor on latents (tests/policy/test_latent_actor.py has the actor itself)."""
from __future__ import annotations

import functools
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
    """Each run's loss weighed by its share of the decisions, the gradient is the whole minibatch's; and the budget shrinks, once
    in an iteration however many times it runs out, as the iteration's minibatches were planned before.  The next minibatch's
    gradient adds to it."""
    weight = learner.trained[0]
    segments = [pt.Segment([None] * n, None, None, None, np.full(1, v)) for n, v in ((3, 1.0), (5, 2.0), (8, -1.0))]

    def loss(run, eps, note):
        if len(run) > 1:
            raise torch.OutOfMemoryError("too big")
        note("tried", 1.0)
        return weight.sum() * sum(len(s.steps) * float(s.names[0]) for s in run) / sum(len(s.steps) for s in run)
    monkeypatch.setattr(learner, "_loss", loss)
    budget, notes = learner.token_budget, []
    learner.shrunk = False
    try:
        learner._backward(segments, 0.2, lambda *a: notes.append(a))     # runs out in 1 piece, then in 2, then fits in 3
        assert learner.token_budget == int(budget * pt.SPLIT_SHRINK)
    finally:
        learner.token_budget = budget
    assert torch.allclose(weight.grad, torch.full_like(weight, (3 * 1.0 + 5 * 2.0 - 8 * 1.0) / 16))
    assert notes == [("tried", 1.0)] * 3 + [("minibatch_decisions", 16)]
    learner._backward(segments[1:2], 0.2, lambda *a: None, 0.5)
    assert torch.allclose(weight.grad, torch.full_like(weight, (3 * 1.0 + 5 * 2.0 - 8 * 1.0) / 16 + 0.5 * 2.0))


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


def test_a_graphs_pass_takes_at_most_the_rows_it_is_given():
    """Against itself the policy has every seat of a game: a collector's passes stay a seat a game wide, as its memory is sized."""
    assert pt._act_slices([None] * 5, graphed=True) == [slice(0, 5)]
    assert pt._act_slices([None] * 5, graphed=True, rows=2) == [slice(0, 2), slice(2, 4), slice(4, 6)]


def test_training_games_are_against_itself_by_a_fixed_share_and_of_other_kinds_than_1v1s_by_the_iteration(learner):
    held = learner.cfg, learner.iteration
    try:
        assert learner.games["self_play_share"] == 0.0 and learner.games["other_kinds_share"] is None, "the kinds' own weights"
        learner.cfg = replace(learner.cfg, self_play=0.9, kinds_ramp=((500, 0.0), (2500, 0.5)))
        shares = []
        for learner.iteration in (0, 500, 1500, 2500, 4000):
            shares.append(learner.games["other_kinds_share"])
        assert shares == pytest.approx([0.0, 0.0, 0.25, 0.5, 0.5]) and learner.games["self_play_share"] == 0.9
    finally:
        learner.cfg, learner.iteration = held


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


def test_each_term_has_its_own_gamma_and_the_advantage_weighs_the_terms_over_their_returns_spread():
    """finish: GAE term by term, at lambda 1 the discounted return less the value.  weigh: a term whose returns spread over 1
    counts for its weight over that spread, one that spreads less for its weight."""
    steps = [pt.Step(*([None] * 12), value=np.zeros(2), reward=np.array(r)) for r in ((4.0, 1.0), (0.0, 1.0), (0.0, 1.0))]
    values, returns = pt.finish([pt.Trajectory(None, None, steps, done=True)], 1.0, np.array([1.0, 0.5]))
    assert values.shape == returns.shape == (3, 2)
    assert [s.ret.tolist() for s in steps] == [s.advantages.tolist() for s in steps] == [[4.0, 1.75], [0.0, 1.5], [0.0, 1.0]]
    scales = pt.Learner.weigh(SimpleNamespace(term_weights=np.array([1.0, 0.5])), steps)
    assert scales.tolist() == pytest.approx([3.6, 1.0])
    assert [s.advantage for s in steps] == pytest.approx([4.0 / 3.6 + 0.875, 0.75, 0.5])


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


def test_a_short_run_trains_leaves_the_encoder_alone_and_goes_on(tmp_path, encoder_path, monkeypatch):
    # Not concurrent, and Muon on the critic: as policy_pipeline runs it.  A small critic, as Muon on the CPU took 89 s of the
    # full one's 110 (2026-11-12), and on the critic only, as on the actor's matrices it takes minutes.
    monkeypatch.setattr(pm, "Critic", functools.partial(pm.Critic, d=64, layers=1, queries=4))
    cfg = _config(encoder_path, concurrent=False, decisions=60, iteration=30, critic_muon_lr=0.004)
    pt.train(cfg, tmp_path, pt.make_sim, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and all(r["masked_out"] == 0 for r in rows)
    marker = json.loads((tmp_path / "latest.json").read_text())
    saved = torch.load(tmp_path / marker["checkpoint"], weights_only=True)
    original = torch.load(encoder_path, weights_only=True)["model"]
    assert torch.equal(saved["actor"]["encoder.encoder.queries"], original["encoder.queries"])
    assert not torch.equal(saved["actor"]["latent_in.weight"], pl.make(VOCABULARY, encoder_path).state_dict()["latent_in.weight"])
    # Going on with the reward's terms apart: the critic's heads start afresh and learn alone, the actor waiting for them.
    terms = (("result", 1.0), ("score", 0.5), ("economy", 0.3))
    pt.train(replace(cfg, decisions=90, terms=terms), tmp_path, pt.make_sim, device="cpu")
    more = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()][len(rows):]
    assert more and all("explained_variance_economy" in r and "return_scale_score" in r for r in more)
    marker = json.loads((tmp_path / "latest.json").read_text())
    later = torch.load(tmp_path / marker["checkpoint"], weights_only=True)
    assert later["terms"] == ["result", "score", "economy"] and later["warm_until"] == saved["iteration"] + cfg.critic_rewarm
    assert later["critic"]["value.weight"].shape[0] == 3 * max(cfg.critic_bins, 1)
    assert torch.equal(later["actor"]["latent_in.weight"], saved["actor"]["latent_in.weight"])


def test_the_optimisers_hold_only_what_trains_each_once_with_the_hidden_matrices_in_muon(tmp_path, encoder_path):
    cfg = _config(encoder_path, critic_muon_lr=0.004, actor_muon_lr=1.3e-4)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [sim_frames(1)[1]], cfg.enabled_kinds), "cpu")
    adam = [id(p) for group in learner.optimizer.param_groups for p in group["params"]]
    muon = [id(p) for group in learner.muon.param_groups for p in group["params"]]
    assert sorted(adam + muon) == sorted(id(p) for p in learner.trained)
    assert not set(adam + muon) & {id(p) for p in learner.actor.encoder.parameters()}
    assert id(learner.critic.value.weight) in adam and id(learner.critic.queries) in adam and id(learner.actor.head_kind.weight) in adam
    assert id(learner.critic.reads[0].kv.weight) in muon and id(learner.critic.unit_in.weight) in muon and id(learner.actor.latent_in.weight) in muon


def test_the_critic_learns_at_its_own_rate_falling_as_the_actors_does_which_also_follows_the_kl(tmp_path, encoder_path):
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
    learner.follow_kl(0.05)     # past twice the target: the actor slows, the critic doesn't
    learner.set_rates()
    assert actor["lr"] == pytest.approx(6.5e-6 / pt.KL_SLOW) and muon["actor"]["lr"] == pytest.approx(1.3e-4 * 0.65 / pt.KL_SLOW)
    assert critic["lr"] == pytest.approx(3e-4 * 0.65)
    learner.follow_kl(0.001)
    learner.follow_kl(0.001)    # under half of it: back to the schedule, and no further
    learner.set_rates()
    assert actor["lr"] == pytest.approx(6.5e-6)
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
        tr.steps.append(pt.Step(frame, *(acts[h][0] for h in pt.act.SLOT_FIELDS + ("logp",)), value[0].copy(), latents=acts["latents"][0]))
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


def test_grpo_judges_each_branch_of_a_copied_game_against_the_other_and_trains_with_no_critic(tmp_path, encoder_path, monkeypatch):
    """A group of two games: each stretch's two branches start from the same frame and what the actor kept.  A decision is judged
    on what its branch scored from it on, against the other branch's from the same moment, its economy on the change from it;
    an optimizer step takes whole groups; and the advantages aren't normalized again by the learner, so all of them above 0 push every action up."""
    cfg = _config(encoder_path, games=2, group=2, horizon=(5.0, 10.0), terms=(("result", 1.0), ("score", 0.5), ("economy", 0.3), ("combat", 0.3)))
    env = pt.make_sim(cfg.games, cfg.seed)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, env.bundles(), cfg.enabled_kinds), "cpu")
    assert learner.critic is None
    opened = {}
    real_open = pt.GroupCollector._open

    def spy(self, seat):
        real_open(self, seat)
        opened.setdefault((self.env.slot, seat % self.env.slots), []).append((self.env.frame(seat).units.copy(), self.current[seat].history.copy()))

    monkeypatch.setattr(pt.GroupCollector, "_open", spy)
    real_terms = env.reward_terms

    def terms(seat, names, b):      # each branch's income balance steady, at a level of its own
        return np.where(np.array(names) == "income", float(seat), real_terms(seat, names, b))

    monkeypatch.setattr(env, "reward_terms", terms)
    segments, tally = pt.GroupCollector(learner, env, np.random.default_rng(1)).collect_tally(40)
    assert tally.stored == sum(len(seg.steps) for seg in segments) >= 40 and tally.groups["branches"] >= 4
    moments: dict[tuple[int, int], list[np.ndarray]] = {}
    for seg in segments:
        for s in seg.steps:
            moments.setdefault((seg.group, s.slot), []).append(s.advantages)
    both = [v for v in moments.values() if len(v) == 2]
    assert len(both) >= 10 and all(np.allclose(a + b, 0.0) for a, b in both)
    assert any(len({tuple(s.advantages) for s in seg.steps}) > 1 for seg in segments)     # from each decision on, not the stretch
    # The economy is judged on the income balance's change from each decision on (Config.economy_change): a steady one is nothing.
    assert all(np.allclose(s.advantages[2], 0.0) for seg in segments for s in seg.steps)
    stretches = sorted(opened)
    assert len(stretches) >= 2 and all(len(opened[k]) == 2 for k in stretches)
    for (units_a, kept_a), (units_b, kept_b) in (opened[k] for k in stretches):
        assert np.array_equal(units_a, units_b) and np.array_equal(kept_a, kept_b)
    assert np.abs(opened[stretches[-1]][0][1]).sum() > 0.0          # a later stretch goes on from what the actor kept
    learner.cfg.group_step = 12
    plan = learner._plan(segments, np.random.default_rng(0))
    groups = [{seg.group for batch in step for seg in batch} for step in plan]
    assert len(plan) >= 2 and sum(map(len, groups)) == len({seg.group for seg in segments})       # no group split over steps
    for seg in segments:
        for s in seg.steps:
            s.advantages = np.abs(s.advantages)
    before = learner.actor.latent_in.weight.detach().clone()
    learner.cfg.group_step = 10**6          # one step, so every ratio is 1 and the loss is minus the advantages' mean
    metrics = learner.train(segments, np.random.default_rng(0))
    assert "loss_value" not in metrics and {f"term_scale_{name}" for name in learner.term_names} <= metrics.keys()
    assert metrics["loss_pg"] < -0.5 * np.mean([s.advantage for seg in segments for s in seg.steps]) < 0.0
    assert not torch.equal(before, learner.actor.latent_in.weight)
    learner.save()
    saved = torch.load(tmp_path / json.loads((tmp_path / "latest.json").read_text())["checkpoint"], weights_only=True)
    assert "critic" not in saved and "ema_critic" not in saved
    # With data workers in processes, each minibatch's decisions are asked of them once, the next one's before this one's loss.
    asked, before_loss = [], []

    def batches(steps, regions):
        asked.append(sum(map(len, steps)))
        return [pt._worker_batch(([(f, id(b), n) for f, b, n in step], regions), learner.workers.state) for step in steps]

    real_loss = learner._loss
    monkeypatch.setattr(learner.workers, "batches", batches)
    monkeypatch.setattr(learner.workers, "pool", object())
    monkeypatch.setattr(learner, "token_budget", 1)         # a segment a minibatch
    monkeypatch.setattr(learner, "_loss", lambda run, eps, note: (before_loss.append(len(asked)), real_loss(run, eps, note))[1])
    learner.train(segments, np.random.default_rng(0))
    assert len(segments) >= 3 and sorted(asked) == sorted(len(seg.steps) for seg in segments) and not learner.ahead
    assert before_loss[:2] == [1, 3]


def test_grpo_against_itself_judges_each_side_against_the_same_side_of_the_other_copies(tmp_path, encoder_path, monkeypatch):
    """Every game the policy's against itself from the first (Config.self_play 1): both sides of a copy decide, each side's
    branches are a group of their own, and with a zero-sum score what one side's decision gained its copy the other's lost."""
    cfg = _config(encoder_path, games=2, group=2, horizon=(5.0, 10.0), self_play=1.0, terms=(("score", 1.0),))
    env = pt.make_sim(cfg.games, cfg.seed)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, env.bundles(), cfg.enabled_kinds), "cpu")
    pt.set_games(env, learner.games)
    # Each copy's score at a level of its own, one side's what the other's isn't.
    monkeypatch.setattr(env, "reward_terms", lambda seat, names, b: np.full(len(names), (1.0 - 2.0 * env.team_of(seat)) * (1.0 + env.game_of(seat))))
    segments, tally = pt.GroupCollector(learner, env, np.random.default_rng(1)).collect_tally(40)
    assert set(env.mode) == {"mirror"} and tally.groups["branches"] >= 4
    moments: dict[int, dict[int, list[float]]] = {}
    for seg in segments:
        for s in seg.steps:
            moments.setdefault(s.slot, {}).setdefault(seg.group, []).append(float(s.advantages[0]))
    whole = [list(sides.values()) for sides in moments.values() if sorted(map(len, sides.values())) == [2, 2]]
    assert len(whole) >= 5
    for one, other in whole:
        assert sum(one) == pytest.approx(0.0) and sorted(one) == pytest.approx(sorted(-x for x in other))
    assert any(abs(x) > 0.1 for one, _ in whole for x in one)


def test_spo_bound_keeps_the_pull_back_on_a_ratio_what_it_is_at_the_edge():
    """Within bound x eps of 1 the objective is SPO's own; past it, its penalty's pull on the ratio grows no more."""
    eps, bound, adv = 0.2, 2.0, torch.tensor(-1.5, dtype=torch.float64)

    def slope(b):
        r = torch.linspace(0.0, 3.0, 301, dtype=torch.float64, requires_grad=True)
        pt.spo_objective(r, adv, eps, b).sum().backward()
        return r.detach(), r.grad
    ratio, plain = slope(0.0)
    _, bounded = slope(bound)
    near = (ratio - 1.0).abs() <= bound * eps
    assert torch.allclose(plain[near], bounded[near])
    pull = (bounded - adv).abs()
    assert torch.allclose(pull[~near], torch.full_like(pull[~near], bound * abs(float(adv)))) and (plain - adv).abs().max() > 3 * bound * abs(float(adv))


def test_tests_of_one_seed_play_the_same_games_and_pair_their_scores(tmp_path, encoder_path):
    """Each test game's record has what its score reads and what pairs it (seed and slot), and a second test of the seed plays
    the same games: here two cut short at their time limit, so drawn."""
    cfg = _config(encoder_path)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, pt.make_sim(2, 7).bundles(), cfg.enabled_kinds), "cpu")

    def short(*args, **kw):
        env = pt.make_sim(*args, **kw)
        env.time_limit = 6
        return env
    first, again = (pt.evaluate(learner, short, "fight", 2, 7)["records"] for _ in range(2))
    assert len(first) == 2 and [(r["slot"], r["map"]) for r in first] == [(r["slot"], r["map"]) for r in again]
    assert all(r["result"] == "drew" and r["seed"] == 7 and r["income_own"] > 0.0 and "damage_dealt" in r for r in first)
    m = pt.score_test("eval_fight", again, first)
    assert m["eval_fight_paired"] == 2 and m["eval_fight_result"] == 0.5 and 0.0 < m["eval_fight_score"] < 1.0


def test_bootstrapped_grpo_adds_the_critics_value_of_where_each_branch_ended_and_trains_the_critic(tmp_path, encoder_path, monkeypatch):
    """A decision's return is its branch's terms to its stretch's end and the critic's value of where the branch ended,
    discounted to it, against its group's: with that value 1 more for one branch's end, each of that branch's decisions gains
    its discount x (1 - 1 / branches), and the group's other branch's lose the same moment's discount / branches.  The critic
    learns the returns; it plays nothing."""
    monkeypatch.setattr(pm, "Critic", functools.partial(pm.Critic, d=64, layers=1, queries=4))
    cfg = _config(encoder_path, games=4, group=2, horizon=(5.0, 10.0), bootstrap=True, critic_bins=51, critic_span=2.0,
                  terms=(("result", 1.0), ("score", 0.5), ("economy", 0.3), ("combat", 0.3)))
    env = pt.make_sim(cfg.games, cfg.seed)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, env.bundles(), cfg.enabled_kinds), "cpu")
    assert learner.critic is not None and learner.player.critic is None
    segments, _ = pt.GroupCollector(learner, env, np.random.default_rng(1)).collect_tally(40)
    learner.prepare(segments)
    before = {id(s): s.advantages.copy() for seg in segments for s in seg.steps}
    raised = next(seg for seg in segments if seg.end is not None)
    assert len({seg.group for seg in segments}) >= 2
    real = learner.values_of
    monkeypatch.setattr(learner, "values_of", lambda frames: [v + (f is raised.end) for (f, _, _), v in zip(frames, real(frames))])
    learner.prepare(segments)
    at = {s.slot: s.discount for seg in segments if seg.branch == raised.branch for s in seg.steps}
    for seg in segments:
        for s in seg.steps:
            if seg.group != raised.group:
                want = 0.0
            elif seg.branch == raised.branch:
                want = s.discount * (1.0 - 1.0 / seg.members)
            else:
                want = -at[s.slot] / seg.members
            assert np.allclose(s.advantages - before[id(s)], want)
    critic = learner.critic.value.weight.detach().clone()
    metrics = learner.train(segments, np.random.default_rng(0))
    assert metrics["loss_value"] > 0.0 and "explained_variance_economy" in metrics and not torch.equal(critic, learner.critic.value.weight)


def test_resignation_plays_games_out_to_measure_then_gives_up_those_under_the_threshold(tmp_path, encoder_path, monkeypatch):
    """As AlphaGo Zero: with no threshold every game is played out, saying the least result the critic gave it where its group
    went on and whether it was then lost; with one, a game the critic puts under it is given up there.  The threshold is the
    highest under which enough played-out games fell and few enough of them weren't lost."""
    monkeypatch.setattr(pm, "Critic", functools.partial(pm.Critic, d=64, layers=1, queries=4))     # fresh: its result is 0 everywhere
    cfg = _config(encoder_path, games=4, group=2, horizon=(5.0, 5.0), bootstrap=True, resign=True, resign_free=0.0, critic_bins=51,
                  critic_span=2.0, terms=(("result", 1.0), ("score", 0.5)))
    env = pt.make_sim(cfg.games, cfg.seed)
    env.time_limit = 20
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, env.bundles(), cfg.enabled_kinds), "cpu")
    collector = pt.GroupCollector(learner, env, np.random.default_rng(1))
    collector.measuring = True
    played, resigned = [], 0
    for _ in range(8):
        tally = collector.collect_tally(20)[1]
        played, resigned = played + tally.lineages, resigned + tally.groups.get("resigned", 0)
        if played:
            break
    assert played and resigned == 0 and all(abs(low) < 1e-6 and kept == 1.0 for low, kept in played)     # drawn at the limit
    collector.resign_at = 0.5
    for _ in range(8):
        resigned += collector.collect_tally(20)[1].groups.get("resigned", 0)
        if resigned:
            break
    assert resigned
    learner.follow_resignation([(-1.0, 0.0)] * 20 + [(-0.65, 1.0)] * 2)
    assert learner.resign_at == -0.7
