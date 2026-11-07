"""The trainer's shortcuts must not change what is learned: heads and critic once per minibatch, the sampled pair gradient, the
frozen player, and collecting during training."""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
import policy_model as pm  # noqa: E402
import policy_toy as toy  # noqa: E402
import policy_toy_env  # noqa: E402
import policy_train as pt  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402

CLIP = 0.2
LENGTHS = (5, 5, 3)     # unequal, to cover segments that end early


def _ignore(*args, **kwargs) -> None:
    pass


@pytest.fixture(scope="module")
def learner(tmp_path_factory):
    bundle = toy.plain()
    cfg = pt.Config(data_workers=0)
    torch.manual_seed(3)
    out = pt.Learner(cfg, tmp_path_factory.mktemp("run"), pt.DataWorkers(0, [bundle], cfg.enabled_kinds), device="cpu")
    with torch.no_grad():
        out.actor.pair_w.normal_(std=0.2)   # it starts at zero, where the bias does nothing
    return out


def _segments(learner) -> list[pt.Segment]:
    """Fights that go on changing size, with actions from the net, and memory left over from an earlier segment."""
    bundle = toy.plain()
    names = np.arange(learner.vocabulary, dtype=np.int32)
    rng = np.random.default_rng(5)
    state = pm.initial_state(1, bundle.regions, "cpu")
    out = []
    for length in LENGTHS:
        game = toy.ToyGame(rng)
        steps = []
        glob, regions = state
        first = glob[0].numpy().astype(np.float16), regions[0].numpy().astype(np.float16)
        for _ in range(length):
            frame = game.frame_for(0)
            items, acts, value, glob, regions = learner.act([(frame, bundle, names)], glob, regions)
            c = items[0].controllable
            steps.append(pt.Step(frame, *(acts[h][0, :c] for h in ("kind", "enemy", "region", "fine", "logp")), float(value[0]),
                                 advantage=float(rng.normal()), ret=float(rng.normal())))
            game.apply(1, toy.script_focus(game, 1, rng))
            game.step()
            game.hp[rng.integers(len(game.hp))] = 0.0      # so that frames differ in size within a segment
        out.append(pt.Segment(steps, first[0], first[1], bundle, names))
        state = glob, regions
    return out


def _per_step_loss(learner, segments) -> torch.Tensor:
    """The loss as the trainer first computed it: everything once per step, written out as the reference."""
    bundle = toy.plain()
    all_adv = np.array([s.advantage for seg in segments for s in seg.steps], np.float32)
    mean, std = float(all_adv.mean()), float(all_adv.std()) + 1e-6
    pg = value_sum = torch.zeros(())
    ent = {h: torch.zeros(()) for h in pt.HEADS}
    ent_n = {h: 0.0 for h in pt.HEADS}
    glob = torch.as_tensor(np.stack([s.glob for s in segments]), dtype=torch.float32)
    regions = torch.as_tensor(np.stack([s.regions for s in segments]), dtype=torch.float32)
    decisions = 0
    for t in range(max(LENGTHS)):
        live = [i for i, seg in enumerate(segments) if t < len(seg.steps)]
        steps = [segments[i].steps[t] for i in live]
        b = next(iter(learner.workers.batches([[(s.frame, bundle, segments[0].names) for s in steps]], bundle.regions))).to("cpu")
        index = torch.as_tensor(live)
        a = pt._stored_actions(steps, b, "cpu")
        _, units, reg_h, new_glob, new_regions = learner.actor.encode(b, glob[index], regions[index])
        logp, entropy = pm.evaluate_actions(learner.actor, b, units, reg_h, a)
        value = learner.critic(b, glob[index])
        old = pt._pad([s.logp for s in steps], b.ctrl_mask.shape, 0.0, "cpu", torch.float32)
        adv = torch.as_tensor([(s.advantage - mean) / std for s in steps], dtype=torch.float32)
        ret = torch.as_tensor([s.ret for s in steps], dtype=torch.float32)
        weights = (b.ctrl_mask & (b.kind_mask.sum(-1) > 1)).float()
        ratio = (logp - old).clamp(-20.0, 20.0).exp()
        per_unit = -torch.min(ratio * adv[:, None], ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * adv[:, None])
        count = weights.sum(-1)
        has = count > 0
        if has.any():
            pg = pg + ((per_unit * weights).sum(-1)[has] / count[has]).sum()
        value_sum = value_sum + (value - ret).pow(2).sum()
        for h in pt.HEADS:
            e, mask = entropy[h]
            ent[h] = ent[h] + (e * mask).sum()
            ent_n[h] += float(mask.sum())
        decisions += len(live)
        glob = glob.clone().index_copy(0, index, new_glob)
        regions = regions.clone().index_copy(0, index, new_regions)
    entropy_loss = sum(learner.entropy_coef[h] * ent[h] / max(ent_n[h], 1.0) for h in pt.HEADS)
    return pg / decisions + learner.cfg.value_weight * value_sum / decisions - entropy_loss


def _gradients(learner, loss_of) -> dict[str, torch.Tensor]:
    params = dict(learner.actor.named_parameters()) | {f"critic.{k}": v for k, v in learner.critic.named_parameters()}
    learner.optimizer.zero_grad(set_to_none=True)
    loss_of().backward()
    return {k: (v.grad.clone() if v.grad is not None else torch.zeros_like(v)) for k, v in params.items()}


def test_heads_and_critic_once_per_minibatch_give_the_same_gradients(learner):
    segments = _segments(learner)
    all_steps = np.arange(max(LENGTHS))
    fast = _gradients(learner, lambda: learner._loss(segments, CLIP, _ignore, np.random.default_rng(0), trained=all_steps))
    slow = _gradients(learner, lambda: _per_step_loss(learner, segments))
    assert any(float(g.abs().max()) > 1e-4 for g in slow.values())
    for name, g in slow.items():
        assert torch.allclose(fast[name], g, rtol=1e-3, atol=1e-5), name


def test_the_sampled_pair_gradient_is_right_on_average(learner):
    segments = _segments(learner)
    steps = max(LENGTHS)
    exact = _gradients(learner, lambda: learner._loss(segments, CLIP, _ignore, np.random.default_rng(0), trained=np.arange(steps)))
    sampled = [_gradients(learner, lambda t=t: learner._loss(segments, CLIP, _ignore, np.random.default_rng(0), trained=np.array([t]))) for t in range(steps)]
    assert float(exact["pair_w"].abs().max()) > 1e-5
    assert torch.allclose(sum(g["pair_w"] for g in sampled) / steps, exact["pair_w"], rtol=1e-3, atol=1e-5)
    # Nothing else moves, and one sample alone isn't the answer: otherwise this test proves nothing.
    for name in exact:
        if name != "pair_w":
            assert torch.allclose(sampled[0][name], exact[name], rtol=1e-3, atol=1e-5), name
    assert not torch.allclose(sampled[0]["pair_w"], exact["pair_w"], rtol=1e-2, atol=1e-6)


def test_stopping_the_pair_gradient_leaves_the_output_alone(learner):
    bundle = toy.plain()
    frame = toy.ToyGame(np.random.default_rng(1)).frame_for(0)
    names = np.arange(learner.vocabulary, dtype=np.int32)
    b = next(iter(learner.workers.batches([[(frame, bundle, names)]], bundle.regions))).to("cpu")
    glob, regions = pm.initial_state(1, bundle.regions, "cpu")
    with torch.no_grad():
        base = learner.actor.encode(b, glob, regions)[0]
        for scale in (0.0, 8.0):
            assert torch.equal(learner.actor.encode(b, glob, regions, pair_grad=scale)[0], base)


def test_a_frozen_player_keeps_its_weights_until_told(learner):
    frozen = learner.frozen_player(learner.workers)
    before = frozen.actor.head_kind.weight.clone()
    with torch.no_grad():
        learner.actor.head_kind.weight.add_(1.0)
    try:
        assert torch.equal(frozen.actor.head_kind.weight, before)
        learner.copy_weights_to(frozen)
        assert torch.equal(frozen.actor.head_kind.weight, learner.actor.head_kind.weight)
    finally:
        with torch.no_grad():
            learner.actor.head_kind.weight.sub_(1.0)


def test_a_minibatch_budgets_its_padded_size(learner):
    """A few frames from a big map can't ride along with many small ones: padded, they would all cost as much as the big map's."""
    def segment(n_units, regions, length):
        frame = rf.Frame(np.zeros(n_units, rf.UNIT_DTYPE), np.zeros(0, rf.SITE_DTYPE), np.zeros(len(rf.GLOBAL_FIELDS), np.float32))
        return pt.Segment([pt.Step(frame, *(np.zeros(0) for _ in range(5)))] * length, np.zeros(1), np.zeros((regions, 1)), SimpleNamespace(regions=regions), np.zeros(1))
    segments = [segment(40, 24, 32) for _ in range(40)] + [segment(20, 76, 32) for _ in range(40)]
    budget = learner.cfg.token_budget
    learner.cfg.token_budget = 100_000
    try:
        batches = learner._minibatches(segments, np.random.default_rng(1))
    finally:
        learner.cfg.token_budget = budget
    assert sorted(id(s) for b in batches for s in b) == sorted(id(s) for s in segments)
    for b in batches:
        decisions = sum(len(s.steps) for s in b)
        padded = max(len(s.steps[0].frame.units) for s in b) + max(s.bundle.regions for s in b) + 2
        assert decisions * padded <= 100_000 or len(b) == 1


def test_big_frames_get_fewer_tokens_in_a_minibatch(learner):
    """Attention's pairs: a frame of many tokens costs more memory per token."""
    def segment(n_units):
        frame = rf.Frame(np.zeros(n_units, rf.UNIT_DTYPE), np.zeros(0, rf.SITE_DTYPE), np.zeros(len(rf.GLOBAL_FIELDS), np.float32))
        return pt.Segment([pt.Step(frame, *(np.zeros(0) for _ in range(5)))] * 32, np.zeros(1), np.zeros((76, 1)), SimpleNamespace(regions=76), np.zeros(1))
    budget = learner.cfg.token_budget
    learner.cfg.token_budget = 165_000
    try:
        size = {n: 32 * max(len(b) for b in learner._minibatches([segment(n) for _ in range(40)], np.random.default_rng(1))) for n in (80, 512)}
    finally:
        learner.cfg.token_budget = budget
    assert size[512] * (512 + 78) < 0.8 * size[80] * (80 + 78)


def test_playing_in_pieces_decides_the_same_as_in_one(learner, monkeypatch):
    bundle = toy.plain()
    names = np.arange(learner.vocabulary, dtype=np.int32)
    rng = np.random.default_rng(2)
    games = [toy.ToyGame(rng) for _ in range(5)]
    for k, game in enumerate(games):
        game.hp[:k * 2] = 0.0       # so that widths differ between the pieces
    frames = [(game.frame_for(0), bundle, names) for game in games]
    glob, regions = pm.initial_state(len(games), bundle.regions, "cpu")
    items, out, value, new_glob, new_regions = learner.act(frames, glob, regions)
    tokens = max(len(t.unit_x) for t in items) + bundle.regions + 2
    monkeypatch.setattr(pt, "ACT_PAIRS", 2 * tokens ** 2)
    assert len(pt._act_slices(items, bundle.regions)) > 1
    cut_items, cut, cut_value, cut_glob, cut_regions = learner.act(frames, glob, regions)
    assert np.allclose(cut_value, value, atol=1e-5)
    assert torch.allclose(cut_glob, new_glob, atol=1e-5)
    assert torch.allclose(cut_regions, new_regions, atol=1e-5)
    for k, t in enumerate(cut_items):
        chosen = cut["kind"][k, :t.controllable]
        assert t.kind_mask[np.arange(t.controllable), chosen].all()


@pytest.mark.parametrize("concurrent", (False, True))
def test_a_short_run_trains_and_stops(tmp_path, concurrent):
    cfg = pt.Config(decisions=120, iteration=60, games=2, passes=1, segment=4, token_budget=4000, data_workers=0, concurrent=concurrent, eval_every=10**6, checkpoint_every=10**6)
    pt.train(cfg, tmp_path, pt.make_toy, device="cpu")
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 2
    assert (tmp_path / "latest.json").exists()


def test_a_slower_policy_does_not_slow_the_script():
    rate = {}
    for every in (1, 2):
        env = policy_toy_env.ToyEnv(1, 3, ["nearest"], decide_every=every)
        env.begin()
        env.games[0].x[:8] = 0.0    # out of range, so nothing ends the game
        calls = {"n": 0}
        real = toy.SCRIPTS["nearest"]

        def counted(game, seat, rng, real=real, calls=calls):
            calls["n"] += 1
            return real(game, seat, rng)
        toy.SCRIPTS["nearest"] = counted
        try:
            frame = env.games[0].frame
            env.step(0.0)
        finally:
            toy.SCRIPTS["nearest"] = real
        rate[every] = (env.games[0].frame - frame, calls["n"])
    assert rate[1] == (15, 1)
    assert rate[2] == (30, 2)


def test_a_slower_policy_is_told_of_every_hit_since_it_last_looked():
    env = policy_toy_env.ToyEnv(1, 3, ["mirror"], decide_every=2)
    env.begin()
    game = env.games[0]
    # Side 1 shoots side 0's unit on the first slot only, then has nobody left in range.
    game.x[8:], game.z[8:] = 0.0, 0.0
    game.x[0], game.z[0] = 0.0, 100.0
    game.x[1:8], game.z[1:8] = 2000.0, 2000.0

    def leave(*args):
        game.x[0], game.z[0] = 2000.0, 2000.0

    real = game._tick
    ticks = {"n": 0}

    def tick():
        real()
        ticks["n"] += 1
        if ticks["n"] == 3:
            leave()
    game._tick = tick
    env.step(0.0)
    assert ticks["n"] == 6
    assert game.damaged[0]
    assert env.games[0].frame_for(0).units[0]["flags"] & rf.DAMAGED


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


def test_a_run_remembers_how_far_self_play_got(tmp_path):
    cfg = pt.Config(env="toy", decisions=0, games=2, iteration=4)
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
