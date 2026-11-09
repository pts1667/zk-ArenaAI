"""The trainer's parts around the actor (tests/policy/test_latent_actor.py has the actor in training): minibatches within their
budget, playing many games in pieces, the move to self-play, and the card's memory shared between runs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import policy_train as pt
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)
from zkbot.policy import rawframe as rf  # noqa: E402

from conftest import sim_frames  # noqa: E402


@pytest.fixture(scope="module")
def learner(tmp_path_factory, encoder_path):
    cfg = pt.Config(data_workers=0, latent_encoder=str(encoder_path))
    torch.manual_seed(3)
    return pt.Learner(cfg, tmp_path_factory.mktemp("run"), pt.DataWorkers(0, [sim_frames(1)[1], sim_frames(40)[1]], cfg.enabled_kinds), device="cpu")


def test_a_minibatch_budgets_its_padded_size(learner):
    """A few frames from a big map can't ride along with many small ones: padded, they would all cost as much as the big map's."""
    def segment(n_units, regions, length):
        frame = rf.Frame(np.zeros(n_units, rf.UNIT_DTYPE), np.zeros(0, rf.SITE_DTYPE), np.zeros(len(rf.GLOBAL_FIELDS), np.float32))
        return pt.Segment([pt.Step(frame, *(np.zeros(0) for _ in range(9)))] * length, np.zeros(1), np.zeros((regions, 1)), SimpleNamespace(regions=regions), np.zeros(1))
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
    """A frame of many tokens costs more memory per token."""
    def segment(n_units):
        frame = rf.Frame(np.zeros(n_units, rf.UNIT_DTYPE), np.zeros(0, rf.SITE_DTYPE), np.zeros(len(rf.GLOBAL_FIELDS), np.float32))
        return pt.Segment([pt.Step(frame, *(np.zeros(0) for _ in range(9)))] * 32, np.zeros(1), np.zeros((76, 1)), SimpleNamespace(regions=76), np.zeros(1))
    budget = learner.cfg.token_budget
    learner.cfg.token_budget = 165_000
    try:
        size = {n: 32 * max(len(b) for b in learner._minibatches([segment(n) for _ in range(40)], np.random.default_rng(1))) for n in (80, 512)}
    finally:
        learner.cfg.token_budget = budget
    assert size[512] * (512 + 78) < 0.8 * size[80] * (80 + 78)


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
