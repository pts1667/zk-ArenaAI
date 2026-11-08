"""The actor on latents: what it reads, what stays fixed, and that the trainer runs it as it plays."""
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
    import frame_vae as fv
    import policy_latent as pl
    import policy_model as pm
    import policy_sim_env  # noqa: F401
    import policy_train as pt
    from zkbot.policy import featurize as fz, rawframe as rf, tables
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

from conftest import sim_frames  # noqa: E402

TABLE = tables.table()
VOCABULARY = len(TABLE.names) + 1


@pytest.fixture()
def actor(encoder_path) -> pl.LatentActor:
    torch.manual_seed(2)
    return pl.make(VOCABULARY, encoder_path)


def _items(count: int = 3, seed: int = 5) -> list[fz.Tensors]:
    frames, bundle = sim_frames(count, seed)
    return [fz.featurize(f, bundle, TABLE, enabled_kinds=fz.STAGE2_KINDS, privileged=True) for f in frames]


def _batch(items: list[fz.Tensors]) -> pm.Batch:
    return pm.collate(items).to("cpu")


def _state(actor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return actor.initial_state(count, 0, "cpu")


def _decided(actor, b, count: int):
    """A decision with no frames before it."""
    return actor.decide(b, *_state(actor, count))


# What it reads ---------------------------------------------------------------------------------------------------------------------------------------


def test_the_encoder_is_the_trained_ones_without_its_decoder(actor, encoder_path):
    trained = fv.load_vae(encoder_path)
    b = _batch(_items())
    assert torch.allclose(actor.encoder(b), trained.latent_tokens(b))
    names = {name for name, _ in actor.encoder.named_parameters()}
    assert not any(n.startswith(("region_out", "where_out", "at_out", "reads")) for n in names)


def test_the_encoder_does_not_train(actor):
    b = _batch(_items())
    assert all(not p.requires_grad for p in actor.encoder.parameters())
    units, reg_h, _, _, _ = _decided(actor, b, 3)
    kind, *_ = actor.heads(b, units, reg_h)
    kind.masked_fill(kind < -1e8, 0.0).sum().backward()
    assert all(p.grad is None for p in actor.encoder.parameters())
    assert float(actor.latent_in.weight.grad.abs().sum()) > 0.0 and float(actor.slots.grad.abs().sum()) > 0.0
    actor.train()
    assert not actor.encoder.training


def test_the_encoder_sees_true_names_and_the_actor_those_it_is_allowed(actor):
    b = _batch(_items(1))
    dropped = replace(b, unit_name=torch.full_like(b.unit_name, VOCABULARY - 1))
    assert torch.equal(actor.encoder(b), actor.encoder(dropped))
    assert not torch.allclose(_decided(actor, b, 1)[0], _decided(actor, dropped, 1)[0])


def test_only_the_units_that_decide_read_the_latents(actor):
    b = _batch(_items(1))
    units = _decided(actor, b, 1)[0].detach()
    c = int(b.ctrl_mask.sum())
    assert units.shape[-1] == pl.D + pl.KEY
    assert float(units[0, :c, :pl.D].abs().sum()) > 0.0
    assert float(units[0, b.ctrl_mask.shape[1]:, :pl.D].abs().sum()) == 0.0


def test_what_it_reads_changes_a_units_decision_but_not_its_key(actor):
    b = _batch(_items(1))
    now = torch.randn(1, actor.encoder.tokens, pl.D)
    before = actor.read(b, now)[0]
    after = actor.read(b, torch.randn(1, actor.encoder.tokens, pl.D))[0]
    assert not torch.allclose(before[..., :pl.D], after[..., :pl.D])
    assert torch.equal(before[..., pl.D:], after[..., pl.D:])


def test_padding_changes_no_decision(actor):
    items = _items(3)
    alone = _batch(items[:1])
    units, reg_h, *_ = _decided(actor, alone, 1)
    kind = actor.heads(alone, units, reg_h)[0]
    both = _batch(items)
    units3, reg_h3, *_ = _decided(actor, both, 3)
    kind3 = actor.heads(both, units3, reg_h3)[0]
    c = items[0].controllable
    assert torch.allclose(kind[0, :c], kind3[0, :c], atol=1e-3)


def test_the_order_of_a_batch_changes_nothing(actor):
    """As a plain transformer's: a decision doesn't depend on what else is in its batch, or on any step before it."""
    items = _items(3)
    b = _batch(items)
    forward = _decided(actor, b, 3)
    backward = _batch(items[::-1])
    units = _decided(actor, backward, 3)
    first = actor.heads(b, forward[0], forward[1])[0][0, :items[0].controllable]
    last = actor.heads(backward, units[0], units[1])[0][2, :items[0].controllable]
    assert torch.allclose(first, last, atol=1e-3)


# Frames in order -------------------------------------------------------------------------------------------------------------------------------------


def _frames(actor, count: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, count, actor.encoder.tokens, actor.encoder.latent, generator=torch.Generator().manual_seed(seed))


def _acts(count: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(1, count, pl.ACTION_TOKENS, pl.D, generator=torch.Generator().manual_seed(seed))


def test_each_layer_looks_back_eight_seconds_at_any_decision_rate(actor):
    assert pl.window_frames(1) == 17 and pl.window_frames(2) == 9
    assert actor.reach == 2 * 16


def test_a_frame_reads_nothing_after_it(actor):
    frames, present, acts = _frames(actor, 8), torch.ones(1, 8, dtype=torch.bool), _acts(8)
    changed = frames.clone()
    changed[:, 5:] += 1.0
    first, second = actor.sequence(frames, present, acts), actor.sequence(changed, present, acts)
    assert torch.allclose(first[:, :5], second[:, :5], atol=1e-5)
    assert not torch.allclose(first[:, 5:], second[:, 5:], atol=1e-3)


def test_a_frame_reads_what_was_done_before_it_but_not_what_it_chose(actor):
    """[frame t] [what was done at t] [frame t + 1] ...: a decision never sees its own choice, and the next one does."""
    frames, present, acts = _frames(actor, 8), torch.ones(1, 8, dtype=torch.bool), _acts(8)
    changed = acts.clone()
    changed[:, 5] = _acts(1, seed=9)[:, 0]          # not a constant shift, which layer norm doesn't see
    first, second = actor.sequence(frames, present, acts), actor.sequence(frames, present, changed)
    assert torch.allclose(first[:, :6], second[:, :6], atol=1e-5)
    assert not torch.allclose(first[:, 6], second[:, 6], atol=1e-3)


def test_a_frame_reads_back_as_far_as_its_reach_and_no_further(actor):
    count = actor.reach + 3
    frames, present, acts = _frames(actor, count), torch.ones(1, count, dtype=torch.bool), _acts(count)
    newest = actor.sequence(frames, present, acts)[:, -1]
    for back, matters in ((count - 1 - actor.reach, True), (count - 2 - actor.reach, False), (0, False), (count - 2, True)):
        changed = frames.clone()
        changed[:, back] += 1.0
        same = torch.allclose(newest, actor.sequence(changed, present, acts)[:, -1], atol=1e-5)
        assert same != matters, back
    # What was done reaches no further.
    changed = acts.clone()
    changed[:, count - 2 - actor.reach] = _acts(1, seed=9)[:, 0]
    assert torch.allclose(newest, actor.sequence(frames, present, changed)[:, -1], atol=1e-5)
    changed = acts.clone()
    changed[:, count - 2] = _acts(1, seed=9)[:, 0]
    assert not torch.allclose(newest, actor.sequence(frames, present, changed)[:, -1], atol=1e-5)


def test_frames_from_before_the_game_read_alike_whatever_they_hold(actor):
    frames, acts = _frames(actor, 6), _acts(6)
    present = torch.tensor([[False, False, True, True, True, True]])
    changed, done = frames.clone(), acts.clone()
    changed[:, :2] = 7.0
    done[:, :2] = 7.0
    assert torch.allclose(actor.sequence(frames, present, acts), actor.sequence(changed, present, done), atol=1e-5)


def test_the_newest_frame_alone_reads_as_in_the_whole_sequence(actor):
    frames, present, acts = _frames(actor, 10), torch.ones(1, 10, dtype=torch.bool), _acts(10)
    whole = actor.sequence(frames, present, acts)[:, -1]
    assert torch.allclose(whole, actor.sequence(frames, present, acts, newest_only=True)[:, -1], atol=1e-5)
    blank = acts.clone()
    blank[:, -1] = 0.0          # what's done at the newest isn't known yet when it decides
    assert torch.allclose(whole, actor.sequence(frames, present, blank, newest_only=True)[:, -1], atol=1e-5)


def test_the_age_bias_is_learned_and_nothing_outside_the_window_is_read(actor):
    block = actor.blocks[0]
    frame = torch.arange(20)
    plain = torch.zeros(20, dtype=torch.bool)
    bias = block.bias(frame, frame, plain, plain)
    assert bias.shape == (pl.HEADS, 20, 20)
    assert torch.isinf(bias[:, 3, 4]).all() and torch.isinf(bias[:, 19, 19 - actor.window]).all()
    assert torch.isfinite(bias[:, 19, 19 - actor.window + 1]).all()
    # A frame's own action tokens: its latent tokens don't see them, they see its latent tokens.
    two, act = torch.tensor([5, 5]), torch.tensor([False, True])
    own = block.bias(two, two, act, act)
    assert torch.isinf(own[:, 0, 1]).all() and torch.isfinite(own[:, 1, 0]).all()
    frames, present = _frames(actor, 4), torch.ones(1, 4, dtype=torch.bool)
    actor.sequence(frames, present, _acts(4))[:, -1].sum().backward()
    assert float(block.age.grad.abs().sum()) > 0.0 and float(block.act_key.grad.abs().sum()) > 0.0


def test_a_seat_keeps_its_frames_newest_first_and_forgets_the_oldest(actor):
    b = _batch(_items(1))
    history, seen = _state(actor, 1)
    marks = iter(range(1, 100))
    actor.encoder.forward = lambda _: torch.full((1, actor.encoder.tokens, actor.encoder.latent), float(next(marks)))
    for _ in range(actor.reach + 5):
        _, _, history, seen, keep = actor.decide(b, history, seen)
    assert keep["latents"].shape == (1, actor.encoder.tokens, actor.encoder.latent)
    assert history.shape == (1, actor.reach, actor.record)
    assert [float(history[0, k, 0]) for k in (0, 1, actor.reach - 1)] == [actor.reach + 5.0, actor.reach + 4.0, 6.0]
    assert float(seen.sum()) == actor.reach


def test_what_it_did_is_kept_as_the_newest_frames_action_tokens(actor):
    b = _batch(_items(1))
    units, reg_h, history, seen, _ = _decided(actor, b, 1)
    cut = actor.encoder.tokens * actor.encoder.latent
    assert float(history[0, 0, cut:].abs().sum()) == 0.0
    a = pm.sample_actions(actor, b, units, reg_h)
    done = actor.acted(b, a, history)
    assert torch.allclose(done[0, 0, cut:], actor.action_tokens(b, a)[0].flatten(), atol=1e-6)
    assert torch.equal(done[0, 0, :cut], history[0, 0, :cut]) and torch.equal(done[:, 1:], history[:, 1:])


def _keep_all(b) -> pm.Actions:
    shape = b.ctrl_mask.shape
    none = torch.full(shape, -1, dtype=torch.long)
    return pm.Actions(torch.full(shape, rf.KEEP, dtype=torch.long), none, none, none, none, torch.zeros(shape))


def test_keeping_everything_reads_alike_whoever_kept(actor):
    """Keep is nothing done: the action tokens say only that."""
    b = _batch(_items(3))
    tokens = actor.action_tokens(b, _keep_all(b))
    assert tokens.shape == (3, pl.ACTION_TOKENS, pl.D)
    assert torch.allclose(tokens[0], tokens[1], atol=1e-5) and torch.allclose(tokens[0], tokens[2], atol=1e-5)


def test_what_a_unit_was_told_and_where_reaches_the_action_tokens(actor):
    b = _batch(_items(1))
    kept = _keep_all(b)
    u = int(torch.nonzero(b.ctrl_mask[0] & b.kind_mask[0, :, rf.MOVE]).flatten()[0])
    regions = torch.nonzero(b.ptr_region[0, u]).flatten()
    moves = []
    for r in regions[:2]:
        kind, region = kept.kind.clone(), kept.region.clone()
        kind[0, u], region[0, u] = rf.MOVE, int(r)
        moves.append(actor.action_tokens(b, pm.Actions(kind, kept.enemy, region, kept.fine, kept.produce, kept.logp)))
    assert not torch.allclose(moves[0], actor.action_tokens(b, kept), atol=1e-4)
    if len(moves) == 2:
        assert not torch.allclose(moves[0], moves[1], atol=1e-5)


# Pointers ------------------------------------------------------------------------------------------------------------------------------------------


def test_a_pointer_score_from_place_depends_on_the_difference_alone(actor):
    query = torch.zeros(1, 1, pl.D)
    gain = torch.ones(1, 1, 2 * pl.BANDS)
    here, there = torch.tensor([[[0.2, -0.3]]]), torch.tensor([[[0.5, 0.1], [-0.7, 0.4]]])
    keys = lambda at: torch.cat((torch.zeros(1, 2, pl.D), actor.code(at)), -1)  # noqa: E731
    first = actor._score(query, gain, actor.code(here), keys(there))
    moved = actor._score(query, gain, actor.code(here + 0.11), keys(there + 0.11))
    assert torch.allclose(first, moved, atol=1e-4)


def test_a_pointer_score_from_place_is_highest_on_its_own_place(actor):
    query = torch.zeros(1, 1, pl.D)
    gain = torch.ones(1, 1, 2 * pl.BANDS)
    here = torch.tensor([[[0.2, -0.3]]])
    places = torch.tensor([[[0.2, -0.3], [0.21, -0.3], [0.3, -0.3], [-0.8, 0.9]]])
    keys = torch.cat((torch.zeros(1, 4, pl.D), actor.code(places)), -1)
    score = actor._score(query, gain, actor.code(here), keys)[0, 0]
    assert int(score.argmax()) == 0 and float(score[1]) > float(score[2]) > float(score[3])


def test_a_target_a_little_further_gets_another_score_whatever_the_latents_say(actor):
    items = _items(1)
    b = _batch(items)
    now = actor.sequence(actor.encoder(b).float()[:, None], torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, pl.ACTION_TOKENS, pl.D))[:, -1]
    assert len(items[0].enemy_tokens) > 0
    units, reg_h = actor.read(b, now)
    enemy = actor.heads(b, units, reg_h)[2]
    moved = b.unit_x.clone()
    moved[0, int(items[0].enemy_tokens[0]), fz._I] += 0.02
    shifted = replace(b, unit_x=moved)
    units2, reg_h2 = actor.read(shifted, now)
    legal = b.ptr_enemy[0, 0, :, 0]
    if legal.any():
        again = actor.heads(shifted, units2, reg_h2)[2]
        assert not torch.allclose(enemy[0, 0, legal, 0], again[0, 0, legal, 0], atol=1e-6)


# Choosing ------------------------------------------------------------------------------------------------------------------------------------------


def _sampled(actor, seed: int = 3):
    items = _items(3, seed)
    b = _batch(items)
    torch.manual_seed(seed)
    units, reg_h, *_ = _decided(actor, b, 3)
    return items, b, units, reg_h, pm.sample_actions(actor, b, units, reg_h)


def test_it_starts_by_mostly_keeping_what_it_is_doing(actor):
    _, b, units, _, _ = _sampled(actor)
    raw = actor.head_kind(units[:, :b.ctrl_mask.shape[1], :pl.D])[b.ctrl_mask]
    assert (raw.argmax(-1) == rf.KEEP).all()    # before the masks, as a unit with no order has nothing to keep


def test_every_choice_is_legal(actor):
    for seed in range(4):
        items, b, _, _, a = _sampled(actor, seed)
        for i, t in enumerate(items):
            c = t.controllable
            assert t.kind_mask[np.arange(c), a.kind[i, :c].numpy()].all()
            used = a.region[i, :c] >= 0
            for u in np.flatnonzero(used.numpy()):
                k = int(a.kind[i, u])
                legal = t.ptr_jump if k in (rf.JUMP, rf.THROW) else t.ptr_wait if k == rf.WAIT_AT_BEACON else t.ptr_region
                assert legal[u, int(a.region[i, u])]


def test_a_choice_scores_the_same_when_it_is_evaluated(actor):
    _, b, units, reg_h, a = _sampled(actor)
    logp, entropy = pm.evaluate_actions(actor, b, units, reg_h, a)
    assert torch.allclose(logp, a.logp, atol=1e-3)
    assert set(entropy) == {"kind", "region", "enemy", "fine", "produce"}


# The size ------------------------------------------------------------------------------------------------------------------------------------------


def test_the_parts_add_up(actor):
    report = pl.parameter_report(actor)
    assert report["total"] == sum(p.numel() for p in actor.parameters())
    assert report["trained"] == sum(p.numel() for p in actor.parameters() if p.requires_grad)


# The trainer ---------------------------------------------------------------------------------------------------------------------------------------


def _config(encoder_path, **kw) -> pt.Config:
    base = dict(decisions=120, iteration=60, games=2, passes=1, segment=4, token_budget=4000, data_workers=0, eval_every=10**6, checkpoint_every=10**6, latent_encoder=str(encoder_path))
    return pt.Config(**{**base, **kw})


@pytest.mark.parametrize("concurrent", (False, True))
def test_a_short_run_trains_leaves_the_encoder_alone_and_goes_on(tmp_path, encoder_path, concurrent):
    cfg = _config(encoder_path, concurrent=concurrent)
    pt.train(cfg, tmp_path, pt.make_sim, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and all(r["masked_out"] == 0 for r in rows)
    marker = json.loads((tmp_path / "latest.json").read_text())
    saved = torch.load(tmp_path / marker["checkpoint"], weights_only=True)
    original = torch.load(encoder_path, weights_only=True)["model"]
    assert torch.equal(saved["actor"]["encoder.encoder.queries"], original["encoder.queries"])
    assert not torch.equal(saved["actor"]["latent_in.weight"], pl.make(VOCABULARY, encoder_path).state_dict()["latent_in.weight"])
    pt.train(replace(cfg, decisions=180), tmp_path, pt.make_sim, device="cpu")
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) > len(rows)


def test_the_optimiser_holds_only_what_trains(tmp_path, encoder_path):
    cfg = _config(encoder_path)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [sim_frames(1)[1]], cfg.enabled_kinds), "cpu")
    held = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert not held & {id(p) for p in learner.actor.encoder.parameters()}
    assert id(learner.actor.latent_in.weight) in held and id(learner.critic.value.weight) in held


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
        tr.steps.append(pt.Step(frame, *(acts[h][0, :c] for h in ("kind", "enemy", "region", "fine", "produce", "logp")), float(value[0]), latents=acts["latents"][0]))
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
        a = pt._stored_actions(steps, whole, "cpu")
        acted = learner.actor.segment_actions(whole, a, lengths)
        units, reg_h = learner.actor.read(whole, learner.actor.segments(history, seen, latents, lengths, acted))
        logp, _ = pm.evaluate_actions(learner.actor, whole, units, reg_h, a)
    for k, s in enumerate(steps):
        assert np.allclose(logp[k, :len(s.logp)].numpy(), s.logp, atol=2e-3), k


def test_a_game_that_ends_forgets_its_frames(encoder_path):
    actor = pl.make(VOCABULARY, encoder_path)
    seats = pt.Seats(2, "cpu", actor)
    seats.history += 1.0
    seats.seen += 1.0
    seats.clear(0)
    assert float(seats.history[0].abs().sum()) == 0.0 and float(seats.seen[0].sum()) == 0.0 and float(seats.seen[1].sum()) > 0.0
