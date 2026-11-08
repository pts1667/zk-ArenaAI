"""The actor on latents: what it reads, what stays fixed, and that the trainer can run it as it runs the first actor."""
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
    import frame_jepa as fj
    import frame_vae as fv
    import policy_latent as pl
    import policy_model as pm
    import policy_toy as toy
    import policy_train as pt
    from zkbot.policy import featurize as fz, net as npnet, rawframe as rf, tables
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

ENCODER = fj.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=0, region_pool=6)
VAE = fv.VaeConfig(latent=8, decoder_d=32, decoder_heads=2, decoder_rounds=1, variational=False, two_part=True, questions=True)
TABLE = tables.table()
VOCABULARY = len(TABLE.names) + 1


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory) -> pathlib.Path:
    torch.manual_seed(1)
    path = tmp_path_factory.mktemp("encoder") / "vae.pt"
    fv.save_vae(path, fv.FrameVae(VAE, ENCODER, TABLE), 1)
    return path


@pytest.fixture()
def actor(encoder_path) -> pl.LatentActor:
    torch.manual_seed(2)
    return pl.make(VOCABULARY, encoder_path)


def _items(count: int = 3, seed: int = 5) -> list[fz.Tensors]:
    bundle = toy.plain()
    rng = np.random.default_rng(seed)
    game = toy.ToyGame(rng)
    out = []
    for _ in range(count):
        out.append(fz.featurize(game.frame_for(0), bundle, TABLE, enabled_kinds=fz.STAGE2_KINDS, privileged=True))
        game.apply(1, toy.script_focus(game, 1, rng))
        game.step()
    return out


def _batch(items: list[fz.Tensors]) -> pm.Batch:
    return pm.collate(items, regions=toy.plain().regions, pairs=False).to("cpu")


def _state(actor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return actor.initial_state(count, toy.plain().regions, "cpu")


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


def test_each_layer_looks_back_eight_seconds_at_any_decision_rate(actor):
    assert pl.window_frames(1) == 17 and pl.window_frames(2) == 9
    assert actor.reach == 2 * 16


def test_a_frame_reads_nothing_after_it(actor):
    frames, present = _frames(actor, 8), torch.ones(1, 8, dtype=torch.bool)
    changed = frames.clone()
    changed[:, 5:] += 1.0
    first, second = actor.sequence(frames, present), actor.sequence(changed, present)
    assert torch.allclose(first[:, :5], second[:, :5], atol=1e-5)
    assert not torch.allclose(first[:, 5:], second[:, 5:], atol=1e-3)


def test_a_frame_reads_back_as_far_as_its_reach_and_no_further(actor):
    count = actor.reach + 3
    frames, present = _frames(actor, count), torch.ones(1, count, dtype=torch.bool)
    newest = actor.sequence(frames, present)[:, -1]
    for back, matters in ((count - 1 - actor.reach, True), (count - 2 - actor.reach, False), (0, False), (count - 2, True)):
        changed = frames.clone()
        changed[:, back] += 1.0
        same = torch.allclose(newest, actor.sequence(changed, present)[:, -1], atol=1e-5)
        assert same != matters, back


def test_frames_from_before_the_game_read_alike_whatever_they_hold(actor):
    frames = _frames(actor, 6)
    present = torch.tensor([[False, False, True, True, True, True]])
    changed = frames.clone()
    changed[:, :2] = 7.0
    assert torch.allclose(actor.sequence(frames, present), actor.sequence(changed, present), atol=1e-5)


def test_the_newest_frame_alone_reads_as_in_the_whole_sequence(actor):
    frames, present = _frames(actor, 10), torch.ones(1, 10, dtype=torch.bool)
    assert torch.allclose(actor.sequence(frames, present)[:, -1], actor.sequence(frames, present, newest_only=True)[:, -1], atol=1e-5)


def test_the_age_bias_is_learned_and_nothing_outside_the_window_is_read(actor):
    block = actor.blocks[0]
    frame = torch.arange(20)
    bias = block.bias(frame, frame)
    assert bias.shape == (pl.HEADS, 20, 20)
    assert torch.isinf(bias[:, 3, 4]).all() and torch.isinf(bias[:, 19, 19 - actor.window]).all()
    assert torch.isfinite(bias[:, 19, 19 - actor.window + 1]).all()
    frames, present = _frames(actor, 4), torch.ones(1, 4, dtype=torch.bool)
    actor.sequence(frames, present)[:, -1].sum().backward()
    assert float(block.age.grad.abs().sum()) > 0.0


def test_a_seat_keeps_its_frames_newest_first_and_forgets_the_oldest(actor):
    b = _batch(_items(1))
    history, seen = _state(actor, 1)
    marks = iter(range(1, 100))
    actor.encoder.forward = lambda _: torch.full((1, actor.encoder.tokens, actor.encoder.latent), float(next(marks)))
    for _ in range(actor.reach + 5):
        _, _, history, seen, keep = actor.decide(b, history, seen)
    assert keep["latents"].shape == (1, actor.encoder.tokens, actor.encoder.latent)
    assert [float(history[0, k, 0, 0]) for k in (0, 1, actor.reach - 1)] == [actor.reach + 5.0, actor.reach + 4.0, 6.0]
    assert float(seen.sum()) == actor.reach


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
    now = actor.sequence(actor.encoder(b).float()[:, None], torch.ones(1, 1, dtype=torch.bool))[:, -1]
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


def test_the_first_actor_keeps_its_memory_and_chooses_as_it_did():
    torch.manual_seed(2)
    first = pm.Actor(VOCABULARY)
    b = pm.collate(_items(2), regions=toy.plain().regions).to("cpu")
    glob, regions = first.initial_state(2, toy.plain().regions, "cpu")
    units, reg_h, new_glob, new_regions, keep = first.decide(b, glob, regions)
    a = pm.sample_actions(first, b, units, reg_h)
    logp, _ = pm.evaluate_actions(first, b, units, reg_h, a)
    assert torch.allclose(logp, a.logp, atol=1e-3)
    assert first.recurrent and first.uses_pairs and b.bucket is not None and keep == {}
    assert new_glob.shape == glob.shape and new_regions.shape == regions.shape


def test_without_pairs_no_bucket_is_built():
    assert pm.collate(_items(1), regions=toy.plain().regions, pairs=False).bucket is None


# The size ------------------------------------------------------------------------------------------------------------------------------------------


def test_the_parts_add_up_and_it_is_smaller_than_the_first_actor(actor):
    report = pl.parameter_report(actor)
    assert report["total"] == sum(p.numel() for p in actor.parameters())
    assert report["trained"] < sum(p.numel() for p in pm.Actor(VOCABULARY).parameters()) / 2
    assert not any(name.startswith(("gru", "mem_in", "reg_mem", "read_region")) for name, _ in actor.named_parameters())


# The trainer ---------------------------------------------------------------------------------------------------------------------------------------


def _config(encoder_path, **kw) -> pt.Config:
    base = dict(decisions=120, iteration=60, games=2, passes=1, segment=4, token_budget=4000, data_workers=0, eval_every=10**6, checkpoint_every=10**6, latent_encoder=str(encoder_path))
    return pt.Config(**{**base, **kw})


@pytest.mark.parametrize("concurrent", (False, True))
def test_a_short_run_trains_leaves_the_encoder_alone_and_goes_on(tmp_path, encoder_path, concurrent):
    cfg = _config(encoder_path, concurrent=concurrent)
    pt.train(cfg, tmp_path, pt.make_toy, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and all(r["masked_out"] == 0 for r in rows)
    marker = json.loads((tmp_path / "latest.json").read_text())
    saved = torch.load(tmp_path / marker["checkpoint"], weights_only=True)
    original = torch.load(encoder_path, weights_only=True)["model"]
    assert torch.equal(saved["actor"]["encoder.encoder.queries"], original["encoder.queries"])
    assert not torch.equal(saved["actor"]["latent_in.weight"], pl.make(VOCABULARY, encoder_path).state_dict()["latent_in.weight"])
    pt.train(replace(cfg, decisions=180), tmp_path, pt.make_toy, device="cpu")
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) > len(rows)


def test_the_optimiser_holds_only_what_trains(tmp_path, encoder_path):
    cfg = _config(encoder_path)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [toy.plain()], cfg.enabled_kinds, False), "cpu")
    held = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    assert not held & {id(p) for p in learner.actor.encoder.parameters()}
    assert id(learner.actor.latent_in.weight) in held and id(learner.critic.value.weight) in held


def test_a_frozen_player_copies_the_actor_and_its_encoder(tmp_path, encoder_path):
    cfg = _config(encoder_path)
    workers = pt.DataWorkers(0, [toy.plain()], cfg.enabled_kinds, False)
    learner = pt.Learner(cfg, tmp_path, workers, "cpu")
    player = learner.frozen_player(workers)
    assert player.actor is not learner.actor and player.actor.uses_pairs is False
    with torch.no_grad():
        learner.actor.latent_in.weight.add_(1.0)
    assert not torch.equal(player.actor.latent_in.weight, learner.actor.latent_in.weight)
    learner.copy_weights_to(player)
    assert torch.equal(player.actor.latent_in.weight, learner.actor.latent_in.weight)


def test_segments_score_in_training_as_their_decisions_did_in_play(tmp_path, encoder_path):
    """All of it: played through the trainer's own path and cut into segments, mid-game ones starting from the frames the seat had kept, then
    featurized again and read, every segment in one forward pass."""
    cfg = _config(encoder_path)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, [toy.plain()], cfg.enabled_kinds, False), "cpu")
    bundle = toy.plain()
    names = np.arange(VOCABULARY, dtype=np.int32)
    rng = np.random.default_rng(5)
    game = toy.ToyGame(rng)
    seats = pt.Seats(1, bundle.regions, "cpu", learner.actor)
    tr = pt.Trajectory(bundle, names)
    tr.glob, tr.regions = pt._kept(seats.glob[0], seats.regions[0], bundle, False)
    segments = []
    for t in range(learner.actor.reach + 12):
        frame = game.frame_for(0)
        items, acts, value, seats.glob, seats.regions = learner.act([(frame, bundle, names)], seats.glob, seats.regions)
        c = items[0].controllable
        tr.steps.append(pt.Step(frame, *(acts[h][0, :c] for h in ("kind", "enemy", "region", "fine", "produce", "logp")), float(value[0]), latents=acts["latents"][0]))
        if t in (6, learner.actor.reach + 4):
            pt._cut(tr, segments, seats.glob[0], seats.regions[0], False)
        game.apply(1, toy.script_focus(game, 1, rng))
        game.step()
    pt._close(tr, segments)
    assert [len(s.steps) for s in segments] == [7, learner.actor.reach - 2, 7]
    assert float(segments[2].regions.sum()) == learner.actor.reach and float(segments[0].regions.sum()) == 0.0
    batch = segments[::-1]
    steps = [s for seg in batch for s in seg.steps]
    chunks = [[(s.frame, bundle, names) for s in steps]]
    whole = pm.join(list(learner.workers.batches(chunks, bundle.regions))).to("cpu")
    lengths = [len(seg.steps) for seg in batch]
    latents = torch.zeros(len(batch), max(lengths), *steps[0].latents.shape)
    for i, seg in enumerate(batch):
        latents[i, :len(seg.steps)] = torch.as_tensor(np.stack([s.latents for s in seg.steps]))
    history = torch.as_tensor(np.stack([seg.glob for seg in batch]))
    seen = torch.as_tensor(np.stack([seg.regions for seg in batch]))
    with torch.no_grad():
        units, reg_h = learner.actor.read(whole, learner.actor.segments(history, seen, latents, lengths))
        logp, _ = pm.evaluate_actions(learner.actor, whole, units, reg_h, pt._stored_actions(steps, whole, "cpu"))
    for k, s in enumerate(steps):
        assert np.allclose(logp[k, :len(s.logp)].numpy(), s.logp, atol=2e-3), k


def test_a_game_that_ends_forgets_its_frames(encoder_path):
    actor = pl.make(VOCABULARY, encoder_path)
    seats = pt.Seats(2, toy.plain().regions, "cpu", actor)
    seats.glob += 1.0
    seats.regions += 1.0
    seats.clear(0)
    assert float(seats.glob[0].abs().sum()) == 0.0 and float(seats.regions[0].sum()) == 0.0 and float(seats.regions[1].sum()) > 0.0
