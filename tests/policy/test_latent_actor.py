"""The actor on latents: what it reads, what stays fixed, and how it chooses.  tests/policy/test_train.py has it in training."""
from __future__ import annotations

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
    ctx, _, _, _ = _decided(actor, b, 3)
    kind, *_ = pm.each_unit_heads(actor, b, ctx)
    kind.masked_fill(kind < -1e8, 0.0).sum().backward()
    assert all(p.grad is None for p in actor.encoder.parameters())
    assert float(actor.latent_in.weight.grad.abs().sum()) > 0.0 and float(actor.slots.grad.abs().sum()) > 0.0
    actor.train()
    assert not actor.encoder.training


def test_the_encoder_sees_true_names_and_the_actor_those_it_is_allowed(actor):
    b = _batch(_items(1))
    dropped = replace(b, unit_name=torch.full_like(b.unit_name, VOCABULARY - 1))
    assert torch.equal(actor.encoder(b), actor.encoder(dropped))
    assert not torch.allclose(_decided(actor, b, 1)[0].hu, _decided(actor, dropped, 1)[0].hu)


def test_what_it_reads_changes_a_units_decision_but_not_its_key(actor):
    b = _batch(_items(1))
    before = actor.context(b, torch.randn(1, actor.encoder.tokens, pl.SIZE.d))
    after = actor.context(b, torch.randn(1, actor.encoder.tokens, pl.SIZE.d))
    kinds = [pm.each_unit_heads(actor, b, ctx)[0] for ctx in (before, after)]
    legal = kinds[0] > -1e8
    assert not torch.allclose(kinds[0][legal], kinds[1][legal])
    assert torch.equal(before.keys, after.keys) and torch.equal(before.hu, after.hu)


def test_padding_changes_no_decision(actor):
    items = _items(3)
    alone = _batch(items[:1])
    kind = pm.each_unit_heads(actor, alone, _decided(actor, alone, 1)[0])[0]
    both = _batch(items)
    kind3 = pm.each_unit_heads(actor, both, _decided(actor, both, 3)[0])[0]
    c = items[0].controllable
    assert torch.allclose(kind[0, :c], kind3[0, :c], atol=1e-3)
    anchor = [actor.anchor_logits(b, ctx, actor.plan(ctx, ctx.hu.new_zeros(len(b.ctrl_mask), 0, pl.SIZE.d)), torch.zeros(len(b.ctrl_mask), 1, b.ctrl_mask.shape[1], dtype=torch.bool))
              for b, ctx in ((alone, _decided(actor, alone, 1)[0]), (both, _decided(actor, both, 3)[0]))]
    assert torch.allclose(anchor[0][0, 0, :c + 1], anchor[1][0, 0, :c + 1], atol=1e-3)


def test_the_order_of_a_batch_changes_nothing(actor):
    """As a plain transformer's: a decision doesn't depend on what else is in its batch, or on any step before it."""
    items = _items(3)
    b = _batch(items)
    backward = _batch(items[::-1])
    first = pm.each_unit_heads(actor, b, _decided(actor, b, 3)[0])[0][0, :items[0].controllable]
    last = pm.each_unit_heads(actor, backward, _decided(actor, backward, 3)[0])[0][2, :items[0].controllable]
    assert torch.allclose(first, last, atol=1e-3)


# Frames in order -------------------------------------------------------------------------------------------------------------------------------------


def _frames(actor, count: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, count, actor.encoder.tokens, actor.encoder.latent, generator=torch.Generator().manual_seed(seed))


def _acts(count: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(1, count, pl.ACTION_TOKENS, pl.SIZE.d, generator=torch.Generator().manual_seed(seed))


def test_each_layer_looks_back_eight_seconds_at_any_decision_rate(actor):
    assert pl.window_frames(1) == 17 and pl.window_frames(2) == 9
    assert actor.reach == actor.size.layers * 16


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
        # Exactly: what's past the reach is masked out, so it changes nothing at all; at the reach, through every layer's
        # attention, an untrained actor's change can be under any tolerance.
        same = torch.equal(newest, actor.sequence(changed, present, acts)[:, -1])
        assert same != matters, back
    # What was done reaches no further.
    changed = acts.clone()
    changed[:, count - 2 - actor.reach] = _acts(1, seed=9)[:, 0]
    assert torch.equal(newest, actor.sequence(frames, present, changed)[:, -1])
    changed = acts.clone()
    changed[:, count - 2] = _acts(1, seed=9)[:, 0]
    assert not torch.equal(newest, actor.sequence(frames, present, changed)[:, -1])


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention is for CUDA")
def test_flex_attention_reads_and_learns_as_the_bias_does(actor):
    """The time transformer by flex attention (a block mask and a score_mod) gives what the dense bias does, and the same gradients,
    over more frames than it reaches, some from before the game, whole and newest only."""
    actor = actor.cuda()
    torch.manual_seed(3)
    with torch.no_grad():
        for block in actor.blocks:          # untrained, they're zeros, which would check nothing
            block.age.normal_()
            block.act_key.normal_()
    count = actor.reach + 6
    frames, acts = _frames(actor, count).cuda(), _acts(count).cuda()
    present = torch.ones(1, count, dtype=torch.bool, device="cuda")
    present[:, :3] = False

    def run(flex: bool, newest_only: bool):
        actor.flex = flex
        actor.zero_grad()
        out = actor.sequence(frames, present, acts, newest_only)
        (out * torch.linspace(-1, 1, out.shape[-1], device="cuda")).sum().backward()
        block = actor.blocks[0]
        return out.detach(), block.age.grad.clone(), block.act_key.grad.clone(), block.q.weight.grad.clone()

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False       # policy_train turns it on; its rounding differs by path, ~3e-3 here
    try:
        for newest_only in (False, True):
            for dense, flex in zip(run(False, newest_only), run(True, newest_only)):
                assert torch.allclose(dense, flex, rtol=1e-4, atol=1e-4), (newest_only, float((dense - flex).abs().max()))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def test_the_age_bias_is_learned_and_nothing_outside_the_window_is_read(actor):
    block = actor.blocks[0]
    frame = torch.arange(20)
    plain = torch.zeros(20, dtype=torch.bool)
    bias = block.bias(frame, frame, plain, plain)
    assert bias.shape == (actor.size.heads, 20, 20)
    assert torch.isinf(bias[:, 3, 4]).all() and torch.isinf(bias[:, 19, 19 - actor.window]).all()
    assert torch.isfinite(bias[:, 19, 19 - actor.window + 1]).all()
    # A frame's own action tokens: its latent tokens don't see them, they see its latent tokens.
    two, act = torch.tensor([5, 5]), torch.tensor([False, True])
    own = block.bias(two, two, act, act)
    assert torch.isinf(own[:, 0, 1]).all() and torch.isfinite(own[:, 1, 0]).all()
    frames, present = _frames(actor, 4), torch.ones(1, 4, dtype=torch.bool)
    out = actor.sequence(frames, present, _acts(4))[:, -1]
    (out * torch.randn_like(out)).sum().backward()      # not a plain sum: that of a LayerNorm's output is 0 whatever goes in
    assert float(block.age.grad.abs().sum()) > 0.0 and float(block.act_key.grad.abs().sum()) > 0.0


def test_the_age_bias_by_frame_is_the_age_bias_by_token_and_learns_the_same():
    """`_age_bias` indexes a pair of frames at a time when it can; indexing every pair of tokens must give the same, and the same gradient."""
    block = pl.TimeBlock(pl.SIZE.d, pl.HEADS, 5)
    with torch.no_grad():
        block.age.copy_(torch.randn_like(block.age))
    frame = torch.arange(7).repeat_interleave(3)
    newest = frame[-3:]
    for q in (frame, newest):
        fast = block._age_bias(q, frame)
        slow = block.age[:, (q[:, None] - frame[None, :]).clamp(0, 4)]
        assert torch.equal(fast, slow)
        weights = torch.randn_like(fast)
        grad_fast, = torch.autograd.grad((fast * weights).sum(), block.age)
        grad_slow, = torch.autograd.grad((slow * weights).sum(), block.age)
        assert torch.allclose(grad_fast, grad_slow, atol=1e-5)


def test_the_bias_built_by_frames_is_the_bias_by_tokens_and_learns_the_same():
    """`frame_bias`, which `sequence` uses because it compiles, against `bias` token by token: the same values and -inf, and the same
    gradients to the age table and the action-key bias; for every frame's queries and for the newest's latent tokens alone."""
    block = pl.TimeBlock(pl.SIZE.d, pl.HEADS, 5)
    with torch.no_grad():
        block.age.copy_(torch.randn_like(block.age))
        block.act_key.copy_(torch.randn_like(block.act_key))
    frames, per, tokens = torch.arange(7), 4, 3
    is_act = torch.arange(per) >= tokens
    frame, act = frames.repeat_interleave(per), is_act.repeat(len(frames))
    for qf, qa, q in ((frames, is_act, slice(None)), (frames[-1:], is_act[:tokens], slice(len(frame) - per, len(frame) - 1))):
        built = block.frame_bias(qf, frames, qa, is_act)
        want = block.bias(frame[q], frame, act[q], act)
        assert torch.equal(torch.isinf(built), torch.isinf(want))
        finite = torch.isfinite(want)
        assert torch.allclose(built[finite], want[finite])
        weights = torch.randn_like(want).masked_fill(~finite, 0.0)
        got = torch.autograd.grad((built.masked_fill(~finite, 0.0) * weights).sum(), (block.age, block.act_key))
        expected = torch.autograd.grad((want.masked_fill(~finite, 0.0) * weights).sum(), (block.age, block.act_key))
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, expected))


def test_a_seat_keeps_its_frames_newest_first_and_forgets_the_oldest(actor):
    b = _batch(_items(1))
    history, seen = _state(actor, 1)
    marks = iter(range(1, 100))
    actor.encoder.forward = lambda _: torch.full((1, actor.encoder.tokens, actor.encoder.latent), float(next(marks)))
    for _ in range(actor.reach + 5):
        _, history, seen, keep = actor.decide(b, history, seen)
    assert keep["latents"].shape == (1, actor.encoder.tokens, actor.encoder.latent)
    assert history.shape == (1, actor.reach, actor.record)
    assert [float(history[0, k, 0]) for k in (0, 1, actor.reach - 1)] == [actor.reach + 5.0, actor.reach + 4.0, 6.0]
    assert float(seen.sum()) == actor.reach


def test_what_it_did_is_kept_as_the_newest_frames_action_tokens(actor):
    b = _batch(_items(1))
    ctx, history, seen, _ = _decided(actor, b, 1)
    cut = actor.encoder.tokens * actor.encoder.latent
    assert float(history[0, 0, cut:].abs().sum()) == 0.0
    a = pm.sample_actions(actor, b, ctx)
    done = actor.acted(b, a, history)
    assert torch.allclose(done[0, 0, cut:], actor.action_tokens(b, a)[0].flatten(), atol=1e-6)
    assert torch.equal(done[0, 0, :cut], history[0, 0, :cut]) and torch.equal(done[:, 1:], history[:, 1:])


def _nothing(count: int, slots: int = 3) -> pm.Actions:
    return pm.actions_from({f: np.full((count, slots), pt.act.UNSET[f]) for f in pt.act.SLOT_FIELDS}, "cpu")


def test_doing_nothing_reads_alike_whoever_did_it(actor):
    """Done at once is nothing done: the action tokens say only that."""
    b = _batch(_items(3))
    tokens = actor.action_tokens(b, _nothing(3))
    assert tokens.shape == (3, pl.ACTION_TOKENS, pl.SIZE.d)
    assert torch.allclose(tokens[0], tokens[1], atol=1e-5) and torch.allclose(tokens[0], tokens[2], atol=1e-5)


def test_who_was_told_what_and_where_reaches_the_action_tokens_in_any_order(actor):
    b = _batch(_items(1))
    nothing = _nothing(1)
    movers = torch.nonzero(b.ctrl_mask[0] & b.kind_mask[0, :, rf.MOVE]).flatten()
    u = int(movers[0])
    regions = torch.nonzero(b.ptr_region[0, u]).flatten()

    def told(*moves) -> pm.Actions:
        a = _nothing(1)
        for k, (unit, region) in enumerate(moves):
            a.anchor[0, k], a.kind[0, k], a.region[0, k], a.fine[0, k] = unit, rf.MOVE, region, 0
        return a

    first = actor.action_tokens(b, told((u, int(regions[0]))))
    assert not torch.allclose(first, actor.action_tokens(b, nothing), atol=1e-4)
    assert not torch.allclose(first, actor.action_tokens(b, told((u, int(regions[1])))), atol=1e-5)
    if len(movers) > 1:
        v = int(movers[1])
        both = actor.action_tokens(b, told((u, int(regions[0])), (v, int(regions[0]))))
        assert not torch.allclose(first, both, atol=1e-5)
        assert torch.allclose(both, actor.action_tokens(b, told((v, int(regions[0])), (u, int(regions[0])))), atol=1e-5)


# Pointers ------------------------------------------------------------------------------------------------------------------------------------------


def test_a_pointer_score_from_place_depends_on_the_difference_alone(actor):
    query = torch.zeros(1, 1, pl.SIZE.d)
    gain = torch.ones(1, 1, 2 * pl.BANDS)
    here, there = torch.tensor([[[0.2, -0.3]]]), torch.tensor([[[0.5, 0.1], [-0.7, 0.4]]])
    keys = lambda at: torch.cat((torch.zeros(1, 2, pl.SIZE.d), actor.code(at)), -1)  # noqa: E731
    first = actor._score(query, gain, actor.code(here), keys(there))
    moved = actor._score(query, gain, actor.code(here + 0.11), keys(there + 0.11))
    assert torch.allclose(first, moved, atol=1e-4)


def test_a_pointer_score_from_place_is_highest_on_its_own_place(actor):
    query = torch.zeros(1, 1, pl.SIZE.d)
    gain = torch.ones(1, 1, 2 * pl.BANDS)
    here = torch.tensor([[[0.2, -0.3]]])
    places = torch.tensor([[[0.2, -0.3], [0.21, -0.3], [0.3, -0.3], [-0.8, 0.9]]])
    keys = torch.cat((torch.zeros(1, 4, pl.SIZE.d), actor.code(places)), -1)
    score = actor._score(query, gain, actor.code(here), keys)[0, 0]
    assert int(score.argmax()) == 0 and float(score[1]) > float(score[2]) > float(score[3])


def test_a_target_a_little_further_gets_another_score_whatever_the_latents_say(actor):
    items = _items(1)
    b = _batch(items)
    now = actor.sequence(actor.encoder(b).float()[:, None], torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, pl.ACTION_TOKENS, pl.SIZE.d))[:, -1]
    assert len(items[0].enemy_tokens) > 0
    enemy = pm.each_unit_heads(actor, b, actor.context(b, now))[2]
    moved = b.unit_x.clone()
    moved[0, int(items[0].enemy_tokens[0]), fz._I] += 0.02
    shifted = replace(b, unit_x=moved)
    legal = b.ptr_enemy[0, 0, :, 0]
    if legal.any():
        again = pm.each_unit_heads(actor, shifted, actor.context(shifted, now))[2]
        assert not torch.allclose(enemy[0, 0, legal, 0], again[0, 0, legal, 0], atol=1e-6)


# Choosing ------------------------------------------------------------------------------------------------------------------------------------------


def _sampled(actor, seed: int = 3, done: float | None = None):
    items = _items(3, seed)
    b = _batch(items)
    torch.manual_seed(seed)
    if done is not None:
        with torch.no_grad():
            actor.head_done.bias[0] = done
    ctx, *_ = _decided(actor, b, 3)
    return items, b, ctx, pm.sample_actions(actor, b, ctx)


def test_it_starts_by_being_done_rather_than_picking_any_one_unit(actor):
    _, b, ctx, _ = _sampled(actor)
    first = actor.anchor_logits(b, ctx, actor.plan(ctx, ctx.hu.new_zeros(3, 0, pl.SIZE.d)), torch.zeros(3, 1, b.ctrl_mask.shape[1], dtype=torch.bool))[:, 0]
    assert (first.argmax(-1) == 0).all()


def test_every_choice_is_legal_and_no_unit_is_picked_twice(actor):
    acted = 0
    for seed in range(4):
        items, b, _, a = _sampled(actor, seed, done=-3.0)
        for i, t in enumerate(items):
            picked = [int(u) for u in a.anchor[i] if u >= 0]
            assert len(set(picked)) == len(picked) and all(t.kind_mask[u].sum() > t.kind_mask[u, rf.KEEP] for u in picked)
            assert (a.anchor[i, len(picked):] == -1).all()
            acted += len(picked)
            for k, u in enumerate(picked):
                kind = int(a.kind[i, k])
                assert kind != rf.KEEP and t.kind_mask[u, kind]
                if a.region[i, k] >= 0:
                    legal = t.ptr_jump if kind in (rf.JUMP, rf.THROW) else t.ptr_wait if kind == rf.WAIT_AT_BEACON else t.ptr_region
                    assert legal[u, int(a.region[i, k])]
    assert acted > 10


@pytest.mark.parametrize("done", (None, -3.0))
def test_a_choice_scores_the_same_when_it_is_evaluated(actor, done):
    _, b, ctx, a = _sampled(actor, done=done)
    logp, entropy = pm.evaluate_actions(actor, b, ctx, a)
    assert torch.allclose(logp, a.logp, atol=1e-3)
    assert set(entropy) == {"anchor", "kind", "region", "enemy", "fine", "produce", "place", "offset", "wreck", "group", "roles"}


def test_a_scripts_actions_score_as_a_set(actor):
    """In cloning, an action's anchor may be any of the units still to be told: so the order they're given in changes nothing."""
    _, b, ctx, a = _sampled(actor, done=-3.0)
    picked = a.anchor[0] >= 0
    n = int(picked.sum())
    assert n >= 2
    labelled = torch.zeros_like(b.ctrl_mask)
    labelled[0, a.anchor[0, :n]] = True
    with torch.no_grad():
        as_set = pm.evaluate_actions(actor, b, ctx, a, labelled)[0][0]
        in_order = pm.evaluate_actions(actor, b, ctx, a)[0][0]
    assert (as_set >= in_order - 1e-5).all() and float(as_set[0]) > float(in_order[0]) + 1e-4
    assert abs(float(as_set[n - 1]) - float(in_order[n - 1])) < 1e-4      # the last can only be what's left


# The size ------------------------------------------------------------------------------------------------------------------------------------------


def test_the_parts_add_up(actor):
    report = pl.parameter_report(actor)
    assert report["total"] == sum(p.numel() for p in actor.parameters())
    assert report["trained"] == sum(p.numel() for p in actor.parameters() if p.requires_grad)


# Its strategy -----------------------------------------------------------------------------------------------------------------------------------------


def test_z_reads_as_nothing_until_it_is_learned_and_then_as_the_strategy(actor):
    """z_in starts at zero, so a checkpoint from before z plays as it did; once trained, z changes what every frame holds."""
    frames, present, acts = _frames(actor, 6), torch.ones(1, 6, dtype=torch.bool), _acts(6)
    z = torch.randn(1, pl.Z_DIM, generator=torch.Generator().manual_seed(3))
    assert torch.equal(actor.sequence(frames, present, acts), actor.sequence(frames, present, acts, z=z))
    with torch.no_grad():
        actor.z_in.weight.normal_()
    assert not torch.allclose(actor.sequence(frames, present, acts), actor.sequence(frames, present, acts, z=z), atol=1e-3)
    assert pl.answers("z_in", actor.z_in)        # AdamW's, not Muon's, which would move a zero matrix by a whole step at once
