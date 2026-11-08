"""Orders someone else gave, read back as the policy's choices: what supervised training learns from."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from zkbot.policy import actions, featurize as fz, rawframe as rf, tables

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_toy as toy
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

TABLE = tables.table()


def _frames(count: int, seed: int):
    bundle = toy.plain()
    rng = np.random.default_rng(seed)
    game = toy.ToyGame(rng)
    for _ in range(count):
        yield fz.featurize(game.frame_for(0), bundle, TABLE, enabled_kinds=fz.STAGE2_KINDS, privileged=True), bundle
        game.apply(1, toy.script_focus(game, 1, rng))
        game.step()


def _random_choice(t: fz.Tensors, rng: np.random.Generator) -> actions.Decision:
    """Any legal choice for each unit, as the policy could make."""
    c = t.controllable
    kind, enemy, region, fine, produce = (np.full(c, v, np.int64) for v in (rf.KEEP, -1, -1, -1, -1))
    for i in range(c):
        k = int(rng.choice(np.flatnonzero(t.kind_mask[i])))
        kind[i] = k
        if k in fz.KINDS_WITH_REGION:
            mask = t.ptr_jump if k in (rf.JUMP, rf.THROW) else t.ptr_wait if k == rf.WAIT_AT_BEACON else t.ptr_region
            region[i] = int(rng.choice(np.flatnonzero(mask[i])))
        if k in fz.ENEMY_POINTER_KINDS:
            enemy[i] = int(rng.choice(np.flatnonzero(t.ptr_enemy[fz.ENEMY_POINTER_KINDS.index(k), i])))
        if k in fz.KINDS_WITH_FINE:
            fine[i] = int(rng.integers(fz.FINE_OPTIONS))
        if k == rf.PRODUCE:
            produce[i] = int(rng.choice(np.flatnonzero(t.ptr_produce[i])))
    return actions.Decision(kind, enemy, region, fine, produce, np.zeros(c, np.float32))


def test_fine_points_are_where_decoding_sends_a_unit():
    t, bundle = next(_frames(1, 3))
    points = actions.fine_points(t, bundle)
    for r in range(0, bundle.regions, 7):
        for o in range(fz.FINE_OPTIONS):
            assert np.allclose(points[r, o], fz.fine_point(t, bundle, r, o), atol=1e-3)


def test_what_decoding_makes_reads_back_as_the_same_choices():
    rng = np.random.default_rng(1)
    checked = 0
    for t, bundle in _frames(12, 4):
        chosen = _random_choice(t, rng)
        x, z = actions.world_positions(t, bundle)
        back, missed = actions.encode(actions.decode(chosen, t, bundle, x, z), t, bundle, x, z)
        assert missed == 0
        for field in ("kind", "enemy", "produce"):
            assert np.array_equal(getattr(back, field), getattr(chosen, field)), field
        # A region and option are the same place, unless two of them coincide.
        for i in np.flatnonzero(np.isin(chosen.kind, fz.KINDS_WITH_REGION)):
            want = fz.fine_point(t, bundle, int(chosen.region[i]), max(int(chosen.fine[i]), 0))
            got = fz.fine_point(t, bundle, int(back.region[i]), max(int(back.fine[i]), 0))
            assert np.hypot(want[0] - got[0], want[1] - got[1]) < 1e-3
            checked += 1
    assert checked > 10


def test_a_place_anywhere_becomes_the_nearest_the_unit_could_go():
    t, bundle = next(_frames(1, 5))
    x, z = actions.world_positions(t, bundle)
    i = next(i for i in range(t.controllable) if t.kind_mask[i, rf.FIGHT])
    target = (float(bundle.centre[3, 0]) + 37.0, float(bundle.centre[3, 1]) - 21.0)
    back, missed = actions.encode([actions.Order(int(t.uids[i]), rf.FIGHT, *target)], t, bundle, x, z)
    assert missed == 0 and back.kind[i] == rf.FIGHT
    points = actions.fine_points(t, bundle)
    best = np.hypot(points[..., 0] - target[0], points[..., 1] - target[1])
    best[~t.ptr_region[i]] = np.inf
    assert np.isclose(np.hypot(*(points[back.region[i], back.fine[i]] - target)), best.min())
    assert (back.kind[np.arange(t.controllable) != i] == rf.KEEP).all()


def test_what_the_policy_cannot_say_is_counted_and_kept():
    t, bundle = next(_frames(1, 6))
    x, z = actions.world_positions(t, bundle)
    i = 0
    stranger = actions.Order(10**6, rf.MOVE, 100.0, 100.0)
    no_target = actions.Order(int(t.uids[i]), rf.ATTACK_UNIT, target=10**6)
    back, missed = actions.encode([stranger, no_target], t, bundle, x, z)
    assert missed == 2 and (back.kind == rf.KEEP).all()
