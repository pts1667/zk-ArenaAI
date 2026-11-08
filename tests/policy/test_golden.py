"""The featurizer on frames recorded in live games, against reviewed output.  A failure here means the policy's view of the
game changed: tools/policy_golden.py rewrites the expected files, once that change is understood and meant."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf, tables

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA = pathlib.Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(ROOT / "tools"))
TOLERANCE = 1e-6


def golden():
    return rf.load_frames(DATA / "golden_frames.npz"), bd.Bundle.load(DATA / "golden_bundle.npz")


def test_sim_frames_read_the_same_way_as_live_ones():
    """The sim's frames, through the same featurizer and the live game's bundle, against reviewed output."""
    bundle = bd.Bundle.load(DATA / "golden_bundle.npz")
    frames = rf.load_frames(DATA / "sim_frames.npz")
    assert len(frames) >= 3
    table = tables.table()
    known = np.concatenate([f.units["knowledge"] for f in frames])
    for state in (rf.KNOW_LOS, rf.KNOW_HIDDEN):
        assert (known == state).any(), state
    for i, frame in enumerate(frames):
        t = fz.featurize(frame, bundle, table, privileged=True)
        assert np.isfinite(t.unit_x).all() and np.isfinite(t.reg_x).all()
        assert t.kind_mask.any(axis=1).all()
        off = fz.featurize(frame, bundle, table)
        with np.load(DATA / f"expected_sim_{i}.npz") as e:
            assert off.controllable == int(e["controllable"])
            for name in e.files:
                if name == "controllable":
                    continue
                got = getattr(off, name)
                assert got.shape == e[name].shape, name
                if got.dtype.kind == "f":
                    assert np.abs(got - e[name]).max(initial=0.0) <= TOLERANCE, name
                else:
                    assert np.array_equal(got, e[name]), name


def test_there_are_five_frames_and_they_show_what_they_should():
    frames, _ = golden()
    assert len(frames) == 5
    known = np.concatenate([f.units["knowledge"] for f in frames])
    for state in (rf.KNOW_LOS, rf.KNOW_RADAR, rf.KNOW_MEMORY):
        assert (known == state).any(), state
    assert not (known == rf.KNOW_HIDDEN).any()      # a live game has no way of knowing these
    assert {int(a) for f in frames for a in f.units["ally_team"]} == {0, 1}
    assert sum(len(f.idle_uids) for f in frames) > 0


@pytest.mark.parametrize("i", range(5))
def test_features_match_the_reviewed_output(i):
    frames, bundle = golden()
    t = fz.featurize(frames[i], bundle, tables.table())
    with np.load(DATA / f"expected_{i}.npz") as e:
        assert t.controllable == int(e["controllable"])
        for name in e.files:
            if name == "controllable":
                continue
            got = getattr(t, name)
            assert got.shape == e[name].shape, name
            if got.dtype.kind == "f":
                assert np.abs(got - e[name]).max(initial=0.0) <= TOLERANCE, name
            else:
                assert np.array_equal(got, e[name]), name


@pytest.mark.parametrize("i", range(5))
def test_live_frames_have_legal_actions(i):
    frames, bundle = golden()
    table = tables.table()
    t = fz.featurize(frames[i], bundle, table)
    assert t.kind_mask.any(axis=1).all()
    assert t.ptr_region.any(axis=1).all()
    for slot, kind in enumerate(fz.ENEMY_POINTER_KINDS):
        assert t.ptr_enemy[slot][t.kind_mask[:, kind]].any(axis=1).all()


def test_economy_inputs_stay_out_at_stage_2():
    frames, bundle = golden()
    table = tables.table()
    off, on = fz.featurize(frames[1], bundle, table), fz.featurize(frames[1], bundle, table, economy=True)
    assert off.glob_x[rf.G["metal"]] == 0.0 and on.glob_x[rf.G["metal"]] > 0.0
    assert (off.site_x[:, 3] == 1.0).all() and not (on.site_x[:, 3] == 1.0).all()
    assert np.array_equal(off.unit_x, on.unit_x)
