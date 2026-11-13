"""libzkpolicy (sim/py/zkpolicy.py) against the Python it ports, to the bit, on the golden frames: five live ones and four from the
sim.  The tens of thousands of recorded frames of every kind of game are tools/policy_parity.py's (02-policy.md, "Deployment")."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "sim" / "py"), str(ROOT / "tools")]
import zkpolicy as zp  # noqa: E402
from policy_parity import decision_diff, orders_diff, random_decision, tensor_diff  # noqa: E402
from zkbot.policy import actions as act, bundle as bd, featurize as fz, groups as gr, rawframe as rf, tables  # noqa: E402

DATA = ROOT / "tests" / "policy" / "data"
# As training featurizes (the sim's kinds, the economy, the critic's units), and featurize's defaults.
SIM_KINDS = tuple(k for k in range(rf.KIND_COUNT) if k not in (rf.ATTACK_GROUND, rf.STOP, rf.REPAIR, rf.RESERVED_KIND))
CONFIGS = (dict(enabled_kinds=SIM_KINDS, economy=True, privileged=True), {})


@pytest.fixture(scope="module")
def golden():
    frames = rf.load_frames(DATA / "golden_frames.npz") + rf.load_frames(DATA / "sim_frames.npz")
    return frames, bd.Bundle.load(DATA / "golden_bundle.npz")


@pytest.fixture(scope="module")
def table_file():
    """The table the bot reads: what tables.py makes now (sim/py/zkpolicy.py table rewrites it)."""
    assert zp.TABLE_FILE.read_bytes() == zp.table_bytes(tables.table()), "policy_table.bin is stale: ~/zkml/bin/python sim/py/zkpolicy.py table"
    return zp.Table.load()


@pytest.mark.parametrize("config", CONFIGS, ids=("training", "defaults"))
def test_featurizes_as_the_python_does(golden, table_file, config):
    frames, bundle = golden
    table = tables.table()
    items = [(f, bundle) for f in frames]
    kept = [np.full(int(zp.bounds(items, table_file, config.get("privileged", False)).max()), 0xff, np.uint8) for _ in items]
    batched = zp.featurize_many(items, table_file, threads=4, buffers=kept, **config)       # a worker's buffers, used before
    for i, (frame, got) in enumerate(zip(frames, batched)):
        want = fz.featurize(frame, bundle, table, **config)
        assert not tensor_diff(want, got), (i, tensor_diff(want, got))
        assert not tensor_diff(want, zp.featurize(frame, bundle, table, **config)), i


def test_decodes_and_encodes_as_the_python_does(golden, table_file):
    """Random legal decisions to orders, those orders back to choices, and a decision to actions and back, each both ways."""
    frames, bundle = golden
    table, rng = tables.table(), np.random.default_rng(3)
    for frame in frames:
        for config in CONFIGS:
            py, cc = fz.featurize(frame, bundle, table, **config), zp.featurize(frame, bundle, table_file, **config)
            x, z = act.world_positions(py, bundle)
            xc, zc = zp.world_positions(cc, bundle)
            assert np.array_equal(x, xc) and np.array_equal(z, zc)
            for _ in range(4):
                d = random_decision(py, rng)
                orders = act.decode(d, py, bundle, x, z)
                assert orders_diff(orders, zp.decode(d, cc, bundle, xc, zc, table_file)) is None
                want, missed = act.encode(orders, py, bundle, x, z)
                got, got_missed = zp.encode(orders, cc, bundle, xc, zc, table_file)
                assert decision_diff(want, got) is None and missed == got_missed
                order = rng.permutation(py.controllable)
                slots = act.to_slots(d, order)
                assert all(np.array_equal(getattr(slots, f), getattr(zp.to_slots(d, order), f)) for f in act.SLOT_FIELDS)
                assert decision_diff(act.to_decision(slots, py.controllable), zp.to_decision(slots, py.controllable)) is None
    for radius in (0.0, 59.9995, 60.0, 61.0, 2250.0, 2251.0):
        assert zp.bucket(radius) == gr.bucket(radius)
