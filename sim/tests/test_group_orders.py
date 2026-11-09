"""Group orders (capi.h zk_group_rec): one order to an anchor and the units of its side, of some roles, within a radius of it."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
RAID, RIOT = 0, 1


def roles() -> np.ndarray:
    out = np.full(len(CAT.units), -1, np.int32)
    out[CAT.index["cloakraid"]] = RAID
    out[CAT.index["cloakriot"]] = RIOT
    out[CAT.index["turretlaser"]] = RAID      # a role, but it doesn't move: never selected
    return out


def test_a_group_order_goes_to_its_anchor_and_the_units_of_its_side_and_roles_about_it():
    w = zksim.World(CAT, zksim.Map.flat(), seed=1)
    w.set_roles(roles())
    anchor = w.add("cloakskirm", 0, 1020, 1020)     # no role of its own: it's told all the same
    picked = [anchor, w.add("cloakraid", 0, 1000, 1000), w.add("cloakraid", 0, 1060, 1000), w.add("cloakriot", 0, 1000, 1080)]
    left = [w.add("cloakraid", 0, 1300, 1000),              # outside the circle
            w.add("cloakraid", 1, 1000, 1020),              # the other side's
            w.add("cloakraid", 0, 1020, 1040, build=0.5),   # not finished
            w.add("turretlaser", 0, 980, 980),              # doesn't move
            w.add("cloakskirm", 0, 1040, 1040)]             # no role
    assert w.order_group(anchor, (1 << RAID) | (1 << RIOT), 100.0, zksim.ORDER_MOVE, 3000.0, 1000.0) == len(picked)
    assert all(w.get(u).order == zksim.ORDER_MOVE for u in picked)
    assert all(w.get(u).order == zksim.ORDER_NONE for u in left)
    assert w.order_group(picked[1], 1 << RIOT, 100.0, zksim.ORDER_FIGHT, 0.0, 0.0) == 2
    assert w.get(picked[3]).order == zksim.ORDER_FIGHT and w.get(picked[2]).order == zksim.ORDER_MOVE


def test_in_a_batch_a_games_unit_orders_go_after_its_group_orders():
    batch = zksim.Batch(CAT, 1, threads=1)
    batch.set_roles(roles())
    types = [CAT.index["cloakraid"]] * 2
    batch.reset(0, zksim.Map.flat(), 1, np.array(types), np.array([0, 0]), np.array([1000.0, 1050.0]), np.array([1000.0, 1000.0]))
    groups = np.array([(0, 1 << RAID, 60.0, zksim.ORDER_MOVE, 3000.0, 1000.0, -1)], zksim.GROUP_DTYPE)
    orders = np.array([(1, zksim.ORDER_MOVE, 1000.0, 3000.0, -1)], zksim.ORDER_DTYPE)
    batch.step(orders, np.array([0, 1], np.int32), ticks=20, groups=groups, group_offsets=np.array([0, 1], np.int32))
    world = batch.records[0, 0, :int(batch.counts[0, 0])]
    by_uid = {int(r["uid"]): r for r in world}
    assert by_uid[0]["x"] > 1200.0 and abs(by_uid[0]["z"] - 1000.0) < 30.0     # the group's move, east
    assert by_uid[1]["z"] > 1200.0 and abs(by_uid[1]["x"] - 1050.0) < 30.0     # its own, south
