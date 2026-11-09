"""Group orders, as the policy gives them: one unit, the anchor, chooses an order and that it goes as well to every unit of its side of
some roles within a radius of it, as a player gives one order to a box of selected units.  The sim takes them as they are (capi.h
zk_group_rec); `expand` gives the unit orders they stand for, for whatever takes only those.

Roles are the bot catalog's (`zkbot.catalog.role`), numbered in sorted order: the army script and the sim number them the same way.  The
radius is one of GROUP_RADII.  The policy groups only orders of GROUP_KINDS.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import rawframe as rf
from .tables import table

GROUP_KINDS = (rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT)
GROUP_RADII = (60.0, 90.0, 135.0, 200.0, 300.0, 450.0, 675.0, 1000.0, 1500.0, 2250.0)     # elmos
GROUP_MARK = 100        # a record's kind in a script's recorded orders: the order before it, of the same unit, went as a group's
                        # anchor, with x the radius and target the roles (zkscripts.h ZKS_GROUP_MARK)


@lru_cache(maxsize=None)
def roles() -> tuple[tuple[str, ...], np.ndarray]:
    """The role names in their order, and each type-table row's role, the unknown row's last ("other")."""
    from zkbot import catalog
    t = table()
    role_of = [catalog.role(name) for name in t.names] + ["other"]
    names = tuple(sorted(set(role_of)))
    assert len(names) <= 32
    return names, np.array([names.index(r) for r in role_of], np.int64)


@lru_cache(maxsize=None)
def choosable() -> int:
    """The roles a group can take, as bits: those of a type that moves.  The others never are selected."""
    t = table()
    _, role_of = roles()
    mobile = np.asarray(t.mobile, bool)[:len(t.names)]
    return int(np.bitwise_or.reduce(1 << np.unique(role_of[:len(t.names)][mobile])))


def bucket(radius: float) -> int:
    """The index in GROUP_RADII of the smallest radius at least this, or -1 if none is."""
    i = int(np.searchsorted(np.asarray(GROUP_RADII), radius - 1e-3))
    return i if i < len(GROUP_RADII) else -1


def expand(anchor: int, roles_mask: int, radius: float, units: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Which of `units` (raw-frame records of the anchor's side, with their type-table rows) a group order gives its order to, as the sim
    selects them: the anchor, and every other that moves, is finished, has a role in the mask and is within the radius of the anchor."""
    at = np.flatnonzero(units["uid"] == anchor)
    if not len(at):
        return np.zeros(len(units), bool)
    a = units[at[0]]
    _, role_of = roles()
    moves = np.asarray(table().mobile, bool)[rows]
    finished = (units["flags"] & rf.BEING_BUILT) == 0
    in_mask = ((roles_mask >> role_of[rows]) & 1) == 1
    dx, dz = (units["x"] - a["x"]).astype(np.float32), (units["z"] - a["z"]).astype(np.float32)
    near = dx * dx + dz * dz <= np.float32(radius) * np.float32(radius)
    return (units["uid"] == anchor) | (moves & finished & in_mask & near)
