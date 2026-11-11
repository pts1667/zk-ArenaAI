"""The featurizer's per-unit loops (featurize.py), compiled with numba when it's installed, plain Python otherwise.

Each gives what the numpy it replaced gave, bit for bit, under numba and without it: float32 stays float32 (as numpy computes
float32 arrays with Python floats), a mean is numpy's pairwise sum, and what numpy summed with bincount is summed in float64 in
the same order.  Under numba a Python float literal would widen an expression to float64, so every constant is a float32 global.
Without numba this is the same code run by Python on numpy scalars: right, and far slower.
"""
from __future__ import annotations

import numpy as np

from . import rawframe as rf

try:
    from numba import njit
except ImportError:         # pragma: no cover - the live bot may run without it
    import warnings
    warnings.warn("numba isn't installed: featurize runs in plain Python, some 50 times slower (pip install numba)", stacklevel=2)

    def njit(*args, **kwargs):
        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else (lambda f: f)

# rawframe's numbers, written out: numba's cache knows only this file, so it would keep code compiled with numbers since changed
# elsewhere.  Checked against rawframe on import.
BEING_BUILT, CLOAKED, PARALYZED, IN_LOS, IDLE, TYPE_KNOWN, DAMAGED, RESERVED, JUMP_READY, PULLING = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
CARRIED, LOADED = 1024, 2048
KNOW_LOS, KNOW_RADAR, KNOW_MEMORY, KNOW_HIDDEN = 0, 1, 2, 3
KEEP, MOVE, FIGHT, ATTACK_UNIT, ATTACK_GROUND, STOP, MANUAL_FIRE, JUMP, CLOAK_ON, CLOAK_OFF = range(10)
BUILD, ASSIST, REPAIR, RECLAIM, PRODUCE, RESERVED_KIND = range(10, 16)
THROW, PUSH, PULL, PLACE_BEACON, WAIT_AT_BEACON = range(16, 21)
LOAD, UNLOAD, DROP = range(21, 24)
ORDER_SLOTS, KIND_COUNT = 16, 24
_MIRRORED = ("BEING_BUILT CLOAKED PARALYZED IN_LOS IDLE TYPE_KNOWN DAMAGED RESERVED JUMP_READY PULLING CARRIED LOADED KNOW_LOS KNOW_RADAR KNOW_MEMORY "
             "KNOW_HIDDEN KEEP MOVE FIGHT ATTACK_UNIT ATTACK_GROUND STOP MANUAL_FIRE JUMP CLOAK_ON CLOAK_OFF BUILD ASSIST REPAIR RECLAIM "
             "PRODUCE RESERVED_KIND THROW PUSH PULL PLACE_BEACON WAIT_AT_BEACON LOAD UNLOAD DROP ORDER_SLOTS KIND_COUNT").split()
_CHANGED = [name for name in _MIRRORED if globals()[name] != getattr(rf, name)]
if _CHANGED:
    raise ImportError(f"rawframe's {_CHANGED} changed: update featurize_kernels' copies")

# unit_x after the type stats (whose width, from the catalog, the stats array brings): instance, order memory, rays, summary.
INSTANCE_COLS = 21
ORDER_COLS = ORDER_SLOTS + 7
RAY_COLS = 24
RAY_CELLS = 12            # 32-elmo steps, to 400 elmos
RAY_STEP = 32.0
RAY_DPS_RADIUS = 600.0
LEAST_COST = 100.0        # what anything counts for at least when choosing the pool: a blip's type may be unknown
ALTITUDE_DEAD_ZONE = 40.0
PASS_CELL = 16.0          # bundle's, checked there

# The table packed for the kernels (featurize._table_pack): one float matrix and one bool matrix, with these columns.
(T_COST, T_SPEED, T_RANGE, T_DPS_GROUND, T_DPS_AIR, T_MANUAL_RANGE, T_MANUAL_RELOAD, T_JUMP_RANGE, T_THROW_RADIUS, T_THROW_REACH,
 T_LIFT_LIGHT, T_LIFT_MEDIUM, T_LIFT_HEAVY, T_LIFT_SIZE, T_LOAD_CLASS, T_LOAD_SIZE) = range(16)
(B_MOBILE, B_AIR, B_ARMED, B_HIT_GROUND, B_HIT_AIR, B_BUILDER, B_FACTORY, B_COMBAT, B_MANUALFIRE, B_JUMP, B_CLOAK, B_SWITCHES,
 B_TELEPORTS, B_BEACON, B_MISSILE, B_LAUNCHER, B_ACTS_IN_PLACE, B_CONSTRUCTOR, B_PRODUCES, B_BUILDS, B_TRANSPORT) = range(21)

_F0, _F1, _F2, _F4 = np.float32(0.0), np.float32(1.0), np.float32(2.0), np.float32(4.0)
_F6, _F15, _F30 = np.float32(6.0), np.float32(15.0), np.float32(30.0)
_F150, _F200, _F300 = np.float32(150.0), np.float32(200.0), np.float32(300.0)
_F1_5 = np.float32(1.5)
_STILL = np.float32(0.5)          # elmos a frame: an enemy faster than this can't be picked up (unit_transport_load_unload.lua)
_AT_ITS_TRANSPORT = np.float32(32.0)      # a carried unit is this near its transport, across: the sim says the same place
_PASS_CELL = np.float32(PASS_CELL)
_LEAST_COST = np.float32(LEAST_COST)
_DEAD_ZONE = np.float32(ALTITUDE_DEAD_ZONE)
_DPS_RADIUS2 = np.float32(RAY_DPS_RADIUS * RAY_DPS_RADIUS)
_PI8, _PI4 = np.float32(np.pi / 8.0), np.float32(np.pi / 4.0)
_RAY_ANGLES = np.arange(8) * (np.pi / 4.0)
_RAY_DX = np.sin(_RAY_ANGLES).astype(np.float32)
_RAY_DZ = np.cos(_RAY_ANGLES).astype(np.float32)
_RAY_T = (np.arange(1, RAY_CELLS + 1) * RAY_STEP).astype(np.float32)
_INSTANCE_BITS = np.array((BEING_BUILT, CLOAKED, PARALYZED, IN_LOS, IDLE, TYPE_KNOWN, DAMAGED), np.int64)


# numpy's own scalar rules: maximum and minimum pass a NaN on, clip is their composition, sign keeps a NaN.
@njit(cache=True)
def _max(a, b):
    return a if (a >= b or a != a) else b


@njit(cache=True)
def _min(a, b):
    return a if (a <= b or a != a) else b


@njit(cache=True)
def _clip(v, lo, hi):
    return _min(_max(v, lo), hi)


@njit(cache=True)
def _sign(a):
    if a > _F0:
        return _F1
    if a < _F0:
        return -_F1
    if a == _F0:
        return _F0
    return a


@njit(cache=True)
def _row(row_of_def, def_id):
    """TypeTable.rows for one def id."""
    last = len(row_of_def) - 1
    if def_id < 0 or def_id >= last:
        return np.int64(row_of_def[last])
    return np.int64(row_of_def[def_id])


@njit(cache=True)
def _cell(x, z, cell, h, w):
    """Bundle._cell for one point."""
    i = np.int64(np.int32(z / cell))
    j = np.int64(np.int32(x / cell))
    return min(max(i, 0), h - 1), min(max(j, 0), w - 1)


@njit(cache=True)
def _relative(v, own, count):
    """featurize.relative_id for one id."""
    v = np.int64(v)
    out = v + 1 if v < own else v
    if v == own:
        out = 0
    if v < 0 or out >= count:
        out = count
    return out


@njit(cache=True)
def _pairwise_block(a, lo, n):
    """numpy's pairwise sum of float32 (loops_utils.h.src) for up to 128 values."""
    if n < 8:
        res = -_F0
        for i in range(n):
            res += a[lo + i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def _mean(a):
    """numpy's mean of float32 to the bit: its pairwise sum, which halves past 128 values (at a multiple of 8) and adds the halves.
    The halving runs on a stack of its own, not by recursion, which numba's cache can't load."""
    n = len(a)
    if n <= 128:
        total = _pairwise_block(a, 0, n)
    else:
        seg_lo = np.empty(64, np.int64)
        seg_n = np.empty(64, np.int64)
        phase = np.empty(64, np.int64)
        sums = np.empty(64, np.float32)
        top, done = 0, 0
        seg_lo[0], seg_n[0], phase[0] = 0, n, 0
        while top >= 0:
            lo, m = seg_lo[top], seg_n[top]
            if m <= 128:
                sums[done] = _pairwise_block(a, lo, m)
                done += 1
                top -= 1
                continue
            half = m // 2
            half -= half % 8
            if phase[top] == 2:             # both halves summed: add them, left first
                sums[done - 2] = sums[done - 2] + sums[done - 1]
                done -= 1
                top -= 1
                continue
            phase[top] += 1
            seg_lo[top + 1], seg_n[top + 1] = (lo, half) if phase[top] == 1 else (lo + half, m - half)
            phase[top + 1] = 0
            top += 1
        total = sums[0]
    return (_F0 + total) / np.float32(n)


@njit(cache=True)
def _lexorder(k1, k2, k3):
    """np.lexsort((k3, k2, k1)): by k1, then k2, then k3, stably."""
    o = np.argsort(k3, kind="mergesort")
    o = o[np.argsort(k2[o], kind="mergesort")]
    return o[np.argsort(k1[o], kind="mergesort")]


@njit(cache=True)
def select(units, row_of_def, tf, tb, player, ally, cap_controllable, cap_units):
    """featurize._select: the catalog row of every record, the indices kept (controllable first), how many are controllable,
    those left over, our army's centroid, and the enemy's (with whether any is in sight or on radar)."""
    n = len(units)
    rows = np.empty(n, np.int64)
    can = np.zeros(n, np.bool_)
    x = np.empty(n, np.float32)
    z = np.empty(n, np.float32)
    goes = np.empty(n, np.int64)
    own = np.empty(n, np.int64)
    seen = np.empty(n, np.int64)
    ng = no = ns = 0
    for i in range(n):
        u = units[i]
        r = _row(row_of_def, u["def_id"])
        rows[i] = r
        x[i], z[i] = u["x"], u["z"]
        flags = u["flags"]
        mine = u["player"] == player
        fights = tb[r, B_COMBAT] and tb[r, B_MOBILE]
        can[i] = (mine and (flags & BEING_BUILT) == 0 and (fights or tb[r, B_ACTS_IN_PLACE] or tb[r, B_CONSTRUCTOR] or tb[r, B_TRANSPORT])
                  and (flags & RESERVED) == 0)
        if can[i] and tb[r, B_MOBILE]:      # a turret isn't where the army is
            goes[ng] = i
            ng += 1
        if mine:
            own[no] = i
            no += 1
        if u["ally_team"] != ally and (u["knowledge"] == KNOW_LOS or u["knowledge"] == KNOW_RADAR):
            seen[ns] = i
            ns += 1
    army = np.zeros(2, np.float32)
    if ng:
        army[0], army[1] = _mean(x[goes[:ng]]), _mean(z[goes[:ng]])
    elif no:
        army[0], army[1] = _mean(x[own[:no]]), _mean(z[own[:no]])
    foe = np.zeros(2, np.float32)
    if ns:
        foe[0], foe[1] = _mean(x[seen[:ns]]), _mean(z[seen[:ns]])

    # Most expensive first; among equals, nearest the army.  uid breaks ties so that record order can't matter.
    cand = np.flatnonzero(can)
    near = np.empty(len(cand), np.float32)
    neg_cost = np.empty(len(cand), np.float32)
    uid = np.empty(len(cand), np.int64)
    for k in range(len(cand)):
        i = cand[k]
        near[k] = np.hypot(x[i] - army[0], z[i] - army[1])
        neg_cost[k] = -tf[rows[i], T_COST]
        uid[k] = units[i]["uid"]
    control = cand[_lexorder(neg_cost, near, uid)][:cap_controllable]
    taken = np.zeros(n, np.bool_)
    for i in control:
        taken[i] = True
    # The rest: what can be aimed at first, in sight before on radar; within each, value over distance from our army.
    rest = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if units[i]["knowledge"] != KNOW_HIDDEN and not taken[i]:
            rest[m] = i
            m += 1
    rest = rest[:m]
    rank = np.empty(m, np.int64)
    neg_score = np.empty(m, np.float32)
    uid = np.empty(m, np.int64)
    for k in range(m):
        i = rest[k]
        u = units[i]
        d = np.hypot(x[i] - army[0], z[i] - army[1])
        neg_score[k] = -(_max(tf[rows[i], T_COST], _LEAST_COST) / (d + _F200))
        hostile = u["ally_team"] != ally
        rank[k] = 0 if hostile and u["knowledge"] == KNOW_LOS else (1 if hostile and u["knowledge"] == KNOW_RADAR else 2)
        uid[k] = u["uid"]
    rest = rest[_lexorder(rank, neg_score, uid)]
    room = cap_units - len(control)
    picked = np.concatenate((control, rest[:room]))
    return rows, picked, len(control), rest[room:].copy(), army, foe, ns > 0


@njit(cache=True)
def _instance_row(u, r, i, stats, tf, height, cell, map_w, map_h, out):
    """Type stats and the instance columns for one unit."""
    at = stats.shape[1]
    for k in range(at):
        out[i, k] = stats[r, k]
    x, z = u["x"], u["z"]
    out[i, at + 0] = x / map_w * _F2 - _F1
    out[i, at + 1] = z / map_h * _F2 - _F1
    # Live, a unit's y is its middle, up to 35 elmos above the ground by type; the sim says the ground.  Past that, it flies.
    ci, cj = _cell(x, z, cell, height.shape[0], height.shape[1])
    above = u["y"] - height[ci, cj]
    out[i, at + 2] = _clip(_sign(above) * _max(abs(above) - _DEAD_ZONE, _F0) / _F200, -_F1, _F4)
    top = _max(tf[r, T_SPEED], _F30) / _F30        # elmos per frame, as vel is
    vx = _clip(u["vx"] / top, -_F2, _F2)
    vz = _clip(u["vz"] / top, -_F2, _F2)
    out[i, at + 3], out[i, at + 4] = vx, vz
    out[i, at + 5] = _min(np.hypot(vx, vz), _F2)
    hp = u["hp_frac"]
    missing = hp != hp
    out[i, at + 6] = _F0 if missing else _clip(hp, _F0, _F1)
    out[i, at + 7] = _F1 if missing else _F0
    out[i, at + 8] = _clip(u["build_frac"], _F0, _F1)
    flags = u["flags"]
    for k in range(len(_INSTANCE_BITS)):
        out[i, at + 9 + k] = _F1 if (flags & _INSTANCE_BITS[k]) != 0 else _F0
    out[i, at + 16] = u["blip_top_speed"] / _F150
    # A remembered structure's ord_age says how long ago it was seen.
    out[i, at + 17] = np.log1p(_max(u["ord_age"], _F0) / _F30) / _F6 if u["knowledge"] == KNOW_MEMORY else _F0
    out[i, at + 18] = _F1 if (flags & JUMP_READY) != 0 else _F0
    out[i, at + 19] = _F1 if (flags & CARRIED) != 0 else _F0
    out[i, at + 20] = _F1 if (flags & LOADED) != 0 else _F0


@njit(cache=True)
def _lifts(tf, me, load):
    """Whether a transport of row `me` can carry a unit of row `load`: a class it lifts, and a width it takes (tables.py)."""
    k = np.int64(tf[load, T_LOAD_CLASS])
    return k > 0 and tf[me, T_LIFT_LIGHT + k - 1] > _F0 and tf[load, T_LOAD_SIZE] <= tf[me, T_LIFT_SIZE]


@njit(cache=True)
def instance_block(units, rows, stats, tf, height, cell, map_w, map_h, out):
    """Type stats and instance columns, for featurize._hidden."""
    for i in range(len(units)):
        _instance_row(units[i], rows[i], i, stats, tf, height, cell, map_w, map_h, out)


@njit(cache=True)
def unit_block(units, rows, all_units, stats, tf, tb, move_class, label, height, passable, cell, map_w, map_h, map_max,
               player, ally, commit_frames, ally_count, player_count, out):
    """The kept units' token rows (type, instance, order memory, rays, DPS around), and what the masks need of them: who is
    commit-locked, each one's region, alliance and player from the viewer, the target tokens (with whether each is a foe's),
    our finished beacons and our own tokens."""
    n = len(units)
    o_at = stats.shape[1] + INSTANCE_COLS
    r_at = o_at + ORDER_COLS
    locked = np.zeros(n, np.bool_)
    region = np.empty(n, np.int32)
    rel_ally = np.empty(n, np.int32)
    rel_player = np.empty(n, np.int32)
    is_target = np.zeros(n, np.bool_)
    is_foe = np.zeros(n, np.bool_)
    is_beacon = np.zeros(n, np.bool_)
    is_own = np.zeros(n, np.bool_)

    # Whatever can be aimed at, kept or not, by uid: where an order's target is now.
    aim = np.empty(len(all_units), np.int64)
    na = 0
    for j in range(len(all_units)):
        a = all_units[j]
        if a["ally_team"] != ally and (a["knowledge"] == KNOW_LOS or a["knowledge"] == KNOW_RADAR):
            aim[na] = j
            na += 1
    aim_uid = np.empty(na, np.int64)
    for k in range(na):
        aim_uid[k] = all_units[aim[k]]["uid"]
    order = np.argsort(aim_uid)
    sorted_uid = aim_uid[order]

    # With a transport of ours among them, what of ours it could pick up is a target too.
    carries = False
    for i in range(n):
        u = units[i]
        carries = carries or (u["player"] == player and tb[rows[i], B_TRANSPORT] and (u["flags"] & BEING_BUILT) == 0)

    rows_p, cols_p = passable.shape
    for i in range(n):
        u = units[i]
        r = rows[i]
        _instance_row(u, r, i, stats, tf, height, cell, map_w, map_h, out)
        ci, cj = _cell(u["x"], u["z"], cell, label.shape[0], label.shape[1])
        region[i] = label[ci, cj]
        rel_ally[i] = _relative(u["ally_team"], ally, ally_count)
        rel_player[i] = _relative(u["player"], player, player_count)
        flags = u["flags"]
        own = u["player"] == player
        hostile = u["ally_team"] != ally
        built = (flags & BEING_BUILT) == 0
        is_own[i] = own
        is_foe[i] = hostile
        # A target is any other alliance's in sight or on radar, or one of ours to help (being built, or a factory) or to carry.
        loads = carries and built and tf[r, T_LOAD_CLASS] > _F0 and (flags & CARRIED) == 0
        is_target[i] = (hostile and (u["knowledge"] == KNOW_LOS or u["knowledge"] == KNOW_RADAR)) or (own and (not built or tb[r, B_FACTORY] or loads))
        is_beacon[i] = own and tb[r, B_BEACON] and built

        # Order memory.
        kind = np.int64(u["ord_kind"])
        has = kind >= 0 and own
        if has:
            out[i, o_at + kind] = _F1
        age = u["ord_age"] if has else _F0
        out[i, o_at + 16] = np.log1p(_max(age, _F0) / _F15) / _F6
        target = u["ord_target"]
        has_target = has and target >= 0 and kind != PRODUCE      # what a factory makes is a type, not a unit
        known = False
        alive = False
        goal_x, goal_z = u["ord_x"], u["ord_z"]
        if has_target and na:
            k = min(np.searchsorted(sorted_uid, target), na - 1)
            if sorted_uid[k] == target:
                # A target's direction is where it is now, in sight or on radar.  Any other goal is where the order said.
                a = all_units[aim[order[k]]]
                known = True
                goal_x, goal_z = a["x"], a["z"]
                alive = a["knowledge"] == KNOW_LOS
        goes = known or (has and kind != ATTACK_UNIT and kind != STOP)
        dx = goal_x - u["x"] if goes else _F0
        dz = goal_z - u["z"] if goes else _F0
        out[i, o_at + 17] = dx / map_w
        out[i, o_at + 18] = dz / map_h
        out[i, o_at + 19] = np.hypot(dx, dz) / map_max
        out[i, o_at + 20] = _F1 if has_target else _F0
        out[i, o_at + 21] = _F1 if has_target and alive else _F0
        fresh = has and age < commit_frames
        out[i, o_at + RESERVED_KIND] = _F1 if own and (flags & PULLING) != 0 else _F0      # no order has this kind
        released = (flags & (DAMAGED | IDLE)) != 0 or (has_target and not alive)
        locked[i] = fresh and not released
        out[i, o_at + 22] = _F1 if locked[i] else _F0

        # Free distance in eight directions, by its own move class; air units see nothing in the way.
        cls = -1 if tb[r, B_AIR] else np.int64(move_class[r])
        for d in range(8):
            first = RAY_CELLS
            if cls >= 0:
                for k in range(RAY_CELLS):
                    pi = np.int64(np.int32(np.floor((u["z"] + _RAY_DZ[d] * _RAY_T[k]) / _PASS_CELL)))
                    pj = np.int64(np.int32(np.floor((u["x"] + _RAY_DX[d] * _RAY_T[k]) / _PASS_CELL)))
                    inside = pi >= 0 and pi < rows_p and pj >= 0 and pj < cols_p
                    v = passable[min(max(pi, 0), rows_p - 1), min(max(pj, 0), cols_p - 1)]
                    if not (inside and ((v >> cls) & 1) != 0):
                        first = k
                        break
            out[i, r_at + d] = np.float32(first / RAY_CELLS)

    # Hostile and friendly DPS within RAY_DPS_RADIUS per 45-degree sector, from each token's own point of view; summed in float64
    # in the order numpy's bincount took them.  Sector 0 is around +z, as the rays are: the angle is measured from z towards x.
    enemy_sum = np.zeros((n, 8), np.float64)
    own_sum = np.zeros((n, 8), np.float64)
    pairs = False
    for a in range(n):
        ua = units[a]
        for b in range(n):
            if a == b:
                continue
            ub = units[b]
            dx = ub["x"] - ua["x"]
            dz = ub["z"] - ua["z"]
            if dx * dx + dz * dz <= _DPS_RADIUS2:
                pairs = True
                sector = np.int64(np.int32((np.arctan2(dx, dz) + _PI8) / _PI4)) % 8
                dps = np.float64(tf[rows[b], T_DPS_GROUND] + tf[rows[b], T_DPS_AIR])
                if ua["ally_team"] == ub["ally_team"]:
                    own_sum[a, sector] += dps
                else:
                    enemy_sum[a, sector] += dps
    if pairs:
        for a in range(n):
            for s in range(8):
                out[a, r_at + 8 + s] = np.float32(np.log1p(enemy_sum[a, s]) / 8.0)
                out[a, r_at + 16 + s] = np.float32(np.log1p(own_sum[a, s]) / 8.0)
    targets = np.flatnonzero(is_target)
    return locked, region, rel_ally, rel_player, targets, is_foe[targets], np.flatnonzero(is_beacon), np.flatnonzero(is_own)


@njit(cache=True)
def region_block(units, rows, features, tf, label, cell, ally, out):
    """featurize._regions' columns from what's on the map now: value, DPS and memories by region, and the metal in sight."""
    r = out.shape[0]
    h, w = label.shape
    if len(features):
        metal = np.zeros(r, np.float64)
        for k in range(len(features)):
            f = features[k]
            ci, cj = _cell(f["x"], f["z"], cell, h, w)
            at = np.int64(label[ci, cj])
            if at >= 0:         # all of what's in sight, kept as a token or not; a point off every region counts for none
                metal[at] += np.float64(f["metal"])
        for k in range(r):
            out[k, 33] = np.float32(np.log1p(metal[k]) / 10.0)
    if not len(units):
        return
    sums = np.zeros((6, r), np.float64)     # value of friends, of foes in sight, on radar; DPS of friends, of foes in sight; memories
    for i in range(len(units)):
        u = units[i]
        row = rows[i]
        ci, cj = _cell(u["x"], u["z"], cell, h, w)
        at = np.int64(label[ci, cj])
        hp = u["hp_frac"]
        if hp != hp:
            hp = _F1
        value = np.float64(tf[row, T_COST] * _clip(hp, _F0, _F1))
        know = u["knowledge"]
        if u["ally_team"] == ally:
            sums[0, at] += value
            sums[3, at] += np.float64(tf[row, T_DPS_GROUND])
        elif know == KNOW_LOS:
            sums[1, at] += value
            sums[4, at] += np.float64(tf[row, T_DPS_GROUND])
        elif know == KNOW_RADAR:
            sums[2, at] += value
        elif know == KNOW_MEMORY:
            sums[5, at] += 1.0
    for k in range(r):
        for j in range(3):
            out[k, 27 + j] = np.float32(np.log1p(sums[j, k]) / 10.0)
        out[k, 30] = np.float32(np.log1p(sums[3, k]) / 8.0)
        out[k, 31] = np.float32(np.log1p(sums[4, k]) / 8.0)
        out[k, 32] = np.float32(np.log1p(sums[5, k]) / 3.0)


@njit(cache=True)
def masks(units, rows, region, targets, foes, beacons, features, locked, c, tf, tb, move_class, produces, builds, producible,
          row_of_def, walk, region_cells, centre, off_centre, enabled):
    """featurize's legal actions: kinds, target pointers, regions to walk, jump or wait at, types to make, and walking times."""
    r = len(centre)
    e = len(targets)
    t = len(producible)
    kind = np.zeros((c, KIND_COUNT), np.bool_)
    ptr_enemy = np.zeros((5, c, e), np.bool_)
    ptr_region = np.zeros((c, r), np.bool_)
    ptr_jump = np.zeros((c, r), np.bool_)
    ptr_wait = np.zeros((c, r), np.bool_)
    route = np.zeros((c, r), np.float32)
    # A factory's or silo's PRODUCE and a constructor's BUILD both choose a type of `producible`; not what it's making already.
    ptr_produce = np.zeros((c, t), np.bool_)
    for i in range(c):
        me = rows[i]
        for k in range(t):
            ptr_produce[i, k] = produces[me, k] or builds[me, k]
        u = units[i]
        if t and u["ord_kind"] == PRODUCE and u["ord_target"] >= 0:
            made = _row(row_of_def, u["ord_target"])
            at = min(np.searchsorted(producible, made), t - 1)
            if producible[at] == made:
                ptr_produce[i, at] = False
    with_beacon = np.zeros(r, np.bool_)
    for b in beacons:
        if region[b] >= 0:
            with_beacon[region[b]] = True
    for i in range(c):
        u = units[i]
        me = rows[i]
        flags = u["flags"]
        free = not locked[i]
        mobile, armed = tb[me, B_MOBILE], tb[me, B_ARMED]
        flies = tb[me, B_AIR]
        cls = np.int64(-1) if flies else np.int64(move_class[me])
        here = np.int64(region[i])
        # Regions it can walk to, by its class; its own always counts, so a unit always has a legal MOVE.  What flies goes
        # anywhere, straight there.
        if flies and mobile:
            speed = _max(tf[me, T_SPEED], _F1)
            for k in range(r):
                ptr_region[i, k] = True
                route[i, k] = np.float32(np.hypot(centre[k, 0] - u["x"], centre[k, 1] - u["z"])) / speed
        elif cls >= 0 and here >= 0:
            speed = _max(tf[me, T_SPEED], _F1)
            for k in range(r):
                w = walk[cls, here, k]
                finite = np.isfinite(w)
                ptr_region[i, k] = finite
                route[i, k] = (w if finite else _F0) / speed
        if here >= 0:
            ptr_region[i, here] = True
        can_go = False
        for k in range(r):
            can_go = can_go or ptr_region[i, k]

        any0 = any1 = any2 = any3 = any4 = False
        lifts = tb[me, B_TRANSPORT] and (flags & LOADED) == 0
        for j in range(e):
            tk = targets[j]
            tr = rows[tk]
            tu = units[tk]
            in_los = tu["knowledge"] == KNOW_LOS
            dist = np.hypot(u["x"] - tu["x"], u["z"] - tu["z"])
            air_target = tb[tr, B_AIR]
            foe = foes[j]
            hits = (tb[me, B_HIT_GROUND] and not air_target) or (tb[me, B_HIT_AIR] and air_target)
            p0 = hits and in_los and foe
            p1 = armed and dist <= tf[me, T_RANGE] * _F1_5 + _F300 and foe
            p2 = tb[me, B_MANUALFIRE] and in_los and dist <= tf[me, T_MANUAL_RANGE] and foe
            # Missiles and the Trinity: at what's on the ground in reach, in sight or on radar, as every enemy token is.
            if tb[me, B_MISSILE]:
                p0 = not air_target and foe and dist <= tf[me, T_RANGE]
            if tb[me, B_LAUNCHER]:
                p2 = not air_target and foe and dist <= tf[me, T_MANUAL_RANGE]
            # ASSIST: a constructor helps build one of ours anywhere (it walks there), or helps a factory with what it makes.
            p3 = tb[me, B_CONSTRUCTOR] and not foe and tk != i
            # LOAD: a transport with nothing in it, at one of ours it can carry, or at another alliance's in sight that stands still.
            p4 = (lifts and tk != i and _lifts(tf, me, tr) and (tu["flags"] & (BEING_BUILT | CARRIED)) == 0
                  and (not foe or (in_los and np.hypot(tu["vx"], tu["vz"]) <= _STILL)))
            ptr_enemy[0, i, j], ptr_enemy[1, i, j], ptr_enemy[2, i, j], ptr_enemy[3, i, j], ptr_enemy[4, i, j] = p0, p1, p2, p3, p4
            any0, any1, any2, any3, any4 = any0 or p0, any1 or p1, any2 or p2, any3 or p3, any4 or p4

        kind[i, KEEP] = (flags & IDLE) == 0 or not mobile
        kind[i, MOVE] = free and mobile and can_go
        kind[i, FIGHT] = free and mobile and armed and can_go
        kind[i, ATTACK_UNIT] = free and any0
        kind[i, ATTACK_GROUND] = free and tb[me, B_HIT_GROUND] and any1
        kind[i, STOP] = free and (mobile or armed)
        fired = u["ord_kind"] == MANUAL_FIRE and u["ord_age"] < tf[me, T_MANUAL_RELOAD]
        kind[i, MANUAL_FIRE] = free and any2 and not fired
        # Where a jump or a throw is told to is a region's centre or near it: one it could walk to, or one in reach (fine positions
        # are this far off the centre).  Not where it couldn't land, which the game refuses.
        ready = cls >= 0 and here >= 0 and (flags & JUMP_READY) != 0
        charged = tb[me, B_JUMP] and ready
        throws = tf[me, T_THROW_RADIUS] > _F0 and ready
        can_land = False
        if charged or throws:
            loaded = charged
            if not charged:     # only with someone beside it to throw (CheckOverlobPrevention): whoever walks, and is ours or in sight
                for j in range(len(units)):
                    o = units[j]
                    ro = rows[j]
                    if (tb[ro, B_MOBILE] and not tb[ro, B_AIR] and o["knowledge"] == KNOW_LOS and (o["flags"] & BEING_BUILT) == 0
                            and o["uid"] != u["uid"] and np.hypot(o["x"] - u["x"], o["z"] - u["z"]) <= tf[me, T_THROW_RADIUS]):
                        loaded = True
                        break
            if loaded:
                reach = tf[me, T_JUMP_RANGE] if charged else tf[me, T_THROW_REACH]
                for k in range(r):
                    away = np.hypot(centre[k, 0] - u["x"], centre[k, 1] - u["z"])
                    lands = ((region_cells[k] >> cls) & 1) != 0
                    near = away <= reach + off_centre[k]
                    ptr_jump[i, k] = ((ptr_region[i, k] or near) if charged else near) and lands
                    can_land = can_land or ptr_jump[i, k]
        kind[i, JUMP] = free and charged and can_land
        kind[i, THROW] = free and throws and can_land
        # UNLOAD: a loaded transport sets its load down where the load can stand, which goes by the load's class: the carried unit
        # at its place.  Not among the tokens, any region something can stand in.  DROP lets it fall, whatever else it's doing: the
        # button's 7 s of reload aren't in the frame, and a drop too soon does nothing.
        loaded = tb[me, B_TRANSPORT] and (flags & LOADED) != 0
        can_set = False
        if loaded:
            load_cls = np.int64(-1)
            for j in range(len(units)):
                o = units[j]
                if ((o["flags"] & CARRIED) != 0 and abs(o["x"] - u["x"]) <= _AT_ITS_TRANSPORT and abs(o["z"] - u["z"]) <= _AT_ITS_TRANSPORT
                        and not tb[rows[j], B_AIR]):
                    load_cls = np.int64(move_class[rows[j]])
                    break
            for k in range(r):
                ptr_jump[i, k] = ((region_cells[k] >> load_cls) & 1) != 0 if load_cls >= 0 else region_cells[k] != 0
                can_set = can_set or ptr_jump[i, k]
        kind[i, LOAD] = free and any4
        kind[i, UNLOAD] = free and loaded and can_set
        kind[i, DROP] = loaded
        cloaked = (flags & CLOAKED) != 0
        kind[i, CLOAK_ON] = free and tb[me, B_CLOAK] and not cloaked
        kind[i, CLOAK_OFF] = free and tb[me, B_CLOAK] and cloaked
        pulling = (flags & PULLING) != 0
        kind[i, PUSH] = free and tb[me, B_SWITCHES] and pulling
        kind[i, PULL] = free and tb[me, B_SWITCHES] and not pulling
        teleports = tb[me, B_TELEPORTS]
        kind[i, PLACE_BEACON] = free and teleports and mobile and can_go
        waits = False
        for k in range(r):
            ptr_wait[i, k] = ptr_region[i, k] and with_beacon[k] and mobile and not teleports
            waits = waits or ptr_wait[i, k]
        kind[i, WAIT_AT_BEACON] = free and waits
        makes = False
        for k in range(t):
            makes = makes or ptr_produce[i, k]
        kind[i, PRODUCE] = free and tb[me, B_PRODUCES] and makes
        kind[i, BUILD] = free and tb[me, B_BUILDS]          # there's always a place: beside itself
        kind[i, ASSIST] = free and any3
        kind[i, RECLAIM] = free and tb[me, B_CONSTRUCTOR] and features > 0      # any feature token: it walks there
        for k in range(KIND_COUNT):
            if not enabled[k]:
                kind[i, k] = False
    return kind, ptr_enemy, ptr_region, ptr_jump, ptr_wait, ptr_produce, route
