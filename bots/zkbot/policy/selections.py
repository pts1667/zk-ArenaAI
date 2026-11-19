"""Orders as a player gives them: select, then command (the user, 2026-11-19; docs/learned-policy/06-select-then-command.md).

A side's orders at a decision are a few commands, in turn.  Each selects some of the side's units, by their roles, a circle
and a portion of what the circle holds, and tells the selection one thing.  Nothing in a command points into the frame: a
place is a fraction of the map each way, a type is its place in `table.producible`, and who a place means (the unit to
select, the enemy to attack, the frame to help) is found when the command is carried out (`decode`).  So a decision's
commands can be written, read and autoencoded with no frame beside them (05-order-autoencoder.md).

The kinds are every kind the sim obeys from the policy (policy_sim_env.KINDS) but keep.  The scripts say twelve of them; the
rest (a jump, a throw, the cloak, push and pull, the beacon's two, the area mex) an autoencoder of these learns from
commands made up (order_vae.synthetic; the user, 2026-11-19).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import featurize as fz
from . import groups as gr
from . import rawframe as rf
from .actions import Order, _producible_index, _within_throw
from .bundle import Bundle
from .tables import table

KINDS = (rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.BUILD, rf.ASSIST, rf.CANCEL_BUILD, rf.RECLAIM, rf.PRODUCE,
         rf.LOAD, rf.UNLOAD, rf.DROP,                                                               # the scripts', and the cancel
         rf.JUMP, rf.THROW, rf.CLOAK_ON, rf.CLOAK_OFF, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.AREA_MEX)
PLACED = (rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.BUILD, rf.ASSIST, rf.RECLAIM, rf.LOAD, rf.UNLOAD,
          rf.JUMP, rf.THROW, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.AREA_MEX)                       # x, z mean something
TYPED = (rf.BUILD, rf.PRODUCE)                                                                    # and its type
AREAS = (rf.RECLAIM, rf.AREA_MEX)                                                                 # and its extent
AIMED = (rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.ASSIST, rf.LOAD)   # the place stands for the unit nearest it that the kind can be aimed at
# The place must be in a region the unit can do it in, by this mask of the frame's.  A MOVE or a FIGHT goes wherever it's told:
# where there's no way, the sim takes a unit as near as a straight line gets (movement.cpp), which is what the scripts' own
# orders rely on.
REGION_MASK = {rf.UNLOAD: "ptr_jump", rf.JUMP: "ptr_jump", rf.THROW: "ptr_jump", rf.PLACE_BEACON: "ptr_region"}
RADII = (0.0,) + gr.GROUP_RADII     # a selection's circle, in elmos.  0: the one unit nearest its middle
PORTIONS = (1.0, 0.5, 0.25)         # of what the circle holds, the nearest to its middle first: FULL, HALF, QUARTER
RECLAIM_LEAST = 32.0                # the sim's least area for a reclaim (capi.h ZK_RECLAIM_RADIUS): an extent of 0
SLOTS = 16                          # commands a decision at most
SAME_ORDER = 200.0                  # elmos: a MOVE or FIGHT this near a unit's last is the same order (army.cpp REORDER)

# One command.  roles: the roles selected, as bits (groups.roles).  radius, portion: places in RADII and PORTIONS.  cx, cz: the
# circle's middle, and x, z: the command's place (PLACED), each as a fraction of the map.  kind: rawframe's number, one of KINDS.
# type: the place in table.producible of what to build or make (TYPED), else -1.  extent: for a RECLAIM or an AREA_MEX
# (AREAS), the place in RADII of its area's radius (0: RECLAIM_LEAST), else -1.
COMMAND_DTYPE = np.dtype([("roles", "u4"), ("radius", "i1"), ("portion", "i1"), ("kind", "i1"), ("extent", "i1"), ("type", "i2"),
                          ("cx", "f4"), ("cz", "f4"), ("x", "f4"), ("z", "f4")])


def command(roles: int, cx: float, cz: float, radius: int, kind: int, x: float = 0.0, z: float = 0.0, type: int = -1,
            extent: int = -1, portion: int = 0) -> np.ndarray:
    out = np.zeros((), COMMAND_DTYPE)
    out["roles"], out["radius"], out["portion"], out["kind"], out["extent"], out["type"] = roles, radius, portion, kind, extent, type
    out["cx"], out["cz"], out["x"], out["z"] = cx, cz, x, z
    return out


@lru_cache(maxsize=None)
def able() -> tuple[np.ndarray, np.ndarray]:
    """What a command can say and be obeyed in, by the type table alone (the user, 2026-11-19: "commands should be
    plausible/valid").  [k, role]: whether any type of the role can ever be told KINDS[k], by what the frame's kind mask asks of a
    unit's type (featurize_kernels).  [k, role, p]: whether one of the role can build or make table.producible[p] under
    KINDS[k]; none for a kind with no type (TYPED).  Whether a unit can now is the frame's to say (`_obeying`)."""
    t = table()
    names, role_of = gr.roles()
    n = len(t.names)
    of_role = role_of[:n, None] == np.arange(len(names))        # type x role
    flag = lambda a: np.asarray(a, bool)[:n]                     # noqa: E731
    mobile, constructor, transport, cloak, switches = flag(t.mobile), flag(t.constructor), flag(t.transport), flag(t.cloak), flag(t.switches)
    builds, produces = np.asarray(t.builds, bool)[:n], np.asarray(t.produces, bool)[:n]
    can = {rf.MOVE: mobile, rf.FIGHT: mobile & flag(t.armed), rf.ATTACK_UNIT: flag(t.can_hit_ground) | flag(t.can_hit_air) | flag(t.missile),
           rf.MANUAL_FIRE: flag(t.manualfire) | flag(t.launcher), rf.BUILD: builds.any(1), rf.ASSIST: constructor, rf.CANCEL_BUILD: builds.any(1),
           rf.RECLAIM: constructor, rf.PRODUCE: produces.any(1), rf.LOAD: transport, rf.UNLOAD: transport, rf.DROP: transport,
           rf.JUMP: flag(t.jump) & mobile, rf.THROW: np.asarray(t.throw_radius)[:n] > 0.0, rf.CLOAK_ON: cloak, rf.CLOAK_OFF: cloak,
           rf.PUSH: switches, rf.PULL: switches, rf.PLACE_BEACON: flag(t.teleports) & mobile, rf.WAIT_AT_BEACON: mobile & ~flag(t.teleports),
           rf.AREA_MEX: flag(t.area_mex)}
    by_role = np.stack([(can[k][:, None] & of_role).any(0) for k in KINDS])
    makes = np.zeros((len(KINDS), len(names), len(t.producible)), bool)
    for k, options in ((rf.BUILD, builds), (rf.PRODUCE, produces)):
        makes[KINDS.index(k)] = (options[:, None, :] & of_role[:, :, None]).any(0)
    assert by_role.any(1).all() and (makes.any(2) == by_role)[[KINDS.index(k) for k in TYPED]].all()
    return by_role, makes


def unit_roles(t: fz.Tensors) -> np.ndarray:
    """C: the role of each controllable unit, as groups.roles numbers them."""
    names = t.unit_name if t.true_name is None else t.true_name
    return gr.roles()[1][np.asarray(names[:t.controllable], np.int64)]


def selected(c: np.ndarray, role: np.ndarray, free: np.ndarray, unit_x: np.ndarray, unit_z: np.ndarray, map_w: float, map_h: float) -> np.ndarray:
    """The controllable units a command selects, as indices, the nearest to its circle's middle first: those of its roles
    that are `free` (not told anything yet in this decision) within its radius, and of them its portion, at least one.  With
    none in the circle, or a radius of 0, the one nearest its middle: a selection is empty only when the side has no such
    unit."""
    of_roles = (((int(c["roles"]) >> role) & 1) == 1) & free
    if not of_roles.any():
        return np.zeros(0, np.int64)
    away = np.hypot(unit_x[:len(role)] - float(c["cx"]) * map_w, unit_z[:len(role)] - float(c["cz"]) * map_h)
    away[~of_roles] = np.inf
    inside = np.flatnonzero(away <= RADII[int(c["radius"])])
    if int(c["radius"]) == 0 or not len(inside):
        return np.array([int(np.argmin(away))], np.int64)
    inside = inside[np.argsort(away[inside], kind="stable")]
    return inside[:math.ceil(PORTIONS[int(c["portion"])] * len(inside))]


def _obeying(cmd: np.ndarray, t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray, role: np.ndarray, free: np.ndarray,
             aim_x: np.ndarray, aim_z: np.ndarray) -> tuple[np.ndarray, float, float, int, float]:
    """The units of a command's selection that obey it, and the order each is given: its x, z, target and radius."""
    none = np.zeros(0, np.int64)
    kind = int(cmd["kind"])
    chosen = selected(cmd, role, free, unit_x, unit_z, bundle.map_w, bundle.map_h)
    chosen = chosen[t.kind_mask[chosen, kind]]
    if not len(chosen):
        return none, 0.0, 0.0, -1, 0.0
    x, z = float(cmd["x"]) * bundle.map_w, float(cmd["z"]) * bundle.map_h
    target, radius = -1, 0.0
    if kind in REGION_MASK:
        region = int(bundle.region_at(x, z))
        if region < 0:          # a place in no region: nobody's to go to
            return none, 0.0, 0.0, -1, 0.0
        chosen = chosen[getattr(t, REGION_MASK[kind])[chosen, region]]
    elif kind in AIMED:
        legal = t.ptr_enemy[fz.ENEMY_POINTER_KINDS.index(kind)][chosen]         # n x E
        aimed_at = np.flatnonzero(legal.any(0))
        if not len(aimed_at):
            return none, 0.0, 0.0, -1, 0.0
        e = int(aimed_at[np.argmin(np.hypot(aim_x[aimed_at] - x, aim_z[aimed_at] - z))])
        chosen = chosen[legal[:, e]]
        token = int(t.enemy_tokens[e])
        x, z, target = float(unit_x[token]), float(unit_z[token]), int(t.uids[token])
        if kind == rf.LOAD:             # one thing is one transport's load: the nearest that can lift it
            chosen = chosen[[int(np.argmin(np.hypot(unit_x[chosen] - x, unit_z[chosen] - z)))]]
    elif kind in TYPED:
        p = int(cmd["type"])
        if not 0 <= p < t.ptr_produce.shape[1]:
            return none, 0.0, 0.0, -1, 0.0
        chosen = chosen[t.ptr_produce[chosen, p]]
        target = int(table().producible[p])
        if kind == rf.PRODUCE:
            x = z = 0.0
    elif kind in AREAS:
        radius = max(RADII[max(int(cmd["extent"]), 0)], RECLAIM_LEAST)
        if kind == rf.RECLAIM and (not len(t.feat_pos) or not (np.hypot(t.feat_pos[:, 0] - x, t.feat_pos[:, 1] - z) <= radius).any()):
            return none, 0.0, 0.0, -1, 0.0
    elif kind == rf.WAIT_AT_BEACON:
        # The beacon of ours nearest the place that any of them can walk to; each that can, goes.
        ok = t.ptr_wait[chosen][:, np.maximum(t.beacon_regions, 0)] & (t.beacon_regions >= 0)     # n x beacons
        open_ = np.flatnonzero(ok.any(0))
        if not len(open_):
            return none, 0.0, 0.0, -1, 0.0
        tokens = t.own_beacons[open_]
        b = int(np.argmin(np.hypot(unit_x[tokens] - x, unit_z[tokens] - z)))
        chosen, target = chosen[ok[:, open_[b]]], int(t.uids[tokens[b]])
        x = z = 0.0
    elif kind not in PLACED:
        x = z = 0.0
    return chosen, x, z, target, radius


def decode(commands: np.ndarray, t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray) -> list[Order]:
    """The orders a decision's commands (COMMAND_DTYPE) come to, each a plain order to one unit.  Of a selection, each unit
    obeys that the command is legal for by the frame's own masks, and the rest keep what they're doing, as a player's units
    ignore a command they haven't got.  A unit that obeyed is in no later selection of the decision.

    Several told to BUILD get one type at one point: they build one thing (SimEnv._placed, the sim's ZK_ORDER_BUILD).  A
    RECLAIM is an area's, its order's radius: a unit is sent only where a feature in sight lies in it.  An AREA_MEX's
    radius is its area's too.  A THROW past a unit's reach goes as far as it can the same way (actions._within_throw)."""
    role, free = unit_roles(t), np.ones(t.controllable, bool)
    orders: list[Order] = []
    aim_x, aim_z = unit_x[t.enemy_tokens], unit_z[t.enemy_tokens]
    for cmd in commands:
        chosen, x, z, target, radius = _obeying(cmd, t, bundle, unit_x, unit_z, role, free, aim_x, aim_z)
        free[chosen] = False
        kind = int(cmd["kind"])
        for i in chosen:
            at = _within_throw(t, int(i), x, z, unit_x, unit_z) if kind == rf.THROW else (x, z)
            orders.append(Order(int(t.uids[i]), kind, *at, target, radius))
    return orders


def _fraction(v: float, size: float) -> float:
    return float(np.float32(min(max(v / size, 0.0), 1.0)))


def last_orders(t: fz.Tensors, units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U: the last order each unit token was given, from the frame's records (rawframe.UNIT_DTYPE): its kind, x and z.  -1 for
    a token with no record."""
    at = {int(u): k for k, u in enumerate(units["uid"])}
    rows = np.array([at.get(int(u), -1) for u in t.uids], np.int64)
    known = rows >= 0
    kind = np.where(known, units["ord_kind"][np.maximum(rows, 0)], -1).astype(np.int64)
    return kind, np.where(known, units["ord_x"][np.maximum(rows, 0)], 0.0), np.where(known, units["ord_z"][np.maximum(rows, 0)], 0.0)


def encode(orders: list[Order], t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray, slots: int | None = SLOTS,
           missed: dict | None = None, last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, int]:
    """`decode` the other way: orders someone else gave, such as a script, as commands that give each unit what the orders
    gave it.  A group order (groups.py) is first its units' own orders, as the sim selects them; a unit's own order replaces
    a group's, and of two of either the later counts.  Who an order is aimed at becomes the place it's at.  An order to a unit
    on a build (featurize_kernels.masks: nothing is legal for it but the cancel) becomes the cancel of that build and is
    itself left out: the unit can be told nothing else in the decision.

    Units told one thing are then one command where a selection takes them and nobody else: their roles, in the smallest
    circle round their middle that holds them, whole, or its nearest half or quarter.  Where none does, they're halved
    across the way they're spread and each half tried, down to a unit on its own: the one of its role nearest where it
    stands.  With `last` (`last_orders`), a circle may also take in whoever's last order was this very one, a MOVE or a FIGHT
    within SAME_ORDER of the place, and who is told nothing else now: told again, they go on as they were.  The most units
    first, so that what's cut at `slots` commands (None: all) is the least.

    Also returns how many orders couldn't be said, and counts them in `missed` by (reason, kind): a unit the side doesn't
    control, a kind that isn't among KINDS or isn't legal for the unit now, a place it can't go, a target that isn't in sight
    or that it can't aim at, a type it can't make.  `missed` also counts the units told one thing with others, and the
    commands that took."""
    c = t.controllable
    role = unit_roles(t)
    token_of = {int(u): k for k, u in enumerate(t.uids) if u >= 0}
    enemy_of = {int(token): e for e, token in enumerate(t.enemy_tokens)}
    w, h = float(bundle.map_w), float(bundle.map_h)
    mobile = np.asarray(table().mobile, bool)
    names = t.unit_name if t.true_name is None else t.true_name
    lost = 0

    def miss(why: str, kind: int) -> None:
        nonlocal lost
        lost += 1
        if missed is not None:
            missed[why, kind] = missed.get((why, kind), 0) + 1

    def told(i: int, o: Order) -> tuple | None:
        """What unit i is told by order o, as a command's (kind, x, z, type, extent), or None with the reason counted."""
        kind = int(o.kind)
        x, z, p, extent = float(o.x), float(o.z), -1, -1
        if kind == rf.ATTACK_UNIT:
            # An attack on what the side has no token of (a building it only remembers) or can't aim at is a fight at its
            # place: there's no saying whom, and that's what brings the unit to it.
            e = enemy_of.get(token_of.get(int(o.target), -1), -1)
            if not t.kind_mask[i, kind] or e < 0 or not t.ptr_enemy[fz.ENEMY_POINTER_KINDS.index(kind), i, e]:
                kind = rf.FIGHT
        if not t.kind_mask[i, kind]:
            if t.kind_mask[i, rf.CANCEL_BUILD]:
                return (rf.CANCEL_BUILD, 0.0, 0.0, -1, -1)
            return miss("kind not legal now", kind)
        if kind in REGION_MASK:
            if not getattr(t, REGION_MASK[kind])[i, int(bundle.region_at(x, z))]:
                return miss("a place it can't go", kind)
        elif kind in AIMED:
            token = token_of.get(int(o.target), -1)
            e = enemy_of.get(token, -1)
            if e < 0:
                return miss("a target not in sight", kind)
            if not t.ptr_enemy[fz.ENEMY_POINTER_KINDS.index(kind), i, e]:
                return miss("a target not its to aim at", kind)
            x, z = float(unit_x[token]), float(unit_z[token])
        elif kind in TYPED:
            p = _producible_index(int(o.target))
            if p < 0 or not t.ptr_produce[i, p]:
                return miss("a type it can't make", kind)
        elif kind == rf.RECLAIM:
            if not (t.feat_ids == int(o.target)).any():
                return miss("a feature not in sight", kind)
            extent = 0
        elif kind == rf.AREA_MEX:
            extent = min(gr.bucket(float(o.radius)) % len(gr.GROUP_RADII) + 1, len(RADII) - 1)
        return (kind, _fraction(x, w), _fraction(z, h), p, extent) if kind in PLACED else (kind, 0.0, 0.0, p, extent)

    want: dict[int, tuple] = {}
    plain = [o for o in orders if not (o.radius > 0.0 and o.kind in gr.GROUP_KINDS)]
    for o in [o for o in orders if o.radius > 0.0 and o.kind in gr.GROUP_KINDS] + plain:
        i = token_of.get(int(o.uid), -1)
        if not 0 <= i < c or int(o.kind) not in KINDS:
            miss("not the side's to command" if not 0 <= i < c else "a kind no command has", int(o.kind))
            continue
        members = [i]
        if o.radius > 0.0 and o.kind in gr.GROUP_KINDS:
            dx, dz = unit_x[:c] - unit_x[i], unit_z[:c] - unit_z[i]
            near = dx * dx + dz * dz <= np.float32(o.radius) * np.float32(o.radius)
            members = np.flatnonzero((mobile[names[:c]] & (((int(o.roles) >> role) & 1) == 1) & near) | (np.arange(c) == i)).tolist()
        for m in members:
            said = told(m, o)
            if said is not None:
                want.pop(m, None)       # the later counts, and comes later
                want[m] = said
    together: dict[tuple, list[int]] = {}
    for i, said in want.items():
        together.setdefault(said, []).append(i)
    free, wanted = np.ones(c, bool), np.zeros(c, bool)
    wanted[list(want)] = True
    aim_x, aim_z = unit_x[t.enemy_tokens], unit_z[t.enemy_tokens]
    commands: list[np.ndarray] = []

    def say(units: np.ndarray, kind: int, x: float, z: float, p: int, extent: int, on_it: np.ndarray) -> None:
        """Commands that tell these units this, and nobody else but those on it already: one circle if one takes them, else
        the two halves of them across the way they're spread, and at the last each its own."""
        if len(units) > 1:
            cx, cz = _fraction(float(unit_x[units].mean()), w), _fraction(float(unit_z[units].mean()), h)
            far = float(np.hypot(unit_x[units] - cx * w, unit_z[units] - cz * h).max())
            radius = int(np.searchsorted(np.asarray(RADII), far))
            roles = int(np.bitwise_or.reduce(1 << role[units]))
            for portion in range(len(PORTIONS) if radius < len(RADII) else 0):
                cmd = command(roles, cx, cz, radius, kind, x, z, p, extent, portion)
                chosen = _obeying(cmd, t, bundle, unit_x, unit_z, role, free, aim_x, aim_z)[0]
                if np.isin(units, chosen).all() and on_it[chosen[~np.isin(chosen, units)]].all():
                    free[chosen] = False
                    commands.append(cmd)
                    return
            wide = unit_x[units] if np.ptp(unit_x[units]) >= np.ptp(unit_z[units]) else unit_z[units]
            across = units[np.argsort(wide, kind="stable")]
            say(across[:len(units) // 2], kind, x, z, p, extent, on_it)
            say(across[len(units) // 2:], kind, x, z, p, extent, on_it)
            return
        i = int(units[0])
        cmd = command(1 << int(role[i]), _fraction(float(unit_x[i]), w), _fraction(float(unit_z[i]), h), 0, kind, x, z, p, extent)
        chosen = _obeying(cmd, t, bundle, unit_x, unit_z, role, free, aim_x, aim_z)[0]
        if len(chosen) != 1 or int(chosen[0]) != i:
            return miss("no command's own", kind)
        free[i] = False
        commands.append(cmd)

    for (kind, x, z, p, extent), units in sorted(together.items(), key=lambda item: -len(item[1])):
        # Whoever's last order was this one may be told it again: a circle may take them in (as the army script's own
        # group orders do, army.cpp `cover`).
        on_it = np.zeros(c, bool)
        if last is not None and kind in (rf.MOVE, rf.FIGHT):
            on_it = (last[0][:c] == kind) & (np.hypot(last[1][:c] - x * w, last[2][:c] - z * h) <= SAME_ORDER) & ~wanted
        before = len(commands)
        say(np.asarray(units, np.int64), kind, x, z, p, extent, on_it)
        if missed is not None and len(units) > 1:      # how those told one thing were said: units, and the commands they took
            missed["units told one thing with others", kind] = missed.get(("units told one thing with others", kind), 0) + len(units)
            missed["the commands that told them", kind] = missed.get(("the commands that told them", kind), 0) + len(commands) - before
    if slots is not None:
        commands = commands[:slots]
    return (np.array(commands, COMMAND_DTYPE) if commands else np.zeros(0, COMMAND_DTYPE)), lost
