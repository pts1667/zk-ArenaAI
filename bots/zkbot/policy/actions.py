"""From the network's choices to orders: one way of reading them, for the live bot and the sim alike."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import featurize as fz
from . import rawframe as rf
from .bundle import Bundle
from .tables import table

GROUND_FINE_RADIUS = 200.0
THROW_SHORT = 0.98      # of its reach at most, or it'd walk off to get in range and leave behind those it meant to throw


@dataclass
class Decision:
    """What the network chose for each controllable unit, in token order."""
    kind: np.ndarray              # C: the kind each unit chose
    enemy: np.ndarray             # C: index into enemy_tokens, or -1
    region: np.ndarray            # C: region index, or -1
    fine: np.ndarray              # C: fine option, or -1
    produce: np.ndarray           # C: place in table.producible, or -1


@dataclass(frozen=True)
class Order:
    """In the policy's terms: target is a uid, or for PRODUCE a row of the type table."""
    uid: int
    kind: int
    x: float = 0.0
    z: float = 0.0
    target: int = -1


def _within_throw(t: fz.Tensors, i: int, x: float, z: float, unit_x: np.ndarray, unit_z: np.ndarray) -> tuple[float, float]:
    """A region counts as in reach if a point of it could be: what's further than that goes as far as it can, the same way."""
    dx, dz = x - float(unit_x[i]), z - float(unit_z[i])
    away = float(np.hypot(dx, dz))
    reach = float(table().throw_reach[t.unit_name[i]]) * THROW_SHORT
    if away <= reach:
        return x, z
    return float(unit_x[i]) + dx * reach / away, float(unit_z[i]) + dz * reach / away


def _beacon_in(t: fz.Tensors, region: int, x: float, z: float, unit_x: np.ndarray, unit_z: np.ndarray) -> int:
    """The nearest beacon of ours in a region: ptr_wait only offers regions with one."""
    here = t.own_beacons[t.beacon_regions == region]
    return int(t.uids[here[np.argmin(np.hypot(unit_x[here] - x, unit_z[here] - z))]])


def decode(decision: Decision, t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray) -> list[Order]:
    """unit_x and unit_z: world position of each unit token (the tensors hold them normalised).  KEEP emits nothing."""
    orders = []
    for i in range(t.controllable):
        kind = int(decision.kind[i])
        if kind == rf.KEEP:
            continue
        uid = int(t.uids[i])
        if kind == rf.WAIT_AT_BEACON:
            orders.append(Order(uid, kind, target=_beacon_in(t, int(decision.region[i]), float(unit_x[i]), float(unit_z[i]), unit_x, unit_z)))
        elif kind in fz.KINDS_WITH_REGION:
            x, z = fz.fine_point(t, bundle, int(decision.region[i]), int(decision.fine[i]))
            if kind == rf.THROW:
                x, z = _within_throw(t, i, x, z, unit_x, unit_z)
            orders.append(Order(uid, kind, x, z))
        elif kind == rf.PRODUCE:
            orders.append(Order(uid, kind, target=int(table().producible[decision.produce[i]])))
        elif kind in (rf.ATTACK_UNIT, rf.MANUAL_FIRE):
            token = int(t.enemy_tokens[decision.enemy[i]])
            orders.append(Order(uid, kind, float(unit_x[token]), float(unit_z[token]), int(t.uids[token])))
        elif kind == rf.ATTACK_GROUND:
            token = int(t.enemy_tokens[decision.enemy[i]])
            x, z = float(unit_x[token]), float(unit_z[token])
            option = int(decision.fine[i])
            if option > 0:
                angle = (option - 1) * np.pi / 4.0
                x += GROUND_FINE_RADIUS * float(np.cos(angle) * t.north[0] + np.sin(angle) * t.north[1])
                z += GROUND_FINE_RADIUS * float(np.cos(angle) * t.north[1] - np.sin(angle) * t.north[0])
            orders.append(Order(uid, kind, x, z))
        else:
            orders.append(Order(uid, kind))
    return orders


def fine_points(t: fz.Tensors, bundle: Bundle) -> np.ndarray:
    """R x FINE_OPTIONS x 2: the point each region and fine option stand for, in elmos, as `fz.fine_point` gives them."""
    angle = np.arange(fz.FINE_OPTIONS - 1) * np.pi / 4.0
    north, east = np.asarray(t.north, np.float64), np.array((t.north[1], -t.north[0]), np.float64)
    compass = np.cos(angle)[:, None] * north + np.sin(angle)[:, None] * east
    centre = bundle.centre[:, :2].astype(np.float64)
    around = centre[:, None, :] + t.region_radius[:, None, None] * compass[None]
    return np.concatenate((centre[:, None, :], around), 1)


def _nearest(points: np.ndarray, legal: np.ndarray, x: float, z: float) -> tuple[int, int]:
    """The legal region, and its option, whose point is nearest.  points: R x O x 2."""
    away = np.hypot(points[..., 0] - x, points[..., 1] - z)
    away[~legal] = np.inf
    r, o = np.unravel_index(int(np.argmin(away)), away.shape)
    return int(r), int(o)


def encode(orders: list[Order], t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray) -> tuple[Decision, int]:
    """`decode` the other way: orders someone else gave, such as a script, as the choices that would have given them.  A unit with no
    order keeps what it has.  A place becomes the nearest region and option the unit could have chosen.  Returns the choices, and how
    many orders couldn't be put in the policy's terms at all (a unit it doesn't control, a kind its masks rule out now, a target that
    isn't a token), which stay KEEP."""
    c = t.controllable
    kind, enemy, region, fine, produce = (np.full(c, v, np.int64) for v in (rf.KEEP, -1, -1, -1, -1))
    token_of = {int(u): k for k, u in enumerate(t.uids) if u >= 0}
    enemy_of = {int(token): e for e, token in enumerate(t.enemy_tokens)}
    points = fine_points(t, bundle) if c else np.zeros((0, fz.FINE_OPTIONS, 2))
    missed = 0
    for o in orders:
        i = token_of.get(int(o.uid), -1)
        if not 0 <= i < c or not t.kind_mask[i, o.kind]:
            missed += 1
            continue
        k = int(o.kind)
        if k in fz.KINDS_WITH_REGION:
            mask = t.ptr_jump if k in (rf.JUMP, rf.THROW) else t.ptr_wait if k == rf.WAIT_AT_BEACON else t.ptr_region
            x, z = o.x, o.z
            if k == rf.WAIT_AT_BEACON:
                beacon = token_of.get(int(o.target), -1)
                if beacon < 0:
                    missed += 1
                    continue
                x, z = float(unit_x[beacon]), float(unit_z[beacon])
            options = points if k in fz.KINDS_WITH_FINE else points[:, :1]
            legal = np.repeat(mask[i][:, None], options.shape[1], 1)
            region[i], option = _nearest(options, legal, x, z)
            fine[i] = option if k in fz.KINDS_WITH_FINE else -1
        elif k in (rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.ATTACK_GROUND):
            pointer = fz.ENEMY_POINTER_KINDS.index(k)
            if k == rf.ATTACK_GROUND:
                legal = t.ptr_enemy[pointer, i]
                if not legal.any():
                    missed += 1
                    continue
                around = np.array([[unit_x[e], unit_z[e]] for e in t.enemy_tokens])
                # Option 0 is on the token, 1-8 the compass points GROUND_FINE_RADIUS from it, as `decode` places them.
                angle = np.arange(fz.FINE_OPTIONS - 1) * np.pi / 4.0
                north = np.asarray(t.north, np.float64)
                compass = GROUND_FINE_RADIUS * (np.cos(angle)[:, None] * north + np.sin(angle)[:, None] * np.array((north[1], -north[0])))
                options = np.concatenate((around[:, None, :], around[:, None, :] + compass[None]), 1)
                enemy[i], fine[i] = _nearest(options, np.repeat(legal[:, None], options.shape[1], 1), o.x, o.z)
            else:
                e = enemy_of.get(token_of.get(int(o.target), -1), -1)
                if e < 0 or not t.ptr_enemy[pointer, i, e]:
                    missed += 1
                    continue
                enemy[i] = e
        elif k == rf.PRODUCE:
            made = table().producible
            p = int(np.searchsorted(made, o.target))
            if p >= len(made) or made[p] != o.target or not t.ptr_produce[i, p]:
                missed += 1
                continue
            produce[i] = p
        kind[i] = k
    return Decision(kind, enemy, region, fine, produce), missed


def world_positions(t: fz.Tensors, bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """Unit-token positions in elmos, undoing the featurizer's normalisation."""
    x = (t.unit_x[:, fz._I + 0] + 1.0) * 0.5 * bundle.map_w
    z = (t.unit_x[:, fz._I + 1] + 1.0) * 0.5 * bundle.map_h
    return x, z
