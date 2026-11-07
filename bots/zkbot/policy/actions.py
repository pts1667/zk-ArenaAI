"""From the network's choices to orders: one way of reading them, for the live bot and the sim alike."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import featurize as fz
from . import rawframe as rf
from .bundle import Bundle
from .net import Decision
from .tables import table

GROUND_FINE_RADIUS = 200.0
THROW_SHORT = 0.98      # of its reach at most, or it'd walk off to get in range and leave behind those it meant to throw


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


def world_positions(t: fz.Tensors, bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """Unit-token positions in elmos, undoing the featurizer's normalisation."""
    x = (t.unit_x[:, fz._I + 0] + 1.0) * 0.5 * bundle.map_w
    z = (t.unit_x[:, fz._I + 1] + 1.0) * 0.5 * bundle.map_h
    return x, z
