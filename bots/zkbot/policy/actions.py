"""From the network's choices to orders: one way of reading them, for the live bot and the sim alike."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import featurize as fz
from . import rawframe as rf
from .bundle import Bundle
from .net import Decision

GROUND_FINE_RADIUS = 200.0


@dataclass(frozen=True)
class Order:
    uid: int
    kind: int
    x: float = 0.0
    z: float = 0.0
    target: int = -1


def decode(decision: Decision, t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray) -> list[Order]:
    """unit_x and unit_z: world position of each unit token (the tensors hold them normalised).  KEEP emits nothing."""
    orders = []
    for i in range(t.controllable):
        kind = int(decision.kind[i])
        if kind == rf.KEEP:
            continue
        uid = int(t.uids[i])
        if kind in (rf.MOVE, rf.FIGHT, rf.JUMP):
            x, z = fz.fine_point(t, bundle, int(decision.region[i]), int(decision.fine[i]))
            orders.append(Order(uid, kind, x, z))
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
