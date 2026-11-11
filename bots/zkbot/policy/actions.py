"""From the network's choices to orders: one way of reading them, for the live bot and the sim alike.

The network gives a decision's orders as actions, one after another (`Slots`, at most SLOTS): each picks a unit, its anchor, and
an order for it, which may go as a group order; then done.  `Decision` is the same as what each unit was told, which is what
`decode` and `encode` read and write; `to_decision` and `to_slots` go between the two."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import featurize as fz
from . import groups as gr
from . import rawframe as rf
from .bundle import Bundle
from .tables import table

GROUND_FINE_RADIUS = 200.0
THROW_SHORT = 0.98      # of its reach at most, or it'd walk off to get in range and leave behind those it meant to throw


FIELDS = ("kind", "enemy", "region", "fine", "produce", "group", "roles", "place", "offset")
SLOT_FIELDS = ("anchor",) + FIELDS
SLOTS = 16              # actions a decision at most.  The two scripts' looks give more than 16 in 1.1% of them, 2.3% of their
                        # orders (03-training.md, "Script data")
UNSET = {"anchor": -1, "kind": rf.KEEP, "enemy": -1, "region": -1, "fine": -1, "produce": -1, "group": -1, "roles": 0, "place": -1, "offset": -1}


@dataclass
class Decision:
    """What the network chose for each controllable unit, in token order."""
    kind: np.ndarray              # C: the kind each unit chose
    enemy: np.ndarray             # C: index into enemy_tokens, or -1
    region: np.ndarray            # C: region index, or -1
    fine: np.ndarray              # C: fine option, or -1
    produce: np.ndarray           # C: place in table.producible, or -1
    group: np.ndarray | None = None     # C: for a kind of groups.GROUP_KINDS, index into GROUP_RADII if it goes as a group order, else -1
    roles: np.ndarray | None = None     # C: that group's roles, as bits (groups.roles), else 0
    place: np.ndarray | None = None     # C: for a BUILD, what it's beside: index into place_tokens, then past them the sites; for a
                                        # RECLAIM, which feature token (index into feat_ids); else -1
    offset: np.ndarray | None = None    # C: for a BUILD, where beside it (fz.PLACE_OFFSETS: on it, then 8 a ring), else -1

    def __post_init__(self):
        if self.group is None:
            self.group = np.full(len(self.kind), -1, np.int64)
        if self.roles is None:
            self.roles = np.zeros(len(self.kind), np.int64)
        if self.place is None:
            self.place = np.full(len(self.kind), -1, np.int64)
        if self.offset is None:
            self.offset = np.full(len(self.kind), -1, np.int64)


@dataclass
class Slots:
    """A decision's actions in the order they were chosen: each field has one entry an action.  anchor is the controllable unit's
    token, or -1 for done and for every slot after it; the rest are as Decision's for that unit (UNSET where there's no action)."""
    anchor: np.ndarray
    kind: np.ndarray
    enemy: np.ndarray
    region: np.ndarray
    fine: np.ndarray
    produce: np.ndarray
    group: np.ndarray
    roles: np.ndarray
    place: np.ndarray
    offset: np.ndarray


def to_decision(s: Slots, controllable: int) -> Decision:
    """What each unit was told: its action's choices, and keep for every unit no action picked."""
    out = {f: np.full(controllable, UNSET[f], np.int64) for f in FIELDS}
    for k in np.flatnonzero(np.asarray(s.anchor) >= 0):
        for f in FIELDS:
            out[f][int(s.anchor[k])] = getattr(s, f)[k]
    return Decision(**out)


def to_slots(d: Decision, order: np.ndarray | None = None, slots: int | None = SLOTS) -> Slots:
    """The units told something other than keep, as actions: in `order` (a permutation of the controllable units; token order
    without one), at most `slots` of them (None: all), done after the last if there's room."""
    acting = np.flatnonzero(np.asarray(d.kind) != rf.KEEP)
    if order is not None:
        rank = np.empty(len(d.kind), np.int64)
        rank[np.asarray(order)] = np.arange(len(order))
        acting = acting[np.argsort(rank[acting], kind="stable")]
    width = len(acting) if slots is None else slots
    acting = acting[:width]
    out = {f: np.full(width, UNSET[f], np.int64) for f in SLOT_FIELDS}
    out["anchor"][:len(acting)] = acting
    for f in FIELDS:
        out[f][:len(acting)] = np.asarray(getattr(d, f))[acting]
    return Slots(**out)


@dataclass(frozen=True)
class Order:
    """In the policy's terms: target is a uid, or for PRODUCE a row of the type table.  A radius makes it a group order
    (groups.py): the unit's, and every unit of its side of `roles` within the radius of it."""
    uid: int
    kind: int
    x: float = 0.0
    z: float = 0.0
    target: int = -1
    radius: float = 0.0
    roles: int = 0


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
            order = Order(uid, kind, target=_beacon_in(t, int(decision.region[i]), float(unit_x[i]), float(unit_z[i]), unit_x, unit_z))
        elif kind in fz.KINDS_WITH_REGION:
            x, z = fz.fine_point(t, bundle, int(decision.region[i]), int(decision.fine[i]))
            if kind == rf.THROW:
                x, z = _within_throw(t, i, x, z, unit_x, unit_z)
            order = Order(uid, kind, x, z)
        elif kind == rf.BUILD:
            # The point it chose; whoever obeys it finds the nearest place there the type fits.
            x, z = place_points(t, unit_x, unit_z)[int(decision.place[i]), int(decision.offset[i])]
            order = Order(uid, kind, float(x), float(z), int(table().producible[decision.produce[i]]))
        elif kind == rf.PRODUCE:
            order = Order(uid, kind, target=int(table().producible[decision.produce[i]]))
        elif kind == rf.RECLAIM:
            f = int(decision.place[i])
            order = Order(uid, kind, float(t.feat_pos[f, 0]), float(t.feat_pos[f, 1]), int(t.feat_ids[f]))
        elif kind in (rf.ATTACK_UNIT, rf.MANUAL_FIRE, rf.ASSIST, rf.LOAD):
            token = int(t.enemy_tokens[decision.enemy[i]])
            order = Order(uid, kind, float(unit_x[token]), float(unit_z[token]), int(t.uids[token]))
        elif kind == rf.ATTACK_GROUND:
            token = int(t.enemy_tokens[decision.enemy[i]])
            x, z = float(unit_x[token]), float(unit_z[token])
            option = int(decision.fine[i])
            if option > 0:
                angle = (option - 1) * np.pi / 4.0
                x += GROUND_FINE_RADIUS * float(np.cos(angle) * t.north[0] + np.sin(angle) * t.north[1])
                z += GROUND_FINE_RADIUS * float(np.cos(angle) * t.north[1] - np.sin(angle) * t.north[0])
            order = Order(uid, kind, x, z)
        else:
            order = Order(uid, kind)
        if kind in gr.GROUP_KINDS and decision.group[i] >= 0:
            order = replace(order, radius=gr.GROUP_RADII[int(decision.group[i])], roles=int(decision.roles[i]))
        orders.append(order)
    return orders


def fine_points(t: fz.Tensors, bundle: Bundle) -> np.ndarray:
    """R x FINE_OPTIONS x 2: the point each region and fine option stand for, in elmos, as `fz.fine_point` gives them."""
    angle = np.arange(fz.FINE_OPTIONS - 1) * np.pi / 4.0
    north, east = np.asarray(t.north, np.float64), np.array((t.north[1], -t.north[0]), np.float64)
    compass = np.cos(angle)[:, None] * north + np.sin(angle)[:, None] * east
    centre = bundle.centre[:, :2].astype(np.float64)
    around = centre[:, None, :] + t.region_radius[:, None, None] * compass[None]
    return np.concatenate((centre[:, None, :], around), 1)


def place_points(t: fz.Tensors, unit_x: np.ndarray, unit_z: np.ndarray) -> np.ndarray:
    """(P + S) x PLACE_OFFSETS x 2: the point of each place a BUILD can choose, in elmos: beside each of `place_tokens`, then each
    site; on it, then 8 compass points on each of fz.PLACE_RINGS, north towards the enemy."""
    base = np.concatenate((np.stack((unit_x[t.place_tokens], unit_z[t.place_tokens]), 1), t.site_pos), 0).astype(np.float64)
    angle = np.arange(8) * np.pi / 4.0
    north, east = np.asarray(t.north, np.float64), np.array((t.north[1], -t.north[0]), np.float64)
    compass = np.cos(angle)[:, None] * north + np.sin(angle)[:, None] * east
    offsets = np.concatenate((np.zeros((1, 2)), (np.array(fz.PLACE_RINGS)[:, None, None] * compass[None]).reshape(-1, 2)))
    return base[:, None, :] + offsets[None, :, :]


def _producible_index(row: int) -> int:
    """Where a type-table row is in `producible`, or -1."""
    made = table().producible
    p = int(np.searchsorted(made, row))
    return p if p < len(made) and made[p] == row else -1


def _nearest(points: np.ndarray, legal: np.ndarray, x: float, z: float) -> tuple[int, int]:
    """The legal region, and its option, whose point is nearest.  points: R x O x 2."""
    away = np.hypot(points[..., 0] - x, points[..., 1] - z)
    away[~legal] = np.inf
    r, o = np.unravel_index(int(np.argmin(away)), away.shape)
    return int(r), int(o)


def encode(orders: list[Order], t: fz.Tensors, bundle: Bundle, unit_x: np.ndarray, unit_z: np.ndarray) -> tuple[Decision, int]:
    """`decode` the other way: orders someone else gave, such as a script, as the choices that would have given them.  A unit with no
    order keeps what it has.  A place becomes the nearest region and option the unit could have chosen, and a BUILD's the nearest
    place and offset (`place_points`).  Returns the choices, and how
    many orders couldn't be put in the policy's terms at all (a unit it doesn't control, a kind its masks rule out now, a target that
    isn't a token), which stay KEEP.  A group order is its unit's choice with its radius and roles, the roles it can't choose
    dropped (groups.choosable); the units it gives the order to keep what they have."""
    c = t.controllable
    kind, enemy, region, fine, produce, group, roles, place, offset = (np.full(c, v, np.int64) for v in (rf.KEEP, -1, -1, -1, -1, -1, 0, -1, -1))
    places = None
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
            mask = t.ptr_jump if k in fz.REACH_KINDS else t.ptr_wait if k == rf.WAIT_AT_BEACON else t.ptr_region
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
        elif k in fz.ENEMY_POINTER_KINDS:
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
        elif k == rf.PRODUCE or k == rf.BUILD:
            p = _producible_index(o.target)
            if p < 0 or not t.ptr_produce[i, p]:
                missed += 1
                continue
            produce[i] = p
            if k == rf.BUILD:
                if places is None:
                    places = place_points(t, unit_x, unit_z)
                away = np.hypot(places[..., 0] - o.x, places[..., 1] - o.z)
                place[i], offset[i] = np.unravel_index(int(np.argmin(away)), away.shape)
        elif k == rf.RECLAIM:
            f = np.flatnonzero(t.feat_ids == int(o.target))
            if not len(f):
                missed += 1     # not among the feature tokens the cap kept
                continue
            place[i] = int(f[0])
        kind[i] = k
        if o.radius > 0.0 and k in gr.GROUP_KINDS:
            group[i] = gr.bucket(o.radius)
            roles[i] = int(o.roles) & gr.choosable() if group[i] >= 0 else 0
    return Decision(kind, enemy, region, fine, produce, group, roles, place, offset), missed


def world_positions(t: fz.Tensors, bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """Unit-token positions in elmos, undoing the featurizer's normalisation."""
    x = (t.unit_x[:, fz._I + 0] + 1.0) * 0.5 * bundle.map_w
    z = (t.unit_x[:, fz._I + 1] + 1.0) * 0.5 * bundle.map_h
    return x, z
