"""Raw frame to tensors and masks.  The one featurizer: the live bot, the sim's training loop and the tests all use it.

Numpy only.  Normalisation is fixed, not learned, so it can't drift with training.  Tokens come in a fixed order:

    controllable units | every other unit, most important first | summaries | regions | sites | features | global

so the first `controllable` tokens are the units that get an action, and an index into the tensors is an index into `uids`.
All units share one pool, and each says whose it is through `unit_ally` and `unit_player`, numbered from the viewer: 0 is the viewer's
own, and any number of alliances and players fit.  Token kinds say only what is known of a unit.
Summaries carry what the caps left out.  `experience` isn't an input: the sim doesn't model it, and a feature the training
never saw vary is one the live game could only mislead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rawframe as rf
from .bundle import PASS_CELL, Bundle
from .tables import TYPE_DIM, TypeTable

CAP_CONTROLLABLE, CAP_UNITS, CAP_HIDDEN = 256, 592, 64
CAP_SITES = 24
CAP_FEATURES = 32           # wrecks and the like in sight.  Past that, a region's metal says what's left out.
CAP_SUMMARIES = 12
# How the rest of the pool is chosen: what can be aimed at comes first, what's in sight before what's on radar, whichever way the
# armies stand.  Within each, value over distance from our army.  A blip's type may be unknown, so nothing counts for less than the floor.
LEAST_COST = 100.0

# Rows of the id embeddings, as well as one for an unknown id.
MAX_ALLY_TEAMS, MAX_PLAYERS = 16, 32

# Token kinds.
TK_SEEN, TK_BLIP, TK_MEMORY, TK_SUMMARY, TK_REGION, TK_SITE, TK_GLOBAL, TK_FEATURE, TK_HIDDEN, TK_MEMORY_STATE = range(10)
TOKEN_KINDS = 10
_KIND_OF_KNOWLEDGE = np.array((TK_SEEN, TK_BLIP, TK_MEMORY, TK_HIDDEN), np.int8)      # by rf.KNOW_*

COMMIT_DECISIONS = 6
DECISION_FRAMES = 15
STAGE2_KINDS = (*range(10), rf.PRODUCE, rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON)

# The three kinds that point at an enemy token, in the order of `ptr_enemy`.
ENEMY_POINTER_KINDS = (rf.ATTACK_UNIT, rf.ATTACK_GROUND, rf.MANUAL_FIRE)
FINE_OPTIONS = 9
# What each kind is told besides itself.  A throw and a jump choose among the same regions: ptr_jump, where this unit's own reach lands.
KINDS_WITH_REGION = (rf.MOVE, rf.FIGHT, rf.JUMP, rf.THROW, rf.PLACE_BEACON, rf.WAIT_AT_BEACON)
KINDS_WITH_FINE = (rf.MOVE, rf.FIGHT, rf.JUMP, rf.ATTACK_GROUND, rf.THROW, rf.PLACE_BEACON)

# unit_x layout.
_INSTANCE = 19
_ORDER = rf.ORDER_SLOTS + 7
_RAYS = 24
_SUMMARY = 3
UNIT_DIM = TYPE_DIM + _INSTANCE + _ORDER + _RAYS + _SUMMARY
ALTITUDE_DEAD_ZONE = 40.0
_I = TYPE_DIM
_O = _I + _INSTANCE
_R = _O + _ORDER
_S = _R + _RAYS

REGION_DIM = 2 + 1 + 2 + 1 + 9 + 6 + 2 + 2 + 2 + 6 + 1
SITE_DIM = 2 + 1 + 4 + 1 + 1
FEATURE_DIM = 2 + 1 + 1 + 1
GLOBAL_DIM = len(rf.GLOBAL_FIELDS) - 2     # the last two say who the viewer is, and aren't inputs

# Pair buckets.  Unit-unit: log2 distance (12) x hostile, same alliance or same player (3), then a unit and what it's ordered to shoot, each way.
# Unit-region: same region or hops 1-6+, or unknown (8).  Region-region: hops 0-6+ (7).  Unit-feature: log2 distance (12).  All else: one per pair of token groups.
_UU = 0
_AIMS_AT = 36
_AIMED_AT_BY = 37
_UR = 38
_RR = 46
_UF = 53
_OTHER = 65
_GROUPS = 6   # unit, summary, region, site, feature, global
NUM_BUCKETS = _OTHER + _GROUPS * _GROUPS + 1   # the last is the memory state's, with anything
MEMORY_BUCKET = NUM_BUCKETS - 1

RAY_CELLS = 12            # 32-elmo steps, to 400 elmos
RAY_STEP = 32.0
RAY_MAX = RAY_CELLS * RAY_STEP
RAY_DPS_RADIUS = 600.0
_RAY_ANGLES = np.arange(8) * (np.pi / 4.0)
_RAY_DX = np.sin(_RAY_ANGLES).astype(np.float32)
_RAY_DZ = np.cos(_RAY_ANGLES).astype(np.float32)
_RAY_T = (np.arange(1, RAY_CELLS + 1) * RAY_STEP).astype(np.float32)


@dataclass
class Tensors:
    unit_x: np.ndarray            # U x UNIT_DIM
    unit_name: np.ndarray         # U: row in the name vocabulary
    reg_x: np.ndarray             # R x REGION_DIM
    site_x: np.ndarray            # S x SITE_DIM
    glob_x: np.ndarray            # GLOBAL_DIM
    tok_kind: np.ndarray          # N
    pair_bucket: np.ndarray       # N x N
    controllable: int             # C: the first C unit tokens get an action
    kind_mask: np.ndarray         # C x KIND_COUNT
    ptr_enemy: np.ndarray         # len(ENEMY_POINTER_KINDS) x C x E: legal targets per kind
    ptr_region: np.ndarray        # C x R: where it can walk to
    ptr_jump: np.ndarray          # C x R: where it can jump or throw to, which can be across what it can't walk
    ptr_wait: np.ndarray          # C x R: where it can walk to a beacon of ours
    ptr_produce: np.ndarray       # C x T: what it can make, by place in table.producible
    route_bias: np.ndarray        # C x R: walking seconds to each region
    gate_a: np.ndarray            # G: region indices of each gate
    gate_b: np.ndarray            # G
    gate_x: np.ndarray            # G x GATE_DIM, normalised
    uids: np.ndarray              # U: uid per unit token, -1 for a summary
    enemy_tokens: np.ndarray      # E: unit-token index of each token that can be aimed at: any other alliance's, in sight or on radar
    north: np.ndarray             # 2: from our army towards the enemy, for fine positions
    region_radius: np.ndarray     # R: half the equivalent radius, for fine positions
    own_beacons: np.ndarray       # B: unit-token index of each finished beacon of ours
    beacon_regions: np.ndarray    # B: the region each is in, or -1
    hidden_x: np.ndarray = field(default_factory=lambda: np.zeros((0, UNIT_DIM), np.float32))   # critic only
    hidden_name: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))              # critic only
    unit_ally: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))                # U: from the viewer, see relative_id
    unit_player: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))              # U
    hidden_ally: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))              # critic only
    hidden_player: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))            # critic only
    feat_x: np.ndarray = field(default_factory=lambda: np.zeros((0, FEATURE_DIM), np.float32))  # F x FEATURE_DIM
    feat_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))                 # F


def _centroid(x: np.ndarray, z: np.ndarray, fallback: tuple[float, float]) -> np.ndarray:
    return np.array([x.mean(), z.mean()], np.float32) if len(x) else np.array(fallback, np.float32)


def relative_id(ids: np.ndarray, own: int, count: int) -> np.ndarray:
    """A game's numbers as the viewer's: its own is 0, and the others count from 1, skipping it.  So no seat is special, both seats of
    a 1v1 see the other as 1, and a number stays the same all game.  Unknown ids, and any past `count`, take the last row."""
    ids = ids.astype(np.int64)
    out = np.where(ids < own, ids + 1, ids)
    out[ids == own] = 0
    out[(ids < 0) | (out >= count)] = count
    return out.astype(np.int32)


def _select(units: np.ndarray, table: TypeTable, player: int, ally: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray | None]:
    """Applies the caps.  Returns the indices kept, how many of them are controllable, those left over, and army and enemy centroids."""
    rows = table.rows(units["def_id"])
    cost = table.cost[rows]
    finished = (units["flags"] & rf.BEING_BUILT) == 0
    own = units["player"] == player
    hostile = units["ally_team"] != ally
    knowledge = units["knowledge"]
    fights = table.combat[rows] & table.mobile[rows]
    can = own & finished & (fights | table.acts_in_place[rows]) & ~table.air[rows] & ((units["flags"] & rf.RESERVED) == 0)
    x, z = units["x"], units["z"]

    def near(idx: np.ndarray, centre: np.ndarray) -> np.ndarray:
        return np.hypot(x[idx] - centre[0], z[idx] - centre[1])

    cand = np.flatnonzero(can)
    goes = np.flatnonzero(can & table.mobile[rows])      # a turret isn't where the army is
    army = _centroid(x[goes], z[goes], (float(x[own].mean()), float(z[own].mean())) if own.any() else (0.0, 0.0))
    # Most expensive first; among equals, nearest the army.  uid breaks ties so that record order can't matter.
    keep_c = cand[np.lexsort((units["uid"][cand], near(cand, army), -cost[cand]))] if len(cand) else cand
    control = keep_c[:CAP_CONTROLLABLE]
    taken = np.zeros(len(units), bool)
    taken[control] = True
    rest = np.flatnonzero((knowledge != rf.KNOW_HIDDEN) & ~taken)
    score = np.maximum(cost[rest], LEAST_COST) / (near(rest, army) + 200.0)
    rank = np.select((hostile[rest] & (knowledge[rest] == rf.KNOW_LOS), hostile[rest] & (knowledge[rest] == rf.KNOW_RADAR)), (0, 1), 2)
    rest = rest[np.lexsort((units["uid"][rest], -score, rank))]
    room = CAP_UNITS - len(control)
    seen = np.flatnonzero(hostile & ((knowledge == rf.KNOW_LOS) | (knowledge == rf.KNOW_RADAR)))
    foe = _centroid(x[seen], z[seen], (0.0, 0.0)) if len(seen) else None
    return np.concatenate((control, rest[:room])), len(control), rest[room:], army, foe


def _instance(units: np.ndarray, rows: np.ndarray, bundle: Bundle, table: TypeTable, out: np.ndarray) -> None:
    x, z = units["x"], units["z"]
    out[:, _I + 0] = x / bundle.map_w * 2.0 - 1.0
    out[:, _I + 1] = z / bundle.map_h * 2.0 - 1.0
    # Live, a unit's y is its middle, up to 35 elmos above the ground by type; the sim says the ground.  Past that, it flies.
    above = units["y"] - bundle.ground_at(x, z)
    out[:, _I + 2] = np.clip(np.sign(above) * np.maximum(np.abs(above) - ALTITUDE_DEAD_ZONE, 0.0) / 200.0, -1.0, 4.0)
    top = np.maximum(table.speed[rows], 30.0) / 30.0        # elmos per frame, as vel is
    vx, vz = np.clip(units["vx"] / top, -2.0, 2.0), np.clip(units["vz"] / top, -2.0, 2.0)
    out[:, _I + 3], out[:, _I + 4] = vx, vz
    out[:, _I + 5] = np.minimum(np.hypot(vx, vz), 2.0)
    hp = units["hp_frac"]
    missing = np.isnan(hp)
    out[:, _I + 6] = np.where(missing, 0.0, np.clip(hp, 0.0, 1.0))
    out[:, _I + 7] = missing
    out[:, _I + 8] = np.clip(units["build_frac"], 0.0, 1.0)
    flags = units["flags"]
    for k, bit in enumerate((rf.BEING_BUILT, rf.CLOAKED, rf.PARALYZED, rf.IN_LOS, rf.IDLE, rf.TYPE_KNOWN, rf.DAMAGED)):
        out[:, _I + 9 + k] = (flags & bit) != 0
    out[:, _I + 16] = units["blip_top_speed"] / 150.0
    # A remembered structure's ord_age says how long ago it was seen.
    is_memory = units["knowledge"] == rf.KNOW_MEMORY
    out[:, _I + 17] = np.where(is_memory, np.log1p(np.maximum(units["ord_age"], 0.0) / 30.0) / 6.0, 0.0)
    out[:, _I + 18] = (flags & rf.JUMP_READY) != 0


def _find(wanted: np.ndarray, uids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each wanted uid is among `uids`: whether it's there, and its index, which means nothing where it isn't."""
    if not len(uids):
        return np.zeros(len(wanted), bool), np.zeros(len(wanted), np.int64)
    order = np.argsort(uids)
    at = order[np.minimum(np.searchsorted(uids, wanted, sorter=order), len(uids) - 1)]
    return uids[at] == wanted, at


def _order_block(units: np.ndarray, is_own: np.ndarray, bundle: Bundle, aimable: np.ndarray, commit_decisions: int, out: np.ndarray) -> np.ndarray:
    """Fills the order-memory columns, and returns who is commit-locked.  `aimable` is every record that can be aimed at, kept or not."""
    kind = units["ord_kind"].astype(np.int64)
    has = (kind >= 0) & is_own
    idx = np.flatnonzero(has)
    out[idx, _O + kind[idx]] = 1.0
    age = np.where(has, units["ord_age"], 0.0)
    out[:, _O + 16] = np.log1p(np.maximum(age, 0.0) / DECISION_FRAMES) / 6.0
    target = units["ord_target"]
    has_target = has & (target >= 0)
    known, at = _find(target, aimable["uid"])
    known &= has_target
    # A target's direction is where it is now, in sight or on radar.  Any other goal is where the order said.
    goal_x, goal_z = units["ord_x"], units["ord_z"]
    if len(aimable):
        goal_x, goal_z = np.where(known, aimable["x"][at], goal_x), np.where(known, aimable["z"][at], goal_z)
    goes = known | (has & (kind != rf.ATTACK_UNIT) & (kind != rf.STOP))
    dx, dz = np.where(goes, goal_x - units["x"], 0.0), np.where(goes, goal_z - units["z"], 0.0)
    out[:, _O + 17] = dx / bundle.map_w
    out[:, _O + 18] = dz / bundle.map_h
    out[:, _O + 19] = np.hypot(dx, dz) / max(bundle.map_w, bundle.map_h)
    alive = known & (aimable["knowledge"][at] == rf.KNOW_LOS) if len(aimable) else known
    out[:, _O + 20] = has_target
    out[:, _O + 21] = has_target & alive
    fresh = has & (age < commit_decisions * DECISION_FRAMES)
    flags = units["flags"]
    out[:, _O + rf.RESERVED_KIND] = is_own & ((flags & rf.PULLING) != 0)       # no order has this kind, and the frames so far hold 0 here
    released = ((flags & (rf.DAMAGED | rf.IDLE)) != 0) | (has_target & ~alive)
    locked = fresh & ~released
    out[:, _O + 22] = locked
    return locked


def _rays(x: np.ndarray, z: np.ndarray, classes: np.ndarray, bundle: Bundle, out: np.ndarray) -> None:
    """Free distance in eight directions, by this unit's own move class; air units see nothing in the way."""
    n = len(x)
    if not n:
        return
    px = x[:, None, None] + _RAY_DX[None, :, None] * _RAY_T[None, None, :]
    pz = z[:, None, None] + _RAY_DZ[None, :, None] * _RAY_T[None, None, :]
    rows, cols = bundle.passable.shape
    i = np.floor(pz / PASS_CELL).astype(np.int32)
    j = np.floor(px / PASS_CELL).astype(np.int32)
    inside = (i >= 0) & (i < rows) & (j >= 0) & (j < cols)
    cells = bundle.passable[np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)]
    flying = classes < 0
    bits = np.where(flying, 0, classes).astype(np.uint8)[:, None, None]
    free = inside & (((cells >> bits) & 1) != 0)
    free[flying] = True
    # The first blocked step along each ray; a ray with none runs the full length.
    blocked = ~free
    first = np.where(blocked.any(axis=2), blocked.argmax(axis=2), RAY_CELLS)
    out[:, _R:_R + 8] = first / RAY_CELLS


def _ray_dps(x: np.ndarray, z: np.ndarray, ally: np.ndarray, dps: np.ndarray, out: np.ndarray) -> None:
    """Hostile and friendly DPS within RAY_DPS_RADIUS per 45-degree sector, from each token's own point of view."""
    n = len(x)
    if not n:
        return
    dx = x[None, :] - x[:, None]
    dz = z[None, :] - z[:, None]
    near = (dx * dx + dz * dz) <= RAY_DPS_RADIUS * RAY_DPS_RADIUS
    np.fill_diagonal(near, False)
    a, b = np.nonzero(near)
    if not len(a):
        return
    # Sector 0 is around +z, as the rays are: the angle is measured from z towards x.
    sector = ((np.arctan2(dx[a, b], dz[a, b]) + np.pi / 8.0) / (np.pi / 4.0)).astype(np.int32) % 8
    friendly = ally[a] == ally[b]
    weights = dps[b]
    enemy_sum = np.bincount(a[~friendly] * 8 + sector[~friendly], weights=weights[~friendly], minlength=n * 8).reshape(n, 8)
    own_sum = np.bincount(a[friendly] * 8 + sector[friendly], weights=weights[friendly], minlength=n * 8).reshape(n, 8)
    out[:, _R + 8:_R + 16] = np.log1p(enemy_sum) / 8.0
    out[:, _R + 16:_R + 24] = np.log1p(own_sum) / 8.0


def _aim_pairs(units: np.ndarray, is_own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tokens with a target among the unit tokens, and the token of that target."""
    aiming = np.flatnonzero(is_own & (units["ord_kind"] >= 0) & (units["ord_target"] >= 0))
    found, at = _find(units["ord_target"][aiming], units["uid"])
    return aiming[found], at[found]


def _distance_bucket(dist: np.ndarray) -> np.ndarray:
    """Log2 distance: under 16 elmos is bucket 0, each doubling one more, 12 in all."""
    return np.clip(np.frexp(np.maximum(dist, 1.0) / 8.0)[1] - 1, 0, 11)


def _pair_buckets(units: np.ndarray, is_own: np.ndarray, unit_region: np.ndarray, summaries: int, bundle: Bundle, sites: int, features: np.ndarray) -> np.ndarray:
    x, z, players, allies = units["x"], units["z"], units["player"], units["ally_team"]
    u = len(units)                    # units, without summaries
    r = bundle.regions
    f = len(features)
    n = u + summaries + r + sites + f + 1
    out = np.empty((n, n), np.int8)
    groups = np.concatenate((np.zeros(u, np.int8), np.full(summaries, 1, np.int8), np.full(r, 2, np.int8), np.full(sites, 3, np.int8), np.full(f, 4, np.int8), np.array([5], np.int8)))
    out[:] = _OTHER + groups[:, None] * _GROUPS + groups[None, :]
    if u and f:
        block = _UF + _distance_bucket(np.hypot(features["x"][None, :] - x[:, None], features["z"][None, :] - z[:, None]))
        first = u + summaries + r + sites
        out[:u, first:first + f] = block
        out[first:first + f, :u] = block.T
    if u:
        dx = x[None, :] - x[:, None]
        dz = z[None, :] - z[:, None]
        dist = np.sqrt(dx * dx + dz * dz)
        buckets = _distance_bucket(dist)
        same_ally = allies[:, None] == allies[None, :]
        # Two units whose player isn't known aren't known to share one.
        same_player = (players[:, None] == players[None, :]) & (players[:, None] >= 0)
        out[:u, :u] = _UU + buckets + 12 * (same_ally.astype(np.int8) + (same_ally & same_player))
        shooter, target = _aim_pairs(units, is_own)
        out[shooter, target] = _AIMS_AT
        out[target, shooter] = _AIMED_AT_BY
    if u and r:
        known = unit_region >= 0
        hops = np.where(known[:, None], bundle.hops[np.maximum(unit_region, 0)], 7).astype(np.int8)
        block = _UR + np.minimum(hops, 7)
        out[:u, u + summaries:u + summaries + r] = block
        out[u + summaries:u + summaries + r, :u] = block.T
    if r:
        out[u + summaries:u + summaries + r, u + summaries:u + summaries + r] = _RR + bundle.hops
    return out


def _summaries(units: np.ndarray, left: np.ndarray, rows: np.ndarray, bundle: Bundle, table: TypeTable, player: int,
               ally: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One token per (whose, what's known of it, coarse role) for what the caps left out: count, hp sum, dps sum, centroid.  The
    viewer's own go together.  With more groups than the cap, the biggest stay."""
    tokens, counts, allies, players = [], [], [], []
    x, z = units["x"], units["z"]
    mine = units["player"][left] == player
    rel_ally = relative_id(units["ally_team"][left], ally, MAX_ALLY_TEAMS)
    keys = np.where(mine, -1, rel_ally.astype(np.int64) * 4 + units["knowledge"][left])
    for key in np.unique(keys):
        group = left[keys == key]
        first_ally = int(rel_ally[keys == key][0])
        r = rows[group]
        coarse = np.where(table.builder[r], 1, np.where(table.mobile[r] & table.combat[r], 0, 2))
        for role in range(3):
            members = group[coarse == role]
            if not len(members):
                continue
            row = np.zeros(UNIT_DIM, np.float32)
            row[_I + 0] = x[members].mean() / bundle.map_w * 2.0 - 1.0
            row[_I + 1] = z[members].mean() / bundle.map_h * 2.0 - 1.0
            hp = np.nan_to_num(units["hp_frac"][members], nan=1.0)
            row[_S + 0] = np.log1p(len(members)) / 5.0
            row[_S + 1] = np.log1p((hp * table.cost[rows[members]]).sum()) / 10.0
            row[_S + 2] = np.log1p(table.dps_ground[rows[members]].sum()) / 8.0
            tokens.append(row)
            counts.append(len(members))
            allies.append(first_ally)
            players.append(0 if key == -1 else MAX_PLAYERS)
    if not tokens:
        return np.zeros((0, UNIT_DIM), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    keep = np.sort(np.argsort(-np.array(counts), kind="stable")[:CAP_SUMMARIES])
    return np.stack(tokens)[keep], np.full(len(keep), table.unknown, np.int32), np.array(allies, np.int32)[keep], np.array(players, np.int32)[keep]


def featurize(frame: rf.Frame, bundle: Bundle, table: TypeTable, commit_decisions: int = COMMIT_DECISIONS,
              enabled_kinds: tuple[int, ...] = STAGE2_KINDS, privileged: bool = False, economy: bool = False) -> Tensors:
    """economy: whether the net sees metal, energy and who holds the sites.  Off at stage 2: the sim has no economy yet, and an
    input the training never saw vary could only mislead it in a real game."""
    all_units = frame.units
    player, ally = int(frame.glob[rf.G["player"]]), int(frame.glob[rf.G["ally_team"]])
    picked, c, left, army, foe = _select(all_units, table, player, ally)
    units = all_units[picked]
    rows = table.rows(units["def_id"])
    u = len(units)
    x, z = units["x"], units["z"]
    is_own = units["player"] == player
    hostile = units["ally_team"] != ally
    knowledge = units["knowledge"]
    unit_region = bundle.region_at(x, z).astype(np.int32)

    all_rows = table.rows(all_units["def_id"])
    summary_x, summary_name, summary_ally, summary_player = _summaries(all_units, left, all_rows, bundle, table, player, ally)
    s = len(summary_x)

    # Unit tokens.
    unit_x = np.zeros((u + s, UNIT_DIM), np.float32)
    head = unit_x[:u]
    head[:, :TYPE_DIM] = table.stats[rows]
    _instance(units, rows, bundle, table, head)
    aimable = all_units[(all_units["ally_team"] != ally) & ((all_units["knowledge"] == rf.KNOW_LOS) | (all_units["knowledge"] == rf.KNOW_RADAR))]
    locked = _order_block(units, is_own, bundle, aimable, commit_decisions, head)
    classes = np.where(table.air[rows], -1, table.move_class[rows])
    _rays(x, z, classes, bundle, head)
    dps = table.dps_ground[rows] + table.dps_air[rows]
    _ray_dps(x, z, units["ally_team"], dps, head)
    if s:
        unit_x[u:] = summary_x
    unit_name = np.concatenate((rows.astype(np.int32), summary_name))
    unit_ally = np.concatenate((relative_id(units["ally_team"], ally, MAX_ALLY_TEAMS), summary_ally))
    unit_player = np.concatenate((relative_id(units["player"], player, MAX_PLAYERS), summary_player))

    # Sites: nearest the base first.
    base = frame.glob[[rf.G["base_x"], rf.G["base_z"]]]
    sites = frame.sites
    if len(sites) > CAP_SITES:
        order = np.lexsort((sites["site_id"], np.hypot(sites["x"] - base[0], sites["z"] - base[1])))
        sites = sites[order[:CAP_SITES]]
    site_regions = bundle.region_at(sites["x"], sites["z"]).astype(np.int32)
    site_x = np.zeros((len(sites), SITE_DIM), np.float32)
    site_x[:, 0] = sites["x"] / bundle.map_w * 2.0 - 1.0
    site_x[:, 1] = sites["z"] / bundle.map_h * 2.0 - 1.0
    site_x[:, 2] = sites["income"] / 5.0
    if len(sites):
        status = sites["status"].astype(np.int64) if economy else np.full(len(sites), rf.SITE_FREE, np.int64)
        site_x[np.arange(len(sites)), 3 + status] = 1.0
        d = bundle.centre[site_regions] - np.stack((sites["x"], sites["z"]), axis=1)
        site_x[:, 7] = np.hypot(d[:, 0], d[:, 1]) / 1000.0
        site_x[:, 8] = np.hypot(sites["x"] - base[0], sites["z"] - base[1]) / max(bundle.map_w, bundle.map_h)

    features = _pick_features(frame.features, army)
    feat_x = np.zeros((len(features), FEATURE_DIM), np.float32)
    feat_x[:, 0] = features["x"] / bundle.map_w * 2.0 - 1.0
    feat_x[:, 1] = features["z"] / bundle.map_h * 2.0 - 1.0
    feat_x[:, 2] = np.log1p(features["metal"]) / 8.0
    feat_x[:, 3] = np.hypot(features["x"] - base[0], features["z"] - base[1]) / max(bundle.map_w, bundle.map_h)
    feat_x[:, 4] = np.hypot(features["x"] - army[0], features["z"] - army[1]) / max(bundle.map_w, bundle.map_h)

    reg_x = _regions(all_units, all_rows, bundle, table, ally, frame.features)
    glob_x = _global(frame, bundle, economy)

    # Token kinds.
    tok_kind = np.concatenate((_KIND_OF_KNOWLEDGE[knowledge], np.full(s, TK_SUMMARY, np.int8), np.full(bundle.regions, TK_REGION, np.int8),
                               np.full(len(sites), TK_SITE, np.int8), np.full(len(features), TK_FEATURE, np.int8), np.array([TK_GLOBAL], np.int8)))
    pair_bucket = _pair_buckets(units, is_own, unit_region, s, bundle, len(sites), features)

    # Actions.
    enemy_tokens = np.flatnonzero(hostile & ((knowledge == rf.KNOW_LOS) | (knowledge == rf.KNOW_RADAR)))
    own_beacons = np.flatnonzero(is_own & table.beacon[rows] & ((units["flags"] & rf.BEING_BUILT) == 0))
    legal = _masks(_Tokens(units, rows, unit_region, enemy_tokens, own_beacons), locked, c, bundle, table, enabled_kinds)

    if foe is None:
        foe = np.array((bundle.map_w * 0.5, bundle.map_h * 0.5), np.float32)
    north = foe - army
    norm = float(np.hypot(north[0], north[1]))
    north = north / norm if norm > 1.0 else np.array((0.0, 1.0), np.float32)

    gate_x = bundle.gate_x.copy()
    gate_x[:, :6] = np.log1p(gate_x[:, :6]) / 8.0
    gate_x[:, 6:12] = gate_x[:, 6:12] / 2000.0
    out = Tensors(
        unit_x=unit_x, unit_name=unit_name, reg_x=reg_x, site_x=site_x, glob_x=glob_x, tok_kind=tok_kind, pair_bucket=pair_bucket,
        controllable=c, kind_mask=legal.kind, ptr_enemy=legal.enemy, ptr_region=legal.region, ptr_jump=legal.jump, ptr_wait=legal.wait,
        ptr_produce=legal.produce, route_bias=legal.route, own_beacons=own_beacons, beacon_regions=unit_region[own_beacons],
        gate_a=bundle.edge_a, gate_b=bundle.edge_b, gate_x=gate_x,
        uids=np.concatenate((units["uid"], np.full(s, -1, np.int64))), enemy_tokens=enemy_tokens, north=north,
        region_radius=(np.sqrt(bundle.area / np.pi) * 0.5).astype(np.float32), unit_ally=unit_ally, unit_player=unit_player,
        feat_x=feat_x, feat_ids=features["feat_id"].astype(np.int64),
    )
    if privileged:
        out.hidden_x, out.hidden_name, out.hidden_ally, out.hidden_player = _hidden(all_units, all_rows, army, bundle, table, player, ally)
    return out


def _pick_features(features: np.ndarray, army: np.ndarray) -> np.ndarray:
    """The most metal for the walk from our army, as the units are chosen.  feat_id breaks ties so that record order can't matter."""
    if len(features) > CAP_FEATURES:
        score = features["metal"] / (np.hypot(features["x"] - army[0], features["z"] - army[1]) + 200.0)
        features = features[np.lexsort((features["feat_id"], -score))[:CAP_FEATURES]]
    return features[np.argsort(features["feat_id"], kind="stable")]


def _regions(units: np.ndarray, rows: np.ndarray, bundle: Bundle, table: TypeTable, ally: int, features: np.ndarray) -> np.ndarray:
    r = bundle.regions
    out = np.zeros((r, REGION_DIM), np.float32)
    if len(features):
        at = bundle.region_at(features["x"], features["z"]).astype(np.int64)
        # All of what's in sight, kept as a token or not.  A point off every region counts for none.
        on = at >= 0
        out[:, 33] = np.log1p(np.bincount(at[on], weights=features["metal"][on], minlength=r)) / 10.0
    out[:, 0] = bundle.centre[:, 0] / bundle.map_w * 2.0 - 1.0
    out[:, 1] = bundle.centre[:, 1] / bundle.map_h * 2.0 - 1.0
    out[:, 2] = np.log1p(bundle.area) / 16.0
    out[:, 3:5] = bundle.region_height / 300.0
    out[:, 5] = bundle.sea
    out[:, 6:15] = bundle.tags
    out[:, 15:21] = bundle.access
    out[:, 21:23] = bundle.shares
    out[:, 23] = bundle.spots[:, 0] / 5.0
    out[:, 24] = bundle.spots[:, 1] / 10.0
    out[:, 25:27] = bundle.gate_counts / 4.0
    if len(units):
        at = bundle.region_at(units["x"], units["z"]).astype(np.int64)
        hp = np.nan_to_num(units["hp_frac"], nan=1.0)
        value = table.cost[rows] * np.clip(hp, 0.0, 1.0)
        dps = table.dps_ground[rows]
        hostile = units["ally_team"] != ally
        knowledge = units["knowledge"]
        friends = ~hostile
        foes = hostile & (knowledge == rf.KNOW_LOS)
        for k, members in enumerate((friends, foes, hostile & (knowledge == rf.KNOW_RADAR))):
            out[:, 27 + k] = np.log1p(np.bincount(at[members], weights=value[members], minlength=r)) / 10.0
        out[:, 30] = np.log1p(np.bincount(at[friends], weights=dps[friends], minlength=r)) / 8.0
        out[:, 31] = np.log1p(np.bincount(at[foes], weights=dps[foes], minlength=r)) / 8.0
        out[:, 32] = np.log1p(np.bincount(at[hostile & (knowledge == rf.KNOW_MEMORY)], minlength=r)) / 3.0
    return out


def _global(frame: rf.Frame, bundle: Bundle, economy: bool) -> np.ndarray:
    g = frame.glob
    out = np.zeros(GLOBAL_DIM, np.float32)
    idx = rf.G
    out[idx["frame"]] = g[idx["frame"]] / 36000.0
    if economy:
        for name in ("metal", "metal_income", "metal_storage", "energy", "energy_income", "energy_storage"):
            out[idx[name]] = np.log1p(max(0.0, float(g[idx[name]]))) / 10.0
        out[idx["unfinished_structures"]] = g[idx["unfinished_structures"]] / 8.0
        out[idx["idle_builders"]] = g[idx["idle_builders"]] / 8.0
    out[idx["commander_hp"]] = g[idx["commander_hp"]]
    out[idx["commander_uid"]] = float(g[idx["commander_uid"]] >= 0.0)    # alive, not who
    out[idx["base_x"]] = g[idx["base_x"]] / bundle.map_w * 2.0 - 1.0
    out[idx["base_z"]] = g[idx["base_z"]] / bundle.map_h * 2.0 - 1.0
    out[idx["map_w"]] = bundle.map_w / 16384.0
    out[idx["map_h"]] = bundle.map_h / 16384.0
    out[idx["frames_since_contact"]] = np.log1p(max(0.0, float(g[idx["frames_since_contact"]])) / 30.0) / 6.0
    return out


@dataclass
class _Tokens:
    """The unit tokens, and which of them an order can be about."""
    units: np.ndarray
    rows: np.ndarray
    regions: np.ndarray
    enemies: np.ndarray
    beacons: np.ndarray


@dataclass
class _Legal:
    kind: np.ndarray
    enemy: np.ndarray
    region: np.ndarray
    jump: np.ndarray
    wait: np.ndarray
    produce: np.ndarray
    route: np.ndarray


def _throw_targets(tokens: _Tokens, i: int, centres_near: np.ndarray, table: TypeTable) -> np.ndarray:
    """Where a thrower can aim, if there's anyone beside it to throw (CheckOverlobPrevention): whoever walks, and is ours or in sight."""
    units, rows = tokens.units, tokens.rows
    around = table.mobile[rows] & ~table.air[rows] & (units["knowledge"] == rf.KNOW_LOS) & ((units["flags"] & rf.BEING_BUILT) == 0) & (units["uid"] != units["uid"][i])
    near = np.hypot(units["x"][around] - units["x"][i], units["z"][around] - units["z"][i]) <= table.throw_radius[rows[i]]
    return centres_near if near.any() else np.zeros_like(centres_near)


def _masks(tokens: _Tokens, locked: np.ndarray, c: int, bundle: Bundle, table: TypeTable, enabled: tuple[int, ...]) -> _Legal:
    units, rows, unit_region, enemy_tokens = tokens.units, tokens.rows, tokens.regions, tokens.enemies
    r = bundle.regions
    e = len(enemy_tokens)
    kind = np.zeros((c, rf.KIND_COUNT), bool)
    ptr_enemy = np.zeros((len(ENEMY_POINTER_KINDS), c, e), bool)
    ptr_region = np.zeros((c, r), bool)
    ptr_jump = np.zeros((c, r), bool)
    ptr_wait = np.zeros((c, r), bool)
    route = np.zeros((c, r), np.float32)
    me = rows[:c]
    ptr_produce = table.produces[me]
    if not c:
        return _Legal(kind, ptr_enemy, ptr_region, ptr_jump, ptr_wait, ptr_produce, route)
    idle = (units["flags"][:c] & rf.IDLE) != 0
    free = ~locked[:c]
    mobile, armed = table.mobile[me], table.armed[me]
    classes = table.move_class[me]
    here = unit_region[:c]
    # Regions a unit can walk to, by its class; its own always counts, so a unit always has a legal MOVE.
    for i in range(c):
        if classes[i] >= 0 and here[i] >= 0:
            walk = bundle.walk[classes[i], here[i]]
            ptr_region[i] = np.isfinite(walk)
            route[i] = np.where(np.isfinite(walk), walk, 0.0) / max(table.speed[me[i]], 1.0)
            ptr_region[i, here[i]] = True
        elif here[i] >= 0:
            ptr_region[i, here[i]] = True
    can_go = ptr_region.any(axis=1)

    if e:
        targets = rows[enemy_tokens]
        in_los = units["knowledge"][enemy_tokens] == rf.KNOW_LOS
        dx = units["x"][:c, None] - units["x"][enemy_tokens][None, :]
        dz = units["z"][:c, None] - units["z"][enemy_tokens][None, :]
        dist = np.hypot(dx, dz)
        air_target = table.air[targets]
        hits = (table.can_hit_ground[me][:, None] & ~air_target[None, :]) | (table.can_hit_air[me][:, None] & air_target[None, :])
        ptr_enemy[0] = hits & in_los[None, :]
        ptr_enemy[1] = armed[:, None] & (dist <= (table.range[me] * 1.5 + 300.0)[:, None])
        ptr_enemy[2] = table.manualfire[me][:, None] & in_los[None, :] & (dist <= table.manual_range[me][:, None])

    kind[:, rf.KEEP] = ~idle | ~mobile
    kind[:, rf.MOVE] = free & mobile & can_go
    kind[:, rf.FIGHT] = free & mobile & armed & can_go
    kind[:, rf.ATTACK_UNIT] = free & ptr_enemy[0].any(axis=1)
    kind[:, rf.ATTACK_GROUND] = free & table.can_hit_ground[me] & ptr_enemy[1].any(axis=1)
    kind[:, rf.STOP] = free & (mobile | armed)
    fired = (units["ord_kind"][:c] == rf.MANUAL_FIRE) & (units["ord_age"][:c] < table.manual_reload[me])
    kind[:, rf.MANUAL_FIRE] = free & ptr_enemy[2].any(axis=1) & ~fired
    # Where it's told to is a region's centre or near it: one it could walk to, or one a jump from here would reach (fine positions are this
    # far off the centre), so that it can cross what it can't walk.  Not where it couldn't land, which the game refuses.
    ready = (classes >= 0) & (here >= 0) & ((units["flags"][:c] & rf.JUMP_READY) != 0)
    charged, throws = table.jump[me] & ready, (table.throw_radius[me] > 0.0) & ready
    if (charged | throws).any():
        cells = bundle.passable[np.clip((bundle.centre[:, 1] / PASS_CELL).astype(np.int64), 0, bundle.passable.shape[0] - 1),
                                np.clip((bundle.centre[:, 0] / PASS_CELL).astype(np.int64), 0, bundle.passable.shape[1] - 1)]
        off_centre = np.sqrt(bundle.area / np.pi) * 0.5
        for i in np.flatnonzero(charged | throws):
            away = np.hypot(bundle.centre[:, 0] - units["x"][i], bundle.centre[:, 1] - units["z"][i])
            lands = ((cells >> classes[i]) & 1).astype(bool)
            if charged[i]:
                ptr_jump[i] = (ptr_region[i] | (away <= table.jump_range[me[i]] + off_centre)) & lands
            else:
                ptr_jump[i] = _throw_targets(tokens, i, (away <= table.throw_reach[me[i]] + off_centre) & lands, table)
    can_land = ptr_jump.any(axis=1)
    kind[:, rf.JUMP] = free & charged & can_land
    kind[:, rf.THROW] = free & throws & can_land
    cloaked = (units["flags"][:c] & rf.CLOAKED) != 0
    kind[:, rf.CLOAK_ON] = free & table.cloak[me] & ~cloaked
    kind[:, rf.CLOAK_OFF] = free & table.cloak[me] & cloaked
    pulling = (units["flags"][:c] & rf.PULLING) != 0
    kind[:, rf.PUSH] = free & table.switches[me] & pulling
    kind[:, rf.PULL] = free & table.switches[me] & ~pulling
    teleports = table.teleports[me]
    kind[:, rf.PLACE_BEACON] = free & teleports & mobile & can_go
    with_beacon = np.zeros(r, bool)
    with_beacon[unit_region[tokens.beacons][unit_region[tokens.beacons] >= 0]] = True
    ptr_wait = ptr_region & with_beacon[None, :] & (mobile & ~teleports)[:, None]
    kind[:, rf.WAIT_AT_BEACON] = free & ptr_wait.any(axis=1)
    kind[:, rf.PRODUCE] = free & ptr_produce.any(axis=1)
    off = np.ones(rf.KIND_COUNT, bool)
    off[list(enabled)] = False
    kind[:, off] = False
    return _Legal(kind, ptr_enemy, ptr_region, ptr_jump, ptr_wait, ptr_produce, route)


def _hidden(units: np.ndarray, rows: np.ndarray, army: np.ndarray, bundle: Bundle, table: TypeTable, player: int,
            ally: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enemies the player can't see, for the critic alone.  The actor's tensors never hold these."""
    idx = np.flatnonzero(units["knowledge"] == rf.KNOW_HIDDEN)
    if len(idx) > CAP_HIDDEN:
        d = np.hypot(units["x"][idx] - army[0], units["z"][idx] - army[1])
        idx = idx[np.lexsort((units["uid"][idx], d))[:CAP_HIDDEN]]
    hidden = units[idx]
    row = rows[idx]
    out = np.zeros((len(idx), UNIT_DIM), np.float32)
    out[:, :TYPE_DIM] = table.stats[row]
    if len(idx):
        _instance(hidden, row, bundle, table, out)
    return out, row.astype(np.int32), relative_id(hidden["ally_team"], ally, MAX_ALLY_TEAMS), relative_id(hidden["player"], player, MAX_PLAYERS)


def fine_point(t: Tensors, bundle: Bundle, region: int, option: int) -> tuple[float, float]:
    """Option 0 is the region's centre, 1-8 the compass points at half its equivalent radius, with north towards the enemy."""
    cx, cz = float(bundle.centre[region, 0]), float(bundle.centre[region, 1])
    if option == 0:
        return cx, cz
    angle = (option - 1) * np.pi / 4.0
    east = (t.north[1], -t.north[0])
    r = float(t.region_radius[region])
    return (cx + r * (float(np.cos(angle)) * t.north[0] + float(np.sin(angle)) * east[0]),
            cz + r * (float(np.cos(angle)) * t.north[1] + float(np.sin(angle)) * east[1]))
