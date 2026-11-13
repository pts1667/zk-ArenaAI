"""Raw frame to tensors and masks.  The one featurizer: the live bot, the sim's training loop and the tests all use it.

Numpy, with the per-unit loops in featurize_kernels (numba when it's installed).  Normalisation is fixed, not learned, so it can't
drift with training.  Tokens come in a fixed order:

    controllable units | every other unit, most important first | summaries | regions | sites | features | global

so the first `controllable` tokens are the units that get an action, and an index into the tensors is an index into `uids`.
All units share one pool, and each says whose it is through `unit_ally` and `unit_player`, numbered from the viewer: 0 is the viewer's
own, and any number of alliances and players fit.  Token kinds say only what is known of a unit.
Summaries carry what the caps left out.  `experience` isn't an input: the sim doesn't model it, and a feature the training
never saw vary is one the live game could only mislead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from operator import attrgetter, is_
from typing import NamedTuple

import numpy as np

from . import featurize_kernels as kernels
from . import rawframe as rf
from .bundle import PASS_CELL, Bundle
from .tables import TYPE_DIM, TypeTable

FEATURE_VERSION = 10         # what a saved encoder or actor was trained to read: raise it with any change to the tensors' meaning
                             # 7: ids sized for free-for-all, 64 players and 8 allyteams, so an unknown id reads 64 and 8 (2026-11-10)
                             # 8: caps raised for long and many-sided games, 1280 units, 512 controllable, 512 features (2026-11-10)
                             # 9: air transports (two instance columns, LOAD, UNLOAD and DROP), and aircraft controllable (2026-11-11)
                             # 10: the DPS sectors even, by floor, and the frame over 80 minutes, not 20 (the user, 2026-11-13)
ENCODER_FEATURE_VERSIONS = (10,)    # those an encoder reads alike
GAME_FRAMES = 144000         # 80 minutes, the sim's time limit: the frame's scale

# The user's, for long and many-sided games (2026-11-10): an 80-minute 1v1 had 656 records in sight on a side and 609 of its own.
# CAP_HIDDEN: the critic's enemies the seat can't see, nearest its army first.  256 since 2026-11-11 (the user): at 64 a late team
# game, free-for-all or chicken game left most of them out.  Only the critic reads them, and no stored data holds them.
CAP_CONTROLLABLE, CAP_UNITS, CAP_HIDDEN = 512, 1280, 256
CAP_SITES = 24
CAP_FEATURES = 512          # wrecks and the like in sight.  Past that, a region's metal says what's left out.
CAP_SUMMARIES = 12
# How the rest of the pool is chosen: what can be aimed at comes first, what's in sight before what's on radar, whichever way the
# armies stand.  Within each, value over distance from our army.  A blip's type may be unknown, so nothing counts for less than the floor.
LEAST_COST = kernels.LEAST_COST
if kernels.PASS_CELL != PASS_CELL:
    raise ImportError("bundle.PASS_CELL changed: update featurize_kernels' copy")

# Rows of the id embeddings, as well as one for an unknown id: enough for free-for-all games (Recoil's teams are players).
MAX_ALLY_TEAMS, MAX_PLAYERS = 8, 64

# Token kinds.
# TK_MEMORY_STATE is the last token of a collated batch (tools/policy_model.py), which the actors read as nothing in particular;
# the embeddings keep its row so saved encoders and actors load.
TK_SEEN, TK_BLIP, TK_MEMORY, TK_SUMMARY, TK_REGION, TK_SITE, TK_GLOBAL, TK_FEATURE, TK_HIDDEN, TK_MEMORY_STATE = range(10)
TOKEN_KINDS = 10
_KIND_OF_KNOWLEDGE = np.array((TK_SEEN, TK_BLIP, TK_MEMORY, TK_HIDDEN), np.int8)      # by rf.KNOW_*

COMMIT_DECISIONS = 6
DECISION_FRAMES = 15
STAGE2_KINDS = (*range(10), rf.PRODUCE, rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.LOAD, rf.UNLOAD, rf.DROP)

ECONOMY_KINDS = (rf.BUILD, rf.ASSIST, rf.RECLAIM)      # a constructor's (02-policy.md, "Kinds")
# A RECLAIM points at one of the feature tokens (wrecks and the like in sight, `feat_ids`), with the same field a BUILD's place has.

# The kinds that point at a target token, in the order of `ptr_enemy`: the first three at an enemy's, ASSIST at one of our own,
# and a transport's LOAD at either.  `enemy_tokens` holds them all, so one pointer, with a mask per kind, serves every kind.
ENEMY_POINTER_KINDS = (rf.ATTACK_UNIT, rf.ATTACK_GROUND, rf.MANUAL_FIRE, rf.ASSIST, rf.LOAD)
FINE_OPTIONS = 9
# What each kind is told besides itself.  A throw and a jump choose among the same regions: ptr_jump, where this unit's own reach lands.
# A transport's UNLOAD chooses there too (it neither jumps nor throws): where what it carries can stand.
KINDS_WITH_REGION = (rf.MOVE, rf.FIGHT, rf.JUMP, rf.THROW, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.UNLOAD)
KINDS_WITH_FINE = (rf.MOVE, rf.FIGHT, rf.JUMP, rf.ATTACK_GROUND, rf.THROW, rf.PLACE_BEACON, rf.UNLOAD)
REACH_KINDS = (rf.JUMP, rf.THROW, rf.UNLOAD)       # those whose regions are ptr_jump's
# A BUILD is a type (the produce head) at a place: beside one of our unit tokens or a metal site (`place_tokens`, then the sites),
# on it or at one of 8 compass points on each ring, north towards the enemy as for fine points; whoever obeys it finds the nearest
# place there the type fits.  On the economy script's builds this lands p50 19 elmos from its place, 99% within 150 (02-policy.md).
PLACE_RINGS = (64.0, 128.0, 256.0)
PLACE_OFFSETS = 1 + 8 * len(PLACE_RINGS)

# unit_x layout: type stats, instance, order memory, rays, summary.  The kernels fill all but the last.
_INSTANCE = kernels.INSTANCE_COLS
_ORDER = kernels.ORDER_COLS
_RAYS = kernels.RAY_COLS
_SUMMARY = 3
UNIT_DIM = TYPE_DIM + _INSTANCE + _ORDER + _RAYS + _SUMMARY
ALTITUDE_DEAD_ZONE = kernels.ALTITUDE_DEAD_ZONE
_I = TYPE_DIM
_O = _I + _INSTANCE
_R = _O + _ORDER
_S = _R + _RAYS

REGION_DIM = 2 + 1 + 2 + 1 + 9 + 6 + 2 + 2 + 2 + 6 + 1
SITE_DIM = 2 + 1 + 4 + 1 + 1
FEATURE_DIM = 2 + 1 + 1 + 1
GLOBAL_DIM = len(rf.GLOBAL_FIELDS) - 2     # the last two say who the viewer is, and aren't inputs

RAY_CELLS, RAY_STEP, RAY_DPS_RADIUS = kernels.RAY_CELLS, kernels.RAY_STEP, kernels.RAY_DPS_RADIUS


@dataclass
class Tensors:
    unit_x: np.ndarray            # U x UNIT_DIM
    unit_name: np.ndarray         # U: row in the name vocabulary
    reg_x: np.ndarray             # R x REGION_DIM
    site_x: np.ndarray            # S x SITE_DIM
    glob_x: np.ndarray            # GLOBAL_DIM
    tok_kind: np.ndarray          # N
    controllable: int             # C: the first C unit tokens get an action
    kind_mask: np.ndarray         # C x KIND_COUNT
    ptr_enemy: np.ndarray         # len(ENEMY_POINTER_KINDS) x C x E: legal targets per kind
    ptr_region: np.ndarray        # C x R: where it can walk to
    ptr_jump: np.ndarray          # C x R: where it can jump or throw to, which can be across what it can't walk; for a loaded
                                  # transport, where it can set its load down
    ptr_wait: np.ndarray          # C x R: where it can walk to a beacon of ours
    ptr_produce: np.ndarray       # C x T: what it can make, by place in table.producible
    route_bias: np.ndarray        # C x R: walking seconds to each region
    gate_a: np.ndarray            # G: region indices of each gate
    gate_b: np.ndarray            # G
    gate_x: np.ndarray            # G x GATE_DIM, normalised
    uids: np.ndarray              # U: uid per unit token, -1 for a summary
    enemy_tokens: np.ndarray      # E: unit-token index of each token that can be aimed at (any other alliance's, in sight or on
                                  # radar), helped (one of our own being built, or a factory) or, with a transport of ours
                                  # about, carried (one of ours that walks): the targets of ptr_enemy
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
    true_name: np.ndarray | None = None     # U: set if unit_name has had this game's names dropped
    place_tokens: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # P: unit-token index of each of ours a BUILD can be beside
    site_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))  # S x 2: each site's x, z in elmos
    feat_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))  # F x 2: each feature token's x, z in elmos


def relative_id(ids: np.ndarray, own: int, count: int) -> np.ndarray:
    """A game's numbers as the viewer's: its own is 0, and the others count from 1, skipping it.  So no seat is special, both seats of
    a 1v1 see the other as 1, and a number stays the same all game.  Unknown ids, and any past `count`, take the last row."""
    ids = ids.astype(np.int64)
    out = np.where(ids < own, ids + 1, ids)
    out[ids == own] = 0
    out[(ids < 0) | (out >= count)] = count
    return out.astype(np.int32)


# What each pack is made from.  A pack is kept on its table or bundle, and made again if any of these is no longer the object it was
# made from: a copy (copy.copy) carries the pack, and may have been given arrays of its own.
_TABLE_SOURCES = attrgetter("row_of_def", "cost", "speed", "range", "dps_ground", "dps_air", "manual_range", "manual_reload", "jump_range",
                            "throw_radius", "throw_reach", "lift", "lift_size", "load_class", "load_size", "transport",
                            "mobile", "air", "armed", "can_hit_ground", "can_hit_air", "builder", "factory",
                            "combat", "manualfire", "jump", "cloak", "switches", "teleports", "beacon", "missile", "launcher",
                            "acts_in_place", "move_class", "stats", "produces", "builds", "producible")
_BUNDLE_SOURCES = attrgetter("cell", "map_w", "map_h", "passable", "centre", "area", "region_height", "sea", "tags", "access", "shares",
                             "spots", "gate_counts", "gate_x")


def _current(owner: object, sources: attrgetter) -> NamedTuple | None:
    pack = owner.__dict__.get("_kernel_pack")
    return pack if pack is not None and all(map(is_, sources(owner), pack.sources)) else None


class _TablePack(NamedTuple):
    """The table as the kernels read it: few arrays, so that a call unpacks few (featurize_kernels' T_ and B_ columns)."""
    row_of_def: np.ndarray
    f: np.ndarray
    b: np.ndarray
    move_class: np.ndarray
    stats: np.ndarray
    produces: np.ndarray
    builds: np.ndarray
    producible: np.ndarray
    sources: tuple


def _table_pack(table: TypeTable) -> _TablePack:
    pack = _current(table, _TABLE_SOURCES)
    if pack is None:
        t = table
        f = np.stack((t.cost, t.speed, t.range, t.dps_ground, t.dps_air, t.manual_range, t.manual_reload, t.jump_range, t.throw_radius,
                      t.throw_reach, t.lift[:, 0], t.lift[:, 1], t.lift[:, 2], t.lift_size, t.load_class, t.load_size), 1).astype(np.float32)
        b = np.stack((t.mobile, t.air, t.armed, t.can_hit_ground, t.can_hit_air, t.builder, t.factory, t.combat, t.manualfire, t.jump,
                      t.cloak, t.switches, t.teleports, t.beacon, t.missile, t.launcher, t.acts_in_place, t.constructor,
                      t.produces.any(axis=1), t.builds.any(axis=1), t.transport), 1)
        c = np.ascontiguousarray
        pack = _TablePack(c(t.row_of_def), c(f), c(b), c(t.move_class), c(t.stats, np.float32), c(t.produces), c(t.builds),
                          c(t.producible, np.int64), _TABLE_SOURCES(t))
        table._kernel_pack = pack
    return pack


class _BundlePack(NamedTuple):
    """What featurize makes of a map's bundle alone, and its sizes as the float32 numpy would make of them."""
    cell: np.float32
    map_w: np.float32
    map_h: np.float32
    map_max: np.float32
    region_radius: np.ndarray       # R: half the equivalent radius, how far a fine position is off the centre
    region_cells: np.ndarray        # R: the passable bits at each region's centre
    reg_x: np.ndarray               # R x REGION_DIM: the columns that don't change
    gate_x: np.ndarray              # G x GATE_DIM, normalised
    sources: tuple


def _bundle_pack(bundle: Bundle) -> _BundlePack:
    pack = _current(bundle, _BUNDLE_SOURCES)
    if pack is None:
        rows, cols = bundle.passable.shape
        cells = bundle.passable[np.clip((bundle.centre[:, 1] / PASS_CELL).astype(np.int64), 0, rows - 1),
                                np.clip((bundle.centre[:, 0] / PASS_CELL).astype(np.int64), 0, cols - 1)]
        reg_x = np.zeros((bundle.regions, REGION_DIM), np.float32)
        reg_x[:, 0] = bundle.centre[:, 0] / bundle.map_w * 2.0 - 1.0
        reg_x[:, 1] = bundle.centre[:, 1] / bundle.map_h * 2.0 - 1.0
        reg_x[:, 2] = np.log1p(bundle.area) / 16.0
        reg_x[:, 3:5] = bundle.region_height / 300.0
        reg_x[:, 5] = bundle.sea
        reg_x[:, 6:15] = bundle.tags
        reg_x[:, 15:21] = bundle.access
        reg_x[:, 21:23] = bundle.shares
        reg_x[:, 23] = bundle.spots[:, 0] / 5.0
        reg_x[:, 24] = bundle.spots[:, 1] / 10.0
        reg_x[:, 25:27] = bundle.gate_counts / 4.0
        gate_x = bundle.gate_x.copy()
        gate_x[:, :6] = np.log1p(gate_x[:, :6]) / 8.0
        gate_x[:, 6:12] = gate_x[:, 6:12] / 2000.0
        pack = _BundlePack(np.float32(bundle.cell), np.float32(bundle.map_w), np.float32(bundle.map_h),
                           np.float32(max(bundle.map_w, bundle.map_h)), (np.sqrt(bundle.area / np.pi) * 0.5).astype(np.float32),
                           np.ascontiguousarray(cells), reg_x, gate_x, _BUNDLE_SOURCES(bundle))
        bundle._kernel_pack = pack
    return pack


@lru_cache(maxsize=None)
def _enabled(kinds: tuple[int, ...]) -> np.ndarray:
    on = np.zeros(rf.KIND_COUNT, bool)
    on[list(kinds)] = True
    return on


def _select(units: np.ndarray, table: TypeTable, player: int, ally: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray | None]:
    """Applies the caps.  Returns the indices kept, how many of them are controllable, those left over, and army and enemy centroids.
    Controllable: most expensive first, then nearest the army.  The rest: what can be aimed at first, in sight before on radar,
    whichever way the armies stand; within each, value over distance from our army.  uid breaks ties."""
    t = _table_pack(table)
    _, picked, c, left, army, foe, seen = kernels.select(units, t.row_of_def, t.f, t.b, player, ally, CAP_CONTROLLABLE, CAP_UNITS)
    return picked, c, left, army, foe if seen else None


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
            # A unit's health can be below zero in the frame it dies: summed into one below -1, it made log1p NaN, and with it every
            # number a cloning pass made from the frame (full-2's clone, 2026-11-11).  At 0 at least, as the unit tokens' is.
            hp = np.maximum(np.nan_to_num(units["hp_frac"][members], nan=1.0), 0.0)
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
    t, b = _table_pack(table), _bundle_pack(bundle)
    all_rows, picked, c, left, army, foe, seen = kernels.select(all_units, t.row_of_def, t.f, t.b, player, ally, CAP_CONTROLLABLE, CAP_UNITS)
    units = all_units[picked]
    rows = all_rows[picked]
    u = len(units)
    summary_x, summary_name, summary_ally, summary_player = _summaries(all_units, left, all_rows, bundle, table, player, ally)
    s = len(summary_x)

    # Unit tokens.
    unit_x = np.zeros((u + s, UNIT_DIM), np.float32)
    locked, unit_region, unit_ally, unit_player, enemy_tokens, foes, own_beacons, own_tokens = kernels.unit_block(
        units, rows, all_units, t.stats, t.f, t.b, t.move_class, bundle.label, bundle.height, bundle.passable, b.cell, b.map_w, b.map_h,
        b.map_max, player, ally, np.float32(commit_decisions * DECISION_FRAMES), MAX_ALLY_TEAMS, MAX_PLAYERS, unit_x)
    if s:
        unit_x[u:] = summary_x
    unit_name = np.concatenate((rows.astype(np.int32), summary_name))
    unit_ally = np.concatenate((unit_ally, summary_ally))
    unit_player = np.concatenate((unit_player, summary_player))

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
    tok_kind = np.concatenate((_KIND_OF_KNOWLEDGE[units["knowledge"]], np.full(s, TK_SUMMARY, np.int8), np.full(bundle.regions, TK_REGION, np.int8),
                               np.full(len(sites), TK_SITE, np.int8), np.full(len(features), TK_FEATURE, np.int8), np.array([TK_GLOBAL], np.int8)))

    # Actions.
    kind, ptr_enemy, ptr_region, ptr_jump, ptr_wait, ptr_produce, route = kernels.masks(
        units, rows, unit_region, enemy_tokens, foes, own_beacons, len(features), locked, c, t.f, t.b, t.move_class, t.produces, t.builds,
        t.producible, t.row_of_def, bundle.walk, b.region_cells, bundle.centre, b.region_radius, _enabled(tuple(enabled_kinds)))

    if not seen:
        foe = np.array((bundle.map_w * 0.5, bundle.map_h * 0.5), np.float32)
    north = foe - army
    norm = float(np.hypot(north[0], north[1]))
    north = north / norm if norm > 1.0 else np.array((0.0, 1.0), np.float32)

    out = Tensors(
        unit_x=unit_x, unit_name=unit_name, reg_x=reg_x, site_x=site_x, glob_x=glob_x, tok_kind=tok_kind,
        controllable=c, kind_mask=kind, ptr_enemy=ptr_enemy, ptr_region=ptr_region, ptr_jump=ptr_jump, ptr_wait=ptr_wait,
        ptr_produce=ptr_produce, route_bias=route, own_beacons=own_beacons, beacon_regions=unit_region[own_beacons],
        gate_a=bundle.edge_a, gate_b=bundle.edge_b, gate_x=b.gate_x.copy(),
        uids=np.concatenate((units["uid"], np.full(s, -1, np.int64))), enemy_tokens=enemy_tokens, north=north,
        region_radius=b.region_radius.copy(), unit_ally=unit_ally, unit_player=unit_player,
        feat_x=feat_x, feat_ids=features["feat_id"].astype(np.int64),
        place_tokens=own_tokens, site_pos=np.stack((sites["x"], sites["z"]), 1).astype(np.float32),
        feat_pos=np.stack((features["x"], features["z"]), 1).astype(np.float32),
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
    out = _bundle_pack(bundle).reg_x.copy()
    kernels.region_block(units, rows, features, _table_pack(table).f, bundle.label, _bundle_pack(bundle).cell, ally, out)
    return out


def _global(frame: rf.Frame, bundle: Bundle, economy: bool) -> np.ndarray:
    g = frame.glob
    out = np.zeros(GLOBAL_DIM, np.float32)
    idx = rf.G
    out[idx["frame"]] = g[idx["frame"]] / float(GAME_FRAMES)
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
    t, b = _table_pack(table), _bundle_pack(bundle)
    kernels.instance_block(hidden, row, t.stats, t.f, bundle.height, b.cell, b.map_w, b.map_h, out)
    return out, row.astype(np.int32), relative_id(hidden["ally_team"], ally, MAX_ALLY_TEAMS), relative_id(hidden["player"], player, MAX_PLAYERS)


def fine_point(t: Tensors, bundle: Bundle, region: int, option: int) -> tuple[float, float]:
    """Option 0 is the region's centre, 1-8 the compass points at half its equivalent radius, with north towards the enemy."""
    cx, cz = float(bundle.centre[region, 0]), float(bundle.centre[region, 1])
    if option == 0:
        return cx, cz
    # In float64, as actions.fine_points has the points encode chooses among (float32 until 2026-11-13: numpy's scalars).
    angle = (option - 1) * np.pi / 4.0
    c, s = float(np.cos(angle)), float(np.sin(angle))
    n0, n1 = float(t.north[0]), float(t.north[1])
    r = float(t.region_radius[region])
    return cx + r * (c * n0 + s * n1), cz + r * (c * n1 - s * n0)
