"""ctypes binding to libzkscripts (sim/scripts/zkscripts.h): the economy script, the aimer, the army script and its production in
C++, for many games at once on a thread pool: what `SimEnv` plays the scripts with.  They were ported from Python, which is gone (it's in
git at 8e9bca7); policy_eco.py and policy_mix.py describe what they do and build the tables they read.

The library is looked for at ZKSCRIPTS_LIB, else beside ZKSIM_LIB (one build makes both), else ~/zksim/build/libzkscripts.so.

`ScriptsData` is what the scripts know that doesn't change: each type-table row, read from `ScriptData`, `EcoData` and the factory menus,
and each map's ground, read from its `Ground` when a game first starts on it.  `Scripts` is a runner: the state of a number of
game slots and a thread pool of its own.
"""
from __future__ import annotations

import ctypes as ct
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot.policy import bundle as bd  # noqa: E402
from zkbot.policy import groups  # noqa: E402

import policy_eco as eco  # noqa: E402
from policy_mix import menu as factory_menu  # noqa: E402
from policy_variants import ROLES, Variant  # noqa: E402

ABI_VERSION = 8
MOVE_CLASSES = 6
SEAT_ARMY = 1
SEAT_ARMY_LABEL = 2     # the army script only advises the seat: its orders are labels, not carried out
SEAT_ECO_LABEL = 4      # the economy script only advises
ROLE_NAMES = ("raid", "riot", "assault", "skirm", "arty", "aa", "snipe", "scout", "builder", "bomb")
SPECIAL = (eco.CERBERUS, eco.LUCIFER, eco.SILO, eco.NUKE, eco.FUSION, eco.PYLON, eco.RADAR)
SECOND_FACTORIES = ("factoryplane", "factorygunship", "striderhub")      # zks_data_desc.second_factories
FINISHED = (eco.CERBERUS, eco.LUCIFER, eco.SILO, eco.NUKE, eco.FUSION, "singularity", "missile")   # zks_eco_stats, after checks
LOG_WHAT = ("start", "missile at", "nuke at")
(MOBILE, AIR, ARMED, BUILDER, FACTORY, FIGHTS, REGENERATES, EXTRACTOR, DEFENCE, KITABLE, HEAVY_RIOT, SLOW_HEAVY, MISSILE, COMMANDER,
 STATIC_ARMED, VALUABLE, CONSTRUCTOR, SINGULARITY) = (1 << k for k in range(18))

P = ct.POINTER
c_f32p, c_f64p, c_i32p, c_i8p, c_i16p, c_u8p, c_u32p = (P(ct.c_float), P(ct.c_double), P(ct.c_int32), P(ct.c_int8), P(ct.c_int16),
                                                         P(ct.c_uint8), P(ct.c_uint32))


class DataDesc(ct.Structure):
    _fields_ = [("rows", ct.c_int), ("n_def", ct.c_int), ("row_of_def", c_i32p), ("bits", c_u32p), ("cost", c_f32p), ("range", c_f32p),
                ("move_class", c_i8p), ("catalog_row", c_i32p), ("role", c_i32p), ("energy", c_f32p), ("pylon", c_f32p),
                ("needs_link", c_f32p), ("radius", c_f32p), ("footing", c_i32p), ("footing_radius", c_f64p), ("best_missile", c_i32p),
                ("generators", c_i32p), ("option_offsets", c_i32p), ("options", c_i32p), ("n_roles", ct.c_int), ("menu", c_i32p),
                ("plate_of", c_i32p),
                ("role_ids", ct.c_int32 * 10), ("special", ct.c_int32 * 7), ("second_factories", ct.c_int32 * 3),
                ("caretaker", ct.c_int32),
                ("group_radii", ct.c_double * len(groups.GROUP_RADII)), ("cerberus_reach", ct.c_double),
                ("missile_reach", ct.c_double), ("extractor_radius", ct.c_float), ("n_footings", ct.c_int)]


class MapDesc(ct.Structure):
    _fields_ = [("rows", ct.c_int), ("cols", ct.c_int), ("cell", ct.c_double), ("map_w", ct.c_double), ("map_h", ct.c_double),
                ("n_spots", ct.c_int), ("spots", c_f32p), ("masks", P(c_u8p)), ("site_counts", c_i32p), ("sites", P(c_f32p)),
                ("labels", P(c_i16p)), ("largest", c_i32p)]


class StepArgs(ct.Structure):
    _fields_ = [("n", ct.c_int), ("games", c_i32p), ("records", ct.c_void_p), ("cap", ct.c_int), ("counts", c_i32p), ("globs", c_f32p),
                ("stats", c_f32p), ("decisions", c_i32p), ("seat_flags", c_i32p), ("busy", c_i32p), ("ticks", ct.c_int), ("tick_frames", ct.c_int),
                ("frames_per_second", ct.c_int), ("script_every", ct.c_int), ("eco_every", ct.c_int), ("pre", ct.c_void_p),
                ("pre_offsets", c_i32p), ("post", ct.c_void_p), ("post_offsets", c_i32p), ("group_orders", ct.c_int),
                ("features", ct.c_void_p), ("feature_cap", ct.c_int), ("feature_counts", c_i32p)]


class Style(ct.Structure):
    """zks_style: the army script's knobs for one side, as policy_variants.Variant gives them."""
    _fields_ = [("role_weight", ct.c_double * len(ROLES)), ("commit", ct.c_double), ("gather", ct.c_double),
                ("patrol_share", ct.c_double), ("patience", ct.c_double), ("retreat_below", ct.c_double)]


def _style(v: Variant) -> Style:
    s = Style(commit=v.commit, gather=v.gather, patrol_share=v.patrol_share, patience=v.patience, retreat_below=v.retreat_below)
    s.role_weight[:] = v.role_weight
    return s


def _library_path() -> str:
    if os.environ.get("ZKSCRIPTS_LIB"):
        return os.environ["ZKSCRIPTS_LIB"]
    sim = os.environ.get("ZKSIM_LIB")
    if sim:
        return str(pathlib.Path(sim).with_name("libzkscripts.so"))
    return os.path.expanduser("~/zksim/build/libzkscripts.so")


_LIB: ct.CDLL | None = None


def lib() -> ct.CDLL:
    global _LIB
    if _LIB is None:
        L = ct.CDLL(_library_path())
        vp = ct.c_void_p
        L.zks_abi_version.restype = ct.c_int
        L.zks_data_new.restype = vp
        L.zks_data_new.argtypes = [P(DataDesc)]
        L.zks_data_add_map.restype = ct.c_int
        L.zks_data_add_map.argtypes = [vp, P(MapDesc)]
        L.zks_data_free.argtypes = [vp]
        L.zks_runner_new.restype = vp
        L.zks_runner_new.argtypes = [vp, ct.c_int, ct.c_int]
        L.zks_runner_free.argtypes = [vp]
        L.zks_reset.argtypes = [vp, ct.c_int, ct.c_int, c_f64p, c_f64p, ct.c_int, ct.c_int, ct.c_uint64, ct.c_uint64, P(Style)]
        L.zks_step.restype = ct.c_int
        L.zks_step.argtypes = [vp, P(StepArgs), vp, ct.c_int, c_i32p, vp, ct.c_int, c_i32p, vp, ct.c_int, c_i32p]
        L.zks_collect.restype = ct.c_int
        L.zks_collect.argtypes = [vp, vp, ct.c_int, c_i32p, vp, ct.c_int, c_i32p, vp, ct.c_int, c_i32p]
        L.zks_last_counts.argtypes = [vp, c_i32p]
        L.zks_seat_counts.argtypes = [vp, c_i32p]
        L.zks_eco_orders.restype = ct.c_int
        L.zks_eco_orders.argtypes = [vp, vp, ct.c_int, c_i32p]
        L.zks_made.restype = ct.c_int
        L.zks_made.argtypes = [vp, ct.c_int, ct.c_int, c_i32p]
        L.zks_eco_stats.argtypes = [vp, ct.c_int, ct.c_int, c_i32p]
        L.zks_log.restype = ct.c_int
        L.zks_log.argtypes = [vp, ct.c_int, ct.c_int, c_f64p, c_i32p, c_i32p, ct.c_int]
        L.zks_tape.argtypes = [vp, ct.c_int, ct.c_int, c_f64p, c_u8p, ct.c_int]
        L.zks_tape_misses.restype = ct.c_int
        L.zks_tape_misses.argtypes = [vp, ct.c_int, ct.c_int]
        L.zks_py_hypot.restype = ct.c_double
        L.zks_py_hypot.argtypes = [ct.c_double, ct.c_double]
        if L.zks_abi_version() != ABI_VERSION:
            raise RuntimeError("libzkscripts has another ABI version: rebuild it")
        _LIB = L
    return _LIB


def _ptr(a: np.ndarray, kind):
    assert a.flags.c_contiguous
    return a.ctypes.data_as(kind)


class ScriptsData:
    """The scripts' static data in C++, shared read-only by every `Scripts` made from it.  Maps are added as games first use them."""

    def __init__(self, script_data, eco_data):
        sd, ed, t = script_data, eco_data, script_data.table
        assert sd.table is ed.table
        n = len(t.names) + 1
        assert (sd.extractor == ed.extractor).all() and (sd.catalog_row == ed.catalog_row).all()
        self.roles = sorted(set(sd.role_of))
        role_index = {r: k for k, r in enumerate(self.roles)}
        bits = np.zeros(n, np.uint32)
        for bit, mask in ((MOBILE, t.mobile), (AIR, t.air), (ARMED, t.armed), (BUILDER, t.builder), (FACTORY, t.factory), (FIGHTS, sd.fights),
                          (REGENERATES, sd.regenerates), (EXTRACTOR, sd.extractor), (DEFENCE, sd.defence), (KITABLE, sd.kitable),
                          (HEAVY_RIOT, sd.heavy_riot), (SLOW_HEAVY, sd.slow_heavy), (MISSILE, ed.missile), (COMMANDER, ed.commander),
                          (STATIC_ARMED, ed.armed), (VALUABLE, ed.valuable), (SINGULARITY, ed.energy >= 100.0)):
            bits[np.asarray(mask, bool)[:n]] |= bit
        bits[[r for r in range(n) if sd.role_of[r] == "builder" and t.mobile[r]]] |= CONSTRUCTOR
        # Footings: their (low, high) water depths index each map's masks and site samples.
        self.footings: list[tuple[float, float]] = sorted({(f.low, f.high) for f in ed.footing.values()})
        footing = np.full(n, -1, np.int32)
        footing_radius = np.zeros(n, np.float64)
        for row, f in ed.footing.items():
            footing[row] = self.footings.index((f.low, f.high))
            footing_radius[row] = f.radius
        generators = np.full((n, 3), -1, np.int32)
        offsets, options = [0], []
        for row in range(n):
            opts = ed.options(row).tolist() if row < len(t.names) else []
            options += opts
            offsets.append(len(options))
            if opts:
                generators[row] = [-1 if g is None else g for g in ed.generators(row)]
        menu = np.full((n, len(self.roles)), -1, np.int32)
        for row in np.flatnonzero(t.factory[:n]):
            for role, unit in factory_menu(sd, int(row)).items():
                menu[row, role_index[role]] = unit
        best = np.full(n, -1, np.int32)
        for row, m in sd.best_missile.items():
            best[row] = m
        ext = np.flatnonzero(ed.extractor)
        self._arrays = dict(
            row_of_def=np.ascontiguousarray(t.row_of_def, np.int32), bits=bits, cost=np.ascontiguousarray(t.cost, np.float32),
            range=np.ascontiguousarray(t.range, np.float32), move_class=np.ascontiguousarray(t.move_class, np.int8),
            catalog_row=np.ascontiguousarray(sd.catalog_row, np.int32), role=np.array([role_index[r] for r in sd.role_of], np.int32),
            energy=np.ascontiguousarray(ed.energy, np.float32), pylon=np.ascontiguousarray(ed.pylon, np.float32),
            needs_link=np.ascontiguousarray(ed.needs_link, np.float32), radius=np.ascontiguousarray(ed.radius, np.float32),
            footing=footing, footing_radius=footing_radius, best_missile=best, generators=generators,
            option_offsets=np.array(offsets, np.int32), options=np.array(options + [0], np.int32), menu=menu,
            plate_of=np.ascontiguousarray(ed.plate_of[:n], np.int32))
        a = self._arrays
        assert tuple(self.roles) == groups.roles()[0], "the policy's group roles are numbered as the scripts' are"
        desc = DataDesc(rows=n, n_def=len(a["row_of_def"]), row_of_def=_ptr(a["row_of_def"], c_i32p), bits=_ptr(bits, c_u32p),
                        cost=_ptr(a["cost"], c_f32p), range=_ptr(a["range"], c_f32p), move_class=_ptr(a["move_class"], c_i8p),
                        catalog_row=_ptr(a["catalog_row"], c_i32p), role=_ptr(a["role"], c_i32p), energy=_ptr(a["energy"], c_f32p),
                        pylon=_ptr(a["pylon"], c_f32p), needs_link=_ptr(a["needs_link"], c_f32p), radius=_ptr(a["radius"], c_f32p),
                        footing=_ptr(footing, c_i32p), footing_radius=_ptr(footing_radius, c_f64p), best_missile=_ptr(best, c_i32p),
                        generators=_ptr(generators, c_i32p), option_offsets=_ptr(a["option_offsets"], c_i32p),
                        options=_ptr(a["options"], c_i32p), n_roles=len(self.roles), menu=_ptr(menu, c_i32p),
                        plate_of=_ptr(a["plate_of"], c_i32p),
                        cerberus_reach=float(ed.reach.get(eco.CERBERUS, 0.0)), missile_reach=float(ed.missile_reach),
                        extractor_radius=float(ed.radius[ext[0]]) if len(ext) else 24.0, n_footings=len(self.footings))
        desc.role_ids[:] = [role_index.get(r, -1) for r in ROLE_NAMES]
        desc.special[:] = [ed.row.get(name, -1) for name in SPECIAL]
        desc.second_factories[:] = [t.names.index(name) if name in t.names else -1 for name in SECOND_FACTORIES]
        desc.caretaker = t.names.index(eco.CARETAKER) if eco.CARETAKER in t.names else -1
        desc.group_radii[:] = groups.GROUP_RADII
        self.lib = lib()
        self.handle = self.lib.zks_data_new(ct.byref(desc))
        self._maps: dict[int, int] = {}         # id(Ground) -> index

    def __del__(self):
        if getattr(self, "handle", None) and _LIB is not None:
            _LIB.zks_data_free(self.handle)
            self.handle = None

    def map_index(self, ground) -> int:
        """The C++ index of this `Ground`'s map, adding it the first time.  Not while any runner of this data is stepping."""
        key = id(ground)
        if key not in self._maps:
            b = ground.bundle
            masks = [np.ascontiguousarray(ground.mask(low, high), np.uint8) for low, high in self.footings]
            sites = [np.ascontiguousarray(ground._sites(low, high), np.float32).reshape(-1, 2) for low, high in self.footings]
            labels = [np.ascontiguousarray(ground.labels(c), np.int16) for c in range(MOVE_CLASSES)]
            largest = np.array([ground._largest_part(c) for c in range(MOVE_CLASSES)], np.int32)
            spots = np.ascontiguousarray(b.spot_list, np.float32).reshape(-1, 3)
            assert all(m.shape == (ground.rows, ground.cols) for m in masks + labels)
            counts = np.array([len(s) for s in sites], np.int32)
            desc = MapDesc(rows=ground.rows, cols=ground.cols, cell=float(bd.PASS_CELL), map_w=float(b.map_w), map_h=float(b.map_h),
                           n_spots=len(spots), spots=_ptr(spots, c_f32p),
                           masks=(c_u8p * len(masks))(*[_ptr(m, c_u8p) for m in masks]), site_counts=_ptr(counts, c_i32p),
                           sites=(c_f32p * len(sites))(*[_ptr(s, c_f32p) for s in sites]),
                           labels=(c_i16p * MOVE_CLASSES)(*[_ptr(lab, c_i16p) for lab in labels]), largest=_ptr(largest, c_i32p))
            self._maps[key] = self.lib.zks_data_add_map(self.handle, ct.byref(desc))
        return self._maps[key]


class Scripts:
    """A runner: the scripts' state for `games` slots, and `threads` threads (the caller's included) to step them on."""

    def __init__(self, data: ScriptsData, games: int, threads: int = 1):
        self.data, self.games, self.lib = data, games, data.lib
        self.handle = self.lib.zks_runner_new(data.handle, games, threads)
        self._out = np.zeros(4096, zksim.ORDER_DTYPE)
        self._army = np.zeros(1024, zksim.ORDER_DTYPE)
        self._eco = np.zeros(1024, zksim.ORDER_DTYPE)
        self._groups = np.zeros(256, zksim.GROUP_DTYPE)
        self._empty = np.zeros(1, zksim.ORDER_DTYPE)

    def __del__(self):
        if getattr(self, "handle", None) and _LIB is not None:
            _LIB.zks_runner_free(self.handle)
            self.handle = None

    def reset(self, game: int, ground, starts: tuple[float, float, float, float], rally: tuple[float, float, float, float],
              economy: bool, aimer: bool, seeds: tuple[int, int] = (0, 0), variants: tuple[Variant, Variant] | None = None) -> None:
        """A new game in this slot.  starts: each side's base, x0, z0, x1, z1.  rally: each side's army script's rally.  variants: each
        side's army script knobs (policy_variants.py), or None for the standard script."""
        s, r = np.array(starts, np.float64), np.array(rally, np.float64)
        styles = None if variants is None else (Style * 2)(*(_style(v) for v in variants))
        self.lib.zks_reset(self.handle, game, self.data.map_index(ground), _ptr(s, c_f64p), _ptr(r, c_f64p), int(economy), int(aimer),
                           int(seeds[0]) & ((1 << 64) - 1), int(seeds[1]) & ((1 << 64) - 1), styles)

    def step(self, batch: zksim.Batch, decisions: np.ndarray, seat_flags: np.ndarray, busy: np.ndarray, ticks: int, script_every: int,
             eco_every: int, games: np.ndarray | None = None, pre: tuple[np.ndarray, np.ndarray] | None = None,
             post: tuple[np.ndarray, np.ndarray] | None = None, group_orders: bool = True) -> tuple[np.ndarray, ...]:
        """Every listed game's orders as `batch.step` takes them, the army script's alone by listed seat, and the group orders:
        (orders, offsets, army, army_offsets, groups, group_offsets).  The orders and groups are views, valid until the next step;
        the army's are a copy, as the policy would give them: alone, or a group order as its anchor's followed by a
        groups.GROUP_MARK record (x the radius, target the roles).  seat_flags, busy: by listed seat.
        pre, post: (orders, offsets by listed seat) to put before the scripts' and after the army script's.  group_orders: whether
        the army script sends what it can as group orders, which the batch then needs the roles of (policy_sim_env sets them)."""
        listed = np.arange(self.games, dtype=np.int32) if games is None else np.ascontiguousarray(games, np.int32)
        n = len(listed)
        flags = np.ascontiguousarray(seat_flags, np.int32)
        busy = np.ascontiguousarray(busy, np.int32)
        decisions = np.ascontiguousarray(decisions, np.int32)
        args = StepArgs(n=n, games=_ptr(listed, c_i32p), records=batch.records.ctypes.data, cap=batch.cap, counts=_ptr(batch.counts, c_i32p),
                        globs=_ptr(batch.globs, c_f32p), stats=_ptr(batch.stats, c_f32p), decisions=_ptr(decisions, c_i32p), seat_flags=_ptr(flags, c_i32p),
                        busy=_ptr(busy, c_i32p), ticks=ticks, tick_frames=zksim.TICK_FRAMES, frames_per_second=zksim.FRAMES_PER_SECOND,
                        script_every=script_every, eco_every=eco_every, group_orders=int(group_orders),
                        features=batch.features.ctypes.data, feature_cap=batch.features.shape[2], feature_counts=_ptr(batch.feature_counts, c_i32p))
        keep = []
        for name, part in (("pre", pre), ("post", post)):
            if part is not None:
                orders, offsets = np.ascontiguousarray(part[0], zksim.ORDER_DTYPE), np.ascontiguousarray(part[1], np.int32)
                assert len(offsets) == 2 * n + 1
                keep += [orders, offsets]
                setattr(args, name, (orders if len(orders) else self._empty).ctypes.data)
                setattr(args, f"{name}_offsets", _ptr(offsets, c_i32p))
        offsets = np.zeros(n + 1, np.int32)
        army_offsets = np.zeros(2 * n + 1, np.int32)
        group_offsets = np.zeros(n + 1, np.int32)
        buffers = lambda: (self._out.ctypes.data, len(self._out), _ptr(offsets, c_i32p), self._army.ctypes.data, len(self._army),  # noqa: E731
                           _ptr(army_offsets, c_i32p), self._groups.ctypes.data, len(self._groups), _ptr(group_offsets, c_i32p))
        total = self.lib.zks_step(self.handle, ct.byref(args), *buffers())
        counts = np.zeros(3, np.int32)
        self.lib.zks_last_counts(self.handle, _ptr(counts, c_i32p))
        if counts[0] > len(self._out) or counts[1] > len(self._army) or counts[2] > len(self._groups):
            self._out = np.zeros(max(2 * int(counts[0]), len(self._out)), zksim.ORDER_DTYPE)
            self._army = np.zeros(max(2 * int(counts[1]), len(self._army)), zksim.ORDER_DTYPE)
            self._groups = np.zeros(max(2 * int(counts[2]), len(self._groups)), zksim.GROUP_DTYPE)
            total = self.lib.zks_collect(self.handle, *buffers())
        return self._out[:total], offsets, self._army[:counts[1]].copy(), army_offsets, self._groups[:counts[2]], group_offsets

    def seat_counts(self, n: int) -> np.ndarray:
        """The last step's orders by part, per listed seat: pre, army script, post, economy script."""
        out = np.zeros((2 * n, 4), np.int32)
        self.lib.zks_seat_counts(self.handle, _ptr(out, c_i32p))
        return out

    def eco_orders(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The last step's economy-script orders alone, a copy, and their offsets by listed seat (2 n + 1)."""
        offsets = np.zeros(2 * n + 1, np.int32)
        total = self.lib.zks_eco_orders(self.handle, self._eco.ctypes.data, len(self._eco), _ptr(offsets, c_i32p))
        if total > len(self._eco):
            self._eco = np.zeros(2 * total, zksim.ORDER_DTYPE)
            self.lib.zks_eco_orders(self.handle, self._eco.ctypes.data, len(self._eco), _ptr(offsets, c_i32p))
        return self._eco[:total].copy(), offsets

    def made(self, game: int, side: int) -> dict[str, int]:
        """The army script's Mix.made: units started by role."""
        out = np.zeros(len(self.data.roles), np.int32)
        self.lib.zks_made(self.handle, game, side, _ptr(out, c_i32p))
        return {role: int(v) for role, v in zip(self.data.roles, out) if v}

    def eco_stats(self, game: int, side: int) -> tuple[dict[str, int], dict[str, int]]:
        """Economy.checks, and how many of each finished big building (Economy.finished, as counts)."""
        out = np.zeros(9, np.int32)
        self.lib.zks_eco_stats(self.handle, game, side, _ptr(out, c_i32p))
        return {"eligible": int(out[0]), "stalled": int(out[1])}, dict(zip(FINISHED, map(int, out[2:])))

    def log(self, game: int, side: int, table) -> list[tuple[float, str]]:
        """The economy's and the aimer's log together, as (seconds, what)."""
        n = self.lib.zks_log(self.handle, game, side, None, None, None, 0)
        seconds, what, row = np.zeros(n, np.float64), np.zeros(n, np.int32), np.zeros(n, np.int32)
        if n:
            self.lib.zks_log(self.handle, game, side, _ptr(seconds, c_f64p), _ptr(what, c_i32p), _ptr(row, c_i32p), n)
        return [(float(s), f"{LOG_WHAT[w]} {table.names[r]}") for s, w, r in zip(seconds, what, row)]

