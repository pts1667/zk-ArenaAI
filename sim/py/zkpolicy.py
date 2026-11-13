"""ctypes binding to libzkpolicy (sim/policy/zkpolicy.h): the type table, the featurizer and the actions in C++, bit for bit with
bots/zkbot/policy's Python, which stays the reference (tools/policy_parity.py checks them against each other).

The library is looked for at ZKPOLICY_LIB, else beside ZKSIM_LIB (one build makes both), else ~/zksim/build/libzkpolicy.so.

    featurize(frame, bundle, table, ...)        fz.featurize's drop-in: the same fz.Tensors, made in C++
    featurize_many(items, table, threads=8)     many (frame, bundle) pairs on C++ threads, each into a numpy buffer of its own
    decode / encode / fine_point / world_positions / to_slots / to_decision      actions.py's
    bucket / expand                             groups.py's

A frame's tensors are views of one buffer, which C++ wrote straight into: each array is C-contiguous and has fz.featurize's dtype
and shape, and gate_a and gate_b are the bundle's own, as fz.featurize gives them.

The type table: C++ reads a flat file of tables.TypeTable as the kernels read it (`table_bytes`; zkpolicy.h), not the catalog's
JSON, so the rules that make the table are Python's alone.  A TypeTable given here is turned into those bytes once (kept on it,
remade if its arrays are replaced, as featurize's own pack is).  The bot, with no Python, reads the same bytes from
bots/zkbot/data/policy_table.bin:

    ~/zkml/bin/python sim/py/zkpolicy.py table        # rewrites it, after the catalog or tables.py changes
"""
from __future__ import annotations

import ctypes as ct
import os
import pathlib
import struct
import sys
from operator import attrgetter, is_

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT / "bots") not in sys.path:
    sys.path.insert(0, str(ROOT / "bots"))
from zkbot import units as bot_units  # noqa: E402
from zkbot.policy import actions as act, bundle as bd, featurize as fz, groups as gr, rawframe as rf, tables  # noqa: E402

ABI_VERSION = 2
TABLE_MAGIC = b"ZKPTABLE"
TABLE_VERSION = 1
TABLE_FILE = ROOT / "bots" / "zkbot" / "data" / "policy_table.bin"

# zkpolicy.h's ZKP_F_*, in order: the name in fz.Tensors, its dtype and its dims.
FIELDS = (
    ("unit_x", np.float32, 2), ("unit_name", np.int32, 1), ("reg_x", np.float32, 2), ("site_x", np.float32, 2), ("glob_x", np.float32, 1),
    ("tok_kind", np.int8, 1), ("kind_mask", np.bool_, 2), ("ptr_enemy", np.bool_, 3), ("ptr_region", np.bool_, 2), ("ptr_jump", np.bool_, 2),
    ("ptr_wait", np.bool_, 2), ("ptr_produce", np.bool_, 2), ("route_bias", np.float32, 2), ("gate_x", np.float32, 2), ("uids", np.int64, 1),
    ("enemy_tokens", np.int64, 1), ("north", np.float32, 1), ("region_radius", np.float32, 1), ("own_beacons", np.int64, 1),
    ("beacon_regions", np.int32, 1), ("hidden_x", np.float32, 2), ("hidden_name", np.int32, 1), ("unit_ally", np.int32, 1),
    ("unit_player", np.int32, 1), ("hidden_ally", np.int32, 1), ("hidden_player", np.int32, 1), ("feat_x", np.float32, 2),
    ("feat_ids", np.int64, 1), ("place_tokens", np.int64, 1), ("site_pos", np.float32, 2), ("feat_pos", np.float32, 2),
)
_FIELD_TYPES = tuple((name, np.dtype(dtype), dims) for name, dtype, dims in FIELDS)
FRAME_DTYPE = np.dtype([("units", "u8"), ("n_units", "i8"), ("sites", "u8"), ("n_sites", "i8"), ("features", "u8"), ("n_features", "i8"),
                        ("glob", "u8"), ("bundle", "u8")])
ORDER_DTYPE = np.dtype([("uid", "i8"), ("kind", "i8"), ("x", "f8"), ("z", "f8"), ("target", "i8"), ("radius", "f8"), ("roles", "i8")])


class BundleArgs(ct.Structure):
    _fields_ = [("map_w", ct.c_double), ("map_h", ct.c_double), ("cell", ct.c_double),
                ("label_h", ct.c_int32), ("label_w", ct.c_int32), ("pass_rows", ct.c_int32), ("pass_cols", ct.c_int32),
                ("regions", ct.c_int32), ("gates", ct.c_int32)] + [
        (name, ct.c_void_p) for name in ("label", "height", "passable", "centre", "area", "region_height", "sea", "tags", "access", "shares",
                                         "spots", "gate_counts", "gate_x", "walk")]


class Options(ct.Structure):
    _fields_ = [("commit_decisions", ct.c_int32), ("privileged", ct.c_int32), ("economy", ct.c_int32), ("enabled", ct.c_uint8 * rf.KIND_COUNT)]


_TENSOR_COUNTS = ("controllable", "units", "enemies", "regions", "beacons", "places", "sites", "features", "producible")
_TENSOR_ARRAYS = (("unit_name", np.int32), ("uids", np.int64), ("enemy_tokens", np.int64), ("kind_mask", np.bool_), ("ptr_enemy", np.bool_),
                  ("ptr_region", np.bool_), ("ptr_jump", np.bool_), ("ptr_wait", np.bool_), ("ptr_produce", np.bool_), ("north", np.float32),
                  ("region_radius", np.float32), ("own_beacons", np.int64), ("beacon_regions", np.int32), ("place_tokens", np.int64),
                  ("site_pos", np.float32), ("feat_pos", np.float32), ("feat_ids", np.int64))


class TensorsArgs(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in _TENSOR_COUNTS] + [(name, ct.c_void_p) for name, _ in _TENSOR_ARRAYS] + [
        ("unit_x", ct.c_void_p), ("unit_z", ct.c_void_p)]


def _library_path() -> str:
    if os.environ.get("ZKPOLICY_LIB"):
        return os.environ["ZKPOLICY_LIB"]
    sim = os.environ.get("ZKSIM_LIB")
    if sim:
        return str(pathlib.Path(sim).with_name("libzkpolicy.so"))
    return os.path.expanduser("~/zksim/build/libzkpolicy.so")


_LIB: ct.CDLL | None = None


def lib() -> ct.CDLL:
    global _LIB
    if _LIB is None:
        L = ct.CDLL(_library_path())
        vp, i64, i32, f64 = ct.c_void_p, ct.c_int64, ct.c_int32, ct.c_double
        L.zkp_abi_version.restype = ct.c_int
        if L.zkp_abi_version() != ABI_VERSION:
            raise ImportError(f"{_library_path()} is ABI {L.zkp_abi_version()}, this binding {ABI_VERSION}: rebuild it")
        L.zkp_table_new.restype = vp
        L.zkp_table_new.argtypes = [ct.c_char_p, i64, ct.c_char_p, i32]
        L.zkp_table_free.argtypes = [vp]
        L.zkp_table_info.argtypes = [vp, vp]
        L.zkp_bundle_new.restype = vp
        L.zkp_bundle_new.argtypes = [ct.POINTER(BundleArgs)]
        L.zkp_bundle_free.argtypes = [vp]
        L.zkp_bounds.argtypes = [vp, vp, i64, i32, vp]
        L.zkp_featurize.restype = i64
        L.zkp_featurize.argtypes = [vp, vp, i64, ct.POINTER(Options), vp, vp, vp, vp, i32]
        L.zkp_decode.restype = i64
        L.zkp_decode.argtypes = [vp, vp, ct.POINTER(TensorsArgs), vp, vp]
        L.zkp_encode.restype = i64
        L.zkp_encode.argtypes = [vp, vp, ct.POINTER(TensorsArgs), vp, i64, vp]
        L.zkp_fine_point.argtypes = [vp, ct.POINTER(TensorsArgs), i64, i64, vp]
        L.zkp_world_positions.argtypes = [vp, i64, i64, i64, f64, f64, vp, vp]
        L.zkp_to_slots.restype = i64
        L.zkp_to_slots.argtypes = [vp, i64, vp, i64, vp]
        L.zkp_acting.restype = i64
        L.zkp_acting.argtypes = [vp, i64]
        L.zkp_to_decision.argtypes = [vp, i64, i64, vp]
        L.zkp_group_bucket.restype = i64
        L.zkp_group_bucket.argtypes = [f64]
        L.zkp_group_expand.argtypes = [vp, i64, i64, f64, vp, i64, vp, vp]
        _LIB = L
    return _LIB


def _ptr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


# The type table ---------------------------------------------------------------------------------------------------------------------------------

def _roles(table: tables.TypeTable) -> tuple[tuple[str, ...], np.ndarray, int]:
    """groups.roles and groups.choosable, for any table: the global one's are groups'."""
    if table is tables.table():
        names, role_of = gr.roles()
        return names, role_of, gr.choosable()
    from zkbot import catalog
    role_of = [catalog.role(name) for name in table.names] + ["other"]
    names = tuple(sorted(set(role_of)))
    rows = np.array([names.index(r) for r in role_of], np.int64)
    mobile = np.asarray(table.mobile, bool)[:len(table.names)]
    return names, rows, int(np.bitwise_or.reduce(1 << np.unique(rows[:len(table.names)][mobile])))


def table_bytes(table: tables.TypeTable) -> bytes:
    """The flat file of a TypeTable (zkpolicy.h): the kernels' pack of it (featurize._table_pack), each row's def id, the group
    roles, the names, and its catalog hash."""
    pack = fz._table_pack(table)
    role_names, role_of, choosable = _roles(table)
    rows = len(table.names) + 1
    def_of_row = np.array([bot_units.id_of(name) if bot_units.id_of(name) is not None else -1 for name in table.names] + [-1], np.int32)
    names = b"".join(name.encode() + b"\0" for name in table.names)
    role_block = b"".join(name.encode() + b"\0" for name in role_names)
    arrays = [np.ascontiguousarray(pack.stats, np.float32), np.ascontiguousarray(pack.f, np.float32), np.ascontiguousarray(pack.b, np.uint8),
              np.ascontiguousarray(pack.move_class, np.int8), np.ascontiguousarray(pack.produces, np.uint8),
              np.ascontiguousarray(pack.builds, np.uint8), np.ascontiguousarray(pack.producible, np.int64),
              np.ascontiguousarray(pack.row_of_def, np.int32), def_of_row, np.ascontiguousarray(role_of, np.int32)]
    head = struct.pack("<8s12I16s", TABLE_MAGIC, TABLE_VERSION, rows, pack.stats.shape[1], pack.f.shape[1], pack.b.shape[1],
                       len(pack.producible), len(pack.row_of_def), len(role_names), len(names), len(role_block), table.unknown, choosable,
                       table.hash.encode()[:16])
    out = [head]
    for block in [a.tobytes() for a in arrays] + [names, role_block]:
        out.append(block + b"\0" * (-len(block) % 8))
    return b"".join(out)


def write_table(path: str | os.PathLike = TABLE_FILE, table: tables.TypeTable | None = None) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_bytes(table_bytes(table or tables.table()))
    return path


class Table:
    """A table in the library: from a TypeTable, or from the flat file as the bot reads it."""

    def __init__(self, data: bytes):
        error = ct.create_string_buffer(256)
        self._data = data
        self.ptr = lib().zkp_table_new(data, len(data), error, len(error))
        if not self.ptr:
            raise ValueError(f"not a policy table: {error.value.decode()}")
        info = np.zeros(8, np.int64)
        lib().zkp_table_info(self.ptr, _ptr(info))
        self.rows, self.type_dim, self.unit_dim, self.producible, self.unknown, self.roles, self.choosable, self.defs = (int(v) for v in info)

    @classmethod
    def load(cls, path: str | os.PathLike = TABLE_FILE) -> "Table":
        return cls(pathlib.Path(path).read_bytes())

    def __del__(self):
        if getattr(self, "ptr", None) and _LIB is not None:
            _LIB.zkp_table_free(self.ptr)
            self.ptr = None


_TABLE_SOURCES = fz._TABLE_SOURCES


def _table(table: tables.TypeTable | Table | None) -> Table:
    if isinstance(table, Table):
        return table
    table = table if table is not None else tables.table()
    held = table.__dict__.get("_zkpolicy")
    if held is not None and all(map(is_, _TABLE_SOURCES(table), held[1])):
        return held[0]
    made = Table(table_bytes(table))
    table._zkpolicy = (made, _TABLE_SOURCES(table))
    return made


# Bundles --------------------------------------------------------------------------------------------------------------------------------------------

_BUNDLE_ARRAYS = (("label", np.int16), ("height", np.float32), ("passable", np.uint8), ("centre", np.float32), ("area", np.float32),
                  ("region_height", np.float32), ("sea", np.float32), ("tags", np.float32), ("access", np.float32), ("shares", np.float32),
                  ("spots", np.float32), ("gate_counts", np.float32), ("gate_x", np.float32), ("walk", np.float32))
_BUNDLE_SOURCES = attrgetter("map_w", "map_h", "cell", *(name for name, _ in _BUNDLE_ARRAYS))


class _Bundle:
    """A bundle in the library: views of its arrays, held here as long as it is."""

    def __init__(self, bundle: bd.Bundle):
        self.arrays = {name: np.ascontiguousarray(getattr(bundle, name), dtype) for name, dtype in _BUNDLE_ARRAYS}
        a = self.arrays
        if a["walk"].shape != (len(bd.MOVE_CLASSES), len(a["centre"]), len(a["centre"])) or a["label"].shape != a["height"].shape:
            raise ValueError("a bundle's arrays don't agree on its regions or cells")
        args = BundleArgs(float(bundle.map_w), float(bundle.map_h), float(bundle.cell), a["label"].shape[0], a["label"].shape[1],
                          a["passable"].shape[0], a["passable"].shape[1], len(a["centre"]), len(a["gate_x"]),
                          *(_ptr(a[name]) for name, _ in _BUNDLE_ARRAYS))
        self.ptr = lib().zkp_bundle_new(ct.byref(args))
        self.edge_a, self.edge_b = bundle.edge_a, bundle.edge_b

    def __del__(self):
        if getattr(self, "ptr", None) and _LIB is not None:
            _LIB.zkp_bundle_free(self.ptr)
            self.ptr = None


def _bundle(bundle: bd.Bundle) -> _Bundle:
    held = bundle.__dict__.get("_zkpolicy")
    if held is not None and all(map(is_, _BUNDLE_SOURCES(bundle), held[1])) and held[0].edge_a is bundle.edge_a and held[0].edge_b is bundle.edge_b:
        return held[0]
    made = _Bundle(bundle)
    bundle._zkpolicy = (made, _BUNDLE_SOURCES(bundle))
    return made


# Featurizing ----------------------------------------------------------------------------------------------------------------------------------------

def _records(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if a.dtype == dtype:
        return np.ascontiguousarray(a)
    out = np.zeros(len(a), dtype)
    for name in a.dtype.names:
        if name in dtype.names:
            out[name] = a[name]
    return out


def _options(commit_decisions: int, enabled_kinds, privileged: bool, economy: bool) -> Options:
    on = fz._enabled(tuple(enabled_kinds))
    return Options(int(commit_decisions), int(bool(privileged)), int(bool(economy)), (ct.c_uint8 * rf.KIND_COUNT)(*on.astype(np.uint8)))


def _frames(items: list[tuple[rf.Frame, bd.Bundle]]) -> tuple[np.ndarray, list, list[_Bundle]]:
    frames = np.zeros(len(items), FRAME_DTYPE)
    held, handles = [], []
    for k, (frame, bundle) in enumerate(items):
        u, s, f = _records(frame.units, rf.UNIT_DTYPE), _records(frame.sites, rf.SITE_DTYPE), _records(frame.features, rf.FEATURE_DTYPE)
        g = np.ascontiguousarray(frame.glob, np.float32)
        b = _bundle(bundle)
        held.append((u, s, f, g))
        handles.append(b)
        frames[k] = (_ptr(u), len(u), _ptr(s), len(s), _ptr(f), len(f), _ptr(g), b.ptr)
    return frames, held, handles


def bounds(items: list[tuple[rf.Frame, bd.Bundle]], table: tables.TypeTable | Table | None = None, privileged: bool = False) -> np.ndarray:
    """The bytes each (frame, bundle)'s tensors take at most: how big a buffer featurize_many needs for it."""
    frames, _held, _handles = _frames(items)
    out = np.empty(len(items), np.int64)
    if len(items):
        lib().zkp_bounds(_table(table).ptr, _ptr(frames), len(items), int(bool(privileged)), _ptr(out))
    return out


def featurize_many(items: list[tuple[rf.Frame, bd.Bundle]], table: tables.TypeTable | Table | None = None, threads: int = 1,
                   commit_decisions: int = fz.COMMIT_DECISIONS, enabled_kinds: tuple[int, ...] = fz.STAGE2_KINDS, privileged: bool = False,
                   economy: bool = False, buffers: list[np.ndarray] | None = None) -> list[fz.Tensors]:
    """fz.featurize of each (frame, bundle), on `threads` C++ threads (made for the call; the GIL is let go meanwhile).

    Each frame's tensors are views of a uint8 buffer: `buffers[k]` if given (C-contiguous, at least `bounds(...)[k]` bytes; a
    worker that keeps its buffers from batch to batch saves the pages' first touch), else one made for it.  A buffer given is
    overwritten, so its tensors are good until it's used again."""
    n = len(items)
    if not n:
        return []
    t = _table(table)
    frames, held, handles = _frames(items)
    L = lib()
    bounds = np.empty(n, np.int64)
    L.zkp_bounds(t.ptr, _ptr(frames), n, int(bool(privileged)), _ptr(bounds))
    if buffers is None:
        buffers = [np.empty(int(size), np.uint8) for size in bounds]
    elif len(buffers) != n or any(b.dtype != np.uint8 or not b.flags.c_contiguous or b.nbytes < size for b, size in zip(buffers, bounds)):
        raise ValueError("featurize_many: a buffer a frame, uint8, C-contiguous and at least bounds() bytes")
    pointers = np.array([_ptr(buf) for buf in buffers], np.uint64)
    layouts = np.zeros((n, len(FIELDS), 4), np.int64)
    controllable = np.zeros(n, np.int64)
    opt = _options(commit_decisions, enabled_kinds, privileged, economy)
    failed = L.zkp_featurize(t.ptr, _ptr(frames), n, ct.byref(opt), _ptr(pointers), _ptr(bounds), _ptr(layouts), _ptr(controllable), int(threads))
    if failed:
        raise RuntimeError(f"zkp_featurize: {failed} of {n} frames failed")
    return [_tensors(buffers[k], layouts[k], int(controllable[k]), handles[k]) for k in range(n)]


def _tensors(buf: np.ndarray, layout: np.ndarray, controllable: int, b: _Bundle) -> fz.Tensors:
    got = {}
    for (name, dtype, dims), (offset, d0, d1, d2) in zip(_FIELD_TYPES, layout.tolist()):
        shape = (d0,) if dims == 1 else (d0, d1) if dims == 2 else (d0, d1, d2)
        got[name] = np.ndarray(shape, dtype, buffer=buf, offset=offset)
    return fz.Tensors(controllable=controllable, gate_a=b.edge_a, gate_b=b.edge_b, **got)


def featurize(frame: rf.Frame, bundle: bd.Bundle, table: tables.TypeTable | Table | None = None, commit_decisions: int = fz.COMMIT_DECISIONS,
              enabled_kinds: tuple[int, ...] = fz.STAGE2_KINDS, privileged: bool = False, economy: bool = False) -> fz.Tensors:
    """fz.featurize's drop-in."""
    return featurize_many([(frame, bundle)], table, 1, commit_decisions, enabled_kinds, privileged, economy)[0]


# Actions --------------------------------------------------------------------------------------------------------------------------------------------

def _tensors_args(t: fz.Tensors, unit_x: np.ndarray, unit_z: np.ndarray) -> tuple[TensorsArgs, list]:
    # A throw's reach goes by the unit's real name, not the one a training game may have dropped (fz.Tensors.true_name).
    names = t.unit_name if t.true_name is None else t.true_name
    arrays = [np.ascontiguousarray(names if name == "unit_name" else getattr(t, name), dtype) for name, dtype in _TENSOR_ARRAYS]
    arrays += [np.ascontiguousarray(unit_x, np.float32), np.ascontiguousarray(unit_z, np.float32)]
    counts = (int(t.controllable), len(t.uids), len(t.enemy_tokens), len(t.region_radius), len(t.own_beacons), len(t.place_tokens),
              len(t.site_pos), len(t.feat_ids), t.ptr_produce.shape[1])
    return TensorsArgs(*counts, *(_ptr(a) for a in arrays)), arrays


def _decision_array(d: act.Decision, controllable: int) -> np.ndarray:
    return np.ascontiguousarray(np.stack([np.asarray(getattr(d, f), np.int64).reshape(controllable) for f in act.FIELDS]))


def decode(decision: act.Decision, t: fz.Tensors, bundle: bd.Bundle, unit_x: np.ndarray, unit_z: np.ndarray,
           table: tables.TypeTable | Table | None = None) -> list[act.Order]:
    """actions.decode's drop-in."""
    args, held = _tensors_args(t, unit_x, unit_z)
    d = _decision_array(decision, t.controllable)
    out = np.zeros(max(t.controllable, 1), ORDER_DTYPE)
    got = lib().zkp_decode(_table(table).ptr, _bundle(bundle).ptr, ct.byref(args), _ptr(d), _ptr(out))
    if got < 0:
        raise IndexError(f"decode: unit {-1 - got}'s choices point past the tensors")
    return [act.Order(uid, kind, x, z, target, radius, roles) for uid, kind, x, z, target, radius, roles in out[:got].tolist()]


def orders_array(orders: list[act.Order]) -> np.ndarray:
    return np.array([(o.uid, o.kind, o.x, o.z, o.target, o.radius, o.roles) for o in orders], ORDER_DTYPE).reshape(len(orders))


def encode(orders: list[act.Order] | np.ndarray, t: fz.Tensors, bundle: bd.Bundle, unit_x: np.ndarray, unit_z: np.ndarray,
           table: tables.TypeTable | Table | None = None) -> tuple[act.Decision, int]:
    """actions.encode's drop-in; orders may be ORDER_DTYPE records."""
    rows = orders if isinstance(orders, np.ndarray) else orders_array(orders)
    rows = np.ascontiguousarray(rows, ORDER_DTYPE)
    args, held = _tensors_args(t, unit_x, unit_z)
    d = np.empty((len(act.FIELDS), t.controllable), np.int64)
    missed = lib().zkp_encode(_table(table).ptr, _bundle(bundle).ptr, ct.byref(args), _ptr(rows), len(rows), _ptr(d))
    return act.Decision(*d), int(missed)


def fine_point(t: fz.Tensors, bundle: bd.Bundle, region: int, option: int) -> tuple:
    """fz.fine_point, as Python floats."""
    args, held = _tensors_args(t, np.zeros(len(t.uids), np.float32), np.zeros(len(t.uids), np.float32))
    out = np.zeros(2)
    lib().zkp_fine_point(_bundle(bundle).ptr, ct.byref(args), int(region), int(option), _ptr(out))
    return float(out[0]), float(out[1])


def world_positions(t: fz.Tensors, bundle: bd.Bundle) -> tuple[np.ndarray, np.ndarray]:
    x, z = np.empty(len(t.unit_x), np.float32), np.empty(len(t.unit_x), np.float32)
    ux = np.ascontiguousarray(t.unit_x, np.float32)
    lib().zkp_world_positions(_ptr(ux), len(ux), ux.shape[1] if ux.ndim == 2 else fz.UNIT_DIM, fz._I, float(bundle.map_w), float(bundle.map_h),
                              _ptr(x), _ptr(z))
    return x, z


def to_slots(d: act.Decision, order: np.ndarray | None = None, slots: int | None = act.SLOTS) -> act.Slots:
    c = len(d.kind)
    a = _decision_array(d, c)
    o = None if order is None else np.ascontiguousarray(order, np.int64)
    width = lib().zkp_acting(_ptr(a), c) if slots is None else int(slots)
    out = np.empty((len(act.SLOT_FIELDS), width), np.int64)
    lib().zkp_to_slots(_ptr(a), c, None if o is None else _ptr(o), width, _ptr(out))
    return act.Slots(*out)


def to_decision(s: act.Slots, controllable: int) -> act.Decision:
    rows = np.ascontiguousarray(np.stack([np.asarray(getattr(s, f), np.int64) for f in act.SLOT_FIELDS]))
    out = np.empty((len(act.FIELDS), controllable), np.int64)
    lib().zkp_to_decision(_ptr(rows), rows.shape[1], controllable, _ptr(out))
    return act.Decision(*out)


def bucket(radius: float) -> int:
    return int(lib().zkp_group_bucket(float(radius)))


def expand(anchor: int, roles_mask: int, radius: float, units: np.ndarray, rows: np.ndarray, table: tables.TypeTable | Table | None = None) -> np.ndarray:
    u = _records(units, rf.UNIT_DTYPE)
    r = np.ascontiguousarray(rows, np.int64)
    out = np.empty(len(u), np.bool_)
    lib().zkp_group_expand(_table(table).ptr, int(anchor), int(roles_mask), float(radius), _ptr(u), len(u), _ptr(r), _ptr(out))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["table"]:
        print(write_table(*sys.argv[2:3]))
    else:
        print(__doc__)
