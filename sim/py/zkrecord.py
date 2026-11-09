"""ctypes binding to libzkrecord (sim/record/zkrecord.h), which records a batch's games straight from its arrays, and the reader of its
files.  zkrecord.h describes the file.

The library is looked for at ZKRECORD_LIB, else beside ZKSIM_LIB (one build makes both), else ~/zksim/build/libzkrecord.so.

`Writer` wraps a recorder: `begin` a slot's run, `frames` then `orders` every step, `finish` it to a file, and `wait` for the files.  `read` gives a file back
as frames, orders and its meta for each side, as `rawframe.Frame`s: what was narrowed comes back within its rounding (zkrecord.h).
"""
from __future__ import annotations

import ctypes as ct
import json
import os
import pathlib

import numpy as np

import zksim
from zkbot.policy import rawframe

ABI_VERSION = 1
MAGIC = b"ZKREC001"
CLAMP_NAMES = ("x", "z", "y", "ord_x", "ord_z", "build_frac", "ord_age", "half", "feature_xz")
GLOBALS = len(rawframe.GLOBAL_FIELDS)
SUFFIX = ".zkr"

# As ZKR_UNIT_FIELDS, ZKR_FEATURE_FIELDS and ZKR_ORDER_FIELDS in record.cpp, in their order.
NARROW_UNIT = np.dtype([("uid", "i4"), ("ord_target", "i4"), ("def_id", "i2"), ("x", "u2"), ("z", "u2"), ("y", "u2"), ("vx", "u2"), ("vz", "u2"),
                        ("hp_frac", "u2"), ("build_frac", "u2"), ("experience", "u2"), ("blip_top_speed", "u2"), ("ord_age", "u2"),
                        ("ord_x", "u2"), ("ord_z", "u2"), ("flags", "u2"), ("knowledge", "u1"), ("ord_kind", "i1"), ("player", "i1"),
                        ("ally_team", "i1")])
NARROW_FEATURE = np.dtype([("feat_id", "i4"), ("x", "u2"), ("z", "u2"), ("metal", "u4"), ("def_id", "i2")])
_HALF = ("vx", "vz", "hp_frac", "experience", "blip_top_speed")


class FramesArgs(ct.Structure):
    _fields_ = [("n", ct.c_int), ("games", ct.POINTER(ct.c_int32)), ("records", ct.c_void_p), ("cap", ct.c_int),
                ("counts", ct.POINTER(ct.c_int32)), ("globs", ct.POINTER(ct.c_float)), ("features", ct.c_void_p), ("feature_cap", ct.c_int),
                ("feature_counts", ct.POINTER(ct.c_int32))]


def _library_path() -> str:
    if os.environ.get("ZKRECORD_LIB"):
        return os.environ["ZKRECORD_LIB"]
    sim = os.environ.get("ZKSIM_LIB")
    if sim:
        return str(pathlib.Path(sim).with_name("libzkrecord.so"))
    return os.path.expanduser("~/zksim/build/libzkrecord.so")


_LIB: ct.CDLL | None = None


def lib() -> ct.CDLL:
    global _LIB
    if _LIB is None:
        L = ct.CDLL(_library_path())
        vp, i32p = ct.c_void_p, ct.POINTER(ct.c_int32)
        L.zkr_abi_version.restype = ct.c_int
        L.zkr_new.restype = vp
        L.zkr_new.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        L.zkr_free.argtypes = [vp]
        L.zkr_begin.argtypes = [vp, ct.c_int, ct.c_float, ct.c_float, ct.c_float, ct.c_float, vp, ct.c_int]
        L.zkr_frames.argtypes = [vp, ct.POINTER(FramesArgs)]
        L.zkr_orders.argtypes = [vp, ct.c_int, i32p, vp, i32p]
        L.zkr_finish.restype = ct.c_int
        L.zkr_finish.argtypes = [vp, ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_int]
        L.zkr_wait.restype = ct.c_int
        L.zkr_wait.argtypes = [vp]
        L.zkr_frames_held.restype = ct.c_int
        L.zkr_frames_held.argtypes = [vp, ct.c_int]
        L.zkr_clamps.argtypes = [vp, ct.POINTER(ct.c_int64)]
        L.zkr_frame_size.restype = ct.c_int64
        L.zkr_frame_size.argtypes = [ct.c_char_p, ct.c_int64]
        L.zkr_decompress.restype = ct.c_int64
        L.zkr_decompress.argtypes = [ct.c_char_p, ct.c_int64, vp, ct.c_int64]
        if L.zkr_abi_version() != ABI_VERSION:
            raise RuntimeError("libzkrecord has another ABI version: rebuild it")
        _LIB = L
    return _LIB


def _i32(a: np.ndarray):
    assert a.dtype == np.int32 and a.flags.c_contiguous
    return a.ctypes.data_as(ct.POINTER(ct.c_int32))


def height_range(heights: np.ndarray) -> tuple[float, float]:
    """The y a run's units are kept in, from its map's heights: the sea floor or the sea (hovers and ships ride it) to the highest
    ground plus aircraft and jumps, with room for the heightmap being finer than `heights`."""
    return min(float(heights.min()), 0.0) - 256.0, max(float(heights.max()), 0.0) + 768.0


class Writer:
    """libzkrecord's recorder, for the games of a `zksim.Batch`: a slot a game, at the game's position in the batch."""

    def __init__(self, slots: int, threads: int = 1, level: int = 3, chunk_frames: int = 64):
        self.lib, self.slots = lib(), slots
        self.handle = self.lib.zkr_new(slots, threads, level, chunk_frames)

    def __del__(self):
        if getattr(self, "handle", None) and _LIB is not None:
            _LIB.zkr_free(self.handle)
            self.handle = None

    def begin(self, slot: int, map_w: float, map_h: float, y_range: tuple[float, float], sites: np.ndarray) -> None:
        sites = np.ascontiguousarray(sites, rawframe.SITE_DTYPE)
        self.lib.zkr_begin(self.handle, slot, map_w, map_h, y_range[0], y_range[1], sites.ctypes.data if len(sites) else None, len(sites))

    def frames(self, batch: zksim.Batch, games: np.ndarray) -> None:
        """Both sides' present frames of these games, read from the batch's arrays."""
        games = np.ascontiguousarray(games, np.int32)
        args = FramesArgs(n=len(games), games=_i32(games), records=batch.records.ctypes.data, cap=batch.cap, counts=_i32(batch.counts),
                          globs=batch.globs.ctypes.data_as(ct.POINTER(ct.c_float)), features=batch.features.ctypes.data,
                          feature_cap=batch.features.shape[2], feature_counts=_i32(batch.feature_counts))
        self.lib.zkr_frames(self.handle, ct.byref(args))

    def orders(self, games: np.ndarray, orders: np.ndarray, offsets: np.ndarray) -> None:
        """The orders given at the frames last taken: seat s's are orders[offsets[s]:offsets[s + 1]], over every seat."""
        games = np.ascontiguousarray(games, np.int32)
        orders = np.ascontiguousarray(orders, zksim.ORDER_DTYPE)
        offsets = np.ascontiguousarray(offsets, np.int32)
        assert len(offsets) == 2 * self.slots + 1
        self.lib.zkr_orders(self.handle, len(games), _i32(games), orders.ctypes.data if len(orders) else None, _i32(offsets))

    def finish(self, slot: int, path: pathlib.Path, meta: dict) -> None:
        """The slot's run to a file, written on the recorder's own thread: `wait` for it."""
        text = json.dumps(meta).encode()
        if self.lib.zkr_finish(self.handle, slot, str(path).encode(), text, len(text)):
            raise ValueError(f"zkrecord: slot {slot} has no run")

    def wait(self) -> None:
        """Until every file finished is written."""
        err = self.lib.zkr_wait(self.handle)
        if err:
            raise OSError(err, f"zkrecord: writing a file: {os.strerror(err)}")

    def frames_held(self, slot: int) -> int:
        return self.lib.zkr_frames_held(self.handle, slot)

    def clamps(self) -> dict[str, int]:
        out = np.zeros(len(CLAMP_NAMES), np.int64)
        self.lib.zkr_clamps(self.handle, out.ctypes.data_as(ct.POINTER(ct.c_int64)))
        return dict(zip(CLAMP_NAMES, map(int, out)))


# Reading -----------------------------------------------------------------------------------------------------------------------------------------

class _Cursor:
    def __init__(self, buf: bytes):
        self.buf, self.at = buf, 0

    def take(self, n: int) -> bytes:
        out = self.buf[self.at:self.at + n]
        if len(out) != n:
            raise ValueError("zkrecord: the file ends early")
        self.at += n
        return out

    def array(self, dtype, n: int) -> np.ndarray:
        dtype = np.dtype(dtype)
        return np.frombuffer(self.take(dtype.itemsize * n), dtype, n)

    def planes(self, dtype, n: int) -> np.ndarray:
        """n records of `dtype` stored a column at a time, each column a byte plane at a time."""
        dtype = np.dtype(dtype)
        if dtype.names is None:
            raw = np.frombuffer(self.take(dtype.itemsize * n), np.uint8).reshape(dtype.itemsize, n)
            return np.ascontiguousarray(raw.T).view(dtype).reshape(n)
        out = np.empty(n, dtype)
        for name in dtype.names:
            out[name] = self.planes(dtype[name], n)
        return out


def _unzstd(blob: bytes) -> bytes:
    L = lib()
    size = L.zkr_frame_size(blob, len(blob))
    if size < 0:
        raise ValueError("zkrecord: a chunk without its size")
    out = ct.create_string_buffer(max(int(size), 1))
    if L.zkr_decompress(blob, len(blob), out, size) != size:
        raise ValueError("zkrecord: a chunk doesn't decompress")
    return out.raw[:size]


def _chains(back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Records linked to the one `back` before them, as chains: an order putting each chain's records together in time, and for each
    record (in that order) where its chain starts in it."""
    idx = np.arange(len(back), dtype=np.int64)
    root = np.where(back > 0, idx - back, idx)
    while True:         # pointer jumping: log2 of the longest chain rounds
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    order = np.argsort(root, kind="stable")
    r = root[order]
    start = np.ones(len(r), bool)
    start[1:] = r[1:] != r[:-1]
    first = np.flatnonzero(start)
    return order, first, np.cumsum(start) - 1


def _undelta(res: np.ndarray, chains: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Each field the wrapping sum of the residuals along its chain."""
    order, first, segment = chains
    out = np.empty_like(res)
    for name in res.dtype.names:
        col = res[name]
        unsigned = col.view(np.dtype(f"u{col.dtype.itemsize}"))
        total = np.cumsum(unsigned[order].astype(np.uint64))
        before = np.zeros(len(first), np.uint64)
        before[1:] = total[first[1:] - 1]
        values = np.empty(len(col), np.uint64)
        values[order] = total - before[segment]
        out[name] = values.astype(unsigned.dtype).view(col.dtype)
    return out


def _fraction(q: np.ndarray, lo: float, span: float) -> np.ndarray:
    return (lo + q.astype(np.float64) * (span / 65535.0)).astype(np.float32)


def _wide_units(q: np.ndarray, map_w: float, map_h: float, y_lo: float, y_hi: float) -> np.ndarray:
    u = np.zeros(len(q), rawframe.UNIT_DTYPE)
    for name in ("uid", "ord_target", "def_id", "flags", "knowledge", "ord_kind", "player", "ally_team"):
        u[name] = q[name]
    u["x"], u["z"] = _fraction(q["x"], 0.0, map_w), _fraction(q["z"], 0.0, map_h)
    u["y"] = _fraction(q["y"], y_lo, y_hi - y_lo)
    u["ord_x"], u["ord_z"] = _fraction(q["ord_x"], -0.25 * map_w, 1.5 * map_w), _fraction(q["ord_z"], -0.25 * map_h, 1.5 * map_h)
    u["build_frac"] = _fraction(q["build_frac"], 0.0, 1.0)
    u["ord_age"] = q["ord_age"].astype(np.float32)
    for name in _HALF:
        u[name] = q[name].view(np.float16).astype(np.float32)
    return u


def _wide_features(q: np.ndarray, map_w: float, map_h: float) -> np.ndarray:
    f = np.zeros(len(q), rawframe.FEATURE_DTYPE)
    f["feat_id"], f["def_id"] = q["feat_id"], q["def_id"]
    f["x"], f["z"] = _fraction(q["x"], 0.0, map_w), _fraction(q["z"], 0.0, map_h)
    f["metal"] = q["metal"].view(np.float32)
    return f


def _split(a: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    return np.split(a, np.cumsum(counts)[:-1]) if len(counts) else []


def _packed(dtype) -> int:
    """Bytes a record of `dtype` takes in a file's planes, which have no padding."""
    dtype = np.dtype(dtype)
    return dtype.itemsize if dtype.names is None else sum(_packed(dtype[name]) for name in dtype.names)


def read(path: str | os.PathLike, side: int | None = None, until: int | None = None) -> tuple[dict, list[tuple[list[rawframe.Frame], list[np.ndarray]]]]:
    """A file's meta, and for side 0 and side 1 its frames and the orders given at each.  side: only that one's (the other's are
    empty); until: only the frames before it, at least (records are deltas along the game, so a frame needs all before it, and
    reading stops after the chunk that holds frame until - 1)."""
    c = _Cursor(pathlib.Path(path).read_bytes())
    if c.take(8) != MAGIC:
        raise ValueError(f"{path}: not a zkrecord file")
    meta = json.loads(c.take(int(c.array("<u4", 1)[0])))
    map_w, map_h, y_lo, y_hi = (float(v) for v in c.array("<f4", 4))
    frames, chunks, _, n_sites = (int(v) for v in c.array("<i4", 4))
    sites = np.frombuffer(_unzstd(c.take(int(c.array("<u4", 1)[0]))), rawframe.SITE_DTYPE, n_sites).copy()
    parts: list[dict[str, list]] = [{k: [] for k in ("units", "features", "orders", "globs", "unit_back", "feature_back", "unit_counts",
                                                     "feature_counts", "order_counts")} for _ in range(2)]
    read_frames = 0
    for _ in range(chunks):
        if until is not None and read_frames >= until:
            break
        k = _Cursor(_unzstd(c.take(int(c.array("<u4", 1)[0]))))
        n = int(k.array("<i4", 1)[0])
        read_frames += n
        for s, part in enumerate(parts):
            counts = [k.array("<i4", n) for _ in range(3)]
            units, feats, orders = (int(v.sum()) for v in counts)
            if side is not None and s != side:
                k.take(4 * GLOBALS * n + (4 + _packed(NARROW_UNIT)) * units + (4 + _packed(NARROW_FEATURE)) * feats
                       + _packed(zksim.ORDER_DTYPE) * orders)
                continue
            for name, v in zip(("unit_counts", "feature_counts", "order_counts"), counts):
                part[name].append(v)
            part["globs"].append(np.stack([k.planes("<u4", n) for _ in range(GLOBALS)], axis=1))
            part["unit_back"].append(k.planes("<i4", units))
            part["units"].append(k.planes(NARROW_UNIT, units))
            part["feature_back"].append(k.planes("<i4", feats))
            part["features"].append(k.planes(NARROW_FEATURE, feats))
            part["orders"].append(k.planes(zksim.ORDER_DTYPE, orders))
    out = []
    for part in parts:
        cat = {name: np.concatenate(v) if v else None for name, v in part.items()}
        if cat["unit_counts"] is None:
            out.append(([], []))
            continue
        globs = (np.cumsum(cat["globs"].astype(np.uint64), axis=0).astype(np.uint32)).view(np.float32)
        units = _wide_units(_undelta(cat["units"], _chains(cat["unit_back"])), map_w, map_h, y_lo, y_hi)
        feats = _wide_features(_undelta(cat["features"], _chains(cat["feature_back"])), map_w, map_h)
        assert len(cat["unit_counts"]) == (frames if until is None else min(frames, read_frames))
        fr = [rawframe.Frame(u, sites, g, f)
              for u, g, f in zip(_split(units, cat["unit_counts"]), globs, _split(feats, cat["feature_counts"]))]
        out.append((fr, _split(cat["orders"], cat["order_counts"])))
    return meta, out


def frame_count(path: str | os.PathLike) -> int:
    """Frames of each side, from the header alone."""
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a zkrecord file")
        f.seek(int(np.frombuffer(f.read(4), "<u4")[0]) + 16, os.SEEK_CUR)
        return int(np.frombuffer(f.read(4), "<i4")[0])
