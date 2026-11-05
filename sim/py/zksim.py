"""ctypes binding to the simulator (sim/include/zksim/capi.h).

Build it in WSL:  cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j 16
ZKSIM_LIB overrides where the library is looked for.
"""
from __future__ import annotations

import ctypes as ct
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "py"))
import zknav  # noqa: E402
ABI_VERSION = 1
MAX_MOUNTS = 8
TICK_FRAMES = 5
FRAMES_PER_SECOND = 30
MOVE_CLASSES = ("kbot", "tank", "spider", "amph", "hover", "ship")
ADVANCE, HOLD, ORDERS = 0, 1, 2
ORDER_NONE, ORDER_MOVE, ORDER_FIGHT, ORDER_ATTACK = 0, 1, 2, 3
MOBILE, AIR, ARMED, FIREPROOF = 1, 2, 4, 8
W_TURRET = 2


class WeaponRec(ct.Structure):
    _fields_ = [("damage", ct.c_float * 5)] + [(n, ct.c_float) for n in (
        "range", "reload", "salvo", "salvo_delay", "projectiles", "speed", "accel", "start_speed",
        "blast_radius", "edge", "height_mod", "spray", "accuracy", "moving_accuracy",
        "para_seconds", "extra_para", "slow_factor", "shield_power", "shield_regen", "shield_radius",
        "vel_threshold", "vel_base", "vel_scale", "alpha_damage",
        "area_dps", "area_radius", "area_plateau", "area_range_fall", "area_plateau_fall", "area_time_fall", "area_frames",
        "burn_frames", "burn_random", "burn_chance", "burn_damage")] + [(n, ct.c_int32) for n in (
            "kind", "flags", "shield_intercept", "intercepted_by")]


class TypeRec(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        "cost", "hp", "speed", "accel", "brake", "turn_rate", "slow_turn_speed",
        "footprint_radius", "hit_radius", "armored_mult", "auto_heal", "max_range")] + [(n, ct.c_int32) for n in (
            "armor_class", "move_class", "category", "flags", "explode_weapon", "n_mounts")] + [
        ("mount_weapon", ct.c_int32 * MAX_MOUNTS), ("mount_only", ct.c_int32 * MAX_MOUNTS),
        ("mount_bad", ct.c_int32 * MAX_MOUNTS)]


class UnitState(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        "x", "z", "vx", "vz", "heading", "speed", "hp", "damage_taken", "para", "shield")] + [
        (n, ct.c_int32) for n in ("alive", "order", "target")]


def _load_library() -> ct.CDLL:
    path = os.environ.get("ZKSIM_LIB") or os.path.expanduser("~/zksim/build/libzksim.so")
    lib = ct.CDLL(path)
    vp, cvp = ct.c_void_p, ct.c_void_p
    fl, it = ct.c_float, ct.c_int
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int)
    lib.zk_catalog_new.restype = vp
    lib.zk_catalog_new.argtypes = [ct.POINTER(TypeRec), it, ct.POINTER(WeaponRec), it, fp]
    lib.zk_catalog_free.argtypes = [vp]
    lib.zk_map_new.restype = vp
    lib.zk_map_new.argtypes = [it, it, fl, fp, fp, it, fp, fp, fp, fp, ct.POINTER(ct.c_short)]
    lib.zk_map_set_routes.restype = None
    lib.zk_map_set_routes.argtypes = [vp, it, it, fp, it, ct.POINTER(ct.c_int), fp, ct.POINTER(ct.c_int)]
    lib.zk_map_free.argtypes = [vp]
    lib.zk_world_new.restype = vp
    lib.zk_world_new.argtypes = [cvp, cvp, ct.c_uint64, it]
    lib.zk_world_free.argtypes = [vp]
    lib.zk_world_add.restype = it
    lib.zk_world_add.argtypes = [vp, it, it, fl, fl, fl, fl, fl]
    lib.zk_world_order.argtypes = [vp, it, it, fl, fl, it]
    lib.zk_world_step.argtypes = [vp, it]
    lib.zk_world_frame.argtypes = [cvp]
    lib.zk_world_set_frame.argtypes = [vp, it]
    lib.zk_world_get.argtypes = [cvp, it, ct.POINTER(UnitState)]
    lib.zk_world_ready_weapons.argtypes = [vp]
    lib.zk_world_kill.argtypes = [vp, it]
    lib.zk_replay.argtypes = [cvp, ct.POINTER(vp), ip, it, ip, ip, ip, fp, fp, fp, fp, ct.POINTER(ct.c_uint8), it,
                              ct.c_uint64, ip, it, it, it, fp]
    if lib.zk_abi_version() != ABI_VERSION:
        raise RuntimeError("libzksim has another ABI version: rebuild it")
    for got, want in ((lib.zk_sizeof_weapon(), WeaponRec), (lib.zk_sizeof_type(), TypeRec), (lib.zk_sizeof_state(), UnitState)):
        if got != ct.sizeof(want):
            raise RuntimeError(f"{want.__name__}: {ct.sizeof(want)} bytes here, {got} in libzksim: rebuild it")
    return lib


_lib = _load_library()


def _fill(rec: ct.Structure, values: dict) -> None:
    for name, kind in rec._fields_:
        if name in values and not name.startswith("mount_"):
            value = values[name]
            setattr(rec, name, (kind)(*value) if isinstance(value, list) else value)


def _floats(a: np.ndarray):
    return a.ctypes.data_as(ct.POINTER(ct.c_float))


def _ints(a: np.ndarray):
    return a.ctypes.data_as(ct.POINTER(ct.c_int))


class Catalog:
    """Unit and weapon records from sim/data, indexed by name."""

    def __init__(self, directory: pathlib.Path | str = DATA):
        directory = pathlib.Path(directory)
        data = json.loads((directory / "catalog.json").read_text("utf8"))
        self.units = data["units"]
        self.weapons = data["weapons"]
        self._install(np.load(directory / "priority.npz")["priority"].astype(np.float32))

    @classmethod
    def from_records(cls, units: list[dict], weapons: list[dict]) -> "Catalog":
        """A small catalog for tests: every weapon rates every unit equally."""
        self = cls.__new__(cls)
        self.units, self.weapons = units, weapons
        self._install(np.zeros((len(weapons), len(units)), np.float32))
        return self

    def _install(self, priority: np.ndarray) -> None:
        self.index = {u["name"]: i for i, u in enumerate(self.units)}
        self.weapon_index = {w["name"]: i for i, w in enumerate(self.weapons)}
        types = (TypeRec * len(self.units))()
        for rec, u in zip(types, self.units):
            _fill(rec, u)
            rec.n_mounts = len(u["mount_weapon"])
            for i in range(rec.n_mounts):
                rec.mount_weapon[i] = u["mount_weapon"][i]
                rec.mount_only[i] = u["mount_only"][i]
                rec.mount_bad[i] = u["mount_bad"][i]
        weapons = (WeaponRec * len(self.weapons))()
        for rec, w in zip(weapons, self.weapons):
            _fill(rec, w)
        self.handle = _lib.zk_catalog_new(types, len(self.units), weapons, len(self.weapons), _floats(priority))

    def weapon_of(self, unit: str, mount: int = 0) -> dict:
        return self.weapons[self.units[self.index[unit]]["mount_weapon"][mount]]


ROUTE_ARRAYS = ("centres", "edge_to", "gates", "next_edge")


def load_routes(data) -> list[dict]:
    """The per-class routes of a map file."""
    return [{k: data[f"class{c}_{k}"] for k in ROUTE_ARRAYS} for c in range(data["speed"].shape[0])]


class Map:
    def __init__(self, height: np.ndarray, speed: np.ndarray, cell: float = 16.0, name: str = "",
                 slope: np.ndarray | None = None, slope_mod: np.ndarray | None = None,
                 segments: np.ndarray | None = None, routes: list[dict] | None = None):
        """segments: per class and cell.  routes: per class, from zknav and in cells.  If routes are missing they're
        worked out here, splitting the segments first; without segments, the whole map starts as one for each class."""
        height = np.asarray(height, np.float32)
        speed = np.asarray(speed, np.float32)
        self.name, self.cell, self.height, self.speed = name, cell, height, speed
        rows, cols = height.shape
        # The downhill direction, of unit length like the engine's GetCenterNormals2D: only the
        # slope map says how steep it is.
        dz, dx = np.gradient(height.astype(np.float64), cell)
        flat = np.maximum(np.sqrt(dx * dx + dz * dz), 1e-9)
        normal_x = (-dx / flat).astype(np.float32)
        normal_z = (-dz / flat).astype(np.float32)
        self.slope = np.asarray(1.0 - 1.0 / np.sqrt(dx * dx + dz * dz + 1.0) if slope is None else slope, np.float32)
        self.slope_mod = np.zeros(speed.shape[0], np.float32) if slope_mod is None else np.asarray(slope_mod, np.float32)
        self.segments = np.zeros(speed.shape, np.int16) if segments is None else np.asarray(segments, np.int16)
        if routes is None:
            built = [zknav.navigation(s, v) for s, v in zip(self.segments, speed)]
            self.segments = np.stack([labels for labels, _ in built])
            routes = [r for _, r in built]
        self.routes = routes
        self.handle = _lib.zk_map_new(cols, rows, cell, _floats(height), _floats(speed), speed.shape[0],
                                      _floats(self.slope), _floats(normal_x), _floats(normal_z), _floats(self.slope_mod),
                                      self.segments.ctypes.data_as(ct.POINTER(ct.c_short)))
        for c, r in enumerate(routes):
            centres = np.asarray(r["centres"], np.float32) * cell
            gates = np.asarray(r["gates"], np.float32) * cell
            edge_to = np.asarray(r["edge_to"], np.int32)
            next_edge = np.asarray(r["next_edge"], np.int32)
            _lib.zk_map_set_routes(self.handle, c, len(centres), _floats(centres), len(edge_to), edge_to.ctypes.data_as(ct.POINTER(ct.c_int)),
                                   _floats(gates), next_edge.ctypes.data_as(ct.POINTER(ct.c_int)))

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Map":
        d = np.load(path)
        return cls(d["height"], d["speed"], float(d["cell"]), str(d["name"]), d["slope"], d["slope_mod"], d["segments"], load_routes(d))

    @classmethod
    def flat(cls, width: float = 4096.0, depth: float = 4096.0, cell: float = 16.0) -> "Map":
        rows, cols = int(depth / cell), int(width / cell)
        return cls(np.zeros((rows, cols), np.float32), np.ones((len(MOVE_CLASSES), rows, cols), np.float32), cell, "flat")


class World:
    def __init__(self, catalog: Catalog, world_map: Map, seed: int = 1, behaviour: int = ORDERS):
        self.catalog, self.map = catalog, world_map
        self.handle = _lib.zk_world_new(catalog.handle, world_map.handle, seed, behaviour)

    def __del__(self):
        if getattr(self, "handle", None):
            _lib.zk_world_free(self.handle)
            self.handle = None

    def add(self, name: str, side: int, x: float, z: float, heading: float = 0.0, hp: float = 1.0, build: float = 1.0) -> int:
        return _lib.zk_world_add(self.handle, self.catalog.index[name], side, x, z, heading, hp, build)

    def move(self, unit: int, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_MOVE, x, z, -1)

    def fight(self, unit: int, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_FIGHT, x, z, -1)

    def kill(self, unit: int) -> None:
        _lib.zk_world_kill(self.handle, unit)

    def attack(self, unit: int, target: int) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_ATTACK, 0.0, 0.0, target)

    def ready_weapons(self) -> None:
        _lib.zk_world_ready_weapons(self.handle)

    def step(self, ticks: int = 1) -> None:
        _lib.zk_world_step(self.handle, ticks)

    @property
    def frame(self) -> int:
        return _lib.zk_world_frame(self.handle)

    def set_frame(self, frame: int) -> None:
        _lib.zk_world_set_frame(self.handle, frame)

    def get(self, unit: int) -> UnitState:
        state = UnitState()
        _lib.zk_world_get(self.handle, unit, ct.byref(state))
        return state


def replay(catalog: Catalog, maps: list[Map], snapshot_map: np.ndarray, offsets: np.ndarray, type_: np.ndarray,
           side: np.ndarray, x: np.ndarray, z: np.ndarray, hp: np.ndarray, build: np.ndarray, tracked: np.ndarray,
           seeds: int, seed: int, horizons: list[int], behaviour: int, threads: int) -> np.ndarray:
    """Losses per snapshot, seed, horizon and side; horizons are in ticks."""
    n = len(snapshot_map)
    i32 = lambda a: np.asarray(a, np.int32)  # noqa: E731
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    snapshot_map, offsets, type_, side = i32(snapshot_map), i32(offsets), i32(type_), i32(side)
    x, z, hp, build = f32(x), f32(z), f32(hp), f32(build)
    tracked = np.asarray(tracked, np.uint8)
    horizon_array = i32(horizons)
    out = np.zeros((n, seeds, len(horizons), 2), np.float32)
    handles = (ct.c_void_p * len(maps))(*[m.handle for m in maps])
    _lib.zk_replay(catalog.handle, handles, _ints(snapshot_map), n, _ints(offsets), _ints(type_), _ints(side),
                   _floats(x), _floats(z), _floats(hp), _floats(build),
                   tracked.ctypes.data_as(ct.POINTER(ct.c_uint8)), seeds, seed, _ints(horizon_array),
                   len(horizons), behaviour, threads, _floats(out))
    return out
