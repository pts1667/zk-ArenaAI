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
sys.path.insert(0, str(ROOT.parent / "bots"))
import zknav  # noqa: E402
from zkbot.policy import rawframe  # noqa: E402
ABI_VERSION = 26
WARD_TARGETS = 4
MAX_MOUNTS = 8
MAX_TEAMS = 8                       # players in a game, and so allyteams too
TICK_FRAMES = 5
FRAMES_PER_SECOND = 30
GRAVITY = 120.0 / 900.0       # elmos a frame squared (world.hpp)
MOVE_CLASSES = ("kbot", "tank", "spider", "amph", "hover", "ship")
ADVANCE, HOLD, ORDERS, SCRIPTED = 0, 1, 2, 3
ORDER_NONE, ORDER_MOVE, ORDER_FIGHT, ORDER_ATTACK, ORDER_MANUAL_FIRE, ORDER_JUMP, ORDER_CLOAK_ON, ORDER_CLOAK_OFF = 0, 1, 2, 3, 6, 7, 8, 9
ORDER_BUILD, ORDER_ASSIST, ORDER_RECLAIM, ORDER_PRODUCE, ORDER_MOVE_STATE, ORDER_FIRE_STATE, ORDER_UNIT_AI, ORDER_SELF_DESTRUCT = 10, 11, 13, 14, 15, 16, 17, 18
ORDER_PLACE_BEACON, ORDER_WAIT_AT_BEACON, ORDER_ON_OFF = 19, 20, 21
ORDER_LOAD, ORDER_UNLOAD, ORDER_DROP = 22, 23, 24        # an air transport's (sim/src/transport.cpp)
MOBILE, AIR, ARMED, FIREPROOF, FACTORY, PLOPPABLE, COMMANDER, CAN_CLOAK, STARTS_CLOAKED, FIRES_CLOAKED, AREA_CLOAK_ON, LEAVES_GHOST, KAMIKAZE_BLIND, FLOATS = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
MISSILE, NEUTRAL, PLANE, LANDS, STRAFES, DONT_COUNT, VITAL = 16384, 32768, 65536, 131072, 262144, 524288, 1048576
FEATURES_PER_SIDE = 512             # records a frame can hold, as many as featurize keeps; past it, those with the most metal left (sim/src/views.cpp)
# Chicken defense (sim/src/chickens.cpp): the difficulties, the roles of the gadget's kinds, and what chicken_state gives.
CHICKEN_DIFFICULTIES = ("Beginner", "Very Easy", "Easy", "Normal", "Hard", "Suicidal")
CHICKEN_OFFENSE, CHICKEN_DEFENSE, CHICKEN_SUPPORT, CHICKEN_SPECIAL = 0, 1, 2, 3
CHICKEN_KINDS, CHICKEN_MINI_QUEENS, CHICKEN_NO_TARGET = 32, 4, 4
CHICKEN_STATE = ("queen", "until_queen", "burrows", "burrows_killed", "chickens", "waves", "aggro", "tech", "queen_time", "queen_hp",
                 "difficulty", "team")
W_TURRET, W_NO_AUTO_TARGET, W_MANUAL, W_UNMODELLED = 2, 16, 512, 2048
W_PIERCES, W_ONE_HIT_EACH, W_THROWS, W_BOMB, W_DROPPED = 524288, 1048576, 2097152, 4194304, 8388608


class WeaponRec(ct.Structure):
    _fields_ = [("damage", ct.c_float * 5)] + [(n, ct.c_float) for n in (
        "range", "reload", "salvo", "salvo_delay", "projectiles", "speed", "accel", "start_speed",
        "blast_radius", "edge", "height_mod", "spray", "accuracy", "moving_accuracy",
        "para_seconds", "extra_para", "slow_factor", "disarm_mult", "disarm_penalty", "disarm_time", "overstun_seconds",
        "overslow", "shield_cost", "dyn_exp", "dyn_range", "climb_frames", "shield_power", "shield_regen", "shield_radius",
        "shield_energy", "arc",
        "vel_threshold", "vel_base", "vel_scale", "alpha_damage",
        "area_dps", "area_radius", "area_plateau", "area_range_fall", "area_plateau_fall", "area_time_fall", "area_frames",
        "burn_frames", "burn_random", "burn_chance", "burn_damage",
        "okp_damage", "okp_frames", "okp_full_frames", "okp_hit_mod", "okp_min_range", "okp_fast_mult", "okp_radar_mult",
        "radar_wobble_penalty", "radar_dot_penalty", "impulse", "gravity",
        "lead_limit", "predict_boost", "move_error", "flight_frames", "turn_rate", "rise")] + [(n, ct.c_int32) for n in (
            "kind", "flags", "shield_intercept", "intercepted_by", "accurate_leading")]


class TypeRec(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        "cost", "hp", "speed", "accel", "brake", "turn_rate", "slow_turn_speed",
        "footprint_radius", "hit_radius", "armored_mult", "max_range", "idle_regen", "idle_time",
        "los_radius", "air_los_radius", "radar_radius", "los_height", "stockpile_frames", "reload_move_penalty",
        "metal_cost", "energy_cost", "build_time", "build_speed", "build_range", "metal_make", "energy_make", "extract_mult", "pylon_range",
        "reclaim_speed", "wreck_metal", "wreck_hp", "heap_metal", "heap_hp", "energy_upkeep", "stockpile_cost", "close_frames",
        "cloak_cost", "cloak_cost_moving", "decloak_distance", "bestowed_decloak", "area_cloak_radius", "area_cloak_upkeep",
        "perceived_range", "heat_per_shot", "heat_decay", "heat_max_slow", "heat_initial", "bait_cost", "kamikaze_distance", "detonate_reach", "drop_height", "drop_gravity", "jump_range", "jump_speed", "jump_height", "jump_reload", "jump_delay", "burrow_speed", "mass", "push_scale", "needed_link", "throw_radius", "teleport_rate", "beacon_frames",
        "cruise_height", "vertical_speed", "turn_speed", "strafe_radius", "strafe_speed", "reammo_frames", "reammo_energy", "pad_bp",
        "lift_light", "lift_medium", "lift_heavy", "lift_size", "load_size", "load_height")] + [(n, ct.c_int32) for n in (
            "move_state", "fire_state", "no_chase", "bait_level", "bait_target", "bait_target_armor",
            "def_id", "wreck_feature", "heap_feature", "armor_class", "move_class", "category", "flags", "explode_weapon", "n_mounts", "landing_weapon", "drop_weapon", "missile_pads", "beacon_type", "ammo", "pad_count", "plate_parent", "load_class")] + [
        ("mount_weapon", ct.c_int32 * MAX_MOUNTS), ("mount_only", ct.c_int32 * MAX_MOUNTS),
        ("mount_bad", ct.c_int32 * MAX_MOUNTS)]


class TacticsRec(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        "search_range", "idle_push_dist_sq", "idle_chase_leeway", "idle_commit_dist", "idle_commit_mult", "idle_enemy_mult",
        "flee_order_dis", "flee_leeway", "flee_distance", "min_flee_range", "flee_vel_prediction",
        "weapon_range", "stopping_distance", "velocity_prediction", "skirm_order_dis", "skirm_order_dis_min", "hug_range", "skirm_leeway",
        "skirm_range_override", "reload_skirm_leeway", "skirm_blocked_frames", "vel_pred_chase", "skirm_keep_leeway", "skirm_only_near", "my_speed",
        "max_swarm_range", "min_swarm_range", "swarm_leeway", "jink_tangent", "jink_parallel", "jink_away_parallel", "strafe_order_length",
        "min_circle_strafe", "jink_period", "skirm_jink", "skirm_heading_block", "ward_leeway", "ward_enable_leeway", "ward_predict",
        "ward_shield", "ward_height", "ward_speed_buffer")] + [
        ("ward_range", ct.c_float * WARD_TARGETS), ("ward_target", ct.c_int32 * WARD_TARGETS), ("waterline", ct.c_float)] + [
        (n, ct.c_int32) for n in ("row", "sea", "fight_only", "flags", "weapon_num", "blocked_on_fight")]


class UnitState(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        "x", "z", "vx", "vz", "heading", "speed", "hp", "damage_taken", "para", "shield", "disarm", "build", "heat", "jump_charge", "altitude")] + [
        (n, ct.c_int32) for n in ("alive", "order", "target", "stock", "cloaked", "chasing", "dodging", "shots_fired", "jumping", "skidding", "low_power", "deployed", "beacon", "pulling", "ammo", "air_state", "carrier", "cargo")]


class ChickenKind(ct.Structure):
    _fields_ = [("type", ct.c_int32), ("role", ct.c_int32)] + [(n, ct.c_float) for n in (
        "time", "obsolete", "squad_size", "cost", "max_aggro", "count", "burrow_ratio", "min_dist", "max_dist", "cooldown")] + [
        (n, ct.c_int32) for n in ("target_human", "tie_to_burrow")]


class ChickenRules(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in ("level", "burrow", "queen", "queen_morph", "mini_queen", "test_building", "test_building_queen")] + [
        ("no_target", ct.c_int32 * CHICKEN_NO_TARGET)] + [(n, ct.c_int32) for n in ("max_tries", "max_tries_small", "n_mini_queen_times", "n_kinds")] + [
        (n, ct.c_float) for n in (
            "spawn_square", "spawn_square_increment", "max_burrows", "min_base_distance", "max_base_distance", "max_age", "egg_decay_time", "burrow_eggs",
            "player_malus", "burrow_spawn_rate", "chicken_spawn_rate", "wave_ratio", "base_wave_size", "wave_size_mult", "queen_spawn_mult",
            "defense_per_wave", "defense_per_burrow_kill", "grace_period", "grace_penalty", "grace_period_min", "ramp_up_time",
            "queen_time", "queen_health_mod")] + [("queen_morph_frames", ct.c_float * 2), ("mini_queen_time", ct.c_float * CHICKEN_MINI_QUEENS)] + [
        (n, ct.c_float) for n in (
            "burrow_queen_time", "burrow_wave_size", "burrow_respawn_chance", "burrow_regress_time",
            "human_aggro_per_burrow", "human_aggro_decay", "human_aggro_min", "human_aggro_max", "human_aggro_wave_factor", "human_aggro_wave_max",
            "human_aggro_defense_factor", "human_aggro_tech_time_progress", "human_aggro_tech_time_regress",
            "human_aggro_queen_time_factor", "human_aggro_queen_time_min", "human_aggro_queen_time_max",
            "tech_accel_per_player", "tech_time_floor_factor", "tech_time_max", "time_spawn_bonus")] + [
        ("kinds", ChickenKind * CHICKEN_KINDS)]


class EggRec(ct.Structure):
    _fields_ = [("feature", ct.c_int32), ("metal", ct.c_float), ("hp", ct.c_float), ("radius", ct.c_float)]


def _load_library() -> ct.CDLL:
    path = os.environ.get("ZKSIM_LIB") or os.path.expanduser("~/zksim/build/libzksim.so")
    lib = ct.CDLL(path)
    vp, cvp = ct.c_void_p, ct.c_void_p
    fl, it = ct.c_float, ct.c_int
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int)
    lib.zk_catalog_new.restype = vp
    lib.zk_catalog_new.argtypes = [ct.POINTER(TypeRec), it, ct.POINTER(WeaponRec), it, fp, ct.POINTER(TacticsRec), it, ct.POINTER(ct.c_uint8)]
    lib.zk_catalog_free.argtypes = [vp]
    lib.zk_catalog_reveal.argtypes = [vp, fl, it]
    lib.zk_catalog_chickens.argtypes = [vp, it, ct.POINTER(ChickenRules)]
    lib.zk_catalog_eggs.argtypes = [vp, ct.POINTER(EggRec)]
    lib.zk_map_new.restype = vp
    lib.zk_map_new.argtypes = [it, it, fl, fp, fp, it, fp, fp, fp, fp, ct.POINTER(ct.c_short)]
    lib.zk_map_set_routes.restype = None
    lib.zk_map_set_routes.argtypes = [vp, it, it, fp, it, ct.POINTER(ct.c_int), fp, ct.POINTER(ct.c_int)]
    lib.zk_map_free.argtypes = [vp]
    lib.zk_world_new.restype = vp
    lib.zk_world_new.argtypes = [cvp, cvp, ct.c_uint64, it]
    lib.zk_world_free.argtypes = [vp]
    lib.zk_world_set_teams.argtypes = [vp, it, ip]
    lib.zk_world_add.restype = it
    lib.zk_world_add.argtypes = [vp, it, it, fl, fl, fl, fl, fl]
    lib.zk_world_order.argtypes = [vp, it, it, fl, fl, it]
    lib.zk_world_step.argtypes = [vp, it]
    lib.zk_world_frame.argtypes = [cvp]
    lib.zk_world_count.argtypes = [cvp]
    lib.zk_world_set_frame.argtypes = [vp, it]
    lib.zk_world_get.argtypes = [cvp, it, ct.POINTER(UnitState)]
    lib.zk_world_ready_weapons.argtypes = [vp]
    lib.zk_world_kill.argtypes = [vp, it]
    lib.zk_world_set_hp.argtypes = [vp, it, fl]
    lib.zk_world_set_shield.argtypes = [vp, it, fl]
    lib.zk_world_economy.argtypes = [vp, it, fp, fl, fl, fl, fl, fl]
    lib.zk_world_resources.argtypes = [cvp, it, fp]
    lib.zk_world_add_feature.restype = it
    lib.zk_world_add_feature.argtypes = [vp, fl, fl, fl, fl, fl, it]
    lib.zk_world_features.restype = it
    lib.zk_world_features.argtypes = [vp, it, vp, it]
    lib.zk_world_view.restype = it
    lib.zk_world_view.argtypes = [vp, it, vp, it, fp, it, fp]
    lib.zk_world_set_reveal.argtypes = [vp, it]
    lib.zk_world_chickens.argtypes = [vp, it, it]
    lib.zk_world_chicken_state.argtypes = [cvp, fp]
    lib.zk_batch_set_reveal.argtypes = [vp, it]
    lib.zk_batch_chickens.argtypes = [vp, it, it, it]
    lib.zk_batch_chicken_state.argtypes = [cvp, it, fp]
    lib.zk_batch_new.restype = vp
    lib.zk_batch_new.argtypes = [cvp, it, it, it]
    lib.zk_batch_free.argtypes = [vp]
    lib.zk_batch_reset.argtypes = [vp, it, cvp, ct.c_uint64, it, it, ip, it, ip, ip, fp, fp, fp, fp, fp, ip, fp]
    lib.zk_batch_world.restype = vp
    lib.zk_batch_world.argtypes = [vp, it]
    lib.zk_batch_copy_game.argtypes = [vp, it, it]
    lib.zk_batch_step.argtypes = [vp, ip, it, vp, ip, vp, ip, it, vp, it, ip, fp, fp, vp, it, ip]
    lib.zk_batch_step_begin.argtypes = [vp, ip, it, vp, ip, vp, ip, it, vp, it, ip, fp, fp, vp, it, ip]
    lib.zk_batch_step_wait.argtypes = [vp]
    lib.zk_batch_set_roles.argtypes = [vp, ip, it]
    lib.zk_world_set_roles.argtypes = [vp, ip, it]
    lib.zk_world_order_group.argtypes = [vp, vp]
    lib.zk_world_order_group.restype = it
    lib.zk_batch_set_threads.argtypes = [vp, it]
    lib.zk_replay.argtypes = [cvp, ct.POINTER(vp), ip, it, ip, ip, ip, fp, fp, fp, fp, ct.POINTER(ct.c_uint8), it,
                              ct.c_uint64, ip, it, it, it, fp]
    if lib.zk_abi_version() != ABI_VERSION:
        raise RuntimeError("libzksim has another ABI version: rebuild it")
    if (lib.zk_sizeof_unit_record() != rawframe.UNIT_DTYPE.itemsize or lib.zk_sizeof_feature_record() != rawframe.FEATURE_DTYPE.itemsize
            or lib.zk_global_count() != len(rawframe.GLOBAL_FIELDS)):
        raise RuntimeError("libzksim was built from another raw frame schema: rebuild it")
    for got, want in ((lib.zk_sizeof_weapon(), WeaponRec), (lib.zk_sizeof_type(), TypeRec), (lib.zk_sizeof_state(), UnitState),
                      (lib.zk_sizeof_chicken_rules(), ChickenRules)):
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
    assert a.flags.c_contiguous, "a column of a record array has gaps in it: C would read past each value"
    return a.ctypes.data_as(ct.POINTER(ct.c_float))


def _ints(a: np.ndarray):
    assert a.flags.c_contiguous, "a column of a record array has gaps in it: C would read past each value"
    return a.ctypes.data_as(ct.POINTER(ct.c_int))


class Catalog:
    """Unit and weapon records from sim/data, indexed by name, with the game's rules (rules.json, tools/sim_rules.py) if there."""

    def __init__(self, directory: pathlib.Path | str = DATA):
        directory = pathlib.Path(directory)
        data = json.loads((directory / "catalog.json").read_text("utf8"))
        self.units = data["units"]
        self.weapons = data["weapons"]
        arrays = np.load(directory / "priority.npz")
        self.tactics, self.relations = data["tactics"], arrays["relations"]
        rules = directory / "rules.json"
        self.rules = json.loads(rules.read_text("utf8")) if rules.exists() else None
        self._install(arrays["priority"].astype(np.float32))

    @classmethod
    def from_records(cls, units: list[dict], weapons: list[dict], rules: dict | None = None) -> "Catalog":
        """A small catalog for tests: every weapon rates every unit equally.  rules: as rules.json, for what its units have."""
        self = cls.__new__(cls)
        self.units, self.weapons, self.rules = units, weapons, rules
        self.tactics, self.relations = [{} for _ in units], np.zeros((len(units), len(units)), np.uint8)
        self._install(np.zeros((len(weapons), len(units)), np.float32))
        return self

    def _install(self, priority: np.ndarray) -> None:
        self.index = {u["name"]: i for i, u in enumerate(self.units)}
        self.weapon_index = {w["name"]: i for i, w in enumerate(self.weapons)}
        dont_count = set((self.rules or {}).get("dont_count", ()))
        vital = set((self.rules or {}).get("vital", ()))
        types = (TypeRec * len(self.units))()
        for rec, u in zip(types, self.units):
            rec.move_state, rec.fire_state = 1, 2      # the engine's defaults, for records that don't say
            rec.plate_parent = -1
            _fill(rec, u)
            if u["name"] in dont_count:
                rec.flags |= DONT_COUNT
            if u["name"] in vital:
                rec.flags |= VITAL
            rec.n_mounts = len(u["mount_weapon"])
            for i in range(rec.n_mounts):
                rec.mount_weapon[i] = u["mount_weapon"][i]
                rec.mount_only[i] = u["mount_only"][i]
                rec.mount_bad[i] = u["mount_bad"][i]
        weapons = (WeaponRec * len(self.weapons))()
        for rec, w in zip(weapons, self.weapons):
            rec.lead_limit, rec.predict_boost = -1.0, 1.0      # the engine's defaults again: no limit to its lead, and no error in it
            _fill(rec, w)
        # A type's own table first, then the other tables some have: for the sea, and against what they fight only when told to.
        tactics = (TacticsRec * len(self.tactics))()
        for row, (rec, t) in enumerate(zip(tactics, self.tactics)):
            rec.row, rec.sea, rec.fight_only, rec.skirm_heading_block, rec.ward_shield = row, -1, -1, 2.0, -1.0
            rec.ward_target = (ct.c_int32 * WARD_TARGETS)(*[-1] * WARD_TARGETS)
            _fill(rec, t)
        relations = np.require(self.relations, np.uint8, "C")
        assert relations.shape == (len(self.tactics), len(self.units)), relations.shape
        self.handle = _lib.zk_catalog_new(types, len(self.units), weapons, len(self.weapons), _floats(priority), tactics, len(self.tactics),
                                          relations.ctypes.data_as(ct.POINTER(ct.c_uint8)))
        if self.rules:
            self._install_rules(self.rules)

    def _install_rules(self, rules: dict) -> None:
        if "reveal" in rules:
            _lib.zk_catalog_reveal(self.handle, rules["reveal"]["value_ratio"], rules["reveal"]["every_frames"])
        chickens = rules.get("chickens")
        if not chickens or chickens["burrow"] not in self.index:
            return
        for difficulty, settings in enumerate(chickens["difficulties"]):
            _lib.zk_catalog_chickens(self.handle, difficulty, ct.byref(self._chicken_rules(chickens, settings)))
        eggs = (EggRec * len(self.units))()
        for rec, u in zip(eggs, self.units):
            egg = chickens["eggs"].get(u["name"])
            rec.feature = -1 if egg is None else egg["feature"]
            if egg is not None:
                rec.metal, rec.hp, rec.radius = egg["metal"], egg["hp"], egg["radius"]
        _lib.zk_catalog_eggs(self.handle, eggs)

    def _chicken_rules(self, chickens: dict, settings: dict) -> ChickenRules:
        """One difficulty's settings as capi.h's zk_chicken_rules: names to types, and -1 for what the gadget leaves to a fallback."""
        def type_of(name):
            return -1 if name is None else self.index[name]

        def given(value):
            return -1.0 if value is None else value

        r = ChickenRules()
        for name, kind in ChickenRules._fields_:
            if kind is ct.c_float and name in settings:
                setattr(r, name, settings[name])
        r.level = settings["level"]
        r.burrow, r.queen, r.queen_morph, r.mini_queen = (type_of(n) for n in (chickens["burrow"], settings["queen"], settings["queen_morph"], settings["mini_queen"]))
        r.test_building, r.test_building_queen = type_of(chickens["test_building"]), type_of(chickens["test_building_queen"])
        no_target = [self.index[n] for n in chickens["no_target"] if n in self.index]
        assert len(no_target) <= CHICKEN_NO_TARGET, "more types chickens aren't sent at than capi.h has room for"
        r.no_target[:] = no_target + [-1] * (CHICKEN_NO_TARGET - len(no_target))
        r.max_tries, r.max_tries_small = chickens["max_tries"], chickens["max_tries_small"]
        assert len(settings["mini_queen_time"]) <= CHICKEN_MINI_QUEENS
        r.n_mini_queen_times = len(settings["mini_queen_time"])
        for i, share in enumerate(settings["mini_queen_time"]):
            r.mini_queen_time[i] = share
        r.queen_morph_frames[:] = settings["queen_morph_frames"]
        kinds = [(role, k) for role, key in ((CHICKEN_OFFENSE, "offense"), (CHICKEN_DEFENSE, "defense"), (CHICKEN_SUPPORT, "support"),
                                             (CHICKEN_SPECIAL, "special")) for k in settings[key]]
        assert len(kinds) <= CHICKEN_KINDS, "more kinds of chicken than capi.h has room for"
        r.n_kinds = len(kinds)
        for rec, (role, k) in zip(r.kinds, kinds):
            rec.type, rec.role = type_of(k["unit"]), role
            rec.time, rec.obsolete, rec.squad_size = k["time"], k["obsolete"], k.get("squad_size", 1.0)
            rec.cost = 1.0 if k.get("cost") is None else k["cost"]     # SpawnTurret: (defenders[turret].cost) or 1
            rec.max_aggro = k.get("max_aggro", 0.0)
            rec.count, rec.burrow_ratio = given(k.get("count")), given(k.get("burrow_ratio"))
            rec.min_dist, rec.max_dist, rec.cooldown = given(k.get("min_dist")), given(k.get("max_dist")), k.get("cooldown", 1.0)
            rec.target_human, rec.tie_to_burrow = int(bool(k.get("target_human"))), int(bool(k.get("tie_to_burrow")))
        return r

    def weapon_of(self, unit: str, mount: int = 0) -> dict:
        return self.weapons[self.units[self.index[unit]]["mount_weapon"][mount]]


# LuaRules/Configs/start_setup.lua.  Storage is a commander's own: the game starts with none besides.
START_METAL, START_ENERGY = 350.0, 350.0
INNATE_METAL, INNATE_ENERGY = 2.0, 2.0
COMMANDER_STORAGE = 500.0


def _difficulty(difficulty: int | str) -> int:
    return CHICKEN_DIFFICULTIES.index(difficulty) if isinstance(difficulty, str) else int(difficulty)


def _set_economy(world, spots: np.ndarray, metal: float, energy: float, innate_metal: float, innate_energy: float, storage: float) -> None:
    triples = np.require(spots, np.float32, "C").reshape(-1, 3)
    _lib.zk_world_economy(world, len(triples), _floats(triples), metal, energy, innate_metal, innate_energy, storage)


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
            # Classes that move alike here get one answer between them: on a map made by hand, usually all of them do.
            known: dict[tuple[bytes, bytes], tuple] = {}
            built = []
            for s, v in zip(self.segments, speed):
                key = (s.tobytes(), v.tobytes())
                if key not in known:
                    known[key] = zknav.navigation(s, v)
                built.append(known[key])
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

    def set_teams(self, ally_of) -> None:
        """Each team's allyteam, team by team: (0, 0, 1) is two players together against a third.  Until set, two teams, each
        its own allyteam.  Before any unit is added."""
        allies = np.require(ally_of, np.int32, "C")
        _lib.zk_world_set_teams(self.handle, len(allies), _ints(allies))

    def add(self, name: str, team: int, x: float, z: float, heading: float = 0.0, hp: float = 1.0, build: float = 1.0) -> int:
        """A unit of this team (its allyteam is the team's).  Returns its handle, or -1 for a team the world hasn't."""
        return _lib.zk_world_add(self.handle, self.catalog.index[name], team, x, z, heading, hp, build)

    def move(self, unit: int, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_MOVE, x, z, -1)

    def fight(self, unit: int, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_FIGHT, x, z, -1)

    def set_move_state(self, unit: int, state: int) -> None:
        """Hold position 0, maneuver 1, roam 2."""
        _lib.zk_world_order(self.handle, unit, ORDER_MOVE_STATE, 0.0, 0.0, state)

    def set_fire_state(self, unit: int, state: int) -> None:
        """Hold fire 0, return fire 1, fire at will 2."""
        _lib.zk_world_order(self.handle, unit, ORDER_FIRE_STATE, 0.0, 0.0, state)

    def jump(self, unit: int, x: float, z: float) -> None:
        """Walks into range and goes when charged.  Nothing happens for what can't jump, or if it couldn't walk where it would land."""
        _lib.zk_world_order(self.handle, unit, ORDER_JUMP, x, z, -1)

    def self_destruct(self, unit: int) -> None:
        """A bomb goes off now, unless it's stunned.  Nothing else does."""
        _lib.zk_world_order(self.handle, unit, ORDER_SELF_DESTRUCT, 0.0, 0.0, 0)

    def set_unit_ai(self, unit: int, on: bool) -> None:
        """The tactical AI, which a player can turn off a unit at a time."""
        _lib.zk_world_order(self.handle, unit, ORDER_UNIT_AI, 0.0, 0.0, 1 if on else 0)

    def set_hp(self, unit: int, hp: float) -> None:
        _lib.zk_world_set_hp(self.handle, unit, hp)

    def kill(self, unit: int) -> None:
        _lib.zk_world_kill(self.handle, unit)

    def set_shield(self, unit: int, charge: float) -> None:
        _lib.zk_world_set_shield(self.handle, unit, charge)

    def attack(self, unit: int, target: int) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_ATTACK, 0.0, 0.0, target)

    def manual_fire(self, unit: int, target: int) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_MANUAL_FIRE, 0.0, 0.0, target)

    def manual_fire_at(self, unit: int, x: float, z: float) -> None:
        """At a place: what throws, throws there."""
        _lib.zk_world_order(self.handle, unit, ORDER_MANUAL_FIRE, x, z, -1)

    def set_on_off(self, unit: int, on: bool) -> None:
        """What has both pulls when off and pushes when on."""
        _lib.zk_world_order(self.handle, unit, ORDER_ON_OFF, 0.0, 0.0, 1 if on else 0)

    def place_beacon(self, unit: int, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_PLACE_BEACON, x, z, -1)

    def set_roles(self, role_of_type: np.ndarray) -> None:
        """Each catalog type's role, 0 to 31, or -1 for none: what group orders select by."""
        roles = np.require(role_of_type, np.int32, "C")
        _lib.zk_world_set_roles(self.handle, _ints(roles), len(roles))

    def order_group(self, anchor: int, roles: int, radius: float, kind: int, x: float = 0.0, z: float = 0.0, target: int = -1) -> int:
        """One order to the anchor and each other mobile, finished unit of its team of a role in the `roles` bits within `radius` of it.
        Returns how many it went to."""
        g = np.array([(anchor, roles, radius, kind, x, z, target)], GROUP_DTYPE)
        return _lib.zk_world_order_group(self.handle, g.ctypes.data)

    def wait_at_beacon(self, unit: int, beacon: int) -> None:
        """beacon: it, or the teleporter it belongs to."""
        _lib.zk_world_order(self.handle, unit, ORDER_WAIT_AT_BEACON, 0.0, 0.0, beacon)

    def load(self, transport: int, unit: int) -> None:
        """An air transport: fly to a unit it can lift, of any team, and pick it up."""
        _lib.zk_world_order(self.handle, transport, ORDER_LOAD, 0.0, 0.0, unit)

    def unload(self, transport: int, x: float, z: float) -> None:
        """Set its load down there, if that's a place the load can stand."""
        _lib.zk_world_order(self.handle, transport, ORDER_UNLOAD, x, z, -1)

    def drop(self, transport: int) -> None:
        """Drop Cargo: let its load fall from where it is.  Its other orders stay."""
        _lib.zk_world_order(self.handle, transport, ORDER_DROP, 0.0, 0.0, -1)

    def count(self) -> int:
        """Handles given out so far, the dead included."""
        return _lib.zk_world_count(self.handle)

    def ready_weapons(self) -> None:
        _lib.zk_world_ready_weapons(self.handle)

    def economy(self, spots: np.ndarray = np.zeros((0, 3)), metal: float = START_METAL, energy: float = START_ENERGY, innate_metal: float = INNATE_METAL,
                innate_energy: float = INNATE_ENERGY, storage: float = COMMANDER_STORAGE) -> None:
        """Turns on resources and construction.  spots: S x 3, x, z, income.  Before adding extractors: they read their spot when added."""
        _set_economy(self.handle, spots, metal, energy, innate_metal, innate_energy, storage)

    def build(self, unit: int, type_name: str, x: float, z: float) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_BUILD, x, z, self.catalog.index[type_name])

    def assist(self, unit: int, target: int) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_ASSIST, 0.0, 0.0, target)

    def produce(self, factory: int, type_name: str | None) -> None:
        _lib.zk_world_order(self.handle, factory, ORDER_PRODUCE, 0.0, 0.0, -1 if type_name is None else self.catalog.index[type_name])

    def reclaim(self, unit: int, feature: int) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_RECLAIM, 0.0, 0.0, feature)

    def cloak(self, unit: int, on: bool) -> None:
        _lib.zk_world_order(self.handle, unit, ORDER_CLOAK_ON if on else ORDER_CLOAK_OFF, 0.0, 0.0, -1)

    def add_feature(self, x: float, z: float, metal: float, hp: float = 100.0, radius: float = 16.0, feature_def: int = -1) -> int:
        return _lib.zk_world_add_feature(self.handle, x, z, metal, hp, radius, feature_def)

    def features(self, team: int) -> np.ndarray:
        """What this team's allyteam has in sight now."""
        records = np.zeros(FEATURES_PER_SIDE, rawframe.FEATURE_DTYPE)
        return records[:_lib.zk_world_features(self.handle, team, records.ctypes.data, FEATURES_PER_SIDE)]

    def set_reveal(self, on: bool) -> None:
        """game_over.lua's reveal rule, on by default where the catalog has rules.json; never in a chicken game."""
        _lib.zk_world_set_reveal(self.handle, int(on))

    def chickens(self, team: int, difficulty: int | str) -> None:
        """Chicken defense: this team, in an allyteam of its own and with no units, becomes the chickens, whom the world spawns and
        orders from now on.  difficulty: an index or name of CHICKEN_DIFFICULTIES."""
        _lib.zk_world_chickens(self.handle, team, _difficulty(difficulty))

    def chicken_state(self) -> dict[str, float]:
        out = np.zeros(len(CHICKEN_STATE), np.float32)
        _lib.zk_world_chicken_state(self.handle, _floats(out))
        return dict(zip(CHICKEN_STATE, map(float, out)))

    def resources(self, team: int) -> dict[str, float]:
        out = np.zeros(6, np.float32)
        _lib.zk_world_resources(self.handle, team, _floats(out))
        return dict(zip(("metal", "energy", "metal_income", "energy_income", "overdrive_metal", "overdrive_energy"), map(float, out)))

    def step(self, ticks: int = 1) -> None:
        _lib.zk_world_step(self.handle, ticks)

    @property
    def frame(self) -> int:
        return _lib.zk_world_frame(self.handle)

    def set_frame(self, frame: int) -> None:
        _lib.zk_world_set_frame(self.handle, frame)

    def view(self, team: int, commander: int = -1, base: tuple[float, float] = (0.0, 0.0), cap: int = 1024) -> "rawframe.Frame":
        """This team's raw frame as of now: its own units, its allies', then what its allyteam knows of the rest."""
        records = np.zeros(cap, rawframe.UNIT_DTYPE)
        glob = np.zeros(len(rawframe.GLOBAL_FIELDS), np.float32)
        where = np.array(base, np.float32)
        n = _lib.zk_world_view(self.handle, team, records.ctypes.data, cap, _floats(glob), commander, _floats(where))
        return rawframe.Frame(records[:n], np.zeros(0, rawframe.SITE_DTYPE), glob, self.features(team))

    def get(self, unit: int) -> UnitState:
        state = UnitState()
        _lib.zk_world_get(self.handle, unit, ct.byref(state))
        return state


ORDER_DTYPE = np.dtype([("unit", "i4"), ("kind", "i4"), ("x", "f4"), ("z", "f4"), ("target", "i4")], align=True)
# capi.h zk_group_rec: one order to the anchor and every unit of its team that's mobile and finished, of a role in `roles`, within `radius` of it
GROUP_DTYPE = np.dtype([("anchor", "i4"), ("roles", "u4"), ("radius", "f4"), ("kind", "i4"), ("x", "f4"), ("z", "f4"), ("target", "i4")], align=True)
# Per team: tracked value, live units, metal a second from reclaim, live units that keep the team in the game, and since the game began
# the damage its units have done to enemies' and had done to them by enemies', in metal (capi.h ZK_STATS).
STATS = 6


class Batch:
    """Games stepping across threads.  `step` leaves frames in preallocated arrays, so reading one is a view, not a copy.

    Every array is [games, teams, ...]: `teams` slots a game, of which a game uses the first `n_teams` its reset gave, and the rest
    read as empty frames (count 0, stats 0)."""

    def __init__(self, catalog: Catalog, games: int, threads: int = 16, cap: int = 1024, teams: int = 2):
        self.catalog, self.games, self.cap, self.teams = catalog, games, cap, teams
        self.handle = _lib.zk_batch_new(catalog.handle, games, threads, teams)
        self.records = np.zeros((games, teams, cap), rawframe.UNIT_DTYPE)
        self.counts = np.zeros((games, teams), np.int32)
        self.globs = np.zeros((games, teams, len(rawframe.GLOBAL_FIELDS)), np.float32)
        self.stats = np.zeros((games, teams, STATS), np.float32)
        self.features = np.zeros((games, teams, FEATURES_PER_SIDE), rawframe.FEATURE_DTYPE)
        self.feature_counts = np.zeros((games, teams), np.int32)
        self.sites: list[np.ndarray] = [np.zeros(0, rawframe.SITE_DTYPE)] * games
        self.positions = np.arange(games, dtype=np.int32)
        self._in_flight = None      # what a step begun and not yet waited for reads

    def __del__(self):
        if getattr(self, "handle", None):
            _lib.zk_batch_free(self.handle)
            self.handle = None

    def reset(self, game: int, world_map: Map, seed: int, type_: np.ndarray, team: np.ndarray, x: np.ndarray, z: np.ndarray,
              heading: np.ndarray | None = None, hp: np.ndarray | None = None, build: np.ndarray | None = None,
              commander=None, base=None, sites: np.ndarray | None = None, behaviour: int = ORDERS, ally_of=None, chickens=None) -> None:
        """team: each unit's.  ally_of: each team's allyteam, no more teams than the batch's slots; None for two teams in allyteams
        0 and 1.  commander: a handle per team, or -1 (all -1 if None); base: x, z per team (all 0 if None).  chickens: (team,
        difficulty) for chicken defense (World.chickens), the team in an allyteam of its own and given no units."""
        n = len(type_)
        i32 = lambda a: np.require(a, np.int32, "C")  # noqa: E731
        f32 = lambda a: np.require(a, np.float32, "C")  # noqa: E731
        allies = i32((0, 1) if ally_of is None else ally_of)
        n_teams = len(allies)
        assert n_teams <= self.teams, f"{n_teams} teams in a batch with room for {self.teams}"
        commander = np.full(n_teams, -1, np.int32) if commander is None else i32(commander)
        base = np.zeros(2 * n_teams, np.float32) if base is None else f32(base)
        assert len(commander) >= n_teams and len(base) >= 2 * n_teams, "commander and base go by the teams"
        zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
        _lib.zk_batch_reset(self.handle, game, world_map.handle, seed, behaviour, n_teams, _ints(allies), n, _ints(i32(type_)), _ints(i32(team)),
                            _floats(f32(x)), _floats(f32(z)), _floats(zeros if heading is None else f32(heading)), _floats(ones if hp is None else f32(hp)),
                            _floats(ones if build is None else f32(build)), _ints(commander), _floats(base))
        if chickens is not None:
            _lib.zk_batch_chickens(self.handle, game, int(chickens[0]), _difficulty(chickens[1]))
        if sites is not None:
            self.sites[game] = sites

    def set_reveal(self, on: bool) -> None:
        """game_over.lua's reveal rule (World.set_reveal) in every game, now and after any reset."""
        _lib.zk_batch_set_reveal(self.handle, int(on))

    def chicken_state(self, game: int) -> dict[str, float]:
        """capi.h ZK_CS_*, by name: all 0 and team -1 in a game without chickens."""
        out = np.zeros(len(CHICKEN_STATE), np.float32)
        _lib.zk_batch_chicken_state(self.handle, game, _floats(out))
        return dict(zip(CHICKEN_STATE, map(float, out)))

    def economy(self, game: int, spots: np.ndarray, metal: float = START_METAL, energy: float = START_ENERGY, innate_metal: float = INNATE_METAL,
                innate_energy: float = INNATE_ENERGY, storage: float = COMMANDER_STORAGE) -> None:
        """Turns on resources and construction in a game just reset."""
        _set_economy(_lib.zk_batch_world(self.handle, game), spots, metal, energy, innate_metal, innate_energy, storage)

    def add_features(self, game: int, features: np.ndarray) -> None:
        """What lies on a map from the start, in a game just reset.  N x 5: x, z, metal, hp, radius; and N: the def."""
        world = _lib.zk_batch_world(self.handle, game)
        for x, z, metal, hp, radius, feature_def in features:
            _lib.zk_world_add_feature(world, x, z, metal, hp, radius, int(feature_def))

    def _step_args(self, orders, order_offsets, games, ticks, groups, group_offsets) -> tuple:
        listed = self.positions if games is None else np.require(games, np.int32, "C")
        if orders is None:
            orders = np.zeros(0, ORDER_DTYPE)
            order_offsets = np.zeros(len(listed) + 1, np.int32)
        orders, order_offsets = np.require(orders, ORDER_DTYPE, "C"), np.require(order_offsets, np.int32, "C")
        if groups is not None:
            groups, group_offsets = np.require(groups, GROUP_DTYPE, "C"), np.require(group_offsets, np.int32, "C")
        call = (self.handle, _ints(listed), len(listed), orders.ctypes.data, _ints(order_offsets),
                None if groups is None else groups.ctypes.data, None if groups is None else _ints(group_offsets), ticks,
                self.records.ctypes.data, self.cap, _ints(self.counts), _floats(self.globs), _floats(self.stats),
                self.features.ctypes.data, FEATURES_PER_SIDE, _ints(self.feature_counts))
        return call, (listed, orders, order_offsets, groups, group_offsets)

    def step(self, orders: np.ndarray | None = None, order_offsets: np.ndarray | None = None, games: np.ndarray | None = None, ticks: int = 3,
             groups: np.ndarray | None = None, group_offsets: np.ndarray | None = None) -> None:
        """orders: ORDER_DTYPE, grouped by listed game; order_offsets: where each listed game's begin, plus the end.  groups, group_offsets:
        GROUP_DTYPE group orders likewise, which go before a game's unit orders."""
        call, _keep = self._step_args(orders, order_offsets, games, ticks, groups, group_offsets)
        _lib.zk_batch_step(*call)

    def step_begin(self, orders: np.ndarray | None = None, order_offsets: np.ndarray | None = None, games: np.ndarray | None = None, ticks: int = 3,
                   groups: np.ndarray | None = None, group_offsets: np.ndarray | None = None) -> None:
        """`step`, on the batch's threads while this returns at once.  The frames are being written until `step_wait`."""
        call, self._in_flight = self._step_args(orders, order_offsets, games, ticks, groups, group_offsets)
        _lib.zk_batch_step_begin(*call)

    def set_roles(self, role_of_type: np.ndarray) -> None:
        """Each catalog type's role, 0 to 31, or -1 for none: what group orders select by, in every game from now on."""
        roles = np.require(role_of_type, np.int32, "C")
        _lib.zk_batch_set_roles(self.handle, _ints(roles), len(roles))

    def step_wait(self) -> None:
        _lib.zk_batch_step_wait(self.handle)
        self._in_flight = None

    def set_threads(self, threads: int) -> None:
        _lib.zk_batch_set_threads(self.handle, threads)

    def copy_game(self, src: int, dst: int) -> None:
        """Game dst becomes a copy of game src as it stands, its random state too (zk_batch_copy_game), and its frames with it."""
        if self._in_flight is not None:
            raise RuntimeError("a step is under way: wait for it first")
        _lib.zk_batch_copy_game(self.handle, src, dst)
        self.sites[dst] = self.sites[src].copy()
        self.refresh(dst)

    def refresh(self, game: int) -> None:
        """This game's frames, in place, without stepping: after a reset, while the rest of the batch stays as it was."""
        empty = np.zeros(1, ORDER_DTYPE)
        _lib.zk_batch_step(self.handle, _ints(self.positions[game:game + 1]), 1, empty.ctypes.data, _ints(np.zeros(2, np.int32)), None, None, 0,
                           self.records[game:].ctypes.data, self.cap, _ints(self.counts[game:]), _floats(self.globs[game:]), _floats(self.stats[game:]),
                           self.features[game:].ctypes.data, FEATURES_PER_SIDE, _ints(self.feature_counts[game:]))

    def frame(self, position: int, team: int, game: int | None = None) -> rawframe.Frame:
        """A team's frame in the position-th listed game, a view into this batch's buffers, so valid until the next step."""
        n = int(self.counts[position, team])
        return rawframe.Frame(self.records[position, team, :n], self.sites[position if game is None else game], self.globs[position, team],
                              self.features[position, team, :int(self.feature_counts[position, team])])

    def frame_hash(self) -> int:
        """Of everything the last step wrote, to compare runs."""
        import hashlib
        h = hashlib.sha256()
        for position in range(self.games):
            for team in range(self.teams):
                n = int(self.counts[position, team])
                h.update(self.records[position, team, :n].tobytes())
                h.update(self.features[position, team, :int(self.feature_counts[position, team])].tobytes())
        h.update(self.stats.tobytes())
        return int.from_bytes(h.digest()[:8], "little")


def replay(catalog: Catalog, maps: list[Map], snapshot_map: np.ndarray, offsets: np.ndarray, type_: np.ndarray,
           side: np.ndarray, x: np.ndarray, z: np.ndarray, hp: np.ndarray, build: np.ndarray, tracked: np.ndarray,
           seeds: int, seed: int, horizons: list[int], behaviour: int, threads: int) -> np.ndarray:
    """Losses per snapshot, seed, horizon and side; horizons are in ticks."""
    n = len(snapshot_map)
    i32 = lambda a: np.require(a, np.int32, "C")  # noqa: E731
    f32 = lambda a: np.require(a, np.float32, "C")  # noqa: E731
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
