"""Checks of the simulator against formulas from the engine source.  In WSL:

  ~/zkml/bin/python sim/tests/test_sim.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402


def scripted_world(*args, **kwargs):
    """Units that do only what they're told and see everything, so a test measures one thing.  test_unit_ai.py is for the rest."""
    kwargs.setdefault("behaviour", zksim.SCRIPTED)
    return zksim.World(*args, **kwargs)

from zkbot.policy import rawframe  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
RAIDER = "cloakraid"
ORDER_DELAY = 15            # a move order waits for the unit's slow update
STATIC = "staticmex"
ARMORY = ("cloakraid", "cloakassault", "shieldraid", "tankassault", "spiderassault", "jumpraid", "hoverraid", "vehassault")


def test_top_speed_frames():
    w = scripted_world(CAT, FLAT)
    u = w.add(RAIDER, 0, 100, 2000, heading=math.pi / 2)
    w.move(u, 3000, 2000)
    d = CAT.units[CAT.index[RAIDER]]
    frames_needed = math.ceil(d["speed"] / d["accel"])
    w.step(math.ceil((frames_needed + ORDER_DELAY) / zksim.TICK_FRAMES))
    assert abs(w.get(u).speed - d["speed"]) < 1e-3


def test_straight_line_arrival():
    w = scripted_world(CAT, FLAT)
    u = w.add(RAIDER, 0, 100, 2000, heading=math.pi / 2)
    w.move(u, 2100, 2000)
    d = CAT.units[CAT.index[RAIDER]]
    w.step(int((2000 / d["speed"] + ORDER_DELAY) / zksim.TICK_FRAMES) + 3)
    s = w.get(u)
    assert abs(s.x - 2100) <= 20 and s.speed < d["speed"], (s.x, s.speed)


def test_turn_around_frames():
    d = CAT.units[CAT.index[RAIDER]]
    w = scripted_world(CAT, FLAT)
    u = w.add(RAIDER, 0, 2000, 2000, heading=0.0)
    w.move(u, 2000, 100)                  # directly behind
    frames = math.ceil(math.pi / d["turn_rate"])
    w.step(math.ceil((frames + ORDER_DELAY) / zksim.TICK_FRAMES))
    assert math.cos(w.get(u).heading) < -0.99, w.get(u).heading


def _blast_catalog(edge: float) -> zksim.Catalog:
    no_mounts = {"mount_weapon": [], "mount_only": [], "mount_bad": []}
    every = 1
    body = {"cost": 100.0, "hp": 100000.0, "speed": 0.0, "accel": 0.0, "brake": 0.0, "turn_rate": 0.0,
            "slow_turn_speed": 0.0, "footprint_radius": 1.0, "hit_radius": 5.0, "armored_mult": 1.0,
            "idle_regen": 0.0, "idle_time": 0.0, "max_range": 0.0, "armor_class": 0, "move_class": -1, "category": every,
            "flags": 0, "explode_weapon": -1}
    shooter = dict(body, name="shooter", flags=zksim.ARMED, max_range=500.0, mount_weapon=[0],
                   mount_only=[every], mount_bad=[0])
    blast = {"name": "blast", "damage": [100.0] * 5, "range": 500.0, "reload": 100000.0, "salvo": 1.0,
             "salvo_delay": 0.0, "projectiles": 1.0, "speed": 0.0, "accel": 0.0, "start_speed": 0.0,
             "blast_radius": 100.0, "edge": edge, "height_mod": 1.0, "spray": 0.0, "accuracy": 0.0,
             "moving_accuracy": 0.0, "para_seconds": 0.0, "extra_para": 0.0, "slow_factor": 0.0,
             "shield_power": 0.0, "shield_regen": 0.0, "shield_radius": 0.0, "vel_threshold": 0.0, "vel_base": 0.0,
             "vel_scale": 0.0, "alpha_damage": 0.0, "area_dps": 0.0, "area_radius": 0.0, "area_plateau": 0.0,
             "area_range_fall": 0.0, "area_plateau_fall": 0.0, "area_time_fall": 0.0, "area_frames": 0.0, "burn_frames": 0.0,
             "burn_random": 0.0, "burn_chance": 0.0, "burn_damage": 0.0, "kind": 0, "flags": zksim.W_TURRET, "shield_intercept": 0,
             "intercepted_by": 0}
    return zksim.Catalog.from_records([shooter, dict(body, name="box", **no_mounts)], [blast])


def test_blast_falloff():
    """GameHelper.cpp:146-150, measured from the target's surface."""
    for edge, gap, expected in ((0.0, 50.0, 50.001 / 100.001), (0.5, 50.0, 50.001 / 75.001), (1.0, 50.0, 1.0), (0.3, 0.5, 1.0)):
        w = scripted_world(_blast_catalog(edge), FLAT, behaviour=zksim.HOLD)
        w.add("shooter", 0, 2000, 1800)
        aimed = w.add("box", 1, 2000, 2000)
        nearby = w.add("box", 1, 2000 + 5.0 + gap, 2000)
        w.ready_weapons()
        w.step(1)
        assert abs(w.get(aimed).damage_taken - 100.0) < 1e-3
        assert abs(w.get(nearby).damage_taken - 100.0 * expected) < 1e-2, (edge, gap, w.get(nearby).damage_taken)


def _sturdy_target(attacker: str) -> str:
    """Unarmed and immobile, so it neither shoots back nor walks off; and the attacker can target it."""
    can_hit = CAT.units[CAT.index[attacker]]["mount_only"][0]
    options = [u for u in CAT.units if not u["flags"] & ~zksim.LEAVES_GHOST and u["hp"] > 2000.0 and u["armor_class"] == 2 and u["category"] & can_hit]
    return min(options, key=lambda u: u["hp"])["name"]


def test_salvo_damage_rate():
    """The raider's salvo: three shots per 9-frame reload, so 10 shots per second at point blank."""
    w = scripted_world(CAT, FLAT, behaviour=zksim.HOLD)
    w.add(RAIDER, 0, 2000, 2000, heading=0.0)
    target = w.add(_sturdy_target(RAIDER), 1, 2000, 2030)
    w.ready_weapons()
    w.step(6 * 6)                                   # 6 s
    weapon = CAT.weapon_of(RAIDER)
    expected = weapon["damage"][2] * 60
    got = w.get(target).damage_taken
    assert abs(got - expected) <= 3 * weapon["damage"][2], (got, expected)


def _hits_then_walks(attacker, gap, target="cloakassault", attack=None, seconds=3.0):
    """Hit the target until a status takes hold, then order it to walk: (state when hit, its walk, world, ids)."""
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    a = w.add(attacker, 0, 1500.0, 2000.0)
    t = w.add(target, 1, 1500.0 + gap, 2000.0)
    w.ready_weapons()
    (attack or w.attack)(a, t)
    for _ in range(240):
        w.step(1)
        state = w.get(t)
        if state.para > 1.0 or state.disarm > 1.0:
            break
    x0 = state.x
    w.move(t, x0 + 1500.0, 2000.0)
    w.step(int(seconds * 6))
    return state, w.get(t).x - x0, w, (a, t)


def _free_walk(target="cloakassault", seconds=3.0):
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    t = w.add(target, 1, 1700.0, 2000.0)
    w.move(t, 3200.0, 2000.0)
    w.step(int(seconds * 6))
    return w.get(t).x - 1700.0


def test_disarm_stops_weapons_and_not_walking():
    state, walked, _, _ = _hits_then_walks("shieldarty", 600.0)
    hp = CAT.units[CAT.index["cloakassault"]]["hp"]
    assert state.disarm > 1.0 and state.para == 0.0, (state.disarm, state.para)
    assert state.hp == hp and state.damage_taken == 0.0, "a disarm-only weapon does no damage"
    assert walked > 0.9 * _free_walk(), (walked, _free_walk())


def test_stun_stops_walking():
    state, walked, _, _ = _hits_then_walks("turretemp", 300.0)
    assert state.para > 1.0
    assert walked < 0.5 * _free_walk(), (walked, _free_walk())


def test_a_disarmed_unit_holds_its_fire_and_its_reload():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    arty = w.add("shieldarty", 0, 1500.0, 2000.0)
    raider = w.add("cloakraid", 1, 2100.0, 2000.0)
    dummy = w.add(_sturdy_target("cloakraid"), 0, 2100.0, 2120.0)
    w.ready_weapons()
    w.attack(arty, raider)
    while w.get(raider).disarm <= 1.0:
        w.step(1)
    w.step(3)            # shots already in the air
    before = w.get(dummy).damage_taken
    while w.get(raider).disarm > 1.0:
        w.step(1)
    assert w.get(dummy).damage_taken == before, (before, w.get(dummy).damage_taken)
    w.step(36)
    assert w.get(dummy).damage_taken > before, "it fires again once the disarm has worn off"


def test_a_disarm_runs_out_on_the_weapons_timer():
    """Racketeer: 6 s, plus the second of overstun if it is hit again near the threshold."""
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    arty = w.add("shieldarty", 0, 1500.0, 2000.0)
    t = w.add("cloakraid", 1, 2100.0, 2000.0)
    w.ready_weapons()
    w.attack(arty, t)
    while w.get(t).disarm <= 1.0:
        w.step(1)
    w.kill(arty)
    level = w.get(t).disarm
    assert abs(level - (1.0 + 6.0 / 40.0)) < 0.01, level
    ticks = 0
    while w.get(t).disarm > 1.0:
        w.step(1)
        ticks += 1
    assert abs(ticks / 6.0 - 6.0) < 0.5, ticks / 6.0
    w.step(6 * 20)
    assert 0.0 < w.get(t).disarm < 0.6, "it declines on below the threshold"


def test_a_hurt_unit_is_disarmed_sooner():
    weapon = next(x for x in CAT.weapons if x["name"] == "shipscout_missile")
    assert weapon["disarm_mult"] == 5.0 and not weapon["flags"] & 256, "this one also does damage"
    levels = {}
    for hp in (1.0, 0.2):
        w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
        scout = w.add("shipscout", 0, 1500.0, 2000.0)
        t = w.add("cloakassault", 1, 1500.0 + weapon["range"] * 0.5, 2000.0, hp=hp)
        w.ready_weapons()
        w.attack(scout, t)
        while w.get(t).disarm == 0.0:
            w.step(1)
        state = w.get(t)
        assert state.damage_taken > 0.0
        levels[hp] = state.disarm
    assert levels[0.2] > 3.0 * levels[1.0], levels


def test_capture_weapons_do_no_damage_and_stay_out_of_armies():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    a = w.add("vehcapture", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1800.0, 2000.0)
    w.ready_weapons()
    w.attack(a, t)
    w.step(36)
    assert w.get(t).damage_taken == 0.0
    flagged = {u["name"] for u in CAT.units if any(CAT.weapons[m]["flags"] & zksim.W_UNMODELLED for m in u["mount_weapon"])}
    assert {"vehcapture", "veharty", "hoverminer", "jumpblackhole"} <= flagged and "amphlaunch" not in flagged, flagged


def test_a_manual_weapon_waits_for_its_order_and_its_stock():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    reef = w.add("shipcarrier", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1500.0, 3500.0)
    w.step(6 * 3)
    assert w.get(reef).stock == 0
    w.manual_fire(reef, t)
    w.step(6 * 20)
    assert w.get(t).disarm == 0.0, "nothing to fire yet: a missile takes 30 s to stockpile"
    assert w.get(reef).order == zksim.ORDER_MANUAL_FIRE, "the order waits"
    ticks = 0
    while w.get(t).disarm == 0.0 and ticks < 6 * 30:
        w.step(1)
        ticks += 1
    state = w.get(reef)
    assert 6 * 6 <= ticks < 6 * 15, ticks      # 30 s in all, from the 23 s gone by, plus its flight
    assert state.stock == 0 and state.order == zksim.ORDER_NONE, "firing uses the stock and finishes the order"
    assert w.get(t).hp == CAT.units[CAT.index["cloakassault"]]["hp"]
    assert abs(w.get(t).disarm - 1.25) < 0.02, w.get(t).disarm    # the missile's 10 s


def test_no_order_means_no_manual_shot():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    reef = w.add("shipcarrier", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1500.0, 3500.0)
    w.ready_weapons()
    w.step(6 * 20)
    assert w.get(reef).stock >= 1 and w.get(t).disarm == 0.0
    w.attack(reef, t)
    w.step(6 * 20)
    assert w.get(t).disarm == 0.0, "an attack order doesn't fire it either"


def test_a_unit_without_a_manual_weapon_ignores_the_order():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    a = w.add("cloakraid", 0, 1500.0, 2000.0)
    t = w.add("cloakraid", 1, 1800.0, 2000.0)
    w.manual_fire(a, t)
    w.step(1)
    assert w.get(a).order == zksim.ORDER_NONE


def test_a_stun_stops_the_stockpile():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    arty = w.add("shieldarty", 0, 1500.0, 2000.0)
    reef = w.add("shipcarrier", 1, 2100.0, 2000.0)
    w.attack(arty, reef)
    w.step(6 * 31)
    assert w.get(reef).stock == 0 or w.get(reef).disarm <= 1.0, (w.get(reef).stock, w.get(reef).disarm)


def test_heat_rays_fade_with_distance():
    taken = {}
    for gap in (60.0, 200.0):
        w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.HOLD)
        a = w.add("vehraid", 0, 1500.0, 2000.0)
        t = w.add(_sturdy_target("vehraid"), 1, 1500.0 + gap, 2000.0)
        w.ready_weapons()
        w.step(30)
        taken[gap] = w.get(t).damage_taken
        assert w.get(a).alive
    assert taken[60.0] > 2.0 * taken[200.0] > 0.0, taken


def test_fake_guns_do_not_fire():
    fake = next(x for x in CAT.weapons if x["name"] == "amphfloater_fake_cannon")
    assert fake["flags"] & zksim.W_NO_AUTO_TARGET
    armcom = next(x for x in CAT.weapons if x["name"] == "armcom0_fakelaser")
    assert not armcom["flags"] & zksim.W_NO_AUTO_TARGET, "commanders aim with their harmless ones"


def test_slow_only_beams_slow():
    state_free = _free_walk("cloakassault")
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
    a = w.add("tankheavyassault", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1800.0, 2000.0)
    w.ready_weapons()
    w.step(6)
    x0 = w.get(t).x
    w.move(t, x0 + 1500.0, 2000.0)
    w.step(18)
    assert w.get(a).alive and w.get(t).x - x0 < 0.85 * state_free, (w.get(t).x - x0, state_free)


def test_ronin_is_slower_while_reloading():
    def walked(fire):
        w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.SCRIPTED)
        a = w.add("cloakskirm", 0, 1500.0, 2000.0)
        w.add(_sturdy_target("cloakskirm"), 1, 1500.0, 2200.0 if fire else 3900.0)
        w.ready_weapons()
        w.move(a, 3000.0, 2000.0)
        w.step(12)
        x0 = w.get(a).x
        w.step(12)
        return w.get(a).x - x0
    assert walked(True) < 0.8 * walked(False), (walked(True), walked(False))


def test_felon_fires_from_its_shield():
    w = scripted_world(CAT, FLAT, seed=3, behaviour=zksim.HOLD)
    a = w.add("shieldfelon", 0, 1500.0, 2000.0)
    t = w.add(_sturdy_target("shieldfelon"), 1, 1800.0, 2000.0)
    w.ready_weapons()
    full = w.get(a).shield
    w.step(6)
    assert w.get(t).damage_taken > 0.0 and w.get(a).shield < full, (w.get(t).damage_taken, w.get(a).shield, full)


def test_emp_decay():
    """Paralysis falls by 1/40 of max HP per second (Game.paralyzeDeclineRate)."""
    weapon = CAT.weapon_of("cloakassault")
    name = _sturdy_target("cloakassault")
    w = scripted_world(CAT, FLAT, behaviour=zksim.HOLD)
    w.add("cloakassault", 0, 2000, 2000)
    target = w.add(name, 1, 2000, 2100)
    w.ready_weapons()
    w.step(1)
    first = w.get(target).para
    hp = CAT.units[CAT.index[name]]["hp"]
    # Stun goes by current health, which this shot has just lowered.
    assert abs(first - (weapon["extra_para"] / (hp - weapon["damage"][2]) - zksim.TICK_FRAMES / 1200.0)) < 1e-4, first
    w.step(10)                                      # 50 frames, less than the 66-frame reload
    assert abs(first - w.get(target).para - 50.0 / 1200.0) < 1e-4


def test_a_move_order_just_before_the_update_waits_for_the_next():
    """Most phases lose most of their speed to a re-order every decision, and none keeps it all."""
    kept = []
    for start in range(15):
        w = scripted_world(CAT, FLAT, seed=1, behaviour=zksim.SCRIPTED)
        w.set_frame(30000 + start)
        u = w.add(RAIDER, 0, 200, 2000, heading=math.pi / 2)
        for tick in range(66):
            if tick % 3 == 0:
                w.move(u, 3900, 2000)
            w.step(1)
        kept.append((w.get(u).x - 200) / (CAT.units[CAT.index[RAIDER]]["speed"] * 66 * zksim.TICK_FRAMES))
    assert max(kept) < 0.97 and sorted(kept)[len(kept) // 2] < 0.5, [round(k, 2) for k in kept]


def test_move_order_stops_the_unit_fight_order_does_not():
    def moved(kind):
        w = scripted_world(CAT, FLAT)
        u = w.add(RAIDER, 0, 200, 2000, heading=math.pi / 2)
        for tick in range(66):              # 11 s
            if tick % 6 == 0:               # an order every second
                (w.move if kind == "move" else w.fight)(u, 3900, 2000)
            w.step(1)
        return w.get(u).x - 200
    assert moved("move") < 0.95 * moved("fight"), (moved("move"), moved("fight"))


def test_climbing_is_slower_than_descending():
    rows, cols = 256, 256
    height = np.tile(np.arange(rows, dtype=np.float32)[:, None] * 1.5, (1, cols))     # rises with z
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32) * 0.8
    slope = np.full((rows, cols), 0.05, np.float32)
    mods = np.array([9.68, 20.0, 5.0, 9.68, 30.0, 0.0], np.float32)
    hill = zksim.Map(height, speed, 16.0, "hill", slope, mods)

    def walk(start_z, goal_z):
        w = scripted_world(CAT, hill)
        u = w.add(RAIDER, 0, 2000, start_z, heading=0.0 if goal_z > start_z else math.pi)
        w.move(u, 2000, goal_z)
        w.step(60)
        return abs(w.get(u).z - start_z)
    assert walk(1000, 3000) < walk(3000, 1000) * 0.9, (walk(1000, 3000), walk(3000, 1000))


def test_unbuilt_does_not_fire():
    w = scripted_world(CAT, FLAT, behaviour=zksim.HOLD)
    w.add(RAIDER, 0, 2000, 2000, build=0.5)
    target = w.add(STATIC, 1, 2000, 2050)
    w.ready_weapons()
    w.step(60)
    assert w.get(target).damage_taken == 0.0


def test_terrain_blocks_and_routes():
    """A wall across the map with a gap: the unit must walk around, not through."""
    rows, cols = 256, 256
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32)
    speed[:, 128, :] = 0.0
    speed[:, 128, 150:160] = 1.0
    segments = np.zeros(speed.shape, np.int16)
    segments[:, 128:, :] = 1
    segments[:, 128, :] = -1
    segments[:, 128, 150:160] = 0
    world_map = zksim.Map(np.zeros((rows, cols), np.float32), speed, 16.0, "wall", segments=segments)
    w = scripted_world(CAT, world_map)
    u = w.add(RAIDER, 0, 1000, 1500)
    w.move(u, 1000, 2500)
    crossed_at = None
    for _ in range(1500):
        w.step(1)
        s = w.get(u)
        if crossed_at is None and s.z > 128 * 16 + 16:
            crossed_at = s.x
        if abs(s.z - 2500) < 32 and abs(s.x - 1000) < 32:
            break
    assert crossed_at is not None and 145 * 16 <= crossed_at <= 165 * 16, crossed_at
    assert abs(s.z - 2500) < 32, (s.x, s.z)


def test_a_blocked_line_inside_a_segment_goes_around():
    rows, cols = 256, 256
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32)
    speed[:, 128, 90:216] = 0.0                     # a wall with a way around, all in one segment
    world_map = zksim.Map(np.zeros((rows, cols), np.float32), speed, 16.0, "one segment")
    w = scripted_world(CAT, world_map)
    u = w.add(RAIDER, 0, 2048, 1800)
    w.move(u, 2048, 2300)
    w.step(600)
    s = w.get(u)
    assert np.hypot(s.x - 2048, s.z - 2300) < 64, (s.x, s.z)


def test_a_stuck_unit_finds_its_own_way_out():
    rows, cols = 256, 256
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32)
    speed[:, 128, 90:170] = 0.0                     # a wall between the unit and its segment's centre
    empty = {"centres": np.array([[130.5, 160.5]], np.float32), "edge_to": np.zeros(0, np.int32),
             "gates": np.zeros((0, 2), np.float32), "next_edge": np.full((1, 1), -1, np.int32)}
    world_map = zksim.Map(np.zeros((rows, cols), np.float32), speed, 16.0, "hidden centre",
                          segments=np.zeros(speed.shape, np.int16), routes=[empty] * len(zksim.MOVE_CLASSES))
    w = scripted_world(CAT, world_map)
    u = w.add(RAIDER, 0, 130.5 * 16, 1900)
    w.move(u, 3200, 2300)                           # heading straight for it slides along the wall and round the end
    w.step(600)
    s = w.get(u)
    assert np.hypot(s.x - 3200, s.z - 2300) < 64, (s.x, s.z)


def test_goals_in_one_segment_share_a_way_in():
    rows, cols = 256, 256
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32)
    speed[:, 128, :] = 0.0
    speed[:, 128, 150:160] = 1.0
    segments = np.zeros(speed.shape, np.int16)
    segments[:, 129:, :] = 1
    world_map = zksim.Map(np.zeros((rows, cols), np.float32), speed, 16.0, "wall", segments=segments)
    w = scripted_world(CAT, world_map)
    units = [w.add(RAIDER, 0, 1000, 1500), w.add(RAIDER, 0, 1100, 1500)]
    goals = [(1000, 2500), (3000, 3000)]
    for u, g in zip(units, goals):
        w.move(u, *g)
    w.step(500)
    for u, g in zip(units, goals):
        s = w.get(u)
        assert np.hypot(s.x - g[0], s.z - g[1]) < 64, (s.x, s.z, g)


def test_segments_cover_the_map_data():
    d = np.load(zksim.DATA / "maps/titanduel_2_2.npz")
    assert d["segments"].shape == d["speed"].shape
    assert (d["segments"][d["speed"] <= 0] == -1).mean() > 0.99


def test_routes_stay_small():
    d = np.load(zksim.DATA / "maps/titanduel_2_2.npz")
    bytes_used = sum(d[f"class{c}_{k}"].nbytes for c in range(d["speed"].shape[0]) for k in zksim.ROUTE_ARRAYS)
    assert bytes_used < 1_000_000, bytes_used


def _melee(seed: int, n: int = 40):
    w = scripted_world(CAT, FLAT, seed=seed, behaviour=zksim.ADVANCE)
    units = []
    rng = np.random.default_rng(1234)
    for side in (0, 1):
        for _ in range(n):
            units.append(w.add(ARMORY[rng.integers(len(ARMORY))], side, 1800 + 400 * side + rng.uniform(-150, 150),
                               1800 + rng.uniform(-300, 300)))
    w.step(120)
    return [(w.get(u).alive, round(w.get(u).x, 3), round(w.get(u).hp, 3)) for u in units]


def test_determinism():
    assert _melee(5) == _melee(5)
    assert _melee(5) != _melee(6)


def test_replay_threads_do_not_change_results():
    rng = np.random.default_rng(9)
    per_snapshot = 30
    n = 64
    names = [CAT.index[a] for a in ARMORY]
    type_ = rng.choice(names, n * per_snapshot)
    side = np.tile(np.arange(per_snapshot) % 2, n)
    x = 1800 + side * 300 + rng.uniform(-200, 200, n * per_snapshot)
    z = 1800 + rng.uniform(-200, 200, n * per_snapshot)
    hp = rng.uniform(0.3, 1.0, n * per_snapshot)
    args = (CAT, [FLAT], np.zeros(n, np.int32), np.arange(n + 1) * per_snapshot, type_, side, x, z, hp,
            np.ones_like(hp), np.ones_like(hp, np.uint8))
    a = zksim.replay(*args, 4, 77, [60, 120], zksim.ADVANCE, 1)
    b = zksim.replay(*args, 4, 77, [60, 120], zksim.ADVANCE, 8)
    assert np.array_equal(a, b)
    assert (a[:, :, 1] >= a[:, :, 0] - 1e-3).all(), "losses don't shrink with time"
    assert a[:, :, 1].sum() > 0


def test_budget_400_units():
    w = scripted_world(CAT, zksim.Map.load(zksim.DATA / "maps/titanduel_2_2.npz"), behaviour=zksim.ADVANCE)
    rng = np.random.default_rng(3)
    for i in range(400):
        w.add(ARMORY[rng.integers(len(ARMORY))], i % 2, rng.uniform(800, 4300), rng.uniform(800, 4300))
    t0 = time.perf_counter()
    w.step(100)
    per_tick = (time.perf_counter() - t0) / 100
    print(f"  400 units: {per_tick * 1e6:.0f} us per tick, single thread (plan: tick budget in 01 6.1)")


def sight_map() -> zksim.Map:
    """Flat at 0, with a tall ridge running north-south at x 1000-1100."""
    rows, cols = 256, 256
    height = np.zeros((rows, cols), np.float32)
    height[:, 1000 // 16:1100 // 16] = 400.0
    return zksim.Map(height, np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32), 16.0, "ridge")


def records(frame) -> dict:
    return {int(r["uid"]): r for r in frame.units}


def test_fog_shows_what_is_in_sight_and_only_the_critic_sees_the_rest():
    w = scripted_world(CAT, FLAT)
    me = w.add(RAIDER, 0, 500.0, 500.0)
    near = w.add(RAIDER, 1, 800.0, 500.0)
    far = w.add(RAIDER, 1, 3000.0, 3000.0)
    r = records(w.view(0))
    assert r[me]["player"] == 0 and r[me]["flags"] & rawframe.IN_LOS
    assert r[near]["knowledge"] == rawframe.KNOW_LOS and r[near]["flags"] & rawframe.TYPE_KNOWN
    assert r[near]["def_id"] == CAT.units[CAT.index[RAIDER]]["def_id"]
    assert r[far]["knowledge"] == rawframe.KNOW_HIDDEN
    # The record is as the live bot gives it: elmos per frame, health as a share.
    assert abs(r[near]["x"] - 800.0) < 1e-3 and r[near]["hp_frac"] == 1.0 and r[me]["ord_kind"] == -1


def test_a_ridge_hides_what_is_behind_it():
    w = scripted_world(CAT, sight_map())
    w.add(RAIDER, 0, 800.0, 2000.0)
    behind = w.add(RAIDER, 1, 1300.0, 2000.0)
    same_range = w.add(RAIDER, 1, 800.0, 1500.0)
    r = records(w.view(0))
    assert r[behind]["knowledge"] == rawframe.KNOW_HIDDEN
    assert r[same_range]["knowledge"] == rawframe.KNOW_LOS


def test_radar_gives_a_blip_with_identity_only_if_seen_since():
    w = scripted_world(CAT, FLAT)
    w.add("staticradar", 0, 500.0, 500.0)
    a = w.add(RAIDER, 1, 2000.0, 500.0)
    r = records(w.view(0))
    assert r[a]["knowledge"] == rawframe.KNOW_RADAR and r[a]["def_id"] == -1 and np.isnan(r[a]["hp_frac"])
    assert not r[a]["flags"] & (rawframe.IN_LOS | rawframe.TYPE_KNOWN)
    # Seen once, it keeps its identity on radar; lost from radar too, it loses it.
    scout = w.add(RAIDER, 0, 1800.0, 500.0)
    assert records(w.view(0))[a]["knowledge"] == rawframe.KNOW_LOS
    w.kill(scout)
    w.step(1)
    r = records(w.view(0))
    assert r[a]["knowledge"] == rawframe.KNOW_RADAR and r[a]["def_id"] > 0 and r[a]["flags"] & rawframe.TYPE_KNOWN
    w2 = scripted_world(CAT, FLAT)
    radar = w2.add("staticradar", 0, 500.0, 500.0)
    b = w2.add(RAIDER, 1, 600.0, 500.0)
    assert records(w2.view(0))[b]["knowledge"] == rawframe.KNOW_LOS
    w2.kill(radar)
    w2.step(1)
    assert records(w2.view(0))[b]["knowledge"] == rawframe.KNOW_HIDDEN
    assert w2.get(b).alive


def test_a_blip_top_speed_is_the_speed_it_has_held():
    w = scripted_world(CAT, FLAT)
    w.add("staticradar", 0, 500.0, 500.0)
    a = w.add(RAIDER, 1, 1500.0, 500.0)
    w.move(a, 2500.0, 2500.0)
    top = CAT.units[CAT.index[RAIDER]]["speed"] * 30.0
    r = None
    for _ in range(12):
        w.step(3)
        r = records(w.view(0))
    assert r[a]["knowledge"] == rawframe.KNOW_RADAR
    assert 0.5 * top < r[a]["blip_top_speed"] <= top * 1.05, r[a]["blip_top_speed"]


def test_a_remembered_building_stays_until_seen_dead():
    w = scripted_world(CAT, FLAT)
    scout = w.add(RAIDER, 0, 500.0, 500.0)
    mex = w.add("staticmex", 1, 800.0, 500.0)
    assert records(w.view(0))[mex]["knowledge"] == rawframe.KNOW_LOS
    w.kill(scout)
    w.step(1)
    r = records(w.view(0))
    assert r[mex]["knowledge"] == rawframe.KNOW_MEMORY and r[mex]["flags"] & rawframe.TYPE_KNOWN and r[mex]["ord_age"] >= 5
    # Killed out of sight, it's still remembered, until someone looks again.
    w.kill(mex)
    w.step(1)
    assert records(w.view(0))[mex]["knowledge"] == rawframe.KNOW_MEMORY
    w.add(RAIDER, 0, 800.0, 400.0)
    w.step(1)
    assert mex not in records(w.view(0))


def test_a_frame_remembers_orders_idleness_and_damage():
    w = scripted_world(CAT, FLAT)
    a = w.add("cloakassault", 0, 500.0, 500.0)
    idle = w.add("cloakassault", 0, 500.0, 700.0)
    shooter = w.add(RAIDER, 1, 800.0, 500.0)
    w.fight(a, 520.0, 500.0)
    w.attack(idle, shooter)
    w.step(3)
    r = records(w.view(0))
    assert r[a]["ord_kind"] == rawframe.FIGHT and r[a]["ord_x"] == 520.0 and r[a]["ord_age"] == 15
    assert r[idle]["ord_kind"] == rawframe.ATTACK_UNIT and r[idle]["ord_target"] == shooter
    assert r[shooter]["knowledge"] == rawframe.KNOW_LOS
    w.kill(shooter)
    w.step(1)
    for _ in range(30):
        w.step(3)
        r = records(w.view(0))
    assert r[idle]["flags"] & rawframe.IDLE
    assert r[a]["flags"] & rawframe.IDLE
    # Damage shows once, the decision after it happened.
    w3 = scripted_world(CAT, FLAT)
    target = w3.add("cloakassault", 0, 500.0, 500.0)
    other = w3.add("cloakassault", 1, 600.0, 500.0)
    w3.ready_weapons()
    assert not records(w3.view(0))[target]["flags"] & rawframe.DAMAGED
    w3.attack(other, target)
    for _ in range(5):
        w3.step(3)
        flags = records(w3.view(0))[target]["flags"]
        if flags & rawframe.DAMAGED:
            break
    assert flags & rawframe.DAMAGED


def batch_game(batch, game, seed, world_map, count=60):
    rng = np.random.default_rng(seed)
    types = rng.integers(0, len(ARMORY), count)
    names = np.array([CAT.index[ARMORY[int(t)]] for t in types])
    side = np.arange(count) % 2
    x = np.where(side == 0, rng.uniform(300, 1500, count), rng.uniform(2500, 3700, count))
    z = rng.uniform(300, 3700, count)
    batch.reset(game, world_map, seed, names, side, x, z, commander=(0, 1), base=(300.0, 300.0, 3700.0, 3700.0))
    return side, x, z


def run_batch(threads: int, games: int = 12, decisions: int = 12) -> list[int]:
    batch = zksim.Batch(CAT, games, threads=threads)
    for g in range(games):
        batch_game(batch, g, 100 + g, FLAT)
    hashes = []
    orders = np.zeros(games * 4, zksim.ORDER_DTYPE)
    for g in range(games):
        for k in range(4):
            orders[g * 4 + k] = (k, zksim.ORDER_FIGHT, 2000.0, 2000.0, -1)
    offsets = np.arange(games + 1, dtype=np.int32) * 4
    for d in range(decisions):
        if d == 0:
            batch.step(orders, offsets)
        else:
            batch.step()
        hashes.append(batch.frame_hash())
    return hashes


def test_a_batch_does_not_depend_on_its_thread_count():
    assert run_batch(1) == run_batch(16) == run_batch(5)


def test_a_live_frame_recreated_here_reads_the_same_to_the_policy():
    """Units from a live game placed in the sim: the featurizer must get the same type, place, health and terrain view.  Fog is
    an approximation, so it only has to stay near; the sim seeing what the game didn't would be the worse mistake."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "tools"))
    import policy_golden
    seen_live = seen_sim = 0
    for index in range(5):
        worst = policy_golden.recreate_live(index)
        for name in ("type", "x,z", "build", "free rays"):
            assert worst[name] < 1e-3, (index, name, worst[name])
        assert worst["hp"] < 0.02, (index, worst["hp"])
        assert worst["ground altitude"] < 0.01, (index, worst["ground altitude"])
        assert worst["seen by the sim alone"] == 0, index
        seen_live += worst["seen live"]
        seen_sim += worst["seen by the sim"]
    assert seen_sim >= 0.7 * seen_live, (seen_sim, seen_live)


def test_a_refreshed_game_gets_its_frames_where_they_belong():
    batch = zksim.Batch(CAT, 3, threads=1)
    for g in range(3):
        batch_game(batch, g, 40 + g, FLAT)
    batch.step(ticks=0)
    before = batch.records[0, 0].copy(), batch.records[2, 1].copy()
    batch_game(batch, 1, 77, FLAT, count=20)
    batch.refresh(1)
    assert batch.counts[1, 0] == 20 and batch.stats[1, 0, 1] == 10
    assert np.array_equal(batch.records[0, 0], before[0]) and np.array_equal(batch.records[2, 1], before[1])


def test_a_batch_step_fills_frames_for_both_sides():
    batch = zksim.Batch(CAT, 2, threads=2)
    batch_game(batch, 0, 7, FLAT)
    batch_game(batch, 1, 8, FLAT)
    batch.step(ticks=3)
    a, b = batch.frame(0, 0), batch.frame(0, 1)
    assert (a.units["player"] == 0).sum() == 30 == (b.units["player"] == 1).sum()
    assert (a.units["ally_team"] == a.units["player"]).all()       # one player to an alliance, for now
    assert a.glob[rawframe.G["player"]] == a.glob[rawframe.G["ally_team"]] == 0 and b.glob[rawframe.G["player"]] == b.glob[rawframe.G["ally_team"]] == 1
    assert len(a.units) == len(b.units) == 60                      # everyone's in the frame, as KNOW_HIDDEN if hidden
    assert a.glob[rawframe.G["frame"]] == 15
    assert a.glob[rawframe.G["commander_uid"]] == 0 and b.glob[rawframe.G["commander_uid"]] == 1
    assert a.glob[rawframe.G["base_x"]] == 300.0 and b.glob[rawframe.G["base_x"]] == 3700.0
    assert batch.stats[0, 0, 1] == 30 and batch.stats[0, 0, 0] > 0
    assert not np.array_equal(batch.records[0, 0, :60], batch.records[1, 0, :60])


COMMANDER = "dyntrainer_strike_base"       # build speed 10, range 144, makes 4 metal and 6 energy
TICKS_PER_SECOND = zksim.FRAMES_PER_SECOND / zksim.TICK_FRAMES


def _economy(metal=1000.0, energy=1000.0, innate=0.0, storage=100000.0, spots=((1000.0, 1000.0, 2.0),)):
    world = scripted_world(CAT, FLAT)
    world.economy(np.array(spots, np.float32).reshape(-1, 3), metal, energy, innate, innate, storage)
    return world


def _ticks_until_built(world, unit, most=4000):
    for n in range(most):
        if world.get(unit).build >= 1.0:
            return n
        world.step(1)
    raise AssertionError(f"not built in {most} ticks")


def test_a_sides_first_factory_is_free_and_instant_and_its_second_is_neither():
    world = _economy()
    commander = world.add(COMMANDER, 0, 500.0, 500.0)
    world.build(commander, "factorycloak", 600.0, 500.0)
    world.step(1)
    factory = commander + 1
    assert world.get(factory).build == 1.0 and world.get(factory).hp == CAT.units[CAT.index["factorycloak"]]["hp"]
    assert world.resources(0)["metal"] >= 1000.0
    assert world.get(commander).order == zksim.ORDER_NONE
    world.build(commander, "factoryveh", 500.0, 650.0)
    world.step(2)
    assert 0.0 < world.get(factory + 1).build < 0.01 and world.resources(0)["metal"] < 1000.0


def test_the_other_side_keeps_its_free_factory():
    world = _economy()
    a, b = world.add(COMMANDER, 0, 500.0, 500.0), world.add(COMMANDER, 1, 2000.0, 500.0)
    world.build(a, "factorycloak", 600.0, 500.0)
    world.build(b, "factorycloak", 2100.0, 500.0)
    world.step(1)
    assert world.get(b + 1).build == 1.0 and world.get(b + 2).build == 1.0


def test_a_build_takes_its_build_time_and_costs_its_price():
    world = _economy(metal=1000.0, energy=1000.0)
    commander = world.add(COMMANDER, 0, 1000.0, 900.0)
    world.build(commander, "staticmex", 1000.0, 1000.0)
    world.step(1)
    ticks = 1 + _ticks_until_built(world, commander + 1)
    assert abs(ticks / TICKS_PER_SECOND - 8.5) < 0.34, ticks          # 85 at build speed 10
    made = 4.0 * ticks / TICKS_PER_SECOND
    assert abs(world.resources(0)["metal"] - (1000.0 - 85.0 + made)) < 1.0, world.resources(0)
    assert world.get(commander).order == zksim.ORDER_NONE


def test_a_frame_gains_health_as_it_builds():
    world = _economy()
    commander = world.add(COMMANDER, 0, 1000.0, 900.0)
    world.build(commander, "energysolar", 1000.0, 1000.0)
    world.step(int(3.5 * TICKS_PER_SECOND))
    frame = world.get(commander + 1)
    hp = CAT.units[CAT.index["energysolar"]]["hp"]
    assert 0.4 < frame.build < 0.6 and abs(frame.hp / hp - frame.build) < 0.02


def test_building_stalls_without_metal_and_without_energy():
    for metal, energy, seconds in ((0.0, 1000.0, 85.0 / 4.0), (1000.0, 0.0, 85.0 / 6.0)):   # on what the commander makes
        world = _economy(metal=metal, energy=energy)
        commander = world.add(COMMANDER, 0, 1000.0, 900.0)
        world.build(commander, "staticmex", 1000.0, 1000.0)
        world.step(1)
        taken = _ticks_until_built(world, commander + 1) / TICKS_PER_SECOND
        assert abs(taken - seconds) < 0.5, (metal, energy, taken)


def test_income_comes_from_what_is_finished_alive_and_on_a_spot():
    world = _economy(innate=2.0, metal=0.0, energy=0.0)
    world.step(1)
    assert world.resources(0)["metal_income"] == 2.0 and world.resources(0)["energy_income"] == 2.0
    on_spot, off_spot = world.add("staticmex", 0, 1000.0, 1000.0), world.add("staticmex", 0, 1500.0, 1000.0)
    world.add("staticmex", 0, 1000.0, 1000.0, build=0.5)
    solar = world.add("energysolar", 0, 1100.0, 1000.0)
    commander = world.add(COMMANDER, 0, 1300.0, 1300.0)
    world.add("cloakcon", 0, 1300.0, 1500.0)        # a builder left when the commander goes: else its team is out (check_vital)
    world.step(1)
    r = world.resources(0)
    assert r["metal_income"] == 2.0 + 2.0 + 4.0 and r["energy_income"] == 2.0 + 2.0 + 6.0, r
    assert world.resources(1)["metal_income"] == 2.0 and on_spot != off_spot
    world.kill(commander)
    world.kill(solar)
    world.step(2)
    r = world.resources(0)
    assert r["metal_income"] == 2.0 + 2.0 and r["energy_income"] == 2.0, r


def test_storage_is_a_ceiling():
    world = _economy(metal=499.9, energy=499.9, storage=500.0, innate=100.0)
    world.step(6)
    assert world.resources(0)["metal"] == 500.0 and world.resources(0)["energy"] == 500.0


def _hurt_then_alone(name):
    """A unit that has just taken damage, with nothing left to hurt it."""
    world = scripted_world(CAT, FLAT)
    unit = world.add(name, 0, 1000.0, 1000.0, hp=0.5)
    enemy = world.add("cloakraid", 1, 1100.0, 1000.0)
    for _ in range(600):
        world.step(1)
        if world.get(unit).damage_taken > 0.0:
            world.kill(enemy)
            return world, unit
    raise AssertionError("never hit")


@pytest.mark.parametrize("name, wait, rate", (("cloakraid", 5.0, 20.0), ("tankraid", 10.0, 5.0), ("vehraid", 60.0, 5.0)))
def test_a_unit_regenerates_at_its_rate_after_its_time_without_damage(name, wait, rate):
    world, unit = _hurt_then_alone(name)
    # A shot may still be in flight, and starts the wait again.  So it's counted from the last hit.
    state = world.get(unit)
    damage, hp, quiet, first_gain = state.damage_taken, state.hp, 0, None
    while first_gain is None:
        world.step(1)
        state = world.get(unit)
        quiet = 0 if state.damage_taken > damage else quiet + 1
        damage = state.damage_taken
        if state.hp > hp and first_gain is None:
            first_gain = quiet
        hp = state.hp
        assert quiet < (wait + 5.0) * TICKS_PER_SECOND
    assert abs(first_gain / TICKS_PER_SECOND - wait) < 0.5
    world.step(int(2 * TICKS_PER_SECOND))
    assert abs(world.get(unit).hp - hp - 2.0 * rate) < 0.5


def test_a_commander_regenerates_without_waiting():
    world = scripted_world(CAT, FLAT)
    commander = world.add(COMMANDER, 0, 1000.0, 1000.0, hp=0.5)
    start = world.get(commander).hp
    world.step(int(2 * TICKS_PER_SECOND))
    assert abs(world.get(commander).hp - start - 10.0) < 0.5


def test_regeneration_stops_at_full_health():
    world = scripted_world(CAT, FLAT)
    unit = world.add("cloakraid", 0, 1000.0, 1000.0, hp=0.99)
    world.step(int(5 * TICKS_PER_SECOND))
    assert world.get(unit).hp == CAT.units[CAT.index["cloakraid"]]["hp"]


OVERDRIVE_SECOND = int(TICKS_PER_SECOND)
LOW = 1e-3


def _extra(income, energy):
    """unit_mex_overdrive.lua: energyToExtraM."""
    return income * 0.25 * math.sqrt(energy)


def _overdriven(world, ticks=2):
    """The first update comes at the first tick, and income shows it from the next."""
    world.step(ticks)
    return world.resources(0)


def test_an_extractor_linked_to_a_solar_takes_all_that_solar_makes_and_no_more():
    world = _economy(metal=0.0, energy=500.0, storage=500.0, innate=2.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    r = _overdriven(world)
    # 4 energy spare, all of it for overdrive at full storage, but the grid can only carry the solar's 2.
    assert abs(r["overdrive_energy"] - 2.0) < LOW and abs(r["overdrive_metal"] - _extra(2.0, 2.0)) < LOW, r
    assert abs(r["metal_income"] - (2.0 + 2.0 + _extra(2.0, 2.0))) < LOW
    assert abs(r["energy_income"] - (4.0 - 2.0)) < LOW, r


@pytest.mark.parametrize("gap, linked", ((149.0, True), (151.0, False)))
def test_a_link_reaches_the_sum_of_two_ranges(gap, linked):
    world = _economy(metal=0.0, energy=500.0, storage=500.0, innate=2.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1000.0 + gap, 1000.0)
    r = _overdriven(world)
    assert (r["overdrive_metal"] > 0.0) == linked, r
    assert (abs(r["energy_income"] - 4.0) > LOW) == linked, r


def test_overdrive_scales_with_how_full_storage_is():
    world = _economy(metal=0.0, energy=250.0, storage=500.0, innate=2.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    world.add("energysolar", 0, 900.0, 1000.0)
    world.step(1)
    # 6 made, and the first tick's income was already in storage when the update looked.
    drawn = 6.0 * (250.0 + 6.0 / TICKS_PER_SECOND) / 500.0
    r = _overdriven(world, 1)
    assert abs(r["overdrive_energy"] - drawn) < LOW and abs(r["overdrive_metal"] - _extra(2.0, drawn)) < LOW, r


def test_extractors_share_by_income_squared():
    spots = ((900.0, 1000.0, 1.0), (1100.0, 1000.0, 3.0))
    world = _economy(metal=0.0, energy=500.0, storage=500.0, spots=spots)
    world.add("staticmex", 0, 900.0, 1000.0)
    world.add("staticmex", 0, 1100.0, 1000.0)
    world.add("energypylon", 0, 1000.0, 1000.0)
    for k in range(10):
        world.add("energysolar", 0, 700.0 + 60.0 * k, 1200.0)
    r = _overdriven(world)
    # 20 energy over 1 and 9 squared.
    assert abs(r["overdrive_energy"] - 20.0) < LOW, r
    assert abs(r["overdrive_metal"] - (_extra(1.0, 2.0) + _extra(3.0, 18.0))) < LOW, r


def test_what_a_full_grid_cant_take_goes_to_the_other_grid():
    spots = ((1000.0, 1000.0, 2.0), (1800.0, 1000.0, 2.0))
    world = _economy(metal=0.0, energy=250.0, storage=500.0, innate=2.0, spots=spots)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1000.0, 1100.0)
    world.add("staticmex", 0, 1800.0, 1000.0)
    for k in range(5):
        angle = k * 2.0 * math.pi / 5.0
        world.add("energysolar", 0, 1800.0 + 100.0 * math.cos(angle), 1000.0 + 100.0 * math.sin(angle))
    world.step(1)
    # 14 made, half of it spare at half storage: 3.5 each, of which the first grid can carry 2.
    energy = 14.0 * (250.0 + 14.0 / TICKS_PER_SECOND) / 500.0
    r = _overdriven(world, 1)
    assert abs(r["overdrive_energy"] - energy) < LOW, r
    assert abs(r["overdrive_metal"] - (_extra(2.0, 2.0) + _extra(2.0, energy - 2.0))) < LOW, r


def test_every_grid_full_leaves_the_rest_in_storage():
    world = _economy(metal=0.0, energy=400.0, storage=500.0, innate=20.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    world.step(OVERDRIVE_SECOND + 1)
    r = world.resources(0)
    assert abs(r["overdrive_energy"] - 2.0) < LOW and abs(r["energy_income"] - 20.0) < LOW, r
    assert abs(r["energy"] - (400.0 + (22.0 + 20.0 * OVERDRIVE_SECOND) / TICKS_PER_SECOND)) < 0.01, r


def test_with_no_extractors_energy_only_collects():
    world = _economy(metal=0.0, energy=100.0, storage=500.0, innate=3.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    r = _overdriven(world, OVERDRIVE_SECOND + 2)
    assert r["overdrive_energy"] == 0.0 and r["energy_income"] == 5.0, r


def test_what_builders_spend_is_not_spare():
    spots = ((1000.0, 1000.0, 2.0), (1000.0, 1500.0, 2.0))
    world = _economy(metal=1000.0, energy=500.0, storage=500.0, spots=spots)
    world.add("staticmex", 0, 1000.0, 1000.0)
    for k in range(3):
        world.add("energysolar", 0, 900.0 + 100.0 * k, 1100.0)
    commander = world.add(COMMANDER, 0, 1000.0, 1450.0)
    world.step(1)
    assert world.resources(0)["overdrive_energy"] == 6.0 and world.get(commander).order == zksim.ORDER_NONE
    world.build(commander, "staticmex", 1000.0, 1500.0)
    world.step(2 * OVERDRIVE_SECOND)
    # 12 made, of which the commander is spending 10 on the extractor.  Its frame takes no share.
    r = world.resources(0)
    assert world.get(commander + 1).build < 1.0
    assert abs(r["overdrive_energy"] - 2.0) < 0.1, r
    assert abs(r["overdrive_metal"] - _extra(2.0, r["overdrive_energy"])) < LOW, r


def test_a_grid_that_loses_a_link_splits():
    world = _economy(metal=0.0, energy=500.0, storage=500.0, innate=2.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    pylon = world.add("energypylon", 0, 1300.0, 1000.0)
    world.add("energysolar", 0, 1700.0, 1000.0)
    assert _overdriven(world)["overdrive_metal"] > 0.0
    world.kill(pylon)
    world.step(OVERDRIVE_SECOND + 1)
    r = world.resources(0)
    assert r["overdrive_metal"] == 0.0 and r["overdrive_energy"] == 0.0, r


def test_an_unfinished_generator_neither_links_nor_carries():
    world = _economy(metal=0.0, energy=500.0, storage=500.0, innate=2.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0, build=0.5)
    r = _overdriven(world)
    assert r["overdrive_metal"] == 0.0 and r["energy_income"] == 2.0, r


def test_a_dead_extractor_no_longer_earns_its_overdrive():
    world = _economy(metal=0.0, energy=500.0, storage=500.0, innate=2.0)
    mex = world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    assert _overdriven(world)["metal_income"] > 4.0
    world.kill(mex)
    world.step(1)
    assert world.resources(0)["metal_income"] == 2.0


def test_a_builder_walks_into_reach_and_no_farther():
    world = _economy()
    commander = world.add(COMMANDER, 0, 200.0, 1000.0)
    world.build(commander, "staticmex", 1000.0, 1000.0)
    world.step(int(5 * TICKS_PER_SECOND))
    assert world.get(commander).order == zksim.ORDER_BUILD and world.get(commander).x > 300.0
    for _ in range(4000):
        world.step(1)
        if world.get(commander).order == zksim.ORDER_NONE:
            break
    assert world.get(commander + 1).build == 1.0
    reach = 144.0 + 24.0
    gap = 1000.0 - world.get(commander).x
    assert reach * 0.8 < gap <= reach + 8.0, gap


def test_a_factory_repeats_its_type_and_an_assistant_doubles_its_pace():
    def raiders_after(seconds, assisted):
        world = _economy(metal=50000.0, energy=50000.0)
        factory = world.add("factorycloak", 0, 1000.0, 1000.0)
        world.add(COMMANDER, 0, 1000.0, 1100.0)
        world.produce(factory, RAIDER)
        if assisted:
            world.assist(factory + 1, factory)
        world.step(int(seconds * TICKS_PER_SECOND))
        return sum(1 for u in world.view(0).units if u["def_id"] == CAT.units[CAT.index[RAIDER]]["def_id"] and not u["flags"] & rawframe.BEING_BUILT)
    assert raiders_after(20.0, False) == 3            # 65 at build speed 10: 6.5 s each
    assert raiders_after(20.0, True) == 6             # ... and 3.25 s with a commander of the same speed


def test_a_plate_makes_units_only_within_its_reach_of_a_finished_factory_of_its_type_and_side():
    """unit_factory_plate.lua: a plate's build range is 420.  Off, it makes nothing, and helping it does nothing."""
    def raiders_after(factory_x=None, factory="factorycloak", side=0, build=1.0, helped=False):
        world = _economy(metal=50000.0, energy=50000.0)
        plate = world.add("platecloak", 0, 1000.0, 1000.0)
        if factory_x is not None:
            world.add(factory, side, factory_x, 1000.0, build=build)
        if helped:
            world.assist(world.add(COMMANDER, 0, 1000.0, 1100.0), plate)
        world.produce(plate, RAIDER)
        world.step(int(20.0 * TICKS_PER_SECOND))
        return sum(1 for u in world.view(0).units if u["def_id"] == CAT.units[CAT.index[RAIDER]]["def_id"] and not u["flags"] & rawframe.BEING_BUILT)
    assert raiders_after(1300.0) == 3                 # as a factory: 6.5 s each
    assert raiders_after() == 0
    assert raiders_after(helped=True) == 0
    assert raiders_after(1500.0) == 0                 # 500 away
    assert raiders_after(1300.0, factory="factoryveh") == 0
    assert raiders_after(1300.0, side=1) == 0
    assert raiders_after(1300.0, build=0.5) == 0


def test_an_idle_caretaker_helps_a_factory_of_its_side_in_its_reach_and_else_a_frame():
    """unit_auto_patrol_nanos.lua, Zero-K's default: an idle caretaker patrols, which helps what's being built near it."""
    def raiders_after(caretaker_x, side=0):
        world = _economy(metal=50000.0, energy=50000.0)
        factory = world.add("factorycloak", 0, 1000.0, 1000.0)
        world.add("staticcon", side, caretaker_x, 1000.0)
        world.produce(factory, RAIDER)
        world.step(int(20.0 * TICKS_PER_SECOND))
        return sum(1 for u in world.view(0).units if u["def_id"] == CAT.units[CAT.index[RAIDER]]["def_id"] and not u["flags"] & rawframe.BEING_BUILT)
    assert raiders_after(1400.0) == 6                 # build speed 10, as the factory's: twice the pace
    assert raiders_after(1600.0) == 3                 # beyond its reach of 500, and the factory's footprint
    assert raiders_after(1400.0, side=1) == 3
    world = _economy()
    caretaker = world.add("staticcon", 0, 1000.0, 1000.0)
    frame = world.add("energysolar", 0, 1200.0, 1000.0, build=0.0)
    world.step(int(10.0 * TICKS_PER_SECOND))
    assert world.get(frame).build == 1.0 and world.get(caretaker).order == zksim.ORDER_NONE


def test_a_factory_stops_when_told():
    world = _economy(metal=50000.0, energy=50000.0)
    factory = world.add("factorycloak", 0, 1000.0, 1000.0)
    world.produce(factory, RAIDER)
    world.step(int(7 * TICKS_PER_SECOND))
    world.produce(factory, None)
    n = len(world.view(0).units)
    world.step(int(20 * TICKS_PER_SECOND))
    assert len(world.view(0).units) == n


def test_an_unfinished_unit_neither_walks_nor_shoots_and_then_does():
    world = _economy(metal=50000.0, energy=50000.0)
    factory = world.add("factorycloak", 0, 1000.0, 1000.0)
    world.produce(factory, RAIDER)
    world.step(int(1 * TICKS_PER_SECOND))
    raider = factory + 1
    x, z = world.get(raider).x, world.get(raider).z
    world.fight(raider, 3000.0, 1000.0)
    world.step(int(3 * TICKS_PER_SECOND))
    assert (world.get(raider).x, world.get(raider).z) == (x, z) and world.get(raider).build < 1.0
    world.step(int(10 * TICKS_PER_SECOND))
    assert world.get(raider).build == 1.0 and world.get(raider).x > x + 50.0


def test_a_new_unit_comes_out_on_ground_it_can_walk():
    rows, cols = 128, 128
    height = np.zeros((rows, cols), np.float32)
    speed = np.ones((len(zksim.MOVE_CLASSES), rows, cols), np.float32)
    speed[:, :, 66:] = 0.0                            # a wall just east of the factory, the way it faces
    cliff = zksim.Map(height, speed, 16.0, "cliff")
    world = scripted_world(CAT, cliff)
    world.economy(np.zeros((0, 3), np.float32), 50000.0, 50000.0, 0.0, 0.0, 100000.0)
    factory = world.add("factorycloak", 0, 1000.0, 1000.0, heading=math.pi / 2)
    world.produce(factory, RAIDER)
    world.step(1)
    frame = world.get(factory + 1)
    assert frame.x < 66 * 16.0 and math.hypot(frame.x - 1000.0, frame.z - 1000.0) > 56.0, (frame.x, frame.z)


def test_losing_the_frame_ends_the_order():
    world = _economy()
    commander = world.add(COMMANDER, 0, 1000.0, 900.0)
    world.build(commander, "staticmex", 1000.0, 1000.0)
    world.step(3)
    world.kill(commander + 1)
    world.step(2)
    assert world.get(commander).order == zksim.ORDER_NONE


def test_an_assistant_waits_for_its_factorys_next_unit():
    world = _economy(metal=50000.0, energy=50000.0)
    factory = world.add("factorycloak", 0, 1000.0, 1000.0)
    commander = world.add(COMMANDER, 0, 1000.0, 1100.0)
    world.assist(commander, factory)
    world.step(10)
    assert world.get(commander).order == zksim.ORDER_ASSIST
    assert world.view(0).glob[rawframe.G["idle_builders"]] == 0


def test_a_world_without_an_economy_is_as_it_was():
    world = scripted_world(CAT, FLAT)
    commander = world.add(COMMANDER, 0, 1000.0, 900.0)
    world.build(commander, "staticmex", 1000.0, 1000.0)
    world.step(30)
    assert len(world.view(0).units) == 1 and world.resources(0)["metal"] == 0.0


SECOND = 30 // zksim.TICK_FRAMES
CON = "cloakcon"
SEEING = (800.0, 800.0)     # near enough to everything these tests put at 1000, 1000


def _type(name):
    return CAT.units[CAT.index[name]]


def _at(world, x, z, side=0):
    """What this side sees lying at a point."""
    return [r for r in world.features(side) if r["x"] == x and r["z"] == z]


def _only_feature(world, side=0):
    seen = world.features(side)
    assert len(seen) == 1, seen
    return seen[0]


@pytest.mark.parametrize("name", ("cloakraid", "tankassault", "energysolar", "staticmex"))
def test_a_unit_that_dies_slowly_leaves_a_wreck_of_forty_percent_of_its_metal(name):
    world = scripted_world(CAT, FLAT)
    world.add(RAIDER, 0, *SEEING)
    victim = world.add(name, 1, 1000.0, 1000.0)
    world.kill(victim)
    world.step(1)
    left = _only_feature(world)
    assert left["metal"] == pytest.approx(0.4 * _type(name)["metal_cost"], rel=0.02)
    assert (left["x"], left["z"]) == (1000.0, 1000.0) and left["def_id"] == _type(name)["wreck_feature"]


def _shot_dead(shooter, hp):
    world = scripted_world(CAT, FLAT)
    gun = world.add(shooter, 0, 1000.0, 1000.0)
    world.ready_weapons()
    victim = world.add(RAIDER, 1, 1250.0, 1000.0, hp=hp)
    world.attack(gun, victim)
    for _ in range(40 * SECOND):
        world.step(1)
        if not world.get(victim).alive:
            return _only_feature(world)
    raise AssertionError("never died")


def test_a_big_hit_leaves_a_heap_and_small_ones_a_wreck():
    d = _type(RAIDER)
    assert _shot_dead("cloaksnipe", 1.0)["metal"] == pytest.approx(d["heap_metal"])         # 1500 damage to 230 hp
    assert _shot_dead(RAIDER, 0.02)["metal"] == pytest.approx(d["wreck_metal"])             # 10 damage to its last 5
    assert _shot_dead(RAIDER, 1.0)["metal"] == pytest.approx(d["wreck_metal"])              # recent damage fades by a tenth a frame, so even all of it


def test_a_frame_leaves_a_heap_in_proportion_and_one_barely_begun_nothing():
    world = scripted_world(CAT, FLAT)
    world.add(RAIDER, 0, *SEEING)
    frame = world.add(RAIDER, 1, 1000.0, 1000.0, build=0.5)
    barely = world.add(RAIDER, 1, 1100.0, 1000.0, build=0.03)
    world.kill(frame)
    world.kill(barely)
    world.step(1)
    assert _only_feature(world)["metal"] == pytest.approx(0.5 * _type(RAIDER)["heap_metal"])


@pytest.mark.parametrize("build, hurts", ((0.5, False), (0.9, True)))
def test_a_frame_only_explodes_once_nearly_done(build, hurts):
    world = scripted_world(CAT, FLAT)
    frame = world.add("energysingu", 1, 1000.0, 1000.0, build=build)
    near = world.add(RAIDER, 0, 1100.0, 1000.0)
    world.kill(frame)
    world.step(2)
    assert bool(world.get(near).alive) != hurts


def test_a_wreck_blown_up_leaves_a_fresh_heap_and_a_heap_nothing():
    world = scripted_world(CAT, FLAT)
    d = _type(RAIDER)
    world.kill(world.add(RAIDER, 1, 1000.0, 1000.0))
    world.step(1)
    world.kill(world.add("energysingu", 1, 1000.0, 1200.0))
    world.step(2)
    world.add(RAIDER, 0, *SEEING)
    heap = _at(world, 1000.0, 1000.0)
    assert len(heap) == 1 and heap[0]["metal"] == pytest.approx(d["heap_metal"]) and heap[0]["def_id"] == d["heap_feature"]
    world.kill(world.add("energysingu", 1, 1000.0, 1200.0))
    world.step(2)
    world.add(RAIDER, 0, *SEEING)
    assert not _at(world, 1000.0, 1000.0)


def test_what_was_taken_from_a_wreck_is_gone_from_its_heap():
    world = _economy(metal=0.0)
    d = _type("tankassault")
    world.kill(world.add("tankassault", 1, 1000.0, 1000.0))
    world.step(1)
    con = world.add(CON, 0, 1060.0, 1000.0)
    world.reclaim(con, int(_only_feature(world)["feat_id"]))
    world.step(10 * SECOND)
    taken = 1.0 - float(_only_feature(world)["metal"]) / d["wreck_metal"]
    assert 0.05 < taken < 0.5
    world.kill(con)
    world.kill(world.add("energysingu", 1, 1000.0, 1100.0))
    world.step(2)
    world.add(RAIDER, 0, *SEEING)
    assert _at(world, 1000.0, 1000.0)[0]["metal"] == pytest.approx(d["heap_metal"] * (1.0 - taken), rel=0.02)


def test_a_feature_no_blast_reaches_lasts():
    world = scripted_world(CAT, FLAT)
    world.add(RAIDER, 0, *SEEING)
    world.add_feature(1000.0, 1000.0, 50.0, hp=1.0)
    world.step(30 * SECOND)
    assert _only_feature(world)["metal"] == 50.0


# ---------------------------------------------------------------- reclaiming it

def test_a_builder_takes_metal_at_its_own_speed_as_income_until_there_is_none():
    world = _economy(metal=0.0)
    con = world.add(CON, 0, 1000.0, 1000.0)
    wreck = world.add_feature(1050.0, 1000.0, 20.0)
    world.reclaim(con, wreck)
    world.step(SECOND)                                      # its order takes a moment to start
    before = world.resources(0)["metal"]
    world.step(2 * SECOND)
    speed = _type(CON)["reclaim_speed"]
    assert world.resources(0)["metal"] - before == pytest.approx(2 * speed, abs=0.2)
    assert world.resources(0)["metal_income"] == pytest.approx(speed, abs=0.01)
    # Shrinks as it goes, in what a side sees of it.  This tick's is paid at the next.
    assert _only_feature(world)["metal"] == pytest.approx(20.0 - world.resources(0)["metal"] - speed / SECOND, abs=0.01)
    world.step(2 * SECOND)
    assert not len(world.features(0)) and world.get(con).order == zksim.ORDER_NONE
    assert world.resources(0)["metal"] == pytest.approx(20.0, abs=0.01)
    world.step(2)
    assert world.resources(0)["metal_income"] == 0.0


def test_builders_on_one_wreck_add_up():
    world = _economy(metal=0.0)
    wreck = world.add_feature(1050.0, 1000.0, 100.0)
    for z in (1000.0, 1040.0):
        world.reclaim(world.add(CON, 0, 1000.0, z), wreck)
    world.step(SECOND)
    before = world.resources(0)["metal"]
    world.step(2 * SECOND)
    assert world.resources(0)["metal"] - before == pytest.approx(4 * _type(CON)["reclaim_speed"], abs=0.2)


def test_what_storage_cant_hold_is_lost_with_the_wreck():
    world = _economy(metal=0.0, storage=3.0)
    con = world.add(CON, 0, 1000.0, 1000.0)
    world.reclaim(con, world.add_feature(1050.0, 1000.0, 10.0))
    world.step(4 * SECOND)
    assert world.resources(0)["metal"] == pytest.approx(3.0) and not len(world.features(0))


def test_a_builder_walks_to_a_wreck():
    world = _economy(metal=0.0)
    con = world.add(CON, 0, 500.0, 1000.0)
    world.reclaim(con, world.add_feature(1500.0, 1000.0, 10.0))
    world.step(SECOND)
    assert world.resources(0)["metal"] == 0.0
    world.step(60 * SECOND)
    assert world.resources(0)["metal"] == pytest.approx(10.0, abs=0.01) and world.get(con).x < 1500.0 - _type(CON)["build_range"] / 2


def test_either_side_may_take_a_wreck_and_a_factory_or_a_fighter_may_not():
    world = _economy(metal=0.0)
    wreck = world.add_feature(1100.0, 1000.0, 100.0)
    world.reclaim(world.add(RAIDER, 0, 1050.0, 1000.0), wreck)
    world.reclaim(world.add("factorycloak", 0, 1100.0, 1100.0), wreck)
    world.step(3 * SECOND)
    assert world.resources(0)["metal"] == 0.0 and world.resources(1)["metal"] == 0.0
    world.reclaim(world.add(CON, 1, 1150.0, 1000.0), wreck)
    world.step(3 * SECOND)
    assert world.resources(0)["metal"] == 0.0 and world.resources(1)["metal"] > 0.0


# ---------------------------------------------------------------- seeing it

def test_a_side_sees_only_the_wrecks_in_its_line_of_sight():
    world = scripted_world(CAT, sight_map())
    world.add(RAIDER, 0, 800.0, 2000.0)
    world.add(RAIDER, 1, 3500.0, 2000.0)
    near = world.add_feature(800.0, 1600.0, 50.0)
    behind_a_ridge = world.add_feature(1300.0, 2000.0, 60.0)
    out_of_range = world.add_feature(800.0, 3000.0, 70.0)
    theirs = world.add_feature(3300.0, 2000.0, 80.0)
    assert {int(r["feat_id"]) for r in world.features(0)} == {near}
    assert {int(r["feat_id"]) for r in world.features(1)} == {theirs}
    assert {int(r["feat_id"]) for r in world.view(0).features} == {near}
    # Out of sight stays out of sight, however it is otherwise known.
    assert behind_a_ridge not in {int(r["feat_id"]) for r in world.features(1)} and out_of_range not in {int(r["feat_id"]) for r in world.features(1)}
    # Something unfinished looks at nothing.
    other = scripted_world(CAT, FLAT)
    other.add(RAIDER, 0, 1000.0, 1000.0, build=0.5)
    other.add_feature(1000.0, 1100.0, 50.0)
    assert not len(other.features(0))


def test_a_batch_brings_each_side_its_own_features():
    batch = zksim.Batch(CAT, 2, threads=1)
    for game in (0, 1):
        side, x, z = batch_game(batch, game, 7, FLAT)
    ours, others = np.flatnonzero(side == 0)[0], np.flatnonzero(side == 1)[0]
    batch.add_features(1, np.array([[x[ours], z[ours], 40.0, 100.0, 16.0, 5.0], [x[others], z[others], 60.0, 100.0, 16.0, 6.0]], np.float32))
    batch.step()
    assert not len(batch.frame(0, 0).features)
    mine, theirs = batch.frame(1, 0).features, batch.frame(1, 1).features
    assert [float(r["metal"]) for r in mine] == [40.0] and int(mine[0]["def_id"]) == 5
    assert [float(r["metal"]) for r in theirs] == [60.0]
    batch.refresh(1)
    assert [float(r["metal"]) for r in batch.frame(1, 0).features] == [40.0]


# ---------------------------------------------------------------- upkeep

def _spent(world, seconds):
    before = world.resources(0)["energy"]
    world.step(seconds * SECOND)
    return before - world.resources(0)["energy"]


def test_upkeep_is_a_charge_in_energy_each_second():
    world = _economy()
    world.add("staticradar", 0, 1000.0, 1000.0)
    metal = world.resources(0)["metal"]
    assert _spent(world, 10) == pytest.approx(10 * _type("staticradar")["energy_upkeep"], abs=0.81)
    assert world.resources(0)["metal"] == metal


def test_an_unfinished_radar_costs_nothing():
    world = _economy()
    world.add("staticradar", 0, 1000.0, 1000.0, build=0.5)
    assert _spent(world, 5) == 0.0


def test_a_radar_nobody_pays_for_goes_dark_until_paid_again():
    world = _economy(energy=2.0)
    world.add("staticradar", 0, 500.0, 500.0)
    enemy = world.add(RAIDER, 1, 2000.0, 500.0)
    assert records(world.view(0))[enemy]["knowledge"] == rawframe.KNOW_RADAR
    world.step(5 * SECOND)
    assert records(world.view(0))[enemy]["knowledge"] == rawframe.KNOW_HIDDEN
    world.add("energysolar", 0, 500.0, 600.0)
    world.step(5 * SECOND)
    assert records(world.view(0))[enemy]["knowledge"] == rawframe.KNOW_RADAR


# ---------------------------------------------------------------- cloak

SHY = "cloakheavyraid"
SOLAR = "energysolar"


def test_a_cloaked_unit_is_gone_from_sight_and_radar_but_not_from_its_owner():
    world = scripted_world(CAT, FLAT)
    hider = world.add(SHY, 1, 1000.0, 1000.0)
    watcher = world.add(RAIDER, 0, 1300.0, 1000.0)
    assert world.get(hider).cloaked and records(world.view(0))[hider]["knowledge"] == rawframe.KNOW_HIDDEN
    world.step(3)
    assert world.get(hider).cloaked and world.get(hider).alive and world.get(watcher).alive
    assert records(world.view(0))[hider]["knowledge"] == rawframe.KNOW_HIDDEN
    own = records(world.view(1))[hider]
    assert own["flags"] & rawframe.CLOAKED and own["knowledge"] == rawframe.KNOW_LOS
    world.add("staticradar", 0, 1300.0, 1100.0)
    assert records(world.view(0))[hider]["knowledge"] == rawframe.KNOW_HIDDEN
    # Shown, it's on both.
    world.cloak(hider, False)
    world.step(3)
    assert records(world.view(0))[hider]["knowledge"] == rawframe.KNOW_LOS


def test_cloak_costs_more_on_the_move_and_nothing_when_off():
    d = _type(SHY)
    world = _economy()
    unit = world.add(SHY, 0, 500.0, 1000.0)
    assert _spent(world, 10) == pytest.approx(10 * d["cloak_cost"], abs=0.2)
    world.move(unit, 3500.0, 1000.0)
    world.step(SECOND)
    assert _spent(world, 10) == pytest.approx(10 * d["cloak_cost_moving"], abs=1.0)
    world.cloak(unit, False)
    world.step(SECOND)
    assert not world.get(unit).cloaked and _spent(world, 5) == 0.0
    world.cloak(unit, True)
    world.step(SECOND)
    assert world.get(unit).cloaked


def test_no_energy_no_cloak():
    world = _economy(energy=0.0)
    unit = world.add(SHY, 0, 1000.0, 1000.0)
    world.step(3 * SECOND)
    assert not world.get(unit).cloaked


def _frames_until_cloaked(world, unit, most=600):
    start = world.frame
    while world.frame - start < most:
        if world.get(unit).cloaked:
            return world.frame - start
        world.step(1)
    raise AssertionError("stayed visible")


def test_an_enemy_close_by_shows_a_cloaked_unit_and_it_hides_again_soon_after():
    world = scripted_world(CAT, FLAT)
    hider = world.add("jumpbomb", 1, 1000.0, 1000.0)
    world.step(SECOND)
    assert world.get(hider).cloaked
    near = world.add(CON, 0, 1000.0 + _type("jumpbomb")["decloak_distance"] - 20.0, 1000.0)
    world.step(SECOND)
    assert not world.get(hider).cloaked
    world.kill(near)
    world.step(1)
    # 45 frames from the last slow update that saw it, and the next such update to act on it.
    assert 30 <= _frames_until_cloaked(world, hider) <= 70


def test_a_hit_shows_a_cloaked_unit_for_three_seconds():
    world = scripted_world(CAT, FLAT)
    hider = world.add(SHY, 1, 1000.0, 1000.0)
    world.step(SECOND)
    world.kill(world.add(SOLAR, 1, 1050.0, 1000.0))      # a blast has no need to see it
    world.step(1)
    assert world.get(hider).damage_taken > 0.0 and not world.get(hider).cloaked
    assert 70 <= _frames_until_cloaked(world, hider) <= 120


def test_nothing_shoots_what_it_cant_see_radar_or_not():
    world = scripted_world(CAT, FLAT)
    hider = world.add(SHY, 1, 1000.0, 1000.0)
    world.step(SECOND)
    gun = world.add(RAIDER, 0, 1150.0, 1000.0)
    world.add("staticradar", 0, 1300.0, 1100.0)
    world.ready_weapons()
    world.attack(gun, hider)
    world.step(10 * SECOND)
    assert world.get(hider).damage_taken == 0.0
    # Shown, it can be.
    world.cloak(hider, False)
    world.attack(gun, hider)
    world.step(10 * SECOND)
    assert world.get(hider).damage_taken > 0.0


def test_firing_shows_a_unit_unless_it_is_made_to_fire_hidden():
    world = scripted_world(CAT, FLAT)
    sniper = world.add("cloaksnipe", 0, 1000.0, 1000.0)
    world.step(SECOND)
    assert world.get(sniper).cloaked
    enemy = world.add("tankassault", 1, 1500.0, 1000.0)
    world.ready_weapons()
    world.attack(sniper, enemy)
    world.step(3 * SECOND)
    assert world.get(enemy).damage_taken > 0.0 and world.get(sniper).cloaked


def _under_area_cloak(energy=100000.0, enemy_at=1400.0):
    world = _economy(energy=energy)
    cloaker = world.add("cloakjammer", 0, 900.0, 1000.0)
    friend = world.add(RAIDER, 0, 1000.0, 1000.0)
    enemy = world.add(RAIDER, 1, enemy_at, 1000.0)
    return world, cloaker, friend, enemy


def test_an_area_cloak_hides_friends_and_charges_for_it():
    world, cloaker, friend, enemy = _under_area_cloak()
    world.step(SECOND)
    assert world.get(friend).cloaked and records(world.view(1))[friend]["knowledge"] == rawframe.KNOW_HIDDEN
    d = _type("cloakjammer")
    assert _spent(world, 10) == pytest.approx(10 * (d["energy_upkeep"] + d["area_cloak_upkeep"]), abs=17.0)
    world.cloak(cloaker, False)
    world.step(SECOND)
    assert not world.get(friend).cloaked and records(world.view(1))[friend]["knowledge"] == rawframe.KNOW_LOS
    assert _spent(world, 10) == pytest.approx(10 * d["energy_upkeep"], abs=1.6)


def test_an_area_cloak_nobody_pays_for_hides_nothing():
    world, _, friend, _ = _under_area_cloak(energy=20.0)
    world.step(5 * SECOND)
    assert not world.get(friend).cloaked


def test_firing_from_under_an_area_cloak_shows_a_unit_for_eight_seconds():
    world, _, friend, enemy = _under_area_cloak(enemy_at=1140.0)
    world.ready_weapons()
    world.step(SECOND)
    assert world.get(enemy).damage_taken > 0.0 and not world.get(friend).cloaked
    world.kill(enemy)
    world.step(1)
    assert 200 <= _frames_until_cloaked(world, friend) <= 270


def test_a_new_commander_has_no_cloak():
    assert not any(u["flags"] & zksim.CAN_CLOAK and u["flags"] & zksim.COMMANDER for u in CAT.units)


# ---------------------------------------------------------------- stockpiling

STOCKER = "turretaaheavy"


def test_a_stockpile_charges_both_resources_as_it_goes():
    d = _type(STOCKER)
    world = _economy()
    unit = world.add(STOCKER, 0, 1000.0, 1000.0)
    start, stock = world.resources(0), world.get(unit).stock
    world.step(5 * SECOND)
    now = world.resources(0)
    charge = d["stockpile_cost"] * 150.0 / d["stockpile_frames"]
    assert start["metal"] - now["metal"] == pytest.approx(charge, rel=0.05) and start["energy"] - now["energy"] == pytest.approx(charge, rel=0.05)
    world.step(int(d["stockpile_frames"] / zksim.TICK_FRAMES) + 1)
    assert world.get(unit).stock == stock + 1


@pytest.mark.parametrize("metal, energy", ((0.0, 1000.0), (1000.0, 0.0)))
def test_a_stockpile_short_of_either_resource_takes_neither(metal, energy):
    world = _economy(metal=metal, energy=energy)
    unit = world.add(STOCKER, 0, 1000.0, 1000.0)
    stock = world.get(unit).stock
    world.step(int(_type(STOCKER)["stockpile_frames"] / zksim.TICK_FRAMES) + 1)
    assert world.get(unit).stock == stock
    now = world.resources(0)
    assert (now["metal"], now["energy"]) == (metal, energy)


# ---------------------------------------------------------------- a solar under fire

SOLAR_REOPENS_AFTER = 120 + 38


def _solar_under_fire(world):
    solar = world.add(SOLAR, 0, 1000.0, 1000.0)
    gun = world.add(RAIDER, 1, 1200.0, 1000.0)
    world.ready_weapons()
    world.attack(gun, solar)
    return solar, gun


def _until_hit(world, unit, most=200):
    for _ in range(most):
        world.step(1)
        if world.get(unit).damage_taken > 0.0:
            return
    raise AssertionError("never hit")


def test_a_solar_that_is_shot_stops_making_energy_and_opens_again_when_left_alone():
    world = _economy()
    world.step(1)
    solar, gun = _solar_under_fire(world)
    world.step(1)
    assert world.resources(0)["energy_income"] == _type(SOLAR)["energy_make"]
    _until_hit(world, solar)
    world.step(1)
    assert world.resources(0)["energy_income"] == 0.0
    # From the last hit, whichever that was: some are still in the air.
    world.kill(gun)
    last, damage = world.frame, world.get(solar).damage_taken
    while world.resources(0)["energy_income"] == 0.0:
        world.step(1)
        if world.get(solar).damage_taken != damage:
            last, damage = world.frame, world.get(solar).damage_taken
    assert world.resources(0)["energy_income"] == _type(SOLAR)["energy_make"]
    assert world.frame - last == pytest.approx(SOLAR_REOPENS_AFTER, abs=2 * zksim.TICK_FRAMES)


def test_a_solar_only_takes_a_third_once_shut():
    world = _economy()
    solar, _ = _solar_under_fire(world)
    _until_hit(world, solar)
    hp = world.get(solar).hp
    # Shutting takes 19 frames, under fire or not: full damage until then.
    world.step(3)
    open_rate = (hp - world.get(solar).hp) / 3
    world.step(3)
    hp = world.get(solar).hp
    world.step(12)
    shut_rate = (hp - world.get(solar).hp) / 12
    assert open_rate > 0.0 and shut_rate == pytest.approx(open_rate * _type(SOLAR)["armored_mult"], rel=0.1)


def test_nothing_else_that_makes_energy_shuts():
    world = _economy()
    fusion = world.add("energyfusion", 0, 1000.0, 1000.0)
    gun = world.add(RAIDER, 1, 1200.0, 1000.0)
    world.ready_weapons()
    world.attack(gun, fusion)
    _until_hit(world, fusion)
    world.step(2)
    assert world.resources(0)["energy_income"] == _type("energyfusion")["energy_make"]


def test_a_shut_solar_gives_its_grid_no_capacity():
    world = _economy(storage=100.0, energy=100.0, spots=((1000.0, 1000.0, 2.0),))
    world.add("staticmex", 0, 1000.0, 1000.0)
    solar = world.add(SOLAR, 0, 1040.0, 1000.0)
    world.step(2)
    assert world.resources(0)["overdrive_energy"] == pytest.approx(_type(SOLAR)["energy_make"])
    gun = world.add(RAIDER, 1, 1240.0, 1000.0)
    world.ready_weapons()
    world.attack(gun, solar)
    _until_hit(world, solar)
    world.step(2 * SECOND)
    assert world.resources(0)["overdrive_energy"] == 0.0 and world.resources(0)["overdrive_metal"] == 0.0


def main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                failed += 1
                print("FAIL", name, e)
    print("failed" if failed else "all passed", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
