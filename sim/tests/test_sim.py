"""Checks of the simulator against formulas from the engine source.  In WSL:

  ~/zkml/bin/python sim/tests/test_sim.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402
from zkbot.policy import rawframe  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
RAIDER = "cloakraid"
ORDER_DELAY = 15            # a move order waits for the unit's slow update
STATIC = "staticmex"
ARMORY = ("cloakraid", "cloakassault", "shieldraid", "tankassault", "spiderassault", "jumpraid", "hoverraid", "vehassault")


def test_top_speed_frames():
    w = zksim.World(CAT, FLAT)
    u = w.add(RAIDER, 0, 100, 2000, heading=math.pi / 2)
    w.move(u, 3000, 2000)
    d = CAT.units[CAT.index[RAIDER]]
    frames_needed = math.ceil(d["speed"] / d["accel"])
    w.step(math.ceil((frames_needed + ORDER_DELAY) / zksim.TICK_FRAMES))
    assert abs(w.get(u).speed - d["speed"]) < 1e-3


def test_straight_line_arrival():
    w = zksim.World(CAT, FLAT)
    u = w.add(RAIDER, 0, 100, 2000, heading=math.pi / 2)
    w.move(u, 2100, 2000)
    d = CAT.units[CAT.index[RAIDER]]
    w.step(int((2000 / d["speed"] + ORDER_DELAY) / zksim.TICK_FRAMES) + 3)
    s = w.get(u)
    assert abs(s.x - 2100) <= 20 and s.speed < d["speed"], (s.x, s.speed)


def test_turn_around_frames():
    d = CAT.units[CAT.index[RAIDER]]
    w = zksim.World(CAT, FLAT)
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
            "auto_heal": 0.0, "max_range": 0.0, "armor_class": 0, "move_class": -1, "category": every,
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
        w = zksim.World(_blast_catalog(edge), FLAT, behaviour=zksim.HOLD)
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
    options = [u for u in CAT.units if not u["flags"] and u["hp"] > 2000.0 and u["armor_class"] == 2 and u["category"] & can_hit]
    return min(options, key=lambda u: u["hp"])["name"]


def test_salvo_damage_rate():
    """The raider's salvo: three shots per 9-frame reload, so 10 shots per second at point blank."""
    w = zksim.World(CAT, FLAT, behaviour=zksim.HOLD)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
        w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
    a = w.add("vehcapture", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1800.0, 2000.0)
    w.ready_weapons()
    w.attack(a, t)
    w.step(36)
    assert w.get(t).damage_taken == 0.0
    flagged = {u["name"] for u in CAT.units if any(CAT.weapons[m]["flags"] & zksim.W_UNMODELLED for m in u["mount_weapon"])}
    assert {"vehcapture", "veharty", "hoverminer", "jumpblackhole", "amphlaunch"} <= flagged, flagged


def test_a_manual_weapon_waits_for_its_order_and_its_stock():
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
    reef = w.add("shipcarrier", 0, 1500.0, 2000.0)
    t = w.add("cloakassault", 1, 1500.0, 3500.0)
    w.ready_weapons()
    w.step(6 * 20)
    assert w.get(reef).stock >= 1 and w.get(t).disarm == 0.0
    w.attack(reef, t)
    w.step(6 * 20)
    assert w.get(t).disarm == 0.0, "an attack order doesn't fire it either"


def test_a_unit_without_a_manual_weapon_ignores_the_order():
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
    a = w.add("cloakraid", 0, 1500.0, 2000.0)
    t = w.add("cloakraid", 1, 1800.0, 2000.0)
    w.manual_fire(a, t)
    w.step(1)
    assert w.get(a).order == zksim.ORDER_NONE


def test_a_stun_stops_the_stockpile():
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
    arty = w.add("shieldarty", 0, 1500.0, 2000.0)
    reef = w.add("shipcarrier", 1, 2100.0, 2000.0)
    w.attack(arty, reef)
    w.step(6 * 31)
    assert w.get(reef).stock == 0 or w.get(reef).disarm <= 1.0, (w.get(reef).stock, w.get(reef).disarm)


def test_heat_rays_fade_with_distance():
    taken = {}
    for gap in (60.0, 200.0):
        w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.HOLD)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
        w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.ORDERS)
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
    w = zksim.World(CAT, FLAT, seed=3, behaviour=zksim.HOLD)
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
    w = zksim.World(CAT, FLAT, behaviour=zksim.HOLD)
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
        w = zksim.World(CAT, FLAT, seed=1, behaviour=zksim.ORDERS)
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
        w = zksim.World(CAT, FLAT)
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
        w = zksim.World(CAT, hill)
        u = w.add(RAIDER, 0, 2000, start_z, heading=0.0 if goal_z > start_z else math.pi)
        w.move(u, 2000, goal_z)
        w.step(60)
        return abs(w.get(u).z - start_z)
    assert walk(1000, 3000) < walk(3000, 1000) * 0.9, (walk(1000, 3000), walk(3000, 1000))


def test_unbuilt_does_not_fire():
    w = zksim.World(CAT, FLAT, behaviour=zksim.HOLD)
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
    w = zksim.World(CAT, world_map)
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
    w = zksim.World(CAT, world_map)
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
    w = zksim.World(CAT, world_map)
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
    w = zksim.World(CAT, world_map)
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
    w = zksim.World(CAT, FLAT, seed=seed, behaviour=zksim.ADVANCE)
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
    w = zksim.World(CAT, zksim.Map.load(zksim.DATA / "maps/titanduel_2_2.npz"), behaviour=zksim.ADVANCE)
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
    w = zksim.World(CAT, FLAT)
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
    w = zksim.World(CAT, sight_map())
    w.add(RAIDER, 0, 800.0, 2000.0)
    behind = w.add(RAIDER, 1, 1300.0, 2000.0)
    same_range = w.add(RAIDER, 1, 800.0, 1500.0)
    r = records(w.view(0))
    assert r[behind]["knowledge"] == rawframe.KNOW_HIDDEN
    assert r[same_range]["knowledge"] == rawframe.KNOW_LOS


def test_radar_gives_a_blip_with_identity_only_if_seen_since():
    w = zksim.World(CAT, FLAT)
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
    w2 = zksim.World(CAT, FLAT)
    radar = w2.add("staticradar", 0, 500.0, 500.0)
    b = w2.add(RAIDER, 1, 600.0, 500.0)
    assert records(w2.view(0))[b]["knowledge"] == rawframe.KNOW_LOS
    w2.kill(radar)
    w2.step(1)
    assert records(w2.view(0))[b]["knowledge"] == rawframe.KNOW_HIDDEN
    assert w2.get(b).alive


def test_a_blip_top_speed_is_the_speed_it_has_held():
    w = zksim.World(CAT, FLAT)
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
    w = zksim.World(CAT, FLAT)
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
    w = zksim.World(CAT, FLAT)
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
    w3 = zksim.World(CAT, FLAT)
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
