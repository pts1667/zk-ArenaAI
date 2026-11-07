"""What some units do to others: grids that run weapons, pushing and pulling, throwing, silos and teleporting.  Measured against a game in 01-simulator.md section 8.17."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 1500.0, 2000.0
TOUGH = 1000000.0
FRAMES = zksim.TICK_FRAMES
NEWTON, SUMO, LOBSTER, DJINN, BEACON, SILO, TACTICAL, CERBERUS, FUSION, SOLAR = (
    "turretimpulse", "jumpsumo", "amphlaunch", "amphtele", "tele_beacon", "staticmissilesilo", "tacnuke", "staticarty", "energyfusion", "energysolar")
GLAIVE, KNIGHT, DETRITER = "cloakraid", "cloakassault", "striderdetriment"


def world(economy: bool = False) -> zksim.World:
    w = zksim.World(CAT, FLAT, seed=1)
    if economy:
        zksim._set_economy(w.handle, np.zeros((0, 3), np.float32), 100000.0, 100000.0, 0.0, 0.0, 100000.0)
    return w


def dummy(w: zksim.World, name: str, dx: float = 0.0, dz: float = 0.0, side: int = 1, hp: float = TOUGH, heading: float = 0.0) -> int:
    u = w.add(name, side, X + dx, Z + dz, heading)
    w.set_fire_state(u, 0)
    w.set_move_state(u, 0)
    w.set_unit_ai(u, False)
    w.set_hp(u, hp)
    return u


def seconds(w: zksim.World, s: float) -> None:
    w.step(int(s * 30 / FRAMES))


def apart(w: zksim.World, a: int, b: int) -> float:
    sa, sb = w.get(a), w.get(b)
    return math.hypot(sa.x - sb.x, sa.z - sb.z)


# Grids ---------------------------------------------------------------------------------------------------------------------------------------------------------


def test_what_needs_a_grid_goes_by_its_def():
    assert CAT.units[CAT.index[CERBERUS]]["needed_link"] == 50.0
    assert CAT.units[CAT.index["turretheavy"]]["needed_link"] > 0.0
    assert CAT.units[CAT.index[KNIGHT]]["needed_link"] == 0.0


def cerberus_on(grid: list[tuple[str, float]]) -> tuple[zksim.World, int, int]:
    """With what's listed, by how far along from it.  A spotter by the target: a Cerberus sees no further than it needs to here."""
    w = world(economy=True)
    gun = w.add(CERBERUS, 0, X, Z)
    w.set_unit_ai(gun, False)
    for name, along in grid:
        w.add(name, 0, X + along, Z)
    target = dummy(w, KNIGHT, 0.0, 1000.0)
    spotter = w.add("cloakcon", 0, X, Z + 950.0)
    w.set_fire_state(spotter, 0)
    return w, gun, target


def test_a_cerberus_with_no_grid_holds_its_fire():
    w, gun, target = cerberus_on([])
    seconds(w, 15.0)
    assert w.get(gun).low_power == 1
    assert w.get(target).damage_taken == 0.0


def test_a_small_grid_is_not_enough():
    w, gun, target = cerberus_on([(SOLAR, 90.0)])
    seconds(w, 15.0)
    assert w.get(gun).low_power == 1
    assert w.get(target).damage_taken == 0.0


def test_a_grid_that_makes_enough_runs_it():
    w, gun, target = cerberus_on([(FUSION, 90.0), (FUSION, 100.0)])
    seconds(w, 15.0)
    assert w.get(gun).low_power == 0
    assert w.get(target).damage_taken > 0.0


def test_generators_out_of_reach_are_no_use():
    w, gun, target = cerberus_on([(FUSION, 1500.0), (FUSION, 1510.0)])
    seconds(w, 15.0)
    assert w.get(gun).low_power == 1


def test_losing_the_generators_turns_it_off():
    w, gun, _ = cerberus_on([(FUSION, 90.0), (FUSION, 100.0)])
    seconds(w, 3.0)
    assert w.get(gun).low_power == 0
    w.kill(gun + 1)
    w.kill(gun + 2)
    seconds(w, 3.0)
    assert w.get(gun).low_power == 1


def test_a_world_without_an_economy_has_no_grids_to_ask():
    w = world()
    gun = w.add(CERBERUS, 0, X, Z)
    target = dummy(w, KNIGHT, 0.0, 1000.0)
    w.add("cloakcon", 0, X, Z + 950.0)
    seconds(w, 15.0)
    assert w.get(gun).low_power == 0
    assert w.get(target).damage_taken > 0.0


# Pushing and pulling -------------------------------------------------------------------------------------------------------------------------------------------


def test_pushing_does_no_damage_and_counts_as_armed():
    for name in (NEWTON, SUMO):
        u = CAT.units[CAT.index[name]]
        assert u["flags"] & zksim.ARMED
        assert u["max_range"] > 0.0
    assert not CAT.weapon_of(NEWTON, 0)["flags"] & zksim.W_UNMODELLED
    assert max(CAT.weapon_of(NEWTON, 0)["damage"][:4]) == 0.0
    assert CAT.weapon_of(NEWTON, 0)["impulse"] != 0.0


def test_a_newton_throws_a_glaive_back_and_it_stops_past_its_range():
    # Facing it, 300 away: at 555 by 60 frames in a game, and no higher than a skid.  Here it's less, for a beam that misses what's moving fast between ticks.
    w = world()
    w.add(NEWTON, 0, X, Z)
    target = dummy(w, GLAIVE, 0.0, 300.0, heading=math.pi)
    seconds(w, 3.0)
    gone = w.get(target).z - Z
    assert 400.0 < gone < 700.0
    assert w.get(target).damage_taken == 0.0
    assert w.get(target).altitude < 1.0


def test_a_knight_facing_it_stays_where_it_is():
    # 300 in a game after 150 frames, against friction.
    w = world()
    w.add(NEWTON, 0, X, Z)
    target = dummy(w, KNIGHT, 0.0, 300.0, heading=math.pi)
    seconds(w, 5.0)
    assert abs(w.get(target).z - Z - 300.0) < 5.0


def test_what_faces_the_way_it_is_pushed_drifts_by_as_much_as_each_hit():
    # A game: 1 elmo a frame for a Glaive, 0.22 for a Knight, 0.11 for a Ravager, 0.04 for a Sumo.
    for name, low, high in ((GLAIVE, 0.6, 1.3), (KNIGHT, 0.14, 0.32), ("tankassault", 0.05, 0.2), ("jumpsumo", 0.01, 0.08)):
        w = world()
        w.add(NEWTON, 0, X, Z)
        target = dummy(w, name, 0.0, 300.0)
        seconds(w, 3.0)
        rate = (w.get(target).z - Z - 300.0) / 90.0
        assert low < rate < high, name


def test_off_it_pulls_and_a_light_unit_is_flung_about_and_hurt():
    # A game: past the gun by 45 frames, with a hard landing.
    w = world()
    newton = w.add(NEWTON, 0, X, Z)
    w.set_on_off(newton, False)
    target = dummy(w, GLAIVE, 0.0, 400.0)
    seconds(w, 3.5)
    assert w.get(newton).pulling == 1
    assert w.get(target).z < Z + 150.0
    seconds(w, 6.5)
    assert w.get(target).damage_taken > 100.0


def test_pulling_a_knight_does_next_to_nothing():
    w = world()
    newton = w.add(NEWTON, 0, X, Z)
    w.set_on_off(newton, False)
    target = dummy(w, KNIGHT, 0.0, 400.0)
    seconds(w, 10.0)
    assert abs(w.get(target).z - Z - 400.0) < 10.0


def test_a_heavy_unit_hardly_moves_where_a_light_one_flies():
    results = {}
    for name in (GLAIVE, DETRITER):
        w = world()
        w.add(NEWTON, 0, X, Z)
        target = dummy(w, name, 0.0, 300.0)
        w.set_hp(target, TOUGH)
        seconds(w, 3.0)
        results[name] = math.hypot(w.get(target).x - X, w.get(target).z - (Z + 300.0))
    assert results[DETRITER] < 10.0
    assert results[GLAIVE] > 10.0 * max(results[DETRITER], 1.0)


def test_what_is_pushed_does_not_walk_until_it_stops():
    w = world()
    w.add(NEWTON, 0, X, Z)
    target = w.add(KNIGHT, 1, X, Z + 200.0)
    w.set_fire_state(target, 0)
    w.set_unit_ai(target, False)
    w.set_hp(target, TOUGH)
    w.move(target, X, Z + 100.0)
    seconds(w, 0.5)
    start = w.get(target).z
    seconds(w, 5.0)
    assert w.get(target).z > start - 15.0     # 400 to 409 in a game, walking at it


def test_a_hard_landing_hurts():
    w = world()
    newtons = [w.add(NEWTON, 0, X + dx, Z) for dx in (-30.0, 0.0, 30.0)]
    for n in newtons:
        w.set_on_off(n, False)
    assert newtons
    target = dummy(w, GLAIVE, 0.0, 250.0)
    seconds(w, 6.0)
    assert w.get(target).damage_taken > 0.0


def test_buildings_and_what_is_being_built_stay_put():
    w = world()
    w.add(NEWTON, 0, X, Z)
    building = dummy(w, "turretlaser", 0.0, 200.0)
    frame = w.add(KNIGHT, 1, X + 100.0, Z + 200.0, build=0.5)
    seconds(w, 2.0)
    assert w.get(building).z == Z + 200.0
    assert w.get(frame).z == Z + 200.0


# Throwing ------------------------------------------------------------------------------------------------------------------------------------------------------


def lobster(w: zksim.World) -> int:
    u = w.add(LOBSTER, 0, X, Z)
    w.ready_weapons()
    w.set_unit_ai(u, False)
    return u


def test_a_lobster_is_not_armed_and_will_not_fire_by_itself():
    assert not CAT.units[CAT.index[LOBSTER]]["flags"] & zksim.ARMED
    w = world()
    lobster(w)
    near = dummy(w, KNIGHT, 50.0, 0.0, side=0)
    dummy(w, KNIGHT, 0.0, 400.0)
    seconds(w, 3.0)
    assert w.get(near).jumping == 0


def test_it_throws_everything_near_it_by_the_same_offset():
    w = world()
    l = lobster(w)
    friend, foe = dummy(w, KNIGHT, 100.0, 0.0, side=0), dummy(w, KNIGHT, 0.0, -100.0)
    far = dummy(w, KNIGHT, 400.0, 0.0, side=0)
    building = dummy(w, "turretlaser", -80.0, 0.0, side=0)
    w.manual_fire_at(l, X, Z + 500.0)
    w.step(1)
    assert w.get(friend).jumping == 2 and w.get(foe).jumping == 2
    assert w.get(far).jumping == 0 and w.get(building).jumping == 0 and w.get(l).jumping == 0
    top = 0.0
    frames = 0
    while w.get(friend).jumping and frames < 600:
        w.step(1)
        frames += FRAMES
        top = max(top, w.get(friend).altitude)
    assert 130 <= frames <= 150
    assert top > 200.0
    assert abs(w.get(friend).x - (X + 100.0)) < 2.0 and abs(w.get(friend).z - (Z + 510.0)) < 2.0   # a game: 511
    assert abs(w.get(foe).x - X) < 2.0 and abs(w.get(foe).z - (Z + 410.0)) < 2.0
    assert w.get(friend).damage_taken == 0.0


def test_it_walks_into_range_then_the_order_ends_and_it_has_to_reload():
    w = world()
    l = lobster(w)
    friend = w.add(KNIGHT, 0, X + 30.0, Z)
    w.set_unit_ai(friend, False)
    w.set_fire_state(friend, 0)
    w.manual_fire_at(l, X, Z + 900.0)
    seconds(w, 1.0)
    assert w.get(friend).jumping == 0
    assert w.get(l).z > Z
    w2 = world()
    l2 = lobster(w2)
    pal = dummy(w2, KNIGHT, 30.0, 0.0, side=0)
    w2.manual_fire_at(l2, X, Z + 300.0)
    w2.step(1)
    assert w2.get(pal).jumping == 2 and w2.get(l2).order == zksim.ORDER_NONE
    seconds(w2, 6.0)
    assert w2.get(pal).jumping == 0
    w2.manual_fire_at(l2, X, Z)
    seconds(w2, 1.0)
    assert w2.get(l2).order == zksim.ORDER_MANUAL_FIRE


def test_with_nothing_to_throw_it_waits():
    w = world()
    l = lobster(w)
    w.manual_fire_at(l, X, Z + 300.0)
    seconds(w, 1.0)
    assert w.get(l).order == zksim.ORDER_MANUAL_FIRE


def test_if_the_thrower_dies_too_soon_the_push_across_is_lost():
    w = world()
    l = lobster(w)
    pal = dummy(w, KNIGHT, 30.0, 0.0, side=0)
    w.manual_fire_at(l, X, Z + 500.0)
    w.step(1)
    w.kill(l)
    seconds(w, 6.0)
    assert abs(w.get(pal).z - Z) < 2.0


def test_it_does_not_throw_what_is_in_the_air():
    w = world()
    l = lobster(w)
    plane = dummy(w, "bomberprec", 40.0, 0.0, side=0)
    pal = dummy(w, KNIGHT, 30.0, 0.0, side=0)
    w.manual_fire_at(l, X, Z + 300.0)
    w.step(1)
    assert w.get(pal).jumping == 2 and w.get(plane).jumping == 0


def test_a_jumper_landing_from_a_throw_sets_off_nothing():
    w = world()
    l = lobster(w)
    sumo = dummy(w, SUMO, 30.0, 0.0, side=0)
    foe = dummy(w, KNIGHT, 30.0, 300.0)
    w.manual_fire_at(l, X, Z + 300.0)
    w.step(1)
    seconds(w, 6.0)
    assert w.get(sumo).jumping == 0
    assert w.get(foe).damage_taken == 0.0


# Silos ---------------------------------------------------------------------------------------------------------------------------------------------------------


def silo(w: zksim.World) -> int:
    u = w.add(SILO, 0, X, Z)
    w.produce(u, TACTICAL)
    return u


def test_a_silo_builds_up_to_as_many_as_it_has_pads_and_no_more():
    w = world(economy=True)
    s = silo(w)
    seconds(w, 300.0)
    handles = [u for u in range(s + 1, s + 9) if u < w.count()]
    assert len(handles) == 4
    for h in handles:
        assert w.get(h).build == 1.0


def test_firing_takes_a_missile_away_and_frees_its_pad_and_leaves_nothing():
    w = world(economy=True)
    s = silo(w)
    target = dummy(w, KNIGHT, 0.0, 1500.0)
    spotter = w.add("cloakcon", 0, X, Z + 1450.0)
    w.set_fire_state(spotter, 0)
    seconds(w, 90.0)
    first = spotter + 1
    assert w.get(first).build == 1.0
    w.attack(first, target)
    seconds(w, 40.0)
    assert not w.get(first).alive
    assert w.get(target).damage_taken > 1000.0
    assert len(w.features(0)) == 0


def test_a_silo_dying_takes_its_missiles_with_it():
    w = world(economy=True)
    s = silo(w)
    seconds(w, 90.0)
    first = w.count() - 1
    w.kill(s)
    seconds(w, 1.0)
    assert not w.get(s).alive
    assert not w.get(first).alive


def test_a_missile_dying_frees_a_pad():
    w = world(economy=True)
    s = silo(w)
    seconds(w, 300.0)
    count = w.count()
    w.kill(s + 1)
    seconds(w, 90.0)
    assert w.count() == count + 1


# Teleporting ---------------------------------------------------------------------------------------------------------------------------------------------------


def djinn(w: zksim.World, dx: float = 0.0, dz: float = 0.0) -> int:
    u = w.add(DJINN, 0, X + dx, Z + dz)
    w.set_unit_ai(u, False)
    return u


def beacon_at(w: zksim.World, u: int, x: float, z: float) -> int:
    w.place_beacon(u, x, z)
    frames = 0
    while w.get(u).beacon < 0 and frames < 1000:
        w.step(1)
        frames += FRAMES
    assert w.get(u).beacon >= 0
    return frames


def test_a_beacon_takes_as_long_as_the_script_says_and_it_stands_there_for_it():
    w = world()
    u = djinn(w)
    frames = beacon_at(w, u, X + 1500.0, Z)
    assert abs(frames - 270) <= FRAMES
    assert abs(w.get(u).x - X) < 1.0
    b = w.get(u).beacon
    assert abs(w.get(b).x - (X + 1500.0)) < 1.0
    assert w.get(u).order == zksim.ORDER_NONE


def test_a_new_beacon_replaces_the_old():
    w = world()
    u = djinn(w)
    beacon_at(w, u, X + 500.0, Z)
    first = w.get(u).beacon
    w.place_beacon(u, X + 900.0, Z)
    seconds(w, 10.0)
    second = w.get(u).beacon
    assert second != first and second >= 0
    assert not w.get(first).alive and w.get(second).alive


def test_a_unit_walks_over_waits_its_turn_and_comes_out_by_the_djinn():
    w = world()
    u = djinn(w)
    beacon_at(w, u, X + 1500.0, Z)
    b = w.get(u).beacon
    near = w.add(KNIGHT, 0, X + 1500.0, Z + 400.0)
    w.set_unit_ai(near, False)
    w.set_fire_state(near, 0)
    w.wait_at_beacon(near, b)
    seconds(w, 30.0)
    assert w.get(u).deployed == 1
    assert w.get(near).order == zksim.ORDER_NONE
    gap = apart(w, near, u)
    assert gap < 200.0, gap


def test_the_time_goes_by_mass_and_the_longest_waiting_goes_first():
    w = world()
    u = djinn(w)
    beacon_at(w, u, X + 1500.0, Z)
    b = w.get(u).beacon
    seconds(w, 2.0)
    first = w.add(KNIGHT, 0, X + 1500.0, Z + 50.0)
    second = w.add(GLAIVE, 0, X + 1500.0, Z - 50.0)
    for unit in (first, second):
        w.set_unit_ai(unit, False)
        w.set_fire_state(unit, 0)
    w.wait_at_beacon(first, b)
    w.step(1)
    w.wait_at_beacon(second, b)
    frames = 0
    while w.get(first).order and frames < 2000:
        w.step(1)
        frames += FRAMES
    want = CAT.units[CAT.index[KNIGHT]]["mass"] / CAT.units[CAT.index[DJINN]]["teleport_rate"]
    assert abs(frames - want) < 40.0, (frames, want)
    assert w.get(second).order == zksim.ORDER_WAIT_AT_BEACON


def test_it_will_not_bring_anything_over_while_walking_and_nothing_goes_to_the_enemy_beacon():
    w = world()
    u = djinn(w)
    beacon_at(w, u, X + 1500.0, Z)
    b = w.get(u).beacon
    w.move(u, X, Z + 1000.0)
    seconds(w, 0.5)
    assert w.get(u).deployed == 0
    foe = w.add(GLAIVE, 1, X + 1500.0, Z + 40.0)
    w.wait_at_beacon(foe, b)
    assert w.get(foe).order != zksim.ORDER_WAIT_AT_BEACON
    w.set_fire_state(foe, 0)


def test_a_djinn_dying_takes_its_beacon():
    w = world()
    u = djinn(w)
    beacon_at(w, u, X + 1500.0, Z)
    b = w.get(u).beacon
    w.kill(u)
    seconds(w, 1.0)
    assert not w.get(b).alive


def test_the_enemy_does_not_shoot_at_a_beacon_by_itself():
    w = world()
    u = djinn(w, dz=-2000.0)
    beacon_at(w, u, X, Z)
    b = w.get(u).beacon
    w.add(KNIGHT, 1, X + 100.0, Z)
    seconds(w, 5.0)
    assert w.get(b).damage_taken == 0.0
