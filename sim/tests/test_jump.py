"""Jumping, digging in and Blastwing's bomb.  Measured against a game in 01-simulator.md section 8.14."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 1000.0, 2000.0
PYRO, JACK, SUMO, SKUTTLE, IMP, SNITCH, BLASTWING, KNIGHT = "jumpraid", "jumpassault", "jumpsumo", "jumpbomb", "cloakbomb", "shieldbomb", "gunshipbomb", "cloakassault"
TOUGH = 1000000.0
FRAMES = zksim.TICK_FRAMES


def world(the_map=FLAT) -> zksim.World:
    return zksim.World(CAT, the_map, seed=1)


def dummy(w: zksim.World, name: str, dx: float, dz: float, hp: float = TOUGH, side: int = 1) -> int:
    u = w.add(name, side, X + dx, Z + dz)
    w.set_fire_state(u, 0)
    w.set_move_state(u, 0)
    w.set_unit_ai(u, False)
    w.set_hp(u, hp)
    return u


def jumper(w: zksim.World, name: str = PYRO) -> int:
    u = w.add(name, 0, X, Z)
    w.set_fire_state(u, 0)
    w.set_unit_ai(u, False)
    return u


def seconds(w: zksim.World, s: float) -> None:
    w.step(int(s * 30 / FRAMES))


def land(w: zksim.World, u: int, most: float = 10.0) -> int:
    """Frames until it's down."""
    frames = 0
    while w.get(u).jumping and frames < most * 30:
        w.step(1)
        frames += FRAMES
    assert not w.get(u).jumping
    return frames


def flight_frames(name: str, line: float) -> int:
    d = CAT.units[CAT.index[name]]
    return math.ceil(2.0 * math.hypot(line * 0.5, d["jump_height"]) / d["jump_speed"])


def test_who_can_jump_goes_by_their_defs():
    can = {u["name"] for u in CAT.units if u["jump_range"] > 0.0}
    assert {"jumpcon", JACK, PYRO, SUMO, SKUTTLE} <= can
    assert not {"factoryjump", "platejump", "cloakraid"} & can


def test_a_jump_goes_along_its_arc_to_where_it_was_told_in_the_time_its_length_takes():
    w = world()
    u = jumper(w)
    w.jump(u, X, Z + 300.0)
    assert w.get(u).jumping == 2
    w.step(5)
    middle = w.get(u)
    assert middle.jumping == 2 and Z < middle.z < Z + 300.0 and middle.vz > 3.0
    frames = 25 + land(w, u)
    s = w.get(u)
    assert abs(frames - flight_frames(PYRO, 300.0)) <= FRAMES
    assert (s.x, s.z) == (X, Z + 300.0) and s.speed == 0.0 and s.order == zksim.ORDER_NONE and s.jump_charge == 0.0


def test_a_short_jump_covers_less_ground_a_frame_because_the_arc_is_as_high():
    assert flight_frames(PYRO, 100.0) > 0.6 * flight_frames(PYRO, 300.0)


def test_out_of_range_it_walks_into_range_first():
    w = world()
    u = jumper(w)
    w.jump(u, X, Z + 700.0)
    assert not w.get(u).jumping
    seconds(w, 1.0)
    assert not w.get(u).jumping and w.get(u).z > Z + 10.0
    while not w.get(u).jumping:
        w.step(1)
    left_from = w.get(u).z
    assert Z + 700.0 - 400.0 <= left_from < Z + 360.0     # It looks every 15 frames.
    land(w, u)
    assert w.get(u).z == Z + 700.0


def test_it_waits_for_its_charge_before_going_again():
    w = world()
    u = jumper(w)
    w.jump(u, X, Z + 300.0)
    land(w, u)
    w.jump(u, X, Z)
    assert not w.get(u).jumping
    seconds(w, 9.0)
    assert not w.get(u).jumping and 0.85 < w.get(u).jump_charge < 1.0
    assert w.get(u).z == Z + 300.0  # In range, so it stands there.
    seconds(w, 1.5)
    assert w.get(u).jumping == 2


def test_an_order_given_in_the_air_leaves_it_standing_where_it_lands_until_it_gets_another():
    for give in (lambda w, u: w.move(u, X + 300.0, Z + 300.0), lambda w, u: w.fight(u, X + 300.0, Z + 300.0), lambda w, u: w.attack(u, dummy(w, KNIGHT, 200.0, 600.0))):
        w = world()
        u = jumper(w)
        w.jump(u, X, Z + 300.0)
        w.step(4)
        give(w, u)
        land(w, u)
        seconds(w, 3.0)
        assert abs(w.get(u).x - X) < 1.0 and abs(w.get(u).z - Z - 300.0) < 1.0
        w.move(u, X + 300.0, Z + 300.0)
        seconds(w, 3.0)
        assert w.get(u).x > X + 50.0


def _stun(w: zksim.World, u: int) -> None:
    """More than enough Imps going off beside it, of the other side."""
    s = w.get(u)
    for _ in range(4):
        w.self_destruct(dummy(w, IMP, s.x - X, s.z - Z + 20.0, hp=100.0))
    w.step(1)
    assert w.get(u).para > 1.0, "this test needs it stunned"


def test_being_stunned_stops_a_take_off_but_not_a_flight():
    w = world()
    u = jumper(w)
    w.jump(u, X, Z + 100.0)
    w.step(2)
    _stun(w, u)
    land(w, u)
    assert w.get(u).z == Z + 100.0
    w = world()
    u = jumper(w)
    _stun(w, u)
    w.jump(u, X, Z + 300.0)
    seconds(w, 3.0)
    s = w.get(u)
    assert not s.jumping and abs(s.z - Z) < 30.0      # The Imps push it a little.


def test_what_cant_jump_doesnt():
    w = world()
    u = w.add("cloakraid", 0, X, Z)
    w.jump(u, X, Z + 300.0)
    assert w.get(u).order == zksim.ORDER_NONE
    seconds(w, 2.0)
    assert w.get(u).z == Z


def _with_wall() -> zksim.Map:
    """A cliff across the map at z 2100, higher than anything can jump."""
    height = np.array(FLAT.height, dtype=np.float32, copy=True)
    speed = np.array(FLAT.speed, dtype=np.float32, copy=True)
    rows = int(2100 / FLAT.cell), int(2200 / FLAT.cell)
    height[rows[0]:rows[1], :] += 1000.0
    speed[:, rows[0]:rows[1], :] = 0.0
    return zksim.Map(height, speed, FLAT.cell)


def test_it_wont_jump_into_the_side_of_a_cliff_or_land_where_it_cant_walk():
    wall = _with_wall()
    w = world(wall)
    u = jumper(w)
    w.jump(u, X, Z + 300.0)
    seconds(w, 1.0)
    assert not w.get(u).jumping and w.get(u).order == zksim.ORDER_JUMP  # It stands there, still meaning to.
    w = world(wall)
    u = jumper(w)
    w.jump(u, X, 2150.0)
    assert w.get(u).order == zksim.ORDER_NONE


def test_jugglenauts_landing_hurts_what_it_lands_near_and_not_it():
    w = world()
    u = jumper(w, SUMO)
    near = dummy(w, KNIGHT, 0.0, 360.0)
    far = dummy(w, KNIGHT, 0.0, 600.0)
    friend = dummy(w, KNIGHT, 80.0, 300.0, side=0)
    w.jump(u, X, Z + 300.0)
    assert w.get(u).jumping == 1  # It takes its time about leaving.
    w.step(3)
    assert w.get(u).z == Z
    w.step(4)
    assert w.get(u).jumping == 2
    land(w, u)
    w.step(1)
    assert TOUGH - w.get(near).hp > 300.0
    assert TOUGH - w.get(friend).hp > 300.0
    assert w.get(far).hp == TOUGH
    assert w.get(u).damage_taken == 0.0


def test_a_landing_blast_is_not_a_gun():
    """They come from the scripts on landing.  Both have ordinary ranges in their defs."""
    w = world()
    u = w.add("striderdetriment", 0, X, Z)
    w.ready_weapons()
    for name in (SUMO, "striderdetriment"):
        d = CAT.units[CAT.index[name]]
        assert d["landing_weapon"] in d["mount_weapon"]
        assert CAT.weapons[d["landing_weapon"]]["flags"] & zksim.W_NO_AUTO_TARGET
    assert w.get(u).alive


def test_a_blast_under_a_jumper_at_the_top_of_its_arc_is_a_long_way_off():
    def lost(jump: bool) -> float:
        w = world()
        u = dummy(w, JACK, 0.0, 0.0 if jump else 140.0, side=0)
        b = dummy(w, SNITCH, 0.0, 150.0, hp=100.0)
        if jump:
            w.jump(u, X, Z + 300.0)
            while w.get(u).z < Z + 150.0:
                w.step(1)
        w.self_destruct(b)
        w.step(3)
        return TOUGH - w.get(u).hp

    assert 0.0 <= lost(True) < 0.5 * lost(False)


def test_it_can_fire_in_the_air_at_what_it_can_still_reach():
    w = world()
    u = w.add(PYRO, 0, X, Z)
    t = dummy(w, KNIGHT, 40.0, 150.0)
    w.ready_weapons()
    w.jump(u, X, Z + 300.0)
    land(w, u)
    assert w.get(t).hp < TOUGH
    # A Jack's spike is too short to reach the ground from up there.
    w = world()
    u = w.add(JACK, 0, X, Z)
    t = dummy(w, KNIGHT, 40.0, 150.0)
    w.ready_weapons()
    w.jump(u, X, Z + 300.0)
    w.step(4)
    while w.get(u).z < Z + 230.0:
        w.step(1)
    assert w.get(t).hp == TOUGH


def test_skuttle_stays_cloaked_and_what_it_passes_over_doesnt_set_it_off():
    w = world()
    b = w.add(SKUTTLE, 0, X, Z)
    w.step(3)
    assert w.get(b).cloaked
    w.jump(b, X, Z + 300.0)
    while w.get(b).jumping:
        assert w.get(b).cloaked
        w.step(1)
    w = world()
    b = w.add(SKUTTLE, 0, X, Z)
    w.set_fire_state(b, 2)
    t = dummy(w, KNIGHT, 0.0, 150.0)
    w.jump(b, X, Z + 300.0)
    land(w, b)
    assert w.get(b).alive and w.get(t).hp == TOUGH
    # Landing on one does.
    w = world()
    b = w.add(SKUTTLE, 0, X, Z)
    w.set_fire_state(b, 2)
    t = dummy(w, KNIGHT, 0.0, 300.0)
    w.jump(b, X, Z + 300.0)
    seconds(w, 8.0)
    assert not w.get(b).alive and w.get(t).hp < TOUGH


def test_imp_snitch_and_flea_dig_in_when_they_stop():
    for name in (IMP, SNITCH, "spiderscout"):
        w = world()
        u = w.add(name, 0, X, Z)
        w.set_fire_state(u, 0)
        w.step(4)
        assert w.get(u).cloaked
        w.move(u, X, Z + 300.0)
        seconds(w, 1.5)
        assert not w.get(u).cloaked
        seconds(w, 3.0)
        assert w.get(u).speed < 0.05 and w.get(u).cloaked


def test_an_enemy_close_by_shows_what_has_dug_in():
    w = world()
    u = w.add(IMP, 0, X, Z)
    w.step(4)
    assert w.get(u).cloaked
    dummy(w, KNIGHT, 60.0, 0.0)
    w.step(3)
    assert not w.get(u).cloaked


def test_a_blastwing_sits_cloaked_when_idle_and_not_when_going_somewhere():
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    w.set_fire_state(u, 0)
    seconds(w, 1.0)
    assert w.get(u).cloaked
    w.move(u, X, Z + 1500.0)
    seconds(w, 1.0)
    assert not w.get(u).cloaked
    seconds(w, 12.0)
    assert w.get(u).order == zksim.ORDER_NONE and w.get(u).cloaked


def _hurt_after_death(w: zksim.World, u: int, t: int, most: float = 12.0) -> tuple[int, float, float]:
    """Frames until it's gone, how far it was from the target then, and what the target has lost a second and a half later."""
    frames, apart = 0, -1.0
    while w.get(u).alive and frames < most * 30:
        a, b = w.get(u), w.get(t)
        apart = math.hypot(a.x - b.x, a.z - b.z)
        w.step(1)
        frames += FRAMES
    seconds(w, 1.5)
    return (frames if not w.get(u).alive else -1), apart, TOUGH - w.get(t).hp


def test_a_blastwing_told_to_attack_goes_off_far_out_at_full_speed_and_its_bomb_lands_on_target():
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    t = dummy(w, KNIGHT, 0.0, 450.0)
    w.attack(u, t)
    died, apart, lost = _hurt_after_death(w, u, t)
    assert died > 0 and 60.0 < apart <= 170.0 + 40.0
    assert 150.0 < lost < 500.0


def test_a_blastwing_on_fire_at_will_goes_off_for_what_it_passes_close_to_and_not_at_hold_fire():
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    t = dummy(w, KNIGHT, 60.0, 800.0)
    w.move(u, X, Z + 1500.0)
    died, _, _ = _hurt_after_death(w, u, t)
    assert died > 0
    # Farther off, only a target it's been given sets it off.
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    t = dummy(w, KNIGHT, 120.0, 800.0)
    w.move(u, X, Z + 1500.0)
    died, _, _ = _hurt_after_death(w, u, t, 5.0)
    assert died < 0
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    w.set_fire_state(u, 0)
    t = dummy(w, KNIGHT, 60.0, 800.0)
    w.move(u, X, Z + 1500.0)
    died, _, lost = _hurt_after_death(w, u, t, 5.0)
    assert died < 0 and lost == 0.0


def test_a_blastwing_killed_at_speed_throws_its_bomb_ahead():
    w = world()
    u = w.add(BLASTWING, 0, X, Z)
    w.set_fire_state(u, 0)
    w.move(u, X, Z + 3000.0)
    seconds(w, 2.0)
    s = w.get(u)
    assert s.speed > 5.0
    frames = math.sqrt(2.0 * 30.0 / 0.12)
    t = dummy(w, KNIGHT, 0.0, s.z - Z + s.speed * frames)
    behind = dummy(w, KNIGHT, 0.0, s.z - Z - 150.0)
    w.kill(u)
    seconds(w, 1.5)
    assert TOUGH - w.get(t).hp > 150.0
    assert w.get(behind).hp == TOUGH


def test_an_unfinished_blastwing_drops_nothing():
    w = world()
    u = w.add(BLASTWING, 0, X, Z, build=0.5)
    t = dummy(w, KNIGHT, 40.0, 0.0)
    w.kill(u)
    seconds(w, 1.5)
    assert w.get(t).hp == TOUGH


def test_a_view_shows_who_is_charged_and_who_is_in_the_air():
    w = world()
    u = jumper(w)
    w.step(1)

    def record():
        units = w.view(0).units
        return units[units["uid"] == u][0]

    assert record()["flags"] & zksim.rawframe.JUMP_READY
    ground = float(record()["y"])
    w.jump(u, X, Z + 300.0)
    w.step(4)
    assert not record()["flags"] & zksim.rawframe.JUMP_READY and float(record()["y"]) > ground + 20.0
    land(w, u)
    assert not record()["flags"] & zksim.rawframe.JUMP_READY and float(record()["y"]) == ground
    seconds(w, 11.0)
    assert record()["flags"] & zksim.rawframe.JUMP_READY
    plain = world()
    other = plain.add("cloakraid", 0, X, Z)
    plain.step(1)
    assert not plain.view(0).units[0]["flags"] & zksim.rawframe.JUMP_READY and other == 0
