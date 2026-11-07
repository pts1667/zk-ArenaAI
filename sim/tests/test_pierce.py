"""Shots that don't stop at what they hit (noExplode).  Measured against a game in 01-simulator.md section 8.16."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 1000.0, 1000.0
TOUGH = 1000000.0
PYRO, GAUSS = "jumpraid", "turretgauss"


def shooter(w: zksim.World, name: str) -> int:
    u = w.add(name, 0, X, Z)
    w.set_move_state(u, 0)
    w.set_unit_ai(u, False)
    w.set_fire_state(u, 2)
    return u


def dummy(w: zksim.World, name: str, dx: float, dz: float, side: int = 1) -> int:
    u = w.add(name, side, X + dx, Z + dz)
    w.set_fire_state(u, 0)
    w.set_unit_ai(u, False)
    w.set_move_state(u, 0)
    w.set_hp(u, TOUGH)
    return u


def lost(w: zksim.World, u: int) -> float:
    return TOUGH - w.get(u).hp


def play(w: zksim.World, frames: int) -> None:
    w.step(frames // zksim.TICK_FRAMES)


def world() -> zksim.World:
    return zksim.World(CAT, FLAT, seed=1)


def weapon(name: str) -> dict:
    return next(w for w in CAT.weapons if w["name"] == name)


def test_which_weapons_go_through_comes_from_their_defs():
    flame, gauss, thermite = weapon("jumpraid_flamethrower"), weapon("turretgauss_gauss"), weapon("bomberassault_thermite_bomb")
    assert flame["flags"] & zksim.W_PIERCES and not flame["flags"] & zksim.W_ONE_HIT_EACH
    assert gauss["flags"] & zksim.W_PIERCES and gauss["flags"] & zksim.W_ONE_HIT_EACH
    assert not thermite["flags"] & zksim.W_PIERCES
    assert not weapon("cloakraid_emg")["flags"] & zksim.W_PIERCES


def test_a_flame_hits_a_wide_target_more_times_than_a_thin_one():
    """A game's Pyro: 2 hits a shot on a Glaive or a Knight, 3 on a Minotaur."""
    hurt = {}
    for name in ("cloakraid", "cloakassault", "tankassault"):
        w = world()
        shooter(w, PYRO)
        t = dummy(w, name, 0.0, 100.0)
        play(w, 150)
        hurt[name] = lost(w, t)
    assert 500.0 < hurt["cloakraid"] < hurt["cloakassault"] < hurt["tankassault"] < 1300.0


def test_a_gauss_slug_hits_each_unit_once_however_wide():
    hurt = []
    for name in ("cloakraid", "tankassault"):
        w = world()
        shooter(w, GAUSS)
        t = dummy(w, name, 0.0, 200.0)
        play(w, 450)
        hurt.append(lost(w, t))
    assert hurt[0] == hurt[1] > 0.0


def test_a_shot_goes_on_to_what_is_behind_its_target():
    for name, frames in ((PYRO, 150), (GAUSS, 450)):
        w = world()
        shooter(w, name)
        near, behind = dummy(w, "cloakassault", 0.0, 100.0), dummy(w, "cloakassault", 0.0, 170.0)
        play(w, frames)
        assert lost(w, near) > 0.0 and lost(w, behind) > 0.0


def test_it_stops_at_the_end_of_its_range():
    w = world()
    shooter(w, GAUSS)
    near, far = dummy(w, "cloakassault", 0.0, 200.0), dummy(w, "cloakassault", 0.0, 900.0)
    play(w, 450)
    assert lost(w, near) > 0.0 and lost(w, far) == 0.0


def test_what_is_off_the_line_is_only_hurt_by_a_blast_that_reaches_it():
    w = world()
    shooter(w, GAUSS)
    aimed, beside = dummy(w, "cloakassault", 0.0, 120.0), dummy(w, "cloakassault", 45.0, 120.0)
    play(w, 450)
    assert lost(w, aimed) > 0.0 and lost(w, beside) == 0.0
    w = world()
    shooter(w, PYRO)
    aimed, beside = dummy(w, "cloakassault", 0.0, 120.0), dummy(w, "cloakassault", 45.0, 120.0)
    play(w, 150)
    assert 0.0 < lost(w, beside) < lost(w, aimed) / 2.0


def test_it_passes_through_a_friend_on_the_line():
    w = world()
    shooter(w, GAUSS)
    friend, foe = dummy(w, "cloakassault", 0.0, 70.0, side=0), dummy(w, "cloakassault", 0.0, 150.0)
    play(w, 450)
    assert lost(w, friend) == 0.0 and lost(w, foe) > 0.0


def test_it_never_hurts_who_fired_it():
    w = world()
    u = shooter(w, PYRO)
    w.set_hp(u, TOUGH)
    dummy(w, "cloakassault", 0.0, 60.0)
    play(w, 150)
    assert lost(w, u) == 0.0


def test_a_shield_still_stops_it():
    w = world()
    shooter(w, PYRO)
    t = dummy(w, "cloakassault", 0.0, 150.0)
    dummy(w, "shieldcon", 0.0, 150.0)       # radius 80, and it starts charged
    play(w, 60)
    assert lost(w, t) == 0.0
