"""Shots that fly: aimed ahead of what moves as the engine aims them (Weapon.cpp GetLeadVec), and landing where they come down, on
whoever is there by then.  And the jinking that Zero-K's tactical AI has raiders do against them (unit_tactical_ai.lua).  Measured
against a game by tools/sim_fights.py (01-simulator.md, "Validation")."""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
ROCKETS = "cloakskirm"          # Ronin: a rocket at 6.7 elmos a frame over 455, with a blast of 24
SHELLS = "spidercrabe"          # Crab: a shell at 9.7 elmos a frame over 600, with a blast of 100
RAIDER = "cloakraid"            # Glaive: 3.85 elmos a frame
EAST = math.pi / 2
Z = 2000.0
TOUGH = 1000.0                  # times its health, so that it's there to be hit again


def _duel(shooter: str, target_x: float, target_z: float):
    """A shooter at x 1000, loaded, and a raider that won't shoot back.  Units that do only what they're told, and shoot what's in range."""
    w = zksim.World(CAT, FLAT, seed=5, behaviour=zksim.SCRIPTED)
    s = w.add(shooter, 0, 1000.0, Z, EAST)
    t = w.add(RAIDER, 1, target_x, target_z, hp=TOUGH)
    w.set_fire_state(t, 0)
    w.ready_weapons()
    return w, s, t


def _until_it_fires(w: zksim.World, s: int) -> None:
    for _ in range(120):
        if w.get(s).shots_fired:
            return
        w.step(1)
    raise AssertionError("it never fired")


def test_a_target_that_moves_off_dodges_a_slow_shot_that_hits_one_standing_still():
    """A rocket takes 60 frames over 400 elmos.  Fired at a raider standing still, it's aimed at it: one that walks off as it's
    fired is 200 elmos away when it arrives."""
    def taken(dodges: bool) -> float:
        w, s, t = _duel(ROCKETS, 1400.0, Z)
        _until_it_fires(w, s)
        if dodges:
            w.move(t, 1400.0, Z + 400.0)
        w.step(16)          # 80 frames: its flight, and less than its reload
        assert w.get(s).shots_fired == 1
        return w.get(t).damage_taken

    assert taken(dodges=False) > 0.0
    assert taken(dodges=True) == 0.0


def test_a_shot_is_aimed_ahead_of_a_target_that_keeps_its_course():
    """A raider crossing a Crab's front at 450 elmos walks 190 in a shell's flight, further than its blast reaches: aimed at where
    the raider is, the shell would do nothing.  It lands on it, and turning back as it's fired is how to get out from under it."""
    full = CAT.weapons[CAT.units[CAT.index[SHELLS]]["mount_weapon"][0]]["damage"][CAT.units[CAT.index[RAIDER]]["armor_class"]]

    def taken(turns: bool) -> float:
        w, s, t = _duel(SHELLS, 1450.0, Z - 700.0)
        w.move(t, 1450.0, Z + 600.0)
        _until_it_fires(w, s)       # as the raider comes into range, at its full speed
        if turns:
            w.move(t, 1450.0, Z - 700.0)
        w.step(20)          # 100 frames: its flight, and less than its reload
        assert w.get(s).shots_fired == 1
        return w.get(t).damage_taken

    assert taken(turns=False) > 0.5 * full
    assert taken(turns=True) == 0.0


def test_a_raider_jinks_up_to_a_slow_shooter_on_a_fight_order_and_not_on_a_move():
    """unit_tactical_ai.lua's swarming: in reach of what's slow to hit it, a raider on a fight order goes from side to side.  A
    plain move order gets none of it."""
    def strays(fights: bool) -> float:
        w = zksim.World(CAT, FLAT, seed=5, behaviour=zksim.ORDERS)
        s = w.add(ROCKETS, 0, 1000.0, Z, EAST, hp=TOUGH)
        w.set_move_state(s, 0)
        w.set_fire_state(s, 0)
        t = w.add(RAIDER, 1, 1700.0, Z, -EAST)
        (w.fight if fights else w.move)(t, 1000.0, Z)
        off = 0.0
        for _ in range(40):
            w.step(1)
            at = w.get(t)
            if at.x - 1000.0 > 200.0:       # until its own gun reaches
                off = max(off, abs(at.z - Z))
        return off

    assert strays(fights=True) > 10.0
    assert strays(fights=False) < 1.0
