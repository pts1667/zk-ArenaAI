"""Aircraft: planes turning in a circle, bombers' sorties and rearming, strafing gunships, and what lands when idle.  The engine's
numbers are from tools/sim_engine.py air (03-training.md 11.14)."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 2000.0, 1500.0
FRAMES = zksim.TICK_FRAMES
FLYING, LANDED, ON_PAD = 0, 1, 4


def world(economy: bool = False) -> zksim.World:
    w = zksim.World(CAT, FLAT, seed=1)
    if economy:
        zksim._set_economy(w.handle, np.zeros((0, 3), np.float32), 100000.0, 100000.0, 0.0, 0.0, 100000.0)
    return w


def sitting(w: zksim.World, name: str, x: float, z: float) -> int:
    u = w.add(name, 1, x, z)
    w.set_fire_state(u, 0)
    w.set_move_state(u, 0)
    w.set_unit_ai(u, False)
    return u


# Engine frames for a half turn, at full speed into it.
@pytest.mark.parametrize("name,half_turn", [("planefighter", 72), ("bomberprec", 54), ("bomberheavy", 90), ("planescout", 46), ("bomberassault", 176)])
def test_a_plane_turns_round_as_in_the_engine(name, half_turn):
    w = world()
    u = w.add(name, 0, X, Z)
    w.move(u, X, Z + 2500)
    w.step(40)
    w.move(u, X, Z - 600)
    h0, frames = w.get(u).heading, 0
    while frames < 400:
        w.step(1)
        frames += FRAMES
        if abs((w.get(u).heading - h0 + math.pi) % (2 * math.pi) - math.pi) > 0.95 * math.pi:
            break
    assert frames == pytest.approx(half_turn, rel=0.1)


def test_a_plane_lands_at_the_end_of_a_move_and_a_gunship_from_its_factory_stays_up():
    """unit_air_plants.lua: the plane plant's land when idle, the gunship plant's hover; a hovering constructor of the plane plant lands too."""
    states = {}
    for name in ("planefighter", "planecon", "gunshipcon", "gunshipraid"):
        w = world()
        u = w.add(name, 0, X, Z)
        w.move(u, X + 800, Z)
        w.step(120)
        s = w.get(u)
        states[name] = (s.air_state, round(s.altitude))
        assert abs(s.x - X - 800) < 100
    assert states["planefighter"] == (LANDED, 0) and states["planecon"] == (LANDED, 0)
    assert states["gunshipcon"][0] == FLYING and states["gunshipcon"][1] > 50
    assert states["gunshipraid"][0] == FLYING


def sortie(name: str, target: str = "tankassault", ticks: int = 400):
    w = world(economy=True)
    w.add("factoryplane", 0, X, Z - 400)
    b = w.add(name, 0, X, Z)
    t = sitting(w, target, X, Z + 1500)
    w.add("planescout", 0, X + 100, Z + 1500)    # sight
    w.attack(b, t)
    rows = []
    for _ in range(ticks):
        w.step(1)
        rows.append(w.get(b))
    return w, b, t, rows


def test_a_raven_hits_once_then_rearms_on_the_pad_and_waits_there():
    w, b, t, rows = sortie("bomberprec")
    first_pad = next(k for k, s in enumerate(rows) if s.air_state == ON_PAD)
    rearmed = next(k for k, s in enumerate(rows) if k > first_pad and s.ammo > 0)
    assert (rearmed - first_pad) * FRAMES == pytest.approx(240, abs=FRAMES)     # reammoseconds 8
    assert w.get(t).hp == pytest.approx(CAT.units[CAT.index["tankassault"]]["hp"] - 800, abs=1.0)    # one hit, its order gone
    end = rows[-1]
    assert end.order == zksim.ORDER_NONE and end.air_state == LANDED and end.ammo == 1
    assert math.hypot(end.x - X, end.z - Z + 400) < 1.0


def test_a_thunderbird_disarms_what_it_runs_over_once_a_sortie():
    w, b, t, rows = sortie("bomberdisarm", ticks=120)
    assert w.get(t).disarm > 1.0
    assert max(s.shots_fired for s in rows) == 80      # bomberdisarm.lua: 80 bolts, then it rearms
    assert rows[-1].ammo == 0


def test_odins_thermite_burns_a_still_target_for_about_what_the_game_does():
    """The engine's run took 14,842 from a staticradar too tough to die in 30 s."""
    w = world(economy=True)
    w.add("factoryplane", 0, X, Z - 400)
    b = w.add("bomberassault", 0, X, Z)
    t = sitting(w, "staticradar", X, Z + 1500)
    w.set_hp(t, 1e6)
    w.add("planescout", 0, X + 100, Z + 1500)
    w.attack(b, t)
    w.step(400)
    assert 1e6 - w.get(t).hp == pytest.approx(14842, rel=0.05)


# Engine radii it circled at, attacking something still.
@pytest.mark.parametrize("name,radius", [("gunshipraid", 199), ("gunshipskirm", 292), ("gunshipassault", 124), ("gunshipheavyskirm", 500)])
def test_a_strafing_gunship_circles_its_target_as_in_the_engine(name, radius):
    w = world()
    g = w.add(name, 0, X, Z)
    t = sitting(w, "staticradar", X, Z + 1000)
    w.set_hp(t, 1e6)
    w.attack(g, t)
    w.step(150)
    seen = []
    for _ in range(100):
        w.step(1)
        s = w.get(g)
        seen.append(math.hypot(s.x - X, s.z - Z - 1000))
    assert np.mean(seen) == pytest.approx(radius, abs=15)
    assert 1e6 - w.get(t).hp > 0

