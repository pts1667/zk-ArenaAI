"""Water: who can aim at whom in it and fire from it, as the engine has it (Weapon.cpp TestTarget and CanFire, TorpedoLauncher.cpp),
what's under water out of the reach of what goes off at the surface, and amphibious units that float up to fire
(unit_impulsefloat_toggle.lua).  Each against what the engine did with the same units (tools/sim_engine.py water;
01-simulator.md, "Water")."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
CELL, ROWS, COLS = 16.0, 128, 256
SHORE = 2048.0          # land at 50 to the west of it, and the sea bed 150 down to the east
_h = np.where((np.arange(COLS) + 0.5) * CELL < SHORE, 50.0, -150.0).astype(np.float32)[None, :].repeat(ROWS, 0)
MAP = zksim.Map(_h, np.ones((len(zksim.MOVE_CLASSES), ROWS, COLS), np.float32), CELL, "shore")
Z = 1000.0
TOUGH = 1000.0          # times its health, so that it's there to be hit again


def _world(*units):
    """Our shooter and their targets, which don't shoot back, as (name, team, x) or (name, team, x, z), loaded."""
    w = zksim.World(CAT, MAP, seed=3, behaviour=zksim.SCRIPTED)
    made = [w.add(u[0], u[1], u[2], u[3] if len(u) > 3 else Z, hp=1.0 if u[1] == 0 else TOUGH) for u in units]
    for u, spec in zip(made, units):
        if spec[1]:
            w.set_fire_state(u, 0)
    w.ready_weapons()
    return w, made


def test_who_can_shoot_whom_in_water_is_as_in_the_engine():
    """A Duck on the sea bed is under water: an LLT riding the water above it and a Glaive on the shore don't fire at it, and an
    Urchin's torpedoes and a Dagger's gauss, water weapons, hit it.  Its own torpedoes hit a ship over it, and not a Glaive on land,
    in their reach but out of the water, while its missiles don't fire from under water.  An Urchin doesn't fire at a Duck on land."""
    cases = [("turretlaser", 3000.0, "amphraid", 3200.0, False), ("cloakraid", 2040.0, "amphraid", 2140.0, False),
             ("turrettorp", 3000.0, "amphraid", 3200.0, True), ("hoverraid", 3000.0, "amphraid", 3150.0, True),
             ("amphraid", 3000.0, "shipscout", 3140.0, True), ("amphraid", 2190.0, "cloakraid", 2040.0, False),
             ("turrettorp", 2300.0, "amphraid", 2040.0, False)]
    for shooter, sx, target, tx, hits in cases:
        w, (s, t) = _world((shooter, 0, sx), (target, 1, tx))
        w.step(48)          # 8 s
        assert (w.get(t).damage_taken > 0.0) == hits, (shooter, target, w.get(s).shots_fired)


def test_a_shell_that_goes_off_at_the_surface_doesnt_reach_the_sea_bed():
    """A Crab's shells, with a blast of 100, at a ship with a Duck on the sea bed beside it: the ship is hit, and the Duck, 136 elmos
    down at its middle, isn't touched.  The same shells at the ship beside a Duck that's ashore hurt it too."""
    w, (_, ship, duck) = _world(("spidercrabe", 0, 1700.0), ("shipscout", 1, 2250.0), ("amphraid", 1, 2250.0, Z + 30.0))
    w.step(60)
    assert w.get(ship).damage_taken > 0.0 and w.get(duck).base_y < -149.0 and w.get(duck).damage_taken == 0.0
    w, (_, ship, duck) = _world(("spidercrabe", 0, 1500.0), ("shipscout", 1, 2060.0), ("amphraid", 1, 2030.0))
    w.step(60)
    assert w.get(ship).damage_taken > 0.0 and w.get(duck).damage_taken > 0.0


def test_a_buoy_floats_up_to_fire_and_sinks_back_once_its_target_is_gone():
    """A Buoy on the sea bed 150 down, a ship 300 off, which it can't fire at from there.  In the engine it rose to the surface, rode
    up out of the water and settled at its float point, firing 60 frames after it left the sea bed; held still meanwhile.  The
    ship gone after 10 s, it was back on the sea bed 6.1 s later.  The engine's heights, 5 frames after it left and every 15 after."""
    engine = [-139, -109, -76, -42, -6, 19, 22, 3, -24, -22, -9, -12, -19, -13, -14, -17]
    w, (buoy, ship) = _world(("amphfloater", 0, 3000.0), ("shipscout", 1, 3300.0))
    ys, start, first, back = [], None, None, None
    for tick in range(int(20 * 30 / 5)):
        if tick == 60:
            w.kill(ship)
        w.step(1)
        s = w.get(buoy)
        ys.append(s.base_y)
        if start is None and s.floating:
            start = tick
        if first is None and s.shots_fired:
            first = tick
        if tick > 60 and back is None and not s.floating:
            back = tick
        assert (s.x, s.z) == (3000.0, Z)
    sim = [ys[start + 3 * k] for k in range(len(engine))]
    assert np.max(np.abs(np.array(sim) - engine)) < 8.0, [round(y) for y in sim]
    assert 45 <= (first - start) * 5 <= 75
    assert back is not None and abs((back - 60) * 5 / 30 - 6.1) < 1.0


def test_what_is_under_water_is_found_only_by_sonar():
    """With Zero-K's requireSonarUnderWater (LosHandler.cpp InLos, InRadar): an LLT, which has no sonar, sees a ship on the water 360
    off but not a Duck on the sea bed 400 off, well inside its sight.  An Athena's sonar reaching the Duck, from beyond the Athena's
    own sight, lets the LLT see it; with the LLT gone, the Duck is a radar dot of the Athena's sonar."""
    know = zksim.rawframe
    w = zksim.World(CAT, MAP, seed=3, behaviour=zksim.ORDERS)
    llt = w.add("turretlaser", 0, 3000.0, Z)
    duck, ship = w.add("amphraid", 1, 3400.0, Z, hp=TOUGH), w.add("shipscout", 1, 3300.0, Z + 200.0, hp=TOUGH)
    for u in (duck, ship):
        w.set_fire_state(u, 0)
        w.set_move_state(u, 0)

    def known(u: int) -> int:
        w.step(1)
        units = w.view(0).units
        return int(units["knowledge"][units["uid"] == u][0])

    assert known(ship) == know.KNOW_LOS and known(duck) == know.KNOW_HIDDEN
    w.add("athena", 0, 3900.0, Z)
    assert known(duck) == know.KNOW_LOS
    w.kill(llt)
    assert known(duck) == know.KNOW_RADAR
