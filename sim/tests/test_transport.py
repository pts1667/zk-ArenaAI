"""Air transports (sim/src/transport.cpp): Charon and Hercules pick a unit up, carry it, set it down or drop it."""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 1500.0, 2000.0
GRAVITY = 120.0 / 900.0
DROP_RELOAD = 210


def unit(name: str) -> dict:
    return CAT.units[CAT.index[name]]


def passive(w: zksim.World, name: str, team: int, x: float, z: float) -> int:
    u = w.add(name, team, x, z)
    w.set_fire_state(u, 0)
    w.set_unit_ai(u, False)
    return u


def until(w: zksim.World, done, ticks: int = 400) -> int:
    """Steps until done() holds; the ticks it took."""
    for n in range(ticks):
        if done():
            return n
        w.step(1)
    raise AssertionError(f"not within {ticks} ticks")


def carrying(w: zksim.World, transport: str, load: str, team: int = 0) -> tuple[int, int]:
    t = w.add(transport, 0, X, Z)
    u = passive(w, load, team, X + 300, Z)
    w.load(t, u)
    until(w, lambda: w.get(t).cargo == u)
    return t, u


def test_a_charon_picks_up_an_ally_carries_it_slower_and_sets_it_down():
    w = zksim.World(CAT, FLAT, seed=1)
    t = w.add("gunshiptrans", 0, X, Z)
    u = w.add("cloakraid", 0, X + 600, Z)
    far = passive(w, "cloakraid", 1, X - 1200, Z + 1800)
    w.move(u, X + 600, Z + 3000)                # it's walking off, and stops for its transport
    w.load(t, u)
    until(w, lambda: w.get(t).cargo == u)
    assert w.get(u).carrier == t and w.get(u).z < Z + 600
    w.unload(t, X + 1800, Z - 800)
    top, seen = 0.0, None
    for _ in range(400):
        w.step(1)
        s, c = w.get(t), w.get(u)
        if c.carrier < 0:
            break
        top = max(top, s.speed)
        assert (c.x, c.z) == (s.x, s.z) and c.altitude == pytest.approx(s.altitude - unit("cloakraid")["load_height"], abs=0.01)
        if seen is None and s.altitude > 100.0:
            # Nothing of the other team is near: it knows of neither.
            theirs = [int(r["uid"]) for r in w.view(1).units if r["knowledge"] != rf.KNOW_HIDDEN]
            own = {int(r["uid"]): int(r["flags"]) for r in w.view(0).units}
            seen = (theirs, own[t] & rf.LOADED, own[u] & rf.CARRIED, own[u] & rf.PARALYZED)
    assert seen == ([far], rf.LOADED, rf.CARRIED, rf.PARALYZED)
    # unit_transport_speed.lua: a light load takes it to 0.7 of its speed.
    assert top == pytest.approx(0.7 * unit("gunshiptrans")["speed"], rel=0.02)
    c = w.get(u)
    # Down where it was told, a tick's walk ago: it goes on to where it had been going.
    assert math.hypot(c.x - X - 1800, c.z - Z + 800) < 5.5 * unit("cloakraid")["speed"]
    assert c.altitude == 0.0 and not c.skidding and c.hp == unit("cloakraid")["hp"] and w.get(t).cargo == -1
    w.step(40)
    assert w.get(u).z > Z - 800 + 300


# A transport, a load, and its speed with it as a share of its own: none for what it can't lift.
@pytest.mark.parametrize("transport,load,share", [
    ("gunshiptrans", "jumpassault", 0.4), ("gunshiptrans", "jumpsumo", None), ("gunshiptrans", "staticmex", None), ("gunshiptrans", "gunshipraid", None),
    ("gunshipheavytrans", "jumpassault", 0.75), ("gunshipheavytrans", "striderdante", 0.5)])
def test_what_each_transport_lifts_and_how_fast(transport, load, share):
    w = zksim.World(CAT, FLAT, seed=1)
    t = w.add(transport, 0, X, Z)
    u = passive(w, load, 0, X + 300, Z)
    w.load(t, u)
    if share is None:
        w.step(120)
        assert w.get(t).cargo == -1 and w.get(t).x == X
        return
    until(w, lambda: w.get(t).cargo == u)
    w.move(t, X + 3500, Z)
    top = 0.0
    for _ in range(150):
        w.step(1)
        top = max(top, w.get(t).speed)
    assert top == pytest.approx(share * unit(transport)["speed"], rel=0.02)


def test_an_enemy_is_picked_up_only_once_it_stands_still():
    w = zksim.World(CAT, FLAT, seed=1)
    t = w.add("gunshiptrans", 0, X, Z)
    e = passive(w, "spiderassault", 1, X + 400, Z)
    w.move(e, X + 400, Z + 900)
    w.load(t, e)
    walking = until(w, lambda: w.get(e).speed < 0.5 and w.frame > 30)
    assert walking > 60 and w.get(t).cargo == -1
    # It was over it all the while, and low: an enemy that moves faster than half an elmo a frame can't be picked up.
    assert math.hypot(w.get(t).x - w.get(e).x, w.get(t).z - w.get(e).z) < 16.0 and w.get(t).altitude == unit("spiderassault")["load_height"]
    until(w, lambda: w.get(t).cargo == e, ticks=60)
    # Its own side sees it go, carried, and can't shoot it there.
    theirs = {int(r["uid"]): int(r["flags"]) for r in w.view(1).units}
    assert theirs[e] & rf.CARRIED and theirs[e] & rf.PARALYZED


def test_drop_cargo_lets_it_fall_hurt_and_reloads_for_seven_seconds():
    w = zksim.World(CAT, FLAT, seed=1)
    t, u = carrying(w, "gunshiptrans", "jumpassault")
    w.move(t, X + 3500, Z)
    w.step(60)
    s, jack = w.get(t), unit("jumpassault")
    height = s.altitude - jack["load_height"]
    assert height > 90.0 and s.speed > 4.0
    w.drop(t)
    dropped = w.frame
    assert w.get(t).cargo == -1 and w.get(u).skidding and w.get(u).altitude == pytest.approx(height, abs=0.01)
    assert w.get(t).order == zksim.ORDER_MOVE      # it flies on
    until(w, lambda: w.get(u).altitude == 0.0 and not w.get(u).skidding)
    # unit_fall_damage.lua: (speed - 3) x mass x 0.6, the speed its fall's and half of what it had across.
    hit = math.hypot(math.sqrt(2.0 * GRAVITY * height), 0.5 * s.speed) - 3.0
    assert jack["hp"] - w.get(u).hp == pytest.approx(hit * jack["mass"] * 0.6, rel=0.15)
    assert w.get(u).x > s.x + 20.0                  # and it kept the transport's speed as it fell
    # Picked up again at once, it isn't dropped until the button has reloaded.
    w.load(t, u)
    until(w, lambda: w.get(t).cargo == u)
    assert w.frame < dropped + DROP_RELOAD
    w.drop(t)
    assert w.get(t).cargo == u
    w.step((dropped + DROP_RELOAD - w.frame) // zksim.TICK_FRAMES + 1)
    w.drop(t)
    assert w.get(t).cargo == -1


def test_a_load_cannot_be_shot_until_its_transport_is_shot_down_and_then_it_falls():
    w = zksim.World(CAT, FLAT, seed=1)
    t, u = carrying(w, "gunshiptrans", "jumpassault")
    w.add("turretmissile", 1, X + 900, Z)
    w.move(t, X + 900, Z)
    full = unit("jumpassault")["hp"]
    until(w, lambda: w.get(t).hp < unit("gunshiptrans")["hp"])
    assert w.get(u).carrier == t
    until(w, lambda: not w.get(t).alive)
    c = w.get(u)
    assert c.alive and c.carrier == -1 and c.hp == full and c.skidding and c.altitude > 50.0
    until(w, lambda: w.get(u).altitude == 0.0)
    assert w.get(u).hp < full
