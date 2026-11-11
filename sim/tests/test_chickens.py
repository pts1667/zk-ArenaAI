"""Chicken defense (sim/src/chickens.cpp, a port of Zero-K's unit_spawner.lua with spawn_defs.lua's settings): burrows, waves,
eggs, the queen and her morph, on the gadget's schedule.  One player, on Normal unless a test says: burrowSpawnRate 45 s, grace 180 s,
a wave every 50 s.  In WSL:

  ~/zkml/bin/python -m pytest -q sim/tests/test_chickens.py
"""
from __future__ import annotations

import copy
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402
from zkbot.policy import rawframe  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()       # 4096 x 4096
NORMAL = zksim.CHICKEN_DIFFICULTIES.index("Normal")
CENTRE = 2048.0
BASE = "energysingu"          # unarmed and tough, so the chickens' numbers are the gadget's alone


def quick_catalog(**changes) -> zksim.Catalog:
    """The catalog with Normal's settings changed, for what a test can't wait an hour for."""
    cat = zksim.Catalog()
    rules = copy.deepcopy(cat.rules)
    rules["chickens"]["difficulties"][NORMAL].update(changes)
    cat._install_rules(rules)
    return cat


def chicken_world(cat: zksim.Catalog = CAT) -> zksim.World:
    world = zksim.World(cat, FLAT, seed=1)
    world.set_teams([0, 1])
    world.add(BASE, 0, CENTRE, CENTRE)
    world.chickens(1, "Normal")
    return world


def chickens_of(world: zksim.World, name: str, team: int = 1) -> list[int]:
    """The chicken team's live units of a type, by its own frame."""
    own = world.view(team).units
    def_id = CAT.units[CAT.index[name]]["def_id"]
    return [int(u) for u, d, p in zip(own["uid"], own["def_id"], own["player"]) if d == def_id and p == team]


def ticks_to(frame: int) -> int:
    return frame // zksim.TICK_FRAMES


def test_a_world_without_chickens_says_so():
    world = zksim.World(CAT, FLAT, seed=1)
    state = world.chicken_state()
    assert state["team"] == -1 and not any(v for k, v in state.items() if k != "team")


def test_burrows_come_on_the_gadgets_schedule_between_700_and_3500_elmos_from_the_players():
    world = chicken_world()
    # The first once burrowSpawnRate / 4 x 1 = 11.25 s have passed, at the first check after ((n + 17) % 30 == 0): frame 343.
    world.step(ticks_to(340))
    assert world.chicken_state()["burrows"] == 0
    world.step(1)
    assert world.chicken_state()["burrows"] == 1
    (burrow,) = chickens_of(world, "roost")
    at = world.get(burrow)
    assert 700.0 <= math.hypot(at.x - CENTRE, at.z - CENTRE) <= 3500.0
    # The next once 45 / 4 x 2 more have: past frame 343 + 675, so at 1033.
    world.step(ticks_to(1030) - ticks_to(345))
    assert world.chicken_state()["burrows"] == 1
    world.step(1)
    assert world.chicken_state()["burrows"] == 2


def test_waves_come_once_the_grace_period_is_over_and_then_every_50_seconds_from_each_burrow():
    world = chicken_world()
    world.step(ticks_to(5400))        # a wave on the first frame past grace x 30, 5401
    state = world.chicken_state()
    assert state["waves"] == 0 and state["chickens"] == 0 and state["burrows"] >= 4
    world.step(1)
    state = world.chicken_state()
    assert state["waves"] == 1 and state["chickens"] == len(chickens_of(world, "chicken")) > 0     # the only kind out yet
    world.step(ticks_to(6900) - ticks_to(5405))      # and the next on the first past 5401 + 1500
    assert world.chicken_state()["waves"] == 1
    world.step(1)
    assert world.chicken_state()["waves"] == 2


def test_a_killed_burrow_angers_the_chickens_brings_the_queen_forward_and_leaves_eggs_that_stay():
    world = chicken_world(quick_catalog(egg_decay_time=6.0))
    world.step(ticks_to(345))
    (burrow,) = chickens_of(world, "roost")
    at = world.get(burrow)
    world.add("staticradar", 0, at.x + 60.0, at.z)        # to see the eggs
    world.kill(burrow)
    world.step(1)
    state = world.chicken_state()
    # Anger was 0, so the queen's time doesn't move; the kill adds humanAggroPerBurrow, 1 for one player.
    assert state["burrows_killed"] == 1 and state["aggro"] == 1.0 and state["queen_time"] == 3600.0
    egg = CAT.rules["chickens"]["eggs"]["chicken"]        # the only kind out at 11 seconds
    eggs = [f for f in world.features(0) if f["def_id"] == egg["feature"]]
    assert len(eggs) == 15 and all(f["metal"] == egg["metal"] for f in eggs)
    assert all(abs(f["x"] - at.x) <= 30.0 and abs(f["z"] - at.z) <= 30.0 for f in eggs)
    # The next burrow killed takes burrowQueenTime x anger (15 x 1) off her time.
    while not chickens_of(world, "roost"):
        world.step(1)
    world.kill(chickens_of(world, "roost")[0])
    world.step(1)
    state = world.chicken_state()
    assert state["burrows_killed"] == 2 and state["queen_time"] == 3585.0 and state["aggro"] == 2.0
    world.step(ticks_to(30 * 30))      # well past eggDecayTime (6 s here): these don't decay
    assert len([f for f in world.features(0) if f["def_id"] == egg["feature"]]) >= 15


def test_a_chicken_that_dies_leaves_an_egg_that_decays():
    world = chicken_world(quick_catalog(grace_period=20.0, grace_period_min=20.0, egg_decay_time=6.0))
    world.step(ticks_to(605))         # the first wave, at frame 601, from the one burrow
    (burrow,) = chickens_of(world, "roost")
    b = world.get(burrow)
    chicken = min(chickens_of(world, "chicken"), key=lambda u: math.hypot(world.get(u).x - b.x, world.get(u).z - b.z))
    at = world.get(chicken)
    world.kill(chicken)
    world.step(1)                     # dies by frame 610: the egg decays from second 26.3, at the check on frame 871
    egg = CAT.rules["chickens"]["eggs"]["chicken"]
    laid = [f for f in world.features(1) if f["x"] == pytest.approx(at.x, abs=40.0) and f["z"] == pytest.approx(at.z, abs=40.0)]
    assert len(laid) == 1 and laid[0]["def_id"] == egg["feature"] and laid[0]["metal"] == egg["metal"]
    world.step(ticks_to(870) - ticks_to(610))
    assert int(laid[0]["feat_id"]) in set(int(f) for f in world.features(1)["feat_id"])
    world.step(1)
    assert int(laid[0]["feat_id"]) not in set(int(f) for f in world.features(1)["feat_id"])


def test_the_queen_comes_at_her_time_morphs_with_her_health_and_her_death_ends_the_chickens():
    world = chicken_world(quick_catalog(queen_time=40.0))
    # Miniqueens at 0.6 of her time, one a player: the check after second 24, frame 733.
    world.step(ticks_to(730))
    assert not chickens_of(world, "chicken_dragon")
    world.step(1)
    assert len(chickens_of(world, "chicken_dragon")) == 1
    # She comes at the check after second 40, frame 1213.
    world.step(ticks_to(1210) - ticks_to(735))
    assert world.chicken_state()["queen"] == 0
    world.step(1)
    state = world.chicken_state()
    assert state["queen"] == 1 and state["until_queen"] == 0.0 and state["queen_hp"] == 1.0
    (queen,) = chickens_of(world, "chickenflyerqueen")
    world.set_hp(queen, 0.5 * CAT.units[CAT.index["chickenflyerqueen"]]["hp"])
    # She turns into her other form 1800 to 3600 frames on, with the share of her health she had (she has healed a little since).
    for _ in range(ticks_to(3600) + 1):
        share = world.chicken_state()["queen_hp"]
        world.step(1)
        if chickens_of(world, "chickenlandqueen"):
            break
    (landed,) = chickens_of(world, "chickenlandqueen")
    assert not chickens_of(world, "chickenflyerqueen") and world.chicken_state()["queen_hp"] == pytest.approx(share, abs=1e-3)
    assert 0.5 < share < 0.6
    world.kill(landed)
    world.step(1)
    state = world.chicken_state()
    assert state["queen"] == 2 and state["chickens"] == 0 and state["burrows"] == 0
    assert not (world.view(1).units["player"] == 1).any()


def test_the_queen_has_more_health_the_more_players_there_are():
    # Beginner's queen is a Dragon.  Three players: queenHealthMod 1 x (0.5 x 40 / 40 + 0.5) x (3 / 2 + 0.5) = 2.
    beginner = zksim.CHICKEN_DIFFICULTIES.index("Beginner")
    cat = zksim.Catalog()
    rules = copy.deepcopy(cat.rules)
    rules["chickens"]["difficulties"][beginner]["queen_time"] = 40.0
    cat._install_rules(rules)
    world = zksim.World(cat, FLAT, seed=1)
    world.set_teams([0, 0, 0, 1])
    world.add(BASE, 0, CENTRE, CENTRE)
    world.chickens(3, "Beginner")
    world.step(ticks_to(1215))        # she comes at frame 1213, and is first told where to go at 1231
    (queen,) = chickens_of(world, "chicken_dragon", 3)
    q = world.get(queen)
    side = 40.0 if q.x < CENTRE else -40.0
    twin = world.add("chicken_dragon", 3, q.x + 2.0 * side, q.z)
    world.kill(world.add("shieldbomb", 0, q.x + side, q.z))     # goes off halfway between them
    hp = CAT.units[CAT.index["chicken_dragon"]]["hp"]
    world.step(1)
    lost, twin_lost = hp - world.get(queen).hp, hp - world.get(twin).hp
    assert twin_lost > 0.0 and lost == pytest.approx(0.5 * twin_lost, rel=0.02)
    assert world.chicken_state()["queen_hp"] == pytest.approx(world.get(queen).hp / hp)


def test_a_batch_game_plays_the_chickens_and_tells_their_state_by_name_with_no_reveal():
    batch = zksim.Batch(CAT, 1, threads=1)
    base = [CAT.index[BASE]] * 4
    batch.reset(0, FLAT, 1, np.array(base), np.zeros(4, np.int32), np.array([1900.0, 2000.0, 2100.0, 2200.0]), np.full(4, CENTRE),
                chickens=(1, "Normal"))
    for _ in range(ticks_to(600) // 3):
        batch.step(ticks=3)
    state = batch.chicken_state(0)
    assert state["team"] == 1 and state["difficulty"] == NORMAL and state["burrows"] == 1
    roost = CAT.units[CAT.index["roost"]]
    # Worth 16,000 to the burrow's 346, but no reveal: the burrow is out of the base's sight.
    assert batch.stats[0, 1, 0] == pytest.approx(roost["cost"]) and batch.stats[0, 1, 3] == 1
    frame = batch.frame(0, 0).units
    assert not ((frame["def_id"] == roost["def_id"]) & (frame["knowledge"] == rawframe.KNOW_LOS)).any()
