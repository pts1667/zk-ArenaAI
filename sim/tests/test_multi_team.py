"""Games of more than two sides: teams (players, each with its own purse, orders and frame) in allyteams (who share sight and
don't fight each other), and Zero-K's sharing of resources inside an allyteam (unit_mex_overdrive.lua).  In WSL:

  ~/zkml/bin/python -m pytest -q sim/tests/test_multi_team.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402
from zkbot.policy import rawframe  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
RAIDER = "cloakraid"
COMMANDER = "dyntrainer_strike_base"       # build speed 10, range 144, makes 4 metal and 6 energy
WALKER = "vehcon"                          # moves, and has no weapon nor cloak: only separation moves it
SECONDS_PER_TICK = zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND


def economy_world(allies, metal=1000.0, energy=1000.0, innate=0.0, storage=100000.0, spots=((1000.0, 1000.0, 2.0),)):
    """Teams in the allyteams `allies` gives, doing only what they're told, with every team's purse alike."""
    world = zksim.World(CAT, FLAT, seed=1, behaviour=zksim.SCRIPTED)
    world.set_teams(allies)
    world.economy(np.array(spots, np.float32).reshape(-1, 3), metal, energy, innate, innate, storage)
    return world


def test_three_allyteams_all_fight_each_other():
    world = zksim.World(CAT, FLAT, seed=1)
    world.set_teams([0, 1, 2])
    fighters = []
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):      # a pair of each, far from the others
        fighters += [world.add(RAIDER, a, 1000.0, 800.0 + 1200.0 * k), world.add(RAIDER, b, 1150.0, 800.0 + 1200.0 * k)]
    world.step(30)
    assert all(world.get(u).damage_taken > 0.0 for u in fighters)


def test_two_teams_of_an_allyteam_dont_fight_share_sight_and_see_each_other_as_allies():
    world = zksim.World(CAT, FLAT, seed=1)
    world.set_teams([0, 0, 1])
    mine, friend = world.add(RAIDER, 0, 1000.0, 1000.0), world.add(RAIDER, 1, 1100.0, 1000.0)
    scout = world.add(RAIDER, 1, 2500.0, 1000.0)
    enemy = world.add("energysolar", 2, 2900.0, 1000.0)       # in the ally's scout's sight, far out of mine
    assert world.add(RAIDER, 3, 0.0, 0.0) == -1                # no such team
    world.step(12)
    assert world.get(mine).damage_taken == 0.0 and world.get(friend).damage_taken == 0.0
    ours, theirs = world.view(0), world.view(1)
    u = ours.units
    assert list(u["uid"]) == [mine, friend, scout, enemy]      # own, the ally's, then the rest
    assert list(u["player"]) == [0, 1, 1, 2] and list(u["ally_team"]) == [0, 0, 0, 1]
    assert (u["knowledge"] == rawframe.KNOW_LOS).all()          # the enemy too, by the ally's sight
    assert ((u["flags"][1:3] & (rawframe.IN_LOS | rawframe.TYPE_KNOWN)) == (rawframe.IN_LOS | rawframe.TYPE_KNOWN)).all()
    assert not (u["flags"][1:3] & rawframe.IDLE).any()           # an ally's unit shows what a seen one does
    assert list(theirs.units["uid"][:3]) == [friend, scout, mine]
    assert ours.glob[rawframe.G["player"]] == 0 and ours.glob[rawframe.G["ally_team"]] == 0
    assert theirs.glob[rawframe.G["player"]] == 1 and theirs.glob[rawframe.G["ally_team"]] == 0


def test_teams_of_an_allyteam_pay_from_their_own_purses_and_each_has_its_first_factory_free():
    world = economy_world([0, 0])
    a, b = world.add(COMMANDER, 0, 500.0, 500.0), world.add(COMMANDER, 1, 1500.0, 500.0)
    world.build(a, "factorycloak", 600.0, 500.0)
    world.build(b, "factorycloak", 1600.0, 500.0)
    world.step(1)
    assert world.get(b + 1).build == 1.0 and world.get(b + 2).build == 1.0
    world.build(a, "staticmex", 550.0, 600.0)
    world.step(12)
    made = 1000.0 + 4.0 * 13 * SECONDS_PER_TICK                 # what a commander makes, and the free factory cost nothing
    assert world.resources(1)["metal"] == pytest.approx(made, abs=1e-3)
    assert world.resources(0)["metal"] == pytest.approx(made - 12 * 10.0 * SECONDS_PER_TICK, abs=0.1)


def test_extractor_income_is_pooled_over_the_allyteam_and_split_evenly_among_its_teams_with_a_unit():
    world = economy_world([0, 0, 0, 1], metal=0.0, energy=0.0, spots=((1000.0, 1000.0, 3.0),))
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add(WALKER, 1, 1500.0, 1000.0)
    third = world.add(WALKER, 2, 1600.0, 1000.0)
    world.add(WALKER, 3, 1700.0, 1000.0)
    world.step(1)
    assert [world.resources(t)["metal_income"] for t in range(4)] == pytest.approx([1.0, 1.0, 1.0, 0.0])
    world.kill(third)
    world.step(2)
    assert [world.resources(t)["metal_income"] for t in range(4)] == pytest.approx([1.5, 1.5, 0.0, 0.0])


def test_overdrive_metal_and_the_energy_it_draws_go_by_what_each_team_put_in():
    world = economy_world([0, 0], metal=0.0, energy=500.0, storage=500.0)
    world.add("staticmex", 0, 1000.0, 1000.0)
    world.add("energysolar", 0, 1100.0, 1000.0)
    for x, z in ((900.0, 1000.0), (1000.0, 1100.0), (1000.0, 900.0)):
        world.add("energysolar", 1, x, z)
    world.step(2)        # the first update comes at the first tick, and income shows it from the next
    # 2 and 6 spare at full storage, 8 in all, which the grid's 8 can carry.
    made = 2.0 * 0.25 * math.sqrt(8.0)
    mine, theirs = world.resources(0), world.resources(1)
    assert mine["overdrive_energy"] == pytest.approx(2.0) and theirs["overdrive_energy"] == pytest.approx(6.0)
    assert mine["overdrive_metal"] == pytest.approx(made / 4.0) and theirs["overdrive_metal"] == pytest.approx(made * 3.0 / 4.0)
    assert mine["metal_income"] == pytest.approx(1.0 + made / 4.0) and theirs["metal_income"] == pytest.approx(1.0 + made * 3.0 / 4.0)
    assert mine["energy_income"] == pytest.approx(0.0, abs=1e-5) and theirs["energy_income"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("allies, theirs", (([0, 0], 496.0), ([0, 1], 490.0)))
def test_metal_over_a_teams_storage_goes_to_an_ally_with_room_and_not_to_an_enemy(allies, theirs):
    world = economy_world(allies, metal=490.0, energy=0.0, storage=500.0)
    world.add(COMMANDER, 0, 1000.0, 1000.0)       # 4 metal a second, its own team's
    world.add(WALKER, 1, 1500.0, 1000.0)
    world.step(24)       # 16 made: 10 fills its team's storage, and 6 is over
    assert world.resources(0)["metal"] == 500.0
    assert world.resources(1)["metal"] == pytest.approx(theirs, abs=0.01)


def test_a_group_order_takes_only_the_anchors_team_and_not_its_allies():
    world = zksim.World(CAT, FLAT, seed=1)
    world.set_teams([0, 0])
    roles = np.full(len(CAT.units), -1, np.int32)
    roles[CAT.index[RAIDER]] = 0
    world.set_roles(roles)
    anchor, own, ally = world.add(RAIDER, 0, 1000.0, 1000.0), world.add(RAIDER, 0, 1030.0, 1000.0), world.add(RAIDER, 1, 1000.0, 1030.0)
    assert world.order_group(anchor, 1, 100.0, zksim.ORDER_MOVE, 3000.0, 1000.0) == 2
    assert world.get(own).order == zksim.ORDER_MOVE and world.get(ally).order == zksim.ORDER_NONE


def test_a_builder_helps_an_allys_frame_and_pays_for_it_but_cant_help_an_enemys():
    world = economy_world([0, 0, 1])
    a, b = world.add(COMMANDER, 0, 1000.0, 900.0), world.add(COMMANDER, 1, 1100.0, 900.0)
    enemy = world.add(COMMANDER, 2, 3000.0, 3000.0)
    world.build(a, "staticmex", 1000.0, 1000.0)
    world.step(1)
    frame = enemy + 1
    world.assist(b, frame)
    world.assist(enemy, frame)
    assert world.get(b).order == zksim.ORDER_ASSIST and world.get(enemy).order == zksim.ORDER_NONE
    world.step(12)
    step = 10.0 * SECONDS_PER_TICK / 85.0           # of the frame, a tick of a commander's help
    assert world.get(frame).build == pytest.approx(25 * step, abs=1e-4)
    income = 4.0 * 13 * SECONDS_PER_TICK
    assert world.resources(1)["metal"] == pytest.approx(1000.0 + income - 12 * 85.0 * step, abs=0.05)
    assert world.resources(2)["metal"] == pytest.approx(1000.0 + income, abs=1e-3)


def test_units_push_apart_the_same_whatever_allyteams_they_are_in():
    """Separation sums what's near a unit as for_near visits it, a cell at a time and each cell's in handle order, however many
    allyteams share the cell: so the sums, and where everyone ends up, come out the same to the bit."""
    def places(allies):
        world = zksim.World(CAT, FLAT, seed=1, behaviour=zksim.SCRIPTED)
        world.set_teams(allies)
        rng = np.random.default_rng(5)
        for k in range(48):
            world.add(WALKER, k % 4, 1000.0 + float(rng.uniform(-50, 50)), 1000.0 + float(rng.uniform(-50, 50)))
        world.step(10)
        return [(world.get(u).x, world.get(u).z) for u in range(48)]
    together = places([0, 0, 0, 0])
    assert places([0, 1, 0, 1]) == together
    assert places([0, 1, 2, 3]) == together
    assert places([3, 2, 1, 0]) == together


def several_teams(batch: zksim.Batch, g: int, allies: list[int], seed: int) -> None:
    """A commander, an extractor and a solar at each team's corner of the map, and a few fighters, with the economy on."""
    rng = np.random.default_rng(seed)
    corners = [(600.0, 600.0), (3500.0, 3500.0), (3500.0, 600.0), (600.0, 3500.0)]
    names, team, x, z, spots = [], [], [], [], []
    for t in range(len(allies)):
        cx, cz = corners[t]
        names.append(COMMANDER)
        names += ["staticmex", "energysolar"] + [str(n) for n in rng.choice(["cloakraid", "shieldraid", "jumpraid", "cloakriot"], 5)]
        team += [t] * 8
        x += [cx, cx + 80.0, cx + 80.0] + list(cx + rng.uniform(-200, 200, 5))
        z += [cz, cz, cz + 80.0] + list(cz + rng.uniform(-200, 200, 5))
        spots.append((cx + 80.0, cz, 2.5))
    commander = [8 * t for t in range(len(allies))]
    base = [v for t in range(len(allies)) for v in corners[t]]
    batch.reset(g, FLAT, seed, np.array([CAT.index[n] for n in names]), np.array(team), np.array(x), np.array(z),
                commander=commander, base=base, ally_of=allies)
    batch.economy(g, np.array(spots, np.float32))


def play(batch: zksim.Batch, games: list[list[int]], decisions: int) -> list[int]:
    """Every team's fighters told to fight their way to the middle, then decisions of 3 ticks."""
    orders, offsets = [], [0]
    for allies in games:
        for t in range(len(allies)):
            orders += [(8 * t + k, zksim.ORDER_FIGHT, 2000.0, 2000.0, -1) for k in range(3, 8)]
        offsets.append(len(orders))
    hashes = []
    for d in range(decisions):
        if d == 0:
            batch.step(np.array(orders, zksim.ORDER_DTYPE), np.array(offsets, np.int32))
        else:
            batch.step()
        hashes.append(batch.frame_hash())
    return hashes


GAMES_OF_TEAMS = ([0, 0, 1], [0, 1, 0, 1], [0, 1, 2, 2], [0, 1], [0, 1, 2], [0, 0, 0, 1])


def test_a_batch_of_four_slots_plays_a_three_team_game_beside_a_two_team_game_and_leaves_the_rest_empty():
    slots = zksim.Batch(CAT, 2, threads=2, teams=4)
    several_teams(slots, 0, [0, 1, 1], 7)
    several_teams(slots, 1, [0, 1], 8)
    alone = zksim.Batch(CAT, 1, threads=1)       # the same two-team game in a batch of two slots
    several_teams(alone, 0, [0, 1], 8)
    for _ in range(4):
        slots.step(ticks=3)
        alone.step(ticks=3)
    assert (slots.counts[0, :3] == 24).all() and slots.counts[0, 3] == 0 and (slots.counts[1, :2] == 16).all() and (slots.counts[1, 2:] == 0).all()
    assert not slots.stats[0, 3].any() and not slots.stats[1, 2:].any() and not slots.globs[1, 2:].any() and not slots.feature_counts[1, 2:].any()
    assert list(slots.stats[0, :3, 1]) == [8, 8, 8]
    for t, ally in enumerate((0, 1, 1)):
        glob = slots.frame(0, t).glob
        assert glob[rawframe.G["player"]] == t and glob[rawframe.G["ally_team"]] == ally
    second = slots.frame(0, 1).units
    assert list(second["player"][:16]) == [1] * 8 + [2] * 8 and (second["ally_team"][:16] == 1).all()
    for t in (0, 1):
        n = int(alone.counts[0, t])
        assert slots.records[1, t, :n].tobytes() == alone.records[0, t, :n].tobytes()
        assert np.array_equal(slots.stats[1, t], alone.stats[0, t]) and np.array_equal(slots.globs[1, t], alone.globs[0, t])


def test_a_game_of_several_teams_and_allyteams_is_the_same_whatever_the_thread_count():
    def run(threads: int) -> list[int]:
        batch = zksim.Batch(CAT, len(GAMES_OF_TEAMS), threads=threads, teams=4)
        for g, allies in enumerate(GAMES_OF_TEAMS):
            several_teams(batch, g, allies, 30 + g)
        return play(batch, list(GAMES_OF_TEAMS), 12)
    alone = run(1)
    assert run(4) == alone
    assert run(6) == alone
