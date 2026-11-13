"""Zero-K's rules for the end of a game (game_over.lua) and what a frame holds of a long one: the reveal rule, the units that keep a
team in the game (stats[3]), an alliance of AIs destroyed when it has no builder or factory left, the tracked value of what costs nothing, and which features a frame keeps past its cap.  In WSL:

  ~/zkml/bin/python -m pytest -q sim/tests/test_game_rules.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402
from zkbot.policy import rawframe  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()       # 4096 x 4096
RICH = "energysingu"          # 4000 metal, unarmed, and sees 273 elmos
HIDDEN = "cloakheavyraid"     # 250 metal, and cloaks


def seen(world: zksim.World, team: int, unit: int) -> bool:
    units = world.view(team).units
    return unit in set(int(u) for u in units["uid"][units["knowledge"] == rawframe.KNOW_LOS])


def test_the_catalog_has_the_rules_zero_k_exports():
    assert CAT.rules["reveal"] == {"value_ratio": 25.0, "every_frames": 45}
    dont_count = set(CAT.rules["dont_count"])
    assert {"spiderscout", "cloakbomb", "shieldbomb", "amphbomb", "gunshipbomb", "terraunit"} <= dont_count
    assert not dont_count & {"cloakraid", "dyntrainer_strike_base", "staticmex", "roost"}


@pytest.mark.parametrize("riches, third, on, shown", (
    (4, False, True, True),       # 16,000 metal against 250: 64 times
    (1, False, True, False),      # 4,000: 16 times
    (4, False, False, False),     # the rule turned off
    (4, True, True, False),       # three alliances left
))
def test_an_alliance_worth_more_than_25_times_the_others_sees_its_units_from_then_on_cloaked_or_not(riches, third, on, shown):
    world = zksim.World(CAT, FLAT, seed=1)
    world.set_teams([0, 1, 2] if third else [0, 1])
    world.set_reveal(on)
    for k in range(riches):
        world.add(RICH, 0, 400.0 + 150.0 * k, 400.0)
    hidden = world.add(HIDDEN, 1, 3600.0, 3600.0)
    world.cloak(hidden, True)
    if third:
        world.add(HIDDEN, 2, 3600.0, 600.0)
    world.step(8)                 # to frame 40: game_over.lua looks every 45 frames
    assert not seen(world, 0, hidden)
    world.step(1)
    assert seen(world, 0, hidden) == shown
    world.step(30)
    assert seen(world, 0, hidden) == shown


def test_what_keeps_a_team_in_the_game_is_its_finished_units_less_those_zero_k_doesnt_count():
    batch = zksim.Batch(CAT, 1, threads=1)
    names = ("dyntrainer_strike_base", "spiderscout", "cloakraid", "cloakbomb", "cloakbomb")
    batch.reset(0, FLAT, 1, np.array([CAT.index[n] for n in names]), np.array([0, 0, 0, 1, 1]), np.array([500.0, 600.0, 700.0, 3500.0, 3600.0]),
                np.array([500.0, 500.0, 500.0, 3500.0, 3500.0]), build=np.array([1.0, 1.0, 0.5, 1.0, 1.0], np.float32))
    batch.refresh(0)
    assert list(batch.stats[0, :, 1]) == [3, 2]     # live units
    assert list(batch.stats[0, :, 3]) == [1, 0]     # the commander: the Flea doesn't count, nor a frame, nor bombs


@pytest.mark.parametrize("left, destroyed", (
    (None, True),                       # nothing but its raider
    (("factorycloak", 1.0), False),     # a factory
    (("staticcon", 1.0), False),        # a caretaker: a builder, and it counts, as the gadget's own test of what doesn't is never true
    (("factorycloak", 0.5), True),      # a factory's frame: not finished
))
def test_an_alliance_of_ais_with_no_finished_builder_or_factory_left_is_destroyed_but_not_one_that_never_had_one(left, destroyed):
    """game_over.lua's vitalAlive: when its commander dies, side 0 is destroyed unless something else finished builds.  Side 1,
    an army set down with no builder, plays on."""
    assert {"dyntrainer_strike_base", "factorycloak", "staticcon", "cloakcon"} <= set(CAT.rules["vital"])
    world = zksim.World(CAT, FLAT, seed=1)
    commander = world.add("dyntrainer_strike_base", 0, 500.0, 500.0)
    raider = world.add("cloakraid", 0, 700.0, 500.0)
    if left:
        world.add(left[0], 0, 500.0, 900.0, build=left[1])
    army = world.add("cloakraid", 1, 3500.0, 3500.0)
    world.step(2)
    world.kill(commander)
    world.step(3)
    assert world.get(raider).alive == (not destroyed) and world.get(army).alive


def test_a_unit_that_costs_nothing_is_worth_its_build_time_and_one_that_costs_something_its_cost():
    batch = zksim.Batch(CAT, 1, threads=1)
    names = ("chicken", "chicken", "cloakraid")
    batch.reset(0, FLAT, 1, np.array([CAT.index[n] for n in names]), np.array([0, 0, 1]), np.array([500.0, 600.0, 3500.0]),
                np.array([500.0, 500.0, 3500.0]), build=np.array([1.0, 0.5, 1.0], np.float32))
    batch.refresh(0)
    chicken, raider = CAT.units[CAT.index["chicken"]], CAT.units[CAT.index["cloakraid"]]
    assert chicken["cost"] == 0.0
    assert batch.stats[0, 0, 0] == pytest.approx(1.5 * chicken["build_time"])
    assert batch.stats[0, 1, 0] == pytest.approx(raider["cost"])


def test_the_damage_a_team_does_is_what_the_other_takes_in_metal_and_a_kill_is_its_victims_cost():
    batch = zksim.Batch(CAT, 1, threads=1)
    names = ("vehraid", "cloakraid")
    batch.reset(0, FLAT, 1, np.array([CAT.index[n] for n in names]), np.array([0, 1]), np.array([2000.0, 2150.0]), np.array([2000.0, 2000.0]))
    for _ in range(100):
        batch.step(ticks=1)
        if batch.stats[0, 1, 1] == 0:
            break
    scorcher, glaive = (CAT.units[CAT.index[n]] for n in names)
    dealt, taken = batch.stats[0, :, 4], batch.stats[0, :, 5]
    # All the Glaive's health and no more, whatever the shot that killed it could have done.
    assert dealt[0] == taken[1] == pytest.approx(glaive["cost"])
    lost = 1.0 - float(batch.frame(0, 0).units["hp_frac"][0])
    assert lost > 0.1 and dealt[1] == taken[0] == pytest.approx(lost * scorcher["cost"], rel=1e-4)


@pytest.mark.parametrize("same", (False, True))
def test_past_the_feature_cap_a_frame_keeps_the_features_with_the_most_metal_left_in_id_order(same):
    world = zksim.World(CAT, FLAT, seed=1)
    world.add("staticradar", 0, 2000.0, 2000.0)       # sees 800 elmos
    n = zksim.FEATURES_PER_SIDE + 50
    metals = np.full(n, 5.0) if same else np.random.default_rng(1).permutation(np.arange(1.0, n + 1.0))
    ids = [world.add_feature(1850.0 + 10.0 * (k % 30), 1850.0 + 10.0 * (k // 30), float(m)) for k, m in enumerate(metals)]
    world.step(1)
    kept = world.features(0)
    assert len(kept) == zksim.FEATURES_PER_SIDE
    assert list(kept["feat_id"]) == sorted(kept["feat_id"])
    if same:        # ties go to the older
        assert list(kept["feat_id"]) == ids[:zksim.FEATURES_PER_SIDE]
    else:
        assert sorted(kept["metal"]) == list(np.arange(n + 1.0 - zksim.FEATURES_PER_SIDE, n + 1.0))
