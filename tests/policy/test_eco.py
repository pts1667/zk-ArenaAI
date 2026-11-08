"""The economy script (tools/policy_eco.py): grids, map control, where big buildings go, what missiles and the Trinity aim at,
how many constructors the factory makes, and that in sim games it grows the economy without its orders counting as the policy's."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import policy_eco as eco  # noqa: E402
import policy_mix as mix  # noqa: E402
import policy_sim_env as se  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = se.SimEnv(1, 5, ["scripts"], threads=1)
    e.begin()
    return e


def _units(env, side: int, *rows: tuple[str, float, float], uid0: int = 100, building: bool = False) -> np.ndarray:
    out = np.zeros(len(rows), rf.UNIT_DTYPE)
    for k, (name, x, z) in enumerate(rows):
        out[k]["uid"] = uid0 + k
        out[k]["def_id"] = se.catalog().units[se.catalog().index[name]]["def_id"]
        out[k]["x"], out[k]["z"] = x, z
        out[k]["player"], out[k]["ally_team"] = side, side
        out[k]["ord_kind"], out[k]["ord_target"] = -1, -1
        out[k]["hp_frac"], out[k]["build_frac"] = 1.0, 1.0
        if building:
            out[k]["flags"] = rf.BEING_BUILT
    return out


def _economy(env, base=(500.0, 500.0), enemy=(7000.0, 7000.0)) -> eco.Economy:
    return eco.Economy(env.eco_data, env.grounds[id(env.bundle[0])], 0, base, enemy, np.random.default_rng(1))


def test_grids_link_by_pylon_reach_and_the_main_one_makes_most(env):
    data = env.eco_data
    own = _units(env, 0, ("staticmex", 0.0, 0.0), ("energysolar", 120.0, 0.0), ("energyfusion", 2000.0, 0.0), ("energypylon", 2600.0, 0.0), ("staticmex", 3080.0, 0.0))
    grid = eco.Grid.of(data, own, env.table.rows(own["def_id"]))
    assert grid.label[0] == grid.label[1]                       # 50 + 100 reach 120 apart
    assert grid.label[2] == grid.label[3] == grid.label[4]      # 150 + 500 reach 600, 500 + 50 reach 480
    assert grid.label[0] != grid.label[2]
    assert grid.main == grid.label[2] and grid.makes[grid.main] == pytest.approx(35.0)


def test_map_control_is_stalled_only_when_neither_side_moves_much_over_two_minutes(env):
    e = _economy(env)
    allowed = max(2.0, eco.STALE_SHARE * len(env.bundle[0].spot_list))
    for t in range(0, 130, 2):
        e.counts.append((float(t), 10, 12 + (t % 4 == 0)))           # an extractor raided and rebuilt over and over
    assert e.stalled(128.0)
    grow = _economy(env)
    for t in range(0, 130, 2):
        grow.counts.append((float(t), 10 + int(t * (allowed + 2) / 120), 12))
    assert not grow.stalled(128.0)
    short = _economy(env)
    for t in range(0, 60, 2):
        short.counts.append((float(t), 10, 12))
    assert not short.stalled(58.0)


def test_a_silo_goes_on_our_half_within_reach_of_the_most_valuable_target(env):
    e = _economy(env, base=(500.0, 500.0), enemy=(9000.0, 500.0))
    enemies = _units(env, 1, ("energyfusion", 6000.0, 500.0), ("energysingu", 7000.0, 600.0), ("staticmex", 5000.0, 500.0), uid0=500)
    site = e._silo_site(enemies, env.table.rows(enemies["def_id"]))
    assert site is not None
    assert env.eco_data.missile_reach == pytest.approx(3500.0)      # the Eos's, which is what a silo makes
    assert np.hypot(site[0] - 7000.0, site[1] - 600.0) <= env.eco_data.missile_reach        # the Singularity: one fusion isn't worth it
    assert e._ours(*site)
    far = _units(env, 1, ("energysingu", 8900.0, 500.0), uid0=600)
    assert e._silo_site(far, env.table.rows(far["def_id"])) is None                          # out of reach from our half


def test_missiles_aim_at_what_is_seen_and_worth_most_and_the_trinity_at_the_most_value_around(env):
    e = eco.Aimer(env.eco_data)
    own = np.concatenate((_units(env, 0, ("tacnuke", 3000.0, 500.0), uid0=10), _units(env, 0, ("staticnuke", 400.0, 400.0), uid0=20)))
    enemies = _units(env, 1, ("energyfusion", 5000.0, 500.0), ("energysingu", 6000.0, 500.0), ("staticmex", 8000.0, 400.0),
                     ("staticmex", 8100.0, 400.0), ("factorycloak", 8050.0, 500.0), ("energyfusion", 8000.0, 700.0), uid0=500)
    rows, erows = env.table.rows(own["def_id"]), env.table.rows(enemies["def_id"])
    built = np.ones(len(own), bool)
    out = e._aim(own, rows, built, enemies, erows, np.array([500, 502, 503, 504, 505]), 600.0)
    by_unit = {int(o[0]): o for o in out}
    assert by_unit[10][1] == rf.ATTACK_UNIT and by_unit[10][4] == 500     # the Singularity is in reach but not seen
    assert by_unit[20][1] == rf.MANUAL_FIRE and by_unit[20][4] in (502, 503, 504, 505)   # the base, not a lone building
    assert e._aim(own, rows, built, enemies, erows, np.array([500]), 610.0) == []          # not again so soon
    # The Trinity only aims at what's in sight or on radar: here the lone fusion, since nothing at the base is.
    lone = eco.Aimer(env.eco_data)._aim(own, rows, built, enemies, erows, np.array([500]), 600.0)
    assert [(int(o[0]), int(o[4])) for o in lone if o[1] == rf.MANUAL_FIRE] == [(20, 500)]


def test_the_factory_makes_constructors_by_the_minute_and_by_income():
    assert mix.constructor_quota(30.0) == 0 and mix.constructor_quota(60.0) == 1 and mix.constructor_quota(600.0) == 5
    assert mix.constructor_quota(600.0, 50.0) == int(50.0 / mix.INCOME_PER_CONSTRUCTOR) == 16
    assert mix.constructor_quota(600.0, 1000.0) == mix.MOST_CONSTRUCTORS


def test_a_silo_is_told_to_make_the_missile_that_hurts_most(env):
    data = env.script_data
    silo = env.table.names.index("staticmissilesilo")
    assert env.table.names[data.best_missile[silo]] == "tacnuke"


def test_the_army_script_aims_missiles_so_that_its_aims_are_labels(env):
    script = env.scripts[0][0]
    assert isinstance(script.aimer, eco.Aimer)
    own = _units(env, 0, ("tacnuke", 3000.0, 500.0), uid0=10)
    enemies = _units(env, 1, ("energysingu", 6000.0, 500.0), uid0=500)
    out = script.orders(own, enemies, 600.0, 0.0, np.array([500]))
    assert [(int(o["unit"]), int(o["kind"]), int(o["target"])) for o in out if o["unit"] == 10] == [(10, rf.ATTACK_UNIT, 500)]


def test_in_sim_games_the_economy_grows_and_its_orders_are_not_the_scripts(env):
    games = se.SimEnv(4, 9, ["scripts"], threads=2)
    games.begin()
    built_by_script = 0
    for _ in range(720):         # 6 minutes
        games.step(0.0)
        built_by_script += sum(int((o["kind"] == zksim.ORDER_BUILD).sum()) for o in games.script_record.values())
        games.finished_seats()
    assert built_by_script == 0
    extractors, builders = [], []
    for seat in range(games.seat_count):
        f = games.frame(seat)
        own = f.units[f.units["player"] == seat % 2]
        rows = games.table.rows(own["def_id"])
        extractors.append(int(games.eco_data.extractor[rows].sum()))
        builders.append(int((games.table.builder[rows] & games.table.mobile[rows] & ~games.eco_data.commander[rows]).sum()))
    assert np.median(extractors) > 6        # more than the opening takes
    assert np.median(builders) >= 1     # a game that has just begun again has none
