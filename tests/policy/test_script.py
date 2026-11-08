"""The fight script, on records made by hand."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_sim_env as se
    import policy_script as ps
    from zkbot.policy import rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

RALLY, ENEMY_START = (500.0, 500.0), (4000.0, 4000.0)


@pytest.fixture(scope="module")
def data():
    return ps.ScriptData(se.tables.table(), se.catalog())


def _def_id(name: str) -> int:
    return int(se.catalog().units[se.catalog().index[name]]["def_id"])


def _units(*specs) -> np.ndarray:
    """Each (uid, name, x, z[, hp_frac])."""
    out = np.zeros(len(specs), rf.UNIT_DTYPE)
    for i, spec in enumerate(specs):
        uid, name, x, z, *hp = spec
        out[i]["uid"], out[i]["def_id"], out[i]["x"], out[i]["z"] = uid, _def_id(name), x, z
        out[i]["hp_frac"] = hp[0] if hp else 1.0
    return out


def _script(data) -> ps.FightScript:
    return ps.FightScript(data, RALLY, ENEMY_START)


def _no_enemies() -> np.ndarray:
    return np.zeros(0, rf.UNIT_DTYPE)


def _goals(out: np.ndarray) -> dict[int, tuple[int, float, float]]:
    return {int(o["unit"]): (int(o["kind"]), float(o["x"]), float(o["z"])) for o in out}


GLAIVE_METAL = 66.0833358765


def test_groups_get_bigger_over_time():
    assert ps.group_metal(179.0) == 0.0
    assert ps.group_metal(180.0) == ps.GROUP_FIRST_METAL
    assert ps.group_metal(240.0) == ps.GROUP_FIRST_METAL + ps.GROUP_METAL_PER_MINUTE
    assert ps.group_metal(600.0) > ps.group_metal(300.0)
    assert ps.group_metal(100000.0) == ps.GROUP_MOST_METAL


def test_before_groups_a_fighter_goes_straight_to_the_enemys_start(data):
    goals = _goals(_script(data).orders(_units((5, "cloakraid", 600.0, 600.0)), _no_enemies(), 60.0))
    assert goals[5] == (rf.FIGHT, *ENEMY_START)


def test_only_finished_fighters_get_orders(data):
    units = _units((5, "cloakraid", 600.0, 600.0), (6, "cloakraid", 600.0, 600.0), (7, "staticmex", 600.0, 600.0), (8, "dyntrainer_strike_base", 600.0, 600.0))
    units[1]["flags"] = rf.BEING_BUILT
    assert set(_goals(_script(data).orders(units, _no_enemies(), 60.0))) == {5}


def test_after_groups_start_fighters_wait_at_the_rally_until_there_are_enough(data):
    script = _script(data)
    few = _units(*[(i, "cloakraid", 600.0, 600.0) for i in range(5)])
    assert all(g == (rf.FIGHT, *RALLY) for g in _goals(script.orders(few, _no_enemies(), 180.0)).values())
    enough = _units(*[(i, "cloakraid", 600.0, 600.0) for i in range(7)])
    assert 7 * GLAIVE_METAL > ps.group_metal(183.0)
    assert all(g == (rf.FIGHT, *ENEMY_START) for g in _goals(script.orders(enough, _no_enemies(), 183.0)).values())


def test_a_group_that_has_gone_stays_gone_and_the_next_waits(data):
    script = _script(data)
    group = [(i, "cloakraid", 600.0, 600.0) for i in range(7)]
    script.orders(_units(*group), _no_enemies(), 183.0)
    goals = _goals(script.orders(_units(*group, (9, "cloakraid", 600.0, 600.0)), _no_enemies(), 186.0))
    assert goals[0] == (rf.FIGHT, *ENEMY_START) and goals[9] == (rf.FIGHT, *RALLY)


def test_the_same_fighters_are_too_few_later_on(data):
    units = _units(*[(i, "cloakraid", 600.0, 600.0) for i in range(7)])
    goals = _goals(_script(data).orders(units, _no_enemies(), 900.0))
    assert 7 * GLAIVE_METAL < ps.group_metal(900.0) and goals[0] == (rf.FIGHT, *RALLY)


def test_a_group_that_cant_fill_goes_anyway_in_the_end(data):
    script = _script(data)
    units = _units((1, "cloakraid", 600.0, 600.0))
    script.waiting_since = 900.0
    assert _goals(script.orders(units, _no_enemies(), 901.0))[1] == (rf.FIGHT, *RALLY)
    assert _goals(script.orders(units, _no_enemies(), 900.0 + ps.LONGEST_WAIT))[1] == (rf.FIGHT, *ENEMY_START)


def test_the_wait_starts_with_the_first_to_wait_not_with_an_earlier_one(data):
    script = _script(data)
    script.orders(_units((1, "cloakraid", 600.0, 600.0)), _no_enemies(), 200.0)
    script.orders(_no_enemies(), _no_enemies(), 300.0)
    assert _goals(script.orders(_units((2, "cloakraid", 600.0, 600.0)), _no_enemies(), 200.0 + ps.LONGEST_WAIT + 10.0))[2] == (rf.FIGHT, *RALLY)


def test_a_hurt_fighter_that_regenerates_goes_back_until_it_has_recovered(data):
    script = _script(data)
    out = script.orders(_units((1, "cloakraid", 3000.0, 3000.0, 0.39)), _no_enemies(), 60.0)
    assert _goals(out)[1] == (rf.MOVE, *RALLY)
    assert _goals(script.orders(_units((1, "cloakraid", 2000.0, 2000.0, 0.8)), _no_enemies(), 63.0))[1] == (rf.MOVE, *RALLY)
    assert _goals(script.orders(_units((1, "cloakraid", 600.0, 600.0, 0.9)), _no_enemies(), 66.0))[1] == (rf.FIGHT, *ENEMY_START)


def test_a_recovered_fighter_joins_the_next_group_once_there_are_groups(data):
    script = _script(data)
    script.orders(_units((1, "cloakraid", 3000.0, 3000.0, 0.2)), _no_enemies(), 170.0)
    assert _goals(script.orders(_units((1, "cloakraid", 600.0, 600.0, 0.95)), _no_enemies(), 200.0))[1] == (rf.FIGHT, *RALLY)


def test_a_hurt_fighter_that_doesnt_regenerate_stays_in(data):
    other = ps.ScriptData(data.table, se.catalog())
    other.regenerates[:] = False
    assert _goals(_script(other).orders(_units((1, "cloakraid", 3000.0, 3000.0, 0.1)), _no_enemies(), 60.0))[1][0] == rf.FIGHT


def test_a_little_hurt_isnt_enough_to_retreat(data):
    assert _goals(_script(data).orders(_units((1, "cloakraid", 3000.0, 3000.0, 0.41)), _no_enemies(), 60.0))[1][0] == rf.FIGHT


def test_having_reached_the_start_fighters_go_for_the_nearest_building_and_dont_come_back(data):
    script = _script(data)
    enemies = _units((50, "staticmex", 3000.0, 3900.0), (51, "energysolar", 3900.0, 3000.0), (52, "cloakraid", 3800.0, 3800.0))
    away = (1, "cloakraid", 2000.0, 2000.0)
    assert _goals(script.orders(_units(away), enemies, 60.0))[1] == (rf.FIGHT, *ENEMY_START)
    at = _goals(script.orders(_units((1, "cloakraid", 3990.0, 3600.0)), enemies, 63.0))
    assert at[1] == (rf.FIGHT, 3900.0, 3000.0)
    assert _goals(script.orders(_units((1, "cloakraid", 2900.0, 3800.0)), enemies, 66.0))[1] == (rf.FIGHT, 3000.0, 3900.0)


def test_with_no_buildings_left_they_go_for_whatever_remains(data):
    script = _script(data)
    script.reached_start = True
    enemies = _units((52, "cloakraid", 3800.0, 3700.0))
    assert _goals(script.orders(_units((1, "cloakraid", 3990.0, 3990.0)), enemies, 63.0))[1] == (rf.FIGHT, 3800.0, 3700.0)


def test_a_uid_past_the_end_of_the_table_is_new(data):
    script = _script(data)
    assert _goals(script.orders(_units((1000, "cloakraid", 600.0, 600.0)), _no_enemies(), 60.0))[1000][0] == rf.FIGHT
    assert script.state[1000] == ps.SENT and script.state[1500] == ps.NEW


def test_fighters_really_do_retreat_group_and_finish_games_in_the_sim():
    env = se.SimEnv(32, 13, ["scripts"], threads=4, scripts=("fight", "fight"))
    env.begin()
    kinds = set()
    for _ in range(int(240 * 2)):
        env.step(0.0)
        for g in range(32):
            for side in (0, 1):
                kinds |= {int(s) for s in env.scripts[g][side].state[env.scripts[g][side].state > 0]}
        env.finished_seats()
    assert {ps.WAITING, ps.SENT, ps.RETREATING} <= kinds
