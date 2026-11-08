"""The army and economy scripts (C++: sim/scripts, tools/policy_scripts_cpp.py): the tables they're given, what they do in short sim
games, the same games whatever the thread count, runners that don't share anything; and that frame games play their own scripts."""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
try:
    import policy_scripts_cpp as psc
    psc.lib()
except OSError:
    pytest.skip("needs the built libzkscripts", allow_module_level=True)
import math  # noqa: E402

import policy_mix as pm  # noqa: E402
import policy_script as ps  # noqa: E402
import policy_sim_env as se  # noqa: E402
import zksim  # noqa: E402
from zkbot import catalog as bc  # noqa: E402


def _play(env: se.SimEnv, decisions: int) -> list:
    """Each decision's frames and script orders, as hashes, and the outcomes."""
    out = []
    for _ in range(decisions):
        env.step(0.0)
        record = tuple((s, env.script_record[s].tobytes()) for s in sorted(env.script_record))
        out.append((env.batch.frame_hash(), hash(record)))
        env.finished_seats()
    return out


def test_math_hypot_is_cpythons():
    rng = np.random.default_rng(3)
    for a, b in rng.normal(0.0, 1000.0, (2000, 2)):
        assert psc.lib().zks_py_hypot(a, b) == math.hypot(a, b)


@pytest.mark.parametrize("threads", (4, 7))
def test_results_do_not_depend_on_the_thread_count(threads):
    one = se.SimEnv(6, 11, ["scripts"], threads=2, script_threads=1)
    many = se.SimEnv(6, 11, ["scripts"], threads=2, script_threads=threads)
    one.begin()
    many.begin()
    assert _play(one, 400) == _play(many, 400)


def test_two_runners_at_once_play_as_each_alone():
    """Two environments with the C++ scripts, stepped at the same time on two Python threads, play as each does by itself."""
    alone = [_play(_begun(seed), 300) for seed in (21, 22)]
    envs = [_begun(seed) for seed in (21, 22)]
    together: list = [None, None]

    def play(k):
        together[k] = _play(envs[k], 300)
    workers = [threading.Thread(target=play, args=(k,)) for k in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert together == alone
    assert alone[0] != alone[1]


def _begun(seed: int) -> se.SimEnv:
    env = se.SimEnv(4, seed, ["scripts"], threads=2, script_threads=3)
    env.begin()
    return env


def test_the_environment_records_productions_and_outcomes():
    env = se.SimEnv(3, 7, ["scripts"], threads=2)
    env.time_limit = 400
    env.begin()
    productions = 0
    for _ in range(400):
        env.step(0.0)
        assert set(env.script_record) == set(range(env.seat_count))
        for seat in range(env.seat_count):
            for uid, name in env.script_production(seat):
                productions += 1
                assert name in se.catalog().index
        env.finished_seats()
    assert productions > 0
    assert len(env.outcomes) >= 2 * 3 and len(env.outcomes) % 2 == 0
    assert all(o["made"].get("raid", 0) > 0 for o in env.outcomes)


def test_policy_seats_get_the_economy_but_not_the_army_script():
    env = se.SimEnv(2, 5, ["fight"], threads=2)
    env.begin()
    economy = np.zeros(env.seat_count, np.int64)
    for _ in range(600):
        env.step(0.0)
        assert set(env.script_record) == {1, 3}         # side 1 is the script's in "fight" mode
        counts = env.cpp.seat_counts(env.count)
        assert counts[[0, 2], 1].sum() == 0
        economy += counts[:, 3]
        env.finished_seats()
    assert (economy > 0).all()          # the economy script plays every side, the policy's too


# The tables the C++ is given.

def test_each_role_on_a_roster_is_its_cheapest_modelled_unit():
    data = ps.ScriptData(se.tables.table(), se.catalog())
    for factory in ("factorycloak", "factoryveh"):
        menu = pm.menu(data, data.row_of_name[factory])
        for name in bc.roster(factory):
            row = data.row_of_name.get(name)
            if row is None or not data.modelled[row]:
                continue
            assert data.role_of[row] in menu
            assert data.table.cost[menu[data.role_of[row]]] <= data.table.cost[row]


def test_a_silo_is_told_to_make_the_missile_that_hurts_most():
    data = ps.ScriptData(se.tables.table(), se.catalog())
    silo = data.table.names.index("staticmissilesilo")
    assert data.table.names[data.best_missile[silo]] == "tacnuke"


# Short sim games.

def test_a_factory_is_told_what_to_build_only_when_that_changes_and_it_is_its_own():
    env = se.SimEnv(8, 3, ["scripts"], threads=4)
    env.begin()
    last: dict[tuple, str] = {}
    told = 0
    for _ in range(400):            # games aren't started afresh: no finished_seats
        env.step(0.0)
        for seat in range(16):
            g, side = divmod(seat, 2)
            for uid, name in env.script_production(seat):
                assert uid == env.seats[g][side].factory_uid
                assert last.get((seat, uid)) != name
                last[(seat, uid)] = name
                told += 1
                assert env.script_data.role_of[env.script_data.row_of_name[name]] in pm.ROLE_ORDER + (pm.SCOUT, pm.BUILDER)
    assert told > 0


def test_in_sim_games_the_economy_grows_and_its_orders_are_not_the_scripts():
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


def test_frame_games_play_their_own_scripts_not_the_cpp_ones():
    import frame_world as fw
    env = fw.FrameGames(2, 3, threads=2)
    assert env.cpp is None
    env.begin()
    fights = 0
    for _ in range(400):
        env.step(0.0)
        assert all(isinstance(s, fw.FrameFight) for pair in env.scripts for s in pair)
        fights += sum(int((o["kind"] == zksim.ORDER_FIGHT).sum()) for o in env.script_record.values())
        env.finished_seats()
    assert fights > 0
