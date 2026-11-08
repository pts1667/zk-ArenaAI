"""The scripts in C++ (sim/scripts, tools/policy_scripts_cpp.py): the same orders as the Python scripts, whatever the thread
count, and runners that don't share anything."""
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

import policy_clone_data as cd  # noqa: E402
import policy_scripts_parity as parity  # noqa: E402
import policy_sim_env as se  # noqa: E402
import zksim  # noqa: E402


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


def test_the_same_orders_as_the_python_scripts_over_a_short_game():
    """Every order of the army script, the aimer, the factory and the economy, decision by decision, from the start to 6 minutes."""
    report = parity.run(games=2, seed=4, decisions=720, threads=2, examples=3)
    assert report["tape_misses"] == 0
    for part in ("army", "eco"):
        r = report[part]
        assert r["orders_py"] > 200
        assert r["orders_identical"] == r["orders_py"] == r["orders_cpp"], report["examples"]
    kinds = report["army"]["orders_compared_by_kind"]
    assert any(k.startswith("fight") for k in kinds) and any(k.startswith("produce") for k in kinds)
    kinds = report["eco"]["orders_compared_by_kind"]
    assert any(k.startswith("build") for k in kinds) and any(k.startswith("assist") for k in kinds)
    assert report["eco"]["calls_drew"] > 0       # draws taken from Python's generator for the C++ economy


@pytest.mark.parametrize("threads", (4, 7))
def test_results_do_not_depend_on_the_thread_count(threads):
    one = se.SimEnv(6, 11, ["scripts"], threads=2, script_impl="cpp", script_threads=1)
    many = se.SimEnv(6, 11, ["scripts"], threads=2, script_impl="cpp", script_threads=threads)
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
    env = se.SimEnv(4, seed, ["scripts"], threads=2, script_impl="cpp", script_threads=3)
    env.begin()
    return env


def test_the_environment_records_productions_and_outcomes_as_with_python(tmp_path):
    env = se.SimEnv(3, 7, ["scripts"], threads=2, script_impl="cpp")
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
    # The recorder works with it.
    env = se.SimEnv(2, 7, ["scripts"], threads=2, script_impl="cpp")
    env.time_limit = 30
    env.begin()
    rec = cd.Recorder(env, tmp_path)
    while rec.decisions < 1:
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    files = sorted(tmp_path.glob("game-*.npz"))
    assert len(files) == 2 * rec.games == 8
    frames, orders, info = cd.load(files[0])
    assert len(frames) == len(orders) == 30 and info["script"] == "army"


def test_policy_seats_get_the_economy_but_not_the_army_script():
    env = se.SimEnv(2, 5, ["fight"], threads=2, script_impl="cpp")
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
