"""The sim as an environment: which maps it plays, and when a match ends."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_sim_env as se
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)


def _env_with_values(a: float, b: float, decide_every: int = 1) -> se.SimEnv:
    """Both sides stay alive at these metal values, whatever the sim does."""
    env = se.SimEnv(1, 3, ["mirror"], units=2, threads=1, decide_every=decide_every)
    env.begin()
    real = env.batch.step

    def step(*args, **kwargs):
        real(*args, **kwargs)
        env.batch.stats[0] = ((a, 5.0), (b, 5.0))
    env.batch.step = step
    return env


def _steps_to_end(env: se.SimEnv, most: int) -> tuple[int, list]:
    for n in range(1, most + 1):
        env.step(0.0)
        if env.done:
            return n, env.done
    return most + 1, []


def test_training_plays_only_its_maps():
    assert len(se.pool()) == len(se.TRAINING_MAPS)
    assert max(b.regions for _, b, _ in se.pool()) <= 35


@pytest.mark.parametrize("a, b, seat_a, seat_b", ((2600.0, 1000.0, 1.0, 0.0), (1000.0, 2600.0, 0.0, 1.0)))
def test_a_side_far_ahead_wins_at_two_minutes(a, b, seat_a, seat_b):
    env = _env_with_values(a, b)
    n, done = _steps_to_end(env, se.EARLY_END_DECISIONS)
    assert n == se.EARLY_END_DECISIONS
    assert sorted(done) == [(0, "mirror", seat_a), (1, "mirror", seat_b)]
    assert env.result[0] == (1.0 if a > b else -1.0)


def test_a_smaller_lead_plays_on():
    env = _env_with_values(2400.0, 1000.0)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS + 5)[1] == []


def test_two_minutes_stay_two_minutes_at_a_slower_decision_rate():
    env = _env_with_values(2600.0, 1000.0, decide_every=2)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS)[0] == se.EARLY_END_DECISIONS // 2


def test_nothing_on_either_side_is_no_lead():
    env = _env_with_values(0.0, 0.0)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS + 5)[1] == []
    assert np.all(env.result == 0.0)
