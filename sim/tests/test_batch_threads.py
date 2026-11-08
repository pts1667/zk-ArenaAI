"""A batch's threads: its results don't depend on how many there are or how games are handed out, batches stepping at once
don't touch each other, and a forked child can go on with its parent's batch.  In WSL:

  ~/zkml/bin/python -m pytest -q sim/tests/test_batch_threads.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
ARMY = ("cloakraid", "cloakassault", "shieldraid", "tankassault", "spiderassault", "jumpraid", "hoverraid", "vehassault", "cloakskirm", "jumpblackhole")
GAMES = 10
DECISIONS = 10


def start(batch: zksim.Batch, seed: int) -> None:
    """Games of different sizes, so that some take far longer than others and the schedule has something to choose."""
    for g in range(batch.games):
        rng = np.random.default_rng(seed + g)
        count = 16 + 24 * (g % 5)
        names = np.array([CAT.index[ARMY[int(t)]] for t in rng.integers(0, len(ARMY), count)])
        side = np.arange(count) % 2
        x = np.where(side == 0, rng.uniform(300, 1500, count), rng.uniform(2500, 3700, count))
        z = rng.uniform(300, 3700, count)
        batch.reset(g, FLAT, seed + g, names, side, x, z, commander=(0, 1), base=(300.0, 300.0, 3700.0, 3700.0))
        batch.add_features(g, np.array([[2000.0, 500.0 + 300 * k, 50.0, 200.0, 16.0, 1] for k in range(8)], np.float32))


def fight_orders(games: int) -> tuple[np.ndarray, np.ndarray]:
    orders = np.zeros(games * 6, zksim.ORDER_DTYPE)
    for g in range(games):
        for k in range(6):
            orders[g * 6 + k] = (k, zksim.ORDER_FIGHT, 2000.0, 2000.0 + 100 * k, -1)
    return orders, np.arange(games + 1, dtype=np.int32) * 6


def play(batch: zksim.Batch, decisions: int = DECISIONS, split: bool = False, between=None) -> list[int]:
    hashes = []
    orders, offsets = fight_orders(batch.games)
    for d in range(decisions):
        if between:
            between(d)
        args = (orders, offsets) if d == 0 else ()
        if split:
            batch.step_begin(*args)
            batch.step_wait()
        else:
            batch.step(*args)
        hashes.append(batch.frame_hash())
    return hashes


def run(threads: int, seed: int = 300, **kwargs) -> list[int]:
    batch = zksim.Batch(CAT, GAMES, threads=threads)
    start(batch, seed)
    return play(batch, **kwargs)


def test_results_are_the_same_for_any_thread_count():
    alone = run(1)
    assert run(2) == alone
    assert run(5) == alone
    assert run(16) == alone


def test_results_are_the_same_for_a_split_step_and_a_changing_thread_count():
    alone = run(1)
    assert run(4, split=True) == alone
    batch = zksim.Batch(CAT, GAMES, threads=1)
    start(batch, 300)
    assert play(batch, between=lambda d: batch.set_threads(1 + (d * 3) % 7)) == alone


def test_results_are_the_same_whichever_order_the_games_are_listed_in():
    """Listed backwards, a game's frames land in the other slot, and the threads take the games in another order."""
    forwards = zksim.Batch(CAT, GAMES, threads=3)
    backwards = zksim.Batch(CAT, GAMES, threads=3)
    start(forwards, 300)
    start(backwards, 300)
    order = np.arange(GAMES, dtype=np.int32)[::-1].copy()
    orders, offsets = fight_orders(GAMES)
    # The orders are the same for every game, so they stay with the right game when the list is turned round.
    for d in range(DECISIONS):
        forwards.step(orders, offsets) if d == 0 else forwards.step()
        backwards.step(orders, offsets, games=order) if d == 0 else backwards.step(games=order)
        for g in range(GAMES):
            p = GAMES - 1 - g
            for side in (0, 1):
                n = int(forwards.counts[g, side])
                assert n == int(backwards.counts[p, side])
                assert forwards.records[g, side, :n].tobytes() == backwards.records[p, side, :n].tobytes(), (d, g, side)
            assert np.array_equal(forwards.stats[g], backwards.stats[p])


def test_two_batches_stepping_at_once_get_what_each_gets_alone():
    alone_a, alone_b = run(3, seed=300), run(3, seed=900)
    got: dict[str, list[int]] = {}
    barrier = threading.Barrier(2)

    def go(name: str, seed: int) -> None:
        batch = zksim.Batch(CAT, GAMES, threads=3)
        start(batch, seed)
        barrier.wait()
        got[name] = play(batch)
    threads = [threading.Thread(target=go, args=("a", 300)), threading.Thread(target=go, args=("b", 900))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got["a"] == alone_a and got["b"] == alone_b


def test_a_forked_child_goes_on_with_its_parents_batch():
    batch = zksim.Batch(CAT, GAMES, threads=4)
    start(batch, 300)
    first = play(batch, decisions=3)
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.write(write, repr(play(batch, decisions=3)).encode())
        finally:
            os._exit(0)
    os.close(write)
    with os.fdopen(read) as f:
        theirs = f.read()
    os.waitpid(child, 0)
    ours = play(batch, decisions=3)
    assert theirs == repr(ours)
    alone = zksim.Batch(CAT, GAMES, threads=1)
    start(alone, 300)
    assert first + ours == play(alone, decisions=3) + play(alone, decisions=3)
