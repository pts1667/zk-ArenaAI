"""What a batch of games costs per decision (S-M5, and S-M6's fog cost).  In WSL, from the repo:

  ~/zkml/bin/python tools/sim_batch_bench.py --games 256 --units 200            # 400 units a game, 16 threads
  ZKSIM_LIB=~/zksim/build-prof/libzksim.so ~/zkml/bin/python tools/sim_batch_bench.py --games 16 --threads 1

The budget is 03-training.md section 2.6: 0.5 ms of wall time per game decision at 400 units with 16 threads, 2x allowed.
With a profiling build it also prints each phase's share, fog included.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy_sim_env as env_module  # noqa: E402
import zksim  # noqa: E402


def counters():
    lib = zksim._lib
    try:
        size = lib.zk_prof_count()
    except AttributeError:
        return None
    lib.zk_prof_name.restype = ct.c_char_p
    names = [lib.zk_prof_name(i).decode() for i in range(size)]
    nanos, calls = (ct.c_longlong * size)(), (ct.c_longlong * size)()

    def take():
        lib.zk_prof_read(nanos, calls, 1)
        return {n: (nanos[i] / 1e9, calls[i]) for i, n in enumerate(names)}
    return take


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--units", type=int, default=200, help="per side")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--decisions", type=int, default=320)
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args()
    take = counters()
    env = env_module.SimEnv(args.games, 1, ["bench"], units=args.units, threads=args.threads)
    t0 = time.perf_counter()
    env.begin()
    print(f"{args.games} games of {args.units * 2} units, {args.threads} threads, started in {time.perf_counter() - t0:.2f} s")
    if take:
        take()
    wall = orders_wall = 0.0
    total_wall = 0.0
    for d in range(1, args.decisions + 1):
        t0 = time.perf_counter()
        parts, offsets = [], [0]
        for g in range(args.games):
            a, b = env.script_orders(g, 0), env.script_orders(g, 1)
            parts += [a, b]
            offsets.append(offsets[-1] + len(a) + len(b))
        all_orders = np.concatenate(parts)
        order_array = np.array(offsets, np.int32)
        t1 = time.perf_counter()
        env.batch.step(all_orders, order_array, ticks=env_module.TICKS_PER_DECISION)
        t2 = time.perf_counter()
        env.decisions += 1
        orders_wall += t1 - t0
        wall += t2 - t1
        total_wall += t2 - t1
        if d % args.window == 0:
            alive = env.batch.stats[:, :, 1].sum(axis=1)
            records = env.batch.counts.mean()
            print(f"decision {d}: {wall / args.window * 1000:.1f} ms per batch step, {wall / args.window / args.games * 1000:.3f} ms per game decision;"
                  f" units alive mean {alive.mean():.0f}, records per side {records:.0f}, orders {orders_wall / args.window * 1000:.1f} ms", flush=True)
            wall = orders_wall = 0.0
    print(f"whole run: {total_wall / args.decisions / args.games * 1000:.3f} ms per game decision (budget 0.5, allowed 1.0)")
    if take:
        phases = take()
        total = sum(v[0] for v in phases.values())
        game_decisions = args.decisions * args.games
        for name, (seconds, calls) in sorted(phases.items(), key=lambda kv: -kv[1][0]):
            print(f"  {name:10s} {seconds / total * 100:5.1f}%  {seconds / game_decisions * 1e6:8.1f} us per game decision  {calls} calls")


if __name__ == "__main__":
    main()
