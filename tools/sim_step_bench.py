"""What zk_batch_step alone costs on scripted games, by game phase and thread count.  In WSL, from the repo:

  PYTHONHASHSEED=0 PYTHONPATH=$HOME/zksim/pt:bots CUDA_VISIBLE_DEVICES= ~/zkml/bin/python tools/sim_step_bench.py --games 128 --threads 1,4,8

Both sides play the scripts (opening, economy and army) from commanders, as SimEnv does for the cloning dataset, with no early end and
no time limit, so a game ends only when one side is gone and is then started afresh.  At each phase (minutes of game time) it times
only the batch step, for --measure decisions per thread count, the scripts still giving orders in between.  The run is deterministic,
so two libraries (ZKSIM_LIB) that agree bit for bit time the same states; each phase prints a hash of what the last step wrote.

The check that a change to the sim plays the same games: the same command with --check under each library, which adds to each
phase's line a hash of every frame every step has written since the start (units, features, stats and resources of every team).
--kinds and --maps play the kinds of game a run plays, on a directory of pool maps (a handful linked into one will do).
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy_sim_env as env_module  # noqa: E402
import zksim  # noqa: E402


def set_threads(batch: zksim.Batch, threads: int) -> None:
    lib = zksim._lib
    if hasattr(lib, "zk_batch_set_threads"):
        lib.zk_batch_set_threads.argtypes = [ct.c_void_p, ct.c_int]
        lib.zk_batch_set_threads(batch.handle, threads)
    else:
        # The baseline's Batch: a catalog pointer, then the thread count.
        ct.c_int.from_address(batch.handle + 8).value = threads


def profile_counters():
    lib = zksim._lib
    try:
        size = lib.zk_prof_count()
    except AttributeError:
        return None
    lib.zk_prof_name.restype = ct.c_char_p
    names = [lib.zk_prof_name(i).decode() for i in range(size)]
    nanos, calls = (ct.c_longlong * size)(), (ct.c_longlong * size)()
    batch_read = getattr(lib, "zk_batch_prof_read", None)

    def take(batch):
        if batch_read is not None:
            batch_read.argtypes = [ct.c_void_p, ct.POINTER(ct.c_longlong), ct.POINTER(ct.c_longlong), ct.c_int]
            batch_read(batch.handle, nanos, calls, 1)
        else:
            lib.zk_prof_read(nanos, calls, 1)
        return {n: (nanos[i] / 1e9, int(calls[i])) for i, n in enumerate(names)}
    return take


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--phases", default="1,8,16", help="minutes of game time at which to measure")
    ap.add_argument("--measure", type=int, default=20, help="decisions timed per thread count, twice over")
    ap.add_argument("--warm-threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kinds", nargs="*", default=None, help="kinds of game by weight, as policy_sim_env.parse_kinds: 1v1s if not given")
    ap.add_argument("--maps", default="", help="a directory of pool maps (policy_sim_env.load_pool): the training maps if not given")
    ap.add_argument("--check", action="store_true", help="hash every step's frames, to compare two libraries' games")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(",")]
    phases = [float(p) for p in args.phases.split(",")]
    take = profile_counters()

    kinds = env_module.parse_kinds(args.kinds) if args.kinds else None
    maps = env_module.load_pool(args.maps) if args.maps else None
    env = env_module.SimEnv(args.games, args.seed, ["scripts"], threads=args.warm_threads, kinds=kinds, maps=maps)
    env.early_end = env.time_limit = 1 << 30
    batch = env.batch
    plain_step = batch.step
    clock = {"on": False, "wall": 0.0}

    def timed_step(*a, **k):
        t0 = time.perf_counter()
        plain_step(*a, **k)
        if clock["on"]:
            clock["wall"] += time.perf_counter() - t0
    batch.step = timed_step
    env.begin()
    seconds_per_decision = env.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
    decision = 0
    rows = []
    started = time.perf_counter()

    every_step = hashlib.blake2b(digest_size=8) if args.check else None

    def advance():
        nonlocal decision
        env.step(0.0)
        if every_step is not None:
            for position in range(batch.games):
                for team in range(batch.teams):
                    every_step.update(batch.records[position, team, :int(batch.counts[position, team])].tobytes())
                    every_step.update(batch.features[position, team, :int(batch.feature_counts[position, team])].tobytes())
            every_step.update(batch.stats.tobytes())
            every_step.update(batch.globs.tobytes())
        env.finished_seats()
        decision += 1

    for minutes in phases:
        set_threads(batch, args.warm_threads)
        while decision * seconds_per_decision < minutes * 60.0:
            advance()
        alive = float(batch.stats[:, :, 1].sum(axis=1).mean())
        worlds = [zksim._lib.zk_batch_world(batch.handle, g) for g in range(args.games)]
        handles = float(np.mean([zksim._lib.zk_world_count(w) for w in worlds]))
        if take:
            take(batch)
        per = {t: 0.0 for t in threads}
        for t in threads + threads[::-1]:
            set_threads(batch, t)
            clock["on"], clock["wall"] = True, 0.0
            for _ in range(args.measure):
                advance()
            clock["on"] = False
            per[t] += clock["wall"]
        steps = 2 * args.measure * args.games
        profile = take(batch) if take else None
        line = {"games": args.games, "minutes": minutes, "alive": alive, "handles": handles, "hash": batch.frame_hash(),
                "steps_per_s": {t: steps / per[t] for t in threads}, "us_per_step": {t: per[t] / steps * 1e6 for t in threads}}
        if profile:
            line["profile"] = profile
        rows.append(line)
        if every_step is not None:
            line["every_step"] = every_step.hexdigest()
        print(f"{args.games} games at {minutes:g} min: {alive:.0f} units alive a game ({handles:.0f} handles), hash {line['hash']:016x}"
              + (f", every step's {line['every_step']}" if every_step is not None else "") + "; "
              + ", ".join(f"{t} thr {steps / per[t]:,.0f} game-steps/s" for t in threads)
              + f"  [{time.perf_counter() - started:.0f} s]", flush=True)
        if profile:
            total = sum(v[0] for k, v in profile.items() if "." not in k)
            for name, (secs, calls) in profile.items():
                if calls:
                    print(f"    {name:28s} {secs / steps * 1e6:9.1f} us/game-step  {secs / total * 100 if total else 0:5.1f}%  {calls} calls")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
