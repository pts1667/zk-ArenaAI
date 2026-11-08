"""Games to learn from before RL: the army script against itself in the sim, every decision of both sides kept with what the script told it.

    ~/zkml/bin/python tools/policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 64 --decisions 2000000 --workers 4 [--maps /mnt/d/zk-learned/maps]
    ~/zkml/bin/python tools/policy_clone_data.py check --data /mnt/d/zk-learned/clone/NAME

Games run as a batch in the sim, which steps them on its threads; the scripts and the writing are Python, so `--workers` runs that many
batches in processes of their own, each with its own seed.

A file a side of a game: its frames in order (rawframe.save_frames), the orders its script gave at each, as the sim took them, and how
the game ended for it.  Frames and orders are kept raw, not as features or choices, so the data outlives changes to the featurizer:
`labels` turns them into the policy's choices when they're read.  The commander's opening orders aren't kept: they aren't the
policy's to give, so the commander's label is to keep doing what it does.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_sim_env as se  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import actions, bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

DATA_VERSION = 3      # 1: the army script with the opening's economy only.  2: with the economy script (policy_eco.py), 2026-11-08.
                      # 3: the army script aims missiles and the Trinity (labels now), the economy spends what it makes, and in the sim
                      # help for a factory still being built builds it, 2026-11-08


def _copy(frame: rf.Frame) -> rf.Frame:
    return rf.Frame(frame.units.copy(), frame.sites.copy(), frame.glob.copy(), frame.features.copy())


class Recorder:
    """Both sides of every game of an environment, until each game ends; then a file each.

    After `close`, games that start aren't kept, and those running are played out, so that the data isn't biased towards short
    games, as it would be if those running were dropped; `closed` says when they're all written."""

    def __init__(self, env: se.SimEnv, out: pathlib.Path, prefix: str = "game"):
        self.env, self.out, self.prefix = env, out, prefix
        out.mkdir(parents=True, exist_ok=True)
        self.slugs = {id(b): f.name for _, b, f in env.maps}
        self.frames: dict[int, list[rf.Frame]] = {s: [] for s in range(env.seat_count)}
        self.orders: dict[int, list[np.ndarray]] = {s: [] for s in range(env.seat_count)}
        self.games = self.decisions = 0
        self.ended: Counter = Counter()
        self.kept = set(range(env.seat_count // 2))     # games whose present run will be written

    def close(self) -> None:
        self.closing = True

    closing = False

    @property
    def closed(self) -> bool:
        return self.closing and not self.kept

    def step(self) -> None:
        env = self.env
        for seat in range(env.seat_count):
            if seat // 2 in self.kept:
                self.frames[seat].append(_copy(env.frame(seat)))
        env.step(0.0)
        for seat in range(env.seat_count):
            if seat // 2 in self.kept:
                self.orders[seat].append(env.script_record.get(seat, np.zeros(0, zksim.ORDER_DTYPE)).copy())
        for g, facts in zip(env._replace, env.outcomes[::2]):     # both sides of each game that ended, side 0 first
            if g in self.kept:
                self.ended[facts["how"]] += 1
        env.outcomes.clear()
        for g in env._replace:
            if g in self.kept:
                for side in (0, 1):
                    self._write(g, side, float(env.result[g]) * (1.0 if side == 0 else -1.0))
                if self.closing:
                    self.kept.discard(g)
        env.finished_seats()        # which starts them afresh

    def _write(self, g: int, side: int, result: float) -> None:
        seat = g * 2 + side
        frames, orders = self.frames[seat], self.orders[seat]
        counts = np.array([len(o) for o in orders], np.int32)
        rf.save_frames(self.out / f"{self.prefix}-{self.games:06d}-{side}.npz", frames, data_version=DATA_VERSION, orders=rf.join(orders, zksim.ORDER_DTYPE),
                       order_counts=counts, result=np.float32(result), map=self.slugs.get(id(self.env.bundle_of(seat)), "?"),
                       script=self.env.script_kinds[g][side])
        self.decisions += len(frames)
        if side == 1:
            self.games += 1
        self.frames[seat], self.orders[seat] = [], []


def load(path: pathlib.Path) -> tuple[list[rf.Frame], list[np.ndarray], dict]:
    """One side of one game: frames, the orders given at each, and the rest."""
    frames = rf.load_frames(path)
    with np.load(path) as d:
        orders = np.split(d["orders"], np.cumsum(d["order_counts"])[:-1]) if len(d["order_counts"]) else []
        info = {"result": float(d["result"]), "map": str(d["map"]), "script": str(d["script"])}
    return frames, orders, info


def labels(frame: rf.Frame, orders: np.ndarray, bundle: bd.Bundle, table: tables.TypeTable, **featurize) -> tuple[fz.Tensors, actions.Decision, int]:
    """A decision as the policy would read it, and what the script did as the policy's choices: the label.  Also how many orders it couldn't say."""
    t = fz.featurize(frame, bundle, table, **featurize)
    x, z = actions.world_positions(t, bundle)
    decision, missed = actions.encode(se.policy_orders(orders), t, bundle, x, z)
    return t, decision, missed


_POOL: dict = {}     # the maps, loaded once before the workers fork, so that they share them: a pool of 298 is 7.5 GB a copy


def _maps(args: argparse.Namespace) -> tuple[list | None, np.ndarray | None]:
    if not args.maps:
        return None, None
    if "maps" not in _POOL:
        from frame_world import held_out_maps, pool_weights
        maps = se.load_pool(args.maps)
        weights = pool_weights(maps)
        weights[held_out_maps(maps)] = 0.0       # kept for testing what's learned on maps it hasn't seen
        _POOL.update(maps=maps, weights=weights / weights.sum())
    return _POOL["maps"], _POOL["weights"]


def _record_part(args: argparse.Namespace, worker: int, decisions: int) -> dict:
    maps, weights = _maps(args)
    env = se.SimEnv(args.games, args.seed * 1000 + worker, ["scripts"], threads=max(1, args.threads // args.workers), maps=maps, weights=weights, scripts=("army", "army"))
    env.begin()
    rec = Recorder(env, pathlib.Path(args.out), f"game-w{worker:02d}")
    began = time.perf_counter()
    while rec.decisions < decisions:
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    return {"games": rec.games, "decisions": rec.decisions, "ended": dict(rec.ended), "seconds": time.perf_counter() - began}


def record(args: argparse.Namespace) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    _maps(args)
    loaded = time.perf_counter() - began
    share = -(-args.decisions // args.workers)
    if args.workers == 1:
        parts = [_record_part(args, 0, share)]
    else:
        # A worker that dies (out of memory, say) fails the whole thing here, rather than leaving the rest waiting for it.
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_record_part, [args] * args.workers, range(args.workers), [share] * args.workers))
    ended: Counter = Counter()
    for part in parts:
        ended.update(part["ended"])
    seconds = time.perf_counter() - began
    decisions = sum(p["decisions"] for p in parts)
    summary = {"games": sum(p["games"] for p in parts), "decisions": decisions, "seconds": seconds, "loading_maps_seconds": loaded,
               "decisions_a_second_playing": decisions / max(max(p["seconds"] for p in parts), 1e-9), "ended": dict(ended),
               "maps": args.maps or "training", "seed": args.seed, "workers": args.workers, "data_version": DATA_VERSION}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def check(args: argparse.Namespace) -> None:
    """What the labels look like: kinds chosen, and how much of what the script said the policy can't."""
    data = pathlib.Path(args.data)
    maps = {f.name: b for _, b, f in (se.load_pool(args.maps) if args.maps else se.pool())}
    table = tables.table()
    kinds, missed, orders, decisions, results, sizes = Counter(), 0, 0, 0, Counter(), []
    paths = sorted(data.glob("game-*.npz"))[:args.files]
    for path in paths:
        frames, given, info = load(path)
        results[info["result"]] += 1
        sizes.append(path.stat().st_size / max(len(frames), 1))
        bundle = maps[info["map"]]
        for frame, o in zip(frames[::args.every], given[::args.every]):
            t, d, m = labels(frame, o, bundle, table, enabled_kinds=se.KINDS)
            kinds.update(int(k) for k in d.kind)
            missed += m
            orders += len(o)
            decisions += 1
    total = max(sum(kinds.values()), 1)
    print(json.dumps({"files": len(paths), "decisions_read": decisions, "orders": orders, "missed": missed, "missed_share": missed / max(orders, 1),
                      "kinds": {rf_name(k): round(v / total, 4) for k, v in sorted(kinds.items())}, "results": {str(k): v for k, v in results.items()},
                      "bytes_per_decision": float(np.mean(sizes)) if sizes else 0.0}, indent=1))


def rf_name(kind: int) -> str:
    names = {rf.KEEP: "keep", rf.MOVE: "move", rf.FIGHT: "fight", rf.ATTACK_UNIT: "attack", rf.ATTACK_GROUND: "attack_ground", rf.STOP: "stop",
             rf.MANUAL_FIRE: "manual_fire", rf.JUMP: "jump", rf.CLOAK_ON: "cloak_on", rf.CLOAK_OFF: "cloak_off", rf.PRODUCE: "produce"}
    return names.get(kind, str(kind))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--games", type=int, default=64)
    r.add_argument("--decisions", type=int, default=200_000, help="of both sides together; once it's reached, games running are played out and no more are started")
    r.add_argument("--maps", help="a map pool from tools/map_gen.py; without it, the five training maps")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--threads", type=int, default=16, help="sim threads, shared among the workers")
    r.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--maps")
    c.add_argument("--files", type=int, default=20)
    c.add_argument("--every", type=int, default=5, help="read every this many decisions")
    args = ap.parse_args()
    {"record": record, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
