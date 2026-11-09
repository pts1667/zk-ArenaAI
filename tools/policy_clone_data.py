"""Games to learn from before RL: the army script against itself in the sim, every decision of both sides kept with what the script told it.

    ~/zkml/bin/python tools/policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 64 --decisions 2000000 --workers 4 [--maps /mnt/d/zk-learned/maps]
    ~/zkml/bin/python tools/policy_clone_data.py check --data /mnt/d/zk-learned/clone/NAME

Games run as a batch in the sim, which steps them on its threads, and the scripts on theirs; `--workers` runs that many batches in
processes of their own, each with its own seed.  `--variants` gives each side's army script its own knobs at each game's start
(policy_variants.py), and each game's meta says which: without it every side plays the standard script, as before.

A file a game (`game-*.zkr`, data version 5): both sides' frames in order, the orders each side's script gave at each, as the policy
would give them (a group order as its anchor's order and a groups.GROUP_MARK record after it), and how the game ended.  libzkrecord writes it straight from the batch's arrays, narrowed and compressed (sim/record/zkrecord.h);
`zkrecord.read` gives it back.  Data versions 1 to 3 were a file a side (`game-*-SIDE.npz`, rawframe.save_frames): `read_game` reads
both.  Frames and orders are kept raw, not as features or choices, so the data outlives changes to the featurizer: `labels` turns them
into the policy's choices when they're read.  The commander's opening orders aren't kept: they aren't the policy's to give, so the
commander's label is to keep doing what it does.
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
import zkrecord  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import actions, bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

DATA_VERSION = 5      # 1: the army script with the opening's economy only.  2: with the economy script (policy_eco.py), 2026-11-08.
                      # 3: the army script aims missiles and the Trinity (labels now), the economy spends what it makes, and in the sim
                      # help for a factory still being built builds it, 2026-11-08.  4: the same games, a .zkr file a game, 2026-11-08.
                      # 5: the army script's group orders, recorded as the policy gives them (groups.GROUP_MARK), air, and a second
                      # factory, 2026-11-08


class Recorder:
    """Both sides of every game of an environment, until each game ends; then a file for it.

    After `close`, games that start aren't kept, and those running are played out, so that the data isn't biased towards short
    games, as it would be if those running were dropped; `closed` says when they're all written."""

    def __init__(self, env: se.SimEnv, out: pathlib.Path, prefix: str = "game", level: int = 3, threads: int = 1):
        self.env, self.out, self.prefix = env, out, prefix
        out.mkdir(parents=True, exist_ok=True)
        self.slugs = {id(b): f.name for _, b, f in env.maps}
        self.writer = zkrecord.Writer(env.count, threads, level)
        self.games = self.decisions = 0
        self.ended: Counter = Counter()
        self.kept = set(range(env.count))     # games whose present run will be written
        self._empty = np.zeros(0, zksim.ORDER_DTYPE)
        for g in self.kept:
            self._begin(g)

    def close(self) -> None:
        self.closing = True

    closing = False

    @property
    def closed(self) -> bool:
        return self.closing and not self.kept

    def _begin(self, g: int) -> None:
        bundle = self.env.bundle[g]
        self.writer.begin(g, bundle.map_w, bundle.map_h, zkrecord.height_range(bundle.height), self.env.batch.sites[g])

    def _orders(self) -> tuple[np.ndarray, np.ndarray]:
        """This step's script orders over every seat, and where each seat's start."""
        env = self.env
        if env.army_orders is not None:
            return env.army_orders
        parts = [env.script_record.get(seat, self._empty) for seat in range(env.seat_count)]
        return rf.join(parts, zksim.ORDER_DTYPE), np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int32)

    def step(self) -> None:
        env = self.env
        listed = np.array(sorted(self.kept), np.int32)
        self.writer.frames(env.batch, listed)
        env.step(0.0)
        self.writer.orders(listed, *self._orders())
        for g, facts in zip(env._replace, env.outcomes[::2]):     # both sides of each game that ended, side 0 first
            if g in self.kept:
                self.ended[facts["how"]] += 1
        env.outcomes.clear()
        ended = [g for g in env._replace if g in self.kept]
        for g in ended:
            self._write(g)
            if self.closing:
                self.kept.discard(g)
        env.finished_seats()        # which starts them afresh
        for g in ended:
            if g in self.kept:
                self._begin(g)
        if ended and self.closed:
            self.writer.wait()      # so that every file is on disk once `closed` says so

    def _write(self, g: int) -> None:
        env = self.env
        frames = self.writer.frames_held(g)
        result = float(env.result[g])
        meta = {"data_version": DATA_VERSION, "schema_version": rf.SCHEMA_VERSION, "map": self.slugs.get(id(env.bundle[g]), "?"),
                "script": list(env.script_kinds[g]), "result": [result, -result]}
        if env.styles[g] is not None:
            meta["variants"] = [v.as_dict() for v in env.styles[g]]     # each side's army script knobs (policy_variants.py)
        self.writer.finish(g, self.out / f"{self.prefix}-{self.games:06d}{zkrecord.SUFFIX}", meta)
        self.decisions += 2 * frames
        self.games += 1


Side = tuple[list[rf.Frame], list[np.ndarray], dict]


def read_game(path: pathlib.Path) -> list[Side]:
    """Every side a file holds, as (frames, the orders given at each, the rest): both of a .zkr game, the one of an .npz."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        frames = rf.load_frames(path)
        with np.load(path) as d:
            orders = np.split(d["orders"], np.cumsum(d["order_counts"])[:-1]) if len(d["order_counts"]) else []
            info = {"result": float(d["result"]), "map": str(d["map"]), "script": str(d["script"])}
        return [(frames, orders, info)]
    meta, sides = zkrecord.read(path)
    return [(frames, orders, {"result": float(meta["result"][s]), "map": meta["map"], "script": meta["script"][s],
                              "variant": meta["variants"][s] if "variants" in meta else None})
            for s, (frames, orders) in enumerate(sides)]


def load(path: pathlib.Path, side: int = 0) -> Side:
    """One side of one game: frames, the orders given at each, and the rest.  An .npz holds one side, its side 0 here."""
    return read_game(path)[side]


def sides(data: pathlib.Path) -> list[tuple[str, int, int]]:
    """Every side of every finished game in a dataset: (path, side in `read_game`, frames)."""
    out = []
    for path in sorted(data.glob("game-*.zkr")):
        try:
            n = zkrecord.frame_count(path)
        except (OSError, ValueError):
            continue
        out += [(str(path), 0, n), (str(path), 1, n)]
    for path in sorted(data.glob("game-*.npz")):
        try:
            with np.load(path) as d:
                n = len(d["unit_counts"])
        except (OSError, ValueError, KeyError, EOFError):        # still being written
            continue
        out.append((str(path), 0, n))
    return [s for s in out if s[2]]


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
        se.share_maps(maps)
        _POOL.update(maps=maps, weights=weights / weights.sum())
    return _POOL["maps"], _POOL["weights"]


def _record_part(args: argparse.Namespace, worker: int, decisions: int) -> dict:
    maps, weights = _maps(args)
    threads = max(1, args.threads // args.workers)
    env = se.SimEnv(args.games, args.seed * 1000 + worker, ["scripts"], threads=threads, maps=maps, weights=weights, variants=args.variants)
    env.begin()
    rec = Recorder(env, pathlib.Path(args.out), f"game-w{worker:02d}", level=args.level, threads=threads)
    began = time.perf_counter()
    while rec.decisions < decisions:
        rec.step()
    rec.close()
    while not rec.closed:
        rec.step()
    return {"games": rec.games, "decisions": rec.decisions, "ended": dict(rec.ended), "seconds": time.perf_counter() - began,
            "clamped": rec.writer.clamps()}


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
    clamped: Counter = Counter()
    for part in parts:
        ended.update(part["ended"])
        clamped.update(part["clamped"])
    seconds = time.perf_counter() - began
    decisions = sum(p["decisions"] for p in parts)
    size = sum(p.stat().st_size for p in out.glob(f"game-*{zkrecord.SUFFIX}"))
    summary = {"games": sum(p["games"] for p in parts), "decisions": decisions, "seconds": seconds, "loading_maps_seconds": loaded,
               "decisions_a_second_playing": decisions / max(max(p["seconds"] for p in parts), 1e-9), "ended": dict(ended),
               "bytes_a_decision": size / max(decisions, 1), "clamped": dict(clamped), "level": args.level,
               "maps": args.maps or "training", "seed": args.seed, "workers": args.workers, "data_version": DATA_VERSION,
               "variants": args.variants}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def check(args: argparse.Namespace) -> None:
    """What the labels look like: kinds chosen, and how much of what the script said the policy can't."""
    data = pathlib.Path(args.data)
    maps = {f.name: b for _, b, f in (se.load_pool(args.maps) if args.maps else se.pool())}
    table = tables.table()
    kinds, missed, orders, decisions, results, sizes, grouped = Counter(), 0, 0, 0, Counter(), [], 0
    picked = sides(data)[:args.files]
    games: dict[str, list[Side]] = {}
    for path, side, n in picked:
        if path not in games:
            games.clear()
            games[path] = read_game(pathlib.Path(path))
            sizes.append(pathlib.Path(path).stat().st_size / max(n * len(games[path]), 1))
        frames, given, info = games[path][side]
        results[info["result"]] += 1
        bundle = maps[info["map"]]
        for frame, o in zip(frames[::args.every], given[::args.every]):
            t, d, m = labels(frame, o, bundle, table, enabled_kinds=se.KINDS)
            kinds.update(int(k) for k in d.kind)
            grouped += int((d.group >= 0).sum())
            missed += m
            orders += len(o)
            decisions += 1
    total = max(sum(kinds.values()), 1)
    print(json.dumps({"sides": len(picked), "decisions_read": decisions, "orders": orders, "missed": missed, "missed_share": missed / max(orders, 1),
                      "group_orders": grouped,
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
    r.add_argument("--level", type=int, default=3, help="zstd's: 3 is about 10%% larger than 9 and 6 times as fast")
    r.add_argument("--variants", action="store_true", help="each side's army script draws its own knobs (policy_variants.py), recorded in each game")
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--maps")
    c.add_argument("--files", type=int, default=20, help="sides read")
    c.add_argument("--every", type=int, default=5, help="read every this many decisions")
    args = ap.parse_args()
    {"record": record, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
