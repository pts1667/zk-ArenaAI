"""A side's orders at a decision as commands (zkbot.policy.selections), from the scripts' games: what the order autoencoder
(order_vae.py; docs/learned-policy/05-order-autoencoder.md) learns from.

    order_data.py record --out DIR --decisions N [--games 32] [--workers 8] [--threads 16] [--seed S] [--maps /mnt/d/zk-learned/maps]
        [--kinds 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1] [--variants]

Every team is a policy's seat that the scripts only advise, with its own economy, and what's carried out is their advice said
as commands and read back (SimEnv.through, selections.encode and decode).  So the games are played in the commands' own
terms, on both sides, and each look's commands are kept.  A look where the scripts say nothing isn't kept: the autoencoder's
training adds its own empty decisions.

A worker writes part-wNN.npz: `commands` (selections.COMMAND_DTYPE, every decision's one after another), `counts` (how many
each decision has), `map_w` and `map_h` (the decision's map, in elmos: a command's places are fractions of it).  summary.json
has what was played and what of the scripts' orders couldn't be said.  No torch; it loads the map pool.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bots"), str(ROOT / "sim" / "py")]
import policy_sim_env as se  # noqa: E402
import zkpolicy  # noqa: E402
from zkbot.policy import actions, selections as sel  # noqa: E402

_POOL: dict = {}     # the maps, loaded once before the workers fork, so that they share them


def _maps(args: argparse.Namespace) -> tuple[list | None, np.ndarray | None]:
    if not args.maps:
        return None, None
    if "maps" not in _POOL:
        from frame_world import held_out_maps, pool_weights
        maps = se.load_pool(args.maps)
        weights = pool_weights(maps)
        weights[held_out_maps(maps)] = 0.0
        se.share_maps(maps)
        _POOL.update(maps=maps, weights=weights / weights.sum())
    return _POOL["maps"], _POOL["weights"]


class Through:
    """SimEnv.through: the scripts' advice as commands and back, each look's commands kept."""

    def __init__(self, env: se.SimEnv, slots: int = sel.SLOTS):
        self.env, self.slots = env, slots
        self.commands: list[np.ndarray] = []
        self.counts: list[int] = []
        self.sizes: list[tuple[float, float]] = []
        self.missed: dict = {}
        self.orders = self.cut = 0

    def __call__(self, seat: int, advice: list) -> list:
        env = self.env
        bundle, frame = env.bundle_of(seat), env.frame(seat)
        t = zkpolicy.featurize(frame, bundle, enabled_kinds=se.KINDS, economy=True)
        x, z = actions.world_positions(t, bundle)
        commands, _ = sel.encode(advice, t, bundle, x, z, slots=None, missed=self.missed, last=sel.last_orders(t, frame.units))
        self.orders += len(advice)
        self.cut += max(0, len(commands) - self.slots)
        commands = commands[:self.slots]
        self.commands.append(commands)
        self.counts.append(len(commands))
        self.sizes.append((float(bundle.map_w), float(bundle.map_h)))
        return sel.decode(commands, t, bundle, x, z)

    def save(self, path: pathlib.Path) -> None:
        sizes = np.array(self.sizes, np.float32).reshape(-1, 2)
        np.savez_compressed(path, commands=np.concatenate(self.commands) if self.commands else np.zeros(0, sel.COMMAND_DTYPE),
                            counts=np.array(self.counts, np.uint8), map_w=sizes[:, 0], map_h=sizes[:, 1])


def _record_part(args: argparse.Namespace, worker: int, decisions: int) -> dict:
    maps, weights = _maps(args)
    env = se.SimEnv(args.games, args.seed * 1000 + worker, ["mirror"], threads=max(1, args.threads // args.workers), maps=maps, weights=weights,
                    variants=args.variants, kinds=se.parse_kinds(args.kinds))
    env.advise, env.policy_economy_share = True, 1.0
    through = env.through = Through(env)
    env.begin()
    began = time.perf_counter()
    games = 0
    while len(through.counts) < decisions:
        env.step(1.0)
        games += len({env.game_of(seat) for seat, _, _ in env.finished_seats()})
    through.save(pathlib.Path(args.out) / f"part-w{worker:02d}.npz")
    return {"decisions": len(through.counts), "commands": int(sum(through.counts)), "orders": through.orders, "cut": through.cut, "games_ended": games,
            "missed": {f"{why} ({kind})": n for (why, kind), n in through.missed.items()}, "seconds": time.perf_counter() - began}


def record(args: argparse.Namespace) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _maps(args)
    share = -(-args.decisions // args.workers)
    if args.workers == 1:
        parts = [_record_part(args, 0, share)]
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_record_part, [args] * args.workers, range(args.workers), [share] * args.workers))
    missed: dict = {}
    for p in parts:
        for why, n in p["missed"].items():
            missed[why] = missed.get(why, 0) + n
    summary = {k: sum(p[k] for p in parts) for k in ("decisions", "commands", "orders", "cut", "games_ended")}
    summary.update(seconds=max(p["seconds"] for p in parts), missed=missed, slots=sel.SLOTS, kinds=list(sel.KINDS), maps=args.maps or "training",
                   seed=args.seed, workers=args.workers, game_kinds=se.parse_kinds(args.kinds), variants=args.variants)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def load(data: pathlib.Path | str) -> list[dict]:
    """Every part of a recording, in name order: each part's arrays (`commands`, `counts`, `map_w`, `map_h`)."""
    return [dict(np.load(p)) for p in sorted(pathlib.Path(data).glob("part-*.npz"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--decisions", type=int, required=True)
    r.add_argument("--games", type=int, default=32, help="a worker's, at a time")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--maps", default="")
    r.add_argument("--kinds", nargs="*", default=None)
    r.add_argument("--variants", action="store_true")
    args = parser.parse_args()
    record(args)


if __name__ == "__main__":
    main()
