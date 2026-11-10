"""Replays for the dashboard (tools/policy_dashboard.py): the first game of each of a SimEnv's first few slots, a frame a second,
each written as gzipped JSON with a line in its directory's index.jsonl.  Set as `env.capture`, SimEnv calls `Capture.after_step`
after every step, so a test's loop needs nothing more.

    ~/zkml/bin/python tools/policy_replay.py demo --maps /mnt/d/zk-learned/maps --out DIR     # the army scripts against each other
    ~/zkml/bin/python tools/policy_replay.py repair DIR...      # name format 1's units again (FORMAT)

A replay holds every unit of both sides, from each side's own records (so what neither side sees is there too):
  {"format": 2, "stage", "label", "slot", "map": {"slug", "width", "height"}, "step_seconds", "every",
   "sides": [{"name", "variant", "factory", "value"}, ...], "result": {"winner": 0, 1 or null, "how", "seconds"},
   "spots": [[x, z, income], ...], "defs": {"def id": {"name", "cost"}}, "unit_fields": UNIT_FIELDS, "side_fields": SIDE_FIELDS,
   "frames": [{"t": seconds, "units": [the unit fields, a unit after another], "sides": [[side 0's], [side 1's]],
               "features": [x, z, metal, ...] (every FEATURE_EVERY frames, and the last: wrecks and whatever else holds metal)}]}
The dashboard adds each def's human name and icon when it serves one.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bots"), str(ROOT / "sim" / "py")]
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402

# 2: "defs" names each engine def id (1 up) by the catalog record holding it.  Format 1 took the catalog's list from 0, so it named
# each def as the next unit (a metal extractor as a missile silo); `repair` names them again from the ids in the frames.
FORMAT = 2
# hp and build are percent; flags: 1 cloaked, 2 paralyzed; order is rawframe's kind (-1 none), to the point (tx, tz) or the unit
# `target` (-1 none).
UNIT_FIELDS = ("uid", "def", "side", "x", "z", "hp", "build", "flags", "order", "tx", "tz", "target")
SIDE_FIELDS = ("metal", "metal_income", "energy_income", "value", "units")
FEATURE_EVERY = 10


def by_def() -> dict[int, dict]:
    """The catalog's records by their engine def id."""
    return {int(u["def_id"]): u for u in json.loads((zksim.DATA / "catalog.json").read_text("utf8"))["units"]}


def named(defs, records: dict[int, dict]) -> dict[str, dict]:
    return {str(d): {"name": records[d]["name"], "cost": records[d]["cost"]} for d in sorted(defs) if d in records}


def save(path: pathlib.Path, replay: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", compresslevel=6) as f:
        json.dump(replay, f, separators=(",", ":"))
    os.replace(temp, path)


def repair(paths) -> list[pathlib.Path]:
    """Names format 1's defs again, from the def ids in each replay's frames; replays already at FORMAT are left alone."""
    records, mended = by_def(), []
    for path in paths:
        with gzip.open(path, "rb") as f:
            if not f.read(16).startswith(b'{"format":1,'):
                continue
        with gzip.open(path, "rt") as f:
            replay = json.load(f)
        n, at = len(replay["unit_fields"]), replay["unit_fields"].index("def")
        used = {u[i] for frame in replay["frames"] for u in (frame["units"],) for i in range(at, len(u), n)}
        replay["defs"], replay["format"] = named(used, records), FORMAT
        save(path, replay)
        mended.append(path)
    return mended


class Capture:
    """Records the first game of each of the env's first `games` slots, from now (after `env.begin()`), and writes each as it ends."""

    def __init__(self, env, out: pathlib.Path | str, games: int, stage: str, label: str, sides: tuple[str, str] = ("policy", "army script"),
                 prefix: str = "", info: dict | None = None, every: float = 1.0):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage, self.label, self.sides, self.prefix, self.info = stage, label, sides, prefix, dict(info or {})
        self.records = by_def()
        self.slugs = {id(b): f.name for _, b, f in env.maps}
        self.step_seconds = env.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        self.every = max(1, round(every / self.step_seconds))
        self.frames: dict[int, list[dict]] = {g: [] for g in range(min(games, env.count))}
        self.defs: dict[int, set[int]] = {g: set() for g in self.frames}
        self.written: list[pathlib.Path] = []
        for g in self.frames:
            self._frame(env, g, features=True)

    @property
    def done(self) -> bool:
        return not self.frames

    def after_step(self, env) -> None:
        ended = set(env._replace)        # games that ended this step, not yet started afresh (SimEnv.finished_seats does that)
        for g in list(self.frames):
            if g in ended:
                self._frame(env, g, features=True)
                self._write(env, g)
            elif env.decisions[g] % self.every == 0:
                self._frame(env, g, features=len(self.frames[g]) % FEATURE_EVERY == 0)

    def _frame(self, env, g: int, features: bool) -> None:
        b = env.batch
        rows, sides, kept = [], [], []
        for side in (0, 1):
            u = b.records[g, side, :int(b.counts[g, side])]
            u = u[u["player"] == side]
            built = (u["flags"] & rf.BEING_BUILT) == 0
            flags = ((u["flags"] & rf.CLOAKED) != 0).astype(np.int64) | 2 * ((u["flags"] & rf.PARALYZED) != 0)
            nowhere = (np.abs(u["ord_x"]) < 0.5) & (np.abs(u["ord_z"]) < 0.5)      # an order on a unit, or none, leaves its point at 0, 0
            rows.append(np.stack([u["uid"], u["def_id"], np.full(len(u), side), np.rint(u["x"]), np.rint(u["z"]),
                                  np.rint(np.clip(u["hp_frac"], 0, 1) * 100), np.where(built, 100, np.rint(np.clip(u["build_frac"], 0, 1) * 100)),
                                  flags, u["ord_kind"], np.where(nowhere, -1, np.rint(u["ord_x"])), np.where(nowhere, -1, np.rint(u["ord_z"])),
                                  u["ord_target"]], axis=1).astype(np.int64))
            self.defs[g].update(int(d) for d in np.unique(u["def_id"]))
            sides.append([round(float(b.globs[g, side, rf.G["metal"]]), 1), round(float(b.globs[g, side, rf.G["metal_income"]]), 2),
                          round(float(b.globs[g, side, rf.G["energy_income"]]), 2), round(float(b.stats[g, side, 0]), 1), int(b.stats[g, side, 1])])
            if features:
                f = b.features[g, side, :int(b.feature_counts[g, side])]
                kept.append(f[f["metal"] > 0])
        frame = {"t": round(float(env.decisions[g]) * self.step_seconds, 2), "units": np.concatenate(rows).ravel().tolist(), "sides": sides}
        if features:
            f = np.concatenate(kept)
            _, first = np.unique(f["feat_id"], return_index=True)
            f = f[first]
            frame["features"] = np.stack([np.rint(f["x"]), np.rint(f["z"]), np.rint(f["metal"])], axis=1).astype(np.int64).ravel().tolist()
        self.frames[g].append(frame)

    def _write(self, env, g: int) -> None:
        frames = self.frames.pop(g)
        defs = self.defs.pop(g)
        result = float(env.result[g])
        winner = 0 if result > 0 else (1 if result < 0 else None)
        facts = [env._game_facts(g, side, winner == side, winner == 1 - side) for side in (0, 1)]
        bundle = env.bundle[g]
        name = f"{self.prefix}game{g}"
        replay = {
            "format": FORMAT, "stage": self.stage, "label": self.label, "slot": g, "info": self.info,
            "map": {"slug": self.slugs.get(id(bundle), "?"), "width": float(bundle.map_w), "height": float(bundle.map_h)},
            "step_seconds": self.step_seconds, "every": self.every * self.step_seconds,
            "sides": [{"name": self.sides[s], "variant": facts[s]["variant"], "factory": facts[s]["own_factory"], "value": facts[s]["own_value"]}
                      for s in (0, 1)],
            "result": {"winner": winner, "how": facts[0]["how"], "seconds": facts[0]["seconds"]},
            "spots": np.round(bundle.spot_list, 2).tolist(),
            "defs": named(defs, self.records),
            "unit_fields": UNIT_FIELDS, "side_fields": SIDE_FIELDS, "frames": frames,
        }
        path = self.out / f"{name}.json.gz"
        save(path, replay)
        line = {"file": path.name, "stage": self.stage, "label": self.label, "slot": g, "map": replay["map"]["slug"],
                "winner": winner, "how": facts[0]["how"], "seconds": facts[0]["seconds"], "sides": replay["sides"],
                "frames": len(frames), "bytes": path.stat().st_size, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **self.info}
        with (self.out / "index.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        self.written.append(path)


def demo(args: argparse.Namespace) -> None:
    """The army scripts against each other on a few maps of the pool, recorded: a replay for the viewer without a trained actor."""
    import tempfile
    import policy_sim_env as se

    maps = pathlib.Path(args.maps)
    slugs = sorted(p.name[:-len(".facts.json")] for p in maps.glob("*.facts.json"))
    rng = np.random.default_rng(args.seed)
    picked = args.only.split(",") if args.only else list(rng.choice(slugs, min(args.games, len(slugs)), replace=False))
    with tempfile.TemporaryDirectory() as some:     # just these maps: the whole pool takes a while to load
        for slug in picked:
            for p in maps.glob(f"{slug}.*"):
                os.symlink(p, pathlib.Path(some) / p.name)
        pool = se.load_pool(some)
    env = se.SimEnv(args.games, args.seed, ["scripts"], maps=pool, variants=True)
    env.begin()
    env.capture = capture = Capture(env, args.out, args.games, args.stage, args.label, ("army script", "army script"))
    while not capture.done:
        env.step(0.0)
        env.finished_seats()
    for p in capture.written:
        print(p, p.stat().st_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="record the army scripts against each other")
    d.add_argument("--maps", default="/mnt/d/zk-learned/maps")
    d.add_argument("--out", required=True)
    d.add_argument("--games", type=int, default=4)
    d.add_argument("--only", help="these maps (slugs, comma-separated) rather than some drawn at random")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--stage", default="demo")
    d.add_argument("--label", default="army script against army script")
    r = sub.add_parser("repair", help="name format 1's units again, in these directories' replays")
    r.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    if args.command == "demo":
        demo(args)
    else:
        mended = repair(p for d in args.dirs for p in sorted(pathlib.Path(d).rglob("*.json.gz")))
        print(f"{len(mended)} replays named again" + "".join(f"\n  {p}" for p in mended))


if __name__ == "__main__":
    main()
