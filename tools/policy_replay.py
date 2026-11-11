"""Replays for the dashboard (tools/policy_dashboard.py): the first game of each of a SimEnv's first few slots, a frame a second,
each written as gzipped JSON with a line in its directory's index.jsonl.  Set as `env.capture`, SimEnv calls `Capture.after_step`
after every step, so a test's loop needs nothing more.

    ~/zkml/bin/python tools/policy_replay.py demo --maps /mnt/d/zk-learned/maps --out DIR     # the army scripts against each other
    ~/zkml/bin/python tools/policy_replay.py repair DIR...      # name format 1's units again (format 2)

A replay holds every unit of every team, from each team's own records (so what no team sees is there too).  Format 3:
  {"format": 3, "stage", "label", "slot", "info", "kind": the env's kind of game ("1v1", "teams", "ffa", "chicken"),
   "map": {"slug", "width", "height"}, "step_seconds", "every",
   "teams": [{"name", "ally", "variant", "factory", "value", "plays", "outcome"}, ...] in team order, `plays` "policy", "script" or
            "chickens", `outcome` the team's result (+1 won, -1 / (A - 1) out, A the alliances: policy_sim_env's),
   "result": {"winner": the winning allyteam or null, "how", "seconds"},
   "spots": [[x, z, income], ...], "defs": {"def id": {"name", "cost"}}, "unit_fields": UNIT_FIELDS, "team_fields": TEAM_FIELDS,
   "frames": [{"t": seconds, "teams": [[team 0's TEAM_FIELDS], ...], "units": [the unit fields, a unit after another],
               "features": [x, z, metal, ...] (every FEATURE_EVERY frames, a page's first and the game's last: wrecks and whatever
               else holds metal)}]}
A long game is paged: once a page holds PAGE_ROWS units, its frames' units and features go to a file of their own,
NAME.pK.json.gz = {"format": 3, "page": K, "first": its first frame, "frames": [{"units", "features"}, ...]}, and the replay keeps
only each frame's "t" and "teams", with "pages": [{"first", "count", "file"}, ...] for every page.  So the viewer holds a few pages
at a time, and a capture one page of each game it keeps, however long the game: an 80-minute 1v1 was 90 MB of JSON (2026-11-10).
Format 2, which the viewer still reads, is format 3 for two teams each its own alliance, without "kind" and the teams' "ally",
"plays" and "outcome", with "sides" for "teams" (the winner a side), "side" for the unit field "team" and "side_fields".
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

# 3: any number of teams in alliances, and long games paged.  2: "defs" names each engine def id (1 up) by the catalog record
# holding it.  Format 1 took the catalog's list from 0, so it named each def as the next unit (a metal extractor as a missile silo);
# `repair` names them again from the ids in the frames, which makes it format 2.
FORMAT = 3
REPAIRED = 2
# hp and build are percent; flags: 1 cloaked, 2 paralyzed; order is rawframe's kind (-1 none), to the point (tx, tz) or the unit
# `target` (-1 none).
UNIT_FIELDS = ("uid", "def", "team", "x", "z", "hp", "build", "flags", "order", "tx", "tz", "target")
TEAM_FIELDS = ("metal", "metal_income", "energy_income", "value", "units")
FEATURE_EVERY = 10
# Units a page holds: about 12 MB of JSON and 1.5 MB gzipped.  A 20-minute 1v1 fits in one; an 80-minute one took eight.
PAGE_ROWS = 250_000


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
    """Names format 1's defs again, from the def ids in each replay's frames, which makes it format 2; others are left alone."""
    records, mended = by_def(), []
    for path in paths:
        with gzip.open(path, "rb") as f:
            if not f.read(16).startswith(b'{"format":1,'):
                continue
        with gzip.open(path, "rt") as f:
            replay = json.load(f)
        n, at = len(replay["unit_fields"]), replay["unit_fields"].index("def")
        used = {u[i] for frame in replay["frames"] for u in (frame["units"],) for i in range(at, len(u), n)}
        replay["defs"], replay["format"] = named(used, records), REPAIRED
        save(path, replay)
        mended.append(path)
    return mended


# What a game's teams are, from the env where it says (policy_sim_env's teams, 2026-11-10), else as SimEnv's two sides were: two
# teams, each its own alliance, the result team 0's.
def n_teams(env, g: int) -> int:
    n = getattr(env, "n_teams", None)
    return 2 if n is None else int(n[g])


def ally_of(env, g: int, team: int) -> int:
    ally = getattr(env, "ally", None)
    return team if ally is None else int(ally[g, team])


def kind_of(env, g: int) -> str:
    kind = getattr(env, "kind", None)
    return "1v1" if kind is None else str(kind[g])


def plays(env, g: int, team: int) -> str:
    if hasattr(env, "plays"):
        return str(env.plays(g, team))
    is_policy = getattr(env, "_is_policy", None)
    return "policy" if is_policy is not None and is_policy(2 * g + team) else "script"


def outcomes(env, g: int, n: int) -> list[float]:
    outcome = getattr(env, "outcome", None)
    if outcome is not None:
        return [float(outcome[g, t]) for t in range(n)]
    result = float(env.result[g])
    return [result, -result]


class _Kept:
    """One game being recorded: every frame's time and teams' numbers, and the page of units not yet written."""

    def __init__(self, name: str):
        self.name = name
        self.head: list[dict] = []
        self.page: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.rows = 0
        self.first = 0
        self.pages: list[dict] = []
        self.defs: set[int] = set()
        self.bytes = 0


def _page_frame(units: np.ndarray, features: np.ndarray | None) -> dict:
    return {"units": units.ravel().tolist(), **({"features": features.ravel().tolist()} if features is not None else {})}


class Capture:
    """Records the first game of each of the env's first `games` slots, from now (after `env.begin()`), and writes each as it ends.
    sides: what to call a team by who plays it, the policy and the scripts (with SimEnv's two sides, side 0 and side 1).  The
    chickens are "chickens".  page_rows: units a page holds (PAGE_ROWS).
    Of the env it reads `batch`'s arrays, `count`, `ticks`, `maps`, `bundle`, `decisions`, `_replace` and `_game_facts`; and its
    teams where it has them (`n_teams`, `ally`, `kind`, `plays`, `outcome`), else two teams each its own alliance and `result`."""

    def __init__(self, env, out: pathlib.Path | str, games: int, stage: str, label: str, sides: tuple[str, str] = ("policy", "army script"),
                 prefix: str = "", info: dict | None = None, every: float = 1.0, page_rows: int = PAGE_ROWS):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage, self.label, self.sides, self.prefix, self.info = stage, label, sides, prefix, dict(info or {})
        self.page_rows = page_rows
        self.records = by_def()
        self.slugs = {id(b): f.name for _, b, f in env.maps}
        self.step_seconds = env.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        self.every = max(1, round(every / self.step_seconds))
        self.kept: dict[int, _Kept] = {g: _Kept(f"{prefix}game{g}") for g in range(min(games, env.count))}
        self.written: list[pathlib.Path] = []
        for g in self.kept:
            self._frame(env, g, features=True)

    @property
    def done(self) -> bool:
        return not self.kept

    def after_step(self, env) -> None:
        ended = set(env._replace)        # games that ended this step, not yet started afresh (SimEnv.finished_seats does that)
        for g in list(self.kept):
            if g in ended:
                self._frame(env, g, features=True)
                self._write(env, g)
            elif env.decisions[g] % self.every == 0:
                self._frame(env, g, features=len(self.kept[g].head) % FEATURE_EVERY == 0)

    def _name(self, env, g: int, team: int) -> str:
        if not hasattr(env, "plays"):
            return self.sides[team]         # as SimEnv's two sides were named
        who = plays(env, g, team)
        return "chickens" if who == "chickens" else self.sides[0] if who == "policy" else self.sides[1]

    def _frame(self, env, g: int, features: bool) -> None:
        kept = self.kept[g]
        if kept.rows >= self.page_rows:
            self._flush(kept)
            features = True         # each page has its own, from its first frame
        b = env.batch
        rows, teams, lying = [], [], []
        for team in range(n_teams(env, g)):
            u = b.records[g, team, :int(b.counts[g, team])]
            u = u[u["player"] == team]          # its own: its allies' are in their own frames
            built = (u["flags"] & rf.BEING_BUILT) == 0
            flags = ((u["flags"] & rf.CLOAKED) != 0).astype(np.int64) | 2 * ((u["flags"] & rf.PARALYZED) != 0)
            nowhere = (np.abs(u["ord_x"]) < 0.5) & (np.abs(u["ord_z"]) < 0.5)      # an order on a unit, or none, leaves its point at 0, 0
            rows.append(np.stack([u["uid"], u["def_id"], np.full(len(u), team), np.rint(u["x"]), np.rint(u["z"]),
                                  np.rint(np.clip(u["hp_frac"], 0, 1) * 100), np.where(built, 100, np.rint(np.clip(u["build_frac"], 0, 1) * 100)),
                                  flags, u["ord_kind"], np.where(nowhere, -1, np.rint(u["ord_x"])), np.where(nowhere, -1, np.rint(u["ord_z"])),
                                  u["ord_target"]], axis=1).astype(np.int32))
            kept.defs.update(int(d) for d in np.unique(u["def_id"]))
            teams.append([round(float(b.globs[g, team, rf.G["metal"]]), 1), round(float(b.globs[g, team, rf.G["metal_income"]]), 2),
                          round(float(b.globs[g, team, rf.G["energy_income"]]), 2), round(float(b.stats[g, team, 0]), 1), int(b.stats[g, team, 1])])
            if features:
                f = b.features[g, team, :int(b.feature_counts[g, team])]
                lying.append(f[f["metal"] > 0])
        units = np.concatenate(rows)
        wrecks = None
        if features:
            f = np.concatenate(lying)
            _, first = np.unique(f["feat_id"], return_index=True)
            f = f[first]
            wrecks = np.stack([np.rint(f["x"]), np.rint(f["z"]), np.rint(f["metal"])], axis=1).astype(np.int32)
        kept.head.append({"t": round(float(env.decisions[g]) * self.step_seconds, 2), "teams": teams})
        kept.page.append((units, wrecks))
        kept.rows += len(units)

    def _flush(self, kept: _Kept) -> None:
        """The page so far to its own file."""
        k = len(kept.pages)
        path = self.out / f"{kept.name}.p{k}.json.gz"
        save(path, {"format": FORMAT, "page": k, "first": kept.first, "frames": [_page_frame(u, f) for u, f in kept.page]})
        kept.pages.append({"first": kept.first, "count": len(kept.page), "file": path.name})
        kept.bytes += path.stat().st_size
        kept.first += len(kept.page)
        kept.page, kept.rows = [], 0

    def _write(self, env, g: int) -> None:
        kept = self.kept.pop(g)
        n = n_teams(env, g)
        allies = [ally_of(env, g, t) for t in range(n)]
        outcome = outcomes(env, g, n)
        winners = [t for t in range(n) if outcome[t] >= 1.0 - 1e-6]       # a sole surviving alliance's teams
        winner = allies[winners[0]] if winners else None
        facts = [env._game_facts(g, t, winner is not None and allies[t] == winner, outcome[t] < 0) for t in range(n)]
        # How the game ended, as a team still in it at the end saw it.
        still = [t for t in range(n) if winner is not None and allies[t] == winner] or [t for t in range(n) if outcome[t] >= 0] or [0]
        end = facts[still[0]]
        if kept.pages:
            self._flush(kept)
            frames = kept.head
        else:
            frames = [{**h, **_page_frame(u, f)} for h, (u, f) in zip(kept.head, kept.page)]
        bundle = env.bundle[g]
        replay = {
            "format": FORMAT, "stage": self.stage, "label": self.label, "slot": g, "info": self.info, "kind": kind_of(env, g),
            "map": {"slug": self.slugs.get(id(bundle), "?"), "width": float(bundle.map_w), "height": float(bundle.map_h)},
            "step_seconds": self.step_seconds, "every": self.every * self.step_seconds,
            "teams": [{"name": self._name(env, g, t), "ally": allies[t], "variant": facts[t].get("variant"), "factory": facts[t].get("own_factory"),
                       "value": facts[t].get("own_value"), "plays": plays(env, g, t), "outcome": round(outcome[t], 4) + 0.0} for t in range(n)],
            "result": {"winner": winner, "how": end["how"], "seconds": end["seconds"]},
            "spots": np.round(bundle.spot_list, 2).tolist(),
            "defs": named(kept.defs, self.records),
            "unit_fields": UNIT_FIELDS, "team_fields": TEAM_FIELDS, "frames": frames,
            **({"pages": kept.pages} if kept.pages else {}),
        }
        path = self.out / f"{kept.name}.json.gz"
        save(path, replay)
        line = {"file": path.name, "format": FORMAT, "stage": self.stage, "label": self.label, "slot": g, "map": replay["map"]["slug"],
                "kind": replay["kind"], "winner": winner, "how": end["how"], "seconds": end["seconds"], "teams": replay["teams"],
                "frames": len(frames), "pages": max(1, len(kept.pages)), "bytes": path.stat().st_size + kept.bytes,
                "written": time.strftime("%Y-%m-%d %H:%M:%S"), **self.info}
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
