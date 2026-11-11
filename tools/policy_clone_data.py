"""Games to learn from before RL: the army script against itself in the sim, every decision of both sides kept with what the script told it.

    ~/zkml/bin/python tools/policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 64 --decisions 2000000 --workers 4 [--maps /mnt/d/zk-learned/maps]
    ~/zkml/bin/python tools/policy_clone_data.py check --data /mnt/d/zk-learned/clone/NAME
    ~/zkml/bin/python tools/policy_clone_data.py zindex --data /mnt/d/zk-learned/clone/NAME --workers 15

Games run as a batch in the sim, which steps them on its threads, and the scripts on theirs; `--workers` runs that many batches in
processes of their own, each with its own seed.  `--variants` gives each side's army script its own knobs at each game's start
(policy_variants.py), and each game's meta says which: without it every side plays the standard script, as before.

A file a game (`game-*.zkr`, data version 7): each recorded side's frames in order, the orders its scripts gave at each, as the
policy would give them (a group order as its anchor's order and a groups.GROUP_MARK record after it; the army script's, then the
economy script's), and how the game ended.  A side is one of the game's teams; every player's is recorded (a chicken game's
chickens, whom no one plays, aren't).  libzkrecord writes it straight from the batch's arrays, narrowed and compressed
(sim/record/zkrecord.h); `zkrecord.read` gives it back.  `--kinds` plays team games, free-for-alls and chicken games too
(policy_sim_env.GAME_KINDS), `--slots` the teams a game has room for, `--chicken-difficulties` how likely each chicken difficulty
is.  Its meta says, a list with one entry a side, each one's team, ally, script, variant, result (`SimEnv.outcome`: 1 won,
0 drew, below 0 lost), how it ended for it (eliminated, lead, limit, or queen: the chickens' queen died) and `until`, the frames
it played: an alliance out of a game that goes on stops there, and `sides` lists only those.  Of the game it says its map and its
kind ("1v1", "teams", "ffa", "chicken"), and a chicken game its difficulty (a name of zksim.CHICKEN_DIFFICULTIES).  Data
versions 4 to 6 held both sides of a 1v1, teams 0 and 1, and said neither teams, allies nor the kind: `side_info` reads them as
that.  Versions 1 to 3 were a file a side (`game-*-SIDE.npz`, rawframe.save_frames): `read_game` reads both.  Frames and orders
are kept raw, not as features or choices, so the data outlives changes to the featurizer: `labels` turns them into the policy's
choices when they're read.  The commander's opening orders aren't kept: they aren't the policy's to give, so the commander's
label is to keep doing what it does.
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

DATA_VERSION = 7      # 1: the army script with the opening's economy only.  2: with the economy script (policy_eco.py), 2026-11-08.
                      # 3: the army script aims missiles and the Trinity (labels now), the economy spends what it makes, and in the sim
                      # help for a factory still being built builds it, 2026-11-08.  4: the same games, a .zkr file a game, 2026-11-08.
                      # 5: the army script's group orders, recorded as the policy gives them (groups.GROUP_MARK), air, and a second
                      # factory, 2026-11-08.  6: the economy script's orders too, after the army script's for each seat, both
                      # scripts looking every 6 slots, 2026-11-09 (03-training.md, "Script data").  7: any number of sides a file
                      # ("ZKREC002"), each side's team and ally, the game's kind and how it ended in the meta, 2026-11-10; then
                      # team games and free-for-alls, with how it ended and `until` for each side (`side_info` reads both)


class Recorder:
    """Every recorded side of every game of an environment, until each game ends; then a file for it.

    After `close`, games that start aren't kept, and those running are played out, so that the data isn't biased towards short
    games, as it would be if those running were dropped; `closed` says when they're all written."""

    def __init__(self, env: se.SimEnv, out: pathlib.Path, prefix: str = "game", level: int = 3, threads: int = 1):
        self.env, self.out, self.prefix = env, out, prefix
        out.mkdir(parents=True, exist_ok=True)
        self.slugs = {id(b): f.name for _, b, f in env.maps}
        self.slots = env.batch.counts.shape[1]      # the batch's teams a game, which its arrays are laid out by
        self.writer = zkrecord.Writer(env.count, threads, level, teams=self.slots)
        self.games = self.decisions = 0
        self.ended: Counter = Counter()
        # By game, by team: how it ended for the team, and how many frames the game had then (an alliance out stops playing, its
        # frames after that empty of its own units; they're recorded with the rest, and `sides` lists only those before).
        self.how: dict[int, dict[int, tuple[str, int]]] = {}
        self.kept = set(range(env.count))     # games whose present run will be written
        self._empty = np.zeros(0, zksim.ORDER_DTYPE)
        for g in self.kept:
            self._begin(g)

    def _sides(self, g: int) -> tuple[list[int], list[int], list[float]]:
        """Game g's teams that are recorded, in the file's order, each one's ally, and how the game ended for each
        (`SimEnv.outcome`): every player's team (not the chickens', whom no one plays)."""
        env = self.env
        teams = [t for t in range(int(env.n_teams[g])) if env.plays(g, t) != "chickens"]
        return teams, [int(env.ally[g, t]) for t in teams], [float(env.outcome[g, t]) for t in teams]

    def close(self) -> None:
        self.closing = True

    closing = False

    @property
    def closed(self) -> bool:
        return self.closing and not self.kept

    def _begin(self, g: int) -> None:
        bundle = self.env.bundle[g]
        self.writer.begin(g, bundle.map_w, bundle.map_h, zkrecord.height_range(bundle.height), self.env.batch.sites[g], teams=self._sides(g)[0])

    def _orders(self) -> tuple[np.ndarray, np.ndarray]:
        """This step's script orders over every seat, and where each seat's start."""
        env = self.env
        if env.army_orders is not None:
            if env.eco_orders is None:
                return env.army_orders
            # Each seat's army-script orders, then its economy script's (data version 6).
            (army, ao), (eco, eo) = env.army_orders, env.eco_orders
            parts = [p for s in range(len(ao) - 1) for p in (army[ao[s]:ao[s + 1]], eco[eo[s]:eo[s + 1]])]
            counts = np.diff(ao) + np.diff(eo)
            return rf.join(parts, zksim.ORDER_DTYPE), np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        parts = [env.script_record.get(seat, self._empty) for seat in range(env.seat_count)]
        return rf.join(parts, zksim.ORDER_DTYPE), np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int32)

    def step(self) -> list[tuple[int, str, float]]:
        """Records a step of the environment, and returns its finished seats (`SimEnv.finished_seats`)."""
        env = self.env
        listed = np.array(sorted(self.kept), np.int32)
        self.writer.frames(env.batch, listed)
        env.step(0.0)
        self.writer.orders(listed, *self._orders())
        for facts in env.outcomes:          # every team whose alliance went out, or whose game ended
            g = facts["game"]
            if g in self.kept:
                self.how.setdefault(g, {})[facts["team"]] = (facts["how"], self.writer.frames_held(g))
        env.outcomes.clear()
        for g in env._replace:
            if g in self.kept:
                self.ended[self.how.get(g, {}).get(0, ("?",))[0]] += 1      # each game once, by how it ended for team 0
        ended = [g for g in env._replace if g in self.kept]
        for g in ended:
            self._write(g)
            if self.closing:
                self.kept.discard(g)
        finished = env.finished_seats()        # which starts them afresh
        for g in ended:
            if g in self.kept:
                self._begin(g)
        if ended and self.closed:
            self.writer.wait()      # so that every file is on disk once `closed` says so
        return finished

    def _write(self, g: int) -> None:
        env = self.env
        frames = self.writer.frames_held(g)
        teams, allies, results = self._sides(g)
        how = self.how.pop(g, {})
        meta = {"data_version": DATA_VERSION, "schema_version": rf.SCHEMA_VERSION, "map": self.slugs.get(id(env.bundle[g]), "?"),
                "script": [env.script_kinds[g][t] for t in teams], "result": results, "teams": teams, "allies": allies,
                "kind": env.kind[g], "how": [how.get(t, (None, frames))[0] for t in teams],
                "until": [how.get(t, (None, frames))[1] for t in teams]}
        if env.chickens[g] >= 0:
            meta["difficulty"] = zksim.CHICKEN_DIFFICULTIES[int(env.difficulty[g])]
        if env.styles[g] is not None:
            meta["variants"] = [env.styles[g][t].as_dict() for t in teams]     # each side's army script knobs (policy_variants.py)
        self.writer.finish(g, self.out / f"{self.prefix}-{self.games:06d}{zkrecord.SUFFIX}", meta)
        self.decisions += sum(meta["until"])
        self.games += 1


Side = tuple[list[rf.Frame], list[np.ndarray], dict]


def side_info(meta: dict, s: int) -> dict:
    """What a file's meta says of its side s: how the game ended for it (result, and how: eliminated, lead, limit or queen), the
    map, its script and variant, its team and ally, the kind of game, a chicken game's difficulty (else None), and `until`, the
    frames it played: up to its alliance going out, all the file's if it was in to the end (None where a file doesn't say, as the
    first files of version 7 don't).  Data version 6 and before held the two sides of a 1v1, side s being team s and ally s; the
    first files of version 7 said one `how` for the game."""
    teams = meta.get("teams", (0, 1))
    allies = meta.get("allies", teams)
    how = meta.get("how")
    return {"result": float(meta["result"][s]), "map": meta["map"], "script": meta["script"][s],
            "variant": meta["variants"][s] if "variants" in meta else None, "team": int(teams[s]), "ally": int(allies[s]),
            "kind": meta.get("kind", "1v1"), "how": how[s] if isinstance(how, list) else how,
            "until": int(meta["until"][s]) if "until" in meta else None, "difficulty": meta.get("difficulty")}


def _meta(path: pathlib.Path) -> dict:
    """A .zkr file's meta alone, from its start."""
    with open(path, "rb") as f:
        f.read(8)
        return json.loads(f.read(int(np.frombuffer(f.read(4), "<u4")[0])))


def read_game(path: pathlib.Path) -> list[Side]:
    """Every side a file holds, as (frames, the orders given at each, the rest): each of a .zkr game's, the one of an .npz."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        frames = rf.load_frames(path)
        with np.load(path) as d:
            orders = np.split(d["orders"], np.cumsum(d["order_counts"])[:-1]) if len(d["order_counts"]) else []
            info = {"result": float(d["result"]), "map": str(d["map"]), "script": str(d["script"])}
        return [(frames, orders, info)]
    meta, sides = zkrecord.read(path)
    return [(frames, orders, side_info(meta, s)) for s, (frames, orders) in enumerate(sides)]


def read_side(path: pathlib.Path, side: int, until: int | None = None) -> Side:
    """One side of a game, decoding only it and only as far as frame `until` (all of a .npz): what a cloning sample needs."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        return read_game(path)[0]
    meta, sides = zkrecord.read(path, side=side, until=until)
    frames, orders = sides[side]
    return frames, orders, side_info(meta, side)


def load(path: pathlib.Path, side: int = 0) -> Side:
    """One side of one game: frames, the orders given at each, and the rest.  An .npz holds one side, its side 0 here."""
    return read_game(path)[side]


def sides(data: pathlib.Path) -> list[tuple[str, int, int]]:
    """Every side of every finished game in a dataset: (path, side in `read_game`, the frames it played).  A side whose alliance
    went out before the end played fewer than its file holds (`side_info`'s until)."""
    out = []
    paths = sorted(data.glob("game-*.zkr"))
    # Each file's length and sides are in its header, and how long each side played in its meta, but reading 84k headers over
    # /mnt/d took 3 to 10 minutes a run: kept in sides.npz, good while the files listed are the same ones.  One written before
    # files had any number of sides has no `sides`, its files two sides each; before sides could stop early, no `side_frames`.
    cache = data / "sides.npz"
    known: dict[str, tuple[int, list[int]]] = {}
    if cache.exists():
        with np.load(cache) as c:
            frames = c["frames"].tolist()
            counts = c["sides"].tolist() if "sides" in c.files else [2] * len(frames)
            each = np.split(c["side_frames"], np.cumsum(counts)[:-1]) if "side_frames" in c.files else [[n] * k for n, k in zip(frames, counts)]
            known = {name: (n, [int(v) for v in e]) for name, n, e in zip(c["names"].tolist(), frames, each)}
    counted: dict[str, tuple[int, list[int]]] = {}
    for path in paths:
        found = known.get(path.name)
        if found is None:
            try:
                frames, teams = zkrecord.header(path)
                until = _meta(path).get("until") or [frames] * len(teams)
            except (OSError, ValueError):
                continue
            found = (frames, [min(int(u), frames) for u in until])
        counted[path.name] = found
        out += [(str(path), s, n) for s, n in enumerate(found[1])]
    if counted.keys() != known.keys():
        try:
            np.savez(cache, names=np.array(list(counted)), frames=np.array([n for n, _ in counted.values()], np.int64),
                     sides=np.array([len(e) for _, e in counted.values()], np.int64),
                     side_frames=np.array([v for _, e in counted.values() for v in e], np.int64))
        except OSError:
            pass
    for path in sorted(data.glob("game-*.npz")):
        try:
            with np.load(path) as d:
                n = len(d["unit_counts"])
        except (OSError, ValueError, KeyError, EOFError):        # still being written
            continue
        out.append((str(path), 0, n))
    return [s for s in out if s[2]]


# A side's strategy, z (04-literature.md P8, C5): what the actor is told it's playing, as AlphaStar's statistic.  Z_DIM numbers:
# whether the knobs are known, whether the counts are, the army script's knobs (policy_variants.py: the role weights and commit,
# gather and patience in log, patrol_share and retreat_below about the standard's), its preset (rush, turtle, skirmish; none is the
# standard), and how many of each of Z_CATEGORIES the side had finished at Z_MINUTES, log1p / 3.  A game over before a minute
# counts as it ended.  Unknown parts are zeros, with their flag 0.
Z_MINUTES = (3, 6, 10)
Z_CATEGORIES = ("extractor", "energy", "factory", "defence", "army")
Z_PRESETS = ("rush", "turtle", "skirmish")
Z_KNOWN, Z_COUNTED = 0, 1
Z_KNOBS = slice(2, 2 + 7 + 5)
Z_PRESET = slice(14, 17)
Z_COUNTS = slice(17, 17 + len(Z_MINUTES) * len(Z_CATEGORIES))
Z_DIM = 32
FRAMES_A_MINUTE = 120           # recorded at 2 a second


def variant_z(variant: dict | None) -> np.ndarray:
    """The knobs' part of z (and its flag), from a recording's variant (Variant.as_dict) or the variant itself; zeros for None."""
    import policy_variants as pv
    z = np.zeros(Z_DIM, np.float32)
    if variant is None:
        return z
    if isinstance(variant, pv.Variant):
        variant = variant.as_dict()
    std = pv.STANDARD
    z[Z_KNOWN] = 1.0
    roles = [np.log(variant["role_weight"][r]) for r in pv.ROLES]
    z[Z_KNOBS] = roles + [np.log(variant["commit"]), np.log(variant["gather"]), np.log(variant["patience"]),
                          (variant["patrol_share"] - std.patrol_share) * 5.0, (variant["retreat_below"] - std.retreat_below) * 5.0]
    if variant["name"] in Z_PRESETS:
        z[Z_PRESET.start + Z_PRESETS.index(variant["name"])] = 1.0
    return z


def _category_rows(table: tables.TypeTable) -> np.ndarray:
    """categories x rows: which type-table rows count as each of Z_CATEGORIES."""
    from policy_eco import EcoData
    eco = EcoData(table, se.catalog())
    mobile, armed = np.asarray(table.mobile), np.asarray(table.armed)
    out = np.zeros((len(Z_CATEGORIES), len(mobile)), bool)
    out[0] = eco.extractor
    out[1] = (eco.energy > 0) & ~eco.extractor & ~eco.commander & ~mobile
    out[2] = np.asarray(table.factory)
    out[3] = ~mobile & armed & ~out[2] & ~eco.commander
    out[4] = mobile & np.asarray(table.combat) & ~np.asarray(table.builder) & ~eco.commander
    return out


def counts_z(frames: list[rf.Frame], player: int, table: tables.TypeTable, rows_of: np.ndarray) -> np.ndarray:
    """The counts' part of z (and its flag), from a side's frames from the start."""
    z = np.zeros(Z_DIM, np.float32)
    z[Z_COUNTED] = 1.0
    out = []
    for m in Z_MINUTES:
        f = frames[min(m * FRAMES_A_MINUTE, len(frames) - 1)]
        u = f.units[(f.units["player"] == player) & ((f.units["flags"] & rf.BEING_BUILT) == 0)]
        rows = table.rows(u["def_id"])
        out += [float(np.log1p(rows_of[c, rows].sum()) / 3.0) for c in range(len(Z_CATEGORIES))]
    z[Z_COUNTS] = out
    return z


_Z: dict = {}


def _zindex_game(path: str) -> tuple[str, np.ndarray, np.ndarray]:
    """A file's sides' z (sides x Z_DIM) and results, by its name."""
    if not _Z:
        _Z["table"] = tables.table()
        _Z["rows"] = _category_rows(_Z["table"])
    meta, every = zkrecord.read(pathlib.Path(path), until=max(Z_MINUTES) * FRAMES_A_MINUTE + 1)
    z = np.zeros((len(every), Z_DIM), np.float32)
    for side, (frames, _) in enumerate(every):
        info = side_info(meta, side)
        played = frames[:info["until"]] if info["until"] else frames      # an alliance out counts as it went out, as a game over does
        z[side] = variant_z(info["variant"]) + counts_z(played, info["team"], _Z["table"], _Z["rows"])
    return pathlib.Path(path).name, z, np.array(meta["result"], np.float32)


def zindex(args: argparse.Namespace) -> None:
    """z.npz in the dataset, a row a side: each side's file name, its side in the file, its z and its result, for the cloning
    trainer (--z, --finish-winners)."""
    data = pathlib.Path(args.data)
    paths = sorted(str(p) for p in data.glob("game-*.zkr"))
    began = time.perf_counter()
    if args.workers <= 1:       # in this process, which may have threads of its own that a fork would leave behind
        out = [_zindex_game(p) for p in paths]
    else:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            out = list(pool.imap(_zindex_game, paths, chunksize=8))
    names = np.array([name for name, z, _ in out for _ in range(len(z))])
    side = np.array([s for _, z, _ in out for s in range(len(z))], np.int32)
    np.savez(data / "z.npz", names=names, side=side, z=np.concatenate([z for _, z, _ in out]) if out else np.zeros((0, Z_DIM), np.float32),
             result=np.concatenate([r for _, _, r in out]) if out else np.zeros(0, np.float32))
    print(json.dumps({"games": len(out), "sides": len(side), "seconds": round(time.perf_counter() - began, 1)}))


def load_z(data: pathlib.Path) -> dict[tuple[str, int], tuple[np.ndarray, float]]:
    """(file name, side) -> (z, result), from `zindex`'s z.npz: a row a side, or, as data version 6's was written, a row a file of
    its two sides."""
    # Each d[key] decompresses the whole array afresh, and a row of it keeps that copy alive: indexing d in the loop held 167k
    # copies of 21 MB and filled the machine's memory (2026-11-09).  Read once; copy the rows.
    with np.load(pathlib.Path(data) / "z.npz") as d:
        names, z, result = d["names"], d["z"], d["result"]
        side = d["side"] if "side" in d.files else None
    if side is None:
        return {(str(n), s): (z[i, s].copy(), float(result[i, s])) for i, n in enumerate(names) for s in range(z.shape[1])}
    return {(str(n), int(s)): (z[i].copy(), float(result[i])) for i, (n, s) in enumerate(zip(names, side))}


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
    env = se.SimEnv(args.games, args.seed * 1000 + worker, ["scripts"], threads=threads, maps=maps, weights=weights, variants=args.variants,
                    teams=args.slots or None, kinds=se.parse_kinds(args.kinds), difficulties=args.chicken_difficulties or None)
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
    difficulties = se.difficulty_weights(args.chicken_difficulties or se.CHICKEN_DIFFICULTY_WEIGHTS)
    summary = {"games": sum(p["games"] for p in parts), "decisions": decisions, "seconds": seconds, "loading_maps_seconds": loaded,
               "decisions_a_second_playing": decisions / max(max(p["seconds"] for p in parts), 1e-9), "ended": dict(ended),
               "bytes_a_decision": size / max(decisions, 1), "clamped": dict(clamped), "level": args.level,
               "maps": args.maps or "training", "seed": args.seed, "workers": args.workers, "data_version": DATA_VERSION,
               "variants": args.variants, "kinds": se.parse_kinds(args.kinds),
               "chicken_difficulties": {zksim.CHICKEN_DIFFICULTIES[d]: w for d, w in difficulties.items()}}
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
            t, d, m = labels(frame, o, bundle, table, enabled_kinds=se.KINDS, economy=True)
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
             rf.MANUAL_FIRE: "manual_fire", rf.JUMP: "jump", rf.CLOAK_ON: "cloak_on", rf.CLOAK_OFF: "cloak_off", rf.PRODUCE: "produce",
             rf.BUILD: "build", rf.ASSIST: "assist", rf.RECLAIM: "reclaim"}
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
    r.add_argument("--kinds", nargs="+", default=["1v1"], metavar="KIND:WEIGHT",
                   help="kinds of game and how likely each, among 1v1, teams, ffa and chicken, as 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1 "
                        "(default: %(default)s)")
    r.add_argument("--slots", type=int, default=0, help="teams a game has room for; 0: what the biggest of --kinds needs")
    r.add_argument("--chicken-difficulties", nargs="+", metavar="NAME:WEIGHT",
                   help="how likely each chicken difficulty is, as very-easy:1 easy:1 normal:1 (default policy_sim_env.CHICKEN_DIFFICULTY_WEIGHTS)")
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--maps")
    c.add_argument("--files", type=int, default=20, help="sides read")
    c.add_argument("--every", type=int, default=5, help="read every this many decisions")
    zi = sub.add_parser("zindex")
    zi.add_argument("--data", required=True)
    zi.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()
    {"record": record, "check": check, "zindex": zindex}[args.command](args)


if __name__ == "__main__":
    main()
