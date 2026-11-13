"""libzkpolicy against the Python it ports (sim/py/zkpolicy.py): the featurizer's every tensor, and the actions, bit for bit, on
recorded script games of every kind; and how fast each is.

    ~/zkml/bin/python tools/policy_parity.py corpus --data /mnt/d/zk-learned/clone/army-7 --out /mnt/d/zk-learned/parity/corpus
    ~/zkml/bin/python tools/policy_parity.py check --corpus /mnt/d/zk-learned/parity/corpus --maps /mnt/d/zk-learned/maps \
        --out /mnt/d/zk-learned/parity/check.json --workers 12
    ~/zkml/bin/python tools/policy_parity.py speed --corpus /mnt/d/zk-learned/parity/corpus --maps /mnt/d/zk-learned/maps \
        --out /mnt/d/zk-learned/parity/speed.json

`corpus` reads games of each kind (1v1, teams, free-for-all, chicken), a side or two of each, and keeps frames spread over each
game, its fullest (most records, most of its own, most wrecks: late 80-minute frames at the caps), and those with a transport
loaded or carried and with a factory plate: one .npz a side (rawframe.save_frames, with the script's orders at each frame).  Only
each game's map bundle is read, never the map pool.

`check`, for every frame: fz.featurize and zkpolicy.featurize as training calls it (the sim's kinds, economy, the critic's hidden
units) and, every 4th frame, with featurize's defaults (every kind, ATTACK_GROUND and STOP among them); every field compared to the
bit.  Also every 4th frame, with all 24 kinds enabled, a copy whose own units are partly made teleporters, beacons and switches
(with_specials), which the scripts never build.  Then on each side's own tensors: world_positions, the script's orders encoded (the cloning labels) and decoded again, a random
legal decision decoded and its orders encoded again, to_slots and to_decision, and group expansion.

`speed`: per frame, numba's featurize against zkpolicy's, and zkpolicy.featurize_many on 1, 8 and 16 threads.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zkpolicy as zp  # noqa: E402
from zkbot import catalog, units as bot_units  # noqa: E402
from zkbot.policy import actions as act, bundle as bd, featurize as fz, groups as gr, rawframe as rf, tables  # noqa: E402

KINDS = ("1v1", "teams", "ffa", "chicken")


# Comparing ------------------------------------------------------------------------------------------------------------------------------------------

def _bits(a: np.ndarray) -> np.ndarray:
    return a.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]) if a.dtype.kind == "f" else a


def tensor_diff(a: fz.Tensors, b: fz.Tensors) -> list[str]:
    """Every field of two fz.Tensors that isn't the same: dtype, shape, and each value to the bit (a float's bits, so -0.0 and 0.0
    differ, and a NaN equals only the same NaN)."""
    bad = []
    for f in dataclasses.fields(fz.Tensors):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            if not (x is None and y is None) and x != y:
                bad.append(f"{f.name}: {x!r} != {y!r}")
            continue
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(f"{f.name}: {x.dtype}{x.shape} != {y.dtype}{y.shape}")
        elif not np.array_equal(_bits(x), _bits(y)):
            at = np.argwhere(_bits(x) != _bits(y))
            bad.append(f"{f.name}: {len(at)} differ, first at {at[0].tolist()}: {x[tuple(at[0])]!r} != {y[tuple(at[0])]!r}")
    return bad


def _same_value(p, q) -> bool:
    p, q = float(p), float(q)
    return p == q or (p != p and q != q)


def orders_diff(a: list[act.Order], b: list[act.Order]) -> str | None:
    if len(a) != len(b):
        return f"{len(a)} orders != {len(b)}"
    for k, (p, q) in enumerate(zip(a, b)):
        same = (int(p.uid) == int(q.uid) and int(p.kind) == int(q.kind) and int(p.target) == int(q.target) and int(p.roles) == int(q.roles)
                and _same_value(p.x, q.x) and _same_value(p.z, q.z) and _same_value(p.radius, q.radius))
        if not same:
            return f"order {k}: {p} != {q}"
    return None


def decision_diff(a: act.Decision, b: act.Decision) -> str | None:
    for f in act.FIELDS:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if x.shape != y.shape or not np.array_equal(x, y):
            return f"{f}: {x.tolist()[:8]} != {y.tolist()[:8]}"
    return None


def random_decision(t: fz.Tensors, rng: np.random.Generator) -> act.Decision:
    """A legal choice for each controllable unit (a kind its mask allows, and what that kind points at), KEEP for about half."""
    c = t.controllable
    d = act.Decision(*(np.full(c, v, np.int64) for v in (rf.KEEP, -1, -1, -1, -1, -1, 0, -1, -1)))
    places = len(t.place_tokens) + len(t.site_pos)
    for i in range(c):
        legal = np.flatnonzero(t.kind_mask[i])
        if not len(legal) or rng.random() < 0.4:
            continue
        k = int(rng.choice(legal))
        d.kind[i] = k
        if k in fz.KINDS_WITH_REGION:
            mask = t.ptr_jump if k in fz.REACH_KINDS else t.ptr_wait if k == rf.WAIT_AT_BEACON else t.ptr_region
            d.region[i] = int(rng.choice(np.flatnonzero(mask[i])))
            d.fine[i] = int(rng.integers(fz.FINE_OPTIONS)) if k in fz.KINDS_WITH_FINE else -1
        elif k in fz.ENEMY_POINTER_KINDS:
            d.enemy[i] = int(rng.choice(np.flatnonzero(t.ptr_enemy[fz.ENEMY_POINTER_KINDS.index(k), i])))
            if k == rf.ATTACK_GROUND:
                d.fine[i] = int(rng.integers(fz.FINE_OPTIONS))
        elif k in (rf.PRODUCE, rf.BUILD):
            d.produce[i] = int(rng.choice(np.flatnonzero(t.ptr_produce[i])))
            if k == rf.BUILD:
                d.place[i], d.offset[i] = int(rng.integers(places)), int(rng.integers(fz.PLACE_OFFSETS))
        elif k == rf.RECLAIM:
            d.place[i] = int(rng.integers(len(t.feat_ids)))
        if k in gr.GROUP_KINDS and rng.random() < 0.5:
            d.group[i], d.roles[i] = int(rng.integers(len(gr.GROUP_RADII))), int(rng.integers(1 << 14))
    return d


# The corpus -----------------------------------------------------------------------------------------------------------------------------------------

def _special_ids(names) -> np.ndarray:
    return np.array(sorted(i for i in (bot_units.id_of(n) for n in names) if i is not None), np.int64)


def _pick(frames: list[rf.Frame], orders: list[np.ndarray], made: np.ndarray, rng: np.random.Generator, plates: np.ndarray, lifts: np.ndarray,
          spread: int) -> list[int]:
    """Which of the made frames to keep: spread over the game (half of them where the scripts gave orders), its fullest, and those
    with transports or plates."""
    stats = []
    for j in made:
        f = frames[j]
        u = f.units
        player = int(f.glob[rf.G["player"]])
        stats.append((int((u["knowledge"] != rf.KNOW_HIDDEN).sum()), int((u["player"] == player).sum()), len(f.features),
                      bool((u["flags"] & (rf.LOADED | rf.CARRIED)).any() or np.isin(u["def_id"][u["player"] == player], lifts).any()),
                      bool(np.isin(u["def_id"], plates).any())))
    stats = np.array(stats, np.int64).reshape(len(made), 5)
    keep = set(np.linspace(0, len(made) - 1, min(spread // 2, len(made))).astype(int).tolist())
    told = np.flatnonzero([len(orders[j]) > 0 for j in made])
    if len(told):
        keep |= set(told[np.linspace(0, len(told) - 1, min(spread // 2, len(told))).astype(int)].tolist())
    for col, top in ((0, 20), (1, 10), (2, 10)):
        keep |= set(np.argsort(-stats[:, col], kind="stable")[:top].tolist())
    for col in (3, 4):
        found = np.flatnonzero(stats[:, col])
        keep |= set(rng.choice(found, min(25, len(found)), replace=False).tolist()) if len(found) else set()
    return sorted(int(made[k]) for k in keep)


def corpus(args: argparse.Namespace) -> None:
    import policy_clone_data as cd
    import zkrecord
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = sorted(pathlib.Path(args.data).glob("game-*.zkr"))
    sample = rng.choice(len(paths), min(args.scan, len(paths)), replace=False)
    by_kind: dict[str, list[tuple[int, pathlib.Path, dict]]] = defaultdict(list)
    for k in sample:
        meta = cd._meta(paths[k])
        by_kind[meta.get("kind", "1v1")].append((max(meta.get("until") or [0]), paths[k], meta))
    plates = _special_ids(catalog.SPECIALS.get("plates", {}))
    lifts = _special_ids(catalog.SPECIALS.get("transports", {}))
    summary = Counter()
    for kind in KINDS:
        games = sorted(by_kind[kind], key=lambda g: -g[0])
        # Half the longest of those scanned (late frames at the caps), half drawn from the rest.
        longest, rest = games[:args.games // 2], games[args.games // 2:]
        chosen = longest + [rest[i] for i in rng.choice(len(rest), min(args.games - len(longest), len(rest)), replace=False)]
        for frames_played, path, meta in chosen:
            sides = list(range(len(meta["teams"])))
            for side in rng.choice(sides, min(args.sides, len(sides)), replace=False).tolist():
                name = out / f"{kind}-{path.stem}-s{side}.npz"
                if name.exists():
                    continue
                every = args.every
                _, parts = zkrecord.read(path, side=side, keep=np.arange(0, 10 ** 6, every))
                frames, orders = parts[side]
                made = np.array([j for j, f in enumerate(frames) if f is not None], np.int64)
                if not len(made):
                    continue
                picked = _pick(frames, orders, made, rng, plates, lifts, args.spread)
                info = cd.side_info(meta, side)
                rf.save_frames(name, [frames[j] for j in picked], orders=rf.join([orders[j] for j in picked], orders[0].dtype),
                               order_counts=np.array([len(orders[j]) for j in picked], np.int32), frame_index=np.array(picked, np.int64),
                               map=info["map"], kind=kind, game=path.name, side=side)
                summary[kind] += len(picked)
                print(f"{name.name}: {len(picked)} frames of {len(frames)}", flush=True)
    print(json.dumps(summary))


# The check --------------------------------------------------------------------------------------------------------------------------------------------

_BUNDLES: dict[str, bd.Bundle] = {}


def _bundle(maps: str, slug: str) -> bd.Bundle:
    if slug not in _BUNDLES:
        _BUNDLES.clear()        # one map's at a time
        _BUNDLES[slug] = bd.Bundle.load(pathlib.Path(maps) / f"{slug}.bundle.npz")
    return _BUNDLES[slug]


def _load(path: pathlib.Path) -> tuple[list[rf.Frame], list[np.ndarray], dict]:
    frames = rf.load_frames(path)
    with np.load(path) as d:
        orders = np.split(d["orders"], np.cumsum(d["order_counts"])[:-1]) if len(d["order_counts"]) else []
        meta = {k: str(d[k]) for k in ("map", "kind", "game")}
    return frames, orders, meta


def check_file(job: tuple[str, str, int]) -> dict:
    import policy_sim_env as se
    path, maps, seed = job
    frames, orders, meta = _load(pathlib.Path(path))
    bundle = _bundle(maps, meta["map"])
    table = tables.table()
    rng = np.random.default_rng(seed)
    out = Counter()
    bad: list[str] = []
    stats = Counter()

    def fail(what: str, k: int, why: str) -> None:
        out[f"bad_{what}"] += 1
        if len(bad) < 5:
            bad.append(f"{pathlib.Path(path).name} frame {k} {what}: {why}")

    for k, original in enumerate(frames):
        configs = [(original, dict(enabled_kinds=se.KINDS, economy=True, privileged=True))] + ([(original, {})] if k % 4 == 0 else [])
        if k % 4 == 2:
            configs.append((with_specials(original, rng), dict(enabled_kinds=tuple(range(rf.KIND_COUNT)), economy=True, privileged=True)))
        for n_config, (frame, cfg) in enumerate(configs):
            py, cc = fz.featurize(frame, bundle, table, **cfg), zp.featurize(frame, bundle, table, **cfg)
            out["tensors"] += 1
            for line in tensor_diff(py, cc):
                fail("tensors", k, line)
            stats["most_units"] = max(stats["most_units"], len(py.uids) - int((py.uids < 0).sum()))
            stats["most_controllable"] = max(stats["most_controllable"], py.controllable)
            stats["most_features"] = max(stats["most_features"], len(py.feat_ids))
            stats["most_hidden"] = max(stats["most_hidden"], len(py.hidden_x))
            stats["frames_at_unit_cap"] += int(len(py.uids) - int((py.uids < 0).sum()) == fz.CAP_UNITS) if n_config == 0 else 0
            stats["frames_at_controllable_cap"] += int(py.controllable == fz.CAP_CONTROLLABLE) if n_config == 0 else 0
            stats["frames_at_feature_cap"] += int(len(py.feat_ids) == fz.CAP_FEATURES) if n_config == 0 else 0
            stats["frames_at_hidden_cap"] += int(len(py.hidden_x) == fz.CAP_HIDDEN) if n_config == 0 else 0
            if n_config == 0:
                stats["frames_with_transport_flags"] += int(bool((frame.units["flags"] & (rf.LOADED | rf.CARRIED)).any()))
                for kind in (rf.LOAD, rf.UNLOAD, rf.DROP):
                    stats[f"frames_with_{rf_name(kind)}_legal"] += int(bool(py.kind_mask[:, kind].any()))
                stats["frames_with_plates"] += int(bool(np.isin(py.unit_name, _plate_rows(table)).any()))
                stats["frames_with_own_plates_controllable"] += int(bool(np.isin(py.unit_name[:py.controllable], _plate_rows(table)).any()))
            x, z = act.world_positions(py, bundle)
            xc, zc = zp.world_positions(cc, bundle)
            if not (np.array_equal(_bits(x), _bits(xc)) and np.array_equal(_bits(z), _bits(zc))):
                fail("world_positions", k, "differ")
            out["actions"] += 1
            if n_config == 0:
                # The script's orders as the policy's choices (the cloning labels), and back.
                given = se.policy_orders(orders[k])
                d_py, m_py = act.encode(given, py, bundle, x, z)
                d_cc, m_cc = zp.encode(given, cc, bundle, xc, zc)
                why = decision_diff(d_py, d_cc) or (None if m_py == m_cc else f"missed {m_py} != {m_cc}")
                out["encoded_orders"] += len(given)
                if why:
                    fail("encode_script", k, why)
                why = orders_diff(act.decode(d_py, py, bundle, x, z), zp.decode(d_cc, cc, bundle, xc, zc))
                if why:
                    fail("decode_script", k, why)
            # A random legal decision, decoded, and its orders encoded again.
            d = random_decision(py, rng)
            o_py, o_cc = act.decode(d, py, bundle, x, z), zp.decode(d, cc, bundle, xc, zc)
            out["decoded_orders"] += len(o_py)
            stats.update(f"decoded_kind_{rf_name(o.kind)}" for o in o_py)
            why = orders_diff(o_py, o_cc)
            if why:
                fail("decode_random", k, why)
            back_py, back_cc = act.encode(o_py, py, bundle, x, z), zp.encode(o_py, cc, bundle, xc, zc)
            why = decision_diff(back_py[0], back_cc[0]) or (None if back_py[1] == back_cc[1] else f"missed {back_py[1]} != {back_cc[1]}")
            if why:
                fail("encode_random", k, why)
            out["round_trip_same"] += int(decision_diff(back_py[0], d) is None)
            order = rng.permutation(py.controllable)
            s_py, s_cc = act.to_slots(d, order), zp.to_slots(d, order)
            if any(not np.array_equal(getattr(s_py, f), getattr(s_cc, f)) for f in act.SLOT_FIELDS):
                fail("to_slots", k, "differ")
            why = decision_diff(act.to_decision(s_py, py.controllable), zp.to_decision(s_py, py.controllable))
            if why:
                fail("to_decision", k, why)
        # Group orders: the anchor's side's records, as the sim selects them.
        frame = original
        own = frame.units[frame.units["player"] == int(frame.glob[rf.G["player"]])]
        if len(own):
            rows = table.rows(own["def_id"]).astype(np.int64)
            for _ in range(3):
                anchor = int(own["uid"][rng.integers(len(own))])
                mask, radius = int(rng.integers(1 << 14)), float(rng.choice(gr.GROUP_RADII))
                if not np.array_equal(gr.expand(anchor, mask, radius, own, rows), zp.expand(anchor, mask, radius, own, rows)):
                    fail("expand", k, f"anchor {anchor}")
                out["expands"] += 1
    return {"file": pathlib.Path(path).name, "kind": meta["kind"], "frames": len(frames), "counts": dict(out), "stats": dict(stats), "bad": bad}


def rf_name(kind: int) -> str:
    import policy_clone_data as cd
    names = {rf.THROW: "throw", rf.PUSH: "push", rf.PULL: "pull", rf.PLACE_BEACON: "place_beacon", rf.WAIT_AT_BEACON: "wait_at_beacon",
             rf.REPAIR: "repair"}
    return names.get(int(kind)) or cd.rf_name(int(kind))


_SPECIALS = ("amphtele", "tele_beacon", "jumpsumo", "turretimpulse")


def with_specials(frame: rf.Frame, rng: np.random.Generator) -> rf.Frame:
    """The frame with up to 24 of its own finished units made into what the scripts never build: teleporters (Djinns) and their
    beacons, and switches (Jugglenauts, Newton firezones), half of them pulling; so PLACE_BEACON, WAIT_AT_BEACON, PUSH and PULL
    come up.  Each keeps its place, so a beacon may stand where none could; the featurizer doesn't mind."""
    units = frame.units.copy()
    own = np.flatnonzero((units["player"] == int(frame.glob[rf.G["player"]])) & ((units["flags"] & rf.BEING_BUILT) == 0)
                         & ((units["flags"] & (rf.CARRIED | rf.LOADED)) == 0))
    rng.shuffle(own)
    for n, i in enumerate(own[:24]):
        units["def_id"][i] = bot_units.id_of(_SPECIALS[n % 4])
        if rng.random() < 0.5:
            units["flags"][i] |= rf.PULLING
    return dataclasses.replace(frame, units=units)


def _plate_rows(table: tables.TypeTable) -> np.ndarray:
    return np.array([table.names.index(n) for n in catalog.SPECIALS.get("plates", {}) if n in table.names], np.int64)


def check(args: argparse.Namespace) -> None:
    files = sorted(pathlib.Path(args.corpus).glob("*.npz"))
    if args.files:
        files = files[:args.files]
    jobs = [(str(p), args.maps, k) for k, p in enumerate(files)]
    started = time.time()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    stats: dict[str, Counter] = defaultdict(Counter)
    bad: list[str] = []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(check_file, jobs)):
            by_kind[r["kind"]]["sides"] += 1
            by_kind[r["kind"]]["frames"] += r["frames"]
            by_kind[r["kind"]].update(r["counts"])
            for key, v in r["stats"].items():
                stats[r["kind"]][key] = max(stats[r["kind"]][key], v) if key.startswith("most") else stats[r["kind"]][key] + v
            bad += r["bad"]
            if n % 20 == 0:
                print(f"{n + 1}/{len(jobs)} files, {time.time() - started:.0f} s, bad {sum(v for c in by_kind.values() for k, v in c.items() if k.startswith('bad'))}",
                      flush=True)
    result = {"files": len(files), "seconds": round(time.time() - started, 1), "by_kind": {k: dict(v) for k, v in by_kind.items()},
              "stats": {k: dict(v) for k, v in stats.items()}, "mismatches": bad[:200]}
    total = Counter()
    for v in by_kind.values():
        total.update(v)
    result["total"] = dict(total)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["total"], indent=1))
    print("mismatches:", len(bad), *bad[:20], sep="\n")


# Speed ------------------------------------------------------------------------------------------------------------------------------------------------

def speed(args: argparse.Namespace) -> None:
    import policy_sim_env as se
    files = sorted(pathlib.Path(args.corpus).glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    items = []
    for path in rng.choice(files, min(args.sides, len(files)), replace=False):
        frames, _, meta = _load(path)
        bundle = bd.Bundle.load(pathlib.Path(args.maps) / f"{meta['map']}.bundle.npz")
        items += [(f, bundle) for f in frames]
    table = tables.table()
    cfg = dict(enabled_kinds=se.KINDS, economy=True, privileged=True)
    for f, b in items[:4]:          # numba's compile and both caches
        fz.featurize(f, b, table, **cfg)
        zp.featurize(f, b, table, **cfg)
    sizes = np.zeros(len(items), np.int64)          # unit tokens: what a frame's cost goes with
    numba_s, single_s = np.zeros(len(items)), np.zeros(len(items))
    for k, (f, b) in enumerate(items):
        t0 = time.perf_counter()
        t = fz.featurize(f, b, table, **cfg)
        t1 = time.perf_counter()
        zp.featurize(f, b, table, **cfg)
        numba_s[k], single_s[k] = t1 - t0, time.perf_counter() - t1
        sizes[k] = int((t.uids >= 0).sum())
    out = {"frames": len(items), "tokens_mean": float(sizes.mean()), "per_frame_ms": {}, "batched": {}}
    for lo, hi in ((0, 128), (128, 384), (384, 768), (768, fz.CAP_UNITS + 1), (0, fz.CAP_UNITS + 1)):
        sel = (sizes >= lo) & (sizes < hi)
        if sel.any():
            out["per_frame_ms"][f"{lo}-{hi - 1} unit tokens ({int(sel.sum())} frames)"] = {
                "numba": round(1e3 * numba_s[sel].mean(), 3), "zkpolicy": round(1e3 * single_s[sel].mean(), 3),
                "speedup": round(numba_s[sel].mean() / single_s[sel].mean(), 2)}
    widest = int(zp.bounds(items, table, privileged=True).max())
    kept = [np.zeros(widest, np.uint8) for _ in range(64)]      # a worker's buffers, kept from batch to batch
    for threads in (1, 8, 16):
        for batch, reuse in ((64, False), (64, True), (len(items), False)):
            t0 = time.perf_counter()
            for at in range(0, len(items), batch):
                part = items[at:at + batch]
                zp.featurize_many(part, table, threads=threads, buffers=kept[:len(part)] if reuse else None, **cfg)
            seconds = time.perf_counter() - t0
            name = f"threads {threads}, batches of {batch}" + (", buffers kept" if reuse else "")
            out["batched"][name] = {"ms_per_frame": round(1e3 * seconds / len(items), 3), "frames_per_s": round(len(items) / seconds, 1),
                                    "speedup_on_numba": round(numba_s.sum() / seconds, 2)}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--scan", type=int, default=2000, help="game files whose kinds are read, of which games are chosen")
    c.add_argument("--games", type=int, default=40, help="a kind")
    c.add_argument("--sides", type=int, default=2, help="a game, at most")
    c.add_argument("--every", type=int, default=3, help="frames apart that are made, to choose from")
    c.add_argument("--spread", type=int, default=100, help="frames a side spread over its game")
    c.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("check")
    k.add_argument("--corpus", required=True)
    k.add_argument("--maps", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--workers", type=int, default=12)
    k.add_argument("--files", type=int, default=0)
    s = sub.add_parser("speed")
    s.add_argument("--corpus", required=True)
    s.add_argument("--maps", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sides", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"corpus": corpus, "check": check, "speed": speed}[args.cmd](args)


if __name__ == "__main__":
    main()
