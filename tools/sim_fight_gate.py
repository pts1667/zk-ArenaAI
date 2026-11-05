"""Stage 1 gate (a): does replaying logged fights in the simulator predict who loses more?

  # in WSL, after tools/sim_data.py and the sim build:
  ~/zkml/bin/python tools/sim_fight_gate.py          # full gate, writes results/simgate/
  ~/zkml/bin/python tools/sim_fight_gate.py --limit 2000 --no-fightnet --tag quick   # a quick look

The protocol is docs/learned-policy/01-simulator.md section 5.1, fixed before results:
eight seeds, losses on tracked units only, score log((their+50)/(our+50)), labelled against
their loss > our loss at 20 s (primary) and 10 s.  The comparators are fightnet retrained
out-of-fold by game and the tracked-value ratio.  Differences get game-block bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim/py"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import fightnet as fn  # noqa: E402

OUT = ROOT / "results/simgate"
SECONDS = (10, 20)
TICKS_PER_SECOND = zksim.FRAMES_PER_SECOND // zksim.TICK_FRAMES
UNKNOWN_BLIP = "def0"
LOSS_MARGIN = 50.0            # as train_fightnet.py's ranking score
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261104
UNTRACKED = fn.FLAG_BLIP | fn.FLAG_MEMORY


def load_fights(games: list[str]) -> tuple[list[dict], list[str]]:
    """The games of tools/splits.json, filtered as train_fightnet.load filters."""
    recs, owners = [], []
    for g in games:
        with open(ROOT / g / "fights-team0.jsonl", encoding="utf8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if len(r.get("out") or {}) < len(fn.HORIZONS) or any(e[1] in fn.NOT_UNITS for e in r["ents"]):
                    continue
                recs.append(r)
                owners.append(g)
    return recs, owners


def _map_slug(game_dir: str) -> str:
    text = pathlib.Path(game_dir, "script.txt").read_text("utf8", "replace")
    name = re.search(r"MapName\s*=\s*([^;]+);", text, re.I).group(1).strip()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class Fights:
    """Logged snapshots as flat arrays, with what the protocol scores them on."""

    def __init__(self, recs: list[dict], games: list[str], cat: zksim.Catalog):
        self.recs, self.games = recs, games
        slugs = {g: _map_slug(g) for g in set(games)}
        self.map_names = sorted(set(slugs.values()))
        where = {s: i for i, s in enumerate(self.map_names)}
        offsets, type_, side, x, z, hp, build, tracked = [0], [], [], [], [], [], [], []
        self.unknown: dict[str, int] = {}
        value = np.zeros((len(recs), 2))
        self.no_retreat = np.ones(len(recs), bool)
        self.no_blip_memory = np.ones(len(recs), bool)
        for i, r in enumerate(recs):
            px, pz = r["pos"]
            for s, name, dx, dz, _dh, frac, flags in r["ents"]:
                frac = max(0.0, frac)           # the logs hold some negative health
                if flags & fn.FLAG_RETREAT:
                    self.no_retreat[i] = False
                if flags & UNTRACKED:
                    self.no_blip_memory[i] = False
                t = cat.index.get(name)
                if t is None:
                    if name != UNKNOWN_BLIP:
                        self.unknown[name] = self.unknown.get(name, 0) + 1
                    continue
                type_.append(t)
                side.append(s)
                x.append(px + dx)
                z.append(pz + dz)
                hp.append(frac)
                # Nanoframes grow health with progress; the log has only the health.
                build.append(min(0.99, frac) if flags & fn.FLAG_UNBUILT else 1.0)
                is_tracked = not flags & UNTRACKED
                tracked.append(is_tracked)
                if is_tracked:
                    value[i, s] += cat.units[t]["cost"] * frac
            offsets.append(len(type_))
        self.snapshot_map = np.array([where[slugs[g]] for g in games], np.int32)
        self.offsets = np.array(offsets, np.int32)
        self.type = np.array(type_, np.int32)
        self.side = np.array(side, np.int32)
        self.x, self.z = np.array(x, np.float32), np.array(z, np.float32)
        self.hp, self.build = np.array(hp, np.float32), np.array(build, np.float32)
        self.tracked = np.array(tracked, np.uint8)
        self.value = value
        self.true_loss = np.array([[r["out"][str(h)] for h in SECONDS] for r in recs], np.float64)  # n, horizon, side
        ug = sorted(set(games))
        order = {g: i for i, g in enumerate(ug)}
        self.game_ids = np.array([order[g] for g in games], np.int32)
        self.n_games = len(ug)


def ranking_score(loss: np.ndarray) -> np.ndarray:
    return np.log((loss[..., 1] + LOSS_MARGIN) / (loss[..., 0] + LOSS_MARGIN))


class WeightedAuc:
    """AUC under per-game bootstrap weights, sharing one sort of the scores."""

    def __init__(self, score: np.ndarray, label: np.ndarray, game: np.ndarray):
        unique, self.inverse = np.unique(score, return_inverse=True)
        self.bins = len(unique)
        self.label = label.astype(np.float64)
        self.game = game

    def __call__(self, game_weight: np.ndarray | None = None) -> float:
        w = np.ones_like(self.label) if game_weight is None else game_weight[self.game]
        pos = np.bincount(self.inverse, w * self.label, self.bins)
        neg = np.bincount(self.inverse, w * (1.0 - self.label), self.bins)
        below = np.cumsum(neg) - neg
        total = pos.sum() * neg.sum()
        return float(((below + 0.5 * neg) * pos).sum() / total) if total > 0 else float("nan")


def percentile_ci(values: np.ndarray) -> list[float]:
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def compare(scores: dict[str, np.ndarray], label: np.ndarray, game: np.ndarray, n_games: int,
            pairs: list[tuple[str, str]]) -> dict:
    """AUC per score, and bootstrap CIs of the pairs' differences."""
    aucs = {k: WeightedAuc(v, label, game) for k, v in scores.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {k: np.empty(BOOTSTRAP) for k in scores}
    for b in range(BOOTSTRAP):
        weight = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games).astype(np.float64)
        for k, a in aucs.items():
            samples[k][b] = a(weight)
    result = {"n": int(len(label)), "positive": float(label.mean()), "auc": {}}
    for k, a in aucs.items():
        result["auc"][k] = {"value": a(), "ci95": percentile_ci(samples[k])}
    result["differences"] = {}
    for first, second in pairs:
        if first in samples and second in samples:
            d = samples[first] - samples[second]
            result["differences"][f"{first} - {second}"] = {"value": aucs[first]() - aucs[second](), "ci95": percentile_ci(d)}
    return result


def calibration(pred_frac: np.ndarray, true_frac: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.array([0.0, 0.01, 0.05, 0.15, 0.4, 1.01])[: bins + 1]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pred_frac >= a) & (pred_frac < b)
        if m.any():
            out.append({"predicted": [float(a), float(b)], "n": int(m.sum()), "mean_predicted": float(pred_frac[m].mean()),
                        "mean_measured": float(true_frac[m].mean())})
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def sim_scores(f: Fights, cat: zksim.Catalog, behaviour: int, seeds: int, seed: int, threads: int) -> np.ndarray:
    world_maps = [zksim.Map.load(ROOT / f"sim/data/maps/{n}.npz") for n in f.map_names]
    horizons = [s * TICKS_PER_SECOND for s in SECONDS]
    t0 = time.time()
    loss = zksim.replay(cat, world_maps, f.snapshot_map, f.offsets, f.type, f.side, f.x, f.z, f.hp, f.build,
                        f.tracked, seeds, seed, horizons, behaviour, threads)
    print(f"  replayed {len(f.recs)} snapshots x {seeds} seeds in {time.time() - t0:.0f}s on {threads} threads")
    return loss.mean(axis=1)  # snapshot, horizon, side: the score is on mean losses


def fightnet_oof(recs: list[dict], games: list[str], splits: dict[str, int], epochs: int, batch: int) -> np.ndarray:
    """Out-of-fold fightnet: for each fold, a model that never saw those games, as train_fightnet.py trains it."""
    import torch
    import torch.nn.functional as F
    import train_fightnet as tf
    X, I, S, M, Y, V, _ = tf.tensorize(recs)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda arr: torch.from_numpy(arr).to(dev)  # noqa: E731
    Xt, It, St, Mt, Yt = T(X), T(I), T(S), T(M), T(Y)
    folds = np.array([splits[g] for g in games])
    pred = np.zeros_like(Y)
    for fold in sorted(set(folds)):
        torch.manual_seed(fold)
        np.random.seed(fold)
        tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
        model = tf.Net(d=64, layers=3, n_out=Y.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        steps = epochs * math.ceil(len(tr) / batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        t0 = time.time()
        for _ in range(epochs):
            idx = np.random.permutation(tr)
            for s in range(0, len(idx), batch):
                b = torch.from_numpy(idx[s:s + batch]).to(dev)
                ids = It[b] * (torch.rand_like(It[b], dtype=torch.float32) > 0.3)
                loss = F.binary_cross_entropy_with_logits(model(Xt[b], ids, St[b], Mt[b]), Yt[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        with torch.no_grad():
            for s in range(0, len(va), 2048):
                b = torch.from_numpy(va[s:s + 2048]).to(dev)
                pred[va[s:s + 2048]] = torch.sigmoid(model(Xt[b], It[b], St[b], Mt[b])).cpu().numpy()
        print(f"  fold {fold}: {len(tr)} train, {len(va)} held out, {time.time() - t0:.0f}s")
    hn = len(fn.HORIZONS)
    return np.stack([np.stack([pred[:, j] * V[:, 0], pred[:, hn + j] * V[:, 1]], axis=-1) for j in range(hn)], axis=1)


def report(f: Fights, sim: dict[str, np.ndarray], net: np.ndarray | None) -> dict:
    out: dict = {"snapshots": len(f.recs), "games": f.n_games, "maps": f.map_names, "unknown_names": f.unknown, "subsets": {}}
    value_score = np.log((f.value[:, 1] + LOSS_MARGIN) / (f.value[:, 0] + LOSS_MARGIN))
    subsets = {"all": np.ones(len(f.recs), bool), "no_retreat": f.no_retreat, "no_blip_memory": f.no_blip_memory,
               "no_retreat_no_blip_memory": f.no_retreat & f.no_blip_memory}
    names = list(sim)
    pairs = [(n, c) for n in names for c in ("fightnet", "value_tracked")]
    for subset, mask in subsets.items():
        out["subsets"][subset] = {"share": float(mask.mean())}
        for h, seconds in enumerate(SECONDS):
            label = f.true_loss[:, h, 1] > f.true_loss[:, h, 0]
            scores = {"value_tracked": value_score[mask]}
            for n, loss in sim.items():
                scores[n] = ranking_score(loss[:, h])[mask]
            if net is not None:
                scores["fightnet"] = ranking_score(net[:, h])[mask]
            out["subsets"][subset][f"{seconds}s"] = compare(scores, label[mask], f.game_ids[mask], f.n_games, pairs)
    for n, loss in sim.items():
        detail = {}
        for h, seconds in enumerate(SECONDS):
            true = f.true_loss[:, h]
            decided = true[:, 0] != true[:, 1]
            agree = np.sign(loss[:, h, 1] - loss[:, h, 0])[decided] == np.sign(true[:, 1] - true[:, 0])[decided]
            value = np.maximum(f.value, 1.0)
            detail[f"{seconds}s"] = {
                "sign_agreement": float(agree.mean()), "decided": float(decided.mean()),
                "mean_loss_predicted": loss[:, h].mean(axis=0).tolist(), "mean_loss_measured": true.mean(axis=0).tolist(),
                "spearman_our": spearman(loss[:, h, 0], true[:, 0]), "spearman_their": spearman(loss[:, h, 1], true[:, 1]),
                "calibration_our": calibration(loss[:, h, 0] / value[:, 0], true[:, 0] / value[:, 0]),
                "calibration_their": calibration(loss[:, h, 1] / value[:, 1], true[:, 1] / value[:, 1]),
            }
        out.setdefault("detail", {})[n] = detail
    return out


def verdict(result: dict) -> list[str]:
    lines = []
    for seconds in ("20s", "10s"):
        cell = result["subsets"]["all"][seconds]
        lines.append(f"all snapshots, {seconds}: " + "  ".join(f"{k} {v['value']:.3f}" for k, v in cell["auc"].items()))
        for k, v in cell["differences"].items():
            lines.append(f"  {k}: {v['value']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    primary = result["subsets"]["all"]["20s"]["differences"]
    need = ("advance - fightnet", "advance - value_tracked")
    if all(k in primary for k in need):
        passed = all(primary[k]["ci95"][0] > 0.0 for k in need)
        lines.append(f"GATE 1(a): {'PASS' if passed else 'FAIL'} (primary: advance at 20 s, both CIs above 0)")
    else:
        lines.append("GATE 1(a): not evaluated, the full protocol wasn't run")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--threads", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="only this many snapshots, evenly spread")
    ap.add_argument("--no-fightnet", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="ignore the saved out-of-fold predictions")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--tag", default="gate")
    a = ap.parse_args()
    t0 = time.time()
    splits = json.loads((ROOT / "tools/splits.json").read_text("utf8"))["games"]
    recs, games = load_fights(sorted(splits))
    if a.limit:
        keep = np.linspace(0, len(recs) - 1, a.limit).astype(int)
        recs, games = [recs[i] for i in keep], [games[i] for i in keep]
    print(f"{len(recs)} snapshots from {len(set(games))} games; load {time.time() - t0:.0f}s")
    cat = zksim.Catalog()
    f = Fights(recs, [str(ROOT / g) for g in games], cat)
    if f.unknown:
        print("  names not in the catalog:", f.unknown)
    sim = {}
    for name, behaviour in (("advance", zksim.ADVANCE), ("hold", zksim.HOLD)):
        print(name)
        sim[name] = sim_scores(f, cat, behaviour, a.seeds, a.seed, a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    net = None
    if not a.no_fightnet:
        cache = OUT / "fightnet_oof.npz"
        if cache.exists() and not a.retrain and not a.limit:
            net = np.load(cache)["loss"]
        else:
            print("fightnet, out of fold")
            net = fightnet_oof(recs, games, splits, a.epochs, a.batch)
            if not a.limit:
                np.savez_compressed(cache, loss=net)
    result = report(f, sim, net)
    (OUT / f"{a.tag}.json").write_text(json.dumps(result, indent=1), "utf8")
    np.savez_compressed(OUT / f"{a.tag}-sim.npz", **sim)
    lines = verdict(result)
    print("\n".join(lines))
    for n, d in result["detail"].items():
        c = d["20s"]
        print(f"{n}, 20 s: sign agreement {c['sign_agreement']:.3f}; mean loss predicted {c['mean_loss_predicted']}, "
              f"measured {c['mean_loss_measured']}; spearman {c['spearman_our']:.3f} {c['spearman_their']:.3f}")
    print(f"written to {OUT / (a.tag + '.json')}")


if __name__ == "__main__":
    main()
