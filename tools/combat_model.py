"""Fit a combat-outcome model on logged engagements (docs/ARCHITECTURE.md, P4).

  python tools/combat_model.py [results/...]   (default: every results/**/engagements-team*.jsonl)

Target: we "won" a fight when the enemy lost more metal than we did in the
20 s after the snapshot (their_lost > our_lost).  Fights where most enemy
units left our sight (their_unknown > their visible loss + cost/2) are
dropped: their outcome is unknown.

Compares, with cross-validation grouped by game (no leakage between
snapshots of the same game):
  rule       the bot's current style: win iff our power >= 1.0x theirs
  logistic   logistic regression on a few interpretable features
  boosting   gradient boosting on the same features (if enough data)

and prints the logistic coefficients, i.e. what actually decides fights:
the log power ratio, range and speed advantage, turret share, height
advantage, and role mix.  The fitted model is saved as JSON coefficients
(data/combat_model.json) only with --save, for the bot to read later.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES = [
    "log_power_ratio", "log_cost_ratio", "log_dps_ratio", "log_hp_ratio",
    "range_adv", "speed_adv", "our_turret_share", "their_turret_share",
    "height_adv", "our_riot_share", "their_riot_share", "our_raid_share",
    "their_raid_share", "our_skirm_share", "their_skirm_share", "minutes",
]


def featurize(r: dict) -> dict | None:
    o, t = r["our"], r["their"]
    if o["cost"] <= 0 or t["cost"] <= 0:
        return None

    def lr(a, b):
        return math.log((a + 1.0) / (b + 1.0))

    def share(side, role):
        n = max(1, side["n"])
        return side["roles"].get(role, 0) / n

    return {
        "log_power_ratio": lr(o["power"], t["power"]),
        "log_cost_ratio": lr(o["cost"], t["cost"]),
        "log_dps_ratio": lr(o["dps"], t["dps"]),
        "log_hp_ratio": lr(o["hp"], t["hp"]),
        "range_adv": (o["range"] - t["range"]) / 100.0,
        "speed_adv": (o["speed"] - t["speed"]) / 10.0,
        "our_turret_share": o["turret_cost"] / o["cost"],
        "their_turret_share": t["turret_cost"] / t["cost"],
        "height_adv": (o["height"] - t["height"]) / 50.0,
        "our_riot_share": share(o, "riot"),
        "their_riot_share": share(t, "riot"),
        "our_raid_share": share(o, "raid"),
        "their_raid_share": share(t, "raid"),
        "our_skirm_share": share(o, "skirm"),
        "their_skirm_share": share(t, "skirm"),
        "minutes": r["t"] / 60.0,
    }


def load(paths: list[pathlib.Path]):
    X, y, groups, raw = [], [], [], []
    for gi, p in enumerate(paths):
        for line in p.read_text(encoding="utf8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["their_unknown"] > r["their_lost"] + 0.5 * r["their"]["cost"]:
                continue
            if r["our_lost"] + r["their_lost"] < 30:
                continue  # nothing happened
            f = featurize(r)
            if f is None:
                continue
            X.append([f[k] for k in FEATURES])
            y.append(1 if r["their_lost"] > r["our_lost"] else 0)
            groups.append(gi)
            raw.append(r)
    return X, y, groups, raw


def main(argv: list[str]) -> int:
    save = "--save" in argv
    argv = [a for a in argv if a != "--save"]
    roots = [pathlib.Path(a) for a in argv] or [ROOT / "results"]
    paths = sorted({p for r in roots for p in r.rglob("engagements-team*.jsonl")})
    X, y, groups, raw = load(paths)
    n_games = len(set(groups))
    print(f"{len(y)} fights with known outcome from {n_games} games ({len(paths)} files); "
          f"we won {sum(y)} ({sum(y) / max(1, len(y)):.0%})")
    if len(y) < 50 or n_games < 3:
        print("not enough data yet: run more games (the logger is on by default)")
        return 0
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    folds = GroupKFold(n_splits=min(5, n_games))
    res = {"rule": [], "logistic": [], "boosting": []}
    for tr, te in folds.split(X, y, groups):
        rule_p = (X[te, FEATURES.index("log_power_ratio")] >= 0).astype(float)
        res["rule"].append((accuracy_score(y[te], rule_p), None))
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        lr.fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])[:, 1]
        res["logistic"].append((accuracy_score(y[te], p >= 0.5), log_loss(y[te], p, labels=[0, 1])))
        if len(tr) >= 300:
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)
            gb.fit(X[tr], y[tr])
            p = gb.predict_proba(X[te])[:, 1]
            res["boosting"].append((accuracy_score(y[te], p >= 0.5), log_loss(y[te], p, labels=[0, 1])))
    for k, v in res.items():
        if not v:
            continue
        acc = sum(a for a, _ in v) / len(v)
        ll = [b for _, b in v if b is not None]
        print(f"{k:9} accuracy {acc:.1%}" + (f"  log-loss {sum(ll) / len(ll):.3f}" if ll else ""))
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    scaler, model = lr.steps[0][1], lr.steps[1][1]
    coefs = sorted(zip(FEATURES, model.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("logistic coefficients (standardised; + favours us):")
    for name, c in coefs:
        print(f"  {name:20} {c:+.2f}")
    if save:
        out = ROOT / "bots" / "zkbot" / "data" / "combat_model.json"
        out.write_text(json.dumps({
            "features": FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "n": int(len(y)),
            "games": n_games,
        }, indent=1))
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
