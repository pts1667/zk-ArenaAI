"""SPSA tuning of the bot's P() parameters in the real engine (ML stage 4).

  python tools/tune.py NAME --params def.base_share,retreat.regen_hp,... \
      [--iters 20] [--games 8] [--jobs 8] [--resume]

Each iteration draws a random +-1 direction over the chosen parameters
(scaled to each one's lo..hi range), plays --games paired games for
theta + c*delta and theta - c*delta (two prun arms: same enemy factory and
side per pair), scores every game and steps theta along the estimated
gradient.  State, per-iteration scores and the current theta are kept in
results/tune-NAME/state.json; --resume continues.

Score per game (higher is better; wins are rare, so continuous measures
carry the signal):
  log(M15) + 0.5 * log(X15) + 1.0 * won
  M = our/CircuitAI metal produced, X = enemy metal killed / ours lost
  (tools/abstat.py), at 15 min, else 10 min.  (A stage-2 value-model
  objective is planned once that model predicts better than these direct
  measures; see docs/ML.md.)

Standard SPSA gains: a_k = a / (k + 1 + A)^0.602, c_k = c / (k + 1)^0.101.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))

import abstat  # noqa: E402


def registry() -> dict:
    """Import every bot module so all P() calls register."""
    import importlib
    import pkgutil

    import zkbot
    from zkbot import params

    for m in pkgutil.iter_modules(zkbot.__path__):
        if m.name not in ("main",):
            try:
                importlib.import_module(f"zkbot.{m.name}")
            except Exception:
                pass
    return params.REGISTRY


def game_score(r: dict) -> float | None:
    m = abstat.game_metrics(r)
    mm = m.get("M15", m.get("M10"))
    xx = m.get("X15", m.get("X10"))
    if mm is None and xx is None:
        return None
    s = 0.0
    if mm is not None:
        s += math.log(mm)
    if xx is not None:
        s += 0.5 * math.log(xx)
    if r.get("won"):
        s += 1.0
    return s


def fmt(theta: dict) -> str:
    return ",".join(f"{k}={v:.4g}" for k, v in theta.items())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--params", required=True, help="comma list of P() names")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=8, help="paired games per iteration")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--a", type=float, default=0.08, help="step gain (fraction of range per unit score)")
    ap.add_argument("--c", type=float, default=0.12, help="perturbation (fraction of range)")
    ap.add_argument("--flags", default="", help="ZKBOT_FLAGS for every game")
    ap.add_argument("--factories", default="cloak,shield,veh,tank,hover,amph,spider,jump")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    reg = registry()
    names = [n.strip() for n in a.params.split(",") if n.strip()]
    for n in names:
        p = reg.get(n)
        if p is None or p.lo is None or p.hi is None:
            raise SystemExit(f"{n}: unknown parameter or no lo/hi range")
    out = RESULTS / f"tune-{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "state.json"
    if a.resume and sp.exists():
        st = json.loads(sp.read_text())
    else:
        st = {"theta": {n: float(reg[n].default) for n in names}, "k": 0, "history": []}
    rng = random.Random(len(st["history"]) * 7919 + 1)
    A = max(1.0, a.iters * 0.1)
    retries = 0
    while st["k"] < a.iters:
        k = st["k"]
        ak = a.a / (k + 1 + A) ** 0.602
        ck = a.c / (k + 1) ** 0.101
        delta = {n: rng.choice((-1.0, 1.0)) for n in names}

        def point(sign):
            th = {}
            for n in names:
                p = reg[n]
                span = p.hi - p.lo
                th[n] = min(p.hi, max(p.lo, st["theta"][n] + sign * ck * span * delta[n]))
            return th

        plus, minus = point(1), point(-1)
        tag = f"tune-{a.name}/it{k:03d}"
        cmd = [sys.executable, str(ROOT / "tools" / "prun.py"), tag,
               "--games", str(a.games), "--jobs", str(a.jobs), "--minutes", str(a.minutes),
               "--arm", f"plus={a.flags}", "--arm", f"minus={a.flags}",
               "--arm-env", f"plus:ZKBOT_PARAMS={fmt(plus)}",
               "--arm-env", f"minus:ZKBOT_PARAMS={fmt(minus)}",
               "--enemy-factories", a.factories]
        print(f"[it {k}] ak={ak:.4f} ck={ck:.4f}\n  plus  {fmt(plus)}\n  minus {fmt(minus)}", flush=True)
        log = (out / f"it{k:03d}.prun.out").open("w", encoding="utf8")
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        base = RESULTS / f"tune-{a.name}"
        rp = abstat.load_rows(base / f"it{k:03d}-plus")
        rm = abstat.load_rows(base / f"it{k:03d}-minus")
        diffs = []
        for g in sorted(set(rp) & set(rm)):
            sp_, sm_ = game_score(rp[g]), game_score(rm[g])
            if sp_ is not None and sm_ is not None:
                diffs.append(sp_ - sm_)
        if not diffs:
            retries += 1
            if retries > 2:
                raise SystemExit(f"iteration {k}: no scored pairs after 3 tries; see {log.name}")
            print("  no scored pairs; retrying iteration", flush=True)
            continue
        retries = 0
        d = sum(diffs) / len(diffs)
        # Gradient in range-normalised units; step and clamp.
        for n in names:
            p = reg[n]
            span = p.hi - p.lo
            g = d / (2.0 * ck) * delta[n]
            st["theta"][n] = min(p.hi, max(p.lo, st["theta"][n] + ak * g * span))
        wins = sum(1 for r in rp.values() if r.get("won")) + sum(1 for r in rm.values() if r.get("won"))
        st["history"].append({"k": k, "plus": plus, "minus": minus, "mean_diff": d, "pairs": len(diffs),
                              "wins": wins, "theta": dict(st["theta"])})
        st["k"] = k + 1
        sp.write_text(json.dumps(st, indent=1))
        print(f"  pairs={len(diffs)} mean(plus-minus)={d:+.3f} wins={wins}\n  theta {fmt(st['theta'])}", flush=True)
    print("final ZKBOT_PARAMS=" + fmt(st["theta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
