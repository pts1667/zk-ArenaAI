"""CircuitAI macro decisions from observer logs (ML stage 3).

  python tools/observe_dataset.py results/o2 [--ours results/d1] [--out results/o2/macro.jsonl]

From observe-team<N>.jsonl ("new" = a unit CircuitAI started, "snap" = its
whole force every 10 s) this builds, per WINDOW seconds:

  state   time, CircuitAI's own value by category, extractor count, the
          enemy value it can see
  target  the metal it committed to each spend category in the window
          (cost of every unit it started: a build decision)

and prints CircuitAI's average spending mix by game phase.  With --ours,
the same mix for our bot (from its state-team logs: change of own value by
category) for a side-by-side comparison.  The examples (--out) are the
training set for a macro head that predicts CircuitAI's spend mix from the
state (behaviour cloning).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot import units  # noqa: E402
from zkbot.statelog import spend_category  # noqa: E402

WINDOW = 30
PHASES = ((0, 180), (180, 420), (420, 720), (720, 1200), (1200, 10**6))


def category(name: str) -> str:
    return spend_category(units.by_name(name))


def phase(t: float) -> str:
    for lo, hi in PHASES:
        if lo <= t < hi:
            return f"{lo // 60}-{hi // 60 if hi < 10**6 else ''}m"
    return "?"


def circuit_examples(path: str):
    recs = [json.loads(x) for x in open(path, encoding="utf8") if x.strip()]
    snaps = [r for r in recs if r["k"] == "snap"]
    news = [r for r in recs if r["k"] == "new"]
    out = []
    for s in snaps:
        t0 = s["t"]
        state = collections.Counter()
        for name, _x, _z, hp, bp in s["ally"]:
            if bp >= 0.99:
                state[category(name)] += units.cost(units.by_name(name))
        n_mex = sum(1 for a in s["ally"] if a[0] == "staticmex" and a[4] >= 0.99)
        seen = collections.Counter()
        for name, *_r in s["enemy"]:
            seen[category(name)] += units.cost(units.by_name(name))
        spend = collections.Counter()
        for n in news:
            if t0 <= n["t"] < t0 + WINDOW:
                spend[category(n["name"])] += units.cost(units.by_name(n["name"]))
        out.append({"t": t0, "own": dict(state), "n_mex": n_mex, "seen": dict(seen), "spend": dict(spend)})
    return out


def our_spend(state_path: str):
    """Our spend per window.  New logs carry "spent" (cumulative metal
    committed per category, dead units included); older ones only the
    standing force, whose increase is a lower bound for units that die."""
    recs = [json.loads(x) for x in open(state_path, encoding="utf8") if x.strip()]
    rows = []
    step = max(1, WINDOW // 10)
    for i in range(0, len(recs) - step, step):
        a, b = recs[i], recs[i + step]
        sp = collections.Counter()
        if "spent" in b:
            for k, v in b["spent"].items():
                sp[k] += max(0.0, v - (a.get("spent") or {}).get(k, 0))
        else:
            for key, pre in (("own_st", ""), ("own_mob", "army_")):
                for k, v in (b.get(key) or {}).items():
                    if k.startswith("n_"):
                        continue
                    kk = k if pre == "" else ("builder" if k == "builder" else pre + k)
                    sp[kk] += max(0.0, v - (a.get(key) or {}).get(k, 0))
        rows.append((a["t"], sp))
    return rows


def mix_table(rows, title):
    agg = collections.defaultdict(collections.Counter)
    for t, sp in rows:
        agg[phase(t)].update(sp)
    print(title)
    cats = sorted({c for c in (k for v in agg.values() for k in v)})
    for ph, _ in sorted(agg.items(), key=lambda kv: int(kv[0].split("-")[0])):
        tot = sum(agg[ph].values()) or 1.0
        top = sorted(((v / tot, c) for c, v in agg[ph].items()), reverse=True)[:9]
        print(f"  {ph:7} " + "  ".join(f"{c} {100 * f:.0f}%" for f, c in top))
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ours", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    ex, rows = [], []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "observe-team*.jsonl"), recursive=True)):
            e = circuit_examples(f)
            for r in e:
                r["game"] = os.path.dirname(f)
                rows.append((r["t"], collections.Counter(r["spend"])))
            ex += e
    print(f"{len(ex)} CircuitAI windows from {len({r['game'] for r in ex})} observer logs")
    mix_table(rows, "CircuitAI spend mix by phase (metal share of units started):")
    if a.ours:
        ours = []
        for d in a.ours:
            for f in sorted(glob.glob(os.path.join(d, "**", "state-team*.jsonl"), recursive=True)):
                ours += our_spend(f)
        mix_table(ours, "our spend mix by phase (metal committed; older logs: net value added):")
    if a.out:
        with open(a.out, "w", encoding="utf8") as f:
            for r in ex:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
