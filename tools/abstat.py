"""Compare arms of a paired A/B batch (or any results dirs).

  python tools/abstat.py results/h120-base results/h120-new [more dirs...]

Per arm: wins, win rate with a 95% Wilson interval and the implied Elo vs
CircuitAI.  For two arms with game numbers in common (tools/prun.py pairs
them: same map, same side), the discordant pairs give an exact two-sided
sign-test p-value: with n games per arm only a large difference shows up,
which is the point of printing it.
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def elo(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return -400 * math.log10(1 / p - 1)


PREFIXES = ("cloak", "shield", "veh", "tank", "hover", "amph", "spider", "jump", "gunship", "plane", "ship")


def enemy_factory_from_log(gdir: pathlib.Path) -> str:
    """CircuitAI's main factory, from the most-seen enemy mobile units."""
    log = next(iter(sorted(gdir.glob("zkbot-team*.log"))), None)
    if log is None:
        return "?"
    seen = [ln for ln in log.read_text(encoding="utf8", errors="replace").splitlines() if "] enemy_seen " in ln]
    if not seen:
        return "?"
    counts: dict[str, float] = {}
    for item in seen[-1].split("enemy_seen ", 1)[1].split():
        name, _, n = item.partition("=")
        for pre in PREFIXES:
            if name.startswith(pre) and not name.endswith("con"):
                counts[pre] = counts.get(pre, 0.0) + float(n or 0)
    return max(counts, key=counts.get) if counts else "?"


def load_rows(d: pathlib.Path) -> dict[int, dict]:
    out = {}
    path = d / "results.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["won"] = r.get("outcome") == "win" and "ControllerAI" in (r.get("winner_sides") or "")
            if not r.get("enemy_factory"):
                r["enemy_factory"] = enemy_factory_from_log(pathlib.Path(r.get("dir") or ""))
            out[int(r["game"])] = r
    return out


def load(d: pathlib.Path) -> dict[int, bool]:
    return {g: r["won"] for g, r in load_rows(d).items()}


MARKS = (600, 900)


def game_metrics(r: dict) -> dict[str, float]:
    """Continuous per-game measures, for batches where almost every game is
    lost and win counts cannot tell arms apart:

      M<min>  our / CircuitAI cumulative metal produced (demo team stats)
      X<min>  metal exchange: enemy value we killed / our value lost
              (the bot's metrics CSV; enemy deaths are attributed from
              EVENT_ENEMY_DESTROYED, so absolute values are approximate)
    """
    out: dict[str, float] = {}
    gdir = pathlib.Path(r.get("dir") or "")
    demo = pathlib.Path(r.get("demo") or "")
    if demo.is_file():
        try:
            from demostats import read_demo_stats, summarize
            from report import our_team
            s = read_demo_stats(demo)
            me = our_team(s["script"])
            for t in MARKS:
                if s["game_time"] >= t:
                    a = summarize(s, me, t).get("metalProduced", 0.0)
                    b = summarize(s, 1 - me, t).get("metalProduced", 0.0)
                    if a > 0 and b > 0:
                        out[f"M{t // 60}"] = a / b
        except Exception:
            pass
    mfiles = sorted(gdir.glob("metrics-team*.csv")) if gdir.is_dir() else []
    if mfiles:
        with mfiles[0].open() as f:
            rows = list(csv.DictReader(f))
        for t in MARKS:
            cand = [x for x in rows if float(x["t"]) <= t]
            if cand and float(rows[-1]["t"]) >= t:
                lost = float(cand[-1]["own_lost_value"] or 0)
                kill = float(cand[-1]["enemy_killed_value"] or 0)
                out[f"X{t // 60}"] = (kill + 50.0) / (lost + 50.0)
    return out


def metric_table(arms: list[tuple[pathlib.Path, dict]]) -> None:
    rows = {d: load_rows(d) for d, _ in arms}
    mets = {d: {g: game_metrics(r) for g, r in rs.items()} for d, rs in rows.items()}
    keys = [f"{k}{t // 60}" for k in ("M", "X") for t in MARKS]

    def gmean(v):
        return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")

    print("geometric means (M = our/their metal, X = metal killed/lost; >1 favours us):")
    for d, _ in arms:
        print(f"  {d.name:<24} " + "  ".join(
            f"{k}={gmean([m[k] for m in mets[d].values() if k in m]):.2f}"
            f"(n={sum(1 for m in mets[d].values() if k in m)})" for k in keys))
    if len(arms) == 2:
        (da, _), (db, _) = arms
        common = sorted(set(mets[da]) & set(mets[db]))
        parts = []
        for k in keys:
            diffs = [math.log(mets[da][g][k] / mets[db][g][k]) for g in common
                     if k in mets[da][g] and k in mets[db][g]]
            if not diffs:
                continue
            better = sum(1 for x in diffs if x > 0)
            p = binom_two_sided(better, len(diffs))
            parts.append(f"{k} x{math.exp(sum(diffs) / len(diffs)):.2f} ({better}/{len(diffs)}, p={p:.2f})")
        print(f"paired {da.name} vs {db.name}: " + "  ".join(parts))


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk + 1e-12))


def main(argv: list[str]) -> int:
    arms = [(pathlib.Path(a), load(pathlib.Path(a))) for a in argv]
    for d, res in arms:
        n, k = len(res), sum(res.values())
        lo, hi = wilson(k, n)
        print(f"{d.name:<24} {k}/{n} = {k / max(1, n):.0%}  95% CI [{lo:.0%}, {hi:.0%}]  "
              f"Elo vs Circuit {elo(k / max(1, n)):+.0f}")
    # Per enemy factory: CircuitAI's random factory choice dominates results.
    table: dict[str, list[str]] = {}
    for d, _res in arms:
        for g, r in load_rows(d).items():
            table.setdefault(r["enemy_factory"], [])
    if table:
        print("by enemy factory: " + "  ".join(
            f"{fac}:" + "/".join(
                f"{sum(1 for r in load_rows(d).values() if r['enemy_factory'] == fac and r['won'])}"
                f"-{sum(1 for r in load_rows(d).values() if r['enemy_factory'] == fac)}"
                for d, _ in arms
            )
            for fac in sorted(table)
        ) + "   (wins-games per arm)")
    if len(arms) == 2:
        (da, a), (db, b) = arms
        common = sorted(set(a) & set(b))
        only_a = sum(1 for g in common if a[g] and not b[g])
        only_b = sum(1 for g in common if b[g] and not a[g])
        n = only_a + only_b
        p = binom_two_sided(only_a, n)
        print(f"paired games={len(common)}: only {da.name} won {only_a}, only {db.name} won {only_b}, "
              f"sign-test p={p:.2f}")
    metric_table(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
