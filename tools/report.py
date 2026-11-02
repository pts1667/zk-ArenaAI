"""Summarise a zk-startup results directory (one or many games).

  python tools/report.py results/h62 [results/h63 ...]

For each game: outcome, length, both teams' metal produced at 5/10/15 min,
kills/deaths, and from the bot's metrics CSV the mex count / builders / army
at the same times.  A final line aggregates win rate and economy ratios.
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from demostats import read_demo_stats, summarize  # noqa: E402

MARKS = (300, 600, 900)


def our_team(script: str) -> int:
    # Demo scripts are lower-cased and put team= before shortname, with a
    # nested [options]{} block, so scan each [aiN] section up to the next
    # top-level header and look for ControllerAI there.
    heads = [m.start() for m in re.finditer(r"\[(ai|team|allyteam|player|modoptions)\d*\]", script, re.I)]
    heads.append(len(script))
    for i, start in enumerate(heads[:-1]):
        seg = script[start:heads[i + 1]]
        if not re.match(r"\[ai\d+\]", seg, re.I):
            continue
        if "controllerai" not in seg.lower():
            continue
        m = re.search(r"team=(\d+);", seg, re.I)
        if m:
            return int(m.group(1))
    return 0


def metrics_at(rows: list[dict], t: float) -> dict:
    cand = [r for r in rows if float(r["t"]) <= t]
    return cand[-1] if cand else {}


def report(dir_: pathlib.Path) -> list[dict]:
    out = []
    csv_path = dir_ / "results.csv"
    if not csv_path.exists():
        print(f"{dir_}: no results.csv")
        return out
    with csv_path.open(encoding="utf-8-sig") as f:
        games = list(csv.DictReader(f))
    for g in games:
        gdir = pathlib.Path(g.get("dir") or "")
        demo = pathlib.Path(g.get("demo") or "")
        won = g.get("outcome") == "win" and "ControllerAI" in (g.get("winner_sides") or "")
        rec = {"dir": gdir.name, "won": won, "outcome": g.get("outcome"), "winner": g.get("winner_sides")}
        line = f"{dir_.name}/{gdir.name}: {'WIN ' if won else 'LOSS'} {g.get('winner_sides','?'):<18}"
        s = None
        if demo.exists():
            try:
                s = read_demo_stats(demo)
            except Exception as exc:
                line += f" (demo unreadable: {exc.__class__.__name__})"
        if s is not None:
            me = our_team(s["script"])
            them = 1 - me
            rec["game_time"] = s["game_time"]
            line += f" t={s['game_time']:4d}s"
            for t in MARKS:
                a = summarize(s, me, t).get("metalProduced", 0.0)
                b = summarize(s, them, t).get("metalProduced", 0.0)
                rec[f"m{t}"] = (a, b)
                line += f" M@{t // 60}m {a / 1000:4.1f}k/{b / 1000:4.1f}k"
            a, b = summarize(s, me), summarize(s, them)
            rec["kd"] = (a.get("unitsKilled", 0), a.get("unitsDied", 0))
            line += f" kill/died {a.get('unitsKilled', 0)}/{a.get('unitsDied', 0)}"
            line += f" dmg {a.get('damageDealt', 0) / 1000:.0f}k/{b.get('damageDealt', 0) / 1000:.0f}k"
        else:
            line += " (no demo)"
        mfiles = sorted(gdir.glob("metrics-team*.csv")) if gdir.exists() else []
        if mfiles:
            with mfiles[0].open() as f:
                rows = list(csv.DictReader(f))
            parts = []
            for t in MARKS:
                r = metrics_at(rows, t)
                if r:
                    parts.append(f"{t // 60}m:mex{r['mex']} b{r['builders_tasked']}/{r['builders']} a{r['army']}")
            if rows:
                last = rows[-1]
                parts.append(f"flees={last['flees']} bdeaths={last['builder_deaths']}")
            line += "\n      " + "  ".join(parts)
        log = next(iter(sorted(gdir.glob("zkbot-team*.log"))), None) if gdir.exists() else None
        if log is not None:
            text = log.read_text(encoding="utf8", errors="replace")
            wd = text.count("WATCHDOG")
            err = text.count("frame error")
            if wd or err:
                line += f"\n      watchdog={wd} frame_errors={err}"
        print(line)
        out.append(rec)
    return out


def main(argv: list[str]) -> int:
    allrecs = []
    for a in argv:
        allrecs += report(pathlib.Path(a))
    if len(allrecs) > 1:
        wins = sum(1 for r in allrecs if r["won"])
        ratios = {}
        for t in MARKS:
            vals = [r[f"m{t}"] for r in allrecs if f"m{t}" in r and r[f"m{t}"][1] > 0]
            if vals:
                ratios[t] = sum(a / b for a, b in vals) / len(vals)
        print(
            f"TOTAL {wins}/{len(allrecs)} wins; mean metal ratio (ours/theirs) "
            + " ".join(f"{t // 60}m={v:.2f}" for t, v in ratios.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
