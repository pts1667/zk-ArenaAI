"""Extract per-team statistics from a Recoil/Spring demo (.sdfz).

The engine appends a TeamStatistics history for every team at the end of the
demo (one sample every ``teamStatPeriod`` seconds).  This gives the enemy's
economy (metal produced/used), damage, and units killed/died without any
cooperation from the opponent AI.

Usage:
  python tools/demostats.py <demo.sdfz> [--timeline] [--every 60]
  python tools/demostats.py results/h62            # resolves demo via results.csv

As a module: ``read_demo_stats(path) -> dict``.
"""
from __future__ import annotations

import csv
import gzip
import pathlib
import struct
import sys

_HEADER_FMT = "<16sii256s16sQ" + "i" * 14
_STAT_FIELDS = (
    "frame",
    "metalUsed", "energyUsed",
    "metalProduced", "energyProduced",
    "metalExcess", "energyExcess",
    "metalReceived", "energyReceived",
    "metalSent", "energySent",
    "damageDealt", "damageReceived",
    "unitsProduced", "unitsDied", "unitsReceived", "unitsSent",
    "unitsCaptured", "unitsOutCaptured", "unitsKilled",
)
_STAT_FMT = "<i12f7i"


def read_demo_stats(path: str | pathlib.Path) -> dict:
    raw = gzip.open(str(path), "rb").read()
    fields = struct.unpack_from(_HEADER_FMT, raw, 0)
    magic = fields[0].rstrip(b"\0")
    if not magic.startswith(b"spring demo"):
        raise ValueError(f"not a demo: {magic!r}")
    (
        header_size, script_size, stream_size, game_time, wall_time,
        num_players, player_stat_size, player_stat_elem,
        num_teams, team_stat_size, team_stat_elem, team_stat_period,
        winners_size,
    ) = (fields[2], fields[6], fields[7], fields[8], fields[9], fields[10],
         fields[11], fields[12], fields[13], fields[14], fields[15], fields[16],
         fields[17])
    pos = header_size
    script = raw[pos:pos + script_size].decode("utf8", "replace")
    pos += script_size + stream_size
    winners = list(raw[pos:pos + winners_size])
    pos += winners_size + player_stat_size
    counts = struct.unpack_from(f"<{num_teams}I", raw, pos)
    pos += 4 * num_teams
    teams: list[list[dict]] = []
    elem = team_stat_elem or struct.calcsize(_STAT_FMT)
    for c in counts:
        hist = []
        for _ in range(c):
            vals = struct.unpack_from(_STAT_FMT, raw, pos)
            hist.append(dict(zip(_STAT_FIELDS, vals)))
            pos += elem
        teams.append(hist)
    return {
        "game_time": game_time,
        "wall_time": wall_time,
        "period": team_stat_period,
        "winners": winners,
        "teams": teams,
        "script": script,
    }


def summarize(stats: dict, team: int, at_seconds: float | None = None) -> dict:
    hist = stats["teams"][team]
    if not hist:
        return {}
    row = hist[-1]
    if at_seconds is not None:
        cand = [h for h in hist if h["frame"] / 30.0 <= at_seconds]
        row = cand[-1] if cand else hist[0]
    return row


def resolve_demo(arg: str) -> list[pathlib.Path]:
    p = pathlib.Path(arg)
    if p.suffix == ".sdfz":
        return [p]
    csv_path = p / "results.csv"
    out = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("demo"):
                    out.append(pathlib.Path(r["demo"]))
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    timeline = "--timeline" in argv
    every = 60
    if "--every" in argv:
        every = int(argv[argv.index("--every") + 1])
    for arg in [a for a in argv if not a.startswith("--") and not a.isdigit()]:
        for demo in resolve_demo(arg):
            if not demo.exists():
                print(f"{demo}: missing")
                continue
            s = read_demo_stats(demo)
            print(f"== {demo.name} game={s['game_time']}s winners(ally)={s['winners']}")
            hdr = "  t     " + "  ".join(
                f"T{i}:Mprod  Mused  kill died dmgOut" for i in range(len(s["teams"]))
            )
            if timeline:
                print(hdr)
                end = s["game_time"]
                t = every
                while t <= end + every:
                    cells = []
                    for i in range(len(s["teams"])):
                        r = summarize(s, i, t)
                        cells.append(
                            f"{r.get('metalProduced',0):7.0f} {r.get('metalUsed',0):6.0f} "
                            f"{r.get('unitsKilled',0):4d} {r.get('unitsDied',0):4d} "
                            f"{r.get('damageDealt',0):7.0f}"
                        )
                    print(f"  {t:4d}  " + "  ".join(cells))
                    t += every
            for i in range(len(s["teams"])):
                r = summarize(s, i)
                print(
                    f"  team{i}: metalProduced={r.get('metalProduced',0):.0f} "
                    f"metalUsed={r.get('metalUsed',0):.0f} "
                    f"unitsProduced={r.get('unitsProduced',0)} died={r.get('unitsDied',0)} "
                    f"killed={r.get('unitsKilled',0)} dmgDealt={r.get('damageDealt',0):.0f} "
                    f"dmgRecv={r.get('damageReceived',0):.0f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
