"""Run a headless game with the simulator's tools loaded as a mutator of Zero-K.

  python tools/sim_engine.py dump                write the engine's defs to results/simtools/simdump
  python tools/sim_engine.py facts MAP [MAP...]  ask each map for its start boxes and metal spots, into sim/data/map_facts.json
  python tools/sim_engine.py overdrive           log what the game's overdrive makes of some layouts (SIMOD lines)
  python tools/sim_engine.py mechanics           log what wrecks, reclaim, upkeep, cloak, stockpiling, a shot solar and shields come to (SIMMECH lines)
  python tools/sim_engine.py pierce              only what shots that go through what they hit come to

These come from a game on the map rather than from its files, since Zero-K decides between its own configs, the map's and a
fallback, and a map's file is Lua that may depend on the number of teams or build its list in code.

tools/sim_battery.py runs the scripted scenarios through this module.

The mutator lives in its own write directory, so neither the Zero-K install nor other
running games are touched.  Zero-K only loads gadgets from archives (VFS.ZIP_ONLY in
LuaRules/gadgets.lua), hence the .sdz.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "simgame"
ZK_DIR = pathlib.Path(os.environ.get("ZK_DIR", pathlib.Path.home() / "Downloads" / "zero-k-portable"))
ENGINE = ZK_DIR / "engine" / "win64" / "2026.07.04" / "spring-headless.exe"
BASE_GAME = "Zero-K v1.14.10.3"
TOOLS_VERSION = "1"
GAME = f"ZK Sim Tools {TOOLS_VERSION}"
WRITE_DIR = ROOT / "results" / "simtools"
FACTS_FILE = ROOT / "sim" / "data" / "map_facts.json"

MODINFO = f"""return {{
  name = "ZK Sim Tools", shortName = "zksim", version = "{TOOLS_VERSION}",
  game = "ZK Sim Tools", shortGame = "zksim", description = "Simulator data and scenarios",
  modtype = 1, depend = {{ "{BASE_GAME}" }},
}}
"""


def build_archive(write_dir: pathlib.Path, extra_files: dict[str, str] | None = None) -> None:
    games = write_dir / "games"
    games.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(games / "zksimtools.sdz", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("modinfo.lua", MODINFO)
        for f in sorted(SRC.rglob("*.lua")):
            z.write(f, f.relative_to(SRC).as_posix())
        for name, text in (extra_files or {}).items():
            z.writestr(name, text)


def start_script(mode: str, map_name: str, port: int) -> str:
    teams = []
    for t in (0, 1):
        teams.append(f"""  [AI{t}]
  {{
    Name=Null{t};
    ShortName=NullAI;
    Team={t};
    Host=0;
  }}
  [TEAM{t}]
  {{
    TeamLeader=0;
    AllyTeam={t};
  }}
  [ALLYTEAM{t}]
  {{
    NumAllies=0;
  }}""")
    body = "\n".join(teams)
    return f"""[GAME]
{{
  GameType={GAME};
  MapName={map_name};
  StartPosType=0;
  IsHost=1;
  HostIP=127.0.0.1;
  HostPort={port};
  MyPlayerName=Spectator;
  [PLAYER0]
  {{
    Name=Spectator;
    Spectator=1;
    Team=0;
  }}
{body}
  [MODOPTIONS]
  {{
    sim_mode={mode};
  }}
}}
"""


def run(mode: str, map_name: str, timeout: float, port: int = 8790,
        extra_files: dict[str, str] | None = None) -> pathlib.Path:
    wd = WRITE_DIR
    wd.mkdir(parents=True, exist_ok=True)
    build_archive(wd, extra_files)
    src = ZK_DIR / "springsettings.cfg"
    if src.exists() and not (wd / "springsettings.cfg").exists():
        shutil.copy(src, wd / "springsettings.cfg")
    script = wd / f"script-{mode}.txt"
    script.write_text(start_script(mode, map_name, port), encoding="utf8")
    env = dict(os.environ, SPRING_DATADIR=str(ZK_DIR))
    log = wd / f"engine-{mode}.log"
    t0 = time.time()
    with log.open("w", encoding="utf8") as out:
        p = subprocess.Popen([str(ENGINE), "--write-dir", str(wd), str(script)], stdout=out,
                             stderr=subprocess.STDOUT, env=env, cwd=str(wd))
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            code = "timeout"
    print(f"{mode}: exit {code} after {time.time() - t0:.0f} s; log {log}")
    return wd


def _slug(name: str) -> str:
    """As tools/sim_data.py names a map's files."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _lists(v):
    """The dump writes every Lua table as an object, so an array comes out keyed "1", "2"..."""
    if isinstance(v, dict):
        if not v:
            return []
        if all(k.isdigit() for k in v):
            return [_lists(v[k]) for k in sorted(v, key=int)]
        return {k: _lists(x) for k, x in v.items()}
    return v


def facts(maps: list[str], timeout: float, launches: int, out: pathlib.Path = FACTS_FILE) -> None:
    """A map may draw its start boxes at random when it loads, so each is launched a few times and every layout seen is kept.  A map that won't load is said and
    passed over, and the table is saved after each map, so that a long list can be stopped and started again: other than the default table, what it has is skipped."""
    table = json.loads(out.read_text(encoding="utf8")) if out.exists() else {}
    made = WRITE_DIR / "simdump" / "map_facts.json"
    for name in maps:
        if out != FACTS_FILE and _slug(name) in table:
            continue
        layouts = []
        got = None
        for _ in range(launches):
            made.unlink(missing_ok=True)
            run("facts", name, timeout)
            if not made.exists():
                continue
            got = _lists(json.loads(made.read_text(encoding="utf8", errors="replace")))
            boxes = got.pop("boxes")
            if boxes not in layouts:
                layouts.append(boxes)
        if got is None:
            print(f"{name}: wrote no facts", flush=True)
            continue
        got["layouts"] = layouts
        table[_slug(name)] = got
        print(f"{name}: {len(layouts)} layouts of {[len(b) for b in layouts]} boxes, {len(got['spots'])} spots, {len(got['features'])} reclaimable features "
              f"of {sum(f[2] for f in got['features']):.0f} metal, on {got['width']:.0f} x {got['height']:.0f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dump", "facts", "overdrive", "mechanics", "pierce", "pierce_slow", "abilities"))
    ap.add_argument("maps", nargs="*")
    ap.add_argument("--map", default="TitanDuel 2.2")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--launches", type=int, default=6, help="facts: how many times to load each map, looking for start boxes it draws at random")
    ap.add_argument("--out", type=pathlib.Path, default=FACTS_FILE, help="facts: the table to add to")
    a = ap.parse_args()
    if a.mode == "facts":
        facts(a.maps, a.timeout, a.launches, a.out)
    else:
        wd = run(a.mode, a.map, a.timeout)
        marker = {"overdrive": "SIMOD", "mechanics": "SIMMECH", "pierce": "SIMMECH", "pierce_slow": "SIMMECH", "abilities": "SIMMECH"}.get(a.mode)
        if marker:
            for line in (wd / f"engine-{a.mode}.log").read_text(encoding="utf8", errors="replace").splitlines():
                if marker in line:
                    print(line[line.index(marker):])


if __name__ == "__main__":
    main()
