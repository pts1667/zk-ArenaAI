"""Run a headless game with the simulator's tools loaded as a mutator of Zero-K.

  python tools/sim_engine.py dump                write the engine's defs to results/simtools/simdump

tools/sim_battery.py runs the scripted scenarios through this module.

The mutator lives in its own write directory, so neither the Zero-K install nor other
running games are touched.  Zero-K only loads gadgets from archives (VFS.ZIP_ONLY in
LuaRules/gadgets.lua), hence the .sdz.
"""
from __future__ import annotations

import argparse
import os
import pathlib
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dump",))
    ap.add_argument("--map", default="TitanDuel 2.2")
    ap.add_argument("--timeout", type=float, default=240.0)
    a = ap.parse_args()
    run(a.mode, a.map, a.timeout)


if __name__ == "__main__":
    main()
