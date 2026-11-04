"""Run zkbot vs CircuitAI headless games in parallel.

  python tools/prun.py <tag> [--diff Hard] [--games 8] [--jobs 8]
                       [--map "TitanDuel 2.2" --map ...] [--minutes 22]
                       [--arm base= --arm new=someflag,other] [--env K=V ...]

Each game is one zk-startup.ps1 invocation with its own ControllerAI port,
engine host port and engine write dir (results/_slots/slotN, reused across
batches so the archive cache is built only once), so several games run at
once without touching each other.  Unlike tools/run.sh it never kills other
spring processes: a human game can run alongside.

Arms (A/B tests): every game index is played once per arm with the same map
and the same side, so the arms are paired.  ZKBOT_FLAGS is set per arm.
Output: results/<tag> with one arm, else results/<tag>-<arm> per arm, each
with game-NNN dirs and a results.csv that tools/report.py reads.  Sides
alternate by game index (odd games: CircuitAI is side 1).
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SLOTS = RESULTS / "_slots"
ZK_DIR = pathlib.Path(os.environ.get("ZK_DIR", r"C:\Users\Thoma\Downloads\zero-k-portable"))
ALL_FACTORIES = ("cloak", "shield", "veh", "tank", "hover", "amph", "spider", "jump", "gunship", "plane", "ship")
FIELDS = ["game", "map", "swapped", "outcome", "winner", "winner_sides", "seconds", "demo", "note", "dir", "arm", "flags", "enemy_factory"]


def slot_dir(slot: int) -> pathlib.Path:
    d = SLOTS / f"slot{slot}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / "springsettings.cfg"
    src = ZK_DIR / "springsettings.cfg"
    if not cfg.exists() and src.exists():
        shutil.copy(src, cfg)
    return d


def free_memory_mb() -> tuple[float, float]:
    """(free physical, free commit) in MB.  A headless engine commits 3.3 GB at
    the start and up to ~4.7 GB in long games, a bot ~0.5 GB; with too little
    commit left the engine
    fails to start ("The paging file is too small")."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**20, m.ullAvailPageFile / 2**20


_START_LOCK = threading.Lock()
_LAST_START = [0.0]


def wait_for_memory(log, need_commit: float = 9000.0) -> None:
    """Block until there is room for one more game.  Starts are spaced 15 s
    apart so the previous engine has committed its memory before the next
    measurement."""
    warned = False
    while True:
        with _START_LOCK:
            phys, commit = free_memory_mb()
            if commit >= need_commit and phys >= 2500 and time.time() - _LAST_START[0] >= 15:
                _LAST_START[0] = time.time()
                return
        if not warned:
            log(f"waiting for memory: free phys {phys:.0f} MB, commit {commit:.0f} MB")
            warned = True
        time.sleep(10)


def run_game(job: dict, slot: int, args, log) -> dict:
    out = job["out"] / f"_g{job['idx']:03d}"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    us = "ControllerAI"
    them = f"CircuitAI:{args.diff}"
    teams = [them, us] if job["swapped"] else [us, them]
    py = os.environ.get("ZKBOT_PY", "python")
    client = f"{py} {job['code']}\\main.py {{team}}"
    if args.observe:
        # CircuitAI vs CircuitAI, each with an allied passive observer that
        # logs its full state (bots/observer): ML stage 3 data.
        teams = [f"CircuitAI:{args.diff}+ControllerAI", f"CircuitAI:{args.diff}+ControllerAI"]
        client = f"{py} {ROOT / 'bots' / 'observer' / 'main.py'} {{team}}"
    cmd = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "zk-startup.ps1"),
        *teams,
        "-Headless", "-NoDownload",
        "-Map", job["map"],
        "-TimeoutMinutes", str(args.minutes),
        "-Games", "1",
        "-OutDir", str(out),
        "-WriteDir", str(slot_dir(slot)),
        "-ControllerPort", str(3100 + 4 * slot),
        "-HostPort", str(8500 + 4 * slot),
        "-Client", client,
    ]
    # The engine pins its main thread to one fixed logical core (0x4000 on
    # this machine) whatever SetCoreAffinity says, so parallel games all
    # shared that core.  zk-startup resets the engine to all cores.
    cmd += ["-EngineAffinity", os.environ.get("ZK_ENGINE_AFFINITY", str((1 << (os.cpu_count() or 16)) - 1))]
    if args.speed:
        # CircuitAI plays worse above ~3x (h120/h121): cap the
        # speed so arms and batches see the same opponent strength.
        cmd += ["-Speed", str(args.speed)]
    if job.get("enemy_factory") and not args.observe:
        # CircuitAI can only start with the one factory left enabled.
        off = "+".join(f"factory{f}" for f in ALL_FACTORIES if f != job["enemy_factory"])
        cmd += ["-CircuitOptions", f"disabledunits={off}"]
    env = dict(os.environ)
    env["ZKBOT_FLAGS"] = job["flags"]
    for kv in args.env + job.get("env", []):
        k, _, v = kv.partition("=")
        env[k] = v
    t0 = time.time()
    with open(job["out"] / f"_g{job['idx']:03d}.out", "w", encoding="utf8") as fout:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fout, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=args.minutes * 60 + 240)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            p.wait()
    row = {"game": job["idx"], "map": job["map"], "swapped": job["swapped"], "outcome": "no result", "dir": ""}
    src_csv = out / "results.csv"
    if src_csv.exists():
        with src_csv.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row.update(rows[0])
    gdir = job["out"] / f"game-{job['idx']:03d}"
    if gdir.exists():
        shutil.rmtree(gdir, ignore_errors=True)
    src_g = out / "game-001"
    if src_g.exists():
        shutil.move(str(src_g), str(gdir))
    demo = pathlib.Path(row.get("demo") or "")
    if row.get("demo") and demo.exists():
        dst = gdir / demo.name
        try:
            shutil.move(str(demo), str(dst))
            row["demo"] = str(dst)
        except OSError:
            pass
    row["dir"] = str(gdir)
    row["game"] = job["idx"]
    row["swapped"] = job["swapped"]
    row["arm"] = job["arm"]
    row["flags"] = job["flags"]
    row["enemy_factory"] = job.get("enemy_factory") or ""
    shutil.rmtree(out, ignore_errors=True)
    won = row.get("outcome") == "win" and "ControllerAI" in (row.get("winner_sides") or "")
    log(f"{job['out'].name}/game-{job['idx']:03d} slot{slot} {'WIN ' if won else 'LOSS'} "
        f"{row.get('outcome')} {job['map']}{' swapped' if job['swapped'] else ''} {time.time() - t0:.0f}s wall")
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--diff", default="Hard")
    ap.add_argument("--games", type=int, default=4, help="games per arm")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--minutes", type=float, default=22)
    ap.add_argument("--arm", action="append", default=[], help="NAME=FLAGS (ZKBOT_FLAGS for this arm)")
    ap.add_argument("--env", action="append", default=[], help="K=V passed to every game")
    ap.add_argument("--first-swapped", action="store_true")
    ap.add_argument("--min-slot", type=int, default=0)
    ap.add_argument("--speed", type=float, default=3, help="cap game speed (default 3: CircuitAI plays as at 1x, h121); 0 = max (CircuitAI weaker)")
    ap.add_argument("--enemy-factories", default="",
                    help="comma list (e.g. cloak,shield,veh,tank,hover,amph,spider,jump) cycled by game "
                         "index: CircuitAI may only start with that factory, so paired games share the matchup")
    ap.add_argument("--observe", action="store_true",
                    help="CircuitAI vs CircuitAI with a passive allied observer per side (ML stage 3 data)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the output dirs and play only the games missing from results.csv")
    ap.add_argument("--arm-env", action="append", default=[], help="NAME:K=V extra env for arm NAME")
    ap.add_argument("--code", action="append", default=[],
                    help="NAME=DIR: bot package dir for arm NAME (default: snapshot of bots/zkbot)")
    args = ap.parse_args(argv)
    maps = args.map or ["TitanDuel 2.2"]
    arms = []
    for a in args.arm or [f"={os.environ.get('ZKBOT_FLAGS', '')}"]:
        name, _, flags = a.partition("=")
        arms.append((name, flags))
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    jobs: queue.Queue = queue.Queue()
    outs = {}
    rows: dict[str, list] = {name: [] for name, _ in arms}
    for name, flags in arms:
        out = RESULTS / (args.tag if len(arms) == 1 else f"{args.tag}-{name}")
        if out.exists() and args.resume:
            if (out / "results.csv").exists():
                with (out / "results.csv").open(encoding="utf8") as f:
                    rows[name] = [r for r in csv.DictReader(f) if r.get("outcome") != "no result"]
        elif out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        outs[name] = out
    # Snapshot the bot so editing bots/zkbot during a batch is safe.  An arm
    # can also run another code version (e.g. a git worktree) via --code.
    snap = RESULTS / f"_code-{args.tag}" / "zkbot"
    if not (args.resume and snap.exists()):  # a resumed batch keeps its code
        if snap.parent.exists():
            shutil.rmtree(snap.parent)
        shutil.copytree(ROOT / "bots" / "zkbot", snap, ignore=shutil.ignore_patterns("__pycache__"))
    code = {name: str(snap) for name, _ in arms}
    for c in args.code:
        name, _, d = c.partition("=")
        code[name] = str(pathlib.Path(d).resolve())
    # Interleave arms so a stopped batch still has paired games.
    efacs = [f.strip() for f in args.enemy_factories.split(",") if f.strip()]
    for i in range(args.games):
        swapped = (i % 2 == 1) != args.first_swapped
        for name, flags in arms:
            if any(int(r["game"]) == i + 1 for r in rows[name]):
                continue  # --resume: already played
            jobs.put({"idx": i + 1, "arm": name, "flags": flags, "map": maps[i % len(maps)],
                      "swapped": swapped, "out": outs[name], "code": code[name],
                      "env": [e.split(":", 1)[1] for e in args.arm_env if e.split(":", 1)[0] == name],
                      "enemy_factory": efacs[(i // 2) % len(efacs)] if efacs else ""})
    total = jobs.qsize()
    log(f"{total} games, {min(args.jobs, total)} parallel, arms={arms} maps={maps} diff={args.diff}")

    def write_results() -> None:
        # Rewritten after every game, so a killed batch keeps its results
        # and can be continued with --resume.
        for name, out in outs.items():
            with (out / "results.csv").open("w", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in sorted(rows[name], key=lambda r: int(r["game"])):
                    w.writerow(r)

    def acquire_slot():
        """A machine-wide free slot (ports + write dir), held by a file lock
        so concurrent prun batches never share one."""
        import msvcrt
        for slot in range(args.min_slot, 64):
            path = SLOTS / f"slot{slot}.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(path, "a+")
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return slot, fh
            except OSError:
                fh.close()
        raise RuntimeError("no free slot")

    def worker(_n: int) -> None:
        with lock:
            slot, fh = acquire_slot()
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                for attempt in range(3):
                    wait_for_memory(log)
                    t0 = time.time()
                    row = run_game(job, slot, args, log)
                    # An engine that died while loading is retried.
                    if row.get("outcome") != "no result" or time.time() - t0 > 120:
                        break
                    log(f"retrying {job['arm']} game {job['idx']}")
                    time.sleep(20)
            except Exception as exc:  # keep the other games going
                log(f"game {job['idx']} {job['arm']} failed: {exc!r}")
                continue
            with lock:
                rows[job["arm"]].append(row)
                write_results()
            # Stagger starts: engine loading is disk heavy.
            time.sleep(1)

    threads = []
    for s in range(min(args.jobs, total)):
        t = threading.Thread(target=worker, args=(s,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(3)
    for t in threads:
        t.join()
    write_results()
    for out in outs.values():
        subprocess.run([sys.executable, str(ROOT / "tools" / "report.py"), str(out)], cwd=ROOT)
    if len(arms) > 1:
        subprocess.run([sys.executable, str(ROOT / "tools" / "abstat.py"), *map(str, outs.values())], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
