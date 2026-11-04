"""Passive observer (ML stage 3): python bots/observer/main.py <team-id>.

Runs as a ControllerAI team allied with a CircuitAI team (zk-startup side
"CircuitAI:Hard+ControllerAI").  ControllerAI's "units" are the *friendly*
units, allies included, so the observer sees CircuitAI's whole army, base
and build decisions, plus everything in the shared line of sight.  It
issues no orders (its own commander idles at its start) and writes

  observe-team<N>.jsonl
    {"k":"snap","t":s,"ally":[[name,x,z,hp,bp],...],"enemy":[[name,x,z,hp],...]}   every 10 s
    {"k":"new","t":s,"name":...,"x":..,"z":..}   each new ally unit (a build decision)
    {"k":"gone","t":s,"name":...}                each ally unit that disappeared

With one observer per side of a CircuitAI-vs-CircuitAI game the two files
together are the full game state.  tools/observe_dataset.py turns them into
training examples for the macro heads.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from zkbot.client import Client  # noqa: E402

SNAP_EVERY = 300


def main() -> int:
    team = int(os.environ.get("CONTROLLERAI_TEAM", sys.argv[1] if len(sys.argv) > 1 else "0"))
    master = os.environ.get("CONTROLLERAI_MASTER", "http://127.0.0.1:3017")
    out = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / f"observer-team{team}.log").open("a", encoding="utf8", buffering=1)
    data_f = (out / f"observe-team{team}.jsonl").open("a", encoding="utf8", buffering=1)

    def log(m: str) -> None:
        logf.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

    def emit(rec: dict) -> None:
        data_f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    client = Client(master, team, log)
    client.connect()
    log("connected")
    commander_sent = settings_sent = False
    known: dict[int, str] = {}
    own: set[int] = set()
    checked: set[int] = set()
    last_snap = -10**9
    while True:
        try:
            msg = client.recv(timeout=1.0)
        except (ConnectionResetError, OSError) as exc:
            log(f"closed: {exc}")
            break
        if msg is None:
            if getattr(client, "_closed", False):
                break
            continue
        if msg.get("type") != "observation":
            continue
        obs = msg.get("data") or {}
        frame = int(obs.get("frame") or 0)
        if frame < 0:
            if not commander_sent:
                client.send({"type": "set_commander", "name": "dyntrainer_support_base"})
                commander_sent = True
            client.send({"type": "finish_frame"})
            continue
        if not settings_sent and frame >= 2:
            settings_sent = True
            try:
                client.set_settings({"block_n_frames": 30})
            except Exception:
                pass
        t = round(frame / 30.0, 1)
        ally = {}
        for uid_s, u in (obs.get("units") or {}).items():
            uid = int(uid_s)
            if uid in own:
                continue
            if uid not in known and uid not in checked and "teamId" not in u:
                # Observations omit the team: ask once for commanders (the
                # only unit the observer itself ever has).
                checked.add(uid)
                name = str(u.get("name") or "")
                if name.startswith("dyn") or "comm" in name:
                    try:
                        full = client.query({"type": "unit_by_id", "unitId": uid}, timeout=5.0) or {}
                        if int(full.get("teamId", -1)) == team:
                            own.add(uid)
                            log(f"own unit {uid} {name}")
                            continue
                    except Exception as exc:
                        log(f"unit_by_id {uid}: {exc}")
            elif int(u.get("teamId", -1)) == team:
                own.add(uid)
                continue
            ally[uid] = u
        for uid, u in ally.items():
            if uid not in known:
                known[uid] = u.get("name", "?")
                p = u.get("pos") or [0, 0, 0]
                emit({"k": "new", "t": t, "id": uid, "name": known[uid], "x": round(p[0]), "z": round(p[2]),
                      "bp": round(float(u.get("buildProgress") or 0.0), 2)})
        for uid in [i for i in known if i not in ally]:
            emit({"k": "gone", "t": t, "id": uid, "name": known.pop(uid)})
        if frame - last_snap >= SNAP_EVERY:
            last_snap = frame

            def row(u, bp=True):
                p = u.get("pos") or [0, 0, 0]
                hp = float(u.get("health") or 0.0) / max(1.0, float(u.get("maxHealth") or 1.0))
                r = [u.get("name", "?"), round(p[0]), round(p[2]), round(hp, 2)]
                if bp:
                    r.append(round(float(u.get("buildProgress") or 0.0), 2))
                return r

            emit({"k": "snap", "t": t,
                  "ally": [row(u) for u in ally.values()],
                  "enemy": [row(u, False) for u in (obs.get("enemies") or {}).values()],
                  "blips": len(obs.get("radarBlips") or {})})
        client.send({"type": "finish_frame"})
    data_f.close()
    log("exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
