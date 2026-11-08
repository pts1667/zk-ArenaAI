"""Entry point: python bots/zkbot/main.py <team-id>.  Connects to ControllerAI and drives the policy host (host.py)."""
from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
import traceback

# Running this file directly puts bots/zkbot on sys.path, but the host uses package-relative imports.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from zkbot.client import Client  # noqa: E402
from zkbot.host import Host  # noqa: E402

BOOTSTRAP = {"game_info", "spawn_boxes", "metadata", "map_features", "heightmap", "settings"}
PROFILE_EVERY = 60.0        # wall seconds


def main() -> int:
    try:
        team_id = int(os.environ.get("CONTROLLERAI_TEAM", sys.argv[1] if len(sys.argv) > 1 else "0"))
    except ValueError:
        print("usage: main.py <team-id>", file=sys.stderr)
        return 2
    master = os.environ.get("CONTROLLERAI_MASTER", "http://127.0.0.1:3017")
    out_dir = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / f"zkbot-team{team_id}.log").open("a", encoding="utf8", buffering=1)
    lock = threading.Lock()
    hosts: list[Host] = []

    def log(message: str) -> None:
        # Wall clock and game time, so that events line up with replays at any game speed.
        seconds = max(0, hosts[0].frame if hosts else 0) // 30
        line = f"[{time.strftime('%H:%M:%S')} g{seconds // 60:02d}:{seconds % 60:02d}] {message}"
        with lock:
            log_file.write(line + "\n")
            print(line, flush=True)

    log(f"start team={team_id} master={master}")
    client = Client(master, team_id, log)
    host = Host(client, team_id, log)
    hosts.append(host)
    # Wall time waiting for the engine, and in the host (queries included).
    prof = {"wait": 0.0, "host": 0.0, "frames": 0, "t0": time.time(), "f0": None}
    last_heartbeat = t_wait = time.time()
    try:
        client.connect()
        log("connected")
        while True:
            try:
                msg = client.recv(timeout=1.0)
            except (ConnectionResetError, OSError) as exc:
                log(f"connection closed: {exc}")
                break
            if msg is None:
                if getattr(client, "_closed", False):
                    log("websocket closed")
                    break
                if time.time() - last_heartbeat > 30.0:
                    last_heartbeat = time.time()
                    log("waiting for observation")
                continue
            mtype, data = msg.get("type"), msg.get("data")
            if mtype in BOOTSTRAP:
                host.on_bootstrap(mtype, data)
            elif mtype == "observation":
                frame = (data or {}).get("frame")
                if frame is not None and int(frame) < 0:
                    try:
                        host.on_startup(data)
                    except Exception:
                        log("startup error:\n" + traceback.format_exc())
                        client.send({"type": "finish_frame"})
                    continue
                t_start = time.time()
                prof["wait"] += t_start - t_wait
                try:
                    host.on_frame(data)
                except Exception:
                    log("frame error:\n" + traceback.format_exc())
                try:
                    client.send_commands(host.take_commands())
                finally:
                    client.finish_frame()
                t_wait = time.time()
                prof["host"] += t_wait - t_start
                prof["frames"] += 1
                if prof["f0"] is None and frame is not None:
                    prof["f0"] = int(frame)
                span = t_wait - prof["t0"]
                if span >= PROFILE_EVERY and frame is not None:
                    game = (int(frame) - prof["f0"]) / 30.0
                    log(f"profile wall={span:.0f}s game={game:.0f}s speed={game / span:.2f}x obs={prof['frames']} engine={prof['wait']:.1f}s "
                        f"host={prof['host'] - client.query_time:.1f}s queries={client.query_time:.1f}s units={len(host.units)}")
                    prof.update(wait=0.0, host=0.0, frames=0, t0=t_wait, f0=int(frame))
                    client.query_time = 0.0
            elif mtype in ("error", "command_error", "query_error", "settings_error", "settings_result", "pause"):
                log(f"{mtype}: {data}")
    except KeyboardInterrupt:
        log("interrupted")
    except Exception:
        log("fatal:\n" + traceback.format_exc())
        return 1
    finally:
        try:
            client.close()
        except Exception:
            pass
        log("exit")
        log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
