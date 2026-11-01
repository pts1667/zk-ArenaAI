"""Entry point: python bots/zkbot/main.py <team-id>."""
from __future__ import annotations

import os
import pathlib
import sys
import time
import traceback

# Running this file directly puts bots/zkbot on sys.path, but the brain uses
# package-relative imports.  Add the package parent and import the package.
_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from zkbot.brain import Brain  # noqa: E402
from zkbot.client import Client  # noqa: E402


def main() -> int:
    try:
        team_id = int(os.environ.get("CONTROLLERAI_TEAM", sys.argv[1] if len(sys.argv) > 1 else "0"))
    except ValueError:
        print("usage: main.py <team-id>", file=sys.stderr)
        return 2
    master = os.environ.get("CONTROLLERAI_MASTER", "http://127.0.0.1:3017")
    out_dir = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"zkbot-team{team_id}.log"
    log_file = log_path.open("a", encoding="utf8", buffering=1)

    lock = __import__("threading").Lock()

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with lock:
            log_file.write(line + "\n")
            print(line, flush=True)

    log(f"start team={team_id} master={master}")
    client = Client(master, team_id, log)
    brain = Brain(client, team_id, log)

    # Keep ControllerAI's defaults (sync mode, one observation per frame).  In
    # sync mode the bot sees every frame, which makes the task state simple and
    # avoids losing commands around settings changes.
    bootstrap_types = {
        "game_info",
        "spawn_boxes",
        "metadata",
        "map_features",
        "heightmap",
        "settings",
    }
    last_heartbeat = time.time()
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
            mtype = msg.get("type")
            data = msg.get("data")
            if mtype in bootstrap_types:
                brain.on_bootstrap(mtype, data)
                continue
            if mtype == "observation":
                frame = (data or {}).get("frame")
                if frame is not None and int(frame) < 0:
                    try:
                        brain.on_startup(data)
                    except Exception:
                        log("startup error:\n" + traceback.format_exc())
                        try:
                            client.send({"type": "finish_frame"})
                        except Exception:
                            pass
                    continue
                try:
                    brain.on_frame(data)
                except Exception:
                    log("frame error:\n" + traceback.format_exc())
                try:
                    client.send_commands(brain.take_commands())
                finally:
                    client.finish_frame()
                continue
            if mtype in ("error", "command_error", "query_error", "settings_error"):
                log(f"{mtype}: {data}")
                continue
            if mtype == "settings_result":
                log(f"settings result: {data}")
                continue
            if mtype == "pause":
                log(f"pause state: {data}")
                continue
            # query_result / settings / etc. that weren't consumed synchronously
            if mtype == "query_result":
                brain.debug_queries += 1
                continue
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
        try:
            log_file.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
