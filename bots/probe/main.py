"""Probe ControllerAI: dump bootstrap data and some observations to the match output folder."""
import json, os, sys, time, pathlib

import requests
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException

MASTER = os.environ.get("CONTROLLERAI_MASTER", "http://127.0.0.1:3017")
TEAM = int(os.environ.get("CONTROLLERAI_TEAM", sys.argv[1] if len(sys.argv) > 1 else "0"))
OUT = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"probe-team{TEAM}.log"

def log(msg):
    with LOG.open("a", encoding="utf8") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)

def find_ws(team):
    for _ in range(300):
        try:
            data = requests.get(MASTER + "/list", timeout=2).json()
        except Exception:
            data = {}
        inst = data.get(str(team))
        if inst and inst.get("reachable", True):
            return inst["wsUrl"]
        time.sleep(0.2)
    raise RuntimeError("no instance")

wsurl = find_ws(TEAM)
log(f"connect {wsurl}")
ws = create_connection(wsurl, timeout=0.5)
ws.settimeout(0.5)
state = {}
frame_count = 0
start = time.time()
last_dump = 0
while time.time() - start < 600:
    try:
        raw = ws.recv()
    except WebSocketTimeoutException:
        continue
    except WebSocketConnectionClosedException:
        log("closed")
        break
    msg = json.loads(raw)
    typ = msg.get("type")
    data = msg.get("data")
    if typ == "observation":
        frame = data.get("frame")
        frame_count += 1
        if frame is not None and frame < 0:
            if not state.get("startup"):
                state["startup"] = True
                gi = state.get("game_info") or {}
                log(f"startup frame={frame} game_info={json.dumps(gi)}")
                ws.send(json.dumps({"type": "set_commander", "name": "dyntrainer_strike_base"}))
                if gi.get("canChooseStartPos"):
                    boxes = state.get("spawn_boxes") or {}
                    box = boxes.get(str(data.get("allyTeamId")))
                    log(f"box={box}")
                    if box:
                        cx = (box["left"] + box["right"]) / 2
                        cz = (box["top"] + box["bottom"]) / 2
                        ws.send(json.dumps({"type": "set_start_pos", "pos": [cx, 0, cz]}))
                    else:
                        ws.send(json.dumps({"type": "finish_frame"}))
                else:
                    ws.send(json.dumps({"type": "finish_frame"}))
            else:
                ws.send(json.dumps({"type": "finish_frame"}))
            continue
        if frame is not None and frame >= 0 and not state.get("started"):
            state["started"] = True
            log(f"GAME START frame={frame}")
            for key in ("game_info", "spawn_boxes", "map_features", "metadata"):
                val = state.get(key)
                if val is not None:
                    path = OUT / f"probe-{key}-team{TEAM}.json"
                    path.write_text(json.dumps(val, indent=1))
                    log(f"wrote {path} ({path.stat().st_size} bytes)")
        if frame is not None and frame >= 0 and frame - last_dump >= 300:
            last_dump = frame
            path = OUT / f"probe-obs-team{TEAM}-{frame}.json"
            path.write_text(json.dumps(data, indent=1))
            log(f"obs frame={frame} units={len(data.get('units',{}))} enemies={len(data.get('enemies',{}))} eco={data.get('economy')}")
        ws.send(json.dumps({"type": "finish_frame"}))
    elif typ in ("game_info", "spawn_boxes", "metadata", "map_features", "heightmap", "settings"):
        state[typ] = data
        log(f"bootstrap {typ}")
    elif typ == "error":
        log(f"ERROR {data}")
        break
log(f"done frames={frame_count}")
