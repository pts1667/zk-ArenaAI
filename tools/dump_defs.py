"""Connect to a ControllerAI instance and dump full unit definitions + bootstrap map data."""
import json, os, sys, time, pathlib, requests
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException

TEAM = int(os.environ.get("CONTROLLERAI_TEAM", sys.argv[1] if len(sys.argv) > 1 else "0"))
OUT = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
OUT.mkdir(parents=True, exist_ok=True)
MASTER = os.environ.get("CONTROLLERAI_MASTER", "http://127.0.0.1:3017")

def log(m):
    print(m, flush=True)

wsurl = None
for _ in range(600):
    try:
        data = requests.get(MASTER + "/list", timeout=2).json()
        inst = data.get(str(TEAM))
        if inst and inst.get("reachable", True):
            wsurl = inst["wsUrl"]; break
    except Exception:
        pass
    time.sleep(0.2)
if not wsurl:
    raise SystemExit("no instance")
log(f"connect {wsurl}")
ws = create_connection(wsurl, timeout=0.5); ws.settimeout(0.5)
bootstrap = {}
defs = {}
pending = set()
sent_queries = False
startup_sent = False
start = time.time()
last_progress = start
while time.time() - start < 240:
    try:
        raw = ws.recv()
    except WebSocketTimeoutException:
        raw = None
    except WebSocketConnectionClosedException:
        log("closed"); break
    if raw:
        try:
            msg = json.loads(raw)
        except Exception:
            msg = None
        if msg:
            typ = msg.get("type"); data = msg.get("data")
            if typ in ("game_info","spawn_boxes","metadata","map_features","heightmap","settings"):
                bootstrap[typ] = data
                log(f"bootstrap {typ}")
                if typ == "metadata":
                    for name, val in bootstrap.items():
                        (OUT / f"dump-{name}.json").write_text(json.dumps(val))
                    if not sent_queries:
                        sent_queries = True
                        for uid_s in data:
                            rid = f"def-{uid_s}"
                            pending.add(rid)
                            ws.send(json.dumps({"type":"query","requestId":rid,"query":{"type":"unit_def_by_id","unitDefId":int(uid_s)}}))
                        for rid, q in [("heightmap",{"type":"height_map"}),("water",{"type":"water_damage"}),("spots",{"type":"resource_spots_by_name","name":"Metal"})]:
                            pending.add(rid)
                            ws.send(json.dumps({"type":"query","requestId":rid,"query":q}))
                        log(f"sent {len(pending)} queries")
            elif typ == "observation":
                frame = (data or {}).get("frame")
                if frame is not None and frame < 0:
                    if not pending and not startup_sent:
                        startup_sent = True
                        gi = bootstrap.get("game_info") or {}
                        ws.send(json.dumps({"type":"set_commander","name":"dyntrainer_strike_base"}))
                        if gi.get("canChooseStartPos"):
                            box = (bootstrap.get("spawn_boxes") or {}).get(str(data.get("allyTeamId")))
                            if box:
                                ws.send(json.dumps({"type":"set_start_pos","pos":[(box["left"]+box["right"])/2,0,(box["top"]+box["bottom"])/2]}))
                            else:
                                ws.send(json.dumps({"type":"finish_frame"}))
                        else:
                            ws.send(json.dumps({"type":"finish_frame"}))
                    elif startup_sent:
                        ws.send(json.dumps({"type":"finish_frame"}))
                elif frame is not None and frame >= 0:
                    (OUT / f"dump-start-{frame}.json").write_text(json.dumps(data, indent=1))
                    break
            elif typ == "query_result":
                rid = data.get("requestId")
                result = data.get("result")
                if rid in pending:
                    pending.discard(rid)
                    if rid.startswith("def-"):
                        defs[rid[4:]] = result
                    else:
                        (OUT / f"dump-{rid}.json").write_text(json.dumps(result))
                if not pending and sent_queries and not startup_sent:
                    log("all queries done, writing defs")
                    (OUT / "dump-unitdefs.json").write_text(json.dumps(defs))
                    log(f"wrote {len(defs)} defs")
            elif typ == "query_error":
                rid = data.get("requestId")
                log(f"query error {rid}: {data}")
                if rid in pending: pending.discard(rid)
            elif typ == "error":
                log(f"error {data}")
    if pending and time.time() - last_progress > 30:
        log(f"still pending {len(pending)}, sample {list(pending)[:5]}")
        last_progress = time.time()
log(f"done, pending={len(pending)} defs={len(defs)}")
