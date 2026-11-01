import json, os, sys, time, pathlib, requests
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
TEAM=int(os.environ.get('CONTROLLERAI_TEAM', sys.argv[1] if len(sys.argv)>1 else '0'))
OUT=pathlib.Path(os.environ.get('ZK_ARENA_OUT','.')); OUT.mkdir(parents=True,exist_ok=True)
MASTER=os.environ.get('CONTROLLERAI_MASTER','http://127.0.0.1:3017')
def log(m):
 print(m,flush=True)
wsurl=None
for _ in range(600):
 try:
  d=requests.get(MASTER+'/list',timeout=2).json(); i=d.get(str(TEAM))
  if i and i.get('reachable',True): wsurl=i['wsUrl']; break
 except Exception: pass
 time.sleep(.2)
if not wsurl: raise SystemExit('no instance')
ws=create_connection(wsurl,timeout=.5); ws.settimeout(.5)
boot={}; pending=set(); started=False; done=False; start=time.time()
KEYS=['mex_count','mex_total','mex_max','wind_min','wind_max','wind_avg','tidal_strength','tidalstrength','map_metal','metal_spots','start_metal','start_energy','startbox_max_n','startbox_recommended_startpos','startbox_n_0','startbox_n_1','max_metal','metal_storage','energy_storage','unit_mex_count_rate','game_frame']
while time.time()-start<120:
 try: raw=ws.recv()
 except WebSocketTimeoutException: raw=None
 except WebSocketConnectionClosedException: break
 if raw:
  try: m=json.loads(raw)
  except Exception: m=None
  if m:
   t=m.get('type'); d=m.get('data')
   if t in ('game_info','spawn_boxes','metadata','map_features','heightmap','settings'):
    boot[t]=d
    if t=='metadata':
     for n,v in boot.items(): (OUT/f'diag-{n}.json').write_text(json.dumps(v,indent=1))
     for rid,q in [
      ('startpos',{'type':'start_position'}),
      ('boxid',{'type':'team_rules_param','key':'start_box_id'}),
      ('params',{'type':'game_rules_params','keys':KEYS}),
      ('tparams',{'type':'team_rules_params','keys':KEYS}),
     ]:
      pending.add(rid); ws.send(json.dumps({'type':'query','requestId':rid,'query':q}))
     log('queries sent')
   elif t=='observation':
    fr=(d or {}).get('frame')
    if fr is not None and fr<0 and not started:
     started=True; gi=boot.get('game_info') or {}
     ws.send(json.dumps({'type':'set_commander','name':'dyntrainer_strike_base'}))
     if gi.get('canChooseStartPos'): ws.send(json.dumps({'type':'finish_frame'}))
     else: ws.send(json.dumps({'type':'finish_frame'}))
    elif fr is not None and fr>=0 and not done:
     done=True
     # give query results a chance
     log(f'frame {fr} units {len((d or {}).get("units",{}))}')
     (OUT/'diag-firstobs.json').write_text(json.dumps(d,indent=1))
     ws.send(json.dumps({'type':'finish_frame'}))
     # wait for remaining queries then exit
     until=time.time()+10
     while pending and time.time()<until:
      try: raw2=ws.recv()
      except WebSocketTimeoutException: continue
      except WebSocketConnectionClosedException: break
      try: m2=json.loads(raw2)
      except Exception: continue
      if m2.get('type')=='query_result':
       rid=m2['data'].get('requestId'); pending.discard(rid)
       (OUT/f'diag-query-{rid}.json').write_text(json.dumps(m2['data'],indent=1)); log(f'query {rid}: {json.dumps(m2["data"]["result"])[:1500]}')
      elif m2.get('type')=='query_error':
       rid=m2['data'].get('requestId'); pending.discard(rid); log(f'qerr {rid}: {m2["data"]}')
     break
   elif t=='query_result':
    rid=d.get('requestId'); pending.discard(rid)
    (OUT/f'diag-query-{rid}.json').write_text(json.dumps(d,indent=1)); log(f'query {rid}: {json.dumps(d["result"])[:1500]}')
   elif t=='query_error':
    rid=d.get('requestId'); pending.discard(rid); log(f'qerr {rid}: {d}')
log('done')
