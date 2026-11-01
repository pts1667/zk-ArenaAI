import json, os, sys, time, pathlib, requests
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
TEAM=int(os.environ.get('CONTROLLERAI_TEAM', sys.argv[1] if len(sys.argv)>1 else '0'))
OUT=pathlib.Path(os.environ.get('ZK_ARENA_OUT','.')); OUT.mkdir(parents=True, exist_ok=True)
LOG=OUT/f'query0-{TEAM}.log'
def log(m):
    line=f'{time.time():.3f} {m}'
    with LOG.open('a',encoding='utf8') as f: f.write(line+'\n')
    print(line,flush=True)
MASTER=os.environ.get('CONTROLLERAI_MASTER','http://127.0.0.1:3017')
wsurl=None
for _ in range(600):
    try:
        data=requests.get(MASTER+'/list',timeout=2).json(); inst=data.get(str(TEAM))
        if inst and inst.get('reachable',True): wsurl=inst['wsUrl']; break
    except Exception: pass
    time.sleep(0.2)
if not wsurl: raise SystemExit('no instance')
log('connect '+wsurl)
ws=create_connection(wsurl,timeout=0.25); ws.settimeout(0.25)
inbox=[]; seq=0
def recv_ws(t=0.25):
    try: raw=ws.recv()
    except WebSocketTimeoutException: return None
    except WebSocketConnectionClosedException: return 'closed'
    try: return json.loads(raw)
    except Exception: return None
def recv(t=0.25):
    if inbox: return inbox.pop(0)
    return recv_ws(t)
def query(q,timeout=8.0):
    global seq
    seq+=1; rid=f'q{seq}'
    ws.send(json.dumps({'type':'query','requestId':rid,'query':q}))
    log('sent query '+rid+' '+q['type'])
    deadline=time.time()+timeout; deferred=[]
    try:
        while time.time()<deadline:
            m=recv(min(0.25,max(0.01,deadline-time.time())))
            if m is None: continue
            if m=='closed': log('closed'); return None
            t=m.get('type'); d=m.get('data') or {}
            if t=='query_result' and d.get('requestId')==rid:
                log('query result '+rid+' in %.2fs: %s'%(timeout-(deadline-time.time()),str(d.get('result'))[:300])); return d.get('result')
            if t=='query_error' and d.get('requestId')==rid:
                log('query error '+rid+': '+str(d)); return None
            deferred.append(m)
        log('QUERY TIMEOUT '+rid)
        return None
    finally:
        inbox.extend(deferred)
start=time.time(); sent=False; queried=False; obs_count=0
while time.time()-start<40:
    m=recv()
    if m is None: continue
    if m=='closed': break
    t=m.get('type'); d=m.get('data') or {}
    if t=='observation':
        obs_count+=1; fr=d.get('frame')
        if fr is not None and fr<0:
            if not sent:
                sent=True
                ws.send(json.dumps({'type':'set_commander','name':'dyntrainer_support_base'}))
                log('sent commander')
            ws.send(json.dumps({'type':'finish_frame'}))
        elif fr is not None and fr>=0:
            if not queried:
                queried=True
                log('frame 0 obs; units='+str(len(d.get('units') or {})))
                cmd=next(iter((d.get('units') or {}).values()),None)
                if cmd:
                    x,z=cmd['pos'][0],cmd['pos'][2]
                    query({'type':'closest_build_site','unitDefId':409,'searchRadius':360,'minDist':90,'pos':[x,0,z],'facing':0})
                    query({'type':'can_build_at','unitDefId':409,'pos':[x+120,0,z],'facing':0})
                    log('after queries, now finish frame 0')
            ws.send(json.dumps({'type':'finish_frame'}))
            if queried and fr>60: break
    elif t in ('game_info','spawn_boxes','metadata','map_features','heightmap','settings'):
        log('bootstrap '+t)
    elif t in ('error','query_error','command_error'):
        log('ERR '+t+' '+str(d))
log('done obs_count='+str(obs_count))
