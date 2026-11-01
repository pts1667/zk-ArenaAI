import json, os, sys, time, pathlib, requests
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException

TEAM=int(os.environ.get('CONTROLLERAI_TEAM', sys.argv[1] if len(sys.argv)>1 else '0'))
OUT=pathlib.Path(os.environ.get('ZK_ARENA_OUT','.')); OUT.mkdir(parents=True, exist_ok=True)
LOG=OUT/f'testplop-{TEAM}.log'
def log(m):
    with LOG.open('a',encoding='utf8') as f: f.write(str(m)+'\n')
    print(m,flush=True)
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
ws=create_connection(wsurl,timeout=0.5); ws.settimeout(0.5)
bootstrap={}; pending={}; cmd=None; start=time.time(); sent_plop=False; lastlog=-1
def recv_msg():
    while True:
        try: raw=ws.recv()
        except WebSocketTimeoutException: return None
        except WebSocketConnectionClosedException: return 'closed'
        try: return json.loads(raw)
        except Exception: continue

def query(q,rid='q'):
    ws.send(json.dumps({'type':'query','requestId':rid,'query':q}))
    while True:
        m=recv_msg()
        if m is None: continue
        if m=='closed': return None
        t=m.get('type'); d=m.get('data')
        if t=='query_result' and d.get('requestId')==rid: return d.get('result')
        if t=='query_error' and d.get('requestId')==rid:
            log('query error '+str(d)); return None
        if t=='observation': obs=d
        if t=='error': log('err '+str(d))

obs=None
while time.time()-start<180:
    m=recv_msg()
    if m is None: continue
    if m=='closed': break
    t=m.get('type'); d=m.get('data')
    if t in ('game_info','spawn_boxes','metadata','map_features','heightmap','settings'):
        bootstrap[t]=d
    elif t=='observation':
        obs=d; fr=obs.get('frame')
        if fr is not None and fr<0:
            gi=bootstrap.get('game_info') or {}
            if not getattr(sys,'_started',False):
                ws.send(json.dumps({'type':'set_commander','name':'dyntrainer_strike_base'}))
                if gi.get('canChooseStartPos'):
                    box=(bootstrap.get('spawn_boxes') or {}).get(str(obs.get('allyTeamId')))
                    if box:
                        ws.send(json.dumps({'type':'set_start_pos','pos':[(box['left']+box['right'])/2,0,(box['top']+box['bottom'])/2]}))
                    else: ws.send(json.dumps({'type':'finish_frame'}))
                else: ws.send(json.dumps({'type':'finish_frame'}))
                sys._started=True
            else: ws.send(json.dumps({'type':'finish_frame'}))
        elif fr is not None and fr>=0:
            units=obs.get('units',{})
            if not sent_plop:
                sent_plop=True
                cmd=next(iter(units.values()),None)
                if cmd:
                    x,z=cmd['pos'][0],cmd['pos'][2]
                    r=query({'type':'closest_build_site','unitDefId':409,'searchRadius':400,'pos':[x,0,z]}, 'site')
                    log('closest site '+str(r))
                    if r and r.get('possible',True):
                        pos=r.get('pos') or [x+150,0,z]
                        ws.send(json.dumps({'type':'build','unitId':int(next(iter(units))),'defId':409,'pos':pos}))
                        log('sent factory build at '+str(pos)+' resources '+str(obs.get('economy')))
                    # also try mex
                # query commander rules params facplop
                if units:
                    uid=int(next(iter(units)))
                    r=query({'type':'unit_rules_params','unitId':uid,'keys':['facplop']}, 'rules')
                    log('comm rules '+str(r))
            if int(fr)//15 != lastlog:
                lastlog=int(fr)//15
                eco=obs.get('economy')
                us=[(u.get('name'),round(u['buildProgress'],3),round(u['health'])) for u in units.values()]
                log(f"f={fr} M={eco['Metal']['current']:.0f}/{eco['Metal']['income']:.1f} E={eco['Energy']['current']:.0f}/{eco['Energy']['income']:.1f} units={us[:12]} n={len(us)}")
            ws.send(json.dumps({'type':'finish_frame'}))
log('done')
