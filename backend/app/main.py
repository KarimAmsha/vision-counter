import asyncio,base64,csv,io,json,os,time,uuid,html
from datetime import datetime,timezone
from pathlib import Path
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,HTTPException,Header,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from sqlalchemy import create_engine,Column,String,Integer,Text,select
from sqlalchemy.orm import DeclarativeBase,Session
from openpyxl import Workbook
from .engine import Detector,Tracker,Rules,draw,COCO
ROOT=Path(__file__).resolve().parents[2]; DATA=Path(os.getenv('DATA_DIR',str(ROOT/'data')));DATA.mkdir(parents=True,exist_ok=True)
DB=create_engine('sqlite:///'+str(DATA/'vision.sqlite'),connect_args={'check_same_thread':False})
class Base(DeclarativeBase):pass
class Event(Base):
    __tablename__='events';id=Column(String,primary_key=True);session_id=Column(String);created_at=Column(String);line_id=Column(String);line_name=Column(String);direction=Column(String);category=Column(String);track_id=Column(Integer);corrected_direction=Column(String,nullable=True);correction_reason=Column(String,nullable=True)
class Recording(Base):
    __tablename__='recordings';id=Column(String,primary_key=True);name=Column(String);status=Column(String);created_at=Column(String);processed_frames=Column(Integer,default=0);error=Column(Text,nullable=True)
Base.metadata.create_all(DB)
MODEL_PATH=os.getenv('MODEL_PATH',str(ROOT/'models'/'yolov8n.onnx'))
API_KEY=os.getenv('VISION_API_KEY','')
if not API_KEY: print('WARNING: VISION_API_KEY is unset. Bind to loopback only; set a long random key before network exposure.')
model=None;model_error=None;model_lock=asyncio.Lock();jobs={}
def auth(token):
    if API_KEY and token!=API_KEY:raise HTTPException(401,'Invalid API key')
def detector():
    global model,model_error
    if model is None:
        try:model=Detector(MODEL_PATH);model_error=None
        except Exception as e:model_error=str(e);raise
    return model
@asynccontextmanager
async def lifespan(app):
    yield
app=FastAPI(title='Vision Counter API',version='0.1.0',lifespan=lifespan)
origins=os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST','PATCH'],allow_headers=['*'])
class Correction(BaseModel):direction:str;reason:str=Field(min_length=3)
class Source(BaseModel):name:str;url:str
def row(e):return {k:getattr(e,k) for k in ('id','session_id','created_at','line_id','line_name','direction','category','track_id','corrected_direction','correction_reason')}
@app.get('/api/health')
def health():
    ok=Path(MODEL_PATH).is_file()
    return {'status':'ready' if ok else 'model_required','model_ready':ok,'model_path':str(Path(MODEL_PATH).name),'time':datetime.now(timezone.utc).isoformat(),'classes':COCO,'model_error':model_error}
@app.get('/api/events')
def get_events(x_api_key:str|None=Header(None),limit:int=200):
    auth(x_api_key)
    with Session(DB) as db:return [row(e) for e in db.scalars(select(Event).order_by(Event.created_at.desc()).limit(min(limit,1000)))]
@app.patch('/api/events/{eid}')
def correct_event(eid:str,payload:Correction,x_api_key:str|None=Header(None)):
    auth(x_api_key)
    if payload.direction not in ('in','out','invalid'):raise HTTPException(400,'Invalid direction')
    with Session(DB) as db:
        e=db.get(Event,eid)
        if not e:raise HTTPException(404)
        e.corrected_direction=payload.direction;e.correction_reason=payload.reason;db.commit();return row(e)
@app.get('/api/summary')
def summary(x_api_key:str|None=Header(None)):
    auth(x_api_key)
    with Session(DB) as db:
        events=list(db.scalars(select(Event)))
        counts={'in':0,'out':0,'invalid':0}
        for e in events:counts[e.corrected_direction or e.direction]+=1
        jobs_list=list(db.scalars(select(Recording).order_by(Recording.created_at.desc()).limit(100)))
    return {'counts':counts,'net':counts['in']-counts['out'],'events':len(events),'recordings':len(jobs_list),'jobs':[{'id':j.id,'name':j.name,'status':j.status,'processed_frames':j.processed_frames,'error':j.error} for j in jobs_list]}
@app.get('/api/export/{format}')
def export(format:str,x_api_key:str|None=Header(None)):
    auth(x_api_key)
    with Session(DB) as db:records=[row(e) for e in db.scalars(select(Event).order_by(Event.created_at))]
    keys=['id','session_id','created_at','line_id','line_name','direction','corrected_direction','correction_reason','category','track_id']
    if format=='csv':
        b=io.StringIO();w=csv.DictWriter(b,fieldnames=keys);w.writeheader();w.writerows(records)
        return StreamingResponse(iter([b.getvalue().encode('utf-8-sig')]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=events.csv'})
    if format=='xlsx':
        wb=Workbook();ws=wb.active;ws.title='Events';ws.append(keys)
        for r in records:ws.append([r[k] for k in keys])
        b=io.BytesIO();wb.save(b);b.seek(0)
        return StreamingResponse(b,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':'attachment; filename=events.xlsx'})
    if format=='pdf':
        from weasyprint import HTML
        counts={'in':0,'out':0,'invalid':0}
        for r in records:counts[r['corrected_direction'] or r['direction']]+=1
        rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k] if r[k] is not None else ""))}</td>' for k in ('created_at','line_name','category','direction','corrected_direction'))+'</tr>' for r in records)
        document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><style>@page{{size:A4 landscape;margin:16mm}}body{{font-family:'Noto Sans Arabic','DejaVu Sans',sans-serif;color:#17283b}}header{{border-bottom:3px solid #1686e4;padding-bottom:10px}}h1{{font-size:23px}}small{{color:#66809c}}.metrics{{display:flex;gap:22px;padding:14px;background:#edf5fc}}table{{width:100%;border-collapse:collapse;font-size:10px;margin-top:18px}}th{{background:#113354;color:white;text-align:left}}th,td{{padding:8px;border:1px solid #d8e2ed;word-wrap:break-word}}tr:nth-child(even){{background:#f4f8fd}}</style><header><h1>VISION COUNTER / Event Report</h1><small>Generated {datetime.now(timezone.utc).isoformat()} · Event timestamps UTC · All recorded sessions and lines · Manual corrections shown alongside originals</small></header><div class="metrics"><strong>Entries: {counts['in']}</strong><strong>Exits: {counts['out']}</strong><strong>Net: {counts['in']-counts['out']}</strong><strong>Invalid: {counts['invalid']}</strong></div><table><thead><tr><th>UTC timestamp</th><th>Gate</th><th>Class</th><th>Original direction</th><th>Corrected direction</th></tr></thead><tbody>{rows}</tbody></table></html>'''
        data=HTML(string=document).write_pdf()
        return StreamingResponse(io.BytesIO(data),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename=events.pdf'})
    raise HTTPException(400,'CSV, XLSX and PDF supported')
def save_events(session_id,events):
    if not events:return
    with Session(DB) as db:
        for e in events:db.add(Event(id=str(uuid.uuid4()),session_id=session_id,created_at=datetime.now(timezone.utc).isoformat(),**e))
        db.commit()
@app.websocket('/ws/live')
async def live(ws:WebSocket):
    if API_KEY and ws.query_params.get('token')!=API_KEY:await ws.close(code=1008);return
    await ws.accept();tracker=Tracker();rules=Rules();sid=str(uuid.uuid4());frame_seq=0
    await ws.send_json({'type':'session','session_id':sid})
    try:
        while True:
            payload=await ws.receive_text();started=time.perf_counter();msg=None
            try:
                msg=json.loads(payload)
                if msg.get('type')!='frame':continue
                frame_seq+=1
                content=base64.b64decode(msg['jpeg']);array=np.frombuffer(content,np.uint8);frame=cv2.imdecode(array,cv2.IMREAD_COLOR)
                if frame is None:raise ValueError('Invalid JPEG frame')
                if frame.shape[0]*frame.shape[1]>1920*1080:raise ValueError('Frame resolution too large')
                async with model_lock:
                    det=tracker.update(await asyncio.to_thread(detector().detect,frame,set(msg.get('classes') or [])))
                lines=msg.get('lines') or [];h,w=frame.shape[:2];events=rules.update(det,lines,w,h)
                await asyncio.to_thread(save_events,sid,events)
                annotated=draw(frame,det,lines)
                _,encoded=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,78])
                await ws.send_json({'type':'result','seq':msg.get('seq',frame_seq),'jpeg':base64.b64encode(encoded).decode(),'width':w,'height':h,'detections':[{'box':d.box,'category':d.category,'confidence':round(d.confidence,3),'track_id':d.track_id} for d in det],'events':events,'counts':[{'line_id':k[0],'direction':k[1],'category':k[2],'value':v} for k,v in rules.counts.items()],'processing_ms':round((time.perf_counter()-started)*1000,1)})
            except Exception as e:await ws.send_json({'type':'error','message':str(e),'seq':msg.get('seq') if isinstance(msg,dict) else None})
    except WebSocketDisconnect:pass
async def process_video(job_id,path):
    cap=cv2.VideoCapture(str(path));tracker=Tracker();frames=0
    try:
        if not cap.isOpened():raise ValueError('Unable to decode video')
        det=detector()
        while True:
            ok,frame=await asyncio.to_thread(cap.read)
            if not ok:break
            async with model_lock:tracker.update(await asyncio.to_thread(det.detect,frame))
            frames+=1
            if frames%15==0:
                with Session(DB) as db:
                    rec=db.get(Recording,job_id);rec.processed_frames=frames;db.commit()
                await asyncio.sleep(0)
        with Session(DB) as db:
            rec=db.get(Recording,job_id);rec.processed_frames=frames;rec.status='complete';db.commit()
    except Exception as e:
        with Session(DB) as db:
            rec=db.get(Recording,job_id);rec.status='error';rec.error=str(e);db.commit()
    finally:cap.release()
@app.post('/api/upload')
async def upload(file:UploadFile=File(...),x_api_key:str|None=Header(None)):
    auth(x_api_key)
    if not file.filename or Path(file.filename).suffix.lower() not in ('.mp4','.mov','.avi','.mkv','.webm'):raise HTTPException(400,'Unsupported video')
    name=Path(file.filename).name;id=str(uuid.uuid4());path=DATA/(id+Path(name).suffix.lower());total=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>500*1024*1024:raise HTTPException(413,'500MB upload limit')
                out.write(chunk)
    except: path.unlink(missing_ok=True);raise
    with Session(DB) as db:db.add(Recording(id=id,name=name,status='queued',created_at=datetime.now(timezone.utc).isoformat()));db.commit()
    task=asyncio.create_task(process_video(id,path));jobs[id]=task;task.add_done_callback(lambda _:jobs.pop(id,None))
    return {'id':id,'status':'queued'}
