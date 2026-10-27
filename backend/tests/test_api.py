from fastapi.testclient import TestClient
from app.main import app,DB,Event
from sqlalchemy.orm import Session
import uuid

def test_health_and_reports():
    client=TestClient(app)
    h=client.get('/api/health')
    assert h.status_code==200 and 'model_ready' in h.json()
    assert client.get('/api/summary').status_code==200
    assert client.get('/api/export/csv').content.startswith(b'\xef\xbb\xbf')
    assert client.get('/api/export/xlsx').content.startswith(b'PK')
    assert client.get('/api/export/pdf').content.startswith(b'%PDF')

def test_manual_correction_keeps_original():
    client=TestClient(app);eid=str(uuid.uuid4())
    with Session(DB) as db:
        db.add(Event(id=eid,session_id='test',created_at='2026-01-01T00:00:00+00:00',line_id='l',line_name='Test gate',direction='in',category='car',track_id=1));db.commit()
    try:
        res=client.patch('/api/events/'+eid,json={'direction':'out','reason':'Manual audit shows reverse crossing'})
        assert res.status_code==200
        assert res.json()['direction']=='in' and res.json()['corrected_direction']=='out'
        assert client.get('/api/summary').json()['counts']['out']>=1
    finally:
        with Session(DB) as db:
            row=db.get(Event,eid);db.delete(row);db.commit()

def test_websocket_session_and_explicit_missing_model():
    import base64,cv2,numpy as np
    ok,buf=cv2.imencode('.jpg',np.zeros((80,80,3),dtype=np.uint8))
    with TestClient(app).websocket_connect('/ws/live') as socket:
        assert socket.receive_json()['type']=='session'
        socket.send_json({'type':'frame','seq':8,'jpeg':base64.b64encode(buf).decode(),'lines':[]})
        answer=socket.receive_json()
        assert answer['seq']==8
        assert answer['type'] in ('result','error')
        if not __import__('pathlib').Path(__import__('app.main',fromlist=['MODEL_PATH']).MODEL_PATH).is_file():
            assert answer['type']=='error' and 'model' in answer['message'].lower()
