from app.engine import Detection,Tracker,Rules,inside_polygon,Detector,ModelUnavailable
import pytest

def test_tracker_keeps_ids():
    t=Tracker();first=t.update([Detection((1,1,30,30),.8,'car')])[0];second=t.update([Detection((3,3,32,32),.8,'car')])[0]
    assert first.track_id==second.track_id

def test_bidirectional_crossing_and_no_duplicate():
    rules=Rules();line=[{'id':'gate','a':[.5,0],'b':[.5,1]}]
    def d(x):return Detection((int((x-.04)*100),40,int((x+.04)*100),60),.9,'car',track_id=10)
    assert rules.update([d(.3)],line,100,100)==[]
    assert len(rules.update([d(.7)],line,100,100))==1
    assert rules.update([d(.72)],line,100,100)==[]
    assert len(rules.update([d(.3)],line,100,100))==1
    assert sorted(rules.counts.values())==[1,1]

def test_polygon():
    sq=[(0,0),(1,0),(1,1),(0,1)]
    assert inside_polygon(.5,.5,sq)
    assert not inside_polygon(1.5,.5,sq)

def test_missing_model_rejected():
    with pytest.raises(ModelUnavailable):Detector('/tmp/nonexistent-vision-counter.onnx')
