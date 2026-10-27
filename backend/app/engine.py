"""OpenCV ONNX YOLOv8/YOLO11 inference, tracking and line crossing."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

@dataclass
class Detection:
    box: tuple[int,int,int,int]
    confidence: float
    category: str
    track_id: int | None = None

class ModelUnavailable(RuntimeError): pass

class Detector:
    def __init__(self, path: str, labels: list[str] | None = None, conf: float=.4, size: int=640):
        self.path=path; self.labels=labels or COCO; self.conf=conf; self.size=size
        if not Path(path).is_file(): raise ModelUnavailable(f'ONNX model missing: {path}. Add a trusted YOLOv8/YOLO11 detect ONNX model to models/ and set MODEL_PATH.')
        try: self.net=cv2.dnn.readNetFromONNX(path)
        except cv2.error as exc: raise ModelUnavailable(f'ONNX model incompatible with OpenCV: {exc}') from exc
    def detect(self, frame: np.ndarray, chosen: set[str]|None=None)->list[Detection]:
        h,w=frame.shape[:2]; s=self.size
        factor=min(s/w,s/h); nw,nh=round(w*factor),round(h*factor)
        resized=cv2.resize(frame,(nw,nh)); canvas=np.full((s,s,3),114,dtype=np.uint8)
        px=(s-nw)//2;py=(s-nh)//2;canvas[py:py+nh,px:px+nw]=resized
        blob=cv2.dnn.blobFromImage(canvas,1/255.,(s,s),swapRB=True,crop=False)
        self.net.setInput(blob); result=self.net.forward()
        out=np.squeeze(result)
        if out.ndim!=2:raise RuntimeError(f'Unexpected model output shape: {result.shape}')
        cls_count=len(self.labels)
        if out.shape[0]==cls_count+4:out=out.T
        if out.shape[1]!=cls_count+4:raise RuntimeError(f'Model output shape {result.shape} incompatible with {cls_count} labels; only YOLOv8/11 detect ONNX is supported')
        boxes=[];scores=[];classes=[]
        for row in out:
            class_id=int(np.argmax(row[4:])); confidence=float(row[4+class_id]); label=self.labels[class_id]
            if confidence<self.conf or (chosen and label not in chosen):continue
            cx,cy,bw,bh=map(float,row[:4]);x1=max(0,int(((cx-bw/2)-px)/factor));y1=max(0,int(((cy-bh/2)-py)/factor))
            x2=min(w,int(((cx+bw/2)-px)/factor));y2=min(h,int(((cy+bh/2)-py)/factor))
            if x2<=x1 or y2<=y1:continue
            boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(confidence);classes.append(label)
        if not boxes:return []
        picks=[]
        for label in set(classes):
            indices=[i for i,c in enumerate(classes) if c==label]
            keep=cv2.dnn.NMSBoxes([boxes[i] for i in indices],[scores[i] for i in indices],self.conf,.45)
            for j in np.array(keep).flatten():picks.append(indices[int(j)])
        return [Detection((boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3]),scores[i],classes[i]) for i in picks]

class Tracker:
    def __init__(self):self.tracks={};self.next_id=1;self.frame=0
    def update(self,detections:list[Detection])->list[Detection]:
        self.frame+=1;used=set()
        for d in sorted(detections,key=lambda x:-x.confidence):
            best=None;score=.25
            for tid,(box,category,last) in self.tracks.items():
                if tid in used or category!=d.category or self.frame-last>12:continue
                xx1=max(box[0],d.box[0]);yy1=max(box[1],d.box[1]);xx2=min(box[2],d.box[2]);yy2=min(box[3],d.box[3]);inter=max(0,xx2-xx1)*max(0,yy2-yy1)
                aa=(box[2]-box[0])*(box[3]-box[1]);bb=(d.box[2]-d.box[0])*(d.box[3]-d.box[1]);iou=inter/max(1,aa+bb-inter)
                if iou>score:score=iou;best=tid
            if best is None:best=self.next_id;self.next_id+=1
            d.track_id=best;used.add(best);self.tracks[best]=(d.box,d.category,self.frame)
        self.tracks={k:v for k,v in self.tracks.items() if self.frame-v[2]<=12}
        return detections

def signed_side(point,a,b):return (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
def inside_polygon(x,y,poly):
    inside=False;j=len(poly)-1
    for i in range(len(poly)):
        xi,yi=poly[i];xj,yj=poly[j]
        if (yi>y)!=(yj>y) and x<(xj-xi)*(y-yi)/(yj-yi)+xi:inside=not inside
        j=i
    return inside

class Rules:
    def __init__(self):self.last_side={};self.counts={};self.latest_seen={}
    def update(self,detections,lines,width,height):
        events=[]
        for d in detections:
            if d.track_id is None:continue
            x=(d.box[0]+d.box[2])/(2*width);y=(d.box[1]+d.box[3])/(2*height)
            for line in lines:
                a=line['a'];b=line['b'];side=signed_side((x,y),a,b)
                length=max(.001,((b[0]-a[0])**2+(b[1]-a[1])**2)**.5)
                if abs(side)/length<.015:continue
                sign=1 if side>0 else -1;key=(line['id'],d.track_id)
                old=self.last_side.get(key)
                if old is not None and old!=sign:
                    direction='in' if old<sign else 'out'
                    count_key=(line['id'],direction,d.category)
                    self.counts[count_key]=self.counts.get(count_key,0)+1
                    events.append({'line_id':line['id'],'line_name':line.get('name',line['id']),'direction':direction,'category':d.category,'track_id':d.track_id})
                self.last_side[key]=sign
        return events

def draw(frame,detections,lines):
    h,w=frame.shape[:2];out=frame.copy()
    for line in lines:
        a=tuple(int(v*s) for v,s in zip(line['a'],(w,h)));b=tuple(int(v*s) for v,s in zip(line['b'],(w,h)))
        cv2.line(out,a,b,(30,210,246),3);cv2.putText(out,line.get('name','Line'),a,cv2.FONT_HERSHEY_SIMPLEX,.65,(30,210,246),2)
    for d in detections:
        x1,y1,x2,y2=d.box;cv2.rectangle(out,(x1,y1),(x2,y2),(56,218,150),2)
        cv2.putText(out,f'{d.category} #{d.track_id} {d.confidence:.2f}',(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(56,218,150),2)
    return out
