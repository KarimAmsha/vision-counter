# Vision Counter (working prototype)

Responsive React/TypeScript PWA and Python/FastAPI video-analysis engine. Live **phone camera** frames are captured continuously and sent over a bidirectional WebSocket; OpenCV runs *real* ONNX inference and returns annotated, sequence-matched JPEG frames with track IDs and line-crossing counts. There is **no simulated detection**. IMPORTANT: this is a working **partial implementation**, not the entire requested production product. See `STATUS.md`.

## Run on macOS/Linux

Prerequisites: Python 3.11 or 3.12, Node.js 20+, 2+ GB free RAM, and a YOLOv8/YOLO11 detection-format ONNX file with 80 COCO classes exported at 640 px (see models/README.md). A CPU is supported but may be slow; performance is not benchmarked.

```bash
cd vision-counter
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
# Place an appropriate licensed ONNX file at models/yolov8n.onnx
export VISION_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MODEL_PATH="$(pwd)/models/yolov8n.onnx"
export DATA_DIR="$(pwd)/data"
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Separate terminal:
```bash
cd vision-counter/frontend
npm install
npm run dev
```

Open `http://localhost:5173` on the same computer. To use a PHONE: camera permissions require **trusted HTTPS** (localhost exceptions do NOT extend to phone accessing your computer's LAN IP). Put a properly configured TLS reverse proxy in front of the frontend and API/WebSocket, use the **same origin**, enable only authenticated access and restrict CORS. Do **not** expose development Uvicorn/Vite to the internet. Paste `VISION_API_KEY` into the settings on your private trusted device. Before first network deployment, add proper session authentication and secret-handling hardening; URL query tokens on WebSockets can leak to infrastructure logs, and current single-user browser token storage has risks.

`curl http://127.0.0.1:8000/api/health` returns `model_required` if weights are missing. For status tests:
```bash
cd backend
python -m pytest -q
cd ../frontend
npm run build
```

## Current behavior and constraints

* Only ONNX YOLOv8/11 detection with 80 COCO labels and 640 square input supported. Tracks use a **basic IoU tracker**, not ByteTrack; IDs are session-local and can be lost under occlusion.
* Add counting lines by dragging across **stopped** video preview. Counts are approximate crossing **events**, not occupancy or unique cars. Corrected direction is retained separately from original event direction.
* Only live browser camera is currently connected to inference + counting; recorded uploads are analyzed frame by frame but do **not** yet produce line events, annotated exports or resume. Large uploads limited to 500 MB.
* A frame is only displayed when its `seq` matches the outstanding capture; engine and end-to-end processing delay are shown separately. When the camera moves/reorients, manually stop, clear gates, and recalibrate.
* SQLite at `data/vision.sqlite`; originals remain at `data/<UUID>.<ext>` and are not overwritten. Retention/backup management not implemented. Treat recordings as private.
* CSV UTF-8 BOM, Excel and PDF exports reflect manual corrections. The frontend uses localStorage for personal UI preferences and access key.
* No microphone permission is requested. Streams stop when closed. No claim of background camera capture.

This prototype uses WebSocket for frame transport, **not WebRTC**, and does not yet offer RTSP/IP cameras, local USB engine cameras, true offline inference, multi-source monitoring, parking occupancy, alerts, heat maps, tests on physical phones, or production-grade roles/security. These require subsequent development and acceptance tests.
