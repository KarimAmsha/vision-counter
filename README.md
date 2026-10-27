# Vision Counter (working prototype)

Responsive React/TypeScript PWA and Python/FastAPI video-analysis engine. Live phone-camera frames stream over WebSocket to OpenCV ONNX inference; annotated sequence-matched frames, tracking IDs and counting results return to the client. Detection is never simulated.

**This is a partial implementation, not the full production product.** See STATUS.md.

## Run locally

Requires Python 3.11/3.12, Node.js 20+, and a licensed YOLOv8/YOLO11 80-class COCO detection ONNX model at 640px.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
export VISION_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MODEL_PATH="$(pwd)/models/yolov8n.onnx"
export DATA_DIR="$(pwd)/data"
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Phone camera requires trusted HTTPS and a securely configured same-origin proxy for API/WebSocket; local IP HTTP is not a secure camera context. Never publish development servers directly. Supply your own ONNX weights; no model is bundled. This version uses WebSocket frame transport rather than WebRTC and basic IoU tracking instead of ByteTrack. It lacks production-ready authentication, RTSP, parking, alerting and verified phone-device acceptance testing.
