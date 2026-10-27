# Implementation status — 2026-10-27

## Implemented in source
- React/TS responsive dashboard, live camera, video uploads, event log/editor, settings, EN/AR/TR labels, RTL, light/dark mode, installable PWA metadata.
- Python FastAPI + OpenCV DNN YOLOv8/11 ONNX inference; no fake detections, image annotation, IoU tracker, line crossing both ways with jitter deadband.
- Browser camera frame streaming over authenticated WebSocket with sequence matching, end-to-end latency indicator, persistent SQLite crossing events, revisions and CSV/XLSX/PDF export.
- Background upload processing and status tracking. Minimal Python tests for core tracking/counting rules.

## Not complete / not verified
- Core features still missing: WebRTC, ByteTrack, polygons/zone logic in running API, parking slots, alerts, RTSP/USB ingestion, annotated video export, resumable chunked uploads, persistent settings/models registry, correction audit revisions, automatic calibration invalidation, backups/restoration, queue fairness/cancellation/restart.
- Frontend package installation/build may require online npm registry access; actual video model not bundled (license/size/availability). No actual real model inference test was possible without weights.
- The server does NOT claim hardened multi-user security: needs production auth/session handling, limits, HTTPS proxy, private media access policy.
- No physical mobile, Safari/iOS, low-end CPU, accuracy ground-truth, TLS, load or reliability tests performed.

Do not treat this as final acceptance. See README for accurate startup instructions and limitations.
