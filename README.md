# Autonomous Retail Checkout System

## Configuration & Usage

1. Copy `.env.example` to `.env` and replace its secret placeholders:
   - `FLASK_SECRET_KEY`
   - `ADMIN_USER`
   - `ADMIN_PASS`
   - `POSTGRES_PASSWORD` (along with `POSTGRES_DATABASE`, `POSTGRES_USER`, etc.)

2. Camera Configuration:
   - Camera sources are configured via the Admin Panel at `http://localhost:5000/admin`.
   - Settings persist automatically across restarts.
   - Camera settings update live instantly in the Surveillance Dashboard without requiring a server restart.
   - The application normalizes base IP Webcam URLs (e.g. `http://HOST:8080` to `http://HOST:8080/video`).

## Two-camera identity flow

- Camera 1 runs YOLO person detection, ByteTrack, then face identification. Only a face-verified customer is added to the active re-identification gallery.
- Camera 2 runs an independent YOLO/ByteTrack instance. Its boxes remain unlabeled while four body frames are collected; only a verified customer ID is shown.
- Camera 2 evaluates appearance, silhouette, entry-to-shelf time, and the valid CAM1→CAM2 route. It assigns the Camera 1 customer ID only at score `>= 0.75`; `0.50–0.75` remains uncertain and lower scores remain unassigned.

The default detector is the fast `yolo26n_openvino_model` at 256px. For distant people, update `MODEL_WEIGHTS` and `YOLO_IMGSZ` in `vision_pipeline.py`; this trades live speed for accuracy. Face, detection, and re-identification tuning constants are also defined in code, keeping `.env` limited to secrets. `torchreid` is listed in `requirements.txt`; when installed it enables OSNet (`osnet_x0_25`). The runtime reports a clear MobileNet fallback if OSNet is unavailable.
