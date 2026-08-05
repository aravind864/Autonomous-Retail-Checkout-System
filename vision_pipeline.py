"""
vision_pipeline.py
------------------
Motion-gated + batched dual-camera detection pipeline.

Algorithm (one cycle):
  Setup  (once)
    1. Export YOLO -> OpenVINO and load a SINGLE shared model instance.
    2. Init MOG2 background subtractor for Cam1.
    3. Init MOG2 background subtractor for Cam2.
    4. Open Cam1 and Cam2 streams.

  Main loop (repeats every cycle via BatchedDetector._detect_loop)
    1.  Read one frame from Cam1  (via CameraGrabber grab thread)
    2.  Read one frame from Cam2
    3.  Apply MOG2 to Cam1 frame -> motion mask, count changed pixels
    4.  Apply MOG2 to Cam2 frame -> motion mask, count changed pixels
    5.  Decide: motion_cam1? motion_cam2?
    6.  If NEITHER has motion -> skip YOLO, show raw frames + STATIC badge, next cycle.
    7.  If EITHER  has motion -> collect both frames into a 2-frame batch.
    8.  Run shared YOLO once on the batch -> results[0]=Cam1, results[1]=Cam2.
    9.  Extract Cam1 result, draw bounding boxes on Cam1 frame.
    10. Extract Cam2 result, draw bounding boxes on Cam2 frame.
    11. Store/display both annotated frames.
"""

import os
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from urllib.parse import urlparse, urlunparse
from reid_engine import PersonReIDEngine

# ── Model selection: prefer fastest available format ───────────────────────────
# Benchmark on Intel Core i3-1005G1:
#   OpenVINO  avg 22ms, best 13ms -> ~45 fps  (3.2x faster than PyTorch)
#   ONNX      avg ~35ms           -> ~28 fps
#   PyTorch   avg 70ms, best 53ms -> ~14 fps
if os.path.isdir("yolo26n_openvino_model"):
    MODEL_WEIGHTS = "yolo26n_openvino_model"
elif os.path.exists("yolo26n.onnx"):
    MODEL_WEIGHTS = "yolo26n.onnx"
elif os.path.exists("yolo26n.pt"):
    MODEL_WEIGHTS = "yolo26n.pt"
else:
    MODEL_WEIGHTS = "yolov8n.pt"

YOLO_IMGSZ    = 256    # Smaller = faster CPU inference
MOTION_THRESH = 0.015  # MOG2 motion threshold (1.5% of pixels)
JPEG_QUALITY  = 80     # JPEG encode quality for streaming

_PALETTE = [
    (255,  56,  56),
    ( 56, 255,  56),
    ( 56,  56, 255),
    (255, 200,  56),
    (200,  56, 255),
]


# ── Setup (once): load single shared YOLO model ───────────────────────────────
print(f"[VISION] Loading shared model: {MODEL_WEIGHTS} ...")
_shared_model = YOLO(MODEL_WEIGHTS)
print("[VISION] Shared YOLO model ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_source(source) -> str:
    if source is None:
        return ""
    s = str(source).strip()
    if not s or "YOUR_PHONE_IP" in s:
        return ""
    if s.isdigit():
        return s
    if not (s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://")):
        s = "http://" + s
    parsed = urlparse(s)
    if parsed.scheme in {"http", "https"}:
        path = parsed.path or ""
        if path in {"", "/"}:
            parsed = parsed._replace(path="/video")
            return urlunparse(parsed)
    return s


def _open_cap(source: str) -> "cv2.VideoCapture | None":
    s = _normalize_source(source)
    if not s:
        return None

    is_ip = not s.isdigit()

    if is_ip:
        # Tell FFmpeg to disable ALL internal buffering for IP streams.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer"
            "|flags;low_delay"
            "|analyzeduration;0"
            "|probesize;32"
            "|rtsp_transport;tcp"
        )
        try:
            cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(s)
        except Exception:
            try:
                cap = cv2.VideoCapture(s)
            except Exception:
                return None
    else:
        # Windows index camera: try DSHOW, MSMF, and default safely
        idx = int(s)
        cap = None
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]:
            try:
                if backend is not None:
                    c = cv2.VideoCapture(idx, backend)
                else:
                    c = cv2.VideoCapture(idx)
                if c and c.isOpened():
                    cap = c
                    break
            except Exception:
                continue

    if cap and cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        except Exception:
            pass

        if is_ip:
            print(f"[CAM] Draining initial buffer for {s} ...")
            try:
                for _ in range(30):
                    cap.grab()
            except Exception:
                pass
            print("[CAM] Buffer drained. Stream is now live.")

    return cap


def _draw_boxes(frame: np.ndarray, boxes: list, cam_title: str = "CAMERA",
                state_mgr=None) -> np.ndarray:
    """Draw bounding boxes and a sleek Active Customer Count overlay."""
    annotated = frame.copy()
    cust_count = len(boxes)

    for b in boxes:
        tid            = b["track_id"]
        x1, y1, x2, y2 = b["bbox"]
        colour         = _PALETTE[tid % len(_PALETTE)]
        cid            = state_mgr.track_to_customer[tid] if (state_mgr and tid in state_mgr.track_to_customer) else f"CUST_{tid}"
        label          = f"{cid}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 12, max(th + 10, y1)), colour, -1)
        cv2.putText(annotated, label, (x1 + 6, max(th + 4, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Top overlay bar (Dark Slate Glass Pill)
    overlay = annotated.copy()
    w = annotated.shape[1]
    banner_w = min(210, w - 24)
    cv2.rectangle(overlay, (12, 12), (12 + banner_w, 48), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (12, 12), (12 + banner_w, 48), (51, 65, 85), 1)

    # Vibrant Live indicator dot
    cv2.circle(annotated, (26, 30), 4, (56, 255, 56), -1, cv2.LINE_AA)

    # Display ONLY Active Customers count
    ui_text = f"Active Customers: {cust_count}"
    cv2.putText(annotated, ui_text, (38, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)

    return annotated


# ── Setup (once): per-camera frame grabbers ───────────────────────────────────

class CameraGrabber:
    """
    Runs a tight grab loop for one camera.
    cap.grab() drains the hardware buffer every cycle so retrieve() always
    returns the newest frame. No YOLO here -- detection is handled in
    BatchedDetector which processes both cameras together.
    """

    _SOURCE_TTL: float = 5.0

    def __init__(self, name: str, config_key: str):
        self.name       = name
        self.config_key = config_key

        self._raw_lock      = threading.Lock()
        self._raw_frame     = None          # type: np.ndarray | None
        self._new_frame_evt = threading.Event()

        self._src_cache      = ""
        self._src_cache_time = 0.0

        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name=f"grab-{name}"
        )

    def start(self):
        self._thread.start()
        print(f"[{self.name}] Grab thread started.")

    def get_frame(self):
        """Return the latest raw frame (or None if not ready yet)."""
        with self._raw_lock:
            return self._raw_frame

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_source(self) -> str:
        now = time.monotonic()
        if now - self._src_cache_time > self._SOURCE_TTL:
            try:
                import config
                self._src_cache      = config.reload().get(self.config_key, "")
                self._src_cache_time = now
            except Exception:
                pass
        return self._src_cache

    def _grab_loop(self):
        cap         = None
        current_src = ""
        last_check  = 0.0

        while True:
            now = time.monotonic()

            if now - last_check > self._SOURCE_TTL:
                new_src    = self._get_source()
                last_check = now
            else:
                new_src = current_src

            # Re-open if source changed or cap died
            if new_src != current_src or cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = _open_cap(new_src)
                current_src = new_src
                if cap is None or not cap.isOpened():
                    time.sleep(1.0)
                    continue
                print(f"[{self.name}] Camera opened: {new_src or '(none)'}")

            # Read fresh frame continuously (non-blocking)
            ret, frame = cap.read()
            if ret and frame is not None:
                with self._raw_lock:
                    self._raw_frame = frame
                self._new_frame_evt.set()
            else:
                time.sleep(0.01)


# ── Main loop: batched detector ───────────────────────────────────────────────

class BatchedDetector:
    """
    Motion-gated batched YOLO detector (the heart of the pipeline).

    Per cycle:
      1. Read latest frame from Cam1 and Cam2.
      2. Apply per-camera MOG2 -> motion_cam1, motion_cam2.
      3. If NEITHER has motion -> store raw frames + STATIC badge, next cycle.
      4. If EITHER  has motion -> stack both frames into a 2-image batch.
      5. Run _shared_model.track(batch) once -> results[0]=Cam1, results[1]=Cam2.
      6. Parse each result, draw boxes, update annotated frames.
    """

    def __init__(self, grabber1: CameraGrabber, grabber2: CameraGrabber,
                 is_entry1: bool = True, state_mgr=None):
        self.grabber1  = grabber1
        self.grabber2  = grabber2
        self.is_entry1 = is_entry1
        self.state_mgr = state_mgr

        # Setup (once): per-camera MOG2 background subtractors (cannot be shared)
        self.bg_sub1 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=24, detectShadows=False
        )
        self.bg_sub2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=24, detectShadows=False
        )

        # Detection state -- written by _detect_loop, read by get_jpeg / get_tracks
        self._det_lock = threading.Lock()
        self._state = {
            "cam1": {"boxes": [], "tracks": {}, "badge": "STATIC | YOLO SKIPPED", "bcolor": (130, 130, 130)},
            "cam2": {"boxes": [], "tracks": {}, "badge": "STATIC | YOLO SKIPPED", "bcolor": (130, 130, 130)},
        }

        self._batch_supported = True
        self.reid_engine = PersonReIDEngine(sim_threshold=0.45)

        # Latest annotated frames (always pre-initialized so video feed never returns None)
        self._frame_lock = threading.Lock()
        self._ann_frame1 = self._make_standby_frame("Entry Camera", "Initializing...")
        self._ann_frame2 = self._make_standby_frame("Shelf Camera", "Initializing...")

        self._thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="batched-detector"
        )

    def _make_standby_frame(self, cam_name: str, message: str) -> np.ndarray:
        """Create a stylized standby frame when a camera is offline or initializing."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 35) # Dark gray background
        cv2.rectangle(img, (20, 20), (620, 460), (60, 60, 70), 2)
        title = f"{cam_name}: {message}"
        cv2.putText(img, title, (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(img, "Check camera config in Admin Panel (/admin)", (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        return img

    def start(self):
        self._thread.start()
        print("[BATCHED] Detector thread started — single model, motion-gated batch inference.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_jpeg1(self):
        return self._encode_jpeg(1)

    def get_jpeg2(self):
        return self._encode_jpeg(2)

    def get_tracks1(self) -> dict:
        with self._det_lock:
            return dict(self._state["cam1"]["tracks"])

    def get_tracks2(self) -> dict:
        with self._det_lock:
            return dict(self._state["cam2"]["tracks"])

    def _encode_jpeg(self, cam_id: int):
        grabber   = self.grabber1 if cam_id == 1 else self.grabber2
        cam_key   = "cam1" if cam_id == 1 else "cam2"
        cam_title = "ENTRY CAM" if cam_id == 1 else "SHELF CAM"

        raw_frame = grabber.get_frame()

        # Handle single-webcam mirror fallback
        if raw_frame is None and cam_id == 2:
            raw_frame = self.grabber1.get_frame()
        elif raw_frame is None and cam_id == 1:
            raw_frame = self.grabber2.get_frame()

        if raw_frame is None:
            src = grabber._get_source() or "None"
            annotated = self._make_standby_frame(cam_title, f"Offline ({src})")
        else:
            with self._det_lock:
                boxes = list(self._state[cam_key].get("boxes", []))
            annotated = _draw_boxes(raw_frame, boxes, cam_title, self.state_mgr)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    # ── Detection loop (the main algorithm) ───────────────────────────────────

    def _motion_ratio(self, bg_sub, frame: np.ndarray) -> float:
        """Apply MOG2 on a downscaled gray frame; return changed-pixel ratio."""
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask  = bg_sub.apply(gray)
        return cv2.countNonZero(mask) / (320.0 * 240.0)

    def _detect_loop(self):
        while True:
            try:
                # Step 1-2: Read one frame from Cam1 and one from Cam2
                frame1 = self.grabber1.get_frame()
                frame2 = self.grabber2.get_frame()

                # If both grabbers point to the same index (e.g. webcam "0"),
                # Windows locks the camera to grabber1. Mirror frame1 to frame2 if frame2 is None.
                if frame2 is None and frame1 is not None:
                    s1 = self.grabber1._get_source()
                    s2 = self.grabber2._get_source()
                    if s1 == s2 or not s2:
                        frame2 = frame1.copy()
                elif frame1 is None and frame2 is not None:
                    s1 = self.grabber1._get_source()
                    s2 = self.grabber2._get_source()
                    if s1 == s2 or not s1:
                        frame1 = frame2.copy()

                ph1_used = (frame1 is None)
                ph2_used = (frame2 is None)

                if ph1_used:
                    src1 = self.grabber1._get_source() or "None"
                    frame1 = self._make_standby_frame("Entry Camera", f"Offline ({src1})")
                if ph2_used:
                    src2 = self.grabber2._get_source() or "None"
                    frame2 = self._make_standby_frame("Shelf Camera", f"Offline ({src2})")

                # Step 3-5: Apply MOG2 -> motion masks -> threshold
                # Skip MOG2 on standby/placeholder frames
                ratio1  = 0.0 if ph1_used else self._motion_ratio(self.bg_sub1, frame1)
                ratio2  = 0.0 if ph2_used else self._motion_ratio(self.bg_sub2, frame2)
                moving1 = ratio1 >= MOTION_THRESH
                moving2 = ratio2 >= MOTION_THRESH

                # Step 6: Neither cam has motion -> skip YOLO, show raw frames
                if not moving1 and not moving2:
                    with self._det_lock:
                        self._state["cam1"].update({"boxes": [], "tracks": {}})
                        self._state["cam2"].update({"boxes": [], "tracks": {}})

                    ann1 = _draw_boxes(frame1, [], "ENTRY CAM", self.state_mgr) if not ph1_used else frame1
                    ann2 = _draw_boxes(frame2, [], "SHELF CAM", self.state_mgr) if not ph2_used else frame2

                    with self._frame_lock:
                        self._ann_frame1 = ann1
                        self._ann_frame2 = ann2

                    time.sleep(0.030)
                    continue

                # Step 7-8: Run shared YOLO on batch (or fallback to per-frame if static OpenVINO shape)
                if self._batch_supported:
                    try:
                        batch = [frame1, frame2]
                        results = _shared_model.track(
                            batch,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=[0],
                            imgsz=YOLO_IMGSZ,
                            conf=0.35,
                            verbose=False,
                        )
                        r1, r2 = results[0], results[1]
                    except Exception:
                        print("[VISION] OpenVINO model has static batch=1 shape. Running per-frame motion-gated ByteTrack inference.")
                        self._batch_supported = False
                        res1 = _shared_model.track(frame1, persist=True, tracker="bytetrack.yaml", classes=[0],
                                                   imgsz=YOLO_IMGSZ, conf=0.35, verbose=False)
                        res2 = _shared_model.track(frame2, persist=True, tracker="bytetrack.yaml", classes=[0],
                                                   imgsz=YOLO_IMGSZ, conf=0.35, verbose=False)
                        r1 = res1[0] if res1 else None
                        r2 = res2[0] if res2 else None
                else:
                    res1 = _shared_model.track(frame1, persist=True, tracker="bytetrack.yaml", classes=[0],
                                               imgsz=YOLO_IMGSZ, conf=0.35, verbose=False)
                    res2 = _shared_model.track(frame2, persist=True, tracker="bytetrack.yaml", classes=[0],
                                               imgsz=YOLO_IMGSZ, conf=0.35, verbose=False)
                    r1 = res1[0] if res1 else None
                    r2 = res2[0] if res2 else None

                # Step 9: Extract Cam1 result, draw bounding boxes on Cam1 frame
                ann1 = self._parse_and_draw(r1, frame1, "ENTRY CAM",
                                             cam_key="cam1", is_entry=self.is_entry1)

                # Step 10: Extract Cam2 result, draw bounding boxes on Cam2 frame
                ann2 = self._parse_and_draw(r2, frame2, "SHELF CAM",
                                             cam_key="cam2", is_entry=False)

                # Step 11: Store annotated frames for JPEG streaming
                with self._frame_lock:
                    self._ann_frame1 = ann1
                    self._ann_frame2 = ann2

            except Exception as e:
                print(f"[BATCHED] detect_loop error (thread kept alive): {e}")
                time.sleep(0.1)

    def _parse_and_draw(self, result, frame: np.ndarray, cam_title: str,
                        cam_key: str, is_entry: bool) -> np.ndarray:
        """Parse one YOLO result, update detection state, return annotated frame."""
        boxes_out  = []
        tracks_out = {}

        if result is not None and result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                tid             = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy          = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # OSNet / Deep Feature Embedding + Cosine Similarity cross-camera matching
                cid = self.reid_engine.match_or_register(frame, (x1, y1, x2, y2), tid, self.state_mgr)

                tracks_out[tid] = (cx, cy)
                boxes_out.append({"track_id": tid, "bbox": (x1, y1, x2, y2), "pos": (cx, cy), "cust_id": cid})

        with self._det_lock:
            self._state[cam_key].update({
                "boxes":  boxes_out,
                "tracks": tracks_out,
            })

        return _draw_boxes(frame, boxes_out, cam_title, self.state_mgr)


# ── VisionSystem: public facade (backward-compatible with app.py and main.py) ──

class VisionSystem:
    """
    Owns two CameraGrabbers + one BatchedDetector.
    Single shared YOLO model loaded once; one inference call per cycle
    regardless of how many cameras triggered motion.
    """

    def __init__(self, state_manager=None):
        self.state_mgr = state_manager

        # Setup (once): open Cam1 and Cam2 streams via grabber threads
        self.grabber1 = CameraGrabber("Entry Camera", config_key="entry_camera")
        self.grabber2 = CameraGrabber("Shelf Camera", config_key="shelf_camera")

        self.detector = BatchedDetector(
            self.grabber1, self.grabber2,
            is_entry1=True,
            state_mgr=state_manager,
        )

        # Back-compat aliases expected by app.py
        self.entry_stream = self
        self.shelf_stream = self

        self.grabber1.start()
        self.grabber2.start()
        self.detector.start()

        print("[VISION] Pipeline LIVE — single batched YOLO model, motion-gated per-cam.")

    # ── Flask MJPEG streaming (app.py) ────────────────────────────────────────

    def entry_get_jpeg(self):
        return self.detector.get_jpeg1()

    def shelf_get_jpeg(self):
        return self.detector.get_jpeg2()

    # Unified getter (used in some app.py routes)
    def get_jpeg(self, cam_id: int = 1):
        return self.detector.get_jpeg1() if cam_id == 1 else self.detector.get_jpeg2()

    # ── Desktop display (main.py) ──────────────────────────────────────────────

    def process_entry_camera(self, frame: np.ndarray) -> np.ndarray:
        """
        Synchronous single-frame path for main.py desktop window.
        MOG2 gate -> YOLO on Cam1 frame only (desktop fallback).
        """
        ratio  = self.detector._motion_ratio(self.detector.bg_sub1, frame)
        moving = ratio >= MOTION_THRESH

        if not moving:
            return _draw_boxes(frame, [], "ENTRY CAM", self.state_mgr)

        results = _shared_model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            imgsz=YOLO_IMGSZ, conf=0.35, verbose=False,
        )
        boxes, tracks = [], {}
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid             = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy          = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cid             = self.detector.reid_engine.match_or_register(frame, (x1, y1, x2, y2), tid, self.state_mgr)
                tracks[tid]     = (cx, cy)
                boxes.append({"track_id": tid, "bbox": (x1, y1, x2, y2), "cust_id": cid})
        with self.detector._det_lock:
            self.detector._state["cam1"]["tracks"] = tracks
        return _draw_boxes(frame, boxes, "ENTRY CAM", self.state_mgr)

    def process_shelf_camera(self, frame: np.ndarray) -> np.ndarray:
        """
        Synchronous single-frame path for main.py desktop window.
        MOG2 gate -> YOLO on Cam2 frame only (desktop fallback).
        """
        ratio  = self.detector._motion_ratio(self.detector.bg_sub2, frame)
        moving = ratio >= MOTION_THRESH

        if not moving:
            return _draw_boxes(frame, [], "SHELF CAM", self.state_mgr)

        results = _shared_model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            imgsz=YOLO_IMGSZ, conf=0.35, verbose=False,
        )
        boxes, tracks = [], {}
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid             = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy          = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cid             = self.detector.reid_engine.match_or_register(frame, (x1, y1, x2, y2), tid, self.state_mgr)
                tracks[tid]     = (cx, cy)
                boxes.append({"track_id": tid, "bbox": (x1, y1, x2, y2), "cust_id": cid})
        with self.detector._det_lock:
            self.detector._state["cam2"]["tracks"] = tracks
        return _draw_boxes(frame, boxes, "SHELF CAM", self.state_mgr)

    # ── Sensor bridge ──────────────────────────────────────────────────────────

    def get_latest_shelf_tracks(self) -> dict:
        return self.detector.get_tracks2()
