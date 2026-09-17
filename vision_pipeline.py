"""Dual-camera motion-gated detection and tracking pipeline."""

import os
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from urllib.parse import urlparse, urlunparse
from reid_engine import PersonReIDEngine
from ai.entry_face_manager import EntryFaceManager, get_customer_color


MODEL_WEIGHTS = "yolo26n_openvino_model"
if not os.path.exists(MODEL_WEIGHTS) and not os.path.isdir(MODEL_WEIGHTS):
    MODEL_WEIGHTS = "yolo26n.pt"

YOLO_IMGSZ = 256
YOLO_CONF = 0.25
MOTION_THRESH = 0.015
JPEG_QUALITY = 80
CAM2_TRACK_OFFSET = 1000

_PALETTE = [
    (255,  56,  56),
    ( 56, 255,  56),
    ( 56,  56, 255),
    (255, 200,  56),
    (200,  56, 255),
]


def _load_tracker_model(camera_name: str):
    """Load an independent YOLO detector and tracker instance per camera."""
    print(f"[VISION] Loading {camera_name} detector/tracker: {MODEL_WEIGHTS} ...")
    return YOLO(MODEL_WEIGHTS, task="detect")


_cam1_model = _load_tracker_model("CAM1")
_cam2_model = None
_cam2_model_lock = threading.Lock()


def _get_cam2_model():
    """Lazily initialize the Camera 2 tracker on first frame."""
    global _cam2_model
    with _cam2_model_lock:
        if _cam2_model is None:
            _cam2_model = _load_tracker_model("CAM2")
        return _cam2_model


def _normalize_source(source) -> str:
    """Normalize input camera index, video file path, or stream URL."""
    if source is None:
        return ""
    s = str(source).strip().strip('"').strip("'")
    if not s or "YOUR_PHONE_IP" in s:
        return ""
    if s.isdigit():
        return s
    if s.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
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
    """Open and configure VideoCapture for IP stream, video file, or webcam."""
    s = _normalize_source(source)
    if not s:
        return None

    is_video_file = s.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))
    is_ip = not s.isdigit() and not is_video_file

    if is_video_file:
        try:
            cap = cv2.VideoCapture(s)
        except Exception:
            return None
    elif is_ip:
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
        idx = int(s)
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            try:
                if backend is not None:
                    c = cv2.VideoCapture(idx, backend)
                else:
                    c = cv2.VideoCapture(idx)
                if c and c.isOpened():
                    ret, test_frame = c.read()
                    if ret and test_frame is not None:
                        cap = c
                        break
                    else:
                        c.release()
            except Exception:
                continue

    if cap and cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not is_video_file:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                state_mgr=None, entry_face_mgr=None) -> np.ndarray:
    """Draw bounding boxes and ID banners for verified customers."""
    annotated = frame.copy()

    for b in boxes:
        tid = b["track_id"]
        x1, y1, x2, y2 = b["bbox"]

        cid = b.get("cust_id")
        if not cid:
            if entry_face_mgr:
                cid, _, _ = entry_face_mgr.get_display_info(tid)
            elif state_mgr and tid in state_mgr.track_to_customer:
                cid = state_mgr.track_to_customer[tid]
            else:
                cid = None

        customer_id = str(cid or "")
        is_verified = bool(customer_id) and not customer_id.startswith(("TRACK_", "CAM2-", "YOLO"))
        if not is_verified:
            continue

        colour = get_customer_color(customer_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(customer_id, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

        banner_top = max(0, y1 - th - 12)
        banner_bottom = y1
        banner_right = min(annotated.shape[1], x1 + tw + 16)
        cv2.rectangle(annotated, (x1, banner_top), (banner_right, banner_bottom), colour, -1)

        text_y = max(th + 4, banner_bottom - 5)
        cv2.putText(annotated, customer_id, (x1 + 6, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


class CameraGrabber:
    """Background frame capture loop for a single camera source."""

    _SOURCE_TTL: float = 5.0

    def __init__(self, name: str, config_key: str, peer_grabber=None):
        self.name = name
        self.config_key = config_key
        self.peer_grabber = peer_grabber

        self._raw_lock = threading.Lock()
        self._raw_frame = None
        self._new_frame_evt = threading.Event()

        self._src_cache = ""
        self._src_cache_time = 0.0

        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name=f"grab-{name}"
        )

    def start(self):
        """Start the background frame capture thread."""
        self._thread.start()
        print(f"[{self.name}] Grab thread started.")

    def get_frame(self):
        """Return the latest raw frame or shared peer frame."""
        with self._raw_lock:
            if self._raw_frame is not None:
                return self._raw_frame
        if self.peer_grabber is not None and self._get_source() == self.peer_grabber._get_source():
            return self.peer_grabber.get_frame()
        return None

    def _get_source(self) -> str:
        """Fetch camera source setting with TTL cache."""
        now = time.monotonic()
        if now - self._src_cache_time > self._SOURCE_TTL:
            try:
                import config
                self._src_cache = config.reload().get(self.config_key, "")
                self._src_cache_time = now
            except Exception:
                pass
        return self._src_cache

    def _grab_loop(self):
        """Continuously grab frames from camera source."""
        cap = None
        current_src = ""
        last_check = 0.0
        fail_count = 0

        while True:
            now = time.monotonic()

            if now - last_check > self._SOURCE_TTL:
                new_src = self._get_source()
                last_check = now
            else:
                new_src = current_src

            if self.peer_grabber is not None:
                peer_src = self.peer_grabber._get_source()
                if new_src and new_src == peer_src:
                    if cap is not None:
                        cap.release()
                        cap = None
                        current_src = ""
                    peer_frame = self.peer_grabber.get_frame()
                    if peer_frame is not None:
                        with self._raw_lock:
                            self._raw_frame = peer_frame.copy()
                        self._new_frame_evt.set()
                    time.sleep(0.02)
                    continue

            if new_src != current_src or cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = _open_cap(new_src)
                current_src = new_src
                fail_count = 0
                if cap is None or not cap.isOpened():
                    time.sleep(2.0)
                    continue
                print(f"[{self.name}] Camera opened: {new_src or '(none)'}")

            ret, frame = cap.read()
            if ret and frame is not None:
                fail_count = 0
                with self._raw_lock:
                    self._raw_frame = frame
                self._new_frame_evt.set()

                if current_src.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    delay = 1.0 / fps if fps > 0 else 0.033
                    time.sleep(delay)
            else:
                fail_count += 1
                if cap and current_src.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.033)
                else:
                    if fail_count > 10:
                        if cap is not None:
                            cap.release()
                            cap = None
                        fail_count = 0
                        time.sleep(2.0)
                    else:
                        time.sleep(0.033)


class BatchedDetector:
    """Dual-camera detector running YOLO tracking and re-identification."""

    def __init__(self, grabber1: CameraGrabber, grabber2: CameraGrabber,
                 is_entry1: bool = True, state_mgr=None):
        self.grabber1 = grabber1
        self.grabber2 = grabber2
        self.is_entry1 = is_entry1
        self.state_mgr = state_mgr

        self.bg_sub1 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=24, detectShadows=False
        )
        self.bg_sub2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=24, detectShadows=False
        )

        self._det_lock = threading.Lock()
        self._state = {
            "cam1": {"boxes": [], "tracks": {}, "badge": "STATIC | YOLO SKIPPED", "bcolor": (130, 130, 130)},
            "cam2": {"boxes": [], "tracks": {}, "badge": "STATIC | YOLO SKIPPED", "bcolor": (130, 130, 130)},
        }

        self.reid_engine = PersonReIDEngine(sim_threshold=0.75)
        self.entry_face_mgr = EntryFaceManager(state_mgr=self.state_mgr, reid_engine=self.reid_engine)

        self._frame_lock = threading.Lock()
        self._ann_frame1 = self._make_standby_frame("Entry Camera", "Initializing...")
        self._ann_frame2 = self._make_standby_frame("Shelf Camera", "Initializing...")

        self._thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="batched-detector"
        )

    def _make_standby_frame(self, cam_name: str, message: str) -> np.ndarray:
        """Create a stylized standby frame when a camera is offline or initializing."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (18, 24, 38)
        cv2.rectangle(img, (20, 20), (620, 460), (38, 50, 75), 2)
        cv2.circle(img, (45, 230), 6, (255, 120, 56), -1, cv2.LINE_AA)
        title = f"{cam_name}: {message}"
        cv2.putText(img, title, (62, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (241, 245, 249), 2, cv2.LINE_AA)
        cv2.putText(img, "Autonomous Retail Surveillance | Configure in Admin Panel (/admin)", (62, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)
        return img

    def start(self):
        """Start background face manager and detection loop."""
        self.entry_face_mgr.start()
        self._thread.start()
        print("[BATCHED] Detector thread started — single model, motion-gated batch inference.")

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
        """Encode current annotated camera frame to JPEG bytes."""
        with self._frame_lock:
            frame = self._ann_frame1 if cam_id == 1 else self._ann_frame2

        if frame is None:
            cam_title = "ENTRY CAM" if cam_id == 1 else "SHELF CAM"
            frame = self._make_standby_frame(cam_title, "Loading...")

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok or buf is None:
            cam_title = "ENTRY CAM" if cam_id == 1 else "SHELF CAM"
            fallback = self._make_standby_frame(cam_title, "Loading...")
            _, buf = cv2.imencode(".jpg", fallback, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if buf is not None else b""

    def _motion_ratio(self, bg_sub, frame: np.ndarray) -> float:
        """Apply MOG2 on a downscaled gray frame and return changed-pixel ratio."""
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = bg_sub.apply(gray)
        return cv2.countNonZero(mask) / (320.0 * 240.0)

    def _detect_loop(self):
        """Main detection loop processing both camera feeds continuously."""
        cycle = 0
        while True:
            cycle += 1
            try:
                frame1 = self.grabber1.get_frame()
                frame2 = self.grabber2.get_frame()

                s1 = self.grabber1._get_source()
                s2 = self.grabber2._get_source()

                if frame2 is None and frame1 is not None:
                    if s1 == s2 or not s2:
                        frame2 = frame1.copy()
                elif frame1 is None and frame2 is not None:
                    if s1 == s2 or not s1:
                        frame1 = frame2.copy()

                ph1_used = (frame1 is None)
                ph2_used = (frame2 is None)

                if ph1_used:
                    src1 = s1 or "None"
                    frame1 = self._make_standby_frame("Entry Camera", f"Offline ({src1})")
                if ph2_used:
                    src2 = s2 or "None"
                    frame2 = self._make_standby_frame("Shelf Camera", f"Offline ({src2})")

                ratio1 = 0.0 if ph1_used else self._motion_ratio(self.bg_sub1, frame1)
                ratio2 = 0.0 if ph2_used else self._motion_ratio(self.bg_sub2, frame2)
                moving1 = ratio1 >= MOTION_THRESH
                moving2 = ratio2 >= MOTION_THRESH

                res1 = _cam1_model.track(frame1, persist=True, tracker="bytetrack.yaml", classes=[0],
                                         imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False) if not ph1_used else None
                res2 = _get_cam2_model().track(frame2, persist=True, tracker="bytetrack.yaml", classes=[0],
                                         imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False) if not ph2_used else None
                r1 = res1[0] if res1 else None
                r2 = res2[0] if res2 else None

                with self._det_lock:
                    self._state["cam1"].update({
                        "badge": "MOTION | DETECTING" if moving1 else "DETECTING",
                        "bcolor": (56, 255, 56) if moving1 else (130, 130, 130),
                    })
                    self._state["cam2"].update({
                        "badge": "MOTION | DETECTING" if moving2 else "DETECTING",
                        "bcolor": (56, 255, 56) if moving2 else (130, 130, 130),
                    })

                ann1 = self._parse_and_draw(r1, frame1, "ENTRY CAM",
                                             cam_key="cam1", is_entry=self.is_entry1) if not ph1_used else frame1
                ann2 = self._parse_and_draw(r2, frame2, "SHELF CAM",
                                             cam_key="cam2", is_entry=False) if not ph2_used else frame2

                with self._frame_lock:
                    self._ann_frame1 = ann1
                    self._ann_frame2 = ann2
                self.reid_engine.clean_stale_tracks()

            except Exception as e:
                print(f"[BATCHED] detect_loop error (thread kept alive): {e}")
                time.sleep(0.1)

    def _parse_and_draw(self, result, frame: np.ndarray, cam_title: str,
                        cam_key: str, is_entry: bool) -> np.ndarray:
        """Parse tracker output, perform identification/re-ID, and render bounding boxes."""
        boxes_out = []
        tracks_out = {}

        if result is not None and result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes_out.append({
                        "track_id": -1,
                        "bbox": (x1, y1, x2, y2),
                        "pos": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "cust_id": "YOLO",
                        "color": (0, 215, 255),
                        "badge": "NO TRACK ID",
                    })
                    continue
                raw_tid = int(box.id.item())
                tid = raw_tid if is_entry else raw_tid + CAM2_TRACK_OFFSET
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                if is_entry:
                    self.entry_face_mgr.process_track_frame(frame, (x1, y1, x2, y2), tid)
                    disp_cid, color, badge = self.entry_face_mgr.get_display_info(tid)
                else:
                    disp_cid, badge = self.reid_engine.associate_cam2_track(
                        frame, (x1, y1, x2, y2), tid, f"CAM2-T{raw_tid}", self.state_mgr
                    )
                    color = get_customer_color(disp_cid)

                tracks_out[tid] = (cx, cy)
                boxes_out.append({
                    "track_id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "pos": (cx, cy),
                    "cust_id": disp_cid,
                    "color": color,
                    "badge": badge,
                })

        with self._det_lock:
            self._state[cam_key].update({
                "boxes": boxes_out,
                "tracks": tracks_out,
            })

        return _draw_boxes(frame, boxes_out, cam_title, self.state_mgr, self.entry_face_mgr)


class VisionSystem:
    """Facade managing camera grabbers and batched detection pipeline."""

    def __init__(self, state_manager=None):
        self.state_mgr = state_manager

        self.grabber1 = CameraGrabber("Entry Camera", config_key="entry_camera")
        self.grabber2 = CameraGrabber("Shelf Camera", config_key="shelf_camera", peer_grabber=self.grabber1)

        self.detector = BatchedDetector(
            self.grabber1, self.grabber2,
            is_entry1=True,
            state_mgr=state_manager,
        )

        self.entry_stream = self
        self.shelf_stream = self

        self.grabber1.start()
        self.grabber2.start()
        self.detector.start()

        print("[VISION] Pipeline LIVE — single batched YOLO model, motion-gated per-cam.")

    def entry_get_jpeg(self):
        return self.detector.get_jpeg1()

    def shelf_get_jpeg(self):
        return self.detector.get_jpeg2()

    def get_jpeg(self, cam_id: int = 1):
        return self.detector.get_jpeg1() if cam_id == 1 else self.detector.get_jpeg2()

    def process_entry_camera(self, frame: np.ndarray) -> np.ndarray:
        """Synchronously process a single entry camera frame."""
        results = _cam1_model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False,
        )
        return self.detector._parse_and_draw(
            results[0] if results else None, frame, "ENTRY CAM", cam_key="cam1", is_entry=True
        )

    def process_shelf_camera(self, frame: np.ndarray) -> np.ndarray:
        """Synchronously process a single shelf camera frame."""
        results = _get_cam2_model().track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False,
        )
        return self.detector._parse_and_draw(
            results[0] if results else None, frame, "SHELF CAM", cam_key="cam2", is_entry=False
        )

    def get_latest_shelf_tracks(self) -> dict:
        """Return the latest tracked customer positions on Camera 2."""
        return self.detector.get_tracks2()
