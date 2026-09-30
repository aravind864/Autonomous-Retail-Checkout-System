"""YOLO11n-Pose based shelf interaction detection.

Pose inference is triggered ONLY when:
1. Camera 2 has a valid person track
2. Track is associated with a customer/session OR is a candidate
3. Person's bbox overlaps the shelf interaction ROI
4. Pose inference cooldown allows it

The pose stage provides evidence of customer-shelf interaction.
It does NOT add products to the cart — that is the weight event's job.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# YOLO-Pose keypoint indices (COCO format)
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# Interaction statuses
STATUS_NOT_IN_SHELF = "NOT_IN_SHELF"
STATUS_APPROACHING = "APPROACHING"
STATUS_INTERACTING = "INTERACTING"
STATUS_LEAVING = "LEAVING"


class PoseInteractionState:
    """Per-track pose interaction state."""

    def __init__(self, cam2_track_id: int):
        self.cam2_track_id = cam2_track_id
        self.customer_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.inside_shelf_roi: bool = False
        self.last_pose_timestamp: float = 0.0
        self.pose_confidence: float = 0.0
        self.keypoints: Optional[np.ndarray] = None
        self.left_wrist: Optional[Tuple[float, float]] = None
        self.right_wrist: Optional[Tuple[float, float]] = None
        self.left_elbow: Optional[Tuple[float, float]] = None
        self.right_elbow: Optional[Tuple[float, float]] = None
        self.left_shoulder: Optional[Tuple[float, float]] = None
        self.right_shoulder: Optional[Tuple[float, float]] = None
        self.interaction_score: float = 0.0
        self.last_interaction_time: float = 0.0
        self.interaction_status: str = STATUS_NOT_IN_SHELF
        self._prev_wrist_positions: list = []
        self._roi_enter_time: float = 0.0
        self._roi_dwell_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for state manager storage."""
        return {
            "cam2_track_id": self.cam2_track_id,
            "customer_id": self.customer_id,
            "session_id": self.session_id,
            "inside_shelf_roi": self.inside_shelf_roi,
            "interaction_status": self.interaction_status,
            "interaction_score": self.interaction_score,
            "last_interaction_time": self.last_interaction_time,
            "pose_confidence": self.pose_confidence,
            "last_pose_timestamp": self.last_pose_timestamp,
        }


class PoseInteractionManager:
    """Manages YOLO11n-Pose model and per-track interaction states.

    The pose model is loaded lazily on first use to avoid startup delay
    if no camera is connected.
    """

    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()
        self._states: Dict[int, PoseInteractionState] = {}
        self._states_lock = threading.Lock()

    def _get_model(self):
        """Lazy-load YOLO11n-Pose model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from ultralytics import YOLO
                        model_path = config.POSE_MODEL_WEIGHTS
                        logger.info(f"[POSE] Loading pose model: {model_path}")
                        self._model = YOLO(model_path, task="pose")
                        logger.info("[POSE] Pose model loaded successfully")
                    except Exception as e:
                        logger.error(f"[POSE] Failed to load pose model: {e}")
                        return None
        return self._model

    def should_run_pose(self, cam2_track_id: int, inside_roi: bool) -> bool:
        """Determine if pose inference should run for this track right now."""
        if not inside_roi:
            return False

        with self._states_lock:
            state = self._states.get(cam2_track_id)
            if state is None:
                return True  # First time — run it
            elapsed = time.monotonic() - state.last_pose_timestamp
            return elapsed >= config.POSE_INFERENCE_INTERVAL

    def run_pose_inference(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           cam2_track_id: int, customer_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> Optional[PoseInteractionState]:
        """Run YOLO11n-Pose on a cropped region and update interaction state."""
        model = self._get_model()
        if model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Pad the crop for better pose estimation
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.10)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 30:
            return None

        try:
            results = model(
                crop,
                conf=config.POSE_CONF,
                imgsz=config.POSE_IMGSZ,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"[POSE] Inference error for track {cam2_track_id}: {e}")
            return None

        if not results or results[0].keypoints is None:
            return None

        result = results[0]
        if result.keypoints.data.shape[0] == 0:
            return None

        # Use the person with largest bbox if multiple detected
        kps_data = result.keypoints.data
        if kps_data.shape[0] > 1 and result.boxes is not None:
            areas = []
            for box in result.boxes:
                bx = box.xyxy[0].tolist()
                areas.append((bx[2] - bx[0]) * (bx[3] - bx[1]))
            best_idx = max(range(len(areas)), key=lambda i: areas[i])
            keypoints = kps_data[best_idx].cpu().numpy()
        else:
            keypoints = kps_data[0].cpu().numpy()

        # Map keypoints back to full-frame coordinates
        offset_x, offset_y = cx1, cy1

        now = time.monotonic()

        with self._states_lock:
            state = self._states.get(cam2_track_id)
            if state is None:
                state = PoseInteractionState(cam2_track_id)
                self._states[cam2_track_id] = state

            state.customer_id = customer_id
            state.session_id = session_id
            state.inside_shelf_roi = True
            state.last_pose_timestamp = now
            state.keypoints = keypoints

            # Extract key body points (x, y, confidence)
            def _get_kp(idx):
                if idx < keypoints.shape[0]:
                    kp = keypoints[idx]
                    if len(kp) >= 3 and kp[2] > 0.3:
                        return (float(kp[0] + offset_x), float(kp[1] + offset_y))
                return None

            state.left_shoulder = _get_kp(KP_LEFT_SHOULDER)
            state.right_shoulder = _get_kp(KP_RIGHT_SHOULDER)
            state.left_elbow = _get_kp(KP_LEFT_ELBOW)
            state.right_elbow = _get_kp(KP_RIGHT_ELBOW)
            state.left_wrist = _get_kp(KP_LEFT_WRIST)
            state.right_wrist = _get_kp(KP_RIGHT_WRIST)

            # Compute lightweight interaction score
            score = self._compute_interaction_score(state, bbox)
            state.interaction_score = score
            state.pose_confidence = float(np.mean(keypoints[:, 2])) if keypoints.shape[1] >= 3 else 0.0

            # Update interaction status (state machine)
            prev_status = state.interaction_status
            state.interaction_status = self._determine_status(state, score)

            if state.interaction_status in (STATUS_INTERACTING, STATUS_APPROACHING):
                state.last_interaction_time = time.time()

            if state.interaction_status != prev_status:
                logger.info(
                    f"[POSE] Track {cam2_track_id}: {prev_status} -> {state.interaction_status} "
                    f"(score={score:.2f}, customer={customer_id})"
                )

            return state

    def _compute_interaction_score(self, state: PoseInteractionState,
                                    bbox: Tuple[int, int, int, int]) -> float:
        """Compute lightweight interaction evidence score from keypoints.

        Factors:
        - Wrist near/inside shelf ROI region
        - Wrist below shoulder (reaching gesture)
        - Stable presence in ROI (dwell time)
        - Body centered in ROI
        """
        score = 0.0
        shelf_roi = config.SHELF_ROI
        rx1, ry1, rx2, ry2 = shelf_roi

        # --- Factor 1: Wrist inside/near shelf ROI (0.0 - 0.35) ---
        wrist_roi_score = 0.0
        for wrist in [state.left_wrist, state.right_wrist]:
            if wrist:
                wx, wy = wrist
                if rx1 <= wx <= rx2 and ry1 <= wy <= ry2:
                    wrist_roi_score = max(wrist_roi_score, 1.0)
                else:
                    rcx = (rx1 + rx2) / 2
                    rcy = (ry1 + ry2) / 2
                    max_dist = ((rx2 - rx1) + (ry2 - ry1)) / 2
                    dist = ((wx - rcx) ** 2 + (wy - rcy) ** 2) ** 0.5
                    proximity = max(0, 1.0 - dist / max(1, max_dist))
                    wrist_roi_score = max(wrist_roi_score, proximity * 0.5)
        score += 0.35 * wrist_roi_score

        # --- Factor 2: Wrist below/near shoulder height (reaching) (0.0 - 0.25) ---
        reach_score = 0.0
        for wrist, shoulder in [(state.left_wrist, state.left_shoulder),
                                 (state.right_wrist, state.right_shoulder)]:
            if wrist and shoulder:
                wy = wrist[1]
                sy = shoulder[1]
                if wy >= sy - 20:
                    reach_score = max(reach_score, 0.8)
                arm_extension = abs(wrist[0] - shoulder[0])
                body_h = max(1, bbox[3] - bbox[1])
                if arm_extension > body_h * 0.15:
                    reach_score = max(reach_score, 0.6)
        score += 0.25 * reach_score

        # --- Factor 3: Dwell time in ROI (0.0 - 0.25) ---
        now = time.monotonic()
        if state._roi_enter_time == 0:
            state._roi_enter_time = now
        state._roi_dwell_time = now - state._roi_enter_time
        dwell_score = min(1.0, state._roi_dwell_time / 3.0)
        score += 0.25 * dwell_score

        # --- Factor 4: Body stable in ROI (0.0 - 0.15) ---
        body_cx = (bbox[0] + bbox[2]) / 2
        roi_cx = (rx1 + rx2) / 2
        centering = max(0, 1.0 - abs(body_cx - roi_cx) / max(1, (rx2 - rx1) / 2))
        score += 0.15 * centering

        return min(1.0, score)

    def _determine_status(self, state: PoseInteractionState, score: float) -> str:
        """Simple state machine for interaction status."""
        if not state.inside_shelf_roi:
            state._roi_enter_time = 0
            state._roi_dwell_time = 0
            return STATUS_NOT_IN_SHELF

        if score >= 0.55:
            return STATUS_INTERACTING
        elif score >= 0.25:
            return STATUS_APPROACHING
        elif state.interaction_status == STATUS_INTERACTING:
            return STATUS_LEAVING
        else:
            return STATUS_APPROACHING

    def update_roi_status(self, cam2_track_id: int, inside_roi: bool,
                          customer_id: Optional[str] = None,
                          session_id: Optional[str] = None):
        """Update ROI status for tracks that don't get pose inference."""
        with self._states_lock:
            state = self._states.get(cam2_track_id)
            if state is None:
                state = PoseInteractionState(cam2_track_id)
                self._states[cam2_track_id] = state

            prev_in_roi = state.inside_shelf_roi
            state.inside_shelf_roi = inside_roi
            state.customer_id = customer_id
            state.session_id = session_id

            if inside_roi and not prev_in_roi:
                state._roi_enter_time = time.monotonic()
                state.interaction_status = STATUS_APPROACHING
                logger.info(f"[POSE] ENTERED SHELF ROI: track {cam2_track_id} customer={customer_id}")
            elif not inside_roi and prev_in_roi:
                state.interaction_status = STATUS_LEAVING
                state._roi_enter_time = 0
                state._roi_dwell_time = 0
                logger.info(f"[POSE] LEFT SHELF ROI: track {cam2_track_id} customer={customer_id}")

    def get_state(self, cam2_track_id: int) -> Optional[PoseInteractionState]:
        """Retrieve pose interaction state for a track."""
        with self._states_lock:
            return self._states.get(cam2_track_id)

    def get_display_info(self, cam2_track_id: int) -> Dict[str, str]:
        """Return display-friendly status strings for camera overlay."""
        with self._states_lock:
            state = self._states.get(cam2_track_id)
            if state is None:
                return {"roi_status": "", "pose_status": "", "interaction_status": ""}

            roi_str = "SHELF" if state.inside_shelf_roi else ""
            pose_str = "POSE ACTIVE" if (time.monotonic() - state.last_pose_timestamp) < 1.0 else ""
            interaction_str = state.interaction_status if state.interaction_status != STATUS_NOT_IN_SHELF else ""

            return {
                "roi_status": roi_str,
                "pose_status": pose_str,
                "interaction_status": interaction_str,
            }

    def sync_to_state_manager(self, cam2_track_id: int, state_mgr):
        """Push current pose state to the global state manager."""
        with self._states_lock:
            state = self._states.get(cam2_track_id)
            if state and state_mgr:
                state_mgr.set_pose_state(cam2_track_id, state.to_dict())

    def clean_stale(self, max_age: float = 30.0):
        """Remove pose states for tracks not updated recently."""
        now = time.monotonic()
        with self._states_lock:
            stale = [
                tid for tid, s in self._states.items()
                if now - s.last_pose_timestamp > max_age and s.last_pose_timestamp > 0
            ]
            for tid in stale:
                del self._states[tid]
