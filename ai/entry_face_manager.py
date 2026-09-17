"""Camera 1 (Entry) face recognition, quality verification, and customer identification."""

import cv2
import time
import queue
import logging
import threading
import numpy as np
from typing import Dict, Optional, Tuple, Any

from config import (
    FACE_MATCH_HIGH_THRESHOLD,
    FACE_MATCH_MEDIUM_THRESHOLD,
    FACE_MATCH_MAX_ATTEMPTS,
)
from services.face_service import process_entry_face_crop
from services.customer_service import identify_customer_by_face_embedding

logger = logging.getLogger(__name__)

CUSTOMER_BOX_COLOR = (0, 220, 0)


def get_customer_color(cid: str, uniform: bool = False) -> Tuple[int, int, int]:
    """Return consistent high-visibility color for customer overlays."""
    return CUSTOMER_BOX_COLOR


class FaceMatchState:
    UNASSIGNED = "UNASSIGNED"
    COLLECTING = "COLLECTING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class TrackProfile:
    """Evaluation state and candidate identities for a tracked person."""

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.state: str = FaceMatchState.UNASSIGNED
        self.customer_id: Optional[str] = None
        self.customer_name: Optional[str] = None
        self.similarity: float = 0.0
        self.attempts: int = 0
        self.candidate_matches: list = []
        self.last_attempt_time: float = 0.0
        self.last_seen_time: float = time.monotonic()
        self.quality_status: str = "Scanning..."
        self.in_flight: bool = False

    @property
    def is_resolved(self) -> bool:
        """Return True if customer identification is finalized."""
        return self.state == FaceMatchState.ACCEPTED

    def get_color(self) -> Tuple[int, int, int]:
        display_id = self.customer_id or f"TRACK_{self.track_id}"
        return get_customer_color(display_id)


class EntryFaceManager:
    """Manages face detection, quality filtering, and customer identification for Camera 1."""

    def __init__(self, state_mgr=None, reid_engine=None):
        self.state_mgr = state_mgr
        self.reid_engine = reid_engine

        self.high_threshold: float = FACE_MATCH_HIGH_THRESHOLD
        self.medium_threshold: float = FACE_MATCH_MEDIUM_THRESHOLD
        self.max_attempts: int = FACE_MATCH_MAX_ATTEMPTS
        self.attempt_interval: float = 0.50

        self._lock = threading.Lock()
        self._profiles: Dict[int, TrackProfile] = {}

        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="entry-face-worker"
        )
        self._running: bool = False

    def start(self):
        """Start the background face recognition worker thread."""
        self._running = True
        self._worker_thread.start()
        logger.info("[ENTRY FACE] Manager started with 3-state confidence matching.")
        print(f"[ENTRY FACE] Engine active (HIGH >= {self.high_threshold}, MEDIUM >= {self.medium_threshold}, MAX_ATTEMPTS = {self.max_attempts})")

    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    def get_profile(self, track_id: int) -> Optional[TrackProfile]:
        """Retrieve a track profile by track ID."""
        with self._lock:
            return self._profiles.get(track_id)

    def get_customer_id(self, track_id: int) -> str:
        """Return resolved customer ID or fallback tracker ID."""
        with self._lock:
            prof = self._profiles.get(track_id)
            if prof and prof.customer_id:
                return prof.customer_id
        if self.state_mgr:
            cid = self.state_mgr.track_to_customer.get(track_id)
            if cid:
                return cid
        return f"TRACK_{track_id}"

    def get_display_info(self, track_id: int) -> Tuple[str, Tuple[int, int, int], str]:
        """Return (label, color, badge) tuple for bounding box rendering."""
        with self._lock:
            prof = self._profiles.get(track_id)

        if prof is None:
            cid = f"TRACK_{track_id}"
            return cid, get_customer_color(cid), "SCANNING"

        prof.last_seen_time = time.monotonic()

        if prof.customer_id and not prof.customer_id.startswith("TRACK_"):
            cid = prof.customer_id
            return cid, get_customer_color(cid), ""

        if prof.state == FaceMatchState.REJECTED:
            status = "UNREGISTERED"
        elif prof.state == FaceMatchState.COLLECTING:
            status = "VERIFYING"
        else:
            status = "SCANNING"
        cid = prof.customer_id or f"TRACK_{track_id}"
        return cid, get_customer_color(cid), status

    def process_track_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int
    ) -> str:
        """Queue candidate face crops from tracked detections for evaluation."""
        now = time.monotonic()

        with self._lock:
            if track_id not in self._profiles:
                self._profiles[track_id] = TrackProfile(track_id)
            prof = self._profiles[track_id]
            prof.last_seen_time = now

            if prof.is_resolved or prof.state == FaceMatchState.REJECTED:
                return prof.customer_id or f"TRACK_{track_id}"

            if prof.in_flight or (now - prof.last_attempt_time < self.attempt_interval):
                return prof.customer_id or f"TRACK_{track_id}"

            prof.in_flight = True
            prof.last_attempt_time = now

        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        pw, ph = x2 - x1, y2 - y1
        pad_x = int(pw * 0.10)
        pad_y = int(ph * 0.10)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(fw, x2 + pad_x)
        py2 = min(fh, y2 + pad_y)

        crop = frame[py1:py2, px1:px2].copy()

        try:
            self._queue.put_nowait((track_id, frame.copy(), crop, bbox))
        except queue.Full:
            with self._lock:
                prof.in_flight = False

        return prof.customer_id or f"TRACK_{track_id}"

    def clean_stale_tracks(self, max_age_seconds: float = 15.0):
        """Remove inactive profiles exceeding the maximum age."""
        now = time.monotonic()
        with self._lock:
            stale = [
                tid for tid, p in self._profiles.items()
                if (now - p.last_seen_time > max_age_seconds)
            ]
            for tid in stale:
                del self._profiles[tid]

    def _worker_loop(self):
        """Background worker thread processing face evaluation queue."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break

                track_id, frame, crop, bbox = item
                try:
                    self._evaluate_face_job(track_id, frame, crop, bbox)
                except Exception as exc:
                    logger.error(f"[ENTRY FACE] Worker error for track {track_id}: {exc}", exc_info=True)
                finally:
                    with self._lock:
                        if track_id in self._profiles:
                            self._profiles[track_id].in_flight = False
                    self._queue.task_done()

            except queue.Empty:
                self.clean_stale_tracks()
                continue
            except Exception as e:
                logger.error(f"[ENTRY FACE] Unexpected worker exception: {e}")

    def _evaluate_face_job(self, track_id: int, frame: np.ndarray, crop: np.ndarray, bbox: tuple):
        """Evaluate face quality, extract embedding, and match against customer database."""
        with self._lock:
            prof = self._profiles.get(track_id)
            if not prof or prof.is_resolved:
                return

        from services.face_service import get_face_app, validate_face_quality_crop, extract_face_embedding_from_crop
        app = get_face_app()
        faces = app.get(frame)

        x1, y1, x2, y2 = bbox
        matched_face = None
        if faces:
            candidates = []
            for f in faces:
                fcx = (f.bbox[0] + f.bbox[2]) / 2.0
                fcy = (f.bbox[1] + f.bbox[3]) / 2.0
                if (x1 - 40 <= fcx <= x2 + 40) and (y1 - 40 <= fcy <= y2 + 40):
                    dist = abs(fcx - (x1 + x2) / 2.0) + abs(fcy - (y1 + y2) / 2.0)
                    candidates.append((dist, f))
            if candidates:
                candidates.sort(key=lambda c: c[0])
                matched_face = candidates[0][1]

        if matched_face is not None:
            ok, reject_reason, metrics = validate_face_quality_crop(matched_face, frame)
            embedding = extract_face_embedding_from_crop(frame, matched_face) if ok else None
        else:
            ok, embedding, reject_reason, metrics = process_entry_face_crop(crop)

        if not ok or embedding is None:
            with self._lock:
                prof.quality_status = reject_reason or "Face check failed"
                prof.attempts += 1
                prof.state = FaceMatchState.COLLECTING
            return

        match = identify_customer_by_face_embedding(embedding)
        sim = match["similarity"] if match else 0.0
        cid = match["customer_id"] if match else None
        name = match.get("name") if match else ""

        with self._lock:
            if cid and sim >= self.medium_threshold:
                prof.customer_id = cid
                prof.customer_name = name
                prof.similarity = max(prof.similarity, sim)
                prof.attempts += 1
                prof.candidate_matches.append((cid, sim))

                candidates = [c for c, _ in prof.candidate_matches]
                vote_count = candidates.count(cid)

                if sim >= self.high_threshold or vote_count >= 2 or prof.attempts >= 3:
                    prof.state = FaceMatchState.ACCEPTED
                    prof.quality_status = f"Identified ({prof.similarity:.2f})"
                    print(f"[ENTRY FACE ACCEPT] Track {track_id} identified as {cid} ({name}) with confidence {sim:.3f}")

                    if self.state_mgr:
                        self.state_mgr.register_customer(cid)
                        self.state_mgr.link_track_to_customer(track_id, cid)

                    if self.reid_engine:
                        try:
                            body_feat = self.reid_engine.extract_embedding(crop)
                            if body_feat is not None:
                                self.reid_engine.register_gallery_feature(cid, body_feat, crop)
                            self.reid_engine.track_to_cid[track_id] = cid
                        except Exception as reid_exc:
                            logger.warning(f"Failed to register body ReID for {cid}: {reid_exc}")
                else:
                    prof.state = FaceMatchState.COLLECTING
                    prof.quality_status = f"Verifying {cid} ({sim:.2f})"
                    print(f"[ENTRY FACE MEDIUM] Track {track_id} candidate {cid} (sim: {sim:.3f}, attempt {prof.attempts}/{self.max_attempts})")

            else:
                prof.attempts += 1
                prof.similarity = sim
                if not prof.customer_id:
                    prof.customer_id = f"TRACK_{track_id}"
                prof.state = FaceMatchState.COLLECTING
                prof.quality_status = f"Scanning ({sim:.2f})"
