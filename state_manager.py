"""Thread-safe global state manager for active customers, sessions, and track associations."""

import logging
import time
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GlobalStateManager:
    """Thread-safe tracker for active store customers, sessions, and ByteTrack associations.

    Maintains in-memory mappings for fast lookup during the vision pipeline loop.
    Database persistence of sessions is handled by the session service layer.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # --- Existing attributes (preserved for backward compatibility) ---
        self.active_customers: set = set()
        self.track_to_customer: dict = {}

        # --- Session management ---
        self._customer_to_session: Dict[str, str] = {}
        self._cam1_track_to_customer: Dict[int, str] = {}
        self._cam1_track_to_session: Dict[int, str] = {}
        self._cam2_track_to_customer: Dict[int, str] = {}
        self._cam2_track_to_session: Dict[int, str] = {}

        # --- Timestamps ---
        self._customer_last_seen: Dict[str, float] = {}
        self._track_last_seen: Dict[int, float] = {}

        # --- Pose interaction state per Camera 2 track ---
        self._pose_states: Dict[int, Dict[str, Any]] = {}

    # ===================================================================
    # Existing public methods — PRESERVED, signatures unchanged
    # ===================================================================

    def register_customer(self, customer_id: str):
        """Register a customer as active in the store."""
        with self._lock:
            is_new = customer_id not in self.active_customers
            self.active_customers.add(customer_id)
            self._customer_last_seen[customer_id] = time.monotonic()
            if is_new:
                logger.info(f"[STATE] Customer {customer_id} registered.")
                print(f"[STATE] Customer {customer_id} registered.")

    def link_track_to_customer(self, track_id: int, customer_id: str):
        """Link a ByteTrack track ID to a customer ID."""
        with self._lock:
            self.track_to_customer[track_id] = customer_id
            self._track_last_seen[track_id] = time.monotonic()

    def get_active_customers(self) -> list:
        """Return list of currently active customer IDs."""
        with self._lock:
            return list(self.active_customers)

    def get_customer_for_track(self, track_id: int):
        """Look up customer ID for a given track ID."""
        with self._lock:
            return self.track_to_customer.get(track_id)

    # ===================================================================
    # Session management — NEW
    # ===================================================================

    def register_session(self, customer_id: str, session_id: str,
                         camera1_track_id: Optional[int] = None):
        """Register an active shopping session for a customer."""
        with self._lock:
            self._customer_to_session[customer_id] = session_id
            if camera1_track_id is not None:
                self._cam1_track_to_customer[camera1_track_id] = customer_id
                self._cam1_track_to_session[camera1_track_id] = session_id
            logger.info(f"[STATE] SESSION CREATED: {session_id} for customer {customer_id}")

    def get_session_for_customer(self, customer_id: str) -> Optional[str]:
        """Return the active session_id for a customer, or None."""
        with self._lock:
            return self._customer_to_session.get(customer_id)

    def get_customer_for_cam1_track(self, track_id: int) -> Optional[str]:
        """Return customer_id linked to a Camera 1 track."""
        with self._lock:
            return self._cam1_track_to_customer.get(track_id)

    def get_session_for_cam1_track(self, track_id: int) -> Optional[str]:
        """Return session_id linked to a Camera 1 track."""
        with self._lock:
            return self._cam1_track_to_session.get(track_id)

    # --- Camera 2 track <-> session linkage ---

    def link_cam2_track(self, cam2_track_id: int, customer_id: str, session_id: str):
        """Associate a Camera 2 track with a customer and session."""
        with self._lock:
            self._cam2_track_to_customer[cam2_track_id] = customer_id
            self._cam2_track_to_session[cam2_track_id] = session_id
            self._track_last_seen[cam2_track_id] = time.monotonic()
            logger.info(f"[STATE] CAM2 TRACK {cam2_track_id} linked to {customer_id} / {session_id}")

    def get_customer_for_cam2_track(self, cam2_track_id: int) -> Optional[str]:
        """Return customer_id for a Camera 2 track."""
        with self._lock:
            return self._cam2_track_to_customer.get(cam2_track_id)

    def get_session_for_cam2_track(self, cam2_track_id: int) -> Optional[str]:
        """Return session_id for a Camera 2 track."""
        with self._lock:
            return self._cam2_track_to_session.get(cam2_track_id)

    def get_all_active_sessions(self) -> Dict[str, str]:
        """Return dict of customer_id -> session_id for all active sessions."""
        with self._lock:
            return dict(self._customer_to_session)

    def get_all_cam2_tracks(self) -> Dict[int, str]:
        """Return dict of cam2_track_id -> customer_id for all active cam2 tracks."""
        with self._lock:
            return dict(self._cam2_track_to_customer)

    # ===================================================================
    # Pose interaction state — NEW
    # ===================================================================

    def set_pose_state(self, cam2_track_id: int, state: Dict[str, Any]):
        """Store pose interaction state for a Camera 2 track."""
        with self._lock:
            self._pose_states[cam2_track_id] = state

    def get_pose_state(self, cam2_track_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve pose interaction state for a Camera 2 track."""
        with self._lock:
            return self._pose_states.get(cam2_track_id)

    def get_all_pose_states(self) -> Dict[int, Dict[str, Any]]:
        """Return all pose states (for customer association during weight events)."""
        with self._lock:
            return dict(self._pose_states)

    def get_interacting_customers_near_shelf(self) -> list:
        """Return list of dicts for customers currently interacting near shelf.

        Each dict contains: customer_id, session_id, cam2_track_id,
        interaction_status, last_interaction_time.
        """
        with self._lock:
            result = []
            for track_id, pose in self._pose_states.items():
                if pose.get("interaction_status") in ("INTERACTING", "APPROACHING"):
                    cid = self._cam2_track_to_customer.get(track_id)
                    sid = self._cam2_track_to_session.get(track_id)
                    if cid and sid:
                        result.append({
                            "customer_id": cid,
                            "session_id": sid,
                            "cam2_track_id": track_id,
                            "interaction_status": pose.get("interaction_status"),
                            "last_interaction_time": pose.get("last_interaction_time", 0),
                            "interaction_score": pose.get("interaction_score", 0),
                        })
            return result

    # ===================================================================
    # Session completion / exit
    # ===================================================================

    def remove_session(self, customer_id: str):
        """Remove an active session and its track associations."""
        with self._lock:
            session_id = self._customer_to_session.pop(customer_id, None)
            self.active_customers.discard(customer_id)
            # Clean cam1 track refs
            for tid in [t for t, c in self._cam1_track_to_customer.items() if c == customer_id]:
                self._cam1_track_to_customer.pop(tid, None)
                self._cam1_track_to_session.pop(tid, None)
            # Clean cam2 track refs
            for tid in [t for t, c in self._cam2_track_to_customer.items() if c == customer_id]:
                self._cam2_track_to_customer.pop(tid, None)
                self._cam2_track_to_session.pop(tid, None)
                self._pose_states.pop(tid, None)
            # Clean track_to_customer
            for tid in [t for t, c in self.track_to_customer.items() if c == customer_id]:
                self.track_to_customer.pop(tid, None)
            if session_id:
                logger.info(f"[STATE] Session {session_id} removed for customer {customer_id}")

    def update_last_seen(self, customer_id: str = None, track_id: int = None):
        """Update last-seen timestamp for a customer or track."""
        now = time.monotonic()
        with self._lock:
            if customer_id:
                self._customer_last_seen[customer_id] = now
            if track_id is not None:
                self._track_last_seen[track_id] = now