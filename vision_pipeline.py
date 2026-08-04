"""
vision_pipeline.py
------------------
VisionSystem – wraps YOLOv8 detection (entry camera) and YOLOv8-Pose
(shelf camera) to feed the GlobalStateManager with person tracks.

Entry camera  → person detection + ByteTrack → register / deregister customers
Shelf camera  → pose estimation + ByteTrack  → expose wrist keypoint positions
                so the sensor-bridge can attribute weight events to the nearest person.
"""

import cv2
from ultralytics import YOLO

# Colour palette for track bounding-boxes
_PALETTE = [
    (255, 56,  56),
    (56,  255, 56),
    (56,  56,  255),
    (255, 200, 56),
    (200, 56,  255),
]


class VisionSystem:
    def __init__(self, state_manager):
        self.state_mgr = state_manager

        # Load models (weights are already present in the project root)
        print("[VISION] Loading YOLOv8n detection model …")
        self.det_model = YOLO("yolov8n.pt")

        print("[VISION] Loading YOLOv8n-Pose model …")
        self.pose_model = YOLO("yolov8n-pose.pt")

        # Latest shelf tracks: {track_id: (centre_x, centre_y)}
        self._shelf_tracks: dict[int, tuple[float, float]] = {}

        print("[VISION] VisionSystem ready.")

    # ------------------------------------------------------------------
    # Entry / Exit camera  (person detection + tracking → customer ReID)
    # ------------------------------------------------------------------
    def process_entry_camera(self, frame):
        """
        Runs YOLOv8 person detection with ByteTrack on the entry-camera frame.
        Registers new track IDs as customers and annotates the frame.
        Returns the annotated frame.
        """
        results = self.det_model.track(
            frame,
            persist=True,
            classes=[0],          # class 0 = person
            conf=0.4,
            verbose=False,
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                colour = _PALETTE[track_id % len(_PALETTE)]

                # Auto-register customer on first sight
                if track_id not in self.state_mgr.track_to_customer:
                    cust_id = f"CUST_{track_id}"
                    self.state_mgr.register_customer(cust_id)
                    self.state_mgr.link_track_to_customer(track_id, cust_id)

                cust_id = self.state_mgr.track_to_customer.get(track_id, f"T{track_id}")

                # Draw bounding box + label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                label = f"{cust_id} | ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return annotated

    # ------------------------------------------------------------------
    # Shelf camera  (pose estimation → wrist proximity for weight events)
    # ------------------------------------------------------------------
    def process_shelf_camera(self, frame):
        """
        Runs YOLOv8-Pose with ByteTrack on the shelf-camera frame.
        Updates internal shelf track positions (body centre) and annotates
        the frame with skeletons + track IDs.
        Returns the annotated frame.
        """
        results = self.pose_model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.4,
            verbose=False,
        )

        annotated = frame.copy()
        new_tracks: dict[int, tuple[float, float]] = {}

        if results and results[0].boxes is not None:
            boxes   = results[0].boxes
            kpss    = results[0].keypoints  # shape: (N, 17, 3)

            for i, box in enumerate(boxes):
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                new_tracks[track_id] = (cx, cy)

                colour = _PALETTE[track_id % len(_PALETTE)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                label = f"Track {track_id}"
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

                # Draw keypoints if available
                if kpss is not None and i < len(kpss.xy):
                    for kx, ky in kpss.xy[i].tolist():
                        if kx > 0 and ky > 0:
                            cv2.circle(annotated, (int(kx), int(ky)), 3, colour, -1)

        self._shelf_tracks = new_tracks
        return annotated

    # ------------------------------------------------------------------
    # Public accessor used by app.py / main.py
    # ------------------------------------------------------------------
    def get_latest_shelf_tracks(self) -> dict[int, tuple[float, float]]:
        """Returns {track_id: (centre_x, centre_y)} for the shelf camera."""
        return dict(self._shelf_tracks)
