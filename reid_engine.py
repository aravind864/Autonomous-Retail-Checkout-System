"""Cross-camera person re-identification — matches Camera 2 tracks to
Camera 1 face-verified customers via body appearance + shape features."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T


class PersonReIDEngine:
    """Aggregate body evidence and associate Camera 2 tracks with Camera 1."""
    W_APPEARANCE, W_SHAPE, W_TIME, W_ROUTE = 0.40, 0.30, 0.15, 0.15
    MATCH_THRESHOLD, UNCERTAIN_THRESHOLD = 0.75, 0.50

    def __init__(self, sim_threshold: float = MATCH_THRESHOLD):
        self.sim_threshold = sim_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_samples = 4
        self.max_samples = 12
        self.min_travel_seconds = 0.0
        self.max_travel_seconds = 300.0
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._build_appearance_model()
        self.gallery: dict[str, dict] = {}
        self.track_to_cid: dict[int, str] = {}
        self._samples: dict[int, dict] = defaultdict(self._new_sample_buffer)

    def _build_appearance_model(self) -> None:
        """Load OSNet if available, otherwise fall back to MobileNetV3."""
        try:
            import torchreid
            name = "osnet_x0_25"
            self.feature_extractor = torchreid.models.build_model(
                name=name, num_classes=1000, loss="softmax", pretrained=True
            ).to(self.device).eval()
            self.pool = None
            print(f"[REID] Using OSNet appearance model: {name}")
            return
        except Exception as exc:
            print(f"[REID] OSNet unavailable ({exc}); using MobileNetV3 appearance fallback.")
        try:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception as exc:
            print(f"[REID] Pretrained MobileNet unavailable ({exc}); using local fallback.")
            backbone = models.mobilenet_v3_small(weights=None)
        self.feature_extractor = backbone.features.to(self.device).eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _new_sample_buffer() -> dict:
        return {"appearance": [], "shape": [], "last_seen": time.monotonic()}

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 12:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            feat = self.feature_extractor(tensor)
            if self.pool is not None:
                feat = self.pool(feat)
            feat = feat.flatten(1)
            feat = feat / (torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-6)
            return feat.cpu().numpy().flatten()
        except Exception as exc:
            print(f"[REID] Appearance extraction failed: {exc}")
            return None

    @staticmethod
    def _extract_shape_feature(crop: np.ndarray) -> np.ndarray:
        """Aspect ratio + eight-band edge silhouette profile."""
        if crop is None or crop.size == 0:
            return np.ones(9, dtype=np.float32) / 3.0
        edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 120)
        widths = []
        for band in np.array_split(edges, 8, axis=0):
            cols = np.where(np.any(band > 0, axis=0))[0]
            widths.append((cols[-1] - cols[0] + 1) / max(1, crop.shape[1]) if cols.size else 0.0)
        descriptor = np.asarray([crop.shape[1] / max(1.0, float(crop.shape[0])), *widths], dtype=np.float32)
        return descriptor / (np.linalg.norm(descriptor) + 1e-6)

    def register_gallery_feature(self, cid: str, emb: np.ndarray, crop: np.ndarray) -> None:
        """Register a Camera 1 customer as a Camera 2 re-ID candidate."""
        if not cid or cid.startswith("TRACK_") or emb is None:
            return
        shape = self._extract_shape_feature(crop)
        old = self.gallery.get(cid)
        if old:
            emb = 0.80 * old["appearance"] + 0.20 * emb
            emb /= np.linalg.norm(emb) + 1e-6
            shape = 0.80 * old["shape"] + 0.20 * shape
            shape /= np.linalg.norm(shape) + 1e-6
        self.gallery[cid] = {"appearance": emb, "shape": shape, "entered_at": time.time()}

    @staticmethod
    def _crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    @staticmethod
    def _mean_normalized(features: list[np.ndarray]) -> Optional[np.ndarray]:
        if not features:
            return None
        mean = np.mean(features, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-6)

    def _active_candidates(self, state_mgr) -> dict[str, dict]:
        if state_mgr is None:
            return dict(self.gallery)
        active = set(state_mgr.get_active_customers())
        return {cid: profile for cid, profile in self.gallery.items() if cid in active}

    def associate_cam2_track(self, frame: np.ndarray, bbox: tuple, track_id: int,
                             display_track_id: str, state_mgr=None) -> tuple[str, str]:
        """Match a Camera 2 track to a Camera 1 customer. Returns (display_id, status)."""
        if track_id in self.track_to_cid:
            return self.track_to_cid[track_id], "MATCHED"
        embedding = self.extract_embedding(self._crop(frame, bbox))
        if embedding is None:
            return display_track_id, "COLLECTING"
        sample = self._samples[track_id]
        sample["last_seen"] = time.monotonic()
        sample["appearance"].append(embedding)
        sample["shape"].append(self._extract_shape_feature(self._crop(frame, bbox)))
        sample["appearance"] = sample["appearance"][-self.max_samples:]
        sample["shape"] = sample["shape"][-self.max_samples:]
        if len(sample["appearance"]) < self.min_samples:
            return display_track_id, f"COLLECTING {len(sample['appearance'])}/{self.min_samples}"
        candidates = self._active_candidates(state_mgr)
        if not candidates:
            return display_track_id, "NO CAM1 CANDIDATE"
        app, shape, now = self._mean_normalized(sample["appearance"]), self._mean_normalized(sample["shape"]), time.time()
        best_cid, best_score = None, -1.0
        for cid, profile in candidates.items():
            elapsed = now - profile["entered_at"]
            if elapsed < self.min_travel_seconds or elapsed > self.max_travel_seconds:
                continue
            s_app = max(0.0, float(np.dot(app, profile["appearance"])))
            s_shape = max(0.0, float(np.dot(shape, profile["shape"])))
            s_time = max(0.0, 1.0 - elapsed / self.max_travel_seconds)
            score = self.W_APPEARANCE * s_app + self.W_SHAPE * s_shape + self.W_TIME * s_time + self.W_ROUTE
            if score > best_score:
                best_cid, best_score = cid, score
        if best_cid is not None and best_score >= self.MATCH_THRESHOLD:
            self.track_to_cid[track_id] = best_cid
            if state_mgr:
                state_mgr.link_track_to_customer(track_id, best_cid)
            print(f"[REID MATCH] {display_track_id} -> {best_cid} (S={best_score:.2f})")
            return best_cid, f"MATCH {best_score:.2f}"
        if best_cid is not None and best_score >= self.UNCERTAIN_THRESHOLD:
            return display_track_id, f"UNCERTAIN {best_score:.2f}"
        return display_track_id, "NO MATCH"

    def match_or_register(self, frame: np.ndarray, bbox: tuple, track_id: int, state_mgr=None) -> str:
        cid, _ = self.associate_cam2_track(frame, bbox, track_id, f"CAM2-T{track_id}", state_mgr)
        return cid

    def clean_stale_tracks(self, max_age_seconds: float = 30.0) -> None:
        now = time.monotonic()
        for tid in [tid for tid, data in self._samples.items() if now - data["last_seen"] > max_age_seconds]:
            self._samples.pop(tid, None)
