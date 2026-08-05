"""
reid_engine.py
--------------
OSNet / Deep Feature Embedding + Cosine Similarity Re-Identification Engine.

Matches persons across different camera streams (Entry Camera & Shelf Camera)
by extracting deep visual feature vectors for each person crop and computing
Cosine Similarity matching against a gallery of known customer embeddings.
"""

import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models

class PersonReIDEngine:
    def __init__(self, sim_threshold: float = 0.45):
        """
        Initialize Deep ReID Feature Extractor using lightweight CNN backbone.
        sim_threshold: Cosine similarity threshold (0.0 to 1.0) for matching.
        """
        self.sim_threshold = sim_threshold
        self.device = torch.device("cpu")

        print("[REID] Initializing OSNet / Deep Feature ReID Engine ...")
        # Lightweight MobileNetV3 backbone for real-time CPU feature extraction
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor.eval()

        # ReID Input Preprocessing (256x128 standard ReID resolution)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Gallery mapping: customer_id -> L2-normalized feature vector (np.ndarray)
        self.gallery = {}
        # Track ID to Customer ID cache mapping
        self.track_to_cid = {}
        self._next_cust_idx = 1
        print("[REID] Re-Identification Engine ready.")

    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract 576-dim L2-normalized feature embedding for a person crop."""
        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return None

        try:
            # Convert BGR -> RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            # Extract features
            feat = self.feature_extractor(tensor)
            feat = self.pool(feat).flatten(1)

            # L2 Normalization
            norm = torch.norm(feat, p=2, dim=1, keepdim=True)
            feat_norm = (feat / (norm + 1e-6)).cpu().numpy().flatten()
            return feat_norm
        except Exception as e:
            return None

    def match_or_register(self, frame: np.ndarray, bbox: tuple, track_id: int, state_mgr=None) -> str:
        """
        Crop person region from frame, extract feature vector, and match via Cosine Similarity.
        Returns persistent customer ID (e.g. CUST_1).
        """
        # If this track ID is already matched in cache, return it
        if track_id in self.track_to_cid:
            return self.track_to_cid[track_id]

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        emb = self.extract_embedding(crop)

        if emb is None:
            cid = f"CUST_{track_id}"
            self.track_to_cid[track_id] = cid
            if state_mgr and cid not in state_mgr.track_to_customer.values():
                state_mgr.register_customer(cid)
                state_mgr.link_track_to_customer(track_id, cid)
            return cid

        best_cid = None
        best_sim = -1.0

        # Compute Cosine Similarity against all customer embeddings in gallery
        for cid, gal_emb in self.gallery.items():
            sim = float(np.dot(emb, gal_emb))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_sim >= self.sim_threshold and best_cid is not None:
            # Match found! Update profile using exponential moving average (EMA)
            self.gallery[best_cid] = 0.85 * self.gallery[best_cid] + 0.15 * emb
            self.gallery[best_cid] /= (np.linalg.norm(self.gallery[best_cid]) + 1e-6)
            cid = best_cid
            print(f"[REID MATCH] Track {track_id} -> {cid} (Cosine Sim: {best_sim:.2f})")
        else:
            # Register new customer profile in gallery
            cid = f"CUST_{self._next_cust_idx}"
            self._next_cust_idx += 1
            self.gallery[cid] = emb
            print(f"[REID NEW] Track {track_id} registered as {cid}")

        self.track_to_cid[track_id] = cid
        if state_mgr:
            state_mgr.register_customer(cid)
            state_mgr.link_track_to_customer(track_id, cid)

        return cid
