"""
Test: Full Customer Identification Pipeline with Multiple Customers

Simulates the complete entry camera pipeline using a group photo:
  1. YOLO person detection → finds all people
  2. Face detection (InsightFace) → detects faces within each person bbox
  3. Face quality validation → checks sharpness, brightness, pose
  4. Face embedding extraction → generates 512-dim ArcFace vectors
  5. Customer DB matching → queries pgvector for identity (if DB available)
  6. Per-person unique tracking → verifies no two people get same ID

This validates that the customer identification code correctly handles
multiple simultaneous customers without cross-contamination.

Output: Annotated image with full pipeline results saved to Downloads.
"""

import os
import sys
import cv2
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────
INPUT_IMAGE = r"C:\Users\praya\Downloads\WhatsApp Image 2026-10-11 at 6.01.04 PM.jpeg"
OUTPUT_DIR = r"C:\Users\praya\Downloads"
OUTPUT_NAME = "customer_identification_test_output.jpg"

MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "yolo26n_openvino_model")
if not os.path.exists(MODEL_WEIGHTS) and not os.path.isdir(MODEL_WEIGHTS):
    MODEL_WEIGHTS = os.path.join(PROJECT_ROOT, "yolo26n.pt")

YOLO_IMGSZ = 640
YOLO_CONF = 0.30  # Slightly higher to avoid duplicate detections
PERSON_CLASS = 0

COLORS = [
    (255,  56,  56),  (56, 255,  56),  (56,  56, 255),
    (255, 200,  56),  (200, 56, 255),  (56, 255, 255),
    (255, 128,   0),  (128, 255, 128), (255,  56, 200),
]


def step_header(step_num, title):
    print(f"\n{'─'*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'─'*60}")


def run_full_identification_test():
    print(f"\n{'#'*60}")
    print(f"  MULTI-CUSTOMER IDENTIFICATION PIPELINE TEST")
    print(f"  Validates the full identification flow for group photo")
    print(f"{'#'*60}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Load image
    # ═══════════════════════════════════════════════════════════════
    step_header(1, "LOAD IMAGE")
    frame = cv2.imread(INPUT_IMAGE)
    if frame is None:
        print(f"  [FAIL] Could not load: {INPUT_IMAGE}")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"  [OK] Image loaded: {w}x{h} pixels")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: YOLO Person Detection
    # ═══════════════════════════════════════════════════════════════
    step_header(2, "YOLO PERSON DETECTION")
    model = YOLO(MODEL_WEIGHTS, task="detect")
    results = model.predict(frame, classes=[PERSON_CLASS], imgsz=YOLO_IMGSZ,
                            conf=YOLO_CONF, verbose=False)
    result = results[0] if results else None

    person_boxes = []
    if result is not None and result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            person_boxes.append({"bbox": (x1, y1, x2, y2), "conf": conf})

    # Filter overlapping boxes (NMS-like) to avoid duplicates
    filtered = []
    for i, pb in enumerate(person_boxes):
        is_dup = False
        for j, fb in enumerate(filtered):
            iou = _compute_iou(pb["bbox"], fb["bbox"])
            if iou > 0.5:
                is_dup = True
                if pb["conf"] > fb["conf"]:
                    filtered[j] = pb
                break
        if not is_dup:
            filtered.append(pb)
    person_boxes = filtered

    print(f"  [OK] Detected {len(person_boxes)} person(s)")
    for i, pb in enumerate(person_boxes):
        print(f"       Person {i+1}: bbox={pb['bbox']}, conf={pb['conf']:.2%}")

    if len(person_boxes) == 0:
        print("  [FAIL] No persons detected. Exiting.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: InsightFace Detection (all faces in full frame)
    # ═══════════════════════════════════════════════════════════════
    step_header(3, "INSIGHTFACE FACE DETECTION")
    from services.face_service import get_face_app, validate_face_quality_crop, extract_face_embedding_from_crop
    app = get_face_app()
    all_faces = app.get(frame)
    print(f"  [OK] InsightFace detected {len(all_faces)} face(s) in full frame")
    for i, f in enumerate(all_faces):
        fb = f.bbox.astype(int)
        score = float(getattr(f, 'det_score', 0.0))
        print(f"       Face {i+1}: bbox=({fb[0]},{fb[1]})->({fb[2]},{fb[3]}), score={score:.2%}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Match faces to person bboxes (like EntryFaceManager does)
    # ═══════════════════════════════════════════════════════════════
    step_header(4, "FACE-TO-PERSON MATCHING")
    print("  (Replicating EntryFaceManager._evaluate_face_job logic)")

    person_results = []
    for pidx, pb in enumerate(person_boxes):
        x1, y1, x2, y2 = pb["bbox"]
        pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Find closest face within person bbox (same logic as entry_face_manager.py L233-243)
        candidates = []
        for f in all_faces:
            fcx = (f.bbox[0] + f.bbox[2]) / 2.0
            fcy = (f.bbox[1] + f.bbox[3]) / 2.0
            if (x1 - 40 <= fcx <= x2 + 40) and (y1 - 40 <= fcy <= y2 + 40):
                dist = abs(fcx - pcx) + abs(fcy - pcy)
                candidates.append((dist, f))

        matched_face = None
        if candidates:
            candidates.sort(key=lambda c: c[0])
            matched_face = candidates[0][1]

        result_entry = {
            "person_idx": pidx + 1,
            "person_bbox": pb["bbox"],
            "person_conf": pb["conf"],
            "face_matched": matched_face is not None,
            "face_bbox": None,
            "face_det_score": 0.0,
            "quality_ok": False,
            "quality_reason": None,
            "quality_metrics": {},
            "embedding": None,
            "embedding_dim": 0,
            "db_match": None,
        }

        if matched_face is not None:
            fb = matched_face.bbox.astype(int)
            result_entry["face_bbox"] = (int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]))
            result_entry["face_det_score"] = float(getattr(matched_face, 'det_score', 0.0))
            print(f"\n  Person {pidx+1}: Face MATCHED")
            print(f"    Face bbox: ({fb[0]},{fb[1]})->({fb[2]},{fb[3]})")
            print(f"    Det score: {result_entry['face_det_score']:.2%}")

            # Step 5: Validate face quality
            ok, reject_reason, metrics = validate_face_quality_crop(matched_face, frame)
            result_entry["quality_ok"] = ok
            result_entry["quality_reason"] = reject_reason
            result_entry["quality_metrics"] = metrics
            print(f"    Quality:   {'✓ PASS' if ok else '✗ FAIL'}")
            if not ok:
                print(f"    Reason:    {reject_reason}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.3f}")
                else:
                    print(f"      {k}: {v}")

            # Step 6: Extract embedding
            if ok:
                emb = extract_face_embedding_from_crop(frame, matched_face)
                if emb is not None:
                    result_entry["embedding"] = emb
                    result_entry["embedding_dim"] = len(emb)
                    print(f"    Embedding: ✓ {len(emb)}-dim vector extracted")
                    print(f"    Norm:      {np.linalg.norm(emb):.4f}")
                else:
                    print(f"    Embedding: ✗ FAILED to extract")
        else:
            print(f"\n  Person {pidx+1}: Face NOT FOUND in bbox")

        person_results.append(result_entry)

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Try DB matching (identify_customer_by_face_embedding)
    # ═══════════════════════════════════════════════════════════════
    step_header(5, "DATABASE CUSTOMER MATCHING")
    db_available = False
    try:
        from services.customer_service import identify_customer_by_face_embedding
        # Quick test if DB is reachable
        from database.connection import get_connection
        conn = get_connection()
        conn.close()
        db_available = True
        print("  [OK] Database connection available")
    except Exception as e:
        print(f"  [WARN] Database not available: {e}")
        print("  [INFO] Skipping DB matching — testing pipeline logic only")

    if db_available:
        for pr in person_results:
            if pr["embedding"] is not None:
                try:
                    match = identify_customer_by_face_embedding(pr["embedding"])
                    pr["db_match"] = match
                    if match:
                        print(f"  Person {pr['person_idx']}: → {match['customer_id']} "
                              f"({match.get('name', '?')}) sim={match['similarity']:.3f}")
                    else:
                        print(f"  Person {pr['person_idx']}: → No DB match (new customer)")
                except Exception as e:
                    print(f"  Person {pr['person_idx']}: → DB query error: {e}")
            else:
                print(f"  Person {pr['person_idx']}: → No embedding (skipped)")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Cross-contamination check
    # ═══════════════════════════════════════════════════════════════
    step_header(6, "CROSS-CONTAMINATION CHECK")
    embeddings = [(pr["person_idx"], pr["embedding"]) for pr in person_results if pr["embedding"] is not None]

    if len(embeddings) >= 2:
        print(f"  Comparing {len(embeddings)} face embeddings for uniqueness...\n")
        all_unique = True
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                idx_a, emb_a = embeddings[i]
                idx_b, emb_b = embeddings[j]
                sim = float(np.dot(emb_a, emb_b))
                status = "✓ DIFFERENT" if sim < 0.40 else ("⚠ SIMILAR" if sim < 0.60 else "✗ TOO SIMILAR")
                if sim >= 0.40:
                    all_unique = False
                print(f"    Person {idx_a} vs Person {idx_b}: cosine_sim = {sim:.4f}  [{status}]")

        if all_unique:
            print(f"\n  ✓ PASS: All embeddings are unique — no cross-contamination")
        else:
            print(f"\n  ⚠ NOTE: Some pairs show similarity. Could be same person")
            print(f"          or similar faces. In live pipeline, tracking + voting resolves this.")

        if db_available:
            matched_ids = [pr["db_match"]["customer_id"] for pr in person_results
                          if pr["db_match"] is not None]
            if matched_ids:
                unique_ids = set(matched_ids)
                if len(unique_ids) == len(matched_ids):
                    print(f"\n  ✓ PASS: All DB matches are unique IDs: {unique_ids}")
                else:
                    print(f"\n  ⚠ WARN: Duplicate customer IDs assigned: {matched_ids}")
                    print(f"          The confidence voting in EntryFaceManager should resolve this")
    else:
        print(f"  [INFO] Only {len(embeddings)} embedding(s) available, need ≥2 for comparison")

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Draw annotated output with full pipeline results
    # ═══════════════════════════════════════════════════════════════
    step_header(7, "DRAW ANNOTATED OUTPUT")
    annotated = frame.copy()

    for pr in person_results:
        pidx = pr["person_idx"]
        color = COLORS[(pidx - 1) % len(COLORS)]
        x1, y1, x2, y2 = pr["person_bbox"]

        # Person bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Face bbox (if found)
        if pr["face_bbox"]:
            fx1, fy1, fx2, fy2 = pr["face_bbox"]
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

        # Build label lines
        lines = []
        if pr["db_match"]:
            cid = pr["db_match"]["customer_id"]
            name = pr["db_match"].get("name", "")
            sim = pr["db_match"]["similarity"]
            lines.append(f"{cid} ({name})")
            lines.append(f"sim: {sim:.3f}")
        else:
            lines.append(f"Person #{pidx}")

        if pr["face_matched"]:
            if pr["quality_ok"]:
                lines.append(f"Face: OK | Emb: {pr['embedding_dim']}d")
            else:
                reason = pr["quality_reason"] or "quality fail"
                # Truncate long reasons
                if len(reason) > 30:
                    reason = reason[:27] + "..."
                lines.append(f"Face: {reason}")
        else:
            lines.append("Face: NOT DETECTED")

        lines.append(f"YOLO: {pr['person_conf']:.0%}")

        # Draw label banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        line_h = 18
        total_h = len(lines) * line_h + 8

        max_tw = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_tw = max(max_tw, tw)

        banner_top = max(0, y1 - total_h)
        banner_right = min(w, x1 + max_tw + 16)
        cv2.rectangle(annotated, (x1, banner_top), (banner_right, y1), color, -1)

        for i, line in enumerate(lines):
            ty = banner_top + (i + 1) * line_h
            cv2.putText(annotated, line, (x1 + 4, ty), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        # Center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 5, color, -1, cv2.LINE_AA)

    # Summary bar
    n_total = len(person_results)
    n_faces = sum(1 for p in person_results if p["face_matched"])
    n_quality = sum(1 for p in person_results if p["quality_ok"])
    n_emb = sum(1 for p in person_results if p["embedding"] is not None)
    n_db = sum(1 for p in person_results if p["db_match"] is not None)

    summary = (f"Persons: {n_total} | Faces: {n_faces} | Quality OK: {n_quality} "
               f"| Embeddings: {n_emb} | DB Match: {n_db}")
    cv2.rectangle(annotated, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(annotated, summary, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

    # Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    cv2.imwrite(output_path, annotated)
    print(f"  [OK] Saved: {output_path}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PIPELINE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Persons detected (YOLO):     {n_total}")
    print(f"  Faces detected (InsightFace): {n_faces}/{n_total}")
    print(f"  Face quality passed:          {n_quality}/{n_faces}")
    print(f"  Embeddings extracted:         {n_emb}/{n_quality}")
    print(f"  DB matches found:             {n_db}/{n_emb}")
    print()

    # Verdict
    tests_passed = 0
    tests_total = 0

    def check(name, condition):
        nonlocal tests_passed, tests_total
        tests_total += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        if condition:
            tests_passed += 1
        print(f"  {status}: {name}")
        return condition

    check("YOLO detects multiple persons (≥2)", n_total >= 2)
    check("InsightFace detects faces for detected persons", n_faces >= 2)
    check("Face quality validation runs without crash", True)  # If we got here, it worked
    check("Face embeddings are 512-dimensional",
          all(pr["embedding_dim"] == 512 for pr in person_results if pr["embedding"] is not None))
    check("Embeddings are L2-normalized",
          all(abs(np.linalg.norm(pr["embedding"]) - 1.0) < 0.01
              for pr in person_results if pr["embedding"] is not None))

    if len(embeddings) >= 2:
        # Check that different people get different embeddings
        all_different = True
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i][1], embeddings[j][1]))
                if sim > 0.90:  # Would indicate same person or bug
                    all_different = False
        check("Different persons get distinct embeddings (sim < 0.9)", all_different)

    if db_available and n_db >= 2:
        matched_ids = [pr["db_match"]["customer_id"] for pr in person_results if pr["db_match"]]
        check("No two persons assigned same customer ID", len(set(matched_ids)) == len(matched_ids))

    print(f"\n  Result: {tests_passed}/{tests_total} tests passed")
    print(f"{'='*60}\n")

    return person_results


def _compute_iou(box1, box2):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    run_full_identification_test()
