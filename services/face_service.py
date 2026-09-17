"""Face detection, validation, alignment, and ArcFace embedding extraction."""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from services.customer_service import enroll_face
from config import (
    FACE_QUALITY_MIN_SHARPNESS,
    FACE_QUALITY_MIN_BRIGHTNESS,
    FACE_QUALITY_MAX_BRIGHTNESS,
    FACE_QUALITY_MIN_SIZE,
    FACE_QUALITY_MAX_YAW,
    FACE_QUALITY_MAX_PITCH,
    FACE_QUALITY_MAX_ROLL,
    FACE_DETECTION_CONF_MIN,
)

logger = logging.getLogger(__name__)

_face_app: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    """Lazy-load the FaceAnalysis model with ArcFace ONNX."""
    global _face_app
    if _face_app is None:
        logger.info("Initializing InsightFace FaceAnalysis (buffalo_s)...")
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        logger.info("InsightFace FaceAnalysis loaded successfully.")
    return _face_app


def estimate_pose(face, img_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """Estimate head pose angles (pitch, yaw, roll) using facial landmarks."""
    img_h, img_w = img_shape[:2]

    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        lmk = face.landmark_3d_68
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float64)

        image_points = np.array([
            lmk[30][:2],
            lmk[8][:2],
            lmk[36][:2],
            lmk[45][:2],
            lmk[48][:2],
            lmk[54][:2]
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return float(angles[0]), float(angles[1]), float(angles[2])

    if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5:
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist > 1e-3:
            mid_x = (left_eye[0] + right_eye[0]) / 2.0
            yaw_approx = ((nose[0] - mid_x) / eye_dist) * 90.0
            return 0.0, float(yaw_approx), 0.0

    return 0.0, 0.0, 0.0


def validate_and_extract_embedding(
    image_bytes: bytes,
    expected_pose: str
) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Validate face image quality and pose, returning aligned 512-dim embedding."""
    if not image_bytes:
        return False, None, f"No image data received for {expected_pose} view."

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False, None, f"Invalid or corrupted image format for {expected_pose} view."

    img_h, img_w = img.shape[:2]
    if img_h < 150 or img_w < 150:
        return False, None, f"Image resolution too low ({img_w}x{img_h}) for {expected_pose} view."

    app = get_face_app()
    faces = app.get(img)
    if len(faces) == 0:
        return False, None, f"No face detected in {expected_pose} image. Please ensure your face is clearly visible."
    if len(faces) > 1:
        return False, None, f"Multiple faces ({len(faces)}) detected in {expected_pose} image. Only one face must be visible."

    face = faces[0]
    bbox = face.bbox.astype(int)
    face_w = max(1, bbox[2] - bbox[0])
    face_h = max(1, bbox[3] - bbox[1])
    face_area_ratio = (face_w * face_h) / float(img_w * img_h)

    if face_area_ratio < 0.05:
        return False, None, f"Face is too small in {expected_pose} image. Please move closer to the camera."

    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(img_w, bbox[2]), min(img_h, bbox[3])
    face_crop = img[y1:y2, x1:x2]

    if face_crop.size > 0:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        if laplacian_var < 25.0:
            return False, None, f"Face image is too blurry (sharpness score: {laplacian_var:.1f}). Please hold steady."

        mean_brightness = np.mean(gray_face)
        if mean_brightness < 30:
            return False, None, f"Face image is too dark (brightness: {mean_brightness:.1f}). Please improve lighting."
        if mean_brightness > 235:
            return False, None, f"Face image is too bright / washed out. Please avoid glare."

    pitch, yaw, roll = estimate_pose(face, (img_h, img_w))
    logger.info(f"Detected pose for {expected_pose}: pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}")

    if abs(roll) > 35.0 and abs(abs(roll) - 180.0) > 35.0:
        return False, None, f"Head tilted too much in {expected_pose} image. Please keep head level."

    pose_upper = expected_pose.upper()
    if pose_upper == "FRONT":
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]
        d_left = abs(nose[0] - left_eye[0])
        d_right = abs(right_eye[0] - nose[0])
        ratio = d_left / max(1.0, d_right)
        if abs(yaw) > 22.0 and (ratio < 0.45 or ratio > 2.2):
            return False, None, f"Front view requires looking straight at the camera (detected yaw: {yaw:.1f}°)."

    elif pose_upper == "LEFT":
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]
        d_left = abs(nose[0] - left_eye[0])
        d_right = abs(right_eye[0] - nose[0])
        ratio = d_left / max(1.0, d_right)
        if abs(yaw) < 8.0 and (0.75 <= ratio <= 1.35):
            return False, None, "Left view requires turning your head to the left."

    elif pose_upper == "RIGHT":
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        nose = face.kps[2]
        d_left = abs(nose[0] - left_eye[0])
        d_right = abs(right_eye[0] - nose[0])
        ratio = d_left / max(1.0, d_right)
        if abs(yaw) < 8.0 and (0.75 <= ratio <= 1.35):
            return False, None, "Right view requires turning your head to the right."

    aligned_face = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    if aligned_face is None:
        return False, None, f"Failed to align face for {expected_pose} view."

    rec_model = app.models.get("recognition")
    if rec_model is None:
        return False, None, "Face recognition model is unavailable on server."

    embedding = rec_model.get(img, face)
    if embedding is None:
        return False, None, f"Failed to generate embedding vector for {expected_pose} view."

    embedding = np.array(embedding, dtype=np.float32).flatten()
    if len(embedding) != 512:
        return False, None, f"Model generated {len(embedding)}-dim vector, expected 512 dimensions."

    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return True, embedding, None


def process_enrollment_images(
    customer_id: str,
    front_bytes: bytes,
    left_bytes: bytes,
    right_bytes: bytes
) -> Dict[str, Any]:
    """Validate front, left, and right enrollment images and persist embeddings."""
    customer_id_clean = customer_id.strip()

    ok, front_vec, err = validate_and_extract_embedding(front_bytes, "FRONT")
    if not ok:
        return {"success": False, "message": f"Front view error: {err}"}

    ok, left_vec, err = validate_and_extract_embedding(left_bytes, "LEFT")
    if not ok:
        return {"success": False, "message": f"Left view error: {err}"}

    ok, right_vec, err = validate_and_extract_embedding(right_bytes, "RIGHT")
    if not ok:
        return {"success": False, "message": f"Right view error: {err}"}

    result = enroll_face(
        customer_id=customer_id_clean,
        front_embedding=front_vec.tolist(),
        left_embedding=left_vec.tolist(),
        right_embedding=right_vec.tolist()
    )

    return result


def validate_face_quality_crop(
    face,
    crop_img: np.ndarray
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validate face quality metrics (size, sharpness, brightness, pose tolerances)."""
    metrics: Dict[str, Any] = {}
    crop_h, crop_w = crop_img.shape[:2]

    det_score = float(getattr(face, "det_score", 0.0))
    metrics["det_score"] = det_score
    if det_score < FACE_DETECTION_CONF_MIN:
        return False, f"Low detection confidence ({det_score:.2f} < {FACE_DETECTION_CONF_MIN})", metrics

    bbox = face.bbox.astype(int)
    fw = max(1, bbox[2] - bbox[0])
    fh = max(1, bbox[3] - bbox[1])
    metrics["face_width"] = fw
    metrics["face_height"] = fh
    if fw < FACE_QUALITY_MIN_SIZE or fh < FACE_QUALITY_MIN_SIZE:
        return False, f"Face too small ({fw}x{fh}px < {FACE_QUALITY_MIN_SIZE}px)", metrics

    face_cy = (bbox[1] + bbox[3]) / 2.0
    metrics["face_cy_ratio"] = face_cy / float(crop_h)

    vis_w = max(0, min(crop_w, bbox[2]) - max(0, bbox[0]))
    vis_h = max(0, min(crop_h, bbox[3]) - max(0, bbox[1]))
    vis_ratio = (vis_w * vis_h) / float(max(1, fw * fh))
    metrics["visible_ratio"] = vis_ratio
    if vis_ratio < 0.65:
        return False, "Face heavily clipped at boundary edge", metrics

    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(crop_w, bbox[2]), min(crop_h, bbox[3])
    face_patch = crop_img[y1:y2, x1:x2]
    if face_patch.size == 0 or face_patch.shape[0] < 10 or face_patch.shape[1] < 10:
        return False, "Empty or invalid face patch", metrics

    gray_face = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray_face))
    metrics["brightness"] = mean_brightness
    if mean_brightness < FACE_QUALITY_MIN_BRIGHTNESS:
        return False, f"Face too dark (brightness: {mean_brightness:.1f} < {FACE_QUALITY_MIN_BRIGHTNESS})", metrics
    if mean_brightness > FACE_QUALITY_MAX_BRIGHTNESS:
        return False, f"Face overexposed/glare (brightness: {mean_brightness:.1f} > {FACE_QUALITY_MAX_BRIGHTNESS})", metrics

    laplacian_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    metrics["sharpness"] = laplacian_var
    if laplacian_var < FACE_QUALITY_MIN_SHARPNESS:
        return False, f"Face too blurry (sharpness: {laplacian_var:.1f} < {FACE_QUALITY_MIN_SHARPNESS})", metrics

    pitch, yaw, roll = estimate_pose(face, (crop_h, crop_w))
    eff_pitch = min(abs(pitch), abs(abs(pitch) - 180.0))
    eff_roll = min(abs(roll), abs(abs(roll) - 180.0))
    eff_yaw = abs(yaw)

    metrics["pitch"] = eff_pitch
    metrics["yaw"] = eff_yaw
    metrics["roll"] = eff_roll

    if eff_yaw > FACE_QUALITY_MAX_YAW:
        return False, f"Head turned too far (yaw: {eff_yaw:.1f}° > {FACE_QUALITY_MAX_YAW}°)", metrics
    if eff_pitch > FACE_QUALITY_MAX_PITCH:
        return False, f"Head tilted up/down (pitch: {eff_pitch:.1f}° > {FACE_QUALITY_MAX_PITCH}°)", metrics
    if eff_roll > FACE_QUALITY_MAX_ROLL:
        return False, f"Head tilted sideways (roll: {eff_roll:.1f}° > {FACE_QUALITY_MAX_ROLL}°)", metrics

    return True, None, metrics


def extract_face_embedding_from_crop(
    crop_img: np.ndarray,
    face
) -> Optional[np.ndarray]:
    """Extract normalized 512-dimensional ArcFace embedding from aligned face crop."""
    try:
        aligned_face = face_align.norm_crop(crop_img, landmark=face.kps, image_size=112)
        if aligned_face is None:
            return None

        app = get_face_app()
        rec_model = app.models.get("recognition")
        if rec_model is None:
            return None

        emb = rec_model.get(crop_img, face)
        if emb is None:
            return None

        emb = np.array(emb, dtype=np.float32).flatten()
        if len(emb) != 512:
            return None

        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        return emb
    except Exception as exc:
        logger.error(f"Error extracting face embedding from crop: {exc}")
        return None


def process_entry_face_crop(
    person_crop: np.ndarray
) -> Tuple[bool, Optional[np.ndarray], Optional[str], Dict[str, Any]]:
    """Detect and validate face in person crop, returning normalized embedding."""
    if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 40 or person_crop.shape[1] < 40:
        return False, None, "Person crop too small or invalid", {}

    app = get_face_app()
    faces = app.get(person_crop)

    if len(faces) == 0:
        return False, None, "No face detected in crop", {}

    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    is_valid, reject_reason, metrics = validate_face_quality_crop(face, person_crop)
    if not is_valid:
        return False, None, reject_reason, metrics

    embedding = extract_face_embedding_from_crop(person_crop, face)
    if embedding is None:
        return False, None, "Failed to generate ArcFace embedding", metrics

    return True, embedding, None, metrics
