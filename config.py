"""Configuration module — credentials from env, runtime settings from code,
camera sources persisted to .camera_settings file."""
import json
import os
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(_env_path)
except ImportError:
    pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_BASE_DIR, ".camera_settings")

ENTRY_CAMERA_DEFAULT = ""
SHELF_CAMERA_DEFAULT = ""
HOST_IP_DEFAULT = "0.0.0.0"
HOST_PORT_DEFAULT = 5000

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "shopflow")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

FACE_MATCH_HIGH_THRESHOLD = 0.24
FACE_MATCH_MEDIUM_THRESHOLD = 0.18
FACE_MATCH_MAX_ATTEMPTS = 5

FACE_QUALITY_MIN_SHARPNESS = 5.0
FACE_QUALITY_MIN_BRIGHTNESS = 30.0
FACE_QUALITY_MAX_BRIGHTNESS = 235.0
FACE_QUALITY_MIN_SIZE = 35
FACE_QUALITY_MAX_YAW = 40.0
FACE_QUALITY_MAX_PITCH = 35.0
FACE_QUALITY_MAX_ROLL = 35.0
FACE_DETECTION_CONF_MIN = 0.40

UNIFORM_BOX_COLOR = False

# === Shopping Session ===
SESSION_ID_PREFIX = "SESS"

# === Shelf Interaction ROI (Camera 2) ===
SHELF_ROI_X1 = int(os.getenv("SHELF_ROI_X1", 200))
SHELF_ROI_Y1 = int(os.getenv("SHELF_ROI_Y1", 100))
SHELF_ROI_X2 = int(os.getenv("SHELF_ROI_X2", 600))
SHELF_ROI_Y2 = int(os.getenv("SHELF_ROI_Y2", 450))
SHELF_ROI = (SHELF_ROI_X1, SHELF_ROI_Y1, SHELF_ROI_X2, SHELF_ROI_Y2)
SHELF_ROI_OVERLAP_THRESHOLD = float(os.getenv("SHELF_ROI_OVERLAP_THRESHOLD", 0.20))
SHELF_ROI_SHOW_OVERLAY = os.getenv("SHELF_ROI_SHOW_OVERLAY", "true").lower() in ("1", "true", "yes")

# === Exit ROI (Camera 1) ===
EXIT_ROI_X1 = int(os.getenv("EXIT_ROI_X1", 0))
EXIT_ROI_Y1 = int(os.getenv("EXIT_ROI_Y1", 350))
EXIT_ROI_X2 = int(os.getenv("EXIT_ROI_X2", 200))
EXIT_ROI_Y2 = int(os.getenv("EXIT_ROI_Y2", 480))
EXIT_ROI = (EXIT_ROI_X1, EXIT_ROI_Y1, EXIT_ROI_X2, EXIT_ROI_Y2)

# === YOLO11n-Pose ===
POSE_MODEL_WEIGHTS = os.getenv("POSE_MODEL_WEIGHTS", "yolo11n-pose.pt")
POSE_CONF = float(os.getenv("POSE_CONF", 0.3))
POSE_IMGSZ = int(os.getenv("POSE_IMGSZ", 256))
POSE_INFERENCE_INTERVAL = float(os.getenv("POSE_INFERENCE_INTERVAL", 0.10))

# === Weight Sensor ===
WEIGHT_CHANGE_MIN_GRAMS = float(os.getenv("WEIGHT_CHANGE_MIN_GRAMS", 20.0))
WEIGHT_STABILITY_TOLERANCE_GRAMS = float(os.getenv("WEIGHT_STABILITY_TOLERANCE_GRAMS", 5.0))
WEIGHT_EVENT_ASSOCIATION_WINDOW_SECONDS = float(os.getenv("WEIGHT_EVENT_ASSOCIATION_WINDOW_SECONDS", 3.0))
WEIGHT_ASSOCIATION_MIN_CONFIDENCE = float(os.getenv("WEIGHT_ASSOCIATION_MIN_CONFIDENCE", 0.5))
WEIGHT_EVENT_DEBOUNCE_SECONDS = float(os.getenv("WEIGHT_EVENT_DEBOUNCE_SECONDS", 1.0))

# === Development Mode ===
DEV_MODE = os.getenv("DEV_MODE", "true").lower() in ("1", "true", "yes")

# === Tax ===
TAX_RATE = float(os.getenv("TAX_RATE", 0.0))


class Config:
    _env_secret = os.getenv("FLASK_SECRET_KEY")
    if not _env_secret:
        import secrets as _secrets
        import warnings as _warnings
        SECRET_KEY = _secrets.token_hex(32)
        _warnings.warn(
            "FLASK_SECRET_KEY is not set — using a random ephemeral key. "
            "Sessions will not survive restarts. Set FLASK_SECRET_KEY in .env for production.",
            stacklevel=1,
        )
    else:
        SECRET_KEY = _env_secret

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DB_USER = POSTGRES_USER
    DB_PASSWORD = POSTGRES_PASSWORD
    DB_HOST = POSTGRES_HOST
    DB_PORT = POSTGRES_PORT
    DB_NAME = POSTGRES_DATABASE

    if POSTGRES_PASSWORD:
        SQLALCHEMY_DATABASE_URI = (
            f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
        )
    else:
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(_BASE_DIR, 'retail_store.db')}"

    HOST = HOST_IP_DEFAULT
    PORT = HOST_PORT_DEFAULT
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = not DEBUG
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit


def _load_settings():
    if not os.path.exists(_SETTINGS_PATH):
        return {}
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(data):
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_camera_config(entry_cam, shelf_cam):
    """Persist camera sources and reload active settings."""
    data = _load_settings()
    data["entry_camera"] = str(entry_cam).strip()
    data["shelf_camera"] = str(shelf_cam).strip()
    _save_settings(data)
    reload()


def reload():
    """Re-read persisted settings and update module-level variables."""
    global ENTRY_CAMERA_INDEX, SHELF_CAMERA_INDEX, HOST_IP, HOST_PORT
    data = _load_settings()
    ENTRY_CAMERA_INDEX = data.get("entry_camera", ENTRY_CAMERA_DEFAULT)
    SHELF_CAMERA_INDEX = data.get("shelf_camera", SHELF_CAMERA_DEFAULT)
    HOST_IP = data.get("host_ip", HOST_IP_DEFAULT)
    HOST_PORT = data.get("host_port", HOST_PORT_DEFAULT)
    return data


ENTRY_CAMERA_INDEX = ""
SHELF_CAMERA_INDEX = ""
HOST_IP = HOST_IP_DEFAULT
HOST_PORT = HOST_PORT_DEFAULT

reload()
