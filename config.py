"""
Configuration loader - reads and saves settings directly to config.json.
Existing imports like `from config import ENTRY_CAMERA_INDEX` continue to work.
"""
import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_EXAMPLE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.example.json")

_DEFAULT_CONFIG = {
    "entry_camera": "",
    "shelf_camera": "",
    "shelf_map": {
        "SENSOR_01": {
            "product_name": "Lays Chips",
            "price": 20,
            "pos_x": 320,
            "pos_y": 240,
            "weight_g": 50,
        },
        "SENSOR_02": {
            "product_name": "Coke Can",
            "price": 35,
            "pos_x": 480,
            "pos_y": 240,
            "weight_g": 330,
        },
    },
    "host_ip": "0.0.0.0",
    "host_port": 5000,
}


def _load_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load():
    """Load config.json (or config.example.json as fallback), with environment overrides."""
    data = dict(_DEFAULT_CONFIG)
    if os.path.exists(_BASE_CONFIG_PATH):
        data.update(_load_file(_BASE_CONFIG_PATH))
    else:
        data.update(_load_file(_EXAMPLE_CONFIG_PATH))


    env_entry = os.getenv("ENTRY_CAMERA")
    if env_entry:
        data["entry_camera"] = env_entry

    env_shelf = os.getenv("SHELF_CAMERA")
    if env_shelf:
        data["shelf_camera"] = env_shelf

    env_host = os.getenv("HOST_IP")
    if env_host:
        data["host_ip"] = env_host

    env_port = os.getenv("HOST_PORT")
    if env_port:
        data["host_port"] = int(env_port)

    return data


def save_camera_config(entry_cam, shelf_cam):
    """Write camera sources directly to config.json and reload active settings."""
    data = _load()
    data["entry_camera"] = str(entry_cam).strip()
    data["shelf_camera"] = str(shelf_cam).strip()
    with open(_BASE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    reload()


def reload():
    """Re-read config.json and return fresh values while updating global variables."""
    global ENTRY_CAMERA_INDEX, SHELF_CAMERA_INDEX, SHELF_MAP, HOST_IP, HOST_PORT
    data = _load()
    ENTRY_CAMERA_INDEX = data.get("entry_camera", "")
    SHELF_CAMERA_INDEX = data.get("shelf_camera", "")
    SHELF_MAP = data.get("shelf_map", {})
    HOST_IP = data.get("host_ip", "0.0.0.0")
    HOST_PORT = data.get("host_port", 5000)
    return data


# --- Expose variables on import ---
ENTRY_CAMERA_INDEX = ""
SHELF_CAMERA_INDEX = ""
SHELF_MAP = {}
HOST_IP = "0.0.0.0"
HOST_PORT = 5000

reload()

