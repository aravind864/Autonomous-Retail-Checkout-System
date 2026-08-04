"""
Configuration loader - reads settings from config.json, with optional local overrides.
Existing imports like `from config import ENTRY_CAMERA_INDEX` continue to work.
"""
import json
import os

_BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.local.json")

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load():
    """Load base config, then local overrides, then environment overrides."""
    data = dict(_DEFAULT_CONFIG)
    data.update(_load_file(_BASE_CONFIG_PATH))
    data.update(_load_file(_LOCAL_CONFIG_PATH))

    data["entry_camera"] = os.getenv("ENTRY_CAMERA", data.get("entry_camera", ""))
    data["shelf_camera"] = os.getenv("SHELF_CAMERA", data.get("shelf_camera", ""))
    data["host_ip"] = os.getenv("HOST_IP", data.get("host_ip", "0.0.0.0"))
    data["host_port"] = int(os.getenv("HOST_PORT", str(data.get("host_port", 5000))))
    return data


def save_camera_config(entry_cam, shelf_cam):
    """Write private camera sources to config.local.json."""
    data = _load()
    data["entry_camera"] = str(entry_cam).strip()
    data["shelf_camera"] = str(shelf_cam).strip()
    with open(_LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def reload():
    """Re-read config files and return fresh values."""
    return _load()


# --- Expose variables on import ---
_cfg = _load()

ENTRY_CAMERA_INDEX = _cfg["entry_camera"]
SHELF_CAMERA_INDEX = _cfg["shelf_camera"]
SHELF_MAP = _cfg["shelf_map"]
HOST_IP = _cfg["host_ip"]
HOST_PORT = _cfg["host_port"]
