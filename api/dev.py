"""Development/test weight event endpoint.

Only enabled when DEV_MODE is True (config.py).
Calls the SAME production weight-event processing pipeline.
"""

from flask import Blueprint, request, jsonify
import logging

import config
from services.weight_event_service import process_weight_event

dev_api = Blueprint("dev_api", __name__)
_logger = logging.getLogger(__name__)

# Reference to global state_mgr, set during app init
_state_mgr = None


def set_state_manager(state_mgr):
    """Set the global state manager reference for dev endpoints."""
    global _state_mgr
    _state_mgr = state_mgr


@dev_api.route("/api/dev/weight-event", methods=["POST"])
def simulate_weight_event():
    """Simulate a weight sensor event for development/testing.

    This endpoint calls the REAL production weight-event processing pipeline.
    No separate fake cart implementation.

    Only available when DEV_MODE is True.

    Request body:
    {
        "event_id": "TEST-EVENT-001",
        "sensor_id": "SENSOR_001",
        "previous_weight_g": 1000,
        "raw_weight_g": 1500,
        "weight_delta_g": 500
    }
    """
    if not config.DEV_MODE:
        return jsonify({
            "success": False,
            "message": "Development endpoint disabled in production mode"
        }), 403

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "message": "JSON request body required"}), 400

    required = ["event_id", "sensor_id"]
    for field in required:
        if not data.get(field):
            return jsonify({"success": False, "message": f"{field} is required"}), 400

    # Require at least raw_weight_g/previous_weight_g or weight_delta_g
    if "weight_delta_g" not in data and ("raw_weight_g" not in data or "previous_weight_g" not in data):
        return jsonify({
            "success": False,
            "message": "Provide weight_delta_g or both raw_weight_g and previous_weight_g"
        }), 400

    try:
        result = process_weight_event(data, state_mgr=_state_mgr)

        status_code = 200 if result.get("success") else 422
        return jsonify(result), status_code

    except Exception as e:
        _logger.error(f"Dev weight event error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Weight event processing failed"}), 500


@dev_api.route("/api/dev/seed-products", methods=["POST"])
def seed_products():
    """Seed test product and sensor mapping data."""
    if not config.DEV_MODE:
        return jsonify({"success": False, "message": "Development endpoint disabled"}), 403

    try:
        from services.product_service import seed_test_data, list_products, list_sensor_mappings
        seed_test_data()
        return jsonify({
            "success": True,
            "products": list_products(),
            "sensor_mappings": list_sensor_mappings(),
        }), 200
    except Exception as e:
        _logger.error(f"Seed products error: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@dev_api.route("/api/dev/active-state", methods=["GET"])
def active_state():
    """Return current in-memory state for debugging."""
    if not config.DEV_MODE:
        return jsonify({"success": False, "message": "Development endpoint disabled"}), 403

    if _state_mgr is None:
        return jsonify({"success": False, "message": "State manager not initialized"}), 500

    return jsonify({
        "success": True,
        "active_customers": _state_mgr.get_active_customers(),
        "active_sessions": _state_mgr.get_all_active_sessions(),
        "cam2_tracks": {str(k): v for k, v in _state_mgr.get_all_cam2_tracks().items()},
        "pose_states": {str(k): v for k, v in _state_mgr.get_all_pose_states().items()},
        "interacting_near_shelf": _state_mgr.get_interacting_customers_near_shelf(),
    }), 200
