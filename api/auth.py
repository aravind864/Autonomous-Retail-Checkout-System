from flask import Blueprint, request, jsonify
import logging

from services.customer_service import login_customer

auth_api = Blueprint("auth_api", __name__)
_logger = logging.getLogger(__name__)


@auth_api.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate customer credentials and return user details."""
    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "Request body is required"
        }), 400

    email = data.get("email")
    password = data.get("password")

    if not email:
        return jsonify({
            "success": False,
            "message": "Email is required"
        }), 400

    if not password:
        return jsonify({
            "success": False,
            "message": "Password is required"
        }), 400

    try:
        result = login_customer(email, password)

        if not result["success"]:
            return jsonify(result), 401

        return jsonify(result), 200

    except Exception as e:
        _logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Login failed"
        }), 500
