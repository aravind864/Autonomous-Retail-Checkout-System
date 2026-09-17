from flask import Blueprint, request, jsonify
import logging

from services.customer_service import register_customer

customer_api = Blueprint("customer_api", __name__)
_logger = logging.getLogger(__name__)


@customer_api.route("/api/customers/register", methods=["POST"])
def register():
    """Register a new customer account."""
    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "Request body is required"
        }), 400

    required_fields = [
        "first_name",
        "last_name",
        "email",
        "password"
    ]

    for field in required_fields:
        if not data.get(field):
            return jsonify({
                "success": False,
                "message": f"{field} is required"
            }), 400

    try:
        result = register_customer(
            data["first_name"],
            data["last_name"],
            data["email"],
            data["password"]
        )

        if not result["success"]:
            return jsonify(result), 409

        return jsonify(result), 201

    except Exception as e:
        _logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Registration failed"
        }), 500
