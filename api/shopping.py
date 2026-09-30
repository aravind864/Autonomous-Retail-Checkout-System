"""Shopping session, cart, and payment REST APIs."""

from flask import Blueprint, request, jsonify
import logging

from services.session_service import (
    get_active_session, get_session_by_id, process_payment, get_customer_history,
)
from services.cart_service import get_cart

shopping_api = Blueprint("shopping_api", __name__)
_logger = logging.getLogger(__name__)


@shopping_api.route("/api/shopping/customer/<customer_id>/active-session", methods=["GET"])
def active_session(customer_id: str):
    """Return the active shopping session and cart summary for a customer."""
    if not customer_id:
        return jsonify({"success": False, "message": "Customer ID is required"}), 400

    try:
        session = get_active_session(customer_id.strip())

        if not session:
            return jsonify({
                "success": True,
                "active": False,
                "session": None,
            }), 200

        cart_data = get_cart(session.session_id)
        session_dict = session.to_dict()
        session_dict["cart"] = {
            "items": cart_data.get("items", []),
            "item_count": cart_data.get("item_count", 0),
            "subtotal": cart_data.get("subtotal", "0.00"),
            "total": cart_data.get("total", "0.00"),
        }

        return jsonify({
            "success": True,
            "active": True,
            "session": session_dict,
        }), 200

    except Exception as e:
        _logger.error(f"Active session lookup error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to get session"}), 500


@shopping_api.route("/api/shopping/session/<session_id>", methods=["GET"])
def session_detail(session_id: str):
    """Return full session state including cart."""
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is required"}), 400

    try:
        session = get_session_by_id(session_id.strip())

        if not session:
            return jsonify({"success": False, "message": "Session not found"}), 404

        cart_data = get_cart(session.session_id)
        session_dict = session.to_dict()
        session_dict["cart"] = {
            "items": cart_data.get("items", []),
            "item_count": cart_data.get("item_count", 0),
            "subtotal": cart_data.get("subtotal", "0.00"),
            "total": cart_data.get("total", "0.00"),
        }

        return jsonify({"success": True, "session": session_dict}), 200

    except Exception as e:
        _logger.error(f"Session detail error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to get session"}), 500


@shopping_api.route("/api/shopping/session/<session_id>/cart", methods=["GET"])
def session_cart(session_id: str):
    """Return cart contents for a session."""
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is required"}), 400

    try:
        cart_data = get_cart(session_id.strip())

        if not cart_data.get("success"):
            return jsonify(cart_data), 404

        return jsonify(cart_data), 200

    except Exception as e:
        _logger.error(f"Cart lookup error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to get cart"}), 500


@shopping_api.route("/api/shopping/session/<session_id>/payment", methods=["POST"])
def payment(session_id: str):
    """Process payment for a shopping session.

    Server-authoritative: the client cannot simply declare payment complete.
    """
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is required"}), 400

    data = request.get_json(silent=True) or {}
    payment_method = data.get("payment_method", "APP")

    try:
        result = process_payment(session_id.strip(), payment_method)

        if not result.get("success"):
            return jsonify(result), 400

        _logger.info(f"[API] PAYMENT SUCCESSFUL: session={session_id} method={payment_method}")
        return jsonify(result), 200

    except Exception as e:
        _logger.error(f"Payment error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Payment processing failed"}), 500


@shopping_api.route("/api/shopping/customer/<customer_id>/history", methods=["GET"])
def customer_history(customer_id: str):
    """Return shopping history for a customer."""
    if not customer_id:
        return jsonify({"success": False, "message": "Customer ID is required"}), 400

    try:
        sessions = get_customer_history(customer_id.strip())
        history_list = []
        for s in sessions:
            cart_data = get_cart(s.session_id)
            s_dict = s.to_dict()
            s_dict["item_count"] = cart_data.get("item_count", 0)
            s_dict["total"] = cart_data.get("total", "0.00")
            s_dict["subtotal"] = cart_data.get("subtotal", "0.00")
            s_dict["items"] = cart_data.get("items", [])
            history_list.append(s_dict)

        return jsonify({
            "success": True,
            "customer_id": customer_id,
            "sessions": history_list,
            "count": len(history_list),
        }), 200

    except Exception as e:
        _logger.error(f"Shopping history error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to get shopping history"}), 500
