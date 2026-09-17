from flask import Blueprint, request, jsonify
import logging

from services.customer_service import get_face_status, enroll_face
from services.face_service import process_enrollment_images

face_api = Blueprint("face_api", __name__)
_logger = logging.getLogger(__name__)


@face_api.route("/api/face/status/<customer_id>", methods=["GET"])
def face_status(customer_id):
    """Return face enrollment status for a customer."""
    if not customer_id:
        return jsonify({
            "success": False,
            "message": "Customer ID is required"
        }), 400

    try:
        result = get_face_status(customer_id.strip())

        if not result["success"]:
            return jsonify({
                "success": False,
                "message": result.get("message", "Customer not found")
            }), 404

        return jsonify({
            "customer_id": result["customer_id"],
            "face_enrolled": result["face_enrolled"]
        }), 200

    except Exception as e:
        _logger.error(f"Face status error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Failed to get face status"
        }), 500


@face_api.route("/api/face/enroll", methods=["POST"])
def enroll():
    """Enroll customer face via multipart images or pre-extracted embedding vectors."""
    if request.files or (request.content_type and "multipart/form-data" in request.content_type):
        customer_id = request.form.get("customer_id") or request.args.get("customer_id")
        if not customer_id:
            return jsonify({
                "success": False,
                "message": "customer_id form field is required"
            }), 400

        front_file = request.files.get("front_image") or request.files.get("front")
        left_file = request.files.get("left_image") or request.files.get("left")
        right_file = request.files.get("right_image") or request.files.get("right")

        if not front_file or not left_file or not right_file:
            return jsonify({
                "success": False,
                "message": "front_image, left_image, and right_image are all required"
            }), 400

        try:
            front_bytes = front_file.read()
            left_bytes = left_file.read()
            right_bytes = right_file.read()

            result = process_enrollment_images(
                customer_id=customer_id.strip(),
                front_bytes=front_bytes,
                left_bytes=left_bytes,
                right_bytes=right_bytes
            )

            status_code = 200 if result.get("success") else 400
            return jsonify(result), status_code

        except Exception as e:
            _logger.error(f"Enrollment image error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "message": "Failed to process enrollment images"
            }), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "message": "Request body or image files are required"
        }), 400

    customer_id = data.get("customer_id")
    front_embedding = data.get("front_embedding")
    left_embedding = data.get("left_embedding")
    right_embedding = data.get("right_embedding")

    if not customer_id:
        return jsonify({
            "success": False,
            "message": "customer_id is required"
        }), 400

    if not front_embedding or not left_embedding or not right_embedding:
        return jsonify({
            "success": False,
            "message": "front_embedding, left_embedding, and right_embedding are all required"
        }), 400

    if len(front_embedding) != 512 or len(left_embedding) != 512 or len(right_embedding) != 512:
        return jsonify({
            "success": False,
            "message": "All embeddings must be exactly 512-dimensional vectors"
        }), 400

    try:
        result = enroll_face(
            customer_id=str(customer_id).strip(),
            front_embedding=front_embedding,
            left_embedding=left_embedding,
            right_embedding=right_embedding
        )

        if not result["success"]:
            return jsonify(result), 404

        return jsonify(result), 200

    except Exception as e:
        _logger.error(f"Face enroll error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Failed to enroll face"
        }), 500
