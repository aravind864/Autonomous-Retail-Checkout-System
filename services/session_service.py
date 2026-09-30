"""Shopping session management service."""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from database.connection import db
from database.models import ShoppingSession

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"SESS-{uuid.uuid4().hex[:12].upper()}"


def get_or_create_active_session(customer_id: str, camera1_track_id: Optional[int] = None) -> Dict[str, Any]:
    """Get existing active session or create a new one for the customer.

    Thread-safe: uses database uniqueness to prevent duplicates.
    If the customer already has an ACTIVE session, returns it.
    Otherwise creates a new one.
    """
    try:
        existing = ShoppingSession.query.filter_by(
            customer_id=customer_id, status="ACTIVE"
        ).first()

        if existing:
            if camera1_track_id is not None and existing.camera1_track_id != camera1_track_id:
                existing.camera1_track_id = camera1_track_id
                existing.updated_at = datetime.utcnow()
                db.session.commit()
            return {"session": existing, "created": False}

        pending = ShoppingSession.query.filter_by(
            customer_id=customer_id, status="PENDING_PAYMENT"
        ).first()
        if pending:
            return {"session": pending, "created": False}

        session_id = generate_session_id()
        new_session = ShoppingSession(
            session_id=session_id,
            customer_id=customer_id,
            status="ACTIVE",
            camera1_track_id=camera1_track_id,
            payment_status="PENDING",
            exit_allowed=False,
        )
        db.session.add(new_session)
        db.session.commit()

        logger.info(f"[SESSION] SESSION CREATED: {session_id} for customer {customer_id}")
        return {"session": new_session, "created": True}

    except Exception as e:
        db.session.rollback()
        logger.error(f"[SESSION] Error creating session for {customer_id}: {e}")
        raise


def get_active_session(customer_id: str) -> Optional[ShoppingSession]:
    """Get the active or pending-payment session for a customer."""
    return ShoppingSession.query.filter(
        ShoppingSession.customer_id == customer_id,
        ShoppingSession.status.in_(["ACTIVE", "PENDING_PAYMENT", "PAID"])
    ).first()


def get_session_by_id(session_id: str) -> Optional[ShoppingSession]:
    """Retrieve a session by its session_id."""
    return ShoppingSession.query.filter_by(session_id=session_id).first()


def get_customer_history(customer_id: str):
    """Retrieve all shopping sessions for a customer, newest first."""
    return ShoppingSession.query.filter_by(
        customer_id=customer_id
    ).order_by(ShoppingSession.started_at.desc()).all()


def process_payment(session_id: str, payment_method: str = "APP") -> Dict[str, Any]:
    """Process payment for a shopping session. Server-authoritative."""
    session = get_session_by_id(session_id)
    if not session:
        return {"success": False, "message": "Session not found"}

    if session.status not in ("ACTIVE", "PENDING_PAYMENT"):
        return {"success": False, "message": f"Session is {session.status}, cannot process payment"}

    if session.payment_status == "PAID":
        return {"success": True, "session_id": session_id, "payment_status": "PAID",
                "exit_allowed": session.exit_allowed, "message": "Already paid"}

    try:
        session.status = "PAID"
        session.payment_status = "PAID"
        session.exit_allowed = True
        session.updated_at = datetime.utcnow()
        db.session.commit()

        logger.info(f"[SESSION] PAYMENT SUCCESSFUL: {session_id} via {payment_method}")
        return {
            "success": True,
            "session_id": session_id,
            "payment_status": "PAID",
            "exit_allowed": True,
        }
    except Exception as e:
        db.session.rollback()
        logger.error(f"[SESSION] Payment error for {session_id}: {e}")
        return {"success": False, "message": "Payment processing failed"}


def complete_session(session_id: str) -> Dict[str, Any]:
    """Mark a session as completed (customer has exited)."""
    session = get_session_by_id(session_id)
    if not session:
        return {"success": False, "message": "Session not found"}

    try:
        session.status = "COMPLETED"
        session.exit_detected_at = datetime.utcnow()
        session.completed_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        db.session.commit()

        logger.info(f"[SESSION] SESSION COMPLETED: {session_id}")
        return {"success": True, "session_id": session_id, "status": "COMPLETED"}
    except Exception as e:
        db.session.rollback()
        logger.error(f"[SESSION] Complete error for {session_id}: {e}")
        return {"success": False, "message": "Failed to complete session"}


def handle_exit_detection(customer_id: str, state_mgr=None) -> Dict[str, Any]:
    """Handle exit detection for a verified customer.

    If paid -> complete session, allow exit.
    If unpaid -> log exit attempt, deny exit.
    """
    session = get_active_session(customer_id)
    if not session:
        logger.warning(f"[SESSION] EXIT ATTEMPT: No active session for {customer_id}")
        return {"exit_allowed": False, "reason": "no_active_session"}

    if session.payment_status == "PAID":
        result = complete_session(session.session_id)
        if state_mgr:
            state_mgr.remove_session(customer_id)
        logger.info(f"[SESSION] EXIT ALLOWED: {customer_id} session {session.session_id}")
        return {"exit_allowed": True, "session_id": session.session_id, "status": "COMPLETED"}
    else:
        session.exit_detected_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        db.session.commit()
        logger.warning(
            f"[SESSION] EXIT BLOCKED: {customer_id} session {session.session_id} "
            f"— payment_status={session.payment_status}"
        )
        return {"exit_allowed": False, "session_id": session.session_id, "reason": "payment_required"}
