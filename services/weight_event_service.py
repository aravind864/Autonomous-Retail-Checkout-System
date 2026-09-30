"""Weight event processing pipeline.

Handles the full flow: receive event -> validate -> find product ->
associate customer -> update cart.
"""

import logging
import time
from typing import Dict, Any

from database.connection import db
from database.models import WeightEvent
from services.product_service import get_product_for_sensor
from services.cart_service import add_item, remove_item
import config

logger = logging.getLogger(__name__)

# In-memory debounce tracker: sensor_id -> last_event_timestamp
_last_event_time: Dict[str, float] = {}


def process_weight_event(event_data: Dict[str, Any], state_mgr=None) -> Dict[str, Any]:
    """Process a weight change event through the full pipeline.

    Steps:
    1. Validate and deduplicate
    2. Determine direction (ADD/REMOVE)
    3. Map sensor to product
    4. Find the customer interacting near the sensor/shelf
    5. Update cart
    6. Persist event record
    """
    event_id = event_data.get("event_id")
    sensor_id = event_data.get("sensor_id")
    raw_weight_g = float(event_data.get("raw_weight_g", 0))
    previous_weight_g = float(event_data.get("previous_weight_g", 0))
    weight_delta_g = float(event_data.get("weight_delta_g", raw_weight_g - previous_weight_g))

    logger.info(f"[WEIGHT] WEIGHT EVENT RECEIVED: {event_id} sensor={sensor_id} delta={weight_delta_g}g")

    # --- Step 1: Check for duplicate event_id ---
    existing = WeightEvent.query.filter_by(event_id=event_id).first()
    if existing:
        logger.warning(f"[WEIGHT] Duplicate event_id: {event_id} — ignored")
        return {"success": False, "message": "Duplicate event_id", "processing_status": "DUPLICATE"}

    # --- Step 2: Debounce ---
    now = time.time()
    last_time = _last_event_time.get(sensor_id, 0)
    if now - last_time < config.WEIGHT_EVENT_DEBOUNCE_SECONDS:
        logger.info(f"[WEIGHT] Debounce: ignoring {event_id} for {sensor_id} (too soon)")
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, "UNKNOWN", processing_status="IGNORED")
        return {"success": False, "message": "Debounced", "processing_status": "IGNORED"}
    _last_event_time[sensor_id] = now

    # --- Step 3: Check minimum weight change ---
    abs_delta = abs(weight_delta_g)
    if abs_delta < config.WEIGHT_CHANGE_MIN_GRAMS:
        logger.info(f"[WEIGHT] Below threshold: {abs_delta}g < {config.WEIGHT_CHANGE_MIN_GRAMS}g")
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, "UNKNOWN", processing_status="IGNORED")
        return {"success": False, "message": "Below minimum weight threshold", "processing_status": "IGNORED"}

    # --- Step 4: Determine direction ---
    if weight_delta_g < -config.WEIGHT_STABILITY_TOLERANCE_GRAMS:
        # Weight decreased on shelf = customer TOOK item = ADD to cart
        direction = "ADD"
    elif weight_delta_g > config.WEIGHT_STABILITY_TOLERANCE_GRAMS:
        # Weight increased on shelf = customer PUT BACK item = REMOVE from cart
        direction = "REMOVE"
    else:
        direction = "UNKNOWN"
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, direction, processing_status="IGNORED")
        return {"success": False, "message": "Weight change within stability tolerance",
                "processing_status": "IGNORED"}

    # --- Step 5: Map sensor to product ---
    product = get_product_for_sensor(sensor_id)
    product_id = product.product_id if product else None
    product_name = product.name if product else None

    if not product:
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, direction, processing_status="UNRESOLVED")
        return {"success": False, "message": f"No product mapped to sensor {sensor_id}",
                "processing_status": "UNRESOLVED"}

    # --- Step 6: Compute quantity ---
    unit_weight = product.unit_weight_g
    quantity = max(1, round(abs_delta / unit_weight)) if unit_weight > 0 else 1

    # --- Step 7: Associate customer ---
    association = _associate_customer(sensor_id, state_mgr)
    customer_id = association.get("customer_id")
    session_id = association.get("session_id")
    cam2_track_id = association.get("cam2_track_id")
    confidence = association.get("confidence", 0)
    method = association.get("method", "NONE")

    if not customer_id or not session_id:
        logger.warning(f"[WEIGHT] AMBIGUOUS WEIGHT EVENT: {event_id} — no confident customer association")
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, direction, product_id=product_id,
                       processing_status="UNRESOLVED",
                       association_confidence=confidence,
                       association_method=method)
        return {
            "success": False,
            "message": "Could not associate weight event with a customer",
            "processing_status": "UNRESOLVED",
            "product_id": product_id,
            "direction": direction,
            "candidates": association.get("candidates", []),
        }

    # --- Step 8: Update cart ---
    if direction == "ADD":
        cart_result = add_item(session_id, product_id, quantity=quantity, sensor_id=sensor_id)
    else:
        cart_result = remove_item(session_id, product_id, quantity=quantity, sensor_id=sensor_id)

    if not cart_result.get("success"):
        _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                       weight_delta_g, direction, product_id=product_id,
                       customer_id=customer_id, session_id=session_id,
                       camera2_track_id=cam2_track_id,
                       processing_status="UNRESOLVED",
                       association_confidence=confidence,
                       association_method=method)
        return {"success": False, "message": cart_result.get("message", "Cart update failed"),
                "processing_status": "UNRESOLVED"}

    # --- Step 9: Persist resolved event ---
    _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                   weight_delta_g, direction, product_id=product_id,
                   customer_id=customer_id, session_id=session_id,
                   camera2_track_id=cam2_track_id,
                   processing_status="RESOLVED",
                   association_confidence=confidence,
                   association_method=method)

    logger.info(
        f"[WEIGHT] CUSTOMER ASSOCIATED WITH SENSOR: customer={customer_id} "
        f"session={session_id} sensor={sensor_id} product={product_id} "
        f"direction={direction} qty={quantity} confidence={confidence:.2f} method={method}"
    )

    return {
        "success": True,
        "processing_status": "RESOLVED",
        "event_id": event_id,
        "sensor_id": sensor_id,
        "product_id": product_id,
        "product_name": product_name,
        "direction": direction,
        "quantity": quantity,
        "customer_id": customer_id,
        "session_id": session_id,
        "camera2_track_id": cam2_track_id,
        "association_confidence": confidence,
        "association_method": method,
        "cart": cart_result,
    }


def _associate_customer(sensor_id: str, state_mgr) -> Dict[str, Any]:
    """Find the customer most likely responsible for a sensor weight change."""
    if state_mgr is None:
        return {"customer_id": None, "session_id": None, "confidence": 0, "method": "NONE"}

    candidates = state_mgr.get_interacting_customers_near_shelf()

    if not candidates:
        # Fallback: if only one active session exists, use it with lower confidence
        all_sessions = state_mgr.get_all_active_sessions()
        if len(all_sessions) == 1:
            cid, sid = next(iter(all_sessions.items()))
            return {
                "customer_id": cid,
                "session_id": sid,
                "cam2_track_id": None,
                "confidence": 0.4,
                "method": "SINGLE_ACTIVE_SESSION",
            }
        return {"customer_id": None, "session_id": None, "confidence": 0,
                "method": "NO_CANDIDATES", "candidates": []}

    now = time.time()
    scored = []
    for c in candidates:
        status_score = 1.0 if c["interaction_status"] == "INTERACTING" else 0.5
        time_delta = now - c.get("last_interaction_time", 0)
        temporal_score = max(0, 1.0 - time_delta / config.WEIGHT_EVENT_ASSOCIATION_WINDOW_SECONDS)
        interaction_score = c.get("interaction_score", 0)

        total = 0.4 * status_score + 0.35 * temporal_score + 0.25 * interaction_score
        scored.append({**c, "total_score": total})

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    best = scored[0]

    # Check for ambiguity
    if len(scored) > 1:
        second = scored[1]
        margin = best["total_score"] - second["total_score"]
        if margin < 0.15:
            logger.warning(
                f"[WEIGHT] AMBIGUOUS: top candidates {best['customer_id']} ({best['total_score']:.2f}) "
                f"vs {second['customer_id']} ({second['total_score']:.2f})"
            )
            return {
                "customer_id": None, "session_id": None, "confidence": 0,
                "method": "AMBIGUOUS",
                "candidates": [{"customer_id": s["customer_id"], "score": s["total_score"]} for s in scored],
            }

    if best["total_score"] < config.WEIGHT_ASSOCIATION_MIN_CONFIDENCE:
        return {
            "customer_id": None, "session_id": None,
            "confidence": best["total_score"],
            "method": "BELOW_THRESHOLD",
            "candidates": [{"customer_id": s["customer_id"], "score": s["total_score"]} for s in scored],
        }

    return {
        "customer_id": best["customer_id"],
        "session_id": best["session_id"],
        "cam2_track_id": best.get("cam2_track_id"),
        "confidence": best["total_score"],
        "method": "POSE+ROI+TEMPORAL_PROXIMITY",
    }


def _persist_event(event_id, sensor_id, raw_weight_g, previous_weight_g,
                   weight_delta_g, direction, product_id=None,
                   customer_id=None, session_id=None, camera2_track_id=None,
                   processing_status="PENDING", association_confidence=None,
                   association_method=None):
    """Persist a weight event record to the database."""
    try:
        evt = WeightEvent(
            event_id=event_id,
            sensor_id=sensor_id,
            raw_weight_g=raw_weight_g,
            previous_weight_g=previous_weight_g,
            weight_delta_g=weight_delta_g,
            direction=direction,
            product_id=product_id,
            customer_id=customer_id,
            session_id=session_id,
            camera2_track_id=camera2_track_id,
            processing_status=processing_status,
            association_confidence=association_confidence,
            association_method=association_method,
        )
        db.session.add(evt)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"[WEIGHT] Failed to persist event {event_id}: {e}")
