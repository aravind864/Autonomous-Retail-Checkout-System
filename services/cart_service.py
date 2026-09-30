"""Shopping cart management service. Uses Decimal for all financial calculations."""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, Any

from database.connection import db
from database.models import ShoppingCartItem, Product, ShoppingSession
import config

logger = logging.getLogger(__name__)


def _quantize(value: Decimal) -> Decimal:
    """Round to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def add_item(session_id: str, product_id: str, quantity: int = 1,
             sensor_id: Optional[str] = None) -> Dict[str, Any]:
    """Add or increment a product in the cart."""
    product = Product.query.filter_by(product_id=product_id, active=True).first()
    if not product:
        return {"success": False, "message": f"Product {product_id} not found"}

    session = ShoppingSession.query.filter_by(session_id=session_id).first()
    if not session:
        return {"success": False, "message": f"Session {session_id} not found"}

    try:
        item = ShoppingCartItem.query.filter_by(
            session_id=session_id, product_id=product_id
        ).first()

        unit_price = Decimal(str(product.price))

        if item:
            item.quantity += quantity
            item.subtotal = _quantize(unit_price * Decimal(str(item.quantity)))
            if sensor_id:
                item.last_sensor_id = sensor_id
        else:
            subtotal = _quantize(unit_price * Decimal(str(quantity)))
            item = ShoppingCartItem(
                session_id=session_id,
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price,
                subtotal=subtotal,
                last_sensor_id=sensor_id,
            )
            db.session.add(item)

        db.session.commit()
        logger.info(f"[CART] CART UPDATED: +{quantity} {product.name} in session {session_id}")
        return {"success": True, "action": "ADD", "product_id": product_id,
                "quantity": item.quantity, "subtotal": str(item.subtotal)}

    except Exception as e:
        db.session.rollback()
        logger.error(f"[CART] Error adding item: {e}")
        return {"success": False, "message": "Failed to add item to cart"}


def remove_item(session_id: str, product_id: str, quantity: int = 1,
                sensor_id: Optional[str] = None) -> Dict[str, Any]:
    """Remove or decrement a product from the cart. Never allows negative quantity."""
    try:
        item = ShoppingCartItem.query.filter_by(
            session_id=session_id, product_id=product_id
        ).first()

        if not item:
            logger.warning(f"[CART] Cannot remove {product_id} from {session_id}: not in cart")
            return {"success": False, "message": "Product not in cart"}

        item.quantity = max(0, item.quantity - quantity)

        if item.quantity == 0:
            db.session.delete(item)
            db.session.commit()
            logger.info(f"[CART] CART UPDATED: removed {product_id} from session {session_id}")
            return {"success": True, "action": "REMOVE", "product_id": product_id,
                    "quantity": 0, "subtotal": "0.00"}

        unit_price = Decimal(str(item.unit_price))
        item.subtotal = _quantize(unit_price * Decimal(str(item.quantity)))
        if sensor_id:
            item.last_sensor_id = sensor_id
        db.session.commit()

        logger.info(f"[CART] CART UPDATED: -{quantity} {product_id} in session {session_id}")
        return {"success": True, "action": "REMOVE", "product_id": product_id,
                "quantity": item.quantity, "subtotal": str(item.subtotal)}

    except Exception as e:
        db.session.rollback()
        logger.error(f"[CART] Error removing item: {e}")
        return {"success": False, "message": "Failed to remove item from cart"}


def get_cart(session_id: str) -> Dict[str, Any]:
    """Retrieve complete cart contents with totals."""
    session = ShoppingSession.query.filter_by(session_id=session_id).first()
    if not session:
        return {"success": False, "message": "Session not found"}

    items = ShoppingCartItem.query.filter_by(session_id=session_id).all()

    item_list = []
    subtotal = Decimal("0.00")
    item_count = 0

    for item in items:
        if item.quantity > 0:
            item_dict = item.to_dict()
            item_list.append(item_dict)
            subtotal += Decimal(str(item.subtotal))
            item_count += item.quantity

    subtotal = _quantize(subtotal)
    tax_rate = Decimal(str(config.TAX_RATE))
    tax = _quantize(subtotal * tax_rate)
    total = _quantize(subtotal + tax)

    result = {
        "success": True,
        "session_id": session_id,
        "customer_id": session.customer_id,
        "items": item_list,
        "item_count": item_count,
        "subtotal": str(subtotal),
        "total": str(total),
    }

    if tax > 0:
        result["tax"] = str(tax)
        result["tax_rate"] = str(tax_rate)

    return result
