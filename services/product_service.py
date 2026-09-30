"""Product catalog and sensor-product mapping service."""

import logging
from typing import Optional, List, Dict

from database.connection import db
from database.models import Product, SensorProductMapping

logger = logging.getLogger(__name__)


def get_product_by_id(product_id: str) -> Optional[Product]:
    """Retrieve a product by its product_id."""
    return Product.query.filter_by(product_id=product_id, active=True).first()


def get_product_for_sensor(sensor_id: str) -> Optional[Product]:
    """Look up the product mapped to a given sensor_id."""
    mapping = SensorProductMapping.query.filter_by(
        sensor_id=sensor_id, active=True
    ).first()
    if not mapping:
        logger.warning(f"[PRODUCT] No active mapping for sensor {sensor_id}")
        return None
    product = get_product_by_id(mapping.product_id)
    if product:
        logger.info(f"[PRODUCT] SENSOR MAPPED TO PRODUCT: {sensor_id} -> {product.product_id} ({product.name})")
    return product


def seed_test_data():
    """Seed development/test products and sensor mappings.

    Only inserts if data doesn't already exist. Safe to call multiple times.
    """
    test_products = [
        {"product_id": "PRODUCT_001", "sku": "SKU-APPLE-01", "name": "Apple (Fuji)",
         "price": 2.50, "unit_weight_g": 200.0},
        {"product_id": "PRODUCT_002", "sku": "SKU-BANANA-01", "name": "Banana Bunch",
         "price": 1.75, "unit_weight_g": 350.0},
        {"product_id": "PRODUCT_003", "sku": "SKU-MILK-01", "name": "Milk 1L",
         "price": 3.99, "unit_weight_g": 1050.0},
        {"product_id": "PRODUCT_004", "sku": "SKU-BREAD-01", "name": "Whole Wheat Bread",
         "price": 4.25, "unit_weight_g": 500.0},
    ]

    test_mappings = [
        {"sensor_id": "SENSOR_001", "product_id": "PRODUCT_001"},
        {"sensor_id": "SENSOR_002", "product_id": "PRODUCT_002"},
        {"sensor_id": "SENSOR_003", "product_id": "PRODUCT_003"},
        {"sensor_id": "SENSOR_004", "product_id": "PRODUCT_004"},
    ]

    try:
        for p in test_products:
            existing = Product.query.filter_by(product_id=p["product_id"]).first()
            if not existing:
                db.session.add(Product(**p))
                logger.info(f"[PRODUCT] Seeded test product: {p['product_id']} ({p['name']})")

        db.session.flush()

        for m in test_mappings:
            existing = SensorProductMapping.query.filter_by(sensor_id=m["sensor_id"]).first()
            if not existing:
                db.session.add(SensorProductMapping(**m))
                logger.info(f"[PRODUCT] Seeded sensor mapping: {m['sensor_id']} -> {m['product_id']}")

        db.session.commit()
        logger.info("[PRODUCT] Test data seeding complete.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"[PRODUCT] Error seeding test data: {e}")


def list_products(active_only: bool = True) -> List[Dict]:
    """Return list of products."""
    query = Product.query
    if active_only:
        query = query.filter_by(active=True)
    return [p.to_dict() for p in query.all()]


def list_sensor_mappings() -> List[Dict]:
    """Return list of sensor-product mappings."""
    mappings = SensorProductMapping.query.filter_by(active=True).all()
    return [{"sensor_id": m.sensor_id, "product_id": m.product_id} for m in mappings]
