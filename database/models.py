from datetime import datetime
from pgvector.sqlalchemy import Vector
from database.connection import db


class Customer(db.Model):
    """Customer database model for store visitors and registered shoppers."""
    __tablename__ = "customers"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(64), unique=True, nullable=True, index=True)
    customer_code = db.Column(db.String(64), unique=True, nullable=True, index=True)
    first_name = db.Column(db.String(128), nullable=True)
    last_name = db.Column(db.String(128), nullable=True)
    name = db.Column(db.String(128), nullable=True)
    email = db.Column(db.String(128), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=True)
    phone = db.Column(db.String(32), nullable=True)
    face_encoding = db.Column(db.Text, nullable=True)
    face_enrolled = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    embeddings = db.relationship("FaceEmbedding", backref="customer", uselist=False)

    def to_dict(self):
        """Serialize customer record to JSON-friendly dictionary (excludes sensitive fields)."""
        return {
            "id": self.id,
            "customer_id": self.customer_id or self.customer_code,
            "customer_code": self.customer_code or self.customer_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "name": self.name or f"{self.first_name or ''} {self.last_name or ''}".strip(),
            "email": self.email,
            "phone": self.phone,
            "face_enrolled": self.face_enrolled,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<Customer {self.customer_id or self.customer_code} - {self.name or self.first_name}>"


class FaceEmbedding(db.Model):
    """Face embedding vectors stored via pgvector for cosine similarity matching."""
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(
        db.Integer, db.ForeignKey("customers.id"), unique=True, nullable=False, index=True
    )
    front_embedding = db.Column(Vector(512), nullable=False)
    left_embedding = db.Column(Vector(512), nullable=True)
    right_embedding = db.Column(Vector(512), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<FaceEmbedding customer_id={self.customer_id}>"


# ====================================================================
# New models — Shopping, Products, Weight Events, Cart
# ====================================================================


class ShoppingSession(db.Model):
    """Persistent shopping session tracking a customer's store visit."""
    __tablename__ = "shopping_sessions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    customer_id = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="ACTIVE")
    camera1_track_id = db.Column(db.Integer, nullable=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    exit_detected_at = db.Column(db.DateTime, nullable=True)
    payment_status = db.Column(db.String(32), nullable=False, default="PENDING")
    exit_allowed = db.Column(db.Boolean, default=False, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)

    cart_items = db.relationship("ShoppingCartItem", backref="session", lazy="dynamic")
    weight_events = db.relationship("WeightEvent", backref="session", lazy="dynamic")

    # Valid statuses: ACTIVE, PENDING_PAYMENT, PAID, COMPLETED, CANCELLED

    def to_dict(self):
        """Serialize session to JSON-friendly dictionary."""
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "status": self.status,
            "camera1_track_id": self.camera1_track_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "exit_detected_at": self.exit_detected_at.isoformat() if self.exit_detected_at else None,
            "payment_status": self.payment_status,
            "exit_allowed": self.exit_allowed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Product(db.Model):
    """Store product catalog."""
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    sku = db.Column(db.String(64), unique=True, nullable=True)
    name = db.Column(db.String(256), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    unit_weight_g = db.Column(db.Float, nullable=False)
    active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Serialize product to JSON-friendly dictionary."""
        return {
            "product_id": self.product_id,
            "sku": self.sku,
            "name": self.name,
            "price": str(self.price),
            "unit_weight_g": self.unit_weight_g,
            "active": self.active,
        }


class SensorProductMapping(db.Model):
    """Maps a physical weight sensor to a specific product."""
    __tablename__ = "sensor_product_mappings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sensor_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    product_id = db.Column(db.String(64), db.ForeignKey("products.product_id"), nullable=False)
    active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    product = db.relationship("Product", backref="sensor_mappings")


class WeightEvent(db.Model):
    """Records a weight change event from a shelf sensor."""
    __tablename__ = "weight_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    event_id = db.Column(db.String(128), unique=True, nullable=False, index=True)
    sensor_id = db.Column(db.String(64), nullable=False, index=True)
    raw_weight_g = db.Column(db.Float, nullable=False)
    previous_weight_g = db.Column(db.Float, nullable=False)
    weight_delta_g = db.Column(db.Float, nullable=False)
    direction = db.Column(db.String(16), nullable=False)  # ADD, REMOVE, UNKNOWN
    event_timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    product_id = db.Column(db.String(64), nullable=True)
    customer_id = db.Column(db.String(64), nullable=True)
    session_id = db.Column(db.String(64), db.ForeignKey("shopping_sessions.session_id"), nullable=True)
    camera2_track_id = db.Column(db.Integer, nullable=True)
    processing_status = db.Column(db.String(32), nullable=False, default="PENDING")
    association_confidence = db.Column(db.Float, nullable=True)
    association_method = db.Column(db.String(128), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # processing_status: PENDING, RESOLVED, UNRESOLVED, DUPLICATE, IGNORED

    def to_dict(self):
        """Serialize weight event to JSON-friendly dictionary."""
        return {
            "event_id": self.event_id,
            "sensor_id": self.sensor_id,
            "raw_weight_g": self.raw_weight_g,
            "previous_weight_g": self.previous_weight_g,
            "weight_delta_g": self.weight_delta_g,
            "direction": self.direction,
            "event_timestamp": self.event_timestamp.isoformat() if self.event_timestamp else None,
            "product_id": self.product_id,
            "customer_id": self.customer_id,
            "session_id": self.session_id,
            "camera2_track_id": self.camera2_track_id,
            "processing_status": self.processing_status,
            "association_confidence": self.association_confidence,
            "association_method": self.association_method,
        }


class ShoppingCartItem(db.Model):
    """Individual item in a shopping session's cart."""
    __tablename__ = "shopping_cart_items"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(64), db.ForeignKey("shopping_sessions.session_id"), nullable=False, index=True)
    product_id = db.Column(db.String(64), db.ForeignKey("products.product_id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    unit_price = db.Column(db.Numeric(10, 2), nullable=False)
    subtotal = db.Column(db.Numeric(10, 2), nullable=False, default=0)
    last_sensor_id = db.Column(db.String(64), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    product = db.relationship("Product", backref="cart_items")

    __table_args__ = (
        db.UniqueConstraint('session_id', 'product_id', name='uq_session_product'),
    )

    def to_dict(self):
        """Serialize cart item to JSON-friendly dictionary."""
        return {
            "product_id": self.product_id,
            "name": self.product.name if self.product else self.product_id,
            "quantity": self.quantity,
            "unit_price": str(self.unit_price),
            "subtotal": str(self.subtotal),
            "last_sensor_id": self.last_sensor_id,
        }
