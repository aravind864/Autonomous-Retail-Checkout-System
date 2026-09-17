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
