import json
import logging
import re
import numpy as np
import bcrypt

from database.connection import get_connection

logger = logging.getLogger(__name__)


def register_customer(first_name, last_name, email, password):
    """Register a new customer with hashed credentials and auto-generated customer ID."""
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM customers WHERE email = %s",
                (email,)
            )

            existing_customer = cursor.fetchone()
            if existing_customer:
                return {
                    "success": False,
                    "message": "Email already registered"
                }

            cursor.execute(
                """
                INSERT INTO customers
                (customer_id, first_name, last_name, email, password_hash)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    "TEMP",
                    first_name,
                    last_name,
                    email,
                    password_hash
                )
            )

            inserted = cursor.fetchone()
            customer_db_id = inserted["id"]
            customer_id = f"CUST{customer_db_id:06d}"

            cursor.execute(
                """
                UPDATE customers
                SET customer_id = %s
                WHERE id = %s
                """,
                (customer_id, customer_db_id)
            )

            connection.commit()

            return {
                "success": True,
                "customer_id": customer_id
            }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def login_customer(email, password):
    """Authenticate customer credentials against stored password hash."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT customer_id, first_name, last_name, email, password_hash
                FROM customers
                WHERE email = %s
                """,
                (email,)
            )

            customer = cursor.fetchone()

            if not customer:
                return {
                    "success": False,
                    "message": "Invalid email or password"
                }

            stored_hash = customer["password_hash"]
            if not stored_hash or not bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                return {
                    "success": False,
                    "message": "Invalid email or password"
                }

            return {
                "success": True,
                "customer_id": customer["customer_id"],
                "first_name": customer["first_name"],
                "last_name": customer["last_name"],
                "email": customer["email"]
            }

    finally:
        connection.close()


def get_face_status(customer_id):
    """Retrieve face enrollment status for a customer."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT customer_id, face_enrolled
                FROM customers
                WHERE customer_id = %s OR LOWER(customer_id) = LOWER(%s)
                LIMIT 1
                """,
                (customer_id, customer_id)
            )

            customer = cursor.fetchone()

            if not customer:
                match = re.match(r"^CUST(\d+)$", customer_id, re.IGNORECASE)
                if match:
                    padded_id = f"CUST{int(match.group(1)):06d}"
                    cursor.execute(
                        """
                        SELECT customer_id, face_enrolled
                        FROM customers
                        WHERE customer_id = %s
                        LIMIT 1
                        """,
                        (padded_id,)
                    )
                    customer = cursor.fetchone()

            if not customer:
                return {
                    "success": False,
                    "message": "Customer not found"
                }

            return {
                "success": True,
                "customer_id": customer_id,
                "face_enrolled": bool(customer["face_enrolled"]) if customer["face_enrolled"] is not None else False
            }

    finally:
        connection.close()


def enroll_face(customer_id, front_embedding, left_embedding, right_embedding):
    """Persist face embeddings (front, left, right) and mark customer enrolled."""
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, customer_id
                FROM customers
                WHERE customer_id = %s OR LOWER(customer_id) = LOWER(%s)
                LIMIT 1
                """,
                (customer_id, customer_id)
            )
            customer = cursor.fetchone()

            if not customer:
                match = re.match(r"^CUST(\d+)$", customer_id, re.IGNORECASE)
                if match:
                    padded_id = f"CUST{int(match.group(1)):06d}"
                    cursor.execute(
                        "SELECT id, customer_id FROM customers WHERE customer_id = %s LIMIT 1",
                        (padded_id,)
                    )
                    customer = cursor.fetchone()

            if not customer:
                return {
                    "success": False,
                    "message": f"Customer '{customer_id}' not found"
                }

            internal_cust_id = customer["id"]

            def format_vec(vec):
                if isinstance(vec, str):
                    return vec
                return json.dumps([float(x) for x in vec])

            front_vec_str = format_vec(front_embedding)
            left_vec_str = format_vec(left_embedding)
            right_vec_str = format_vec(right_embedding)

            cursor.execute(
                """
                INSERT INTO face_embeddings (customer_id, front_embedding, left_embedding, right_embedding, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (customer_id)
                DO UPDATE SET
                    front_embedding = EXCLUDED.front_embedding,
                    left_embedding = EXCLUDED.left_embedding,
                    right_embedding = EXCLUDED.right_embedding,
                    created_at = NOW()
                """,
                (internal_cust_id, front_vec_str, left_vec_str, right_vec_str)
            )

            cursor.execute(
                """
                UPDATE customers
                SET face_enrolled = TRUE
                WHERE id = %s
                """,
                (internal_cust_id,)
            )

            connection.commit()

            return {
                "success": True,
                "customer_id": customer["customer_id"],
                "face_enrolled": True,
                "message": "Face enrolled successfully"
            }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def identify_customer_by_face_embedding(embedding, limit=1):
    """Match face embedding against customer database using pgvector cosine similarity."""
    if embedding is None:
        return None

    if isinstance(embedding, np.ndarray):
        vec = embedding.flatten().astype(np.float32)
    else:
        vec = np.array(embedding, dtype=np.float32).flatten()

    if len(vec) != 512:
        return None

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm

    vec_str = json.dumps(vec.tolist())
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.customer_id,
                    c.first_name,
                    c.last_name,
                    GREATEST(
                        1.0 - (fe.front_embedding <=> %s::vector),
                        COALESCE(1.0 - (fe.left_embedding <=> %s::vector), -1.0),
                        COALESCE(1.0 - (fe.right_embedding <=> %s::vector), -1.0)
                    ) AS similarity
                FROM face_embeddings fe
                JOIN customers c ON c.id = fe.customer_id
                WHERE c.face_enrolled = TRUE
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (vec_str, vec_str, vec_str, limit)
            )

            row = cursor.fetchone()
            if row and row["similarity"] is not None:
                sim = float(row["similarity"])
                if not np.isnan(sim):
                    first_name = row["first_name"] or ""
                    last_name = row["last_name"] or ""
                    name = f"{first_name} {last_name}".strip()
                    return {
                        "customer_id": row["customer_id"],
                        "first_name": first_name,
                        "last_name": last_name,
                        "name": name if name else row["customer_id"],
                        "similarity": sim
                    }
            return None

    except Exception as e:
        logger.error(f"Error querying pgvector for face identification: {e}")
        return None

    finally:
        connection.close()
