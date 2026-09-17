import psycopg2
from psycopg2.extras import RealDictCursor
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE

db = SQLAlchemy()


def get_connection():
    """Create and return a raw PostgreSQL database connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DATABASE,
        cursor_factory=RealDictCursor
    )


def init_db(app):
    """Initialize database connection with Flask app and auto-create tables.

    Enables the pgvector extension (for face embedding storage) and
    creates all tables defined by SQLAlchemy models on startup.
    """
    db.init_app(app)
    with app.app_context():
        try:
            # Enable pgvector extension for Vector column support
            db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.warning(f"pgvector extension setup skipped: {exc}")
        try:
            db.create_all()
        except Exception as exc:
            app.logger.warning(f"Database initialization warning: {exc}")

