import psycopg2
from psycopg2.extras import RealDictCursor
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from contextlib import contextmanager
from flask import has_app_context
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE

db = SQLAlchemy()
_app = None


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


def get_app():
    """Return the initialized Flask application instance."""
    return _app


@contextmanager
def app_context_guard():
    """Context manager ensuring a valid Flask application context is active."""
    if has_app_context():
        yield
    elif _app is not None:
        with _app.app_context():
            yield
    else:
        yield


def init_db(app):
    """Initialize database connection with Flask app and auto-create tables.

    Enables the pgvector extension (for face embedding storage) and
    creates all tables defined by SQLAlchemy models on startup.
    """
    global _app
    _app = app
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
