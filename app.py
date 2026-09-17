import functools
import hmac
import logging
import os
import time
from urllib.parse import urlparse, urlunparse
import cv2
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for

import config
from config import Config, save_camera_config
from database import init_db
from database.connection import get_connection
from api.customer import customer_api
from api.auth import auth_api
from api.face import face_api
from state_manager import GlobalStateManager
from vision_pipeline import VisionSystem

state_mgr = GlobalStateManager()
vision_sys = VisionSystem(state_mgr)

ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PASS = os.getenv("ADMIN_PASS")


def normalize_camera_source(source):
    """Normalize camera input strings: digit indices, video files, or IP stream URLs."""
    if source is None:
        return ""
    s = str(source).strip().strip('"').strip("'")
    if not s or "YOUR_PHONE_IP" in s:
        return ""
    if s.isdigit():
        return s
    if s.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
        return s
    if not (s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://")):
        s = "http://" + s
    parsed = urlparse(s)
    if parsed.scheme in {"http", "https"}:
        path = parsed.path or ""
        if path in {"", "/"}:
            parsed = parsed._replace(path="/video")
            return urlunparse(parsed)
    return s


def admin_required(f):
    """Decorator to protect admin routes behind session login."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


def generate_entry_frames():
    while True:
        try:
            jpg = vision_sys.entry_get_jpeg()
            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        except Exception:
            pass
        time.sleep(0.033)


def generate_shelf_frames():
    while True:
        try:
            jpg = vision_sys.shelf_get_jpeg()
            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        except Exception:
            pass
        time.sleep(0.033)



def create_app(config_class=Config):
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    init_db(app)

    app.register_blueprint(customer_api)
    app.register_blueprint(auth_api)
    app.register_blueprint(face_api)

    @app.after_request
    def set_security_headers(response):
        """Inject security headers on every response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'"
        )
        return response

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "ShopFlow Retail Backend",
            "version": "2.0.0"
        }), 200

    @app.route("/api/test-db")
    @admin_required
    def test_db():
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS count FROM customers")
                result = cursor.fetchone()

            return {
                "success": True,
                "customers": result["count"]
            }
        finally:
            connection.close()

    @app.route('/')
    @admin_required
    def dashboard():
        return render_template('dashboard.html')

    @app.route('/video_feed_entry')
    @admin_required
    def video_feed_entry():
        return Response(generate_entry_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/video_feed_shelf')
    @admin_required
    def video_feed_shelf():
        return Response(generate_shelf_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/admin/login', methods=['GET', 'POST'])
    def admin_login():
        if session.get('admin_logged_in'):
            return redirect(url_for('admin_panel'))

        error = None
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            if (ADMIN_USER and ADMIN_PASS
                    and hmac.compare_digest(username, ADMIN_USER)
                    and hmac.compare_digest(password, ADMIN_PASS)):
                session['admin_logged_in'] = True
                return redirect(url_for('admin_panel'))
            else:
                error = "Invalid credentials. Please try again."

        return render_template('admin_login.html', error=error)

    @app.route('/admin/logout')
    def admin_logout():
        session.pop('admin_logged_in', None)
        return redirect(url_for('admin_login'))

    @app.route('/admin')
    @admin_required
    def admin_panel():
        cfg = config.reload()
        return render_template(
            'admin_panel.html',
            entry_cam=cfg.get("entry_camera", ""),
            shelf_cam=cfg.get("shelf_camera", ""),
            host_ip=cfg.get("host_ip", "0.0.0.0"),
            host_port=cfg.get("host_port", 5000)
        )

    @app.route('/admin/api/save-cameras', methods=['POST'])
    @admin_required
    def api_save_cameras():
        data = request.json or {}
        entry_cam = normalize_camera_source(data.get('entry_cam', ''))
        shelf_cam = normalize_camera_source(data.get('shelf_cam', ''))

        if not entry_cam or not shelf_cam:
            return jsonify({"status": "error", "message": "Both camera fields are required."}), 400

        try:
            save_camera_config(entry_cam, shelf_cam)
            return jsonify({"status": "success", "message": "Configuration saved."})
        except Exception as e:
            logging.getLogger(__name__).error(f"Camera config save failed: {e}")
            return jsonify({"status": "error", "message": "Failed to save configuration."}), 500

    @app.route('/admin/api/test-camera', methods=['POST'])
    @admin_required
    def api_test_camera():
        """Test if camera stream inputs can be opened."""
        data = request.json or {}
        entry_cam = normalize_camera_source(data.get('entry_cam', ''))
        shelf_cam = normalize_camera_source(data.get('shelf_cam', ''))

        def test_source(src):
            if not src:
                return False
            try:
                cap_arg = int(src) if src.isdigit() else src
                cap = cv2.VideoCapture(cap_arg)
                ok = False
                if cap.isOpened():
                    ret, _ = cap.read()
                    ok = ret
                cap.release()
                return ok
            except Exception:
                return False

        entry_ok = test_source(entry_cam)
        shelf_ok = test_source(shelf_cam)

        return jsonify({"entry_ok": entry_ok, "shelf_ok": shelf_ok})

    @app.route('/admin/api/current-config')
    @admin_required
    def api_current_config():
        cfg = config.reload()
        return jsonify({
            "entry_cam": cfg.get("entry_camera", ""),
            "shelf_cam": cfg.get("shelf_camera", ""),
            "host_ip": cfg.get("host_ip", "0.0.0.0"),
            "host_port": cfg.get("host_port", 5000)
        })

    @app.route('/api/dashboard_status')
    @admin_required
    def api_dashboard_status():
        """Returns live customer tracks and motion state for dashboard polling."""
        with vision_sys.detector._det_lock:
            cam1_state = dict(vision_sys.detector._state["cam1"])
            cam2_state = dict(vision_sys.detector._state["cam2"])

        entry_moving = "MOTION" in cam1_state.get("badge", "")
        shelf_moving = "MOTION" in cam2_state.get("badge", "")

        return jsonify({
            "status": "online",
            "active_customers": len(state_mgr.get_active_customers()),
            "tracks": vision_sys.get_latest_shelf_tracks(),
            "motion_state": {
                "entry_camera_moving": entry_moving,
                "shelf_camera_moving": shelf_moving,
                "entry_badge": cam1_state.get("badge", ""),
                "shelf_badge": cam2_state.get("badge", "")
            }
        })

    return app


app = create_app()

if __name__ == '__main__':
    host = config.HOST_IP
    port = config.HOST_PORT
    debug = app.config.get("DEBUG", False)
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
