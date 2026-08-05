from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2
import math
import qrcode
import io
import base64
import functools
import os
import time
from urllib.parse import urlparse, urlunparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config

from config import ENTRY_CAMERA_INDEX, SHELF_CAMERA_INDEX, SHELF_MAP, HOST_IP, HOST_PORT
from config import save_camera_config
from state_manager import GlobalStateManager
from vision_pipeline import VisionSystem

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)

# --- Admin Credentials ---
ADMIN_USER = os.getenv("ADMIN_USER", "__SET_ADMIN_USER__")
ADMIN_PASS = os.getenv("ADMIN_PASS", "__SET_ADMIN_PASS__")
# Initialize core engines
state_mgr = GlobalStateManager()
vision_sys = VisionSystem(state_mgr)

# ─── Camera Management ───
cap_entry = None
cap_shelf = None

def normalize_camera_source(source):
    """Normalize common camera inputs.

    - USB webcams stay numeric.
    - IP Webcam base URLs such as http://host:8080 become http://host:8080/video.
    """
    s = str(source).strip()
    if s.isdigit():
        return s

    parsed = urlparse(s)
    if parsed.scheme in {"http", "https"}:
        path = parsed.path or ""
        if path in {"", "/"}:
            parsed = parsed._replace(path="/video")
            return urlunparse(parsed)

    return s


def open_camera(source):
    """Open a camera source with the correct backend (USB vs IP)."""
    if source is None or str(source).strip() == "":
        return None
    s = normalize_camera_source(source)
    if s.isdigit():
        return cv2.VideoCapture(int(s), cv2.CAP_DSHOW)
    else:
        return cv2.VideoCapture(s)

def get_cameras():
    """Return current camera captures, reopening if needed from config.json."""
    global cap_entry, cap_shelf
    cfg = config.reload()
    entry_src = cfg.get("entry_camera", "")
    shelf_src = cfg.get("shelf_camera", "")

    # Reopen entry camera if not opened or not working
    if cap_entry is None or not cap_entry.isOpened():
        if entry_src:
            cap_entry = open_camera(entry_src)

    # Reopen shelf camera if not opened or not working
    if cap_shelf is None or not cap_shelf.isOpened():
        if shelf_src:
            cap_shelf = open_camera(shelf_src)

    return cap_entry, cap_shelf

# ─── Auth Decorator ───
def admin_required(f):
    """Decorator to protect admin routes behind session login."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

# --- Frame Generators for Web Video Streaming ---
def generate_entry_frames():
    while True:
        entry, _ = get_cameras()
        if entry is None or not entry.isOpened():
            time.sleep(0.5)
            continue
        success, frame = entry.read()
        if not success:
            print("[WARN] Entry camera read failed. Reconnecting...")
            try:
                entry.release()
            except Exception:
                pass
            global cap_entry
            cap_entry = None
            time.sleep(0.5)
            continue
        annotated_frame = vision_sys.process_entry_camera(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_shelf_frames():
    while True:
        _, shelf = get_cameras()
        if shelf is None or not shelf.isOpened():
            time.sleep(0.5)
            continue
        success, frame = shelf.read()
        if not success:
            print("[WARN] Shelf camera read failed. Reconnecting...")
            try:
                shelf.release()
            except Exception:
                pass
            global cap_shelf
            cap_shelf = None
            time.sleep(0.5)
            continue
        annotated_frame = vision_sys.process_shelf_camera(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---
@app.route('/')
@admin_required
def dashboard():
    """Renders Store Monitoring Dashboard."""
    return render_template('dashboard.html')

@app.route('/video_feed_entry')
@admin_required
def video_feed_entry():
    return Response(generate_entry_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_shelf')
@admin_required
def video_feed_shelf():
    return Response(generate_shelf_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ═══════════════════════════════════════════════════════
#  ADMIN PANEL ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page — credentials: admin / admin."""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_panel'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == ADMIN_USER and password == ADMIN_PASS:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_panel'))
        else:
            error = "Invalid credentials. Please try again."

    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    """Logs out admin and redirects to login page."""
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_panel():
    """Admin configuration panel — camera IPs, shelf map, server info."""
    cfg = config.reload()
    return render_template('admin_panel.html',
                           entry_cam=cfg["entry_camera"],
                           shelf_cam=cfg["shelf_camera"],
                           shelf_map=cfg["shelf_map"],
                           host_ip=cfg["host_ip"],
                           host_port=cfg["host_port"])

@app.route('/admin/api/save-cameras', methods=['POST'])
@admin_required
def api_save_cameras():
    """Save camera URLs/indices to config.local.json."""
    data = request.json
    entry_cam = data.get('entry_cam', '').strip()
    shelf_cam = data.get('shelf_cam', '').strip()

    if not entry_cam or not shelf_cam:
        return jsonify({"status": "error", "message": "Both camera fields are required."}), 400

    try:
        save_camera_config(entry_cam, shelf_cam)
        return jsonify({"status": "success", "message": "Camera config saved."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/api/test-cameras', methods=['POST'])
@admin_required
def api_test_cameras():
    """Quick connectivity test for camera URLs."""
    data = request.json
    entry_cam = data.get('entry_cam', '').strip()
    shelf_cam = data.get('shelf_cam', '').strip()

    def test_source(source):
        try:
            s = normalize_camera_source(source)
            if s.isdigit():
                cap = cv2.VideoCapture(int(s), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(s)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                ok = ret
            cap.release()
            return ok
        except Exception:
            return False

    entry_ok = test_source(entry_cam)
    shelf_ok = test_source(shelf_cam)

    return jsonify({
        "entry_ok": entry_ok,
        "shelf_ok": shelf_ok
    })

@app.route('/admin/api/current-config')
@admin_required
def api_current_config():
    """Return current config values (live reload from config files)."""
    cfg = config.reload()
    return jsonify({
        "entry_cam": cfg["entry_camera"],
        "shelf_cam": cfg["shelf_camera"],
        "host_ip": cfg["host_ip"],
        "host_port": cfg["host_port"]
    })

# --- API Endpoints ---
@app.route('/weight_event', methods=['POST'])
def handle_weight_event():
    """Endpoint for ESP32 / Load Cell HTTP requests."""
    data = request.json
    sensor_id = data.get('sensor_id')
    action = data.get('action')
    
    product_data = SHELF_MAP.get(sensor_id)
    if not product_data:
        return jsonify({"status": "error", "message": "Unknown Sensor"}), 400

    latest_tracks = vision_sys.get_latest_shelf_tracks()
    
    if not latest_tracks:
        closest_track_id = 1
        if 1 not in state_mgr.track_to_customer:
            state_mgr.register_customer("CUST_1")
            state_mgr.link_track_to_customer(1, "CUST_1")
    else:
        sensor_x, sensor_y = product_data["pos_x"], product_data["pos_y"]
        closest_track_id = None
        min_distance = float('inf')
        for track_id, pos in latest_tracks.items():
            dist = math.hypot(pos[0] - sensor_x, pos[1] - sensor_y)
            if dist < min_distance:
                min_distance = dist
                closest_track_id = track_id

    if closest_track_id is not None:
        cart_action = "add" if action == "item_taken" else "remove"
        state_mgr.update_cart(closest_track_id, product_data, action=cart_action)

    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    print(f"[SYSTEM] Starting Autonomous Retail Web Server on http://{HOST_IP}:{HOST_PORT}")
    print(f"[ADMIN]  Admin Panel -> http://localhost:{HOST_PORT}/admin/login")
    if ADMIN_USER.startswith("__SET_") or ADMIN_PASS.startswith("__SET_"):
        print("[WARN] ADMIN_USER / ADMIN_PASS are not set. Configure environment variables before public deployment.")
    if isinstance(app.secret_key, (bytes, bytearray)):
        print("[INFO] Flask secret key is generated per process because FLASK_SECRET_KEY is not set.")
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True)
