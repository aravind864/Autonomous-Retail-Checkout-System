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

# ─── Camera Utility (used only by admin test-camera API) ───
def normalize_camera_source(source):
    """Normalize common camera inputs (kept for admin API camera test)."""
    if source is None:
        return ""
    s = str(source).strip()
    if not s or "YOUR_PHONE_IP" in s:
        return ""
    if s.isdigit():
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

def reset_cameras():
    """Signal CameraStream threads to re-read config on next cycle.
    Camera streams are self-managed — they detect source changes automatically."""
    print("[INFO] reset_cameras() called — CameraStream threads will reload config on next cycle.")


# ─── Auth Decorator ───
def admin_required(f):
    """Decorator to protect admin routes behind session login."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

# ── Frame Generators ────────────────────────────────────────────────────────
# Each call to get_jpeg() composites: newest raw frame + cached YOLO boxes.
# Raw frame is from grab_thread (camera native FPS, no YOLO delay).
# YOLO boxes are from yolo_thread (updated async, ~1fps CPU / ~60fps GPU).
# Result: stream is always current — zero delay even when YOLO is slow.

def generate_entry_frames():
    while True:
        jpg = vision_sys.entry_get_jpeg()
        if jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.033)

def generate_shelf_frames():
    while True:
        jpg = vision_sys.shelf_get_jpeg()
        if jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.033)



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
    """Save camera URLs/indices directly to config.json and reset active streams."""
    data = request.json
    entry_cam = normalize_camera_source(data.get('entry_cam', ''))
    shelf_cam = normalize_camera_source(data.get('shelf_cam', ''))

    if not entry_cam or not shelf_cam:
        return jsonify({"status": "error", "message": "Both camera fields are required."}), 400

    try:
        save_camera_config(entry_cam, shelf_cam)
        reset_cameras()
        return jsonify({"status": "success", "message": "Camera config saved to config.json and live streams updated."})
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
    data = request.json or {}
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
        sensor_x, sensor_y = product_data.get("pos_x", 0), product_data.get("pos_y", 0)
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


@app.route('/api/dashboard_status')
@admin_required
def api_dashboard_status():
    """Returns live status of customer carts, tracks, and motion state for dashboard polling."""
    from checkout import calculate_bill
    carts = state_mgr.get_all_carts()
    
    formatted_carts = {}
    total_revenue = 0
    total_items_count = 0
    
    for cust_id, items in carts.items():
        subtotal = calculate_bill(items)
        formatted_carts[cust_id] = {
            "items": items,
            "subtotal": subtotal,
            "item_count": len(items)
        }
        total_revenue += subtotal
        total_items_count += len(items)

    # Read live motion state from the shared BatchedDetector
    with vision_sys.detector._det_lock:
        cam1_state = dict(vision_sys.detector._state["cam1"])
        cam2_state = dict(vision_sys.detector._state["cam2"])

    entry_moving = "MOTION" in cam1_state.get("badge", "")
    shelf_moving = "MOTION" in cam2_state.get("badge", "")

    return jsonify({
        "status": "online",
        "active_customers": len(carts),
        "total_items": total_items_count,
        "total_revenue": total_revenue,
        "carts": formatted_carts,
        "tracks": vision_sys.get_latest_shelf_tracks(),
        "motion_state": {
            "entry_camera_moving": entry_moving,
            "shelf_camera_moving": shelf_moving,
            "entry_badge": cam1_state.get("badge", ""),
            "shelf_badge": cam2_state.get("badge", "")
        }
    })


@app.route('/api/checkout/<customer_id>', methods=['GET', 'POST'])
@admin_required
def api_checkout(customer_id):
    """Generates dynamic itemized receipt and UPI QR payment code for customer."""
    from checkout import calculate_bill, generate_qr_base64
    cart = state_mgr.get_cart(customer_id)
    total = calculate_bill(cart)
    
    if request.method == 'POST':
        # Clear cart on successful payment confirmation
        if customer_id in state_mgr.shopping_carts:
            state_mgr.shopping_carts[customer_id] = []
        return jsonify({"status": "success", "message": f"Customer {customer_id} cart checked out and cleared."})
        
    qr_base64 = generate_qr_base64(total, customer_id)
    
    return jsonify({
        "customer_id": customer_id,
        "items": cart,
        "total_amount": total,
        "qr_code": qr_base64
    })


if __name__ == '__main__':
    print(f"[SYSTEM] Starting Autonomous Retail Web Server on http://{HOST_IP}:{HOST_PORT}")
    print(f"[ADMIN]  Admin Panel -> http://localhost:{HOST_PORT}/admin/login")
    if ADMIN_USER.startswith("__SET_") or ADMIN_PASS.startswith("__SET_"):
        print("[WARN] ADMIN_USER / ADMIN_PASS are not set. Configure environment variables before public deployment.")
    if isinstance(app.secret_key, (bytes, bytearray)):
        print("[INFO] Flask secret key is generated per process because FLASK_SECRET_KEY is not set.")
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True)

