from flask import Flask, render_template, Response, jsonify, request
import cv2
import math
import qrcode
import io
import base64

from config import ENTRY_CAMERA_INDEX, SHELF_CAMERA_INDEX, SHELF_MAP, HOST_IP, HOST_PORT
from state_manager import GlobalStateManager
from vision_pipeline import VisionSystem

app = Flask(__name__)

# Initialize core engines
state_mgr = GlobalStateManager()
vision_sys = VisionSystem(state_mgr)

# Initialize Camera Hardware
cap_entry = cv2.VideoCapture(ENTRY_CAMERA_INDEX, cv2.CAP_DSHOW)
cap_shelf = cv2.VideoCapture(SHELF_CAMERA_INDEX, cv2.CAP_DSHOW)

# --- Frame Generators for Web Video Streaming ---
def generate_entry_frames():
    while True:
        success, frame = cap_entry.read()
        if not success:
            break
        annotated_frame = vision_sys.process_entry_camera(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_shelf_frames():
    while True:
        success, frame = cap_shelf.read()
        if not success:
            break
        annotated_frame = vision_sys.process_shelf_camera(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---
@app.route('/')
def dashboard():
    """Renders Store Monitoring Dashboard."""
    return render_template('dashboard.html')

@app.route('/app')
def customer_app():
    """Renders Customer Mobile Application UI."""
    return render_template('customer_app.html')

@app.route('/video_feed_entry')
def video_feed_entry():
    return Response(generate_entry_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_shelf')
def video_feed_shelf():
    return Response(generate_shelf_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/api/cart/<customer_id>', methods=['GET'])
def get_cart(customer_id):
    """Returns active cart and dynamic payment QR code for mobile app."""
    cart = state_mgr.get_cart(customer_id)
    total = sum(item['price'] for item in cart)
    
    # Generate QR Code as Base64 image string
    qr_b64 = ""
    if total > 0:
        upi_url = f"upi://pay?pa=store@upi&pn=AutonomousRetail&am={total}"
        qr = qrcode.make(upi_url)
        buf = io.BytesIO()
        qr.save(buf, format='PNG')
        qr_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        "customer_id": customer_id,
        "cart": cart,
        "total": total,
        "qr_code_base64": qr_b64
    })

if __name__ == '__main__':
    print(f"[SYSTEM] Starting Autonomous Retail Web Server on http://{HOST_IP}:{HOST_PORT}")
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True)