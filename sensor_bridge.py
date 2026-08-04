from flask import Flask, request, jsonify
import threading

app = Flask(__name__)
event_callback = None

@app.route('/weight_event', methods=['POST'])
def handle_weight_event():
    """Receives JSON: {'sensor_id': 'SENSOR_01', 'action': 'item_taken'}"""
    data = request.json
    sensor_id = data.get('sensor_id')
    action = data.get('action') # 'item_taken' or 'item_returned'
    
    if event_callback and sensor_id:
        event_callback(sensor_id, action)
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "error", "message": "Handler unavailable"}), 400

def start_sensor_server(callback, host="0.0.0.0", port=5000):
    global event_callback
    event_callback = callback
    server_thread = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False))
    server_thread.daemon = True
    server_thread.start()
    print(f"[SENSOR BRIDGE] Listening for load cell updates on port {port}...")