import cv2
import time
import math
from config import ENTRY_CAMERA_INDEX, SHELF_CAMERA_INDEX, SHELF_MAP, HOST_IP, HOST_PORT
from state_manager import GlobalStateManager
from vision_pipeline import VisionSystem
from sensor_bridge import start_sensor_server
from checkout import calculate_bill, generate_qr_payment

# Initialize State and Vision
state_mgr = GlobalStateManager()
vision_sys = VisionSystem(state_mgr)

def handle_sensor_trigger(sensor_id, action):
    """Fired automatically when a load cell detects a weight change."""
    print(f"\n[EVENT TRIGGERED] Sensor: {sensor_id} | Action: {action}")
    
    product_data = SHELF_MAP.get(sensor_id)
    if not product_data:
        print(f"[ERROR] Unknown Sensor ID: {sensor_id}")
        return

    latest_tracks = vision_sys.get_latest_shelf_tracks()
    
    # Fallback for testing: Assign to Track 1 if no active track is visible
    if not latest_tracks:
        print("[INFO] No active camera track detected. Defaulting to Track ID 1 for testing.")
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

def main():
    # 1. Start HTTP Listener for Microcontroller
    start_sensor_server(callback=handle_sensor_trigger, host=HOST_IP, port=HOST_PORT)

    # 2. Open Cameras using Windows DirectShow backend
    cap_entry = cv2.VideoCapture(ENTRY_CAMERA_INDEX, cv2.CAP_DSHOW)
    cap_shelf = cv2.VideoCapture(SHELF_CAMERA_INDEX, cv2.CAP_DSHOW)

    # Set Resolution for stability
    cap_entry.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_entry.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_shelf.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_shelf.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[SYSTEM READY] Press 'q' to exit, 'c' to trigger checkout.")

    while True:
        ret_entry, frame_entry = cap_entry.read()
        ret_shelf, frame_shelf = cap_shelf.read()

        # Safely process Camera 1
        if ret_entry and frame_entry is not None:
            vision_sys.process_entry_camera(frame_entry)
            cv2.imshow("Camera 1 - Entry/Exit", frame_entry)

        # Safely process Camera 2
        if ret_shelf and frame_shelf is not None:
            vision_sys.process_shelf_camera(frame_shelf)
            cv2.imshow("Camera 2 - Shelf View", frame_shelf)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cust_id = "CUST_1"
            cart = state_mgr.get_cart(cust_id)
            total = calculate_bill(cart)
            generate_qr_payment(total, cust_id)

    cap_entry.release()
    cap_shelf.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()