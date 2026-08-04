# Hardware & Camera Settings
ENTRY_CAMERA_INDEX = 0      # WebCam 1 (Entry/Exit)
SHELF_CAMERA_INDEX = 1      # WebCam 2 (Overhead Shelf)

# Physical Shelf Coordinates mapped to overhead camera view (pixels)
SHELF_MAP = {
    "SENSOR_01": {"product_name": "Lays Chips", "price": 20, "pos_x": 320, "pos_y": 240, "weight_g": 50},
    "SENSOR_02": {"product_name": "Coke Can",   "price": 35, "pos_x": 480, "pos_y": 240, "weight_g": 330},
}

# Sensor Server Settings
HOST_IP = "0.0.0.0"
HOST_PORT = 5000