import time
import math

class ShelfSensor:
    def __init__(self, sensor_id, product_details, pin_dt, pin_sck):
        self.sensor_id = sensor_id
        self.product = product_details
        self.base_weight = 0
        self.current_weight = 0
        
        # Initialize HX711 (Requires RPi.GPIO and hx711 library on actual hardware)
        # self.hx = HX711(dout_pin=pin_dt, pd_sck_pin=pin_sck)
        # self.hx.tare()

    def check_weight_change(self):
        """Reads load cell and detects if an item was picked up or placed back."""
        # Mocking weight read
        # reading = self.hx.get_weight_mean(5)
        reading = self.current_weight # Placeholder
        
        weight_diff = reading - self.base_weight
        item_weight = self.product['weight_grams']
        
        if abs(weight_diff) >= item_weight * 0.8: # 80% threshold for variance
            if weight_diff < 0:
                return "item_taken"
            else:
                return "item_returned"
        return "no_change"

def find_nearest_track_id(pose_data, sensor_location):
    """
    Calculates the Euclidean distance between a person's hand (from YOLO-Pose)
    and the physical sensor location to assign the item to the correct person.
    """
    nearest_track = None
    min_dist = float('inf')
    
    # Mock logic: iterate through pose keypoints to find the closest hand to the shelf
    # dist = math.hypot(hand_x - sensor_location[0], hand_y - sensor_location[1])
    
    return nearest_track # Returns the track_id