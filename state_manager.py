import threading

class GlobalStateManager:
    def __init__(self):
        # Lock for thread safety across Flask and OpenCV threads
        self._lock = threading.Lock()
        # Maps CustomerID -> Cart Items (List of Products)
        self.shopping_carts = {}
        # Maps TrackID (from ByteTrack) -> CustomerID
        self.track_to_customer = {}

    def register_customer(self, customer_id):
        """Registers a new customer entering the store."""
        with self._lock:
            if customer_id not in self.shopping_carts:
                self.shopping_carts[customer_id] = []
                print(f"[STATE] Customer {customer_id} registered.")

    def link_track_to_customer(self, track_id, customer_id):
        """Links a physical person track to a recognized customer profile."""
        with self._lock:
            self.track_to_customer[track_id] = customer_id

    def update_cart(self, track_id, product, action="add"):
        """Updates the cart based on shelf interactions."""
        with self._lock:
            customer_id = self.track_to_customer.get(track_id)
            if not customer_id:
                print(f"[WARNING] Unrecognized track ID {track_id} interacting with shelf.")
                return

            item_name = product.get('product_name', 'Unknown Item')

            if action == "add":
                self.shopping_carts[customer_id].append(product)
                print(f"[CART] Added {item_name} to Customer {customer_id}'s cart.")
            elif action == "remove":
                if product in self.shopping_carts[customer_id]:
                    self.shopping_carts[customer_id].remove(product)
                    print(f"[CART] Removed {item_name} from Customer {customer_id}'s cart.")

    def get_cart(self, customer_id):
        with self._lock:
            return list(self.shopping_carts.get(customer_id, []))

    def get_all_carts(self):
        with self._lock:
            return {cid: list(items) for cid, items in self.shopping_carts.items()}