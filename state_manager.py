import threading


class GlobalStateManager:
    """Thread-safe tracker for active store customers and ByteTrack associations."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active_customers = set()
        self.track_to_customer = {}

    def register_customer(self, customer_id: str):
        with self._lock:
            if customer_id not in self.active_customers:
                self.active_customers.add(customer_id)
                print(f"[STATE] Customer {customer_id} registered.")

    def link_track_to_customer(self, track_id: int, customer_id: str):
        with self._lock:
            self.track_to_customer[track_id] = customer_id

    def get_active_customers(self) -> list:
        with self._lock:
            return list(self.active_customers)

    def get_customer_for_track(self, track_id: int):
        with self._lock:
            return self.track_to_customer.get(track_id)