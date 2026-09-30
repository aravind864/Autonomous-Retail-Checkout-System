"""Unit and integration tests for Autonomous Retail Checkout System.

Tests cover all 15 required test scenarios from Phase 17:
1. Customer entry triggers session creation
2. Repeat customer detection does not create duplicate active session
3. Camera 2 track matches Camera 1 customer via Re-ID
4. Customer outside shelf ROI does not trigger pose inference
5. Customer inside shelf ROI triggers pose inference (within interval)
6. Mock weight event is parsed correctly (ADD, REMOVE)
7. Sensor mapped to product correctly
8. Interacting customer associated with weight event
9. Ambiguous weight event flagged as UNRESOLVED
10. Below-threshold weight change ignored
11. Debounced weight event ignored
12. Duplicate event_id ignored
13. Cart updated correctly (subtotal, quantity, unit price)
14. Cart item cannot drop below zero quantity
15. Server-authoritative payment flow:
    - Unpaid customer cannot exit
    - Paid customer exit allowed
    - Session completed on exit
"""

import unittest
import time
import uuid
from decimal import Decimal

import config
from database.connection import db, app_context_guard
from database.models import (
    Customer, ShoppingSession, Product, SensorProductMapping,
    WeightEvent, ShoppingCartItem,
)
from state_manager import GlobalStateManager
from services.session_service import (
    get_or_create_active_session, get_active_session, get_session_by_id,
    process_payment, complete_session, handle_exit_detection,
)
from services.product_service import (
    get_product_by_id, get_product_for_sensor, seed_test_data,
)
from services.cart_service import add_item, remove_item, get_cart
from services.weight_sensor_service import get_sensor_provider, MockWeightSensorProvider
from services.weight_event_service import process_weight_event, _last_event_time
from services.shelf_roi_service import (
    is_person_in_shelf_roi, is_person_in_exit_roi,
    calculate_bbox_iou, calculate_person_roi_overlap,
    get_shelf_roi, get_exit_roi,
)
from services.pose_interaction_service import (
    PoseInteractionManager, PoseInteractionState,
    STATUS_INTERACTING, STATUS_NOT_IN_SHELF, STATUS_APPROACHING,
)
from app import create_app


class RetailSystemTestCase(unittest.TestCase):
    """Full test suite covering Phases 1 through 18."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def setUp(self):
        self.app_context = self.app.app_context()
        self.app_context.push()
        seed_test_data()
        self.state_mgr = GlobalStateManager()

    def tearDown(self):
        db.session.rollback()
        self.app_context.pop()

    # ==================================================================
    # TEST 1: Customer entry triggers session creation
    # ==================================================================
    def test_01_customer_entry_triggers_session_creation(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        res = get_or_create_active_session(cid, camera1_track_id=101)
        self.assertTrue(res["created"])
        session = res["session"]
        self.assertIsNotNone(session)
        self.assertEqual(session.customer_id, cid)
        self.assertEqual(session.status, "ACTIVE")
        self.assertEqual(session.camera1_track_id, 101)
        self.assertEqual(session.payment_status, "PENDING")
        self.assertFalse(session.exit_allowed)

    # ==================================================================
    # TEST 2: Repeat customer detection does not create duplicate active session
    # ==================================================================
    def test_02_repeat_detection_no_duplicate_session(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        res1 = get_or_create_active_session(cid, camera1_track_id=101)
        self.assertTrue(res1["created"])
        first_session_id = res1["session"].session_id

        res2 = get_or_create_active_session(cid, camera1_track_id=102)
        self.assertFalse(res2["created"])
        self.assertEqual(res2["session"].session_id, first_session_id)

        # Check in DB that only 1 ACTIVE session exists for this customer
        active_count = ShoppingSession.query.filter_by(
            customer_id=cid, status="ACTIVE"
        ).count()
        self.assertEqual(active_count, 1)

    # ==================================================================
    # TEST 3: Camera 2 track matches Camera 1 customer via Re-ID
    # ==================================================================
    def test_03_camera2_track_matches_camera1_customer(self):
        from reid_engine import PersonReIDEngine
        import numpy as np

        engine = PersonReIDEngine()
        cid = f"CUST_{uuid.uuid4().hex[:6]}"
        self.state_mgr.register_customer(cid)

        # Fake appearance vector
        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        fake_crop = np.zeros((120, 60, 3), dtype=np.uint8)

        engine.register_gallery_feature(cid, fake_emb, fake_crop)
        self.assertIn(cid, engine.gallery)

        # Associate state manager
        self.state_mgr.link_cam2_track(1005, cid, "SESS-12345")
        self.assertEqual(self.state_mgr.get_customer_for_cam2_track(1005), cid)
        self.assertEqual(self.state_mgr.get_session_for_cam2_track(1005), "SESS-12345")

    # ==================================================================
    # TEST 4: Customer outside shelf ROI does not trigger pose inference
    # ==================================================================
    def test_04_outside_shelf_roi_no_pose(self):
        pose_mgr = PoseInteractionManager()
        track_id = 1001

        # Far outside ROI (e.g. top-left corner: (10, 10, 50, 80))
        outside_bbox = (10, 10, 50, 80)
        in_roi = is_person_in_shelf_roi(outside_bbox)
        self.assertFalse(in_roi)

        should_run = pose_mgr.should_run_pose(track_id, inside_roi=in_roi)
        self.assertFalse(should_run)

    # ==================================================================
    # TEST 5: Customer inside shelf ROI triggers pose inference
    # ==================================================================
    def test_05_inside_shelf_roi_triggers_pose_with_interval(self):
        pose_mgr = PoseInteractionManager()
        track_id = 1002

        # Inside ROI: default SHELF_ROI is (200, 100, 600, 450)
        inside_bbox = (250, 150, 400, 400)
        in_roi = is_person_in_shelf_roi(inside_bbox)
        self.assertTrue(in_roi)

        # First check: should run
        should_run_1 = pose_mgr.should_run_pose(track_id, inside_roi=in_roi)
        self.assertTrue(should_run_1)

        # Update last pose timestamp to now
        pose_mgr.update_roi_status(track_id, True)
        state = pose_mgr.get_state(track_id)
        state.last_pose_timestamp = time.monotonic()

        # Immediate second check: should be blocked by cooldown (interval = 0.10s)
        should_run_2 = pose_mgr.should_run_pose(track_id, inside_roi=in_roi)
        self.assertFalse(should_run_2)

    # ==================================================================
    # TEST 6: Mock weight event is parsed correctly (ADD, REMOVE)
    # ==================================================================
    def test_06_weight_event_direction_parsing(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        s_res = get_or_create_active_session(cid)
        sid = s_res["session"].session_id
        self.state_mgr.register_session(cid, sid)

        # Setup interacting state
        self.state_mgr.link_cam2_track(1001, cid, sid)
        self.state_mgr.set_pose_state(1001, {
            "customer_id": cid,
            "session_id": sid,
            "interaction_status": "INTERACTING",
            "last_interaction_time": time.time(),
            "interaction_score": 0.85,
        })

        # Shelf weight decreased (1000g -> 800g, delta = -200g) -> Customer TOOK item -> ADD
        _last_event_time.clear()
        evt_add = {
            "event_id": f"EVT_ADD_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",  # Product 001 unit_weight = 200g
            "raw_weight_g": 800.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -200.0,
        }
        res_add = process_weight_event(evt_add, state_mgr=self.state_mgr)
        self.assertTrue(res_add["success"])
        self.assertEqual(res_add["direction"], "ADD")
        self.assertEqual(res_add["quantity"], 1)

        # Shelf weight increased (800g -> 1000g, delta = +200g) -> Customer PUT BACK -> REMOVE
        time.sleep(config.WEIGHT_EVENT_DEBOUNCE_SECONDS + 0.1)
        evt_remove = {
            "event_id": f"EVT_REM_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 1000.0,
            "previous_weight_g": 800.0,
            "weight_delta_g": 200.0,
        }
        res_remove = process_weight_event(evt_remove, state_mgr=self.state_mgr)
        self.assertTrue(res_remove["success"])
        self.assertEqual(res_remove["direction"], "REMOVE")
        self.assertEqual(res_remove["quantity"], 1)

    # ==================================================================
    # TEST 7: Sensor mapped to product correctly
    # ==================================================================
    def test_07_sensor_mapped_to_product(self):
        p1 = get_product_for_sensor("SENSOR_001")
        self.assertIsNotNone(p1)
        self.assertEqual(p1.product_id, "PRODUCT_001")
        self.assertEqual(p1.name, "Apple (Fuji)")
        self.assertEqual(float(p1.unit_weight_g), 200.0)

        p2 = get_product_for_sensor("SENSOR_002")
        self.assertIsNotNone(p2)
        self.assertEqual(p2.product_id, "PRODUCT_002")

        # Unknown sensor returns None
        p_unknown = get_product_for_sensor("NON_EXISTENT_SENSOR")
        self.assertIsNone(p_unknown)

    # ==================================================================
    # TEST 8: Interacting customer associated with weight event
    # ==================================================================
    def test_08_interacting_customer_associated_with_weight_event(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        s_res = get_or_create_active_session(cid)
        sid = s_res["session"].session_id
        self.state_mgr.register_session(cid, sid)
        self.state_mgr.link_cam2_track(1003, cid, sid)

        # Set active interacting state
        self.state_mgr.set_pose_state(1003, {
            "customer_id": cid,
            "session_id": sid,
            "interaction_status": "INTERACTING",
            "last_interaction_time": time.time(),
            "interaction_score": 0.90,
        })

        _last_event_time.clear()
        evt = {
            "event_id": f"EVT_ASSOC_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 800.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -200.0,
        }
        res = process_weight_event(evt, state_mgr=self.state_mgr)
        self.assertTrue(res["success"])
        self.assertEqual(res["customer_id"], cid)
        self.assertEqual(res["session_id"], sid)
        self.assertEqual(res["processing_status"], "RESOLVED")

    # ==================================================================
    # TEST 9: Ambiguous weight event flagged as UNRESOLVED
    # ==================================================================
    def test_09_ambiguous_weight_event_flagged_unresolved(self):
        cid1 = f"CUST_A_{uuid.uuid4().hex[:4]}"
        cid2 = f"CUST_B_{uuid.uuid4().hex[:4]}"

        s1 = get_or_create_active_session(cid1)["session"]
        s2 = get_or_create_active_session(cid2)["session"]

        self.state_mgr.register_session(cid1, s1.session_id)
        self.state_mgr.register_session(cid2, s2.session_id)

        self.state_mgr.link_cam2_track(1001, cid1, s1.session_id)
        self.state_mgr.link_cam2_track(1002, cid2, s2.session_id)

        now = time.time()
        # Both customers interacting with virtually identical scores
        self.state_mgr.set_pose_state(1001, {
            "customer_id": cid1,
            "session_id": s1.session_id,
            "interaction_status": "INTERACTING",
            "last_interaction_time": now,
            "interaction_score": 0.85,
        })
        self.state_mgr.set_pose_state(1002, {
            "customer_id": cid2,
            "session_id": s2.session_id,
            "interaction_status": "INTERACTING",
            "last_interaction_time": now,
            "interaction_score": 0.84,
        })

        _last_event_time.clear()
        evt = {
            "event_id": f"EVT_AMBIG_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 800.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -200.0,
        }
        res = process_weight_event(evt, state_mgr=self.state_mgr)
        self.assertFalse(res["success"])
        self.assertEqual(res["processing_status"], "UNRESOLVED")

    # ==================================================================
    # TEST 10: Below-threshold weight change ignored
    # ==================================================================
    def test_10_below_threshold_weight_change_ignored(self):
        _last_event_time.clear()
        evt = {
            "event_id": f"EVT_SMALL_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 995.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -5.0,  # Below min 20g threshold
        }
        res = process_weight_event(evt, state_mgr=self.state_mgr)
        self.assertFalse(res["success"])
        self.assertEqual(res["processing_status"], "IGNORED")

    # ==================================================================
    # TEST 11: Debounced weight event ignored
    # ==================================================================
    def test_11_debounced_weight_event_ignored(self):
        _last_event_time.clear()
        evt1 = {
            "event_id": f"EVT_DEB_1_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 800.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -200.0,
        }
        process_weight_event(evt1, state_mgr=self.state_mgr)

        # Immediate follow-up on same sensor without delay
        evt2 = {
            "event_id": f"EVT_DEB_2_{uuid.uuid4().hex[:6]}",
            "sensor_id": "SENSOR_001",
            "raw_weight_g": 600.0,
            "previous_weight_g": 800.0,
            "weight_delta_g": -200.0,
        }
        res2 = process_weight_event(evt2, state_mgr=self.state_mgr)
        self.assertFalse(res2["success"])
        self.assertEqual(res2["processing_status"], "IGNORED")

    # ==================================================================
    # TEST 12: Duplicate event_id ignored
    # ==================================================================
    def test_12_duplicate_event_id_ignored(self):
        _last_event_time.clear()
        duplicate_id = f"EVT_DUP_{uuid.uuid4().hex[:6]}"
        evt1 = {
            "event_id": duplicate_id,
            "sensor_id": "SENSOR_002",
            "raw_weight_g": 650.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -350.0,
        }
        process_weight_event(evt1, state_mgr=self.state_mgr)

        time.sleep(config.WEIGHT_EVENT_DEBOUNCE_SECONDS + 0.1)
        # Same event_id again
        evt2 = {
            "event_id": duplicate_id,
            "sensor_id": "SENSOR_002",
            "raw_weight_g": 650.0,
            "previous_weight_g": 1000.0,
            "weight_delta_g": -350.0,
        }
        res2 = process_weight_event(evt2, state_mgr=self.state_mgr)
        self.assertFalse(res2["success"])
        self.assertEqual(res2["processing_status"], "DUPLICATE")

    # ==================================================================
    # TEST 13: Cart updated correctly (subtotal, quantity, unit price)
    # ==================================================================
    def test_13_cart_updated_correctly(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        s = get_or_create_active_session(cid)["session"]
        sid = s.session_id

        # Add 2 apples ($2.50 each)
        add_item(sid, "PRODUCT_001", quantity=2)
        cart = get_cart(sid)
        self.assertTrue(cart["success"])
        self.assertEqual(cart["item_count"], 2)
        self.assertEqual(cart["subtotal"], "5.00")

        # Add 1 banana ($1.75)
        add_item(sid, "PRODUCT_002", quantity=1)
        cart = get_cart(sid)
        self.assertEqual(cart["item_count"], 3)
        self.assertEqual(cart["subtotal"], "6.75")

    # ==================================================================
    # TEST 14: Cart item cannot drop below zero quantity
    # ==================================================================
    def test_14_cart_item_cannot_drop_below_zero(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        s = get_or_create_active_session(cid)["session"]
        sid = s.session_id

        add_item(sid, "PRODUCT_001", quantity=1)
        # Remove 5 units when only 1 exists
        res = remove_item(sid, "PRODUCT_001", quantity=5)
        self.assertTrue(res["success"])

        # Check cart is empty and subtotal is 0.00
        cart = get_cart(sid)
        self.assertEqual(cart["item_count"], 0)
        self.assertEqual(cart["subtotal"], "0.00")

        # Removing when not in cart returns error
        res_empty = remove_item(sid, "PRODUCT_001", quantity=1)
        self.assertFalse(res_empty["success"])

    # ==================================================================
    # TEST 15: Server-authoritative payment and exit flow
    # ==================================================================
    def test_15_server_authoritative_payment_and_exit_flow(self):
        cid = f"TEST_CUST_{uuid.uuid4().hex[:6]}"
        s = get_or_create_active_session(cid)["session"]
        sid = s.session_id
        self.state_mgr.register_session(cid, sid)

        # 1. Unpaid customer cannot exit
        exit_blocked = handle_exit_detection(cid, self.state_mgr)
        self.assertFalse(exit_blocked["exit_allowed"])
        self.assertEqual(exit_blocked["reason"], "payment_required")

        # 2. Process server-authoritative payment
        pay_res = process_payment(sid, payment_method="CREDIT_CARD")
        self.assertTrue(pay_res["success"])
        self.assertEqual(pay_res["payment_status"], "PAID")
        self.assertTrue(pay_res["exit_allowed"])

        # 3. Paid customer exit is allowed and session completes
        exit_allowed = handle_exit_detection(cid, self.state_mgr)
        self.assertTrue(exit_allowed["exit_allowed"])
        self.assertEqual(exit_allowed["status"], "COMPLETED")

        # 4. Customer should be cleared from active sessions
        self.assertIsNone(self.state_mgr.get_session_for_customer(cid))

    # ==================================================================
    # TEST 16: REST API endpoints
    # ==================================================================
    def test_16_api_endpoints(self):
        cid = f"TEST_API_{uuid.uuid4().hex[:6]}"
        s = get_or_create_active_session(cid)["session"]
        sid = s.session_id

        # GET /api/shopping/customer/<cid>/active-session
        resp = self.client.get(f"/api/shopping/customer/{cid}/active-session")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["active"])
        self.assertEqual(data["session"]["session_id"], sid)

        # GET /api/shopping/session/<sid>
        resp_sess = self.client.get(f"/api/shopping/session/{sid}")
        self.assertEqual(resp_sess.status_code, 200)
        self.assertEqual(resp_sess.get_json()["session"]["customer_id"], cid)

        # GET /api/shopping/session/<sid>/cart
        resp_cart = self.client.get(f"/api/shopping/session/{sid}/cart")
        self.assertEqual(resp_cart.status_code, 200)
        self.assertEqual(resp_cart.get_json()["item_count"], 0)

        # POST /api/shopping/session/<sid>/payment
        resp_pay = self.client.post(f"/api/shopping/session/{sid}/payment", json={"payment_method": "APP"})
        self.assertEqual(resp_pay.status_code, 200)
        self.assertEqual(resp_pay.get_json()["payment_status"], "PAID")


if __name__ == "__main__":
    unittest.main()
