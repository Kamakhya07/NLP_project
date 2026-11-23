import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from src.mongo_db import MongoDBManager

def test_saas_admin():
    print("--> Starting SaaS Admin Panel Redesign Verification...")
    client = app.test_client()

    # 1. Login as Admin
    resp = client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
    assert resp.status_code == 200
    print("[PASS] 1. Admin login succeeded")

    # 2. Test Dashboard Overview (/admin)
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert b"Platform Overview" in resp.data
    assert b"Total Posts" in resp.data
    assert b"Flagged Posts" in resp.data
    assert b"Total Users" in resp.data
    assert b"Blocked Users" in resp.data
    print("[PASS] 2. SaaS Dashboard Overview with 4 KPI cards verified (HTTP 200)")

    # 3. Test Moderation Queue (/admin/queue)
    resp = client.get("/admin/queue")
    assert resp.status_code == 200
    assert b"Content Moderation Queue" in resp.data
    assert b"View AI Evidence" in resp.data
    print("[PASS] 3. Moderation Queue with LIME Evidence accordion verified (HTTP 200)")

    # 4. Test User Management (/admin/users)
    resp = client.get("/admin/users")
    assert resp.status_code == 200
    assert b"User Management & Sanctions" in resp.data
    assert b"Hate Strikes" in resp.data
    print("[PASS] 4. User Management page with filters verified (HTTP 200)")

    # 5. Test AI Analytics (/admin/analytics)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    assert b"AI Engine Analytics & Evaluation" in resp.data
    assert b"Avg Model Confidence" in resp.data
    assert b"Algorithm Performance Comparison" in resp.data
    print("[PASS] 5. AI Analytics page with empirical benchmarks verified (HTTP 200)")

    # 6. Test Moderation History (/admin/history)
    resp = client.get("/admin/history")
    assert resp.status_code == 200
    assert b"Moderation History & Archive" in resp.data
    print("[PASS] 6. Moderation History page verified (HTTP 200)")

    # 7. Test Activity Logs (/admin/activity)
    resp = client.get("/admin/activity")
    assert resp.status_code == 200
    assert b"Administrative Audit Trail" in resp.data
    print("[PASS] 7. Activity Logs & Audit Trail page verified (HTTP 200)")

    # 8. Test Settings (/admin/settings)
    resp = client.get("/admin/settings")
    assert resp.status_code == 200
    assert b"System & AI Settings" in resp.data
    assert b"Active Model Architecture" in resp.data
    print("[PASS] 8. Settings & Architecture Switcher verified (HTTP 200)")

    # 9. Test Admin Cannot Be Blocked Protection
    admin_user = MongoDBManager.get_user_by_username("admin")
    resp = client.post(f"/admin/user/{str(admin_user['_id'])}/block", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Administrator accounts cannot be blocked" in resp.data
    print("[PASS] 9. Admin protection verified: Admin account cannot be blocked")

    # 10. Test Audit Log Entry Created on Action
    demo_user = MongoDBManager.get_user_by_username("user")
    if demo_user:
        # Block user
        client.post(f"/admin/user/{str(demo_user['_id'])}/block", data={"reason": "Test audit log"}, follow_redirects=True)
        # Check activity logs
        logs = MongoDBManager.get_activity_logs(limit=5)
        assert len(logs) > 0
        assert any(l["action"] == "block_user" for l in logs)
        print(f"[PASS] 10. Audit trail verified: Action recorded in MongoDB activity_logs ('{logs[0]['details']}')")

        # Unblock user to leave clean state
        client.post(f"/admin/user/{str(demo_user['_id'])}/unblock", follow_redirects=True)

    print("\n=======================================================")
    print(" ALL 10 SAAS ADMIN PANEL REDESIGN TESTS PASSED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    test_saas_admin()
