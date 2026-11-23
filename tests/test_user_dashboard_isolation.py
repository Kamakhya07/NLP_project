"""
Test User Data Isolation on Dashboard:
Verifies that specific users can only see their own prediction data and
analytics on /dashboard, and never data from other users.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from src.mongo_db import MongoDBManager

def run_tests():
    print("--> Starting User Data Dashboard Isolation Verification...")
    client = app.test_client()

    # 1. Clean previous test artifacts for both demo users
    user_a = MongoDBManager.get_user_by_username("user")
    user_b = MongoDBManager.get_user_by_username("sarah_m")
    assert user_a is not None, "Demo user 'user' not found"
    assert user_b is not None, "Demo user 'sarah_m' not found"

    uid_a = str(user_a["_id"])
    uid_b = str(user_b["_id"])

    MongoDBManager.clear_user_prediction_history(uid_a)
    MongoDBManager.clear_user_prediction_history(uid_b)

    # 2. Login as User A (Alex Rivera)
    resp = client.get("/quick-login/user", follow_redirects=True)
    assert resp.status_code == 200

    # User A initially has 0 evaluations
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"No Evaluations Recorded for @user Yet" in resp.data or b"Private to @user" in resp.data
    print("[PASS] 1. User A sees their own empty dashboard without other users' data")

    # 3. User A submits a unique evaluation
    unique_text_a = "Hello this is a special test comment uniquely created by Alex Rivera 12345"
    resp = client.post("/analyze", data={"text": unique_text_a, "model_type": "logistic_regression"}, follow_redirects=True)
    assert resp.status_code == 200

    # User A views dashboard -> Must see their evaluation
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Alex Rivera 12345" in resp.data
    assert b"Total Evaluated" in resp.data
    assert b"Private to @user" in resp.data
    print("[PASS] 2. User A sees their own evaluation on /dashboard")

    # 4. Logout User A and Login as User B (Sarah Miller)
    client.get("/logout", follow_redirects=True)

    resp = client.get("/quick-login/sarah_m", follow_redirects=True)
    assert resp.status_code == 200

    # User B views dashboard -> MUST NOT see User A's unique text!
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Alex Rivera 12345" not in resp.data, "CRITICAL: User B saw User A's private data!"
    assert b"Private to @sarah_m" in resp.data
    assert b"No Evaluations Recorded for @sarah_m Yet" in resp.data
    print("[PASS] 3. Data Isolation Verified: User B CANNOT see User A's prediction history or stats")

    # 5. User B submits their own unique evaluation
    unique_text_b = "AI research notes from Sarah Miller for cyber safety testing 998877"
    resp = client.post("/analyze", data={"text": unique_text_b, "model_type": "logistic_regression"}, follow_redirects=True)
    assert resp.status_code == 200

    # User B views dashboard -> Must see User B's text, but NOT User A's
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Sarah Miller for cyber safety" in resp.data
    assert b"Alex Rivera 12345" not in resp.data, "CRITICAL: User A's data leaked to User B's dashboard!"
    print("[PASS] 4. User B sees their own evaluation only")

    # 6. Switch back to User A -> Must see User A's text, but NOT User B's
    client.get("/logout", follow_redirects=True)
    client.get("/quick-login/user", follow_redirects=True)

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Alex Rivera 12345" in resp.data
    assert b"Sarah Miller for cyber safety" not in resp.data, "CRITICAL: User B's data leaked to User A's dashboard!"
    print("[PASS] 5. Symmetrical Isolation: User A cannot see User B's data either")

    # 7. User A clears their history
    resp = client.post("/clear-history", follow_redirects=True)
    assert resp.status_code == 200

    # User A now has empty dashboard
    resp = client.get("/dashboard")
    assert b"Alex Rivera 12345" not in resp.data

    # User B logs in -> User B's history must still exist unharmed
    client.get("/logout", follow_redirects=True)
    client.get("/quick-login/sarah_m", follow_redirects=True)

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Sarah Miller for cyber safety" in resp.data, "User B's history was erroneously deleted by User A!"
    print("[PASS] 6. Clearing history operates strictly on user's own data without affecting others")

    print("\n=======================================================")
    print(" ALL USER DASHBOARD DATA ISOLATION TESTS PASSED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
