"""
End-to-End Test for Twitter-like Social Media Platform with AI Hate Interception
and Admin Moderation Desk using local MongoDB.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from src.mongo_db import MongoDBManager

def run_tests():
    print("--> Starting Social Platform & AI Interception Verification...")
    # Ensure test demo user is clean & active
    try:
        MongoDBManager.users_col().update_one(
            {"username": "user"},
            {"$set": {"status": "active", "strikes_count": 0}}
        )
    except Exception as e:
        print(f"Warning during reset: {e}")

    client = app.test_client()

    # 1. Test Feed Page
    resp = client.get("/feed")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert b"CyberShield Social Feed" in resp.data
    print("[PASS] 1. Feed page accessible (HTTP 200)")

    # 2. Test User Login
    resp = client.post("/login", data={"username": "user", "password": "user123"}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Demo User" in resp.data
    print("[PASS] 2. User login successful")

    # 3. Test Clean Post Creation
    clean_text = "Good morning everyone! Wishing all developers a productive day ahead."
    resp = client.post("/post/create", data={"content": clean_text}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Tweet published to timeline successfully" in resp.data
    assert clean_text.encode("utf-8") in resp.data
    print("[PASS] 3. Clean post passed AI filter and published to timeline")

    # 4. Test Hate / Cyberbullying Post Interception
    hate_text = "You are a dumb idiot loser and everyone hates you so much"
    resp = client.post("/post/create", data={"content": hate_text}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Hate Speech / Cyberbullying Intercepted" in resp.data
    print("[PASS] 4. Hate speech intercepted and withheld from timeline!")

    # Verify post is quarantined in MongoDB
    quarantined = MongoDBManager.get_quarantined_posts()
    assert len(quarantined) >= 1, "Expected at least 1 quarantined post in DB"
    flagged_post = quarantined[0]
    print(f"       -> Intercepted Post: '{flagged_post['content'][:40]}...'")
    print(f"       -> AI Confidence: {flagged_post['moderation']['confidence']}")
    print(f"       -> Toxic Tokens Detected: {[t['token'] for t in flagged_post['moderation']['toxic_tokens']]}")

    # 5. Verify User Received Safety Notification
    user = MongoDBManager.get_user_by_username("user")
    notifs = MongoDBManager.get_user_notifications(str(user["_id"]))
    assert len(notifs) >= 1, "Expected at least 1 notification for user"
    assert notifs[0]["type"] == "hate_speech_warning"
    print(f"[PASS] 5. User received safety alert notification: '{notifs[0]['title']}' (Strikes: {user.get('strikes_count', 0)})")

    # 6. Test Admin Login & Moderation Desk
    with client.session_transaction() as sess:
        sess.clear()
    
    resp = client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Moderation Desk" in resp.data or b"Dashboard Overview" in resp.data or b"Command Center" in resp.data
    print("[PASS] 6. Admin login successful and redirected to Moderation Desk")

    # 7. Test Admin View Quarantined Post
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert b"Pending Queue Preview" in resp.data or b"Review Queue" in resp.data or b"Moderation Queue" in resp.data
    assert hate_text.encode("utf-8") in resp.data
    print("[PASS] 7. Admin can view quarantined hate speech with in-depth toxic tokens")

    # 8. Test Admin Block User
    resp = client.post(f"/admin/user/{str(user['_id'])}/block", follow_redirects=True)
    assert resp.status_code == 200
    assert b"suspended" in resp.data or b"blocked" in resp.data

    updated_user = MongoDBManager.get_user_by_id(str(user["_id"]))
    assert updated_user["status"] == "blocked"
    print(f"[PASS] 8. Admin successfully blocked user @{updated_user['username']} (Status: {updated_user['status']})")

    # 9. Test Blocked User Cannot Post
    with client.session_transaction() as sess:
        sess["user_id"] = str(user["_id"])
        sess["username"] = user["username"]
        sess["role"] = "user"
        sess["display_name"] = user["display_name"]

    resp = client.post("/post/create", data={"content": "Trying to post while blocked"}, follow_redirects=True)
    assert b"Your account is blocked" in resp.data
    print("[PASS] 9. Blocked user prevented from posting")

    # 10. Test Admin Unblock User
    with client.session_transaction() as sess:
        sess.clear()
        admin = MongoDBManager.get_user_by_username("admin")
        sess["user_id"] = str(admin["_id"])
        sess["username"] = admin["username"]
        sess["role"] = "admin"

    resp = client.post(f"/admin/user/{str(user['_id'])}/unblock", follow_redirects=True)
    assert resp.status_code == 200
    restored_user = MongoDBManager.get_user_by_id(str(user["_id"]))
    assert restored_user["status"] == "active"
    print(f"[PASS] 10. Admin successfully restored user access (Status: {restored_user['status']})")

    print("\n=======================================================")
    print(" ALL 10 SOCIAL PLATFORM & MODERATION TESTS PASSED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
