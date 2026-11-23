"""
CyberShield MongoDB Database Integration
Provides unified data access layer for:
- Users (authentication, roles, status, strikes)
- Posts (Twitter-like feed, AI interception, moderation metadata)
- Notifications (user alerts for intercepted hate comments, account warnings)
- Predictions (system inference history and dashboard analytics)
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("MONGO_DB_NAME", "cybershield_db")


class MongoDBManager:
    """Manages connections, schema initialization, and operations for CyberShield."""

    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            cls._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        return cls._client

    @classmethod
    def get_db(cls) -> Database:
        if cls._db is None:
            client = cls.get_client()
            cls._db = client[DB_NAME]
        return cls._db

    @classmethod
    def users_col(cls) -> Collection:
        return cls.get_db()["users"]

    @classmethod
    def posts_col(cls) -> Collection:
        return cls.get_db()["posts"]

    @classmethod
    def notifications_col(cls) -> Collection:
        return cls.get_db()["notifications"]

    @classmethod
    def predictions_col(cls) -> Collection:
        return cls.get_db()["predictions"]

    @classmethod
    def activity_logs_col(cls) -> Collection:
        return cls.get_db()["activity_logs"]

    @classmethod
    def initialize_database(cls) -> None:
        """Create collections, indexes, and seed default admin/demo accounts."""
        db = cls.get_db()
        logger.info(f"Initializing MongoDB database '{DB_NAME}' at {MONGO_URI}...")

        # 1. Indexes for users
        users = cls.users_col()
        users.create_index([("username", ASCENDING)], unique=True)
        users.create_index([("role", ASCENDING)])
        users.create_index([("status", ASCENDING)])

        # 2. Indexes for posts
        posts = cls.posts_col()
        posts.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        posts.create_index([("user_id", ASCENDING)])

        # 3. Indexes for notifications
        notifications = cls.notifications_col()
        notifications.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        notifications.create_index([("user_id", ASCENDING), ("is_read", ASCENDING)])

        # 4. Indexes for predictions
        preds = cls.predictions_col()
        preds.create_index([("timestamp", DESCENDING)])

        # 5. Indexes for activity logs
        logs = cls.activity_logs_col()
        logs.create_index([("timestamp", DESCENDING)])
        logs.create_index([("action", ASCENDING)])

        # 5. Seed default accounts if they do not exist
        cls._seed_defaults()
        logger.info("MongoDB database initialized successfully.")

    @classmethod
    def _seed_defaults(cls) -> None:
        users = cls.users_col()

        # Seed Admin
        if not users.find_one({"username": "admin"}):
            admin_user = {
                "username": "admin",
                "email": "admin@cybershield.org",
                "password_hash": generate_password_hash("admin123"),
                "display_name": "Shield Administrator",
                "role": "admin",
                "status": "active",
                "strikes_count": 0,
                "bio": "Platform Security & NLP Content Moderation Supervisor",
                "created_at": datetime.utcnow()
            }
            users.insert_one(admin_user)
            logger.info("Seeded default admin account: admin / admin123")

        # Seed Demo User
        demo_user = users.find_one({"username": "user"})
        if not demo_user:
            demo_user = {
                "username": "user",
                "email": "user@example.com",
                "password_hash": generate_password_hash("user123"),
                "display_name": "Demo User",
                "role": "user",
                "status": "active",
                "strikes_count": 0,
                "bio": "Active community member & social explorer",
                "created_at": datetime.utcnow()
            }
            res = users.insert_one(demo_user)
            demo_user["_id"] = res.inserted_id
            logger.info("Seeded default demo user account: user / user123")

        # Seed a welcoming verified post if feed is empty
        posts = cls.posts_col()
        if posts.count_documents({}) == 0:
            welcome_post = {
                "user_id": str(demo_user["_id"]) if "_id" in demo_user else "system",
                "username": "CyberShield_Bot",
                "display_name": "CyberShield AI System",
                "content": "Welcome to CyberShield Social! 🛡️ All posts are monitored in real-time by AI models to keep online spaces respectful and bully-free.",
                "image_path": None,
                "status": "published",
                "created_at": datetime.utcnow(),
                "likes_count": 12,
                "moderation": {
                    "prediction": "Non-Cyberbullying",
                    "confidence": 0.998,
                    "risk_score": "Safe",
                    "model_used": "System Verified",
                    "toxic_tokens": [],
                    "ocr_text": None,
                    "intercepted": False
                }
            }
            posts.insert_one(welcome_post)
            logger.info("Seeded welcome tweet on timeline.")

    # --------------------------------------------------------------------------
    # User Authentication & Management
    # --------------------------------------------------------------------------
    @classmethod
    def create_user(
        cls,
        username: str,
        password: str,
        email: str,
        display_name: Optional[str] = None,
        role: str = "user"
    ) -> Dict[str, Any]:
        users = cls.users_col()
        username_clean = username.strip().lower()

        if users.find_one({"username": username_clean}):
            raise ValueError(f"Username '{username_clean}' is already taken.")

        new_user = {
            "username": username_clean,
            "email": email.strip().lower(),
            "password_hash": generate_password_hash(password),
            "display_name": (display_name or username).strip(),
            "role": role if role in ("user", "admin") else "user",
            "status": "active",
            "strikes_count": 0,
            "bio": "Member of CyberShield community",
            "created_at": datetime.utcnow()
        }
        res = users.insert_one(new_user)
        new_user["_id"] = res.inserted_id
        return new_user

    @classmethod
    def authenticate(cls, username: str, password: str) -> Optional[Dict[str, Any]]:
        users = cls.users_col()
        user = users.find_one({"username": username.strip().lower()})
        if user and check_password_hash(user["password_hash"], password):
            return user
        return None

    @classmethod
    def get_user_by_id(cls, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            return cls.users_col().find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    @classmethod
    def get_user_by_username(cls, username: str) -> Optional[Dict[str, Any]]:
        return cls.users_col().find_one({"username": username.strip().lower()})

    @classmethod
    def block_user(cls, user_id: str, reason: str = "Violation of cyberbullying policies", admin_username: str = "admin") -> bool:
        users = cls.users_col()
        target = users.find_one({"_id": ObjectId(user_id)})
        if not target:
            return False
        if target.get("role") == "admin":
            raise ValueError("Administrator accounts cannot be blocked.")

        res = users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": "blocked", "blocked_at": datetime.utcnow(), "block_reason": reason}}
        )
        if res.modified_count > 0:
            cls.log_activity(
                admin_username=admin_username,
                action="block_user",
                target_type="user",
                target_id=user_id,
                details=f"Blocked @{target.get('username')}: reason '{reason}'"
            )
            cls.create_notification(
                user_id=user_id,
                type_="account_blocked",
                title="⛔ Account Blocked by Administrator",
                message=f"Your account has been blocked due to: {reason}. You can no longer publish posts."
            )
            return True
        return False

    @classmethod
    def unblock_user(cls, user_id: str, admin_username: str = "admin") -> bool:
        users = cls.users_col()
        target = users.find_one({"_id": ObjectId(user_id)})
        res = users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": "active"}, "$unset": {"blocked_at": "", "block_reason": ""}}
        )
        if res.modified_count > 0:
            cls.log_activity(
                admin_username=admin_username,
                action="unblock_user",
                target_type="user",
                target_id=user_id,
                details=f"Restored account access for @{target.get('username') if target else user_id}"
            )
            cls.create_notification(
                user_id=user_id,
                type_="account_restored",
                title="✅ Account Access Restored",
                message="Your account has been unblocked by an administrator. Please keep interactions respectful."
            )
            return True
        return False

    @classmethod
    def increment_user_strikes(cls, user_id: str) -> int:
        res = cls.users_col().find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$inc": {"strikes_count": 1}},
            return_document=True
        )
        return res.get("strikes_count", 1) if res else 1

    @classmethod
    def get_all_users(cls) -> List[Dict[str, Any]]:
        return list(cls.users_col().find().sort("created_at", DESCENDING))

    @classmethod
    def get_users_filtered(
        cls,
        search: str = "",
        role: str = "",
        status: str = "",
        sort_by: str = "created_at"
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if search:
            s_regex = {"$regex": re.escape(search), "$options": "i"}
            query["$or"] = [
                {"username": s_regex},
                {"display_name": s_regex},
                {"email": s_regex}
            ]
        if role and role != "all":
            query["role"] = role
        if status and status != "all":
            query["status"] = status

        sort_field = "created_at"
        sort_dir = DESCENDING
        if sort_by == "strikes":
            sort_field = "strikes_count"
            sort_dir = DESCENDING
        elif sort_by == "username":
            sort_field = "username"
            sort_dir = ASCENDING

        return list(cls.users_col().find(query).sort(sort_field, sort_dir))

    # --------------------------------------------------------------------------
    # Activity Logs & Moderation History
    # --------------------------------------------------------------------------
    @classmethod
    def log_activity(
        cls,
        admin_username: str,
        action: str,
        target_type: str,
        target_id: str,
        details: str
    ) -> Dict[str, Any]:
        """Logs an admin operation for the audit trail."""
        entry = {
            "admin_username": admin_username,
            "action": action, # 'approve_post', 'reject_post', 'block_user', 'unblock_user'
            "target_type": target_type, # 'post', 'user'
            "target_id": str(target_id),
            "details": details,
            "timestamp": datetime.utcnow()
        }
        res = cls.activity_logs_col().insert_one(entry)
        entry["_id"] = res.inserted_id
        return entry

    @classmethod
    def get_activity_logs(cls, limit: int = 100, action_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {}
        if action_filter and action_filter != "all":
            query["action"] = action_filter
        return list(cls.activity_logs_col().find(query).sort("timestamp", DESCENDING).limit(limit))

    @classmethod
    def get_moderation_history(cls, limit: int = 100, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns posts that have gone through human moderation review."""
        query: Dict[str, Any] = {
            "$or": [
                {"moderation.review_action": {"$exists": True}},
                {"status": {"$in": ["quarantined", "rejected"]}}
            ]
        }
        if status_filter and status_filter != "all":
            query["status"] = status_filter
        return list(cls.posts_col().find(query).sort("created_at", DESCENDING).limit(limit))

    # --------------------------------------------------------------------------
    # Posts & Twitter-like Social Feed
    # --------------------------------------------------------------------------
    @classmethod
    def create_post(
        cls,
        user_id: str,
        username: str,
        display_name: str,
        content: str,
        image_path: Optional[str] = None,
        moderation_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates a post. If moderation_info indicates cyberbullying, status is 'quarantined',
        otherwise 'published'.
        """
        is_quarantined = False
        if moderation_info:
            pred = moderation_info.get("prediction", "")
            if pred in ("Bullying", "Cyberbullying") or moderation_info.get("intercepted") is True:
                is_quarantined = True

        post = {
            "user_id": user_id,
            "username": username,
            "display_name": display_name,
            "content": content.strip(),
            "image_path": image_path,
            "status": "quarantined" if is_quarantined else "published",
            "created_at": datetime.utcnow(),
            "likes_count": 0,
            "moderation": moderation_info or {
                "prediction": "Non-Cyberbullying",
                "confidence": 0.5,
                "risk_score": "Low",
                "model_used": "default",
                "toxic_tokens": [],
                "ocr_text": None,
                "intercepted": False
            }
        }
        res = cls.posts_col().insert_one(post)
        post["_id"] = res.inserted_id

        # Dispatch immediate notification to user if flagged
        if is_quarantined:
            cls.increment_user_strikes(user_id)
            conf_pct = round(post["moderation"].get("confidence", 0.0) * 100, 1)
            cls.create_notification(
                user_id=user_id,
                post_id=str(post["_id"]),
                type_="hate_speech_warning",
                title="⚠️ Post Intercepted for Review",
                message=(
                    f"Your tweet was flagged by CyberShield AI as offensive / cyberbullying "
                    f"({conf_pct}% confidence) and held from the public timeline for admin moderation."
                )
            )

        return post

    @classmethod
    def get_feed_posts(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns verified, published posts for the public timeline."""
        return list(
            cls.posts_col()
            .find({"status": "published"})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

    @classmethod
    def get_quarantined_posts(cls) -> List[Dict[str, Any]]:
        """Returns all hate/bullying posts awaiting admin moderation."""
        return list(
            cls.posts_col()
            .find({"status": "quarantined"})
            .sort("created_at", DESCENDING)
        )

    @classmethod
    def get_post_by_id(cls, post_id: str) -> Optional[Dict[str, Any]]:
        try:
            return cls.posts_col().find_one({"_id": ObjectId(post_id)})
        except Exception:
            return None

    @classmethod
    def approve_post(cls, post_id: str, admin_name: str) -> bool:
        """Admin overrides AI and allows post onto public feed."""
        post = cls.get_post_by_id(post_id)
        if not post:
            return False

        res = cls.posts_col().update_one(
            {"_id": ObjectId(post_id)},
            {
                "$set": {
                    "status": "published",
                    "moderation.reviewed_by": admin_name,
                    "moderation.review_action": "approved",
                    "moderation.reviewed_at": datetime.utcnow()
                }
            }
        )
        if res.modified_count > 0:
            cls.log_activity(
                admin_username=admin_name,
                action="approve_post",
                target_type="post",
                target_id=post_id,
                details=f"Approved post by @{post.get('username')}: '{post.get('content')[:50]}...'"
            )
            cls.create_notification(
                user_id=str(post["user_id"]),
                post_id=post_id,
                type_="post_approved",
                title="✅ Post Approved by Moderator",
                message="An administrator reviewed your intercepted post and approved it for the public timeline."
            )
            return True
        return False

    @classmethod
    def reject_post(cls, post_id: str, admin_name: str, reason: str = "Hate Speech / Harassment") -> bool:
        """Admin confirms toxicity and permanently rejects post."""
        post = cls.get_post_by_id(post_id)
        if not post:
            return False

        res = cls.posts_col().update_one(
            {"_id": ObjectId(post_id)},
            {
                "$set": {
                    "status": "rejected",
                    "moderation.reviewed_by": admin_name,
                    "moderation.review_action": "rejected",
                    "moderation.rejection_reason": reason,
                    "moderation.reviewed_at": datetime.utcnow()
                }
            }
        )
        if res.modified_count > 0:
            cls.log_activity(
                admin_username=admin_name,
                action="reject_post",
                target_type="post",
                target_id=post_id,
                details=f"Rejected toxic post by @{post.get('username')}: reason '{reason}'"
            )
            cls.create_notification(
                user_id=str(post["user_id"]),
                post_id=post_id,
                type_="post_rejected",
                title="❌ Post Rejected & Deleted",
                message=f"Your post was permanently removed by a moderator for: {reason}."
            )
            return True
        return False

    # --------------------------------------------------------------------------
    # Notifications Management
    # --------------------------------------------------------------------------
    @classmethod
    def create_notification(
        cls,
        user_id: str,
        type_: str,
        title: str,
        message: str,
        post_id: Optional[str] = None
    ) -> Dict[str, Any]:
        notif = {
            "user_id": str(user_id),
            "post_id": str(post_id) if post_id else None,
            "type": type_,
            "title": title,
            "message": message,
            "is_read": False,
            "created_at": datetime.utcnow()
        }
        res = cls.notifications_col().insert_one(notif)
        notif["_id"] = res.inserted_id
        return notif

    @classmethod
    def get_user_notifications(cls, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return list(
            cls.notifications_col()
            .find({"user_id": str(user_id)})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

    @classmethod
    def get_unread_count(cls, user_id: str) -> int:
        return cls.notifications_col().count_documents({"user_id": str(user_id), "is_read": False})

    @classmethod
    def mark_notifications_as_read(cls, user_id: str) -> int:
        res = cls.notifications_col().update_many(
            {"user_id": str(user_id), "is_read": False},
            {"$set": {"is_read": True}}
        )
        return res.modified_count

    # --------------------------------------------------------------------------
    # Predictions History & Analytics Dashboard
    # --------------------------------------------------------------------------
    @classmethod
    def save_prediction(
        cls,
        text_snippet: str,
        prediction: str,
        confidence: float,
        model_type: str,
        user_id: Optional[str] = None
    ) -> str:
        record = {
            "text_snippet": text_snippet[:300],
            "prediction": prediction,
            "confidence": float(confidence),
            "model_type": model_type,
            "user_id": str(user_id) if user_id else "anonymous",
            "timestamp": datetime.utcnow()
        }
        res = cls.predictions_col().insert_one(record)
        return str(res.inserted_id)

    @classmethod
    def get_prediction_history(cls, limit: int = 200) -> List[Dict[str, Any]]:
        return list(
            cls.predictions_col()
            .find()
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )

    @classmethod
    def clear_prediction_history(cls) -> int:
        res = cls.predictions_col().delete_many({})
        return res.deleted_count

    @classmethod
    def get_platform_analytics(cls) -> Dict[str, Any]:
        """Calculates global platform KPIs and model breakdown."""
        total_posts = cls.posts_col().count_documents({})
        published_posts = cls.posts_col().count_documents({"status": "published"})
        quarantined_posts = cls.posts_col().count_documents({"status": "quarantined"})
        rejected_posts = cls.posts_col().count_documents({"status": "rejected"})

        total_users = cls.users_col().count_documents({})
        blocked_users = cls.users_col().count_documents({"status": "blocked"})
        active_users = total_users - blocked_users

        total_predictions = cls.predictions_col().count_documents({})
        cyberbullying_predictions = cls.predictions_col().count_documents({"prediction": "Cyberbullying"})
        non_cyberbullying_predictions = total_predictions - cyberbullying_predictions

        # Calculate average confidence
        pipeline = [
            {"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}}
        ]
        conf_res = list(cls.predictions_col().aggregate(pipeline))
        avg_confidence = round(conf_res[0]["avg_conf"] * 100, 1) if conf_res else 0.0

        return {
            "total_posts": total_posts,
            "published_posts": published_posts,
            "quarantined_posts": quarantined_posts,
            "rejected_posts": rejected_posts,
            "total_users": total_users,
            "active_users": active_users,
            "blocked_users": blocked_users,
            "total_predictions": total_predictions,
            "cyberbullying_predictions": cyberbullying_predictions,
            "non_cyberbullying_predictions": non_cyberbullying_predictions,
            "avg_confidence": avg_confidence
        }
