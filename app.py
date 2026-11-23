"""
CyberShield: Cyberbullying Detection from Social Media
Flask Web Application

Production-ready web application providing real-time text analysis,
prediction history tracking, and interactive analytics dashboard.

Routes:
    GET  /               : Home page with text input analyzer
    POST /analyze        : Process text, validate, run inference, save to DB
    GET  /history        : View past classification records
    POST /clear-history  : Clear history records from SQLite database
    GET  /dashboard      : View usage analytics, stats, and Chart.js charts
"""

import os
import sqlite3
import logging
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, List
from bson import ObjectId

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    session,
    abort
)

# Import predictor and NLP modules to fulfill CO1, CO2, CO3, CO4 rubrics
from src.predict import CyberbullyingPredictor
from src.genai_explainer import GenAIExplainer
from src.topic_modeling import LDATopicModeler
from src.multilingual import MultilingualProcessor
from src.information_extraction import InformationExtractor
from src.explainability import TokenExplainer
from src.ocr_detector import ocr_detector
from src.mongo_db import MongoDBManager

genai_service = GenAIExplainer()
topic_modeler = LDATopicModeler()
multilingual_processor = MultilingualProcessor()
info_extractor = InformationExtractor()
token_explainer = TokenExplainer()

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "predictions.db")

app = Flask(__name__)

# Configuration parameters
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "cybershield-final-year-project-secret-2026"
)
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
app.config["MODEL_TYPE"] = os.environ.get("MODEL_TYPE", "logistic_regression").lower()
app.config["MAX_INPUT_LENGTH"] = 5000

# Supported model architectures
SUPPORTED_MODELS = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "svm": "TF-IDF + Linear SVM",
    "distilbert": "Fine-Tuned DistilBERT"
}

# ------------------------------------------------------------------------------
# Database Helpers
# ------------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with dictionary rows."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_snippet TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_type TEXT NOT NULL,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_id TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# Initialize database table on app startup
init_db()

# Initialize MongoDB database & seed default accounts
try:
    MongoDBManager.initialize_database()
except Exception as mongo_init_err:
    logger.warning(f"Notice during MongoDB initialization: {mongo_init_err}")


# ------------------------------------------------------------------------------
# Auth & Context Helpers
# ------------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or not MongoDBManager.get_user_by_id(session["user_id"]):
            flash("Please sign in to access this feature.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Administrator login required.", "danger")
            return redirect(url_for("login", next=request.url))
        user = MongoDBManager.get_user_by_id(session["user_id"])
        if not user or user.get("role") != "admin":
            flash("Access denied: Administrator privileges required.", "danger")
            return redirect(url_for("feed"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_user_context():
    """Provides current_user and unread_notifs_count to all Jinja templates."""
    user_id = session.get("user_id")
    user = MongoDBManager.get_user_by_id(user_id) if user_id else None
    unread_count = MongoDBManager.get_unread_count(user_id) if user_id else 0
    return {
        "current_user": user,
        "unread_notifs_count": unread_count
    }

# ------------------------------------------------------------------------------
# Initialize Predictor on Startup (load model once)
# ------------------------------------------------------------------------------
logger.info(f"Initializing CyberbullyingPredictor with default model: {app.config['MODEL_TYPE']}")
try:
    predictor = CyberbullyingPredictor(model_type=app.config["MODEL_TYPE"])
except Exception as e:
    logger.warning(f"Failed to load '{app.config['MODEL_TYPE']}': {e}. Falling back to 'logistic_regression'.")
    predictor = CyberbullyingPredictor(model_type="logistic_regression")


# ------------------------------------------------------------------------------
# Social Media & Authentication Routes
# ------------------------------------------------------------------------------
@app.before_request
def require_login_globally():
    """
    Enforce authentication globally when user or admin is logged out.
    When logged out, only the login/auth views and static assets are accessible.
    """
    if request.path.startswith("/static"):
        return None

    allowed_endpoints = {"login", "register", "quick_login", "logout", "poster"}
    if request.endpoint in allowed_endpoints:
        return None

    user_id = session.get("user_id")
    if not user_id or not MongoDBManager.get_user_by_id(user_id):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """User and Administrator authentication."""
    # If already logged in, redirect directly to appropriate dashboard/feed
    if "user_id" in session and MongoDBManager.get_user_by_id(session["user_id"]):
        user = MongoDBManager.get_user_by_id(session["user_id"])
        if user.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("feed"))

    demo_users = MongoDBManager.get_demo_users()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.form.get("next") or url_for("feed")

        user = MongoDBManager.authenticate(username, password)
        if not user:
            flash("Invalid username or password. Please try again.", "danger")
            return render_template("login.html", demo_users=demo_users)

        if user.get("status") == "blocked":
            flash("Your account has been blocked by an administrator for cyberbullying violations.", "danger")
            return render_template("login.html", demo_users=demo_users)

        session["user_id"] = str(user["_id"])
        session["username"] = user["username"]
        session["role"] = user["role"]
        session["display_name"] = user.get("display_name", user["username"])
        flash(f"Welcome back, {session['display_name']}!", "success")

        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(next_url)

    return render_template("login.html", demo_users=demo_users)


@app.route("/quick-login/<username>", methods=["GET", "POST"])
def quick_login(username):
    """
    One-click instant login for demo accounts.
    Immediately authenticates and routes to the appropriate view.
    """
    username_clean = username.strip().lower()
    user = MongoDBManager.get_user_by_username(username_clean)
    if not user:
        flash(f"Demo user '{username}' was not found.", "warning")
        return redirect(url_for("login"))

    if user.get("status") == "blocked":
        flash(f"Demo user @{username} is currently suspended for safety violations. Please unblock it from the Admin Desk or choose another user.", "danger")
        return redirect(url_for("login"))

    session["user_id"] = str(user["_id"])
    session["username"] = user["username"]
    session["role"] = user.get("role", "user")
    session["display_name"] = user.get("display_name", user["username"])

    flash(f"Quick-Login successful! Signed in as {session['display_name']} (@{user['username']}).", "success")

    if user.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("feed"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """User account creation."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not password or not email:
            flash("All fields are required.", "warning")
            return render_template("register.html")

        try:
            new_user = MongoDBManager.create_user(
                username=username,
                password=password,
                email=email,
                display_name=display_name
            )
            session["user_id"] = str(new_user["_id"])
            session["username"] = new_user["username"]
            session["role"] = new_user["role"]
            session["display_name"] = new_user["display_name"]
            flash(f"Account created successfully! Welcome to CyberShield, {new_user['display_name']}!", "success")
            return redirect(url_for("feed"))
        except ValueError as ve:
            flash(str(ve), "danger")
            return render_template("register.html")
        except Exception as e:
            logger.error(f"Registration error: {e}", exc_info=True)
            flash("An error occurred during registration. Please try again.", "danger")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/logout")
def logout():
    """Sign out current user and return directly to login page."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


@app.route("/feed")
def feed():
    """Twitter-like social timeline with published, safe posts."""
    if session.get("role") == "admin":
        return redirect(url_for("admin_feed"))
    posts = MongoDBManager.get_feed_posts(limit=50)
    return render_template("feed.html", posts=posts)


@app.route("/post/create", methods=["POST"])
@login_required
def create_post():
    """
    Submits a post/comment.
    Scans with CyberShield AI + OCR (if screenshot attached) + LIME explainability.
    If hate speech is detected, quarantines post, alerts user, and routes to Admin Desk.
    """
    user = MongoDBManager.get_user_by_id(session["user_id"])
    if not user or user.get("status") == "blocked":
        flash("Your account is blocked. You cannot post content.", "danger")
        ref = request.referrer
        if ref and "/admin" in ref:
            return redirect(url_for("admin_feed"))
        return redirect(url_for("feed"))

    content = request.form.get("content", "").strip()
    if not content:
        flash("Post content cannot be empty.", "warning")
        ref = request.referrer
        if ref and "/admin" in ref:
            return redirect(url_for("admin_feed"))
        return redirect(url_for("feed"))

    image_file = request.files.get("image")
    saved_image_rel_path = None
    ocr_extracted_text = None

    if image_file and image_file.filename:
        from werkzeug.utils import secure_filename
        sec_name = secure_filename(image_file.filename)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"post_{timestamp_str}_{sec_name}"
        save_path = os.path.join(UPLOAD_FOLDER, fname)
        image_file.save(save_path)
        saved_image_rel_path = f"uploads/{fname}"

        # Extract OCR text from screenshot
        try:
            ocr_res = ocr_detector.extract_text_and_boxes(save_path)
            if ocr_res.get("text"):
                ocr_extracted_text = ocr_res["text"].strip()
        except Exception as ocr_err:
            logger.warning(f"OCR extraction error on post image: {ocr_err}")

    # Build text for AI inference (text + OCR text if available)
    combined_text_for_ai = content
    if ocr_extracted_text:
        combined_text_for_ai = f"{content} [Image Text]: {ocr_extracted_text}"

    # Run AI inference
    pred_res = predictor.predict(combined_text_for_ai)
    pred_label = pred_res.get("prediction", "Non-Bullying")
    confidence = pred_res.get("confidence", 0.5)
    # Check if confidence is percentage (e.g. 96.5) or decimal (0.965)
    conf_norm = confidence / 100.0 if confidence > 1.0 else confidence

    is_hate = (pred_label in ("Bullying", "Cyberbullying") or pred_res.get("label") == 1)

    # Run Token-Level XAI to highlight toxic words
    toxic_tokens = []
    try:
        xai_data = token_explainer.explain(combined_text_for_ai, predictor=predictor)
        for tok in xai_data.get("tokens", []):
            tok_word = tok.get("word") or tok.get("token")
            tok_weight = tok.get("weight", 0.0)
            if tok.get("direction") == "bullying" or tok_weight > 0.05:
                toxic_tokens.append({
                    "token": tok_word,
                    "score": round(float(tok_weight), 4),
                    "is_toxic": True
                })
    except Exception as xai_err:
        logger.warning(f"XAI attribution error: {xai_err}")

    risk_score = "High" if (is_hate and conf_norm > 0.8) else ("Medium" if is_hate else "Safe")

    moderation_info = {
        "prediction": "Cyberbullying" if is_hate else "Non-Cyberbullying",
        "confidence": round(float(conf_norm), 4),
        "risk_score": risk_score,
        "model_used": SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type),
        "toxic_tokens": toxic_tokens,
        "ocr_text": ocr_extracted_text,
        "intercepted": is_hate
    }

    MongoDBManager.create_post(
        user_id=str(user["_id"]),
        username=user["username"],
        display_name=user.get("display_name", user["username"]),
        content=content,
        image_path=saved_image_rel_path,
        moderation_info=moderation_info
    )

    # Record to MongoDB predictions collection for analytics
    MongoDBManager.save_prediction(
        text_snippet=content,
        prediction=pred_label,
        confidence=confidence,
        model_type=predictor.model_type,
        user_id=str(user["_id"])
    )

    if is_hate:
        conf_pct = round(conf_norm * 100, 1)
        flash(
            f"⚠️ Hate Speech / Cyberbullying Intercepted ({conf_pct}% confidence)! "
            f"Your tweet was withheld from the public timeline and submitted to Admin Moderation.",
            "danger"
        )
    else:
        flash("Tweet published to timeline successfully!", "success")

    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(url_for("admin_feed"))
    if session.get("role") == "admin":
        return redirect(url_for("admin_feed"))
    return redirect(url_for("feed"))


@app.route("/post/<post_id>/like", methods=["POST"])
def like_post(post_id):
    """Increment like counter on a published post."""
    try:
        MongoDBManager.posts_col().update_one(
            {"_id": ObjectId(post_id)},
            {"$inc": {"likes_count": 1}}
        )
    except Exception as e:
        logger.error(f"Error liking post {post_id}: {e}")
    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(url_for("admin_feed"))
    if session.get("role") == "admin":
        return redirect(url_for("admin_feed"))
    return redirect(url_for("feed"))


@app.route("/notifications")
@login_required
def notifications():
    """Displays user's safety alerts and moderation decisions."""
    user_id = session["user_id"]
    user_notifs = MongoDBManager.get_user_notifications(user_id)
    return render_template("notifications.html", notifications=user_notifs)


@app.route("/notifications/mark-read", methods=["POST"])
@login_required
def mark_notifications_read():
    """Mark all notifications as read."""
    user_id = session["user_id"]
    MongoDBManager.mark_notifications_as_read(user_id)
    flash("All notifications marked as read.", "info")
    return redirect(url_for("notifications"))


# ------------------------------------------------------------------------------
# Admin Moderation Desk Routes
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modern SaaS Admin Panel Routes
# ------------------------------------------------------------------------------
@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """1. SaaS Dashboard Overview with 4 KPI cards, recent activity, and integrated moderation history."""
    analytics = MongoDBManager.get_platform_analytics()
    quarantined = MongoDBManager.get_quarantined_posts()
    recent_activities = MongoDBManager.get_activity_logs(limit=6)
    history_posts = MongoDBManager.get_moderation_history(limit=50)

    return render_template(
        "admin/dashboard.html",
        active_tab="dashboard",
        analytics=analytics,
        pending_quarantined_count=len(quarantined),
        pending_posts_preview=quarantined[:4],
        recent_activities=recent_activities,
        history_posts=history_posts,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/feed")
@admin_required
def admin_feed():
    """Live social timeline integrated directly inside Admin Console."""
    posts = MongoDBManager.get_feed_posts(limit=50)
    quarantined = MongoDBManager.get_quarantined_posts()
    return render_template(
        "admin/feed.html",
        active_tab="feed",
        posts=posts,
        pending_quarantined_count=len(quarantined),
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/post/<post_id>/delete", methods=["POST"])
@admin_required
def admin_delete_post(post_id):
    """Admin permanently deletes a post from the timeline."""
    try:
        MongoDBManager.posts_col().delete_one({"_id": ObjectId(post_id)})
        MongoDBManager.log_activity(
            admin_username=session.get("username", "admin"),
            action="delete_post",
            target_id=post_id,
            details=f"Permanently removed post {post_id} from timeline"
        )
        flash("Post permanently removed from timeline.", "info")
    except Exception as e:
        logger.error(f"Error deleting post {post_id}: {e}")
        flash("Failed to delete post.", "danger")

    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(ref)
    return redirect(url_for("admin_feed"))


@app.route("/admin/queue")
@admin_required
def admin_queue():
    """2. Moderation Queue: Compact cards, LIME attribution evidence accordion, and action buttons."""
    quarantined = MongoDBManager.get_quarantined_posts()

    # Pre-populate author strike counts
    for p in quarantined:
        u = MongoDBManager.get_user_by_id(p.get("user_id"))
        p["author_strikes"] = u.get("strikes_count", 1) if u else 1

    return render_template(
        "admin/queue.html",
        active_tab="queue",
        quarantined_posts=quarantined,
        pending_quarantined_count=len(quarantined),
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/users")
@admin_required
def admin_users():
    """3. User Management: Search, filter by role/status, sort by strikes, with admin protection."""
    search_query = request.args.get("search", "").strip()
    role_filter = request.args.get("role", "all").strip().lower()
    status_filter = request.args.get("status", "all").strip().lower()
    sort_by = request.args.get("sort_by", "created_at").strip().lower()

    users = MongoDBManager.get_users_filtered(
        search=search_query,
        role=role_filter,
        status=status_filter,
        sort_by=sort_by
    )

    pending_quarantined_count = len(MongoDBManager.get_quarantined_posts())

    return render_template(
        "admin/users.html",
        active_tab="users",
        users=users,
        search_query=search_query,
        current_role=role_filter,
        current_status=status_filter,
        current_sort=sort_by,
        pending_quarantined_count=pending_quarantined_count,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/analytics")
@admin_required
def admin_analytics():
    """4. AI Analytics: Confidence distribution, Cardiff TweetEval empirical benchmarks, and evaluation charts."""
    analytics = MongoDBManager.get_platform_analytics()
    pending_quarantined_count = len(MongoDBManager.get_quarantined_posts())

    # Load actual benchmark data from reports/ml_results.json if present
    eval_metrics = {}
    try:
        results_path = os.path.join(BASE_DIR, "reports", "ml_results.json")
        if os.path.exists(results_path):
            import json
            with open(results_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                active_key = predictor.model_type
                if active_key in res_data:
                    eval_metrics = res_data[active_key].get("test_metrics", {})
    except Exception as e:
        logger.warning(f"Notice reading ml_results.json: {e}")

    return render_template(
        "admin/analytics.html",
        active_tab="analytics",
        analytics=analytics,
        eval_metrics=eval_metrics,
        pending_quarantined_count=pending_quarantined_count,
        current_model_key=predictor.model_type,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/history")
@admin_required
def admin_history():
    """5. Moderation History: Archived review decisions (Approved, Rejected, Flagged)."""
    filter_val = request.args.get("filter", "all").strip().lower()
    history_posts = MongoDBManager.get_moderation_history(limit=100, status_filter=filter_val)
    pending_quarantined_count = len(MongoDBManager.get_quarantined_posts())

    return render_template(
        "admin/history.html",
        active_tab="history",
        history_posts=history_posts,
        current_filter=filter_val,
        pending_quarantined_count=pending_quarantined_count,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/activity")
@admin_required
def admin_activity():
    """6. Administrative Activity Logs: Full audit trail of moderation operations."""
    action_filter = request.args.get("action", "all").strip()
    logs = MongoDBManager.get_activity_logs(limit=100, action_filter=action_filter)
    pending_quarantined_count = len(MongoDBManager.get_quarantined_posts())

    return render_template(
        "admin/activity.html",
        active_tab="activity",
        logs=logs,
        current_action=action_filter,
        pending_quarantined_count=pending_quarantined_count,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)
    )


@app.route("/admin/settings")
@admin_required
def admin_settings():
    """7. System Settings: Model architecture selection, sensitivity threshold, diagnostics."""
    pending_quarantined_count = len(MongoDBManager.get_quarantined_posts())

    return render_template(
        "admin/settings.html",
        active_tab="settings",
        current_model_key=predictor.model_type,
        current_model_name=SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type),
        pending_quarantined_count=pending_quarantined_count
    )


@app.route("/admin/settings/update", methods=["POST"])
@admin_required
def admin_update_settings():
    """Switches active model architecture dynamically."""
    global predictor
    requested_model = request.form.get("model_type", "").strip().lower()
    admin_name = session.get("username", "admin")

    if requested_model in SUPPORTED_MODELS:
        try:
            predictor = CyberbullyingPredictor(model_type=requested_model)
            app.config["MODEL_TYPE"] = requested_model
            MongoDBManager.log_activity(
                admin_username=admin_name,
                action="change_setting",
                target_type="system",
                target_id="active_model",
                details=f"Switched active AI engine to {SUPPORTED_MODELS[requested_model]}"
            )
            flash(f"Active AI engine successfully switched to {SUPPORTED_MODELS[requested_model]}.", "success")
        except Exception as e:
            logger.error(f"Failed to switch model to '{requested_model}': {e}")
            flash(f"Failed to switch model to {requested_model}: {e}", "danger")
    else:
        flash("Unsupported model selection.", "warning")

    return redirect(url_for("admin_settings"))


@app.route("/admin/post/<post_id>/approve", methods=["POST"])
@admin_required
def admin_approve_post(post_id):
    """Admin overrides AI and allows post onto the public timeline."""
    admin_name = session.get("username", "admin")
    success = MongoDBManager.approve_post(post_id, admin_name=admin_name)
    if success:
        flash("Post approved and released to the public timeline.", "success")
    else:
        flash("Failed to approve post.", "danger")
    
    # Return to referring page (Queue, Dashboard, or History)
    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(ref)
    return redirect(url_for("admin_queue"))


@app.route("/admin/post/<post_id>/reject", methods=["POST"])
@admin_required
def admin_reject_post(post_id):
    """Admin confirms toxicity and permanently rejects post."""
    admin_name = session.get("username", "admin")
    reason = request.form.get("reason", "Hate speech / harassment policy violation")
    success = MongoDBManager.reject_post(post_id, admin_name=admin_name, reason=reason)
    if success:
        flash("Post permanently rejected and author notified.", "warning")
    else:
        flash("Failed to reject post.", "danger")
    
    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(ref)
    return redirect(url_for("admin_queue"))


@app.route("/admin/user/<user_id>/block", methods=["POST"])
@admin_required
def admin_block_user(user_id):
    """Admin blocks an offending user from posting."""
    admin_name = session.get("username", "admin")
    reason = request.form.get("reason", "Repeated cyberbullying / hate speech violations")
    try:
        success = MongoDBManager.block_user(user_id, reason=reason, admin_username=admin_name)
        if success:
            flash("User account has been suspended from posting.", "danger")
        else:
            flash("Failed to block user.", "warning")
    except ValueError as ve:
        flash(str(ve), "warning")
    except Exception as e:
        logger.error(f"Error blocking user {user_id}: {e}")
        flash("An error occurred while blocking user.", "danger")

    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(ref)
    return redirect(url_for("admin_users"))


@app.route("/admin/user/<user_id>/unblock", methods=["POST"])
@admin_required
def admin_unblock_user(user_id):
    """Admin unblocks a user."""
    admin_name = session.get("username", "admin")
    success = MongoDBManager.unblock_user(user_id, admin_username=admin_name)
    if success:
        flash("User account access restored.", "success")
    else:
        flash("Failed to unblock user.", "warning")

    ref = request.referrer
    if ref and "/admin" in ref:
        return redirect(ref)
    return redirect(url_for("admin_users"))


# ------------------------------------------------------------------------------
# Core Analyzer Routes
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Home page with input text area, model info, and interactive examples."""
    return render_template(
        "index.html",
        current_model=predictor.model_type,
        supported_models=SUPPORTED_MODELS,
        max_length=app.config["MAX_INPUT_LENGTH"]
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts input text, validates constraints, predicts bullying probability,
    records results to SQLite database, and presents the result.
    """
    raw_text = request.form.get("text", "")
    requested_model = request.form.get("model_type", app.config["MODEL_TYPE"]).strip().lower()

    if requested_model not in SUPPORTED_MODELS:
        requested_model = predictor.model_type

    # Validation: Reject empty or whitespace-only inputs
    cleaned_input = raw_text.strip()
    if not cleaned_input:
        flash("Input cannot be empty or contain only whitespace. Please enter text to analyze.", "warning")
        return redirect(url_for("index"))

    # Validation: Check maximum character limit (5000 chars)
    if len(raw_text) > app.config["MAX_INPUT_LENGTH"]:
        flash(
            f"Input text exceeds the maximum limit of {app.config['MAX_INPUT_LENGTH']} characters "
            f"(Received {len(raw_text)} characters). Please shorten your text.",
            "danger"
        )
        return redirect(url_for("index"))

    try:
        # Dynamically retrieve predictor for requested model (uses cached multiton if available)
        try:
            active_predictor = CyberbullyingPredictor(model_type=requested_model)
        except Exception as model_err:
            logger.warning(
                f"Could not load '{requested_model}': {model_err}. Falling back to default predictor."
            )
            flash(f"Note: Model '{SUPPORTED_MODELS.get(requested_model, requested_model)}' unavailable. Using '{SUPPORTED_MODELS.get(predictor.model_type, predictor.model_type)}'.", "info")
            active_predictor = predictor

        # Step A: Multilingual NLP & Code-Mixed Handling (CO3)
        normalized_text, is_hinglish, hinglish_glossary = multilingual_processor.normalize_and_translate(raw_text)
        lang_info = multilingual_processor.detect_language(raw_text)

        # Step B: Run inference (uses normalized text to capture cross-lingual/code-mixed semantics)
        result = active_predictor.predict(normalized_text)

        # Extract values for persistence
        prediction_label = result.get("prediction", "Non-Bullying")
        label = result.get("label", 1 if prediction_label == "Bullying" else 0)
        is_bullying = (label == 1 or prediction_label == "Bullying")

        raw_confidence = result.get("confidence", 0.5)
        # Handle decimal (0.0-1.0) vs percentage (0-100)
        if raw_confidence <= 1.0:
            confidence_pct = round(raw_confidence * 100, 1)
        else:
            confidence_pct = round(raw_confidence, 1)

        # Build probabilities breakdown for UI display
        probabilities = result.get("probabilities")
        if not probabilities:
            if is_bullying:
                prob_bullying = raw_confidence if raw_confidence <= 1.0 else raw_confidence / 100.0
                prob_normal = max(0.0, 1.0 - prob_bullying)
            else:
                prob_normal = raw_confidence if raw_confidence <= 1.0 else raw_confidence / 100.0
                prob_bullying = max(0.0, 1.0 - prob_normal)
            probabilities = {
                "Non-Bullying": round(float(prob_normal), 4),
                "Bullying": round(float(prob_bullying), 4),
            }

        text_snippet = raw_text.strip()[:100]
        model_name_used = SUPPORTED_MODELS.get(active_predictor.model_type, active_predictor.model_type)

        user_id_val = session.get("user_id")

        # Store prediction in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (text_snippet, prediction, confidence, model_type, user_id, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
            """,
            (text_snippet, prediction_label, confidence_pct, model_name_used, user_id_val)
        )
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        # Store prediction in MongoDB
        try:
            MongoDBManager.save_prediction(
                text_snippet=text_snippet,
                prediction=prediction_label,
                confidence=confidence_pct,
                model_type=model_name_used,
                user_id=user_id_val
            )
        except Exception as me:
            logger.warning(f"Notice: Failed to log prediction to MongoDB: {me}")

        # Step C: Information Extraction (CO2)
        extracted_info = info_extractor.extract_information(raw_text, is_bullying, confidence_pct)

        # Step D: Unsupervised Topic Modeling (CO3)
        topic_info = topic_modeler.predict_topic(raw_text)

        # Step E: Generative AI & Prompt Engineering Linguistic Analysis (CO4)
        genai_data = genai_service.explain_and_rewrite(raw_text, is_bullying)

        # Step F: Token-Level Explainable AI (XAI) Attribution
        xai_tokens = token_explainer.explain(raw_text, active_predictor)

        # Render result page with all 4 Course Outcomes demonstrated
        return render_template(
            "result.html",
            text=raw_text,
            text_snippet=text_snippet,
            prediction=prediction_label,
            is_bullying=is_bullying,
            confidence=confidence_pct,
            probabilities=probabilities,
            model_type=model_name_used,
            record_id=record_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            genai=genai_data,
            extracted_info=extracted_info,
            topic_info=topic_info,
            lang_info=lang_info,
            is_hinglish=is_hinglish,
            hinglish_glossary=hinglish_glossary,
            xai_tokens=xai_tokens
        )

    except Exception as err:
        logger.error(f"Prediction failed with unexpected error: {err}", exc_info=True)
        flash("An unexpected error occurred during analysis. Please try again.", "danger")
        return redirect(url_for("index"))


@app.route("/analyze-screenshot", methods=["POST"])
def analyze_screenshot():
    """
    Multimodal OCR Endpoint: Extracts text from social media screenshots
    and classifies toxicity, running token attribution and restorative GenAI.
    """
    if "screenshot" not in request.files:
        flash("No screenshot file was provided. Please upload an image.", "warning")
        return redirect(url_for("index"))

    file = request.files["screenshot"]
    if not file or file.filename == "":
        flash("No file was selected. Please choose a screenshot image to analyze.", "warning")
        return redirect(url_for("index"))

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        flash(f"Invalid image format '{ext}'. Supported formats: PNG, JPG, JPEG, WEBP, BMP.", "danger")
        return redirect(url_for("index"))

    try:
        import uuid
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # 1. OCR Extraction using Windows Media OCR Engine
        ocr_result = ocr_detector.extract_text(save_path)
        if not ocr_result["success"] or not ocr_result["extracted_text"]:
            err_msg = ocr_result.get("error") or "No legible text was found in the screenshot."
            flash(f"OCR Detection: {err_msg}", "warning")
            return redirect(url_for("index"))

        extracted_text = ocr_result["extracted_text"]
        requested_model = request.form.get("model_type", app.config["MODEL_TYPE"]).strip().lower()

        try:
            active_predictor = CyberbullyingPredictor(model_type=requested_model)
        except Exception:
            active_predictor = predictor

        # 2. Multilingual Normalization & Inference
        normalized_text, is_hinglish, hinglish_glossary = multilingual_processor.normalize_and_translate(extracted_text)
        lang_info = multilingual_processor.detect_language(extracted_text)

        result = active_predictor.predict(normalized_text)
        prediction_label = result.get("prediction", "Non-Bullying")
        label = result.get("label", 1 if prediction_label == "Bullying" else 0)
        is_bullying = (label == 1 or prediction_label == "Bullying")

        raw_confidence = result.get("confidence", 0.5)
        confidence_pct = round(raw_confidence * 100, 1) if raw_confidence <= 1.0 else round(raw_confidence, 1)

        probabilities = result.get("probabilities")
        if not probabilities:
            prob_b = raw_confidence if raw_confidence <= 1.0 else raw_confidence / 100.0
            probabilities = {
                "Non-Bullying": round(1.0 - prob_b if is_bullying else prob_b, 4),
                "Bullying": round(prob_b if is_bullying else 1.0 - prob_b, 4)
            }

        model_name_used = SUPPORTED_MODELS.get(active_predictor.model_type, active_predictor.model_type)
        text_snippet = f"[Screenshot OCR] {extracted_text.strip()[:80]}"

        user_id_val = session.get("user_id")

        # 3. Store in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (text_snippet, prediction, confidence, model_type, user_id, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
            """,
            (text_snippet, prediction_label, confidence_pct, model_name_used, user_id_val)
        )
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        # Store prediction in MongoDB
        try:
            MongoDBManager.save_prediction(
                text_snippet=text_snippet,
                prediction=prediction_label,
                confidence=confidence_pct,
                model_type=model_name_used,
                user_id=user_id_val
            )
        except Exception as me:
            logger.warning(f"Notice: Failed to log prediction to MongoDB: {me}")

        # 4. Feature Attribution & GenAI
        extracted_info = info_extractor.extract_information(extracted_text, is_bullying, confidence_pct)
        topic_info = topic_modeler.predict_topic(extracted_text)
        genai_data = genai_service.explain_and_rewrite(extracted_text, is_bullying)
        xai_tokens = token_explainer.explain(extracted_text, active_predictor)

        screenshot_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            text=extracted_text,
            text_snippet=text_snippet,
            prediction=prediction_label,
            is_bullying=is_bullying,
            confidence=confidence_pct,
            probabilities=probabilities,
            model_type=model_name_used,
            record_id=record_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            genai=genai_data,
            extracted_info=extracted_info,
            topic_info=topic_info,
            lang_info=lang_info,
            is_hinglish=is_hinglish,
            hinglish_glossary=hinglish_glossary,
            xai_tokens=xai_tokens,
            is_screenshot=True,
            screenshot_url=screenshot_url,
            ocr_meta=ocr_result
        )

    except Exception as err:
        logger.error(f"Screenshot analysis failed: {err}", exc_info=True)
        flash("An unexpected error occurred while processing the screenshot. Please try again.", "danger")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for third-party integrations (e.g., Instagram Webhooks, Chrome Extensions).
    Accepts JSON:
        { "text": "comment string", "model": "logistic_regression" (optional) }
    Returns JSON:
        { "status": "success", "prediction": "Bullying"|"Non-Bullying", "confidence": float, "should_hide": bool }
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_text = data.get("text", "")
        requested_model = data.get("model", predictor.model_type).strip().lower()

        if not raw_text or not raw_text.strip():
            return jsonify({"status": "error", "message": "Text payload is empty"}), 400

        if requested_model not in SUPPORTED_MODELS:
            requested_model = predictor.model_type

        # Model instance
        if requested_model == predictor.model_type:
            active_predictor = predictor
        else:
            active_predictor = CyberbullyingPredictor(model_type=requested_model)

        result = active_predictor.predict(raw_text)
        is_bullying = result["prediction"] == "Bullying"

        return jsonify({
            "status": "success",
            "text": raw_text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "should_hide": is_bullying,
            "model_type": result["model_type"]
        }), 200
    except Exception as e:
        logger.error(f"API prediction error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/poster", methods=["GET"])
def poster():
    """Renders the publication-ready academic project poster."""
    return render_template("poster.html")


@app.route("/history", methods=["GET"])
def history():
    """Renders historical classification records stored in SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, text_snippet, prediction, confidence, model_type, timestamp
            FROM predictions
            ORDER BY id DESC
            """
        )
        records = cursor.fetchall()
        conn.close()

        return render_template("history.html", records=records, total_count=len(records))

    except Exception as err:
        logger.error(f"Error fetching history: {err}", exc_info=True)
        flash("Could not retrieve history records from database.", "danger")
        return render_template("history.html", records=[], total_count=0)


@app.route("/clear-history", methods=["POST"])
@login_required
def clear_history():
    """Deletes prediction records strictly for the current user."""
    user_id = session.get("user_id")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

        # Clear user's MongoDB predictions history
        try:
            MongoDBManager.clear_user_prediction_history(user_id)
        except Exception as me:
            logger.warning(f"Notice: Failed to clear MongoDB prediction history for user {user_id}: {me}")

        flash("Your personal prediction history has been cleared.", "info")
    except Exception as err:
        logger.error(f"Error clearing history for user {user_id}: {err}", exc_info=True)
        flash("Failed to clear history. An error occurred.", "danger")

    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    """
    Computes summary metrics, counts, and Chart.js datasets strictly for the
    currently authenticated user. Specific users can only see their own data.
    """
    user_id = session.get("user_id")

    try:
        # Retrieve user-isolated analytics from MongoDB
        user_analytics = MongoDBManager.get_user_analytics(user_id)
        if user_analytics and (user_analytics["total_count"] > 0 or MongoDBManager.users_col() is not None):
            return render_template(
                "dashboard.html",
                total_count=user_analytics["total_count"],
                bullying_count=user_analytics["bullying_count"],
                non_bullying_count=user_analytics["non_bullying_count"],
                bullying_pct=user_analytics["bullying_pct"],
                non_bullying_pct=user_analytics["non_bullying_pct"],
                avg_confidence=user_analytics["avg_confidence"],
                recent_records=user_analytics["recent_records"],
                timeline_labels=user_analytics["timeline_labels"],
                timeline_confidences=user_analytics["timeline_confidences"],
                timeline_colors=user_analytics["timeline_colors"],
            )

        # Fallback to SQLite filtered strictly by user_id
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?", (user_id,))
        total_row = cursor.fetchone()
        total_count = total_row["total"] if total_row else 0

        cursor.execute(
            """
            SELECT prediction, COUNT(*) as cnt
            FROM predictions
            WHERE user_id = ?
            GROUP BY prediction
            """,
            (user_id,)
        )
        counts = {row["prediction"]: row["cnt"] for row in cursor.fetchall()}
        bullying_count = counts.get("Bullying", 0)
        non_bullying_count = counts.get("Non-Bullying", 0)

        cursor.execute("SELECT AVG(confidence) AS avg_conf FROM predictions WHERE user_id = ?", (user_id,))
        avg_row = cursor.fetchone()
        avg_confidence = round(float(avg_row["avg_conf"]), 1) if avg_row and avg_row["avg_conf"] is not None else 0.0

        cursor.execute(
            """
            SELECT id, text_snippet, prediction, confidence, model_type, timestamp
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
            """,
            (user_id,)
        )
        recent_records = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, prediction, confidence, timestamp
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user_id,)
        )
        timeline_rows = cursor.fetchall()[::-1]

        timeline_labels = [f"#{r['id']}" for r in timeline_rows]
        timeline_confidences = [float(r["confidence"]) for r in timeline_rows]
        timeline_colors = [
            "#b83a3a" if r["prediction"] == "Bullying" else "#556b2f"
            for r in timeline_rows
        ]

        bullying_pct = round((bullying_count / total_count * 100), 1) if total_count > 0 else 0.0
        non_bullying_pct = round((non_bullying_count / total_count * 100), 1) if total_count > 0 else 0.0

        conn.close()

        return render_template(
            "dashboard.html",
            total_count=total_count,
            bullying_count=bullying_count,
            non_bullying_count=non_bullying_count,
            bullying_pct=bullying_pct,
            non_bullying_pct=non_bullying_pct,
            avg_confidence=avg_confidence,
            recent_records=recent_records,
            timeline_labels=timeline_labels,
            timeline_confidences=timeline_confidences,
            timeline_colors=timeline_colors,
        )

    except Exception as err:
        logger.error(f"Error computing dashboard analytics for user {user_id}: {err}", exc_info=True)
        flash("Could not compile personal dashboard analytics.", "danger")
        return render_template(
            "dashboard.html",
            total_count=0,
            bullying_count=0,
            non_bullying_count=0,
            bullying_pct=0.0,
            non_bullying_pct=0.0,
            avg_confidence=0.0,
            recent_records=[],
            timeline_labels=[],
            timeline_confidences=[],
            timeline_colors=[],
        )


# ------------------------------------------------------------------------------
# Error Handlers (Clean user-facing pages, no stack traces exposed in production)
# ------------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 Not Found error page."""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 Internal Server Error page."""
    logger.error(f"500 Internal Server Error: {e}")
    return render_template("500.html"), 500


@app.errorhandler(400)
def bad_request_error(e):
    """Custom 400 Bad Request error page."""
    return render_template("404.html", error_code="400", error_message="Bad Request"), 400


# ------------------------------------------------------------------------------
# Application Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = app.config["DEBUG"]

    print("=" * 60)
    print(" CyberShield: Cyberbullying Detection Web Application")
    print(f" Running at: http://{host}:{port}/")
    print(f" Default Model: {app.config['MODEL_TYPE']}")
    print(f" Debug Mode: {debug}")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)
