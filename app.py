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
from typing import Optional, Dict, Any, List

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    abort
)

# Import predictor from src.predict as required
from src.predict import CyberbullyingPredictor
from src.genai_explainer import GenAIExplainer

genai_service = GenAIExplainer()

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# Initialize database table on app startup
init_db()

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
# Routes
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

        # Run inference
        result = active_predictor.predict(raw_text)

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

        # Store prediction in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (text_snippet, prediction, confidence, model_type, timestamp)
            VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
            """,
            (text_snippet, prediction_label, confidence_pct, model_name_used)
        )
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        # Generative AI & Prompt Engineering Linguistic Analysis
        genai_data = genai_service.explain_and_rewrite(raw_text, is_bullying)

        # Render result page
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
            genai=genai_data
        )

    except Exception as err:
        logger.error(f"Prediction failed with unexpected error: {err}", exc_info=True)
        flash("An unexpected error occurred during analysis. Please try again.", "danger")
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
def clear_history():
    """Deletes all records from the predictions table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        conn.commit()
        conn.close()

        flash("Prediction history has been permanently cleared.", "info")
    except Exception as err:
        logger.error(f"Error clearing history: {err}", exc_info=True)
        flash("Failed to clear history. An error occurred.", "danger")

    return redirect(url_for("history"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Computes summary metrics, counts, and Chart.js datasets for the
    analytics dashboard.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total predictions count
        cursor.execute("SELECT COUNT(*) AS total FROM predictions")
        total_row = cursor.fetchone()
        total_count = total_row["total"] if total_row else 0

        # Bullying vs Non-Bullying counts
        cursor.execute(
            """
            SELECT prediction, COUNT(*) as cnt
            FROM predictions
            GROUP BY prediction
            """
        )
        counts = {row["prediction"]: row["cnt"] for row in cursor.fetchall()}
        bullying_count = counts.get("Bullying", 0)
        non_bullying_count = counts.get("Non-Bullying", 0)

        # Average confidence calculation
        cursor.execute("SELECT AVG(confidence) AS avg_conf FROM predictions")
        avg_row = cursor.fetchone()
        avg_confidence = round(float(avg_row["avg_conf"]), 1) if avg_row and avg_row["avg_conf"] is not None else 0.0

        # Recent predictions (last 10 records for quick view)
        cursor.execute(
            """
            SELECT id, text_snippet, prediction, confidence, model_type, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
            """
        )
        recent_records = cursor.fetchall()

        # Recent predictions for timeline / bar chart (last 10 IDs in chronological order)
        cursor.execute(
            """
            SELECT id, prediction, confidence, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
            """
        )
        timeline_rows = cursor.fetchall()[::-1]

        timeline_labels = [f"#{r['id']}" for r in timeline_rows]
        timeline_confidences = [float(r["confidence"]) for r in timeline_rows]
        timeline_colors = [
            "rgba(239, 68, 68, 0.85)" if r["prediction"] == "Bullying" else "rgba(16, 185, 129, 0.85)"
            for r in timeline_rows
        ]

        # Calculate percentages
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
        logger.error(f"Error computing dashboard analytics: {err}", exc_info=True)
        flash("Could not compile dashboard analytics.", "danger")
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
