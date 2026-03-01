from __future__ import annotations

import json
import logging
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
)

from database import (
    authenticate_user,
    create_user,
    get_user_history,
    get_user_preferences,
    get_user_stats,
    save_query_history,
)

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


def _base_dir() -> Path:
    return Path(current_app.config["BASE_DIR"])


def _service():
    return current_app.config["ANALYSIS_SERVICE"]


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return wrapper


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/models/<path:filename>")
def serve_models_json(filename: str):
    data_dir = _base_dir() / "data"
    file_path = data_dir / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(data_dir), filename)


@bp.route("/api/auth/register", methods=["POST"])
def register():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request"}), 400

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        if len(password) < 4:
            return jsonify({"error": "Password must be at least 4 characters"}), 400

        ok, msg = create_user(username, password, _base_dir())
        if not ok:
            return jsonify({"error": msg}), 400

        return jsonify({"message": msg})

    except Exception as e:
        logger.exception("Registration error")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@bp.route("/api/auth/login", methods=["POST"])
def login():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request"}), 400

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        user = authenticate_user(username, password, _base_dir())
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        session["user_id"] = user["id"]
        session["username"] = user["username"]

        return jsonify({"message": "Login successful", "user": user})

    except Exception as e:
        logger.exception("Login error")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@bp.route("/api/auth/logout", methods=["POST"])
def logout_route():
    session.clear()
    return jsonify({"message": "Logout successful"})


@bp.route("/api/auth/status")
def auth_status():
    if "user_id" in session:
        return jsonify(
            {
                "authenticated": True,
                "user": {"id": session["user_id"], "username": session["username"]},
            }
        )
    return jsonify({"authenticated": False})


@bp.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        service = _service()
        current_mode = current_app.config.get("CURRENT_ANALYSIS_MODE", "efficientnet_b0")

        result = service.analyze_request(request, mode_override=current_mode)

        # Зберігаємо в history тільки для авторизованого користувача
        if "user_id" in session and not result.get("error"):
            meta = result.get("_meta", {})
            style_data = result.get("architectural_style", {})
            top_pred = style_data.get("top_prediction", {})

            ai_text = ""
            gemini = result.get("gemini_analysis", {}) or {}
            ai_text = (
                gemini.get("analysis")
                or gemini.get("summary")
                or gemini.get("historical_context")
                or gemini.get("error")
                or ""
            )

            save_query_history(
                user_id=session["user_id"],
                image_name=meta.get("image_name", "uploaded_image"),
                image_thumbnail=meta.get("image_thumbnail", ""),
                architectural_style=top_pred.get("style_uk") or top_pred.get("style") or "Невідомо",
                confidence=float(top_pred.get("confidence", 0.0)),
                ai_analysis=ai_text,
                base_dir=_base_dir(),
            )

        result.pop("_meta", None)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Analyze error")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@bp.route("/api/user/history")
@login_required
def user_history():
    try:
        data = get_user_history(session["user_id"], _base_dir())
        return jsonify({"history": data})
    except Exception as e:
        logger.exception("History error")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/user/stats")
@login_required
def user_stats():
    try:
        data = get_user_stats(session["user_id"], _base_dir())
        return jsonify(data)
    except Exception as e:
        logger.exception("Stats error")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/user/preferences")
@login_required
def user_preferences():
    try:
        data = get_user_preferences(session["user_id"], _base_dir())
        return jsonify({"preferences": data})
    except Exception as e:
        logger.exception("Preferences error")
        return jsonify({"error": str(e)}), 500


# Model mode endpoints
@bp.route("/api/models/available")
def available_models():
    try:
        service = _service()
        trained = service.get_trained_models()
        current = current_app.config.get("CURRENT_ANALYSIS_MODE", "efficientnet_b0")

        return jsonify(
            {
                "available": True,
                "models": service.get_available_modes(),
                "trained_models": trained,
                "current_model": current,
                "count": len(service.get_available_modes()),
            }
        )
    except Exception as e:
        logger.exception("Available models error")
        return jsonify({"available": False, "message": str(e), "models": []}), 500


@bp.route("/api/models/current")
def current_model():
    return jsonify({"model_type": current_app.config.get("CURRENT_ANALYSIS_MODE", "efficientnet_b0")})


@bp.route("/api/models/switch", methods=["POST"])
def switch_model():
    try:
        data = request.get_json(silent=True) or {}
        model_type = data.get("model_type")

        if not model_type:
            return jsonify({"success": False, "message": "Не вказано тип моделі"}), 400

        ok, msg = _service().switch_mode(model_type)
        if not ok:
            return jsonify({"success": False, "message": msg}), 400

        current_app.config["CURRENT_ANALYSIS_MODE"] = model_type

        names = {
            "efficientnet_b0": "EfficientNet-B0",
            "resnet50": "ResNet-50",
            "ensemble": "Ensemble (обидві моделі)",
        }
        return jsonify({"success": True, "message": f"Режим аналізу: {names.get(model_type, model_type)}"})

    except Exception as e:
        logger.exception("Switch model error")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/api/models/info/<model_type>")
def model_info(model_type: str):
    info = _service().get_model_info(model_type)
    if not info:
        return jsonify({"error": "Модель не знайдена"}), 404
    return jsonify(info)


# Dataset background images
@bp.route("/api/dataset/images")
def dataset_images():
    return jsonify({"images": _service().get_dataset_images(limit=20)})


@bp.route("/api/dataset/image/<path:filepath>")
def dataset_image(filepath: str):
    resolved = _service().resolve_dataset_image(filepath)
    if not resolved:
        return jsonify({"error": "Image not found"}), 404

    directory, filename = resolved
    return send_from_directory(directory, filename)


# Checkpoints
@bp.route("/api/training/checkpoints")
def training_checkpoints():
    return jsonify({"checkpoints": _service().list_checkpoints()})