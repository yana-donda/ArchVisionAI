from __future__ import annotations

import logging
import re
from functools import wraps
from pathlib import Path
from werkzeug.exceptions import RequestEntityTooLarge

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
    update_query_history_ai_analysis,
)

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)

USERNAME_PATTERN = re.compile(r"^[A-Za-zА-Яа-яІіЇїЄєҐґ0-9_]+$")

PASSWORD_PATTERN = re.compile(r"^[A-Za-z0-9!@#$%_.?\-]+$")


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
    data_dir = (_base_dir() / "data").resolve()
    resolved = (data_dir / filename).resolve()

    try:
        relative_path = resolved.relative_to(data_dir)
    except ValueError:
        return jsonify({"error": "File not found"}), 404

    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(str(data_dir), relative_path.as_posix())


@bp.route("/api/auth/register", methods=["POST"])
def register():
    try:
        if not request.is_json:
            return jsonify({"error": "Некоректний запит"}), 400

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "Вкажіть ім’я користувача та пароль"}), 400

        if len(username) < 4:
            return jsonify({"error": "Логін має містити щонайменше 4 символи"}), 400

        if len(username) > 50:
            return jsonify({"error": "Логін має містити не більше 50 символів"}), 400

        if not USERNAME_PATTERN.fullmatch(username):
            return jsonify({
                "error": "Логін може містити тільки літери української або англійської абетки, цифри та нижнє підкреслення"
            }), 400

        if len(password) < 8:
            return jsonify({"error": "Пароль має містити щонайменше 8 символів"}), 400

        if len(password) > 20:
            return jsonify({"error": "Пароль має містити не більше 20 символів"}), 400

        if not PASSWORD_PATTERN.fullmatch(password):
            return jsonify({
                "error": "Пароль може містити тільки англійські літери, цифри та символи: ! @ # $ % _ - . ?"
            }), 400

        if not re.search(r"[A-Za-z]", password):
            return jsonify({"error": "Пароль має містити щонайменше одну англійську літеру"}), 400

        if not re.search(r"\d", password):
            return jsonify({"error": "Пароль має містити щонайменше одну цифру"}), 400

        ok, msg = create_user(username, password, _base_dir())
        if not ok:
            return jsonify({"error": msg}), 400

        return jsonify({"message": msg})

    except Exception as e:
        logger.exception("Registration error")
        return jsonify({"error": f"Помилка сервера: {str(e)}"}), 500


@bp.route("/api/auth/login", methods=["POST"])
def login():
    try:
        if not request.is_json:
            return jsonify({"error": "Некоректний запит"}), 400

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "Ім'я користувача та пароль є обов'язковими"}), 400

        user = authenticate_user(username, password, _base_dir())
        if not user:
            return jsonify({"error": "Неправильне ім'я користувача або пароль"}), 401

        session["user_id"] = user["id"]
        session["username"] = user["username"]

        return jsonify({"message": "Вхід успішно виконано", "user": user})

    except Exception as e:
        logger.exception("Login error")
        return jsonify({"error": f"Помилка сервера: {str(e)}"}), 500


@bp.route("/api/auth/logout", methods=["POST"])
def logout_route():
    session.clear()
    return jsonify({"message": "Вихід успішно виконано"})


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

        result = service.analyze_request(request, include_gemini=False)

        if "user_id" in session and not result.get("error"):
            meta = result.get("_meta", {})
            style_data = result.get("architectural_style", {})
            top_pred = style_data.get("top_prediction", {})

            history_id = save_query_history(
                user_id=session["user_id"],
                image_name=meta.get("image_name", "uploaded_image"),
                image_thumbnail=meta.get("image_thumbnail", ""),
                architectural_style=top_pred.get("style_uk") or top_pred.get("style") or "Невідомо",
                confidence=float(top_pred.get("confidence", 0.0)),
                ai_analysis="",
                base_dir=_base_dir(),
            )

            result["history_id"] = history_id

        result.pop("_meta", None)
        return jsonify(result)

    except RequestEntityTooLarge:
        return jsonify({
            "error": "Файл занадто великий. Максимальний розмір — 20 МБ."
        }), 413
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Analyze error")
        return jsonify({"error": f"Помилка аналізу: {str(e)}"}), 500
    
    
@bp.route("/api/analyze/gemini", methods=["POST"])
@login_required
def analyze_gemini():
    try:
        data = request.get_json(silent=True) if request.is_json else {}
        history_id = data.get("history_id")

        result = _service().analyze_gemini_request(request)

        gemini = result.get("gemini_analysis", {}) or {}
        ai_text = (
            gemini.get("analysis")
            or gemini.get("summary")
            or gemini.get("description")
            or gemini.get("historical_context")
            or gemini.get("error")
            or ""
        )

        if history_id and ai_text:
            try:
                update_query_history_ai_analysis(
                    user_id=session["user_id"],
                    history_id=int(history_id),
                    ai_analysis=ai_text,
                    base_dir=_base_dir(),
                )
            except (TypeError, ValueError):
                logger.warning("Invalid history_id for Gemini update: %r", history_id)

        return jsonify(result)

    except RequestEntityTooLarge:
        return jsonify({
            "error": "Файл занадто великий. Максимальний розмір — 20 МБ."
        }), 413
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Gemini analyze error")
        return jsonify({"error": f"Помилка Gemini: {str(e)}"}), 500


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


@bp.route("/api/models/available")
def available_models():
    try:
        service = _service()
        modes = service.get_available_modes()

        return jsonify(
            {
                "available": True,
                "models": modes,
                "current_model": service.get_current_mode(),
                "count": len(modes),
            }
        )
    except Exception as e:
        logger.exception("Available models error")
        return jsonify({"available": False, "message": str(e), "models": []}), 500


@bp.route("/api/models/current")
def current_model():
    return jsonify({"model_type": _service().get_current_mode()})


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