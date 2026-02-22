from flask import Blueprint, jsonify, render_template, request, session

from services.analysis_service import AnalysisService
from database import create_user, authenticate_user

bp = Blueprint("main", __name__)
analysis_service = AnalysisService()


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Backend skeleton is running",
        "analysis_ready": analysis_service.is_ready()
    })


# Auth
@bp.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    ok, result = create_user(
        username=data.get("username"),
        password=data.get("password"),
    )

    if not ok:
        return jsonify({"error": result}), 400

    # Автологін після реєстрації
    session["user_id"] = result["id"]
    session["username"] = result["username"]

    return jsonify({
        "message": "Registration successful",
        "user": result
    })


@bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    user = authenticate_user(
        username=data.get("username"),
        password=data.get("password"),
    )

    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    return jsonify({
        "message": "Login successful",
        "user": user
    })


@bp.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logout successful"})


@bp.route("/api/auth/status", methods=["GET"])
def auth_status():
    if "user_id" not in session:
        return jsonify({"authenticated": False})

    return jsonify({
        "authenticated": True,
        "user": {
            "id": session["user_id"],
            "username": session["username"]
        }
    })
