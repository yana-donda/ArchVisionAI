from flask import Blueprint, jsonify, render_template

from services.analysis_service import AnalysisService

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