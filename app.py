from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify

from database import init_db
from routes import bp
from services.analysis_service import AnalysisService


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    # Basic config
    app.config["BASE_DIR"] = base_dir
    secret_key = os.getenv("SECRET_KEY")
    app.config["SECRET_KEY"] = secret_key
    app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # DB init
    init_db(base_dir)

    # Services
    app.config["ANALYSIS_SERVICE"] = AnalysisService(base_dir)

    # Routes
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "Розмір файлу занадто великий. Максимальний розмір — 15 МБ."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)