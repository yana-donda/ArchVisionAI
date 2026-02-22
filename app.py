import os
from flask import Flask
from flask_cors import CORS
from flask_session import Session

from database import init_db
from routes import bp as main_bp


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["SECRET_KEY"] = os.getenv("ARCHVISION_SECRET_KEY", "archvision-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    # Flask session
    os.makedirs("sessions", exist_ok=True)
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = os.path.abspath("sessions")
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_USE_SIGNER"] = True

    CORS(app)
    Session(app)

    # Ініціалізація БД
    init_db()

    # Підключення роутів
    app.register_blueprint(main_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)