from __future__ import annotations

import base64
import io
import shutil
from pathlib import Path

import pytest
from flask import Flask, jsonify
from PIL import Image

from database import get_user_history, get_user_preferences, get_user_stats, init_db
from routes import bp
from services.analysis_service import AnalysisService
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture()
def real_base_dir(tmp_path: Path) -> Path:
    project_dir = Path(__file__).resolve().parents[1]

    source_data_dir = project_dir / "data"
    source_checkpoints_dir = project_dir / "checkpoints"
    source_dataset_dir = project_dir / "dataset"

    target_data_dir = tmp_path / "data"
    target_checkpoints_dir = tmp_path / "checkpoints"
    target_dataset_dir = tmp_path / "dataset"

    if not source_data_dir.exists():
        pytest.fail(f"Missing data folder: {source_data_dir}")

    if not (source_data_dir / "class_mapping.json").exists():
        pytest.fail(
            f"Missing required file: {source_data_dir / 'class_mapping.json'}"
        )

    shutil.copytree(source_data_dir, target_data_dir, dirs_exist_ok=True)

    if source_checkpoints_dir.exists():
        shutil.copytree(
            source_checkpoints_dir,
            target_checkpoints_dir,
            dirs_exist_ok=True,
        )

    if source_dataset_dir.exists():
        shutil.copytree(
            source_dataset_dir,
            target_dataset_dir,
            dirs_exist_ok=True,
        )

    if not (target_data_dir / "class_mapping.json").exists():
        pytest.fail(
            f"class_mapping.json was not copied to test folder: "
            f"{target_data_dir / 'class_mapping.json'}"
        )

    return tmp_path


@pytest.fixture()
def app(real_base_dir: Path) -> Flask:
    project_dir = Path(__file__).resolve().parents[1]

    app = Flask(
        __name__,
        template_folder=str(project_dir / "templates"),
        static_folder=str(project_dir / "static"),
    )

    app.config["TESTING"] = True
    app.config["BASE_DIR"] = real_base_dir
    app.config["SECRET_KEY"] = "test-secret-key"
    app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
    app.config["ANALYSIS_SERVICE"] = AnalysisService(real_base_dir)

    init_db(real_base_dir)
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(
            {
                "error": "Розмір файлу занадто великий. Максимальний розмір — 15 МБ."
            }
        ), 413

    return app


@pytest.fixture()
def client(app: Flask):
    return app.test_client()


def create_test_image_bytes(
    size: tuple[int, int] = (224, 224),
    image_format: str = "JPEG",
) -> bytes:
    image = Image.new("RGB", size, color=(180, 180, 180))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer.getvalue()


def read_sample_architecture_image_bytes(base_dir: Path) -> bytes:
    """
    Reads a real architecture image for tests.

    The project demo dataset may be either:
    1. flat: dataset/*.jpg
    2. structured: dataset/train/architecture or dataset/val/architecture

    The helper first tries structured folders, then falls back to any image
    inside dataset, except folders named not_architecture.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    preferred_dirs = [
        base_dir / "dataset" / "val" / "architecture",
        base_dir / "dataset" / "train" / "architecture",
        base_dir / "dataset",
    ]

    for directory in preferred_dirs:
        if not directory.exists():
            continue

        for image_path in sorted(directory.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            if "not_architecture" in {part.lower() for part in image_path.parts}:
                continue

            return image_path.read_bytes()

    pytest.fail(
        "No architecture test image found. "
        "Put at least one building image into dataset/."
    )


def create_not_architecture_image_bytes(
    size: tuple[int, int] = (224, 224),
    image_format: str = "JPEG",
) -> bytes:
    """
    Creates a simple non-architecture image for testing the binary filter.
    """
    image = Image.new("RGB", size, color=(180, 180, 180))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer.getvalue()


def image_to_base64_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def assert_real_model_ready(base_dir: Path, model_type: str) -> None:
    checkpoint_path = base_dir / "checkpoints" / f"{model_type}_best.pth"
    mapping_path = base_dir / "data" / "class_mapping.json"

    assert mapping_path.exists(), f"Missing required file: {mapping_path}"
    assert checkpoint_path.exists(), f"Missing required checkpoint: {checkpoint_path}"


def register_and_login(client, username: str = "testuser", password: str = "Password123"):
    register_response = client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    assert register_response.status_code == 200

    login_response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login_response.status_code == 200

    return login_response.get_json()["user"]


def test_index_page_opens(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data


def test_real_register_login_status_logout_flow(client):
    register_response = client.post(
        "/api/auth/register",
        json={"username": "realuser", "password": "Password123"},
    )

    assert register_response.status_code == 200
    assert "Реєстрація успішна" in register_response.get_json()["message"]

    login_response = client.post(
        "/api/auth/login",
        json={"username": "realuser", "password": "Password123"},
    )

    assert login_response.status_code == 200
    assert login_response.get_json()["user"]["username"] == "realuser"

    status_response = client.get("/api/auth/status")

    assert status_response.status_code == 200
    assert status_response.get_json()["authenticated"] is True

    logout_response = client.post("/api/auth/logout")

    assert logout_response.status_code == 200

    status_after_logout = client.get("/api/auth/status")

    assert status_after_logout.status_code == 200
    assert status_after_logout.get_json()["authenticated"] is False


def test_real_auth_rejects_wrong_password(client):
    register_response = client.post(
        "/api/auth/register",
        json={"username": "wrongpass", "password": "Password123"},
    )
    assert register_response.status_code == 200

    login_response = client.post(
        "/api/auth/login",
        json={"username": "wrongpass", "password": "WrongPassword123"},
    )

    assert login_response.status_code == 401
    assert "Неправильне" in login_response.get_json()["error"]


def test_real_protected_routes_require_login(client):
    for url in (
        "/api/user/history",
        "/api/user/stats",
        "/api/user/preferences",
        "/api/analyze/gemini",
    ):
        response = client.get(url) if "gemini" not in url else client.post(url, json={})

        assert response.status_code == 401


def test_real_available_models_endpoint(client):
    response = client.get("/api/models/available")

    assert response.status_code == 200

    data = response.get_json()

    assert data["available"] is True
    assert data["current_model"] == "efficientnet_b0"
    assert data["count"] == 3

    model_types = {model["type"] for model in data["models"]}

    assert {"efficientnet_b0", "resnet50", "ensemble"} <= model_types


def test_real_switch_model_endpoint(client):
    response = client.post(
        "/api/models/switch",
        json={"model_type": "resnet50"},
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True

    current_response = client.get("/api/models/current")

    assert current_response.status_code == 200
    assert current_response.get_json()["model_type"] == "resnet50"

    invalid_response = client.post(
        "/api/models/switch",
        json={"model_type": "unknown_model"},
    )

    assert invalid_response.status_code == 400
    assert invalid_response.get_json()["success"] is False


def test_real_model_info_endpoint(client):
    response = client.get("/api/models/info/efficientnet_b0")

    assert response.status_code == 200

    data = response.get_json()

    assert data["type"] == "efficientnet_b0"
    assert data["name"] == "EfficientNet-B0"
    assert data["input_size"] == 224

    missing_response = client.get("/api/models/info/unknown_model")

    assert missing_response.status_code == 404


def test_real_analyze_requires_image_data(client):
    response = client.post("/api/analyze", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Image data required" in response.get_json()["error"]


def test_real_analyze_rejects_invalid_image(client):
    response = client.post(
        "/api/analyze",
        data={"file": (io.BytesIO(b"not a real image"), "broken.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 500 or response.status_code == 400
    assert "error" in response.get_json()


def test_real_analyze_returns_413_for_real_large_request(client):
    original_limit = client.application.config["MAX_CONTENT_LENGTH"]
    client.application.config["MAX_CONTENT_LENGTH"] = 100

    try:
        image_bytes = create_test_image_bytes(size=(512, 512))

        response = client.post(
            "/api/analyze",
            data={"file": (io.BytesIO(image_bytes), "large.jpg")},
            content_type="multipart/form-data",
        )
    finally:
        client.application.config["MAX_CONTENT_LENGTH"] = original_limit

    assert response.status_code == 413
    assert "занадто великий" in response.get_json()["error"]


def test_real_analyze_file_upload_with_efficientnet(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)

    response = client.post(
        "/api/analyze",
        data={
            "file": (io.BytesIO(image_bytes), "test_building.jpg"),
            "model_type": "efficientnet_b0",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data.get("is_architecture") is True
    assert "architectural_style" in data
    assert "supported_styles" in data
    assert "geographical_data" in data

    top_prediction = data["architectural_style"]["top_prediction"]

    assert top_prediction["style"]
    assert top_prediction["style_uk"]
    assert 0 <= top_prediction["confidence"] <= 1


def test_real_analyze_rejects_not_architecture_image(client):
    image_bytes = create_not_architecture_image_bytes()

    response = client.post(
        "/api/analyze",
        data={
            "file": (io.BytesIO(image_bytes), "not_architecture_sample.jpg"),
            "model_type": "efficientnet_b0",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data["is_architecture"] is False
    assert data["skip_history"] is True
    assert "architecture_check" in data
    assert data["architectural_style"]["model"] == "Architecture binary filter"
    assert data["architectural_style"]["top_prediction"]["style"] == "not_architecture"


def test_real_analyze_base64_image_with_efficientnet(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)
    image_data = image_to_base64_data_url(image_bytes)

    response = client.post(
        "/api/analyze",
        json={
            "image": image_data,
            "model_type": "efficientnet_b0",
            "use_tta": False,
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data.get("is_architecture") is True
    assert data["architectural_style"]["model"] == "EfficientNet-B0"
    assert len(data["architectural_style"]["all_predictions"]) > 0


def test_real_analyze_with_tta(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)
    image_data = image_to_base64_data_url(image_bytes)

    response = client.post(
        "/api/analyze",
        json={
            "image": image_data,
            "model_type": "efficientnet_b0",
            "use_tta": True,
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data.get("is_architecture") is True
    assert data["architectural_style"]["tta_augmentations"] == 5
    assert "TTA" in data["architectural_style"]["model"]


def test_real_analyze_with_resnet50(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "resnet50")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)
    image_data = image_to_base64_data_url(image_bytes)

    response = client.post(
        "/api/analyze",
        json={
            "image": image_data,
            "model_type": "resnet50",
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data.get("is_architecture") is True
    assert data["architectural_style"]["model"] == "ResNet-50"


def test_real_analyze_with_ensemble(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")
    assert_real_model_ready(real_base_dir, "resnet50")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)
    image_data = image_to_base64_data_url(image_bytes)

    response = client.post(
        "/api/analyze",
        json={
            "image": image_data,
            "model_type": "ensemble",
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert data.get("is_architecture") is True
    assert data["architectural_style"]["model"] == "Ensemble (EfficientNet-B0 + ResNet-50)"


def test_real_logged_in_analyze_saves_history_stats_and_preferences(
    client,
    real_base_dir: Path,
):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")

    user = register_and_login(client, username="historyuser", password="Password123")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)

    analyze_response = client.post(
        "/api/analyze",
        data={
            "file": (io.BytesIO(image_bytes), "history_building.jpg"),
            "model_type": "efficientnet_b0",
        },
        content_type="multipart/form-data",
    )

    assert analyze_response.status_code == 200

    analyze_data = analyze_response.get_json()

    assert "error" not in analyze_data
    assert "history_id" in analyze_data

    history_response = client.get("/api/user/history")
    stats_response = client.get("/api/user/stats")
    preferences_response = client.get("/api/user/preferences")

    assert history_response.status_code == 200
    assert stats_response.status_code == 200
    assert preferences_response.status_code == 200

    history = history_response.get_json()["history"]
    stats = stats_response.get_json()
    preferences = preferences_response.get_json()["preferences"]

    assert len(history) == 1
    assert history[0]["image_name"] == "history_building.jpg"
    assert history[0]["confidence"] >= 0

    assert stats["total_analyses"] == 1
    assert stats["favorite_style"] is not None
    assert len(stats["popular_styles"]) == 1

    assert len(preferences) == 1
    assert preferences[0]["preference_score"] == 1.0

    db_history = get_user_history(user["id"], real_base_dir)
    db_stats = get_user_stats(user["id"], real_base_dir)
    db_preferences = get_user_preferences(user["id"], real_base_dir)

    assert len(db_history) == 1
    assert db_stats["total_analyses"] == 1
    assert len(db_preferences) == 1


def test_real_anonymous_analyze_does_not_save_history(client, real_base_dir: Path):
    assert_real_model_ready(real_base_dir, "efficientnet_b0")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)

    response = client.post(
        "/api/analyze",
        data={
            "file": (io.BytesIO(image_bytes), "anonymous.jpg"),
            "model_type": "efficientnet_b0",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "error" not in data
    assert "history_id" not in data


def test_real_gemini_analysis_returns_response(client, real_base_dir: Path):
    service = client.application.config["ANALYSIS_SERVICE"]

    if not service.analyzer.gemini_keys:
        pytest.fail(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY_1 in environment or .env file."
        )

    register_and_login(client, username="geminiuser", password="Password123")

    image_bytes = read_sample_architecture_image_bytes(real_base_dir)

    response = client.post(
        "/api/analyze/gemini",
        data={"file": (io.BytesIO(image_bytes), "gemini_test_image.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "gemini_analysis" in data

    gemini = data["gemini_analysis"]

    assert "error" not in gemini, gemini.get("technical_error", gemini.get("error"))

    gemini_text = (
        gemini.get("analysis")
        or gemini.get("description")
        or gemini.get("summary")
        or ""
    ).strip()

    assert gemini_text
    assert len(gemini_text) > 20