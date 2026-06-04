from __future__ import annotations

import base64
import csv
import io
import os
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pytest
import requests
from PIL import Image


BASE_URL = os.getenv("ARCHVISION_BASE_URL", "").rstrip("/")
USERS_COUNT = int(os.getenv("ARCHVISION_LOAD_USERS", "5"))
REQUEST_TIMEOUT = int(os.getenv("ARCHVISION_REQUEST_TIMEOUT", "240"))

RESULTS_DIR = Path("test_results")
RESULTS_FILE = RESULTS_DIR / "deployed_load_results.csv"


def require_base_url() -> str:
    if not BASE_URL:
        pytest.fail(
            "ARCHVISION_BASE_URL is not set. "
            "Set it to the deployed Azure application URL."
        )
    return BASE_URL


def create_test_image_bytes() -> bytes:
    image = Image.new("RGB", (224, 224), color=(180, 180, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def image_to_base64_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def measure_request(method: str, url: str, **kwargs) -> tuple[requests.Response, float]:
    started_at = time.perf_counter()
    response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    duration = time.perf_counter() - started_at
    return response, duration


def measure_session_request(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs,
) -> tuple[requests.Response, float]:
    started_at = time.perf_counter()
    response = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    duration = time.perf_counter() - started_at
    return response, duration


def save_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "scenario",
        "user_index",
        "model_type",
        "status_code",
        "duration_seconds",
        "success",
    ]

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def register_and_login(session: requests.Session, username: str, password: str) -> None:
    base_url = require_base_url()

    register_response, _ = measure_session_request(
        session,
        "POST",
        f"{base_url}/api/auth/register",
        json={"username": username, "password": password},
    )

    assert register_response.status_code == 200, register_response.text

    login_response, _ = measure_session_request(
        session,
        "POST",
        f"{base_url}/api/auth/login",
        json={"username": username, "password": password},
    )

    assert login_response.status_code == 200, login_response.text

    status_response, _ = measure_session_request(
        session,
        "GET",
        f"{base_url}/api/auth/status",
    )

    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["authenticated"] is True


@pytest.mark.deployed
def test_deployed_home_page_and_static_resources_are_available():
    base_url = require_base_url()

    response, duration = measure_request("GET", base_url + "/")

    assert response.status_code == 200
    assert response.status_code < 500
    assert "<html" in response.text.lower()

    static_paths = set(
        re.findall(r'''(?:src|href)=["']([^"']*?/static/[^"']+)["']''', response.text)
    )

    rows = [
        {
            "scenario": "home_page",
            "user_index": "",
            "model_type": "",
            "status_code": response.status_code,
            "duration_seconds": round(duration, 3),
            "success": response.status_code == 200,
        }
    ]

    for path in sorted(static_paths):
        static_url = urljoin(base_url + "/", path)
        static_response, static_duration = measure_request("GET", static_url)

        assert static_response.status_code == 200
        assert static_response.status_code < 500

        rows.append(
            {
                "scenario": "static_resource",
                "user_index": "",
                "model_type": "",
                "status_code": static_response.status_code,
                "duration_seconds": round(static_duration, 3),
                "success": static_response.status_code == 200,
            }
        )

    save_results(rows)


@pytest.mark.deployed
def test_deployed_repeated_page_auth_and_analysis_requests():
    base_url = require_base_url()

    session = requests.Session()
    username = f"repeat_user_{uuid.uuid4().hex[:8]}"
    password = "Password123"

    register_and_login(session, username, password)

    rows = []

    for index in range(3):
        page_response, page_duration = measure_session_request(session, "GET", base_url + "/")

        assert page_response.status_code == 200
        assert page_response.status_code < 500

        rows.append(
            {
                "scenario": "repeated_home_page",
                "user_index": index,
                "model_type": "",
                "status_code": page_response.status_code,
                "duration_seconds": round(page_duration, 3),
                "success": page_response.status_code == 200,
            }
        )

    image_bytes = create_test_image_bytes()

    test_modes = [
        ("efficientnet_b0", False),
        ("resnet50", False),
        ("ensemble", False),
        ("ensemble", True),
    ]

    for model_type, use_tta in test_modes:
        image_data = image_to_base64_data_url(image_bytes)

        analyze_response, analyze_duration = measure_session_request(
            session,
            "POST",
            f"{base_url}/api/analyze",
            json={
                "image": image_data,
                "model_type": model_type,
                "use_tta": use_tta,
            },
        )

        assert analyze_response.status_code == 200, analyze_response.text
        assert analyze_response.status_code < 500

        data = analyze_response.json()

        assert "error" not in data
        assert "architectural_style" in data
        assert "top_prediction" in data["architectural_style"]

        rows.append(
            {
                "scenario": "repeated_analysis_tta" if use_tta else "repeated_analysis",
                "user_index": "",
                "model_type": f"{model_type}{'_tta' if use_tta else ''}",
                "status_code": analyze_response.status_code,
                "duration_seconds": round(analyze_duration, 3),
                "success": analyze_response.status_code == 200,
            }
        )

    history_response, history_duration = measure_session_request(
        session,
        "GET",
        f"{base_url}/api/user/history",
    )

    assert history_response.status_code == 200
    assert history_response.status_code < 500
    assert len(history_response.json()["history"]) >= 1

    rows.append(
        {
            "scenario": "history_after_analysis",
            "user_index": "",
            "model_type": "",
            "status_code": history_response.status_code,
            "duration_seconds": round(history_duration, 3),
            "success": history_response.status_code == 200,
        }
    )

    save_results(rows)


def run_concurrent_user_scenario(user_index: int) -> dict:
    base_url = require_base_url()

    session = requests.Session()
    username = f"load_user_{user_index}_{uuid.uuid4().hex[:8]}"
    password = "Password123"

    model_type = ["efficientnet_b0", "resnet50", "ensemble"][user_index % 3]

    started_at = time.perf_counter()

    page_response, _ = measure_session_request(session, "GET", base_url + "/")
    assert page_response.status_code == 200
    assert page_response.status_code < 500

    register_and_login(session, username, password)

    image_bytes = create_test_image_bytes()
    image_data = image_to_base64_data_url(image_bytes)

    analyze_response, _ = measure_session_request(
        session,
        "POST",
        f"{base_url}/api/analyze",
        json={
            "image": image_data,
            "model_type": model_type,
            "use_tta": False,
        },
    )

    assert analyze_response.status_code == 200, analyze_response.text
    assert analyze_response.status_code < 500

    history_response, _ = measure_session_request(
        session,
        "GET",
        f"{base_url}/api/user/history",
    )

    assert history_response.status_code == 200
    assert history_response.status_code < 500
    assert len(history_response.json()["history"]) >= 1

    duration = time.perf_counter() - started_at

    return {
        "scenario": "concurrent_user_flow",
        "user_index": user_index,
        "model_type": model_type,
        "status_code": 200,
        "duration_seconds": round(duration, 3),
        "success": True,
    }


@pytest.mark.deployed
@pytest.mark.load
def test_deployed_multiple_users_can_work_concurrently():
    require_base_url()

    results = []

    with ThreadPoolExecutor(max_workers=USERS_COUNT) as executor:
        futures = [
            executor.submit(run_concurrent_user_scenario, user_index)
            for user_index in range(USERS_COUNT)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    durations = [result["duration_seconds"] for result in results]

    assert len(results) == USERS_COUNT
    assert max(durations) < REQUEST_TIMEOUT

    save_results(results)

    print("\nDeployed load test results:")
    print(f"Users: {USERS_COUNT}")
    print(f"Average duration: {statistics.mean(durations):.2f}s")
    print(f"Max duration: {max(durations):.2f}s")


@pytest.mark.deployed
@pytest.mark.gemini
def test_deployed_gemini_returns_response():
    if os.getenv("RUN_DEPLOYED_GEMINI_TEST") != "1":
        pytest.skip("Set RUN_DEPLOYED_GEMINI_TEST=1 to run deployed Gemini test.")

    base_url = require_base_url()

    session = requests.Session()
    username = f"gemini_deployed_{uuid.uuid4().hex[:8]}"
    password = "Password123"

    register_and_login(session, username, password)

    image_bytes = create_test_image_bytes()

    response, duration = measure_request(
        "POST",
        f"{base_url}/api/analyze/gemini",
        files={"file": ("gemini_deployed.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        cookies=session.cookies,
    )

    assert response.status_code == 200, response.text
    assert response.status_code < 500

    data = response.json()

    assert "gemini_analysis" in data

    gemini = data["gemini_analysis"]

    assert "error" not in gemini, gemini.get("error")

    gemini_text = (
        gemini.get("analysis")
        or gemini.get("description")
        or gemini.get("summary")
        or ""
    ).strip()

    assert gemini_text
    assert len(gemini_text) > 20

    save_results(
        [
            {
                "scenario": "deployed_gemini",
                "user_index": "",
                "model_type": "gemini",
                "status_code": response.status_code,
                "duration_seconds": round(duration, 3),
                "success": True,
            }
        ]
    )