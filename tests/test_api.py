"""Integration tests for FastAPI application endpoints."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from kyc_risk.synth import render_document
from kyc_risk.train import train_pipeline


@pytest.fixture(scope="module")
def test_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Fixture initializing FastAPI TestClient with tiny pre-trained artifacts."""
    tmp_dir = tmp_path_factory.mktemp("api_artifacts")
    train_pipeline(n=300, epochs=1, out_dir=str(tmp_dir), seed=42, batch_size=32)
    app = create_app(artifacts_dir=str(tmp_dir))
    return TestClient(app)


def test_health_endpoint(test_client: TestClient) -> None:
    """Verify /health returns 200 status code and readiness confirmation."""
    response = test_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


@pytest.mark.parametrize("kind", ["genuine", "tampered"])
def test_sample_document_endpoint(test_client: TestClient, kind: str) -> None:
    """Verify /sample/{kind} generates valid PNG image responses."""
    response = test_client.get(f"/sample/{kind}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 100

    # Ensure returned payload can be read as a valid PIL image
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (96, 96)


def test_sample_invalid_kind(test_client: TestClient) -> None:
    """Verify /sample/{kind} returns 400 Bad Request for unrecognized document kinds."""
    response = test_client.get("/sample/corrupted")
    assert response.status_code == 400
    assert "Invalid sample kind" in response.json()["detail"]


def test_score_endpoint_valid_applicant(test_client: TestClient) -> None:
    """Verify /score endpoint processes valid multipart request and returns risk assessment."""
    doc_arr = render_document(np.random.default_rng(99), tampered=False, size=96)
    img = Image.fromarray(doc_arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    applicant = {
        "age": 31,
        "account_age_days": 240,
        "country_risk": 0.15,
        "device_risk": 0.22,
        "num_failed_logins_30d": 1,
        "selfie_match_score": 0.94,
        "address_mismatch": 0,
        "pep_flag": 0,
        "velocity_signups_same_device": 1,
    }

    response = test_client.post(
        "/score",
        files={"file": ("doc.png", img_bytes, "image/png")},
        data={"applicant": json.dumps(applicant)},
    )
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["decision"] in ("manual_review", "auto_approve")
    assert isinstance(data["signals"], list)


def test_score_endpoint_invalid_applicant_json(test_client: TestClient) -> None:
    """Verify /score endpoint returns 422 Unprocessable Entity on malformed JSON string."""
    fake_img = io.BytesIO(b"fake_image_bytes")
    response = test_client.post(
        "/score",
        files={"file": ("doc.png", fake_img, "image/png")},
        data={"applicant": "{invalid_json_formatting:"},
    )
    assert response.status_code == 422


def test_score_endpoint_pydantic_validation_failure(test_client: TestClient) -> None:
    """Verify /score endpoint returns 422 when required fields violate schema constraints."""
    doc_arr = render_document(np.random.default_rng(1), tampered=False, size=96)
    img = Image.fromarray(doc_arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Underage applicant (< 18) violates Pydantic constraints
    invalid_applicant = {
        "age": 16,
        "account_age_days": 10,
        "country_risk": 0.2,
        "device_risk": 0.1,
        "num_failed_logins_30d": 0,
        "selfie_match_score": 0.8,
        "address_mismatch": 0,
        "pep_flag": 0,
        "velocity_signups_same_device": 1,
    }

    response = test_client.post(
        "/score",
        files={"file": ("doc.png", img_bytes, "image/png")},
        data={"applicant": json.dumps(invalid_applicant)},
    )
    assert response.status_code == 422
