"""FastAPI application providing health checks, synthetic samples, and multimodal scoring."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from kyc_risk.predict import load, predict
from kyc_risk.synth import render_document
from kyc_risk.train import train_pipeline


class ApplicantData(BaseModel):
    """Schema for validating applicant tabular risk features."""

    age: int = Field(..., ge=18, le=120, description="Applicant age in years.")
    account_age_days: int = Field(..., ge=0, description="Days since user registered account.")
    country_risk: float = Field(..., ge=0.0, le=1.0, description="Jurisdiction risk score.")
    device_risk: float = Field(..., ge=0.0, le=1.0, description="Device fingerprint risk score.")
    num_failed_logins_30d: int = Field(..., ge=0, description="Failed logins in past 30 days.")
    selfie_match_score: float = Field(..., ge=0.0, le=1.0, description="Facial verification score.")
    address_mismatch: int = Field(..., ge=0, le=1, description="1 if address mismatch, else 0.")
    pep_flag: int = Field(..., ge=0, le=1, description="1 if politically exposed person, else 0.")
    velocity_signups_same_device: int = Field(..., ge=1, description="Signups from device.")


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    """Application factory initializing API routes and preloading risk model artifacts.

    Args:
        artifacts_dir: Optional custom artifacts directory.

    Returns:
        Configured FastAPI application instance.
    """
    if artifacts_dir is None:
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")

    artifacts_path = Path(artifacts_dir)
    model_checkpoint = artifacts_path / "model.pt"
    scaler_file = artifacts_path / "scaler.json"

    # Train quickly if artifacts are missing
    if not model_checkpoint.exists() or not scaler_file.exists():
        train_pipeline(n=1500, epochs=2, out_dir=str(artifacts_path))

    artifacts = load(artifacts_path)

    app = FastAPI(
        title="Multimodal KYC Risk API",
        version="1.0.0",
        description=(
            "High-throughput multimodal KYC risk assessment and document tampering scoring."
        ),
    )

    # Store loaded artifacts in application state
    app.state.artifacts = artifacts
    app.state.artifacts_dir = str(artifacts_path)
    app.state.rng = np.random.default_rng(1234)

    @app.get("/health", summary="Health check")
    def health() -> dict[str, Any]:
        """Return operational status and model readiness."""
        return {
            "status": "ok",
            "model_loaded": True,
            "version": "1.0.0",
        }

    @app.get("/sample/{kind}", summary="Get synthetic document sample")
    def get_sample(kind: str) -> Response:
        """Generate and return a synthetic ID document PNG (genuine or tampered)."""
        kind_lower = kind.lower().strip()
        if kind_lower not in ("genuine", "tampered"):
            raise HTTPException(
                status_code=400,
                detail="Invalid sample kind. Choose either 'genuine' or 'tampered'.",
            )

        is_tampered = kind_lower == "tampered"
        doc_arr = render_document(app.state.rng, tampered=is_tampered, size=96)
        img = Image.fromarray(doc_arr, mode="L")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/score", summary="Score applicant risk using document image and tabular data")
    async def score_applicant(
        file: UploadFile = File(..., description="Document image file (PNG/JPEG)."),
        applicant: str = Form(..., description="JSON-serialized ApplicantData object."),
    ) -> dict[str, Any]:
        """Score multimodal KYC application using uploaded document image and applicant data."""
        # 1. Parse and validate JSON form field
        try:
            applicant_json = json.loads(applicant)
        except Exception as err:
            raise HTTPException(
                status_code=422,
                detail=f"Applicant field is not valid JSON: {err}",
            ) from err

        try:
            applicant_obj = ApplicantData.model_validate(applicant_json)
        except ValidationError as val_err:
            raise HTTPException(
                status_code=422,
                detail=val_err.errors(),
            ) from val_err

        # 2. Read and parse image contents
        try:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("L")
            pil_image = pil_image.resize((96, 96), resample=Image.Resampling.BILINEAR)
        except Exception as img_err:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to decode uploaded image: {img_err}",
            ) from img_err

        # 3. Predict multimodal risk
        applicant_dict = applicant_obj.model_dump()
        result = predict(
            image=pil_image,
            tabular=applicant_dict,
            artifacts=app.state.artifacts,
        )

        return result

    return app


app = create_app()
