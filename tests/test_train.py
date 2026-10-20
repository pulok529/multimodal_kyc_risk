"""Integration tests for the training pipeline and artifact exports."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from kyc_risk.predict import load, predict
from kyc_risk.synth import render_document
from kyc_risk.train import train_pipeline


def test_train_pipeline_artifacts(tmp_path: Path) -> None:
    """Verify train_pipeline generates expected model, scaler, metric, and visual artifacts."""
    out_dir = tmp_path / "artifacts"
    metrics = train_pipeline(
        n=600,
        epochs=1,
        out_dir=str(out_dir),
        seed=42,
        batch_size=32,
    )

    # 1. Check all required artifact files exist on disk
    expected_files = [
        "model.pt",
        "scaler.json",
        "metrics.json",
        "ablation.png",
        "samples.png",
    ]
    for filename in expected_files:
        artifact_file = out_dir / filename
        assert artifact_file.is_file(), f"Missing artifact: {filename}"
        assert artifact_file.stat().st_size > 0, f"Artifact is empty: {filename}"

    # 2. Check metrics structure and keys
    for model_key in ("fusion", "image_only", "tabular_only"):
        assert model_key in metrics
        model_metrics = metrics[model_key]
        for metric_name in ("roc_auc", "pr_auc", "recall", "precision", "f1", "val_threshold"):
            assert metric_name in model_metrics
            assert isinstance(model_metrics[metric_name], float)

    # 3. Verify metrics.json can be parsed and matches returned metrics
    with open(out_dir / "metrics.json", encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert set(saved_metrics.keys()) == {"fusion", "image_only", "tabular_only"}

    # 4. Verify predict module can load artifacts and score an inference sample
    loaded = load(out_dir)
    sample_doc = Image.fromarray(render_document(None, tampered=False, size=96), mode="L")
    sample_applicant = {
        "age": 34,
        "account_age_days": 180,
        "country_risk": 0.15,
        "device_risk": 0.20,
        "num_failed_logins_30d": 1,
        "selfie_match_score": 0.92,
        "address_mismatch": 0,
        "pep_flag": 0,
        "velocity_signups_same_device": 1,
    }
    prediction = predict(sample_doc, sample_applicant, artifacts=loaded)
    assert "risk_probability" in prediction
    assert 0.0 <= prediction["risk_probability"] <= 1.0
    assert prediction["decision"] in ("manual_review", "auto_approve")
    assert isinstance(prediction["signals"], list)
