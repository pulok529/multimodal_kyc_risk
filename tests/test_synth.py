"""Unit tests for synthetic document generation and tabular applicant data synthesis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kyc_risk.synth import generate, render_document

EXPECTED_COLUMNS = [
    "age",
    "account_age_days",
    "country_risk",
    "device_risk",
    "num_failed_logins_30d",
    "selfie_match_score",
    "address_mismatch",
    "pep_flag",
    "velocity_signups_same_device",
]


def test_render_document_shapes_and_types() -> None:
    """Test individual synthetic document rendering shapes and uint8 output range."""
    rng = np.random.default_rng(101)
    genuine_doc = render_document(rng, tampered=False, size=96)
    assert isinstance(genuine_doc, np.ndarray)
    assert genuine_doc.shape == (96, 96)
    assert genuine_doc.dtype == np.uint8

    tampered_doc = render_document(rng, tampered=True, size=96)
    assert isinstance(tampered_doc, np.ndarray)
    assert tampered_doc.shape == (96, 96)
    assert tampered_doc.dtype == np.uint8


def test_tampered_differs_from_genuine() -> None:
    """Test that tampered documents exhibit measurable differences from genuine renders."""
    # Compare mean pixel differences over a batch of renders
    rng_gen = np.random.default_rng(42)
    rng_tam = np.random.default_rng(42)

    # For the same seed sequence, tampered logic branches and alters document regions
    gen = render_document(rng_gen, tampered=False, size=96)
    tam = render_document(rng_tam, tampered=True, size=96)

    diff = np.abs(gen.astype(float) - tam.astype(float))
    assert np.max(diff) > 0.0
    assert np.mean(diff) > 0.5


def test_generate_shapes_and_columns() -> None:
    """Test batch data generator output shapes, types, and expected tabular columns."""
    n_samples = 150
    images, tabular, labels = generate(n=n_samples, seed=12)

    assert isinstance(images, np.ndarray)
    assert images.shape == (n_samples, 1, 96, 96)
    assert images.dtype == np.float32

    assert isinstance(tabular, pd.DataFrame)
    assert len(tabular) == n_samples
    assert list(tabular.columns) == EXPECTED_COLUMNS

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (n_samples,)
    assert labels.dtype == np.float32


def test_generate_value_ranges() -> None:
    """Test range boundaries for image pixels, tabular attributes, and labels."""
    images, tabular, labels = generate(n=300, seed=77)

    # Image normalization range [0, 1]
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0

    # Labels must be strictly binary {0, 1}
    unique_labels = set(np.unique(labels))
    assert unique_labels.issubset({0.0, 1.0})

    # Tabular attribute valid domains
    assert (tabular["age"] >= 18).all() and (tabular["age"] <= 120).all()
    assert (tabular["account_age_days"] >= 0).all()
    assert (tabular["country_risk"] >= 0.0).all() and (tabular["country_risk"] <= 1.0).all()
    assert (tabular["device_risk"] >= 0.0).all() and (tabular["device_risk"] <= 1.0).all()
    assert (tabular["selfie_match_score"] >= 0.0).all() and (
        tabular["selfie_match_score"] <= 1.0
    ).all()
    assert set(tabular["address_mismatch"].unique()).issubset({0, 1})
    assert set(tabular["pep_flag"].unique()).issubset({0, 1})
    assert (tabular["num_failed_logins_30d"] >= 0).all()
    assert (tabular["velocity_signups_same_device"] >= 1).all()


def test_label_rate_range() -> None:
    """Test that synthetic dataset produces fraud rate in target bounds [0.07, 0.18]."""
    _, _, labels = generate(n=2000, seed=3)
    rate = float(np.mean(labels))
    assert 0.07 <= rate <= 0.18, f"Label rate {rate:.4f} outside target bounds [0.07, 0.18]"
