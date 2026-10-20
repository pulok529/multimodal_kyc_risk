"""Synthetic document image and tabular applicant data generator for KYC risk scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def render_document(
    rng: np.random.Generator | None = None, tampered: bool = False, size: int = 96
) -> np.ndarray:
    """Render a synthetic grayscale ID card document with PIL.

    Args:
        rng: Optional NumPy random generator instance. If None, default_rng is used.
        tampered: Whether to introduce subtle forgery artefacts.
        size: Width and height of the rendered square image.

    Returns:
        Grayscale image as a 2D uint8 NumPy array of shape (size, size).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Base canvas with light card background
    bg_color = int(rng.integers(220, 236))
    img = Image.new("L", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 1. Guilloche-like sine pattern background
    for offset in range(10, size - 8, 12):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        freq = float(rng.uniform(0.08, 0.16))
        amp = float(rng.uniform(1.8, 3.5))
        x_pts = np.linspace(4, size - 4, num=16)
        points = [
            (float(x), float(np.clip(offset + amp * np.sin(freq * x + phase), 3, size - 4)))
            for x in x_pts
        ]
        draw.line(points, fill=int(rng.integers(196, 210)), width=1)

    # 2. Outer card border
    draw.rectangle([3, 3, size - 4, size - 4], outline=80, width=1)
    draw.rectangle([5, 5, size - 6, size - 6], outline=170, width=1)

    # 3. Header bar / emblem placeholder
    header_len = int(rng.integers(40, size - 16))
    draw.rectangle([8, 8, 8 + header_len, 11], fill=110)

    # 4. Photo box (portrait silhouette)
    photo_box = [8, 16, 38, 56]
    draw.rectangle(photo_box, fill=int(rng.integers(195, 210)), outline=95, width=1)
    # Head and torso silhouette
    draw.ellipse([18, 22, 28, 32], fill=135)
    draw.rectangle([13, 36, 33, 55], fill=135)

    # 5. 4 to 6 text-like horizontal bars
    num_bars = int(rng.integers(4, 7))
    bar_positions = [17, 24, 31, 38, 45, 52][:num_bars]
    for y_bar in bar_positions:
        bar_len = int(rng.integers(22, 44))
        draw.rectangle([43, y_bar, min(size - 8, 43 + bar_len), y_bar + 3], fill=85)

    # 6. MRZ-like stripe of small blocks at the bottom (2 rows)
    for y_mrz in (66, 76):
        for x_mrz in range(8, size - 8, 4):
            if rng.uniform() > 0.18:
                draw.rectangle([x_mrz, y_mrz, x_mrz + 2, y_mrz + 4], fill=70)

    # Tampering artefacts (subtle, chosen at random)
    if tampered:
        artefacts = [
            "pasted_rectangle",
            "misaligned_mrz",
            "blurred_region",
            "copy_moved_text",
        ]
        num_artefacts = int(rng.integers(1, 3))
        chosen = rng.choice(artefacts, size=num_artefacts, replace=False)

        for art in chosen:
            if art == "pasted_rectangle":
                # Subtle altered patch over part of the photo box
                patch_fill = int(205 + rng.integers(-15, 16))
                draw.rectangle([12, 20, 32, 44], fill=patch_fill, outline=90, width=2)
            elif art == "misaligned_mrz":
                # Break and vertically offset a segment of the MRZ stripe
                draw.rectangle([20, 64, 52, 82], fill=bg_color)
                for x_mrz in range(20, 52, 4):
                    draw.rectangle([x_mrz + 1, 62, x_mrz + 3, 66], fill=70)
                    draw.rectangle([x_mrz + 1, 72, x_mrz + 3, 76], fill=70)
            elif art == "blurred_region":
                # Apply localized gaussian blur over a card section
                blur_box = (10, 18, 36, 54)
                patch = img.crop(blur_box).filter(ImageFilter.GaussianBlur(radius=2.6))
                img.paste(patch, blur_box)
            elif art == "copy_moved_text":
                # Copy a text bar and paste duplicate with slight displacement
                text_crop = img.crop((43, 23, 76, 28))
                img.paste(text_crop, (45, 34))

    # Apply sensor noise, small random rotation, and brightness variations to ALL images
    rot_angle = float(rng.uniform(-2.0, 2.0))
    img = img.rotate(rot_angle, resample=Image.Resampling.BILINEAR, fillcolor=bg_color)

    arr = np.asarray(img, dtype=np.float32)
    contrast = float(rng.uniform(0.94, 1.06))
    brightness = float(rng.uniform(-8.0, 8.0))
    sensor_noise = rng.normal(0.0, 3.5, size=arr.shape)

    arr = arr * contrast + brightness + sensor_noise
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def generate(
    n: int = 6000, seed: int = 3
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Generate synthetic multimodal KYC dataset.

    Args:
        n: Total number of applicant records to generate.
        seed: Random seed for deterministic generation.

    Returns:
        images: Float32 array of shape (n, 1, 96, 96) in [0.0, 1.0].
        tabular: DataFrame with 9 applicant risk features.
        labels: Binary float32 array of shape (n,) where 1 is high risk.
    """
    rng = np.random.default_rng(seed)

    # 1. Document tampering status (~14% altered documents)
    tampered_flags = rng.binomial(1, 0.14, size=n).astype(bool)

    # 2. Render document images
    images = np.empty((n, 1, 96, 96), dtype=np.float32)
    for i in range(n):
        doc_arr = render_document(rng, tampered=tampered_flags[i], size=96)
        images[i, 0] = doc_arr.astype(np.float32) / 255.0

    # 3. Tabular applicant features
    age = rng.integers(18, 76, size=n)
    account_age_days = rng.integers(1, 1200, size=n)
    country_risk = np.round(rng.beta(1.5, 4.0, size=n), 3)
    device_risk = np.round(rng.beta(1.5, 3.5, size=n), 3)
    num_failed_logins_30d = np.clip(rng.poisson(0.7, size=n), 0, 10)
    selfie_match_score = np.round(rng.beta(8.0, 2.0, size=n), 3)
    address_mismatch = rng.binomial(1, 0.12, size=n)
    pep_flag = rng.binomial(1, 0.025, size=n)
    velocity_signups_same_device = np.clip(rng.poisson(1.2, size=n), 1, 10)

    tabular_df = pd.DataFrame(
        {
            "age": age,
            "account_age_days": account_age_days,
            "country_risk": country_risk,
            "device_risk": device_risk,
            "num_failed_logins_30d": num_failed_logins_30d,
            "selfie_match_score": selfie_match_score,
            "address_mismatch": address_mismatch,
            "pep_flag": pep_flag,
            "velocity_signups_same_device": velocity_signups_same_device,
        }
    )

    # 4. Multimodal risk label generation
    # Both modalities contribute strongly to fraud / high-risk label (~12% positive rate)
    z_tab = (
        1.6 * device_risk
        + 1.5 * (1.0 - selfie_match_score)
        + 1.0 * country_risk
        + 1.2 * address_mismatch
        + 1.8 * pep_flag
        + 0.35 * num_failed_logins_30d
        + 0.25 * (velocity_signups_same_device - 1)
        - 0.0008 * account_age_days
    )
    z_tab_norm = (z_tab - np.mean(z_tab)) / np.std(z_tab)

    tampered_numeric = tampered_flags.astype(float)
    noise = rng.normal(0.0, 0.45, size=n)
    logit = (
        -2.85
        + 2.6 * tampered_numeric
        + 1.5 * z_tab_norm
        + 0.8 * (tampered_numeric * z_tab_norm)
        + noise
    )
    risk_probs = 1.0 / (1.0 + np.exp(-logit))
    labels = (risk_probs >= 0.5).astype(np.float32)

    return images, tabular_df, labels
