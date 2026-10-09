import cv2
import numpy as np
import pytest

from yeast_vacuole_pipeline.features_extraction.extractors.geometric_features import (
    ContourNotFoundError,
    extract_geometric_features,
)


def disk(radius=20, size=80):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, 255, -1)
    return img


def test_disk_is_nearly_circular():
    f = extract_geometric_features(disk())
    # Pixelated contours overestimate the perimeter, so circularity lands below 1.
    assert 0.8 < f["circularity"] <= 1.05
    assert f["eccentricity"] < 0.2
    assert f["solidity"] == pytest.approx(1.0, abs=0.05)
    assert f["total_area"] == pytest.approx(np.pi * 20**2, rel=0.05)


def test_ellipse_is_more_eccentric_than_disk():
    img = np.zeros((80, 80), dtype=np.uint8)
    cv2.ellipse(img, (40, 40), (30, 10), 0, 0, 360, 255, -1)
    assert extract_geometric_features(img)["eccentricity"] > 0.8


def test_empty_image_raises_or_returns_no_area():
    empty = np.zeros((80, 80), dtype=np.uint8)
    try:
        f = extract_geometric_features(empty)
    except ContourNotFoundError:
        return
    assert f.get("total_area", 0) == 0
