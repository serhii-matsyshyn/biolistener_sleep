"""
Feature extraction tests - synthetic 10 Hz sine should yield α-dominant
band power; θ/α ratio should be small.
"""
from __future__ import annotations

import numpy as np

from app.eeg.features import (
    BAND_ALPHA, BAND_THETA, FEATURE_NAMES,
    extract_features, feature_vector,
)
from app.eeg.montage import OCCIPITAL_ALPHA


def _make_alpha_signal(n_channels: int = 8, sr: int = 500,
                       duration_s: float = 4.0, alpha_hz: float = 10.0,
                       seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    data = np.zeros((n_channels, n), dtype=np.float64)
    for ch in range(n_channels):
        data[ch] = (20.0 * np.sin(2 * np.pi * alpha_hz * t)
                    + 3.0 * rng.standard_normal(n))
    return data


def test_alpha_dominant_signal_has_high_alpha_power():
    data = _make_alpha_signal()
    snap = extract_features(data, sampling_rate=500, montage=OCCIPITAL_ALPHA)
    assert snap.alpha_pwr > snap.theta_pwr * 3, \
        f"α not dominant: α={snap.alpha_pwr:.4f}, θ={snap.theta_pwr:.4f}"
    assert snap.theta_alpha_ratio < 0.5, \
        f"θ/α ratio too high for α-dominant signal: {snap.theta_alpha_ratio}"


def test_theta_dominant_signal_has_high_theta_alpha_ratio():
    """Inject 6 Hz theta - θ/α ratio should INCREASE (drowsiness marker)."""
    sr = 500
    duration = 4.0
    n = int(sr * duration)
    t = np.arange(n) / sr
    rng = np.random.default_rng(13)
    data = np.zeros((8, n), dtype=np.float64)
    for ch in range(8):
        data[ch] = (15.0 * np.sin(2 * np.pi * 6.0 * t)
                    + 3.0 * rng.standard_normal(n))
    snap = extract_features(data, sampling_rate=sr, montage=OCCIPITAL_ALPHA)
    assert snap.theta_pwr > snap.alpha_pwr * 2, \
        f"θ not dominant: θ={snap.theta_pwr:.4f}, α={snap.alpha_pwr:.4f}"
    assert snap.theta_alpha_ratio > 1.5


def test_feature_vector_length_stable():
    data = _make_alpha_signal()
    snap = extract_features(data, sampling_rate=500, montage=OCCIPITAL_ALPHA)
    vec = feature_vector(snap)
    assert len(vec) == len(FEATURE_NAMES) == 9
    assert vec.dtype == np.float32


def test_short_window_returns_zero_features():
    sr = 500
    # Only 100 samples (0.2s) - too short for PSD
    data = np.zeros((8, 100), dtype=np.float64)
    snap = extract_features(data, sampling_rate=sr, montage=OCCIPITAL_ALPHA)
    assert snap.alpha_pwr == 0.0
    assert snap.theta_pwr == 0.0
