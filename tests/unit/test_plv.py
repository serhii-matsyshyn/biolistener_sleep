"""
PLV tests - identical signals -> PLV=1; orthogonal/random signals -> PLV~0.
"""
from __future__ import annotations

import numpy as np

from app.eeg.plv import plv, plv_strobe_to_eeg


def test_plv_of_identical_signals_is_one():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1000)
    assert abs(plv(sig, sig) - 1.0) < 1e-6


def test_plv_of_independent_random_signals_is_low():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    assert plv(a, b) < 0.15


def test_plv_strobe_to_eeg_zero_when_strobe_disabled():
    sr = 500
    t = np.arange(2 * sr) / sr
    eeg = np.sin(2 * np.pi * 10 * t)
    env = np.zeros_like(eeg)
    assert plv_strobe_to_eeg(eeg, env, sr, 10.0) == 0.0


def test_plv_strobe_to_eeg_high_when_eeg_follows_strobe():
    """Construct EEG that contains the strobe signal - PLV should be high."""
    sr = 500
    dt = 1.0 / sr
    duration = 4.0
    n = int(sr * duration)
    t = np.arange(n) * dt
    strobe_hz = 8.0

    # Strobe envelope: square wave 20% duty
    env = (np.mod(t * strobe_hz, 1.0) < 0.20).astype(np.float64)
    # EEG that follows the strobe (sine at same freq + small noise)
    rng = np.random.default_rng(42)
    eeg = np.sin(2 * np.pi * strobe_hz * t) + 0.1 * rng.standard_normal(n)
    plv_val = plv_strobe_to_eeg(eeg, env, sr, strobe_hz, bandwidth_hz=2.0)
    assert plv_val > 0.7, f"PLV {plv_val:.3f} too low for following signal"
