"""
Pin the deterministic audio-shaper curves.

The shaper owns two knobs the agent can't learn (warmth + master vol);
its mapping is the *only* thing that taper the music as the subject
descends to N1. A silent regression here would mean the demo session
sounds the same volume + timbre at minute 5 as at minute 0.
"""
from __future__ import annotations

import math

from app.audio.shaper import AudioShaper


def test_drowsiness_saturates_below_baseline_and_above_full():
    s = AudioShaper()
    assert s.drowsiness(0.0) == 0.0
    assert s.drowsiness(s.theta_alpha_baseline) == 0.0
    assert s.drowsiness(s.theta_alpha_full) == 1.0
    assert s.drowsiness(100.0) == 1.0


def test_drowsiness_linear_in_window():
    s = AudioShaper(theta_alpha_baseline=0.5, theta_alpha_full=3.0)
    mid = (0.5 + 3.0) / 2.0   # 1.75
    assert math.isclose(s.drowsiness(mid), 0.5, abs_tol=1e-9)


def test_warmth_runs_from_baseline_to_full_endpoints():
    """Warmth = bright at alert, dark at drowsy. Bounds must match config."""
    s = AudioShaper(warmth_at_baseline_pct=80.0, warmth_at_full_pct=25.0)
    # Alert subject (θ/α at baseline): bright music
    assert math.isclose(s.warmth_pct(s.theta_alpha_baseline), 80.0)
    # Fully drowsy: dark music
    assert math.isclose(s.warmth_pct(s.theta_alpha_full), 25.0)
    # Mid-point: linear interp
    mid = (s.theta_alpha_baseline + s.theta_alpha_full) / 2.0
    assert math.isclose(s.warmth_pct(mid), (80.0 + 25.0) / 2.0)


def test_master_vol_at_start_alert_equals_initial():
    """t=0, fully alert: shaper outputs the initial volume unmodified.
    Any other answer means the agent's commanded volume is being shadowed
    even when there's no drowsiness signal yet."""
    s = AudioShaper(initial_master_vol_pct=50.0)
    v = s.master_vol_pct(theta_alpha_smoothed=0.0,
                          time_into_session_s=0.0,
                          session_duration_s=300.0)
    assert math.isclose(v, 50.0)


def test_master_vol_drops_with_drowsiness_in_early_session():
    """Fully drowsy in the first 25% of the session: only the drowsy
    taper applies; no end-fade yet."""
    s = AudioShaper(initial_master_vol_pct=50.0, drowsy_vol_taper=0.4)
    v = s.master_vol_pct(theta_alpha_smoothed=10.0,    # well above 'full'
                          time_into_session_s=30.0,    # 10% into session
                          session_duration_s=300.0)
    # 50 * (1 - 0.4) = 30
    assert math.isclose(v, 30.0)


def test_master_vol_end_fade_at_session_end():
    """At the very last tick (t=session_duration), volume should be at
    `end_fade_min_factor` × pre-fade value. With initial=50 and no
    drowsiness, pre-fade=50, end-fade=0.4 -> final=20."""
    s = AudioShaper(initial_master_vol_pct=50.0,
                     end_fade_starts_at_frac=0.75,
                     end_fade_min_factor=0.4)
    v = s.master_vol_pct(theta_alpha_smoothed=0.0,
                          time_into_session_s=300.0,
                          session_duration_s=300.0)
    assert math.isclose(v, 20.0, abs_tol=1e-6)


def test_master_vol_combined_drowsy_and_end_fade():
    """Final tick + fully drowsy: BOTH tapers compound. With defaults this
    is 50 × (1 - 0.4) × 0.4 = 12."""
    s = AudioShaper()
    v = s.master_vol_pct(theta_alpha_smoothed=10.0,
                          time_into_session_s=300.0,
                          session_duration_s=300.0)
    expected = 50.0 * (1 - 0.4) * 0.4
    assert math.isclose(v, expected, abs_tol=1e-6)


def test_master_vol_monotonic_in_time_when_drowsiness_fixed():
    """For any constant drowsiness level, master volume must be
    monotonically non-increasing as the session progresses (it can only
    flatten or fade — never rise). A failure here would mean the music
    gets *louder* near the end which is the opposite of what we want."""
    s = AudioShaper()
    prev = float("inf")
    for t_frac in [0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0]:
        v = s.master_vol_pct(theta_alpha_smoothed=2.0,
                              time_into_session_s=t_frac * 300.0,
                              session_duration_s=300.0)
        assert v <= prev + 1e-9, f"non-monotonic at t_frac={t_frac}: {prev} -> {v}"
        prev = v


def test_master_vol_handles_zero_duration():
    """Edge case: if session_duration_s is 0 (misconfig), the shaper
    must not divide by zero — just skip the end-fade."""
    s = AudioShaper(initial_master_vol_pct=50.0)
    v = s.master_vol_pct(theta_alpha_smoothed=0.0,
                          time_into_session_s=10.0,
                          session_duration_s=0.0)
    assert math.isclose(v, 50.0)
