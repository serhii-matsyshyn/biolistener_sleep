"""
ObsState shape + parity tests.
"""
from __future__ import annotations

import math

import numpy as np

from app.agent.env import EnvConfig, SleepOnsetEnv
from app.agent.obs_state import ObsState


def test_obs_state_first_update_seeds_ema_with_raw_value():
    """First sample should be passed through verbatim (no smoothing yet),
    otherwise the policy sees a multi-tick warmup of zeros which doesn't
    match training-env initial conditions."""
    s = ObsState()
    s.update(raw_theta_alpha=0.42, raw_plv=0.17)
    assert s.smoothed_theta_alpha == 0.42
    assert s.smoothed_plv == 0.17
    # Δ is zero on the first tick (only one value in the ring buffer)
    assert s.delta_theta_alpha == 0.0


def test_obs_state_ema_decay_matches_2s_tau_at_4hz_tick():
    """τ=2s, tick_hz=4Hz -> α_decay ≈ 0.118. After a step input the EMA
    should converge to ~63% of the new value in 2 s (8 ticks)."""
    s = ObsState(tick_hz=4.0, ema_tau_s=2.0)
    s.update(0.0, 0.0)
    # Drive a step input to 1.0 for 8 ticks (2 s)
    for _ in range(8):
        s.update(1.0, 0.0)
    # Theoretical EMA response to a unit step after n ticks (initial value
    # 0.0): 1 - (1 - α)^n. With α = 1-exp(-1/8) ≈ 0.1175 and n=8: ≈ 0.633.
    expected = 1.0 - (1.0 - s.ema_alpha_decay) ** 8
    assert math.isclose(s.smoothed_theta_alpha, expected, rel_tol=1e-3)
    assert 0.55 < s.smoothed_theta_alpha < 0.75


def test_obs_state_delta_window_is_16_ticks_at_4hz():
    """Δ window = 4 s × 4 Hz = 16 ticks. Δ should be 0 until the buffer
    is full, then start reporting the smoothed change over that horizon."""
    s = ObsState(tick_hz=4.0, delta_window_s=4.0)
    assert s.delta_window_ticks == 16
    # Constant input: Δ should stay ~0 throughout (smoothed value is also
    # constant after the warmup tick).
    for _ in range(20):
        s.update(0.5, 0.5)
    assert abs(s.delta_theta_alpha) < 1e-9


def test_obs_state_delta_signals_sustained_rise():
    """If we drive θ/α from 0 to 1, the smoothed EMA rises and the Δ
    should be strictly positive after a few ticks."""
    s = ObsState(tick_hz=4.0)
    s.update(0.0, 0.0)
    for _ in range(20):
        s.update(1.0, 0.0)
    # After 20 ticks at α≈0.118 starting from 0, smoothed value sits around
    # 0.92; the ring buffer's 0-position holds the value from 16 ticks ago
    # (~0.46), giving Δ ≈ 0.45 - strictly positive and well above noise.
    assert s.delta_theta_alpha > 0.3, (
        f"sustained rise should produce positive Δ but got {s.delta_theta_alpha}"
    )


def test_env_obs_shape_is_15_when_restfulness_off():
    """The deployed cohort was trained with include_restfulness=False.
    The corresponding obs vector is 9 features + plv + 3 last-action + 2
    temporal context = 15 dims. The runtime daemon MUST produce the same
    shape or SAC.predict raises."""
    env = SleepOnsetEnv(EnvConfig(include_restfulness=False, episode_s=2.0))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,), f"got {obs.shape}"
    assert env.observation_space.shape == (15,)


def test_env_obs_shape_is_16_when_restfulness_on():
    """Sanity: with restfulness enabled, obs gains 1 dim -> 16. Currently
    not used in cohort but reserved if we re-add it later."""
    # include_restfulness=True instantiates RestfulnessMetric which may
    # fail to prepare on some boxes; skip if the metric setup raises.
    try:
        env = SleepOnsetEnv(EnvConfig(include_restfulness=True, episode_s=2.0))
    except Exception:
        import pytest
        pytest.skip("RestfulnessMetric unavailable on this host")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,), f"got {obs.shape}"


def test_env_obs_finite_after_first_tick():
    """No NaN/Inf in the obs vector - would silently nuke SAC.predict
    with a torch shape error rather than an obvious failure."""
    env = SleepOnsetEnv(EnvConfig(include_restfulness=False, episode_s=2.0))
    obs, _ = env.reset(seed=0)
    obs2, _, _, _, _ = env.step(np.zeros(env.action_space.shape,
                                            dtype=np.float32))
    assert np.all(np.isfinite(obs2)), (
        f"non-finite values in obs: {obs2}"
    )
