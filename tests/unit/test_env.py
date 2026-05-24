"""
env smoke tests - reset, step, action/observation shapes.
"""
from __future__ import annotations

import numpy as np

from app.agent.env import EnvConfig, SleepOnsetEnv
from app.agent.policies import FixedFrequencyPolicy, LinearRampDownPolicy
from app.brain_sim.domain_random import RandomizationRanges


def test_env_reset_returns_well_shaped_obs():
    env = SleepOnsetEnv(EnvConfig(episode_s=2.0, include_restfulness=False))
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert "subject" in info
    env.close()


def test_env_step_runs_and_produces_reward():
    env = SleepOnsetEnv(EnvConfig(episode_s=2.0, include_restfulness=False))
    env.reset(seed=42)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert "features" in info and "stim" in info
    env.close()


def test_env_terminates_via_panic():
    env = SleepOnsetEnv(EnvConfig(episode_s=10.0, include_restfulness=False))
    env.reset(seed=0)
    env._supervisor.panic("test")
    obs, reward, term, trunc, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert term is True, "Episode should terminate when panic is latched"
    # Stim must be off
    assert info["stim"].enabled is False
    env.close()


def test_fixed_freq_policy_runs_short_episode():
    env = SleepOnsetEnv(EnvConfig(episode_s=2.0, include_restfulness=False))
    obs, _ = env.reset(seed=99)
    policy = FixedFrequencyPolicy(frequency_hz=10.0, fill=0.20, brightness=0.30)
    total_reward = 0.0
    truncated = False
    while not truncated:
        action, _ = policy.predict(obs)
        obs, reward, term, truncated, info = env.step(action)
        total_reward += reward
        if term:
            break
    assert isinstance(total_reward, float)
    env.close()


def test_linear_ramp_policy_emits_decreasing_frequency():
    """LinearRampDownPolicy starts ~start_hz and ends ~end_hz."""
    policy = LinearRampDownPolicy(
        start_hz=11.0, end_hz=4.0, duration_s=0.4,
    )
    policy.reset()
    obs = np.zeros(20, dtype=np.float32)  # observation shape doesn't matter
    # Right at start
    a_start, _ = policy.predict(obs)
    # The norm-encoded action[0] should be close to encoding 11 Hz
    from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ
    decoded_start = FREQ_MIN_HZ + (a_start[0] + 1.0) / 2.0 \
                    * (FREQ_MAX_HZ - FREQ_MIN_HZ)
    # Sleep enough to be near end
    import time as _time
    _time.sleep(0.45)
    a_end, _ = policy.predict(obs)
    decoded_end = FREQ_MIN_HZ + (a_end[0] + 1.0) / 2.0 \
                  * (FREQ_MAX_HZ - FREQ_MIN_HZ)
    assert decoded_start > decoded_end
    assert decoded_start > 10.0  # near 11 at start
    assert decoded_end < 6.0     # near 4 at end
