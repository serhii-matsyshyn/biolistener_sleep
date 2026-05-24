"""
Smoke tests for the SubprocVecEnv training path.

Goal: catch regressions where someone changes ``make_vec_env`` /
``_env_factory`` / ``EnvConfig`` and accidentally breaks vec training.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.agent.train import TrainConfig, _env_factory, make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def _shrink_cfg(n_envs: int) -> TrainConfig:
    """Tiny config so a smoke run is sub-second."""
    return TrainConfig(
        n_envs=n_envs, total_steps=10, episode_s=2.0,
        include_restfulness=False,
    )


def test_n_envs_1_returns_dummy_vec_env():
    env = make_vec_env(_shrink_cfg(1))
    try:
        assert isinstance(env, DummyVecEnv)
        assert env.num_envs == 1
    finally:
        env.close()


def test_n_envs_gt_1_returns_subproc_vec_env():
    env = make_vec_env(_shrink_cfg(4))
    try:
        assert isinstance(env, SubprocVecEnv)
        assert env.num_envs == 4
    finally:
        env.close()


def test_vec_env_reset_and_step_shapes():
    """Reset returns a batched obs; step returns the 4-tuple SB3 expects."""
    env = make_vec_env(_shrink_cfg(2))
    try:
        obs = env.reset()
        assert obs.shape == (2, env.observation_space.shape[0])
        actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, env.observation_space.shape[0])
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()


def test_env_factory_per_rank_seeds_diverge():
    """Each rank's factory seeds the env with cfg.seed+rank so subprocesses
    don't all sample the same NMM subject. Verify the seeds applied differ."""
    cfg = _shrink_cfg(1)
    cfg.seed = 100
    f0 = _env_factory(cfg, rank=0)
    f1 = _env_factory(cfg, rank=1)
    e0, e1 = f0(), f1()
    try:
        # The factory calls env.reset(seed=base+rank); the env stores its
        # subject under ``_subject``. Different seeds -> different subject.
        assert e0.env._subject is not None
        assert e1.env._subject is not None
        # At minimum the seeds should differ; subject params *usually* differ
        # too but that depends on RandomizationRanges granularity.
        assert e0.env._subject.seed != e1.env._subject.seed
    finally:
        e0.close(); e1.close()


def test_vec_env_clamps_to_cpu_count(monkeypatch):
    """Asking for more envs than CPUs gets clamped silently."""
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    cfg = _shrink_cfg(8)
    env = make_vec_env(cfg)
    try:
        assert env.num_envs == 2
    finally:
        env.close()
