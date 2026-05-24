"""
SAC training loop for the sleep-onset RL agent.

Trains Stable-Baselines3 SAC on ``SleepOnsetEnv`` with per-episode
domain randomization. MLflow logs hyperparams, per-episode rewards, and
the final policy artifact (.zip).

Baseline comparison: same env, same number of episodes, evaluating
:class:`FixedFrequencyPolicy` and :class:`LinearRampDownPolicy` for
reference reward curves logged to the same MLflow experiment.

Usage::

    biolistener-sleep train --config configs/<profile>.yaml
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from app.agent.env import EnvConfig, SleepOnsetEnv
from app.agent.policies import FixedFrequencyPolicy, LinearRampDownPolicy
from app.agent.reward import RewardWeights
from app.agent.reward_variants import resolve as resolve_reward_variant
from app.brain_sim.domain_random import RandomizationRanges

logger = logging.getLogger("agent.train")


@dataclass
class TrainConfig:
    total_steps: int = 20_000
    episode_s: float = 30.0
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"
    seed: int = 42
    include_restfulness: bool = True
    checkpoint_dir: str = "app/agent/checkpoints"
    experiment_name: str = "sleep_onset_sac"
    # Reward shaping. Either:
    #   - ``reward_variant``: name from REWARD_VARIANTS (e.g. "full_shaped");
    #     plugged in by ``_runner.train_from_config``. None -> use default.
    #   - ``reward_weights``: explicit RewardWeights instance. Wins over
    #     ``reward_variant`` if both set.
    # Logged to MLflow for ablation comparison.
    reward_variant: Optional[str] = None
    reward_weights: Optional["RewardWeights"] = None
    # Number of parallel envs collecting transitions during training.
    # 1 -> single-process (DummyVecEnv, no fork). >1 -> SubprocVecEnv with
    # one Python process per env on its own CPU core. The NMM brain sim is
    # pure numpy and fork-safe, so this is a clean speedup for collection
    # without touching SAC's gradient code. Clamped to os.cpu_count() at
    # runtime so we never over-allocate. Post-training eval (baselines +
    # SAC eval) always runs on a single fresh env regardless of this.
    n_envs: int = 1


class MLflowCallback(BaseCallback):
    """Logs episode rewards & lengths to MLflow."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_count += 1
                mlflow.log_metric("episode_reward", float(ep_info["r"]),
                                  step=self.episode_count)
                mlflow.log_metric("episode_length", float(ep_info["l"]),
                                  step=self.episode_count)
        return True


def _resolve_reward_weights(cfg: TrainConfig) -> Optional[RewardWeights]:
    # Explicit weights win, then named variant, then EnvConfig's own
    # default (None -> SleepOnsetEnv builds RewardWeights()).
    if cfg.reward_weights is not None:
        return cfg.reward_weights
    if cfg.reward_variant is not None:
        return resolve_reward_variant(cfg.reward_variant)
    return None


def make_env(cfg: TrainConfig) -> Monitor:
    """Single Monitor-wrapped SleepOnsetEnv (used for post-training eval
    + baseline rollouts where the rest of the code expects single-env API)."""
    env_cfg = EnvConfig(
        episode_s=cfg.episode_s,
        include_restfulness=cfg.include_restfulness,
        randomization=RandomizationRanges(),
        reward_weights=_resolve_reward_weights(cfg),
    )
    return Monitor(SleepOnsetEnv(env_cfg))


def _env_factory(cfg: TrainConfig, rank: int):
    """Closure returning a *factory* (zero-arg callable) for one env in
    a vec collection. Per-env seed = ``cfg.seed + rank`` so subprocesses
    don't all sample the same NMM subject. SubprocVecEnv calls this once
    in each child process; the resulting env stays in that process for
    the run."""
    base_seed = cfg.seed
    rw = _resolve_reward_weights(cfg)
    def _init() -> Monitor:
        env_cfg = EnvConfig(
            episode_s=cfg.episode_s,
            include_restfulness=cfg.include_restfulness,
            randomization=RandomizationRanges(),
            reward_weights=rw,
        )
        env = SleepOnsetEnv(env_cfg)
        env.reset(seed=base_seed + rank)   # warm the per-env RNG
        return Monitor(env)
    return _init


def make_vec_env(cfg: TrainConfig) -> VecEnv:
    """Build the training env. ``cfg.n_envs == 1`` -> in-process
    ``DummyVecEnv`` (cheap, debuggable). ``cfg.n_envs > 1`` ->
    ``SubprocVecEnv`` with one process per env. Clamped to
    ``os.cpu_count()`` so we never over-allocate."""
    n = max(1, int(cfg.n_envs))
    cpu = os.cpu_count() or 1
    if n > cpu:
        logger.warning("n_envs=%d > cpu_count=%d; clamping to %d", n, cpu, cpu)
        n = cpu
    factories = [_env_factory(cfg, rank) for rank in range(n)]
    if n == 1:
        return DummyVecEnv(factories)
    logger.info("Building SubprocVecEnv with %d parallel envs", n)
    return SubprocVecEnv(factories, start_method="fork")


def evaluate_baseline(policy, env: Monitor, n_episodes: int = 5) -> list:
    """Run baseline policy over n_episodes; return list of episode rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        total = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = policy.predict(obs)
            obs, r, done, truncated, _ = env.step(action)
            total += float(r)
        rewards.append(total)
    return rewards


def train(cfg: Optional[TrainConfig] = None) -> str:
    """
    Run a full training experiment + baseline comparisons.
    Returns the path to the saved SAC checkpoint.
    """
    cfg = cfg or TrainConfig()
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(cfg.experiment_name)
    # Encode variant+seed into the run/checkpoint name so the ablation
    # notebook + eval CLI can map "winning variant" -> checkpoint path
    # without having to query MLflow for artifact URIs.
    variant_slug = cfg.reward_variant or "custom"
    run_name = f"sac_{variant_slug}_seed{cfg.seed}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "algo": "SAC",
            "total_steps": cfg.total_steps,
            "episode_s": cfg.episode_s,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "gamma": cfg.gamma,
            "tau": cfg.tau,
            "seed": cfg.seed,
            "include_restfulness": cfg.include_restfulness,
        })

        # Reward variant + coefficients - surfaces in MLflow so the
        # ablation notebook can ``runs.search(experiment, filter="tags.reward_variant=...")``
        # and group learning curves by variant without inspecting code.
        rw = (cfg.reward_weights
              or (resolve_reward_variant(cfg.reward_variant)
                  if cfg.reward_variant else RewardWeights()))
        variant_label = cfg.reward_variant or "custom"
        mlflow.set_tag("reward_variant", variant_label)
        mlflow.log_params({
            "reward_variant":          variant_label,
            "k_theta_alpha":           rw.k_theta_alpha,
            "k_theta_alpha_level":     rw.k_theta_alpha_level,
            "k_hypnagogic":            rw.k_hypnagogic,
            "k_plv":                   rw.k_plv,
            "k_descent":               rw.k_descent,
            "k_alpha_pwr":             rw.k_alpha_pwr,
            "k_disconnect":            rw.k_disconnect,
            "flat_plv":                rw.flat_plv,
        })

        # Training env: SubprocVecEnv when cfg.n_envs > 1, DummyVecEnv otherwise.
        train_env = make_vec_env(cfg)
        mlflow.log_param("n_envs", cfg.n_envs)
        model = SAC(
            "MlpPolicy", train_env,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef=cfg.ent_coef,
            seed=cfg.seed,
            verbose=0,
        )

        logger.info("Training SAC for %d steps (n_envs=%d)...",
                    cfg.total_steps, cfg.n_envs)
        try:
            model.learn(
                total_timesteps=cfg.total_steps,
                callback=MLflowCallback(),
                progress_bar=False,
            )
        finally:
            # Always close vec env workers, even on training failure -
            # leaked subprocesses can wedge subsequent sweep iterations.
            train_env.close()

        # Save checkpoint
        ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.zip"
        model.save(str(ckpt_path))
        mlflow.log_artifact(str(ckpt_path), artifact_path="policy")

        # Eval phase uses a fresh SINGLE env - baseline policies +
        # evaluate_baseline() + the SAC eval loop all expect single-env
        # Gym API (5-tuple step return). Cheap to build, isolates eval
        # randomness from the training vec state.
        eval_env = make_env(cfg)
        logger.info("Evaluating baselines...")
        fixed = FixedFrequencyPolicy(frequency_hz=10.0)
        ramp = LinearRampDownPolicy(
            start_hz=11.0, end_hz=4.0, duration_s=cfg.episode_s,
        )
        fixed_rewards = evaluate_baseline(fixed, eval_env, n_episodes=5)
        ramp_rewards = evaluate_baseline(ramp, eval_env, n_episodes=5)
        sac_eval = []
        for _ in range(5):
            obs, _ = eval_env.reset()
            total = 0.0
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = eval_env.step(action)
                total += float(r)
            sac_eval.append(total)

        mlflow.log_metrics({
            "baseline_fixed_mean": float(np.mean(fixed_rewards)),
            "baseline_ramp_mean": float(np.mean(ramp_rewards)),
            "sac_eval_mean": float(np.mean(sac_eval)),
            "baseline_fixed_std": float(np.std(fixed_rewards)),
            "baseline_ramp_std": float(np.std(ramp_rewards)),
            "sac_eval_std": float(np.std(sac_eval)),
        })

        eval_env.close()
        logger.info("Done. SAC mean=%.3f, Fixed=%.3f, Ramp=%.3f",
                    np.mean(sac_eval), np.mean(fixed_rewards),
                    np.mean(ramp_rewards))
        logger.info("Checkpoint: %s", ckpt_path)
        logger.info("MLflow run: %s", run.info.run_id)

    return str(ckpt_path)
