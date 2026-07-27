"""
Roll a trained SAC policy on N fresh domain-randomised NMM subjects and
dump per-tick trajectories to a tidy DataFrame.

Used by Phase B of the ablation experiment: "the winning reward variant
generalises across simulated subjects". The output Parquet feeds notebook
cell 4 (cross-subject consistency plot).

CLI wrapper: ``biolistener-sleep eval --checkpoint <path> --n 100 --out file.parquet``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from app.agent.env import EnvConfig, SleepOnsetEnv
from app.brain_sim.domain_random import RandomizationRanges

logger = logging.getLogger("agent.eval_rollout")


def rollout_episodes(
    checkpoint_path: Path,
    n_episodes: int = 100,
    episode_s: float = 60.0,
    randomization: Optional[RandomizationRanges] = None,
    base_seed: int = 1000,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Load an SAC checkpoint and run ``n_episodes`` against fresh
    domain-randomised NMM subjects (one new sample per reset).

    Returns a tidy DataFrame keyed by ``(episode, t_s)`` with the columns
    the cross-subject consistency plot needs:

        episode, t_s, strobe_freq_hz, fill, brightness, alpha, theta,
        beta, hypno_theta, sigma, theta_alpha_ratio, plv, reward, enabled

    Per-episode randomisation seeds are derived from ``base_seed`` so the
    same call is reproducible.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    env_cfg = EnvConfig(
        episode_s=episode_s,
        include_restfulness=False,        # not needed for trajectory dumps
        randomization=randomization or RandomizationRanges(),
    )
    env = SleepOnsetEnv(env_cfg)

    model = SAC.load(str(checkpoint_path), device="cpu")
    logger.info("Loaded checkpoint %s; rolling out %d episode(s)",
                checkpoint_path.name, n_episodes)

    rows: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            feats = info["features"]
            stim  = info["stim"]
            rows.append({
                "episode":           ep,
                "t_s":               float(info["t"]),
                "strobe_freq_hz":    float(stim.frequency_hz),
                "fill":              float(stim.fill),
                "brightness":        float(stim.brightness),
                "enabled":           bool(stim.enabled),
                "alpha":             float(feats.alpha_pwr),
                "theta":             float(feats.theta_pwr),
                "beta":              float(feats.beta_pwr),
                "hypno_theta":       float(feats.hypnagogic_theta_pwr),
                "sigma":             float(feats.sigma_pwr),
                "theta_alpha_ratio": float(feats.theta_alpha_ratio),
                "plv":               float(info["plv"]),
                "plv_2f":            float(info.get("plv_2f", 0.0)),
                "reward":            float(reward),
            })

    env.close()
    return pd.DataFrame(rows)


def write_parquet(df: pd.DataFrame, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote %d rows × %d cols -> %s", len(df), df.shape[1], out)


def summary(df: pd.DataFrame) -> dict:
    """Quick numbers for a one-line CLI report."""
    if df.empty:
        return {"n_episodes": 0}
    by_ep = df.groupby("episode").agg(
        total_reward=("reward", "sum"),
        final_ratio=("theta_alpha_ratio", "last"),
        final_strobe=("strobe_freq_hz", "last"),
        mean_plv=("plv", "mean"),
    )
    return {
        "n_episodes":               int(by_ep.shape[0]),
        "mean_total_reward":        float(by_ep["total_reward"].mean()),
        "mean_final_theta_alpha":   float(by_ep["final_ratio"].mean()),
        "mean_final_strobe_hz":     float(by_ep["final_strobe"].mean()),
        "std_final_strobe_hz":      float(by_ep["final_strobe"].std()),
        "median_final_strobe_hz":   float(by_ep["final_strobe"].median()),
        "frac_final_below_6hz":     float((by_ep["final_strobe"] < 6.0).mean()),
        "frac_final_below_8hz":     float((by_ep["final_strobe"] < 8.0).mean()),
        "mean_plv":                 float(by_ep["mean_plv"].mean()),
    }
