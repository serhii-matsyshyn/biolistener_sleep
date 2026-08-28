"""
Roll a stimulation policy on N fresh domain-randomised NMM subjects and
dump per-tick trajectories to a tidy DataFrame.

One harness drives every experimental arm - no-stim, fixed-frequency,
scripted ramp, SAC, yoked replay and oracle open-loop - on *bit-identical*
subjects, so the arms differ only in what they command.

Three properties matter for that comparison and are enforced here:

1. **Policy injection.** ``policy`` accepts anything with
   ``predict(obs, deterministic=True) -> (action, state)``; ``checkpoint_path``
   is the SAC convenience path. ``policy_factory`` builds a fresh policy
   per episode from the subject's static parameters - needed by the
   oracle arm (which is allowed to see ``alpha_hz``) and the yoked arm
   (which replays a donor subject's recorded trajectory).

2. **Simulated clock.** Any policy exposing a ``tick_hz`` field gets it
   bound to the env's control rate, so time-driven open-loop policies
   advance one control tick per ``predict()`` call instead of reading the
   wall clock - which in simulation would collapse a 300 s ramp into a
   couple of seconds. A policy that looks time-driven but exposes no
   ``tick_hz`` field is rejected outright rather than silently mis-run.

3. **Per-episode reset.** ``reset()`` is called at the start of every
   episode, or a tick counter carries over and the ramp arms break
   silently after episode 0.

The frame carries the *clamped* physical command per tick (including the
audio dims), which is what makes the yoked arm's dose matching exact:
:mod:`app.agent.policies.yoked` re-normalises those columns through the
inverse of ``SleepOnsetEnv._decode_action`` and replays them verbatim.

CLI wrapper: ``biolistener-sleep eval --checkpoint <path> --n 100 --out file.parquet``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.agent.env import CONTROL_TICK_HZ, EnvConfig, SleepOnsetEnv
from app.brain_sim.domain_random import RandomizationRanges

logger = logging.getLogger("agent.eval_rollout")


def bind_sim_clock(policy: object, tick_hz: float) -> None:
    """Point a time-driven policy at the simulated clock.

    Policies that measure elapsed time expose a ``tick_hz`` field: ``None``
    means "read the wall clock" (correct in a live session) and a number
    means "one ``predict()`` call is 1 / tick_hz of simulated time". In
    simulation the second is the only correct reading, so bind it.

    Raises if a policy carries a wall-clock start stamp but offers no
    ``tick_hz`` field: that must fail loudly rather than produce a ramp
    that finishes in the first two seconds of a 300 s episode.
    """
    if hasattr(policy, "tick_hz"):
        if getattr(policy, "tick_hz") is None:
            policy.tick_hz = float(tick_hz)
        return
    if hasattr(policy, "_start_t"):
        raise TypeError(
            f"{type(policy).__name__} measures wall-clock time (_start_t) but "
            "exposes no tick_hz field; it cannot be run against a simulated "
            "clock. Add `tick_hz: Optional[float] = None` and count "
            "predict() calls (see LinearRampDownPolicy)."
        )


def rollout_episodes(
    checkpoint_path: Optional[Path] = None,
    n_episodes: int = 100,
    episode_s: float = 60.0,
    randomization: Optional[RandomizationRanges] = None,
    base_seed: int = 1000,
    deterministic: bool = True,
    *,
    policy: Optional[object] = None,
    policy_factory: Optional[Callable[[int, dict], object]] = None,
    tick_hz: float = CONTROL_TICK_HZ,
    progress_every: int = 0,
    label: str = "",
    diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Run ``n_episodes`` against fresh domain-randomised NMM subjects (one
    new sample per reset) under a fixed policy.

    Exactly one of ``checkpoint_path``, ``policy`` or ``policy_factory``
    must be given. ``policy_factory(episode_index, subject_info)`` is
    called after each ``reset()`` and receives the dict returned in
    ``info["subject"]``.

    Per-episode randomisation seeds are derived from ``base_seed`` so the
    same call is reproducible and so every arm sees the same subjects.
    """
    n_sources = sum(x is not None
                    for x in (checkpoint_path, policy, policy_factory))
    if n_sources != 1:
        raise ValueError(
            "pass exactly one of checkpoint_path / policy / policy_factory "
            f"(got {n_sources})"
        )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        from stable_baselines3 import SAC
        policy = SAC.load(str(checkpoint_path), device="cpu")
        logger.info("Loaded checkpoint %s", checkpoint_path.name)

    env_cfg = EnvConfig(
        episode_s=episode_s,
        tick_hz=tick_hz,
        include_restfulness=False,        # not needed for trajectory dumps
        randomization=randomization or RandomizationRanges(),
        diagnostics=diagnostics,
    )
    env = SleepOnsetEnv(env_cfg)

    if policy is not None:
        bind_sim_clock(policy, tick_hz)

    rows: list[dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        obs, info0 = env.reset(seed=base_seed + ep)
        subject = dict(info0.get("subject", {}))
        ep_policy = policy
        if policy_factory is not None:
            ep_policy = policy_factory(ep, subject)
            bind_sim_clock(ep_policy, tick_hz)
        # REQUIRED: tick-counting policies carry their counter across
        # episodes otherwise, and every arm after the first is invalid.
        if hasattr(ep_policy, "reset"):
            ep_policy.reset()

        done = trunc = False
        tick = 0
        while not (done or trunc):
            n_ev = env._supervisor.event_count
            action, _ = ep_policy.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            new_events = env._supervisor.events_since(n_ev)
            feats = info["features"]
            stim = info["stim"]
            diag = info.get("diag") or {}
            rows.append({
                **{f"d_{k}": float(v) for k, v in diag.items()},
                "episode":           ep,
                "tick":              tick,
                "t_s":               float(info["t"]),
                "strobe_freq_hz":    float(stim.frequency_hz),
                "fill":              float(stim.fill),
                "brightness":        float(stim.brightness),
                "enabled":           bool(stim.enabled),
                "audio_depth_pct":   float(stim.audio_strobe_depth_pct),
                "audio_vol_pct":     float(stim.audio_master_vol_pct),
                "alpha":             float(feats.alpha_pwr),
                "theta":             float(feats.theta_pwr),
                "beta":              float(feats.beta_pwr),
                "hypno_theta":       float(feats.hypnagogic_theta_pwr),
                "sigma":             float(feats.sigma_pwr),
                "theta_alpha_ratio": float(feats.theta_alpha_ratio),
                "plv":               float(info["plv"]),
                "plv_2f":            float(info.get("plv_2f", 0.0)),
                "reward":            float(reward),
                # Realised plant state for the coupling sweep: resonance
                # drift is an outcome of the policy, not a setting.
                "effective_resonance_hz": float(info["effective_resonance_hz"]),
                "shift_drive":       float(info["shift_drive"]),
                # Safety accounting per tick.
                "n_safety_events":   len(new_events),
                "ramp_limited_freq": any(e.kind == "ramp_limit"
                                         and e.field == "frequency_hz"
                                         for e in new_events),
                # Subject statics, repeated per tick so a single tidy frame
                # supports the cortical-gain-tercile stratification without
                # a join.
                "responder_type":    subject.get("responder", ""),
                "alpha_hz":          float(subject.get("alpha_hz", np.nan)),
                "cortical_gain":     float(subject.get("cortical_gain", np.nan)),
                "resonance_width_hz": float(subject.get("resonance_width_hz",
                                                        np.nan)),
                "shift_per_unit_ratio": float(
                    subject.get("shift_per_unit_ratio", np.nan)),
                "exogenous_shift":   bool(subject.get("exogenous_shift", False)),
            })
            tick += 1

        if progress_every and (ep + 1) % progress_every == 0:
            el = time.time() - t0
            eta = el / (ep + 1) * (n_episodes - ep - 1)
            print(f"[{label or 'rollout'}] {ep + 1}/{n_episodes} episodes  "
                  f"{el / 60:.1f} min elapsed  eta {eta / 60:.1f} min",
                  flush=True)

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
    # Frequency metrics read DELIVERED stimulation, not the commanded
    # register: an arm with the light off still holds a frequency value, and
    # a descent on zero photic energy is not a descent.
    d = df.copy()
    delivered = (d["fill"] > 0.0) & (d["brightness"] > 0.0)
    if "enabled" in d.columns:
        delivered &= d["enabled"].astype(bool)
    d.loc[~delivered, "strobe_freq_hz"] = np.nan
    by_ep = d.groupby("episode").agg(
        total_reward=("reward", "sum"),
        final_ratio=("theta_alpha_ratio", "last"),
        final_strobe=("strobe_freq_hz", "last"),
        mean_plv=("plv", "mean"),
        frac_delivering=("strobe_freq_hz", lambda c: float(c.notna().mean())),
    )
    # One subset for every field. `agg("last")` skips NaN, so `final_strobe` is the
    # last DELIVERING tick; an episode that never delivered has NaN there.
    # Those episodes are dropped from every frequency statistic, so the medians
    # and frac_below_* are computed over the same subset.
    delivering_eps = by_ep["final_strobe"].notna()
    by_f = by_ep[delivering_eps]
    return {
        "n_episodes":               int(by_ep.shape[0]),
        "mean_total_reward":        float(by_ep["total_reward"].mean()),
        "mean_final_theta_alpha":   float(by_ep["final_ratio"].mean()),
        "n_episodes_delivering":    int(delivering_eps.sum()),
        "mean_final_strobe_hz":     float(by_f["final_strobe"].mean()),
        "std_final_strobe_hz":      float(by_f["final_strobe"].std()),
        "median_final_strobe_hz":   float(by_f["final_strobe"].median()),
        "frac_final_below_6hz":     float((by_f["final_strobe"] < 6.0).mean()),
        "frac_final_below_8hz":     float((by_f["final_strobe"] < 8.0).mean()),
        "mean_plv":                 float(by_ep["mean_plv"].mean()),
        "frac_ticks_delivering":    float(by_ep["frac_delivering"].mean()),
    }
