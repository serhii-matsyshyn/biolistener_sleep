"""
Yoked-replay control arm: a dose-matched, non-adaptive baseline.

The closed-loop arm differs from an open-loop arm in *two* ways at once:
it delivers a different dose, and it adapts. A yoked control removes the
first difference so only the second remains. We record the closed-loop
arm's **clamped** per-tick command on every subject and replay subject
*i*'s recorded trajectory on subject *(i + 1) mod n* - a fixed
derangement, so no subject ever replays its own trajectory. The yoked arm
then receives exactly the dose the closed-loop arm delivered, in exactly
the same order, but blind to the brain in front of it. Combined with the
scripted schedules in :mod:`app.agent.policies.scheduled_ramp`, this
isolates the contribution of adaptivity from that of light dose.

**Dose equivalence is exact, not approximate.** ``SleepOnsetEnv._decode_action``
is an affine map from normalised [-1, 1] to physical units, so the
physical -> normalised inverse implemented here round-trips to float
precision. Replaying values that were *already* clamped means the
SafetySupervisor passes them through unchanged: the ramp limiter sees the
same per-tick deltas that it itself produced, and the bounds checks see
values already inside the bounds. ``assert_dose_equivalence`` verifies
this on the actual output rather than assuming it.

The closed-loop arm oscillates - it travels ~6x further in frequency
than its net descent requires - and the yoked arm replays those
oscillations verbatim. That is what "dose-matched" means, not a defect
of the replay.

**What yoking does NOT match.** The recorded trajectory was generated in
response to subject *i*'s state. Replayed on subject *i+1* it produces a
different θ/α evolution and therefore - because the resonance peak is
keyed to θ/α - a different plant trajectory. The arms match on commanded
dose, never on realised plant state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Physical <-> normalised action columns, in action-vector order. Must
# mirror SleepOnsetEnv._decode_action; the unit tests assert the
# round-trip against that method directly so the two cannot drift.
_ACTION_COLUMNS = (
    "strobe_freq_hz", "fill", "brightness", "audio_depth_pct", "audio_vol_pct",
)


def normalise_clamped_actions(df_ep: pd.DataFrame) -> np.ndarray:
    """Invert ``SleepOnsetEnv._decode_action`` over one episode's rows.

    Returns ``(n_ticks, 5)`` float64 - deliberately not float32, so the
    encode/decode round-trip contributes ~1e-15 Hz rather than ~1e-6 Hz
    and the dose-equivalence assertion tests the design rather than the
    dtype.
    """
    from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ

    missing = [c for c in _ACTION_COLUMNS if c not in df_ep.columns]
    if missing:
        raise KeyError(
            f"donor frame is missing clamped-action columns {missing}; "
            "it was written by an older rollout harness"
        )
    freq = df_ep["strobe_freq_hz"].to_numpy(np.float64)
    fill = df_ep["fill"].to_numpy(np.float64)
    bri = df_ep["brightness"].to_numpy(np.float64)
    depth = df_ep["audio_depth_pct"].to_numpy(np.float64)
    vol = df_ep["audio_vol_pct"].to_numpy(np.float64)
    return np.stack([
        2.0 * (freq - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ) - 1.0,
        2.0 * fill - 1.0,
        2.0 * bri - 1.0,
        2.0 * (depth / 100.0) - 1.0,
        2.0 * (vol / 100.0) - 1.0,
    ], axis=1)


@dataclass
class YokedReplayPolicy:
    """Replays a recorded per-tick action trajectory, ignoring observations."""
    trajectory: np.ndarray          # (n_ticks, 5), NORMALISED actions
    donor_episode: int = -1
    _t: int = 0

    def reset(self) -> None:
        self._t = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        # Donor and recipient episodes are the same length by construction,
        # so this clamp is unreachable in normal use; it keeps a truncated
        # donor trajectory from crashing a long sweep.
        a = self.trajectory[min(self._t, len(self.trajectory) - 1)]
        self._t += 1
        return a, None


def build_yoked_factory(
    donor_df: pd.DataFrame,
    *,
    offset: int = 1,
    n_episodes: Optional[int] = None,
):
    """Build a ``policy_factory`` replaying donor episode ``(e - offset) % n``.

    ``donor_df`` is the tidy per-tick frame produced by
    :func:`app.agent.eval_rollout.rollout_episodes` for the closed-loop
    arm on the *same* subjects at the *same* coupling level.
    """
    episodes = sorted(int(e) for e in donor_df["episode"].unique())
    n = int(n_episodes if n_episodes is not None else len(episodes))
    if n < 2:
        raise ValueError("yoked replay needs >= 2 episodes to derange")
    if offset % n == 0:
        raise ValueError(
            f"offset={offset} with n={n} makes every subject replay its own "
            "trajectory - that is not a yoked control"
        )
    by_ep = {int(e): normalise_clamped_actions(g)
             for e, g in donor_df.groupby("episode", sort=True)}
    missing = [e for e in range(n) if e not in by_ep]
    if missing:
        raise KeyError(f"donor frame lacks episodes {missing[:5]}...")

    def _factory(episode: int, subject: dict) -> YokedReplayPolicy:
        donor = (episode - offset) % n
        return YokedReplayPolicy(trajectory=by_ep[donor], donor_episode=donor)

    return _factory


def assert_dose_equivalence(
    donor_df: pd.DataFrame,
    yoked_df: pd.DataFrame,
    *,
    offset: int = 1,
    tol: float = 1e-6,
    columns: tuple = _ACTION_COLUMNS,
) -> dict:
    """Verify the yoked arm commanded exactly the donor arm's dose.

    Compares, per recipient episode, the clamped command sequence against
    the donor episode it replayed.

    Exactness holds only where neither arm dropped out. The supervisor's
    electrode-disconnect guard is state-dependent: when the simulated EEG
    rails it forces ``enabled=False, brightness=0`` for that tick and
    returns without updating the ramp limiter's reference action. Subject
    *i + 1* rails at different ticks from subject *i*, so from the first
    such tick the two arms' clamped outputs diverge and the ramp limiter
    keeps them apart - unavoidable while the guard remains a genuine
    safety response to the brain in front of it.

    The assertion is therefore enforced on dropout-free episode pairs
    only; pairs containing a dropout are counted and their divergence
    measured so the split is reported rather than averaged over.
    """
    n = int(yoked_df["episode"].nunique())
    donor_groups = {int(e): g.sort_values("tick")
                    for e, g in donor_df.groupby("episode")}
    has_enabled = "enabled" in donor_df.columns and "enabled" in yoked_df.columns

    exact_worst = {c: 0.0 for c in columns}
    exact_where = {c: None for c in columns}
    div_worst = {c: 0.0 for c in columns}
    exact_eps: list[int] = []
    div_eps: list[int] = []
    div_freq_deltas: list[float] = []

    for ep, g in yoked_df.groupby("episode"):
        d = donor_groups[(int(ep) - offset) % n]
        g = g.sort_values("tick")
        m = min(len(g), len(d))
        dropped = bool(
            (~g["enabled"].to_numpy()[:m]).any()
            or (~d["enabled"].to_numpy()[:m]).any()
        ) if has_enabled else False
        deltas = {}
        for c in columns:
            deltas[c] = float(np.max(np.abs(
                g[c].to_numpy(np.float64)[:m] - d[c].to_numpy(np.float64)[:m]
            ))) if m else 0.0
        if dropped:
            div_eps.append(int(ep))
            div_freq_deltas.append(deltas["strobe_freq_hz"])
            for c, v in deltas.items():
                div_worst[c] = max(div_worst[c], v)
        else:
            exact_eps.append(int(ep))
            for c, v in deltas.items():
                if v > exact_worst[c]:
                    exact_worst[c] = v
                    exact_where[c] = int(ep)

    bad = {c: v for c, v in exact_worst.items() if v > tol}
    if bad:
        raise AssertionError(
            f"dose equivalence violated on dropout-free pairs (tol={tol}): "
            + ", ".join(f"{c}: max|Δ|={v:.3e} at episode {exact_where[c]}"
                        for c, v in bad.items())
        )
    out = {f"max_abs_delta_{c}": v for c, v in exact_worst.items()}
    out.update({
        "n_pairs": int(len(exact_eps) + len(div_eps)),
        "n_exact": len(exact_eps),
        "n_divergent_dropout": len(div_eps),
        "exact_episodes": exact_eps,
        "max_abs_delta_freq_divergent": div_worst["strobe_freq_hz"],
        "median_abs_delta_freq_divergent": (
            float(np.median(div_freq_deltas)) if div_freq_deltas else 0.0),
    })
    return out
