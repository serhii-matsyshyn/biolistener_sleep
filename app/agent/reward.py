"""
Reward function for the RL agent - sleep-onset shaping.

Thesis goal: drive the user from wake into N1 hypnagogia. The reward is
designed so a policy that **parks at high-α with great phase-locking
forever** gets near-zero reward; the only way to climb is to actually
ramp the strobe down through the sleep-conducive bands while the brain
follows.

Term-by-term:

1. ``Δ(θ/α)``  - the validated drowsiness biomarker's rate of change.
   Positive when drowsiness biomarker grows tick-over-tick.
2. ``(θ/α − 1.0)+``  - sustained-level bonus. Keeps paying out while
   the ratio stays high, so the agent doesn't learn to oscillate it
   for harvestable deltas.
3. ``Δ hypnagogic-θ`` - explicit 4–7 Hz growth (the band that defines
   N1 hypnagogia).
4. ``band_weight(strobe_freq) · PLV``  - entrainment-fidelity bonus,
   but ONLY when the strobe is in a sleep-conducive band.  This is the
   change vs. the previous version: PLV at 12 Hz earns zero, PLV at
   6 Hz earns the full coefficient.  Closes the loophole where the
   policy could collect a constant PLV trickle at wake-frequency.
5. ``+k · max(0, prev_freq − curr_freq)`` - frequency-descent shaping.
   Tiny per-tick reward for ramping the strobe DOWN, zero for holds,
   negative for upward jumps (the SafetySupervisor's ramp-rate cap
   bounds this so the agent can't game it with infinite-rate descents).
6. ``k_disconnect`` (large negative) - electrode-disconnect bailout.

EMG penalty REMOVED - the v1 BioListener montage has no dedicated EMG
electrodes; the previous γ-band fallback was confounded with cortical γ
and added noise without measuring what it claimed to measure.

All terms are computed from MANUAL features + the stim action - no
MLModel dependency. The reward signature is stable across the project
lifetime so MLflow runs are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.eeg.features import FeatureSnapshot

# θ/α baseline above which the sustained-level bonus turns on.  ~1.0 is
# the "drowsy" inflection in resting-eyes-closed adult EEG literature
# (Berger 1929; Cantero 2002).  Tunable per subject if needed.
THETA_ALPHA_BASELINE = 1.0


@dataclass
class RewardWeights:
    """Default coefficients - tunable per training experiment.

    Two extra fields enable the **ablation-study baselines** (the
    masters-defence "show that naive rewards get exploited" figure):

    - ``k_alpha_pwr``: rewards raw α band power. Off (0) for the
      shaped reward; ON for the ``naive_alpha_max`` variant, where the
      policy is expected to discover that strobing at α-frequencies
      drives big α-power readings -> exploits the reward.
    - ``flat_plv``: when True, ``compute_reward`` skips the sleep-band
      weighting on PLV and gives the full coefficient at ANY strobe
      frequency. ON for the ``naive_plv_only`` variant, where the
      policy is expected to lock to whatever frequency it can entrain
      (often high-α) and park there forever.
    """
    k_theta_alpha: float = 1.0  # Δ(θ/α)
    k_theta_alpha_level: float = 0.2  # sustained (θ/α − baseline)+
    k_hypnagogic: float = 0.5  # Δ hypnagogic θ
    k_plv: float = 0.4  # entrainment fidelity (band-weighted)
    k_descent: float = 0.05  # per-Hz reward for downward ramps
    k_disconnect: float = -10.0  # electrode disconnect bailout
    k_alpha_pwr: float = 0.0  # ABLATION: raw α power (naive baseline)
    flat_plv: bool = False  # ABLATION: skip band_weight on PLV


def band_weight(strobe_freq_hz: float) -> float:
    """Sleep-band weighting for the PLV bonus.

    Closes the "park at 10 Hz forever" loophole - locking to high-α earns
    only a third of the bonus, locking to β earns nothing, locking to θ
    earns the full bonus (this is the band that defines hypnagogia).

    Bands:
        f < 1 Hz   ->  0.0   (strobe effectively off)
        1–4   Hz   ->  0.7   (δ - bonus only once the brain has descended)
        4–8   Hz   ->  1.0   (θ - peak; hypnagogic territory)
        8–12  Hz   ->  0.3   (α - entry phase, mild)
        f ≥ 12 Hz  ->  0.0   (β / wake - no credit for entraining here)
    """
    f = float(strobe_freq_hz)
    if f < 1.0:   return 0.0
    if f < 4.0:   return 0.7
    if f < 8.0:   return 1.0
    if f < 12.0:  return 0.3
    return 0.0


def compute_reward(
        prev: Optional[FeatureSnapshot],
        curr: FeatureSnapshot,
        plv_strobe: float,
        strobe_freq_hz: float = 0.0,
        prev_strobe_freq_hz: Optional[float] = None,
        weights: RewardWeights = RewardWeights(),
) -> float:
    """
    Per-tick reward.

    A reward computed BEFORE the first tick (prev=None) returns 0 - no
    direction to reward yet. ``strobe_freq_hz`` / ``prev_strobe_freq_hz``
    are the current/previous SAFE strobe frequency in Hz (after the
    SafetySupervisor clamps), used for band-weighting PLV and shaping
    descent. Both default to safe no-ops so unit tests that pre-date the
    signature change continue to work.
    """
    # 5) Electrode disconnect - immediate large negative, override everything
    if curr.railed_pct > 80.0:
        return float(weights.k_disconnect)

    if prev is None:
        return 0.0

    # 1) Δ(θ/α) - drowsiness biomarker rate of change
    dr_theta_alpha = curr.theta_alpha_ratio - prev.theta_alpha_ratio
    # 2) Sustained θ/α level - bonus that keeps paying out above baseline
    level_term = max(0.0, curr.theta_alpha_ratio - THETA_ALPHA_BASELINE)
    # 3) Δ hypnagogic θ power
    dr_hyp = curr.hypnagogic_theta_pwr - prev.hypnagogic_theta_pwr
    # 4) Entrainment bonus - band-weighted by default; the ablation's
    #    ``naive_plv_only`` variant sets ``flat_plv=True`` to skip the
    #    band weighting (so the policy can park at any frequency and
    #    still collect the bonus, demonstrating the loophole).
    raw_plv = max(0.0, min(1.0, plv_strobe))
    weight = 1.0 if weights.flat_plv else band_weight(strobe_freq_hz)
    plv_term = raw_plv * weight
    # 6) Frequency-descent shaping - positive on downward ramps, negative on up
    if prev_strobe_freq_hz is None:
        descent_term = 0.0
    else:
        descent_term = prev_strobe_freq_hz - strobe_freq_hz  # +Hz/tick on descent

    return float(
        weights.k_theta_alpha * dr_theta_alpha
        + weights.k_theta_alpha_level * level_term
        + weights.k_hypnagogic * dr_hyp
        + weights.k_plv * plv_term
        + weights.k_descent * descent_term
        + weights.k_alpha_pwr * curr.alpha_pwr  # naive baseline
    )
