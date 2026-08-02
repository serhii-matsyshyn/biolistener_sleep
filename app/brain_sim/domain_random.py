"""
Domain randomization for sim-to-real transfer.

Each training episode samples a "subject" from a distribution of plausible
brains and photic responsiveness profiles. The RL agent learns a policy
that is robust across the distribution, so the real human is just another
sample from the same distribution.

Standard sim-to-real technique (Tobin et al. 2017; Sadeghi & Levine 2017).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nmm import JansenRitParams
from .photic_input import PhoticInputConfig


@dataclass
class SubjectSample:
    """A draw from the subject distribution - one simulated 'person'."""
    seed: int
    alpha_hz: float
    nmm_params: JansenRitParams
    photic_config: PhoticInputConfig
    responder_type: str  # "strong" | "moderate" | "non_responder"
    # This subject's arousal dynamics, surfaced for stratified analysis:
    # the headline question is whether adaptation helps the subjects who
    # have headroom, so every result must be splittable on sol_scale_s.
    sol_scale_s: float = 300.0
    arousal_floor: float = 0.3


@dataclass
class RandomizationRanges:
    """Bounds for the subject distribution. Tune to widen / narrow training."""
    alpha_hz_range: tuple = (8.5, 11.5)        # individual alpha frequency
    cortical_gain_range: tuple = (100.0, 700.0)  # photic responsiveness
    resonance_width_range: tuple = (2.0, 4.5)
    p_mean_range: tuple = (200.0, 240.0)       # background cortical drive
    p_std_range: tuple = (15.0, 30.0)          # noise level
    non_responder_prob: float = 0.10            # ~10% of subjects don't entrain

    # --- per-subject arousal dynamics -----------------------------------
    # How fast THIS subject transitions wake -> drowsy, in simulated seconds.
    # Lognormal, so the distribution is right-skewed like real sleep-onset
    # latency.
    #
    # Calibrated against the NO-STIMULATION resonance excursion, not against
    # clock time: if the peak does not move without stimulation, the "moving
    # target" a closed loop is supposed to track is manufactured by the
    # actuator (strobe low -> raise theta/alpha -> lower the peak -> justify
    # strobing low), which is circular. Measured with these values, n=10
    # no-stim episodes: 4 stationary / 3 moderate (0.05-2 Hz) / 3 large
    # (>=2 Hz), theta/alpha reaching median 0.98 and max 5.29.
    #
    # Aim for all three regimes to be populated, so that no single fixed
    # schedule fits both tails - NOT for a large MEAN excursion, which means
    # every subject completes its whole transition inside one episode and a
    # fixed schedule is optimal by construction. These are simulated seconds
    # and are not calibrated to minutes: no result derived from them may be
    # stated as a sleep-onset latency.
    sol_scale_s_median: float = 300.0
    sol_scale_s_gsd: float = 2.4                # geometric SD (multiplicative)
    # Deep floors matter: a subject that only gets part-way through its
    # transition still has to reach a low enough arousal for theta/alpha to
    # cross shift_baseline_ratio, or its resonance peak never moves at all
    # and there is no target for a controller to track.
    arousal_floor_range: tuple = (0.02, 0.15)   # how deep this subject goes

    # TRAINING hook. When set, the state->plant coupling gain is *sampled*
    # per subject from this range, so the policy sees the whole coupling
    # family during domain randomisation instead of only the default 2.0.
    # Without this, every checkpoint is trained at exactly one coupling and
    # a coupling sweep at evaluation time is off-distribution everywhere
    # else - which confounds "feedback has less to offer here" with "the
    # policy has never seen this plant".
    #
    # The draw is conditional, so leaving this None keeps the RNG sequence
    # bit-identical to the pre-existing subject distribution.
    shift_per_unit_ratio_range: Optional[tuple] = None

    # EVALUATION hook. When set, overrides PhoticInputConfig.shift_per_unit_ratio
    # on every sampled subject. 0.0 disables the state-dependent resonance
    # shift entirely, which is how we test whether a learned descent depends
    # on that modelling assumption. Applied AFTER all RNG draws, so the
    # subject sequence for a given seed is bit-identical at every level of
    # the coupling sweep. Wins over ``shift_per_unit_ratio_range``.
    shift_per_unit_ratio_override: Optional[float] = None

    # Contrast condition: drive the resonance shift from a fixed time
    # course rather than from measured θ/α (see PhoticInputConfig.
    # exogenous_shift). Applied after all RNG draws.
    exogenous_shift: bool = False

    def sample(self, seed: Optional[int] = None) -> SubjectSample:
        rng = np.random.default_rng(seed)
        alpha_hz = float(rng.uniform(*self.alpha_hz_range))
        resonance_width = float(rng.uniform(*self.resonance_width_range))

        is_non_responder = rng.random() < self.non_responder_prob
        if is_non_responder:
            cortical_gain = float(rng.uniform(20.0, 80.0))
            responder = "non_responder"
        else:
            cortical_gain = float(rng.uniform(*self.cortical_gain_range))
            responder = "strong" if cortical_gain > 400 else "moderate"

        nmm = JansenRitParams(
            p_mean=float(rng.uniform(*self.p_mean_range)),
            p_std=float(rng.uniform(*self.p_std_range)),
        )
        # Lognormal transition timescale. Drawn here, inside the RNG block,
        # so it is part of the reproducible subject stream.
        sol_scale_s = float(
            self.sol_scale_s_median
            * np.exp(np.log(self.sol_scale_s_gsd) * rng.standard_normal())
        )
        arousal_floor = float(rng.uniform(*self.arousal_floor_range))

        photic = PhoticInputConfig(
            cortical_gain=cortical_gain,
            base_resonance_hz=alpha_hz,
            resonance_width_hz=resonance_width,
            sol_scale_s=sol_scale_s,
            arousal_floor=arousal_floor,
        )
        # Last RNG draw, and conditional: when the range is None (the
        # evaluation setting) no number is consumed, so subject sequences
        # stay bit-identical to every previously generated artifact.
        if self.shift_per_unit_ratio_range is not None:
            photic.shift_per_unit_ratio = float(
                rng.uniform(*self.shift_per_unit_ratio_range)
            )
        # No RNG below this line - the override must not perturb the draw
        # sequence, so ablation and baseline see identical subjects.
        if self.shift_per_unit_ratio_override is not None:
            photic.shift_per_unit_ratio = float(self.shift_per_unit_ratio_override)
        photic.exogenous_shift = bool(self.exogenous_shift)
        return SubjectSample(
            seed=int(seed if seed is not None else rng.integers(0, 2**31)),
            alpha_hz=alpha_hz,
            nmm_params=nmm,
            photic_config=photic,
            responder_type=responder,
            sol_scale_s=sol_scale_s,
            arousal_floor=arousal_floor,
        )
