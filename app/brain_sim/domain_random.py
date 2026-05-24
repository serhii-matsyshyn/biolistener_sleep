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


@dataclass
class RandomizationRanges:
    """Bounds for the subject distribution. Tune to widen / narrow training."""
    alpha_hz_range: tuple = (8.5, 11.5)        # individual alpha frequency
    cortical_gain_range: tuple = (100.0, 700.0)  # photic responsiveness
    resonance_width_range: tuple = (2.0, 4.5)
    p_mean_range: tuple = (200.0, 240.0)       # background cortical drive
    p_std_range: tuple = (15.0, 30.0)          # noise level
    non_responder_prob: float = 0.10            # ~10% of subjects don't entrain

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
        photic = PhoticInputConfig(
            cortical_gain=cortical_gain,
            base_resonance_hz=alpha_hz,
            resonance_width_hz=resonance_width,
        )
        return SubjectSample(
            seed=int(seed if seed is not None else rng.integers(0, 2**31)),
            alpha_hz=alpha_hz,
            nmm_params=nmm,
            photic_config=photic,
            responder_type=responder,
        )
