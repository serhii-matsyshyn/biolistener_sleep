"""
Named reward-formulation presets for the ablation experiment.

This module pins down the four formulations we compare so the experiment
is reproducible from a single registry rather than scattered configs.

The four variants:

- ``naive_alpha_max``: the proposal's original "α-desync" idea, naively
  inverted to "maximise α". Has only one term - raw α-band power.
  *Expected exploit*: the policy parks the strobe at an α-driving
  frequency (~10 Hz) because driving α produces a big α-power reading.
  The agent never ramps down. **This is the figure that proves you
  needed the shaped reward.**

- ``naive_plv_only``: rewards phase-locking value at ANY frequency
  (``flat_plv=True`` skips band weighting). *Expected exploit*: the
  policy locks PLV at whatever strobe frequency it can entrain - often
  high-α - and parks forever. Demonstrates the band-weighting fix.

- ``theta_alpha_delta``: drowsiness-biomarker terms only (Δθ/α,
  sustained-level, hypnoθ). No PLV, no descent shaping. *Expected
  behaviour*: wanders. The biomarker rewards are noisy without an
  entrainment bonus to anchor the policy to a frequency.

- ``full_shaped``: the recommended current default - biomarker terms +
  band-weighted PLV + descent shaping. *Expected behaviour*: ramps the
  strobe from α through θ as the brain follows.

The ``k_disconnect`` safety bailout stays identical across all variants;
swapping it out would conflate reward-shaping experiments with safety
experiments.
"""
from __future__ import annotations

from typing import Union

from app.agent.reward import RewardWeights


REWARD_VARIANTS: dict[str, RewardWeights] = {
    "naive_alpha_max": RewardWeights(
        k_theta_alpha=0.0,
        k_theta_alpha_level=0.0,
        k_hypnagogic=0.0,
        k_plv=0.0,
        k_descent=0.0,
        k_alpha_pwr=1.0,
        flat_plv=False,
        k_disconnect=-10.0,
    ),
    "naive_plv_only": RewardWeights(
        k_theta_alpha=0.0,
        k_theta_alpha_level=0.0,
        k_hypnagogic=0.0,
        k_plv=1.0,
        k_descent=0.0,
        k_alpha_pwr=0.0,
        flat_plv=True,         # ← unweighted PLV; locks to any frequency
        k_disconnect=-10.0,
    ),
    "theta_alpha_delta": RewardWeights(
        k_theta_alpha=1.0,
        k_theta_alpha_level=0.2,
        k_hypnagogic=0.5,
        k_plv=0.0,
        k_descent=0.0,
        k_alpha_pwr=0.0,
        flat_plv=False,
        k_disconnect=-10.0,
    ),
    "full_shaped": RewardWeights(           # current default (matches reward.py defaults)
        k_theta_alpha=1.0,
        k_theta_alpha_level=0.2,
        k_hypnagogic=0.5,
        k_plv=0.4,
        k_descent=0.05,
        k_alpha_pwr=0.0,
        flat_plv=False,
        k_disconnect=-10.0,
    ),
}


def resolve(variant_or_weights: Union[str, RewardWeights, None]) -> RewardWeights:
    """
    Coerce a config value (registry name, ``RewardWeights`` instance, or
    ``None``) into a concrete ``RewardWeights``.

    >>> resolve("full_shaped") is REWARD_VARIANTS["full_shaped"]
    True
    >>> isinstance(resolve(None), RewardWeights)
    True
    """
    if variant_or_weights is None:
        return RewardWeights()
    if isinstance(variant_or_weights, RewardWeights):
        return variant_or_weights
    if isinstance(variant_or_weights, str):
        try:
            return REWARD_VARIANTS[variant_or_weights]
        except KeyError as err:
            known = ", ".join(sorted(REWARD_VARIANTS))
            raise ValueError(
                f"Unknown reward variant {variant_or_weights!r}; "
                f"known: {known}"
            ) from err
    raise TypeError(
        f"Expected str | RewardWeights | None, got {type(variant_or_weights).__name__}"
    )
