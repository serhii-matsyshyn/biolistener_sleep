"""
Probe scheduling for a frequency sweep.

Probes are visited in a randomised order so that frequency is not confounded
with time-in-block: an ascending sweep would let habituation and electrode drift
masquerade as a falling response at high frequencies.

Randomisation is not free. The supervisor rate-limits frequency changes, so a
permutation that zig-zags across the grid spends longer in transit than the
block allows. :func:`plan_sweep` samples permutations and keeps one whose
transition cost fits the budget, which preserves the randomisation inside the
feasible set instead of falling back to a monotonic order.

The plan is nominal. What actually reached the mask is recorded per tick, and
:func:`settled_mask` selects analysable samples from the recording rather than
from this schedule.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from app.agent.safety import MAX_FREQ_STEP_HZ_PER_S


@dataclass(frozen=True)
class SweepStep:
    index: int
    target_hz: float
    t_command_s: float      # when the target is first asked for
    t_dwell_start_s: float  # when the rate limiter should have arrived
    t_end_s: float

    @property
    def transition_s(self) -> float:
        return self.t_dwell_start_s - self.t_command_s

    @property
    def dwell_s(self) -> float:
        return self.t_end_s - self.t_dwell_start_s


@dataclass(frozen=True)
class SweepPlan:
    order: Tuple[float, ...]
    steps: Tuple[SweepStep, ...]
    duration_s: float
    transition_cost_s: float
    seed: int
    tries: int

    def frequency_at(self, t_s: float) -> Optional[float]:
        """Target to command at ``t_s``; ``None`` past the end of the sweep."""
        for st in self.steps:
            if st.t_command_s <= t_s < st.t_end_s:
                return st.target_hz
        return None

    def step_at(self, t_s: float) -> Optional[SweepStep]:
        for st in self.steps:
            if st.t_command_s <= t_s < st.t_end_s:
                return st
        return None

    def phase_at(self, t_s: float) -> str:
        st = self.step_at(t_s)
        if st is None:
            return "idle"
        return "dwell" if t_s >= st.t_dwell_start_s else "transition"


def transition_cost_s(
    order: Sequence[float], start_hz: Optional[float] = None
) -> float:
    """Total time the rate limiter needs to walk this order."""
    if not order:
        return 0.0
    seq = list(order) if start_hz is None else [start_hz, *order]
    return sum(abs(b - a) for a, b in zip(seq, seq[1:])) / MAX_FREQ_STEP_HZ_PER_S


def plan_sweep(
    probe_hz: Sequence[float],
    dwell_s: float,
    duration_s: float,
    seed: int,
    start_hz: Optional[float] = None,
    max_tries: int = 2000,
) -> SweepPlan:
    """
    Randomised probe order whose transitions fit inside ``duration_s``.

    ``start_hz`` is the frequency the mask is already holding, so the walk into
    the first probe is charged to the budget too.
    """
    probes = list(probe_hz)
    if not probes:
        raise ValueError("no probe frequencies")
    budget = duration_s - len(probes) * dwell_s
    if budget < 0:
        raise ValueError(
            f"{len(probes)} probes at {dwell_s} s need {len(probes) * dwell_s} s "
            f"of dwell alone, block is {duration_s} s")

    rng = random.Random(seed)
    chosen: Optional[Tuple[float, ...]] = None
    cost = 0.0
    tries = 0
    for tries in range(1, max_tries + 1):
        cand = tuple(rng.sample(probes, len(probes)))
        cost = transition_cost_s(cand, start_hz)
        if cost <= budget:
            chosen = cand
            break
    if chosen is None:
        raise ValueError(
            f"no randomised order fits: {max_tries} tries, cheapest exceeded the "
            f"{budget:.0f} s transition budget. Lengthen the block, shorten the "
            f"dwell, or thin the probe grid.")

    steps, t, prev = [], 0.0, start_hz
    for i, f in enumerate(chosen):
        trans = 0.0 if prev is None else abs(f - prev) / MAX_FREQ_STEP_HZ_PER_S
        t_dwell = t + trans
        steps.append(SweepStep(i, f, t, t_dwell, t_dwell + dwell_s))
        t = t_dwell + dwell_s
        prev = f

    return SweepPlan(
        order=chosen, steps=tuple(steps), duration_s=duration_s,
        transition_cost_s=cost, seed=seed, tries=tries,
    )


def settled_mask(
    t_s: np.ndarray,
    actual_hz: np.ndarray,
    target_hz: np.ndarray,
    tol_hz: float = 0.1,
    settle_s: float = 1.0,
    sample_dt_s: Optional[float] = None,
) -> np.ndarray:
    """
    Analysable samples: delivered frequency is on target and has been for a while.

    Selects from what the mask actually did, so a transition that overran, a
    clamp, or a dropped command removes those samples instead of silently
    contaminating a probe's estimate. The first ``settle_s`` after arrival is
    also dropped, leaving out the entrainment build-up.
    """
    t_s = np.asarray(t_s, dtype=float)
    on_target = np.abs(np.asarray(actual_hz, float)
                       - np.asarray(target_hz, float)) <= tol_hz
    if not on_target.any():
        return on_target

    if sample_dt_s is None:
        sample_dt_s = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    need = max(1, int(round(settle_s / sample_dt_s)))

    out = np.zeros_like(on_target)
    run = 0
    for i, ok in enumerate(on_target):
        run = run + 1 if ok else 0
        out[i] = run >= need
    return out
