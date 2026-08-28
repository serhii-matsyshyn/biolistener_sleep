"""Sweep scheduling stays random, stays inside budget, and marks settled data."""
import numpy as np
import pytest

from app.agent.safety import MAX_FREQ_STEP_HZ_PER_S
from app.protocol.spec import DWELL_S, PROBE_HZ, SWEEP_BLOCK_S
from app.protocol.sweep import (
    plan_sweep,
    settled_mask,
    transition_cost_s,
)


def test_plan_visits_every_probe_exactly_once():
    p = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=0)
    assert sorted(p.order) == sorted(PROBE_HZ)
    assert len(p.steps) == len(PROBE_HZ)


def test_plan_fits_inside_the_block():
    for seed in range(20):
        p = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=seed, start_hz=11.0)
        assert p.steps[-1].t_end_s <= SWEEP_BLOCK_S


def test_every_probe_gets_its_full_dwell():
    p = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=3)
    for st in p.steps:
        assert st.dwell_s == pytest.approx(DWELL_S)


def test_transitions_respect_the_rate_limit():
    p = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=5, start_hz=11.0)
    prev = 11.0
    for st in p.steps:
        assert st.transition_s == pytest.approx(
            abs(st.target_hz - prev) / MAX_FREQ_STEP_HZ_PER_S)
        prev = st.target_hz


def test_the_same_seed_gives_the_same_order():
    a = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=7)
    b = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=7)
    assert a.order == b.order


def test_different_seeds_give_different_orders():
    orders = {plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=s).order
              for s in range(12)}
    assert len(orders) > 1


def test_order_is_not_monotonic():
    # An ascending sweep would confound probe frequency with time in block.
    non_monotonic = 0
    for seed in range(12):
        o = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=seed).order
        if list(o) != sorted(o) and list(o) != sorted(o, reverse=True):
            non_monotonic += 1
    assert non_monotonic == 12


def test_planning_fails_loudly_when_no_order_can_fit():
    with pytest.raises(ValueError, match="no randomised order fits"):
        plan_sweep(PROBE_HZ, DWELL_S, len(PROBE_HZ) * DWELL_S + 1.0, seed=0)


def test_planning_rejects_a_block_too_short_for_the_dwells():
    with pytest.raises(ValueError, match="dwell alone"):
        plan_sweep(PROBE_HZ, DWELL_S, 10.0, seed=0)


def test_transition_cost_counts_the_walk_into_the_first_probe():
    order = (10.0, 4.0)
    assert transition_cost_s(order) == pytest.approx(6.0 / MAX_FREQ_STEP_HZ_PER_S)
    assert transition_cost_s(order, start_hz=13.0) == pytest.approx(
        (3.0 + 6.0) / MAX_FREQ_STEP_HZ_PER_S)


def test_phase_reports_transition_then_dwell():
    p = plan_sweep(PROBE_HZ, DWELL_S, SWEEP_BLOCK_S, seed=0, start_hz=11.0)
    st = p.steps[1]
    assert p.phase_at(st.t_command_s + 0.01) == "transition"
    assert p.phase_at(st.t_dwell_start_s + 0.01) == "dwell"
    assert p.phase_at(SWEEP_BLOCK_S + 1) == "idle"


class TestSettledMask:
    def test_rejects_samples_off_target(self):
        t = np.arange(0, 10, 0.25)
        target = np.full_like(t, 10.0)
        actual = np.full_like(t, 6.0)
        assert not settled_mask(t, actual, target).any()

    def test_drops_the_build_up_after_arrival(self):
        t = np.arange(0, 10, 0.25)
        target = np.full_like(t, 10.0)
        actual = np.where(t < 4, 6.0, 10.0)
        m = settled_mask(t, actual, target, settle_s=1.0)
        assert t[m][0] == pytest.approx(4.75)

    def test_a_glitch_restarts_the_settling_clock(self):
        t = np.arange(0, 10, 0.25)
        target = np.full_like(t, 10.0)
        actual = np.full_like(t, 10.0)
        actual[20] = 4.0                      # one bad sample mid-dwell
        m = settled_mask(t, actual, target, settle_s=1.0)
        assert not m[20]
        assert not m[21]                      # clock restarted, not resumed
        assert m[24]

    def test_tolerance_admits_small_offsets(self):
        t = np.arange(0, 6, 0.25)
        target = np.full_like(t, 10.0)
        actual = np.full_like(t, 10.05)
        assert settled_mask(t, actual, target, tol_hz=0.1).any()
        assert not settled_mask(t, actual, target, tol_hz=0.01).any()
