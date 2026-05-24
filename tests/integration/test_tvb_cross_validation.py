"""
Regression test: hand-written Jansen-Rit must match TVB's canonical
implementation on the natural α peak.

This is the thesis-defence answer to *"your simulator is hand-coded -
how do we know it's correct?"*. TVB is the canonical NMM library and
ships Jansen & Rit (1995) as a built-in model. When parameterised
identically (v0=6.0 mV per the 1995 paper, TVB ships v0=5.52 from an
alternate variant), both implementations independently produce the same
α-band peak.

Notebook 01 (section 4) runs the same comparison interactively with a
PSD overlay plot; this test is the CI-friendly version.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# Skip if TVB isn't installed (optional dep - only needed for this test)
tvb = pytest.importorskip("tvb.simulator.lab",
                            reason="tvb-library not installed")

SAMPLING_RATE = 500.0
DURATION_S = 8.0
SETTLE_S = 2.0
SEED = 13


def _run_ours() -> np.ndarray:
    from app.brain_sim.nmm import JansenRitColumn, JansenRitParams
    col = JansenRitColumn(JansenRitParams(), dt=1.0 / SAMPLING_RATE, seed=SEED)
    n = int(DURATION_S * SAMPLING_RATE)
    out = np.zeros(n)
    for i in range(n):
        out[i] = col.step(photic_input=0.0)
    return out[int(SETTLE_S * SAMPLING_RATE):]


def _run_tvb() -> np.ndarray:
    from tvb.simulator.lab import integrators, monitors, simulator
    from tvb.simulator.coupling import Linear
    from tvb.simulator.models.jansen_rit import JansenRit
    from tvb.datatypes.connectivity import Connectivity

    conn = Connectivity(
        weights=np.zeros((1, 1)),
        tract_lengths=np.zeros((1, 1)),
        region_labels=np.array(["V1"]),
        centres=np.zeros((1, 3)),
    )
    conn.configure()

    # v0=6.0 matches our JansenRitParams default and the 1995 paper.
    # Deterministic integrator avoids tangling with TVB's stochastic
    # noise model (different scaling than our single-draw-per-step).
    model = JansenRit(v0=np.array([6.0]))
    integrator = integrators.HeunDeterministic(dt=2 ** -4)
    mon = monitors.TemporalAverage(period=2.0)   # 500 Hz output

    sim = simulator.Simulator(
        model=model, connectivity=conn,
        coupling=Linear(a=np.array([0.0])),
        integrator=integrator, monitors=(mon,),
        simulation_length=DURATION_S * 1000.0,   # TVB uses ms
    )
    sim.configure()
    _, data = sim.run()[0]
    y = data[:, 1, 0, 0] - data[:, 2, 0, 0]
    return y[int(SETTLE_S * SAMPLING_RATE):]


def _peak_freq(trace: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.signal import welch
    f, psd = welch(trace, fs=SAMPLING_RATE, nperseg=min(1024, len(trace)))
    rng = (f >= 1) & (f <= 30)
    f, psd = f[rng], psd[rng]
    return float(f[np.argmax(psd)]), f, psd


def test_jansen_rit_alpha_peak_matches_tvb():
    """Both implementations must peak in α band within ≤2 Hz of each
    other. This regression-tests for any future refactor of our NMM
    that drifts away from canonical JR behaviour."""
    ours = _run_ours()
    tvb_trace = _run_tvb()
    p_ours, _, _ = _peak_freq(ours)
    p_tvb, _, _ = _peak_freq(tvb_trace)

    # Both must land in α band
    assert 8.0 <= p_ours <= 13.0, f"our peak {p_ours} Hz not in α"
    assert 8.0 <= p_tvb <= 13.0,  f"TVB peak {p_tvb} Hz not in α"

    # Peaks must agree
    assert abs(p_ours - p_tvb) <= 2.0, (
        f"peak mismatch: ours={p_ours:.2f}, TVB={p_tvb:.2f}, "
        "Δ exceeds 2 Hz tolerance"
    )


def test_jansen_rit_psd_shape_correlates_with_tvb():
    """Beyond the peak frequency, the overall PSD shape across [1, 30] Hz
    must correlate strongly with TVB's. This catches drift in the
    high-frequency tail or sub-α structure that a single-peak check
    would miss."""
    ours = _run_ours()
    tvb_trace = _run_tvb()
    _, _, psd_a = _peak_freq(ours)
    _, _, psd_b = _peak_freq(tvb_trace)
    log_a = np.log10(psd_a + 1e-30)
    log_b = np.log10(psd_b + 1e-30)
    corr = np.corrcoef(log_a, log_b)[0, 1]
    assert corr >= 0.7, (
        f"log-PSD correlation only {corr:.3f}; expected ≥ 0.7 - "
        "our impl has drifted from TVB-canonical JR behaviour"
    )
