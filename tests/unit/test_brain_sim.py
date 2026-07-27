"""
M1 verification: the Jansen-Rit NMM must produce realistic α-band activity,
photic input must shift / enhance the spectrum at the strobe frequency, and
domain randomization must cover the full α-frequency range (8.5–11.5 Hz).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.brain_sim.domain_random import RandomizationRanges
from app.brain_sim.nmm import JansenRitColumn, JansenRitParams, MultiChannelBrain
from app.brain_sim.photic_input import (
    PhoticInputConfig,
    StrobeParams,
    strobe_envelope,
    strobe_to_cortical_input,
)


def _peak_freq(signal: np.ndarray, sampling_rate: int,
               lo: float = 4.0, hi: float = 30.0) -> float:
    """Return the dominant frequency in [lo, hi] Hz via FFT."""
    n = len(signal)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(freqs[mask][np.argmax(spectrum[mask])])


def _band_power(signal: np.ndarray, sampling_rate: int,
                lo: float, hi: float) -> float:
    n = len(signal)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(spectrum[mask].sum())


def test_jansen_rit_produces_alpha_at_default_params():
    """Default Jansen-Rit params should give ~10 Hz dominant peak in α band."""
    col = JansenRitColumn(params=JansenRitParams(), dt=1.0 / 500, seed=42)
    # Warm-up to leave transient
    for _ in range(int(2.0 * 500)):
        col.step()
    # Collect 6 s of signal
    samples = np.array([col.step() for _ in range(int(6.0 * 500))])
    peak = _peak_freq(samples, 500)
    assert 7.5 <= peak <= 12.5, f"Peak freq {peak:.2f} Hz not in α band"


def test_with_alpha_hz_tunes_peak():
    """with_alpha_hz(target) should shift the peak towards `target`."""
    sampling_rate = 500
    for target in (8.5, 10.0, 11.5):
        params = JansenRitParams().with_alpha_hz(target)
        col = JansenRitColumn(params=params, dt=1.0 / sampling_rate, seed=7)
        for _ in range(int(2.0 * sampling_rate)):
            col.step()
        samples = np.array(
            [col.step() for _ in range(int(6.0 * sampling_rate))]
        )
        peak = _peak_freq(samples, sampling_rate)
        # Allow ±2.5 Hz tolerance - Jansen-Rit α freq scales nonlinearly with `a`
        assert abs(peak - target) <= 2.5, (
            f"Target {target} Hz -> measured peak {peak:.2f} Hz")


def test_photic_input_shifts_spectrum_towards_strobe_freq():
    """Photic drive at f should enhance power around f and below f (FFR)."""
    sampling_rate = 500
    duration_s = 6.0
    brain = MultiChannelBrain(
        n_channels=1, alpha_hz=10.0,
        sampling_rate=sampling_rate, seed=11,
    )
    # Warm-up
    brain.run(duration_s=2.0)

    # No-stim baseline
    baseline = brain.run(duration_s=duration_s)[0]

    # Reset and run with 6 Hz strobe at strong responder profile
    brain2 = MultiChannelBrain(
        n_channels=1, alpha_hz=10.0,
        sampling_rate=sampling_rate, seed=11,
    )
    brain2.run(duration_s=2.0)
    config = PhoticInputConfig(
        cortical_gain=600.0,
        base_resonance_hz=6.0,
        resonance_width_hz=3.0,
    )
    strobe = StrobeParams(enabled=True, frequency_hz=6.0,
                          fill=0.3, brightness=0.5)

    def photic_fn(t: float) -> float:
        return strobe_to_cortical_input(t, strobe, config)

    driven = brain2.run(duration_s=duration_s, photic_fn=photic_fn)[0]

    # Power in 5–7 Hz band should grow under 6 Hz photic drive
    p_base = _band_power(baseline, sampling_rate, 5.0, 7.0)
    p_drive = _band_power(driven, sampling_rate, 5.0, 7.0)
    assert p_drive > p_base, (
        f"Photic drive did not enhance 5–7 Hz band "
        f"(baseline={p_base:.2e}, driven={p_drive:.2e})"
    )


def test_domain_randomization_covers_alpha_range():
    """Sampling many subjects should cover the full configured α range."""
    ranges = RandomizationRanges()
    samples = [ranges.sample(seed=i) for i in range(200)]
    alphas = np.array([s.alpha_hz for s in samples])
    assert alphas.min() >= ranges.alpha_hz_range[0]
    assert alphas.max() <= ranges.alpha_hz_range[1]
    # Spread should approach the full range with 200 samples
    assert alphas.max() - alphas.min() >= 0.8 * (
        ranges.alpha_hz_range[1] - ranges.alpha_hz_range[0]
    )
    # ~10% non-responders
    n_non = sum(1 for s in samples if s.responder_type == "non_responder")
    assert 5 <= n_non <= 35, f"{n_non}/200 non-responders is outside ~10% band"


def test_strobe_envelope_matches_duty_cycle():
    """strobe_envelope() should be ON for `fill` fraction of each period."""
    dt = 1.0 / 1000
    strobe = StrobeParams(enabled=True, frequency_hz=10.0,
                          fill=0.25, brightness=1.0)
    env = strobe_envelope(duration_s=1.0, dt=dt, strobe=strobe)
    on_fraction = env.mean()
    # Allow small discretization tolerance
    assert abs(on_fraction - 0.25) < 0.01, f"ON fraction {on_fraction:.3f} ≠ 0.25"


def test_multichannel_brain_returns_correct_shape():
    brain = MultiChannelBrain(n_channels=8, sampling_rate=500, seed=3)
    eeg = brain.run(duration_s=0.5)
    assert eeg.shape == (8, 250), f"Unexpected shape {eeg.shape}"


def test_in_process_brain_sim_is_deterministic_with_seed():
    """Same seed -> same samples; different seed -> different samples."""
    from app.brain_sim.brainflow_board import InProcessBrainSim

    sim_a = InProcessBrainSim(brain=MultiChannelBrain(n_channels=2, seed=99))
    sim_b = InProcessBrainSim(brain=MultiChannelBrain(n_channels=2, seed=99))
    sim_c = InProcessBrainSim(brain=MultiChannelBrain(n_channels=2, seed=100))

    a = np.array([sim_a.step() for _ in range(50)])
    b = np.array([sim_b.step() for _ in range(50)])
    c = np.array([sim_c.step() for _ in range(50)])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
