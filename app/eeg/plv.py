"""
Phase-Locking Value - entrainment fidelity metric.

PLV measures how consistently the phase difference between two signals
stays constant over a window. For our use case:

- signal A = the strobe ON/OFF envelope (sampled at EEG sample rate)
- signal B = the primary EEG channel (band-pass filtered around the
  strobe frequency)

If the brain is following the strobe (SSVEP / photic driving), the
phase difference is stable -> PLV close to 1. If the brain is doing its
own thing, the phase difference walks -> PLV close to 0.

This does not cleanly separate entrainment from endogenous rhythm. It
measures phase locking between the EEG and the delivered strobe envelope,
and both are band-passed in a fixed 2 Hz window around the current
commanded frequency - so an endogenous rhythm that happens to sit at the
strobe frequency scores the same as driven activity. It is also not
comparable across arms with different ramp rates: at the ramp-limiter cap a
4 s window can sweep 2 Hz, attenuating both signals. Treat PLV as
"phase locking to the delivered stimulus".

Reference: Lachaux et al. 1999, "Measuring phase synchrony in brain
signals", Hum. Brain Mapp. 8(4):194-208.

Not available in BrainFlow's DataFilter, so implemented here in numpy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
)


def hilbert_phase(signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase via analytic signal (no scipy dependency)."""
    n = len(signal)
    if n == 0:
        return np.zeros(0)
    # FFT-based Hilbert: zero negative frequencies, double positives
    spectrum = np.fft.fft(signal)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    return np.angle(analytic)


def plv(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Phase-locking value in [0, 1].

    Parameters
    ----------
    signal_a, signal_b:
        Same-length 1-D arrays. Must already be filtered to the same
        narrow frequency band of interest.
    """
    if len(signal_a) != len(signal_b):
        raise ValueError("PLV requires equal-length signals")
    if len(signal_a) < 16:
        return 0.0
    phase_diff = hilbert_phase(signal_a) - hilbert_phase(signal_b)
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def plv_strobe_to_eeg(
    eeg_channel: np.ndarray,
    strobe_envelope: np.ndarray,
    sampling_rate: int,
    strobe_freq_hz: float,
    bandwidth_hz: float = 2.0,
) -> float:
    """
    PLV between the strobe envelope and the EEG, with both bandpass-filtered
    around the strobe frequency.

    Returns 0.0 if the strobe envelope is all-zero (strobe disabled) or
    arrays are too short.
    """
    if np.allclose(strobe_envelope, 0):
        return 0.0
    if strobe_freq_hz <= 0:
        return 0.0
    if len(eeg_channel) != len(strobe_envelope):
        # Truncate to common length
        n = min(len(eeg_channel), len(strobe_envelope))
        eeg_channel = eeg_channel[:n]
        strobe_envelope = strobe_envelope[:n]

    eeg = eeg_channel.copy().astype(np.float64)
    env = strobe_envelope.copy().astype(np.float64)

    lo = max(0.5, strobe_freq_hz - bandwidth_hz / 2)
    hi = min(sampling_rate / 2 - 0.5, strobe_freq_hz + bandwidth_hz / 2)
    if hi <= lo:
        return 0.0

    DataFilter.perform_bandpass(eeg, sampling_rate, lo, hi, 4,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    DataFilter.perform_bandpass(env, sampling_rate, lo, hi, 4,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    return plv(eeg, env)


def plv_strobe_harmonic(
    eeg_channel: np.ndarray,
    strobe_envelope: np.ndarray,
    sampling_rate: int,
    strobe_freq_hz: float,
    harmonic: int = 2,
    bandwidth_hz: float = 2.0,
) -> float:
    """
    PLV between EEG and strobe envelope at an integer harmonic of the
    strobe frequency.

    A square strobe with duty cycle != 50 % carries real power at 2f, and
    photic driving is known to produce harmonic responses, so entrainment
    can be present at 2f while the fundamental band looks quiet. The
    fundamental-only measurement in :func:`plv_strobe_to_eeg` cannot see
    that.

    Reporting f and 2f side by side also helps separate genuine neural
    entrainment from stimulus-locked contamination on real hardware: a
    photoelectric or photomyogenic artifact tracks the drive waveform's
    own harmonic structure, whereas a neural response need not.

    This is a **diagnostic** measure. It is deliberately kept out of the
    observation vector and the reward so that trained policies remain
    comparable across it.
    """
    if harmonic < 1:
        raise ValueError("harmonic must be >= 1")
    return plv_strobe_to_eeg(
        eeg_channel,
        strobe_envelope,
        sampling_rate,
        strobe_freq_hz * harmonic,
        bandwidth_hz=bandwidth_hz,
    )
