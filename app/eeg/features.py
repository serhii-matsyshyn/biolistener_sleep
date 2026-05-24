"""
Manual feature extraction - primary observation pipeline for the RL agent.

Uses BrainFlow's ``DataFilter`` for filtering / PSD / band-power
(no MNE on the live path). All features are explicit and citable:

- Band powers at the primary EEG channel (typically O1 for occipital α,
  or Fpz for frontal α), with custom hypnagogic-θ band (4–7 Hz) and
  spindle σ band (12–16 Hz) on top of the standard δ/θ/α/β/γ.
- θ/α ratio - well-known drowsiness biomarker
- EMG RMS - broadband (20–200 Hz) high-frequency activity proxy
- EOG energy - high-frequency oscillation in front polar channels
- Railed-percentage check for electrode-disconnect detection

The exact filter chain mirrors the user's existing BioListener
visualization script (already validated on real BioListener data):
    detrend(CONSTANT) -> bandpass(3–45, Butterworth zero-phase, order=2)
    -> bandstop(48–52)   (EU grid)
    -> bandstop(58–62)   (US grid)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
    WindowOperations,
)

from app.eeg.montage import Montage

logger = logging.getLogger("eeg.features")


# Frequency bands (Hz)
BAND_DELTA = (1.0, 4.0)
BAND_THETA = (4.0, 8.0)
BAND_HYPNAGOGIC_THETA = (4.0, 7.0)   # proposal §3.B - drowsy θ
BAND_ALPHA = (8.0, 13.0)
BAND_SPINDLE_SIGMA = (12.0, 16.0)    # sleep-spindle band
BAND_BETA = (13.0, 30.0)
BAND_GAMMA = (30.0, 45.0)


@dataclass
class FeatureSnapshot:
    """One per control-loop tick - what the agent sees."""
    t_ms: int
    alpha_pwr: float                  # mean α power over primary EEG channels
    theta_pwr: float
    beta_pwr: float
    hypnagogic_theta_pwr: float       # 4–7 Hz, drowsy θ
    sigma_pwr: float                  # 12–16 Hz, sleep spindles
    theta_alpha_ratio: float          # drowsiness biomarker
    emg_rms: float                    # broadband 20–45 Hz energy proxy
    eog_energy: float                 # variance across EOG channels
    railed_pct: float                 # 0..100 - electrode-disconnect signal
    n_samples: int                    # window length used


def _safe_band_power(psd: tuple, lo: float, hi: float) -> float:
    try:
        return float(DataFilter.get_band_power(psd, lo, hi))
    except Exception as err:    # BrainFlow raises if PSD too short etc.
        logger.debug("get_band_power(%g,%g) failed: %s", lo, hi, err)
        return 0.0


def _filter_chain(channel: np.ndarray, sampling_rate: int) -> np.ndarray:
    """User-validated chain (mirrors the existing BioListener viz script)."""
    DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(channel, sampling_rate, 3.0, 45.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    DataFilter.perform_bandstop(channel, sampling_rate, 48.0, 52.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    DataFilter.perform_bandstop(channel, sampling_rate, 58.0, 62.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    return channel


def extract_features(
    data: np.ndarray,
    sampling_rate: int,
    montage: Montage,
    t_ms: int = 0,
) -> FeatureSnapshot:
    """
    Compute the per-tick observation vector.

    Parameters
    ----------
    data:
        2-D array shape (channels, samples). Channels are indexed as in
        the BoardShim layout - use ``BoardShim.get_eeg_channels(board_id)``
        to pick the rows, then pass the sliced rows as ``data``.
    sampling_rate:
        EEG sampling rate in Hz (e.g., 500 for BIOLISTENER default).
    montage:
        Drives which rows of ``data`` are primary EEG, EMG, EOG.
    """
    n_samples = data.shape[1]
    if n_samples < int(sampling_rate * 0.5):
        # Less than 0.5s - PSD is meaningless; return zeros + railed flag
        logger.debug("Window too short (%d samples) - returning zero features",
                     n_samples)
        return FeatureSnapshot(
            t_ms=t_ms, alpha_pwr=0.0, theta_pwr=0.0, beta_pwr=0.0,
            hypnagogic_theta_pwr=0.0, sigma_pwr=0.0,
            theta_alpha_ratio=0.0, emg_rms=0.0, eog_energy=0.0,
            railed_pct=0.0, n_samples=n_samples,
        )

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    if n_samples < nfft:
        # Pad-or-clamp: if user gave us less than one FFT window, downsize nfft
        nfft = 1 << (n_samples - 1).bit_length() >> 1
        nfft = max(nfft, 64)

    alpha_vals, theta_vals, beta_vals = [], [], []
    hyp_vals, sigma_vals, railed_vals = [], [], []

    for ch_idx in montage.primary_eeg_channels:
        ch = data[ch_idx].copy().astype(np.float64)
        railed_vals.append(_railed_percentage(ch, sampling_rate))
        _filter_chain(ch, sampling_rate)
        psd = DataFilter.get_psd_welch(
            ch, nfft, nfft // 2, sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value,
        )
        alpha_vals.append(_safe_band_power(psd, *BAND_ALPHA))
        theta_vals.append(_safe_band_power(psd, *BAND_THETA))
        beta_vals.append(_safe_band_power(psd, *BAND_BETA))
        hyp_vals.append(_safe_band_power(psd, *BAND_HYPNAGOGIC_THETA))
        sigma_vals.append(_safe_band_power(psd, *BAND_SPINDLE_SIGMA))

    alpha_pwr = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    theta_pwr = float(np.mean(theta_vals)) if theta_vals else 0.0
    beta_pwr = float(np.mean(beta_vals)) if beta_vals else 0.0
    hyp_theta = float(np.mean(hyp_vals)) if hyp_vals else 0.0
    sigma_pwr = float(np.mean(sigma_vals)) if sigma_vals else 0.0
    railed_pct = float(np.mean(railed_vals)) if railed_vals else 0.0
    ta_ratio = theta_pwr / (alpha_pwr + 1e-12)

    # EMG RMS - broadband HF energy across EMG channels.
    # Falls back to gamma-band energy on primary EEG if no dedicated EMG.
    emg_rms = _emg_rms(data, montage, sampling_rate, nfft)

    # EOG energy - variance across EOG channels (eye-blink detection)
    eog_energy = 0.0
    if montage.eog_channels:
        eog_data = np.stack([data[c] for c in montage.eog_channels])
        eog_energy = float(np.var(eog_data))

    return FeatureSnapshot(
        t_ms=t_ms,
        alpha_pwr=alpha_pwr, theta_pwr=theta_pwr, beta_pwr=beta_pwr,
        hypnagogic_theta_pwr=hyp_theta, sigma_pwr=sigma_pwr,
        theta_alpha_ratio=ta_ratio,
        emg_rms=emg_rms, eog_energy=eog_energy,
        railed_pct=railed_pct,
        n_samples=n_samples,
    )


def _emg_rms(data: np.ndarray, montage: Montage,
             sampling_rate: int, nfft: int) -> float:
    """
    EMG RMS estimate.

    If montage has dedicated EMG channels: compute RMS of raw signal after
    a high-pass filter at 20 Hz (standard EMG isolation).
    Otherwise: fall back to γ-band power (30–45 Hz) on the primary EEG
    channel, which is dominated by muscle artifact when present.
    """
    if montage.emg_channels:
        rms_vals = []
        for ch_idx in montage.emg_channels:
            ch = data[ch_idx].copy().astype(np.float64)
            DataFilter.detrend(ch, DetrendOperations.CONSTANT.value)
            DataFilter.perform_highpass(
                ch, sampling_rate, 20.0, 4,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0,
            )
            rms_vals.append(float(np.sqrt(np.mean(ch ** 2))))
        return float(np.mean(rms_vals))
    # Fallback: γ-band on primary EEG
    if not montage.primary_eeg_channels:
        return 0.0
    ch = data[montage.primary_eeg_channels[0]].copy().astype(np.float64)
    _filter_chain(ch, sampling_rate)
    psd = DataFilter.get_psd_welch(
        ch, nfft, nfft // 2, sampling_rate,
        WindowOperations.BLACKMAN_HARRIS.value,
    )
    return _safe_band_power(psd, *BAND_GAMMA)


def _railed_percentage(channel: np.ndarray, sampling_rate: int) -> float:
    """Wrapper around BrainFlow's get_railed_percentage. Returns 0..100."""
    try:
        return float(DataFilter.get_railed_percentage(
            channel, sampling_rate, sampling_rate,
        ))
    except Exception:
        # Older BrainFlow versions may lack this - fall back to manual estimate
        amax = float(np.abs(channel).max() + 1e-12)
        clipped = np.sum(np.abs(channel) > 0.95 * amax)
        return 100.0 * clipped / len(channel)


def feature_vector(snap: FeatureSnapshot) -> np.ndarray:
    """
    Convert FeatureSnapshot -> fixed-length numeric vector for the RL agent
    observation. Order is stable for the lifetime of the project - if you
    add a feature, append it; don't reorder.
    """
    return np.array([
        snap.alpha_pwr,
        snap.theta_pwr,
        snap.beta_pwr,
        snap.hypnagogic_theta_pwr,
        snap.sigma_pwr,
        snap.theta_alpha_ratio,
        snap.emg_rms,
        snap.eog_energy,
        snap.railed_pct,
    ], dtype=np.float32)


FEATURE_NAMES: List[str] = [
    "alpha_pwr", "theta_pwr", "beta_pwr",
    "hypnagogic_theta_pwr", "sigma_pwr",
    "theta_alpha_ratio", "emg_rms", "eog_energy", "railed_pct",
]
