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

Filter chain, applied identically on the training and deployment paths:
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
BAND_HYPNAGOGIC_THETA = (4.0, 7.0)   # drowsy θ
BAND_ALPHA = (8.0, 13.0)
BAND_SPINDLE_SIGMA = (12.0, 16.0)    # sleep-spindle band
BAND_BETA = (13.0, 30.0)
BAND_GAMMA = (30.0, 45.0)

# Amplifier gain assumed when the caller does not say. Matches the value
# `configs/daemon_prod.yaml` sends to the device, so the railed detector
# reconstructs the right ADC full scale.
DEFAULT_PGA_GAIN = 8


# --- stimulus-artifact diagnostic -------------------------------------------
# Harmonics of the strobe fundamental that are excised from the PSD when the
# diagnostic is requested. A square strobe with duty 0.2-0.4 puts substantial
# power in 2f and 3f, and for a descending policy (f ~ 4-6 Hz) 2f lands
# squarely in the alpha band - i.e. the *denominator* of theta/alpha is
# contaminated as well as the numerator.
STIM_NOTCH_HARMONICS: tuple = (1, 2, 3)

# The diagnostic computes its own PSD at finer resolution than the production
# one. At the production nfft (512 @ 500 Hz, bin width ~0.98 Hz) the 4-8 Hz
# theta band is only ~4 bins wide, and a Blackman-Harris main lobe spans ~3
# bins, so the stimulus peak cannot be separated from the band it sits in.
# At nfft=1024 (bin width ~0.49 Hz) theta is ~8 bins, excising +/-1 bin
# captures ~94% of a pure tone.
# Welch still averages ~3 segments over the 4 s window at this size.
DIAG_NFFT_MAX = 1024

# Excision half-widths in bins, reported as a pair so the contamination
# estimate is bracketed rather than asserted at one arbitrary width.
# 1 bin (+/-0.49 Hz) captures ~94% of a tone; 2 bins (+/-0.98 Hz) ~100%.
DIAG_HALFWIDTHS = {"narrow": 1, "wide": 2}

# Frequency range the decoy centres are allowed to occupy - the passband of
# the production filter chain, so decoy bins live in the same spectral
# territory as the real ones.
DIAG_DECOY_RANGE_HZ = (3.0, 45.0)


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
    # Optional stimulus-artifact diagnostic. ``None`` unless
    # ``extract_features`` was called with ``stim_notch_hz``. Never enters
    # :func:`feature_vector`, so enabling it cannot change what the agent
    # observes or what the reward pays - it is a parallel measurement
    # channel, not a replacement pipeline.
    diagnostic: Optional[dict] = None


def _safe_band_power(psd: tuple, lo: float, hi: float) -> float:
    try:
        return float(DataFilter.get_band_power(psd, lo, hi))
    except Exception as err:    # BrainFlow raises if PSD too short etc.
        logger.debug("get_band_power(%g,%g) failed: %s", lo, hi, err)
        return 0.0


def _excise_psd_bins(
    amp: np.ndarray,
    freq: np.ndarray,
    centers_hz,
    halfwidth_bins: int,
) -> tuple:
    """
    Replace PSD bins around each ``centers_hz`` by linear interpolation.

    Spectral excision rather than time-domain filtering: at nfft=512 /
    fs=500 the bin width is ~0.98 Hz, so the theta band (4-8 Hz) is only
    about five bins wide and a time-domain bandstop of any usable order
    would take out most of the band. Interpolating across the stimulus
    bins removes the narrow driven peak while leaving the surrounding
    band intact.

    Returns ``(amp_excised, n_bins_excised)``. ``amp`` is not modified.
    """
    a = np.array(amp, dtype=np.float64, copy=True)
    f = np.asarray(freq, dtype=np.float64)
    n = a.size
    if n == 0:
        return a, 0

    f_max = float(f[-1])
    mask = np.zeros(n, dtype=bool)
    for c in centers_hz:
        c = float(c)
        if not np.isfinite(c) or c <= 0.0 or c > f_max:
            continue
        i = int(np.argmin(np.abs(f - c)))
        lo = max(0, i - halfwidth_bins)
        hi = min(n - 1, i + halfwidth_bins)
        mask[lo:hi + 1] = True

    if not mask.any():
        return a, 0

    # Interpolate across each contiguous run of excised bins.
    idx = np.flatnonzero(mask)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        lo, hi = int(run[0]), int(run[-1])
        left, right = lo - 1, hi + 1
        if left < 0 and right >= n:
            continue                      # whole spectrum excised - give up
        if left < 0:
            a[lo:hi + 1] = a[right]
        elif right >= n:
            a[lo:hi + 1] = a[left]
        else:
            a[lo:hi + 1] = np.linspace(a[left], a[right], hi - lo + 3)[1:-1]

    return a, int(mask.sum())


def _band_set(psd: tuple, suffix: str) -> dict:
    """All reported band powers (+ θ/α) from one PSD, keys suffixed."""
    alpha = _safe_band_power(psd, *BAND_ALPHA)
    theta = _safe_band_power(psd, *BAND_THETA)
    return {
        f"alpha{suffix}": alpha,
        f"theta{suffix}": theta,
        f"hypno_theta{suffix}": _safe_band_power(psd, *BAND_HYPNAGOGIC_THETA),
        f"sigma{suffix}": _safe_band_power(psd, *BAND_SPINDLE_SIGMA),
        f"beta{suffix}": _safe_band_power(psd, *BAND_BETA),
        f"theta_alpha{suffix}": theta / (alpha + 1e-12),
    }


def _decoy_centres(
    true_centres,
    df: float,
    halfwidth_bins: int,
    lo_hz: float = DIAG_DECOY_RANGE_HZ[0],
    hi_hz: float = DIAG_DECOY_RANGE_HZ[1],
) -> list:
    """
    Stimulus-free frequencies for the band-loss control, one per harmonic.

    Each decoy is placed by searching outward from its harmonic for the first
    slot separated from every true centre - and from every decoy already
    placed - by more than one excision width, so no decoy's interpolation
    anchors sit inside a real peak's leakage skirt.
    """
    sep = (2 * halfwidth_bins + 2) * df
    placed: list = []
    for c in true_centres:
        chosen = None
        for step in range(1, 12):
            for sign in (1, -1):
                cand = c + sign * step * sep
                if not (lo_hz <= cand <= hi_hz):
                    continue
                if any(abs(cand - t) <= sep for t in true_centres):
                    continue
                if any(abs(cand - p) <= sep for p in placed):
                    continue
                chosen = cand
                break
            if chosen is not None:
                break
        if chosen is not None:
            placed.append(chosen)
    return placed


def stim_artifact_diagnostic(
    channel: np.ndarray,
    sampling_rate: int,
    stim_hz: float,
    harmonics=STIM_NOTCH_HARMONICS,
) -> dict:
    """
    Band powers with and without the stimulus peaks.

    ``channel`` must already have passed through :func:`_filter_chain`, so
    the only difference between this and the production path is the FFT
    size and the excised bins.

    Three channels are reported per band, at two excision widths:

    ``_raw``      nothing removed (at the diagnostic's own resolution)
    ``_notch_*``  stimulus harmonics excised
    ``_decoy_*``  the same number of bins excised at stimulus-free
                  frequencies - the band-loss control

    Contamination attributable to the stimulus is ``decoy - notch``: the
    raw-minus-notch drop includes power lost simply by deleting bins, and
    subtracting the decoy removes that. ``narrow``/``wide`` bracket the
    estimate, since one bin captures ~94% of a tone and two ~100%.
    """
    n = int(np.asarray(channel).size)
    nfft = 1 << max(6, (n.bit_length() - 1))     # largest power of two <= n
    nfft = min(nfft, DIAG_NFFT_MAX, n)
    psd = DataFilter.get_psd_welch(
        np.asarray(channel, dtype=np.float64).copy(),
        nfft, nfft // 2, sampling_rate,
        WindowOperations.BLACKMAN_HARRIS.value,
    )
    amp, freq = psd[0], psd[1]
    df = float(np.asarray(freq)[1] - np.asarray(freq)[0])
    true_centres = [h * stim_hz for h in harmonics]

    out: dict = {"stim_notch_hz": float(stim_hz), "diag_nfft": float(nfft)}
    out.update(_band_set(psd, "_raw"))

    for width_name, hw in DIAG_HALFWIDTHS.items():
        decoys = _decoy_centres(true_centres, df, hw)
        a_true, n_true = _excise_psd_bins(amp, freq, true_centres, hw)
        a_decoy, n_decoy = _excise_psd_bins(amp, freq, decoys, hw)
        out.update(_band_set((a_true, freq), f"_notch_{width_name}"))
        out.update(_band_set((a_decoy, freq), f"_decoy_{width_name}"))
        out[f"n_bins_notch_{width_name}"] = float(n_true)
        out[f"n_bins_decoy_{width_name}"] = float(n_decoy)
    return out


def _filter_chain(channel: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Detrend, bandpass 3-45 Hz, notch both mains frequencies."""
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
    stim_notch_hz: Optional[float] = None,
    pga_gain: int = DEFAULT_PGA_GAIN,
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
    stim_notch_hz:
        When given, additionally computes the stimulus-artifact diagnostic
        (see :func:`stim_artifact_diagnostic`) from the same PSD and stores
        it on ``FeatureSnapshot.diagnostic``. Purely additive: the returned
        band powers, θ/α and :func:`feature_vector` output are byte-identical
        to a call without it, so enabling the diagnostic cannot change agent
        behaviour.
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
        # Fewer samples than one FFT window: downsize nfft rather than pad.
        nfft = 1 << (n_samples - 1).bit_length() >> 1
        nfft = max(nfft, 64)

    alpha_vals, theta_vals, beta_vals = [], [], []
    hyp_vals, sigma_vals, railed_vals = [], [], []
    diag_vals: List[dict] = []

    for ch_idx in montage.primary_eeg_channels:
        ch = data[ch_idx].copy().astype(np.float64)
        railed_vals.append(_railed_percentage(ch, pga_gain))
        _filter_chain(ch, sampling_rate)
        psd = DataFilter.get_psd_welch(
            ch, nfft, nfft // 2, sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value,
        )
        if stim_notch_hz is not None:
            # Same filtered channel the production PSD comes from, but the
            # diagnostic re-transforms it at finer resolution (see
            # DIAG_NFFT_MAX). Its ``_raw`` channel is therefore the correct
            # internal baseline for its own notched channels, and is close
            # to - not identical with - the production band powers above.
            diag_vals.append(
                stim_artifact_diagnostic(ch, sampling_rate, stim_notch_hz)
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

    diagnostic = None
    if diag_vals:
        # Mean across primary channels, matching how the band powers above
        # are pooled. Bin counts are identical per channel, so mean == value.
        diagnostic = {
            k: float(np.mean([d[k] for d in diag_vals]))
            for k in diag_vals[0]
        }

    return FeatureSnapshot(
        t_ms=t_ms,
        alpha_pwr=alpha_pwr, theta_pwr=theta_pwr, beta_pwr=beta_pwr,
        hypnagogic_theta_pwr=hyp_theta, sigma_pwr=sigma_pwr,
        theta_alpha_ratio=ta_ratio,
        emg_rms=emg_rms, eog_energy=eog_energy,
        railed_pct=railed_pct,
        n_samples=n_samples,
        diagnostic=diagnostic,
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


# Reading at or above this fraction of full scale on the *uncorrected* signal
# means the offset has consumed the converter and clipping is either happening
# or imminent - the point at which the DC component stops being harmless.
SATURATION_PCT = 95.0


def _railed_percentage(channel: np.ndarray, pga_gain: int = DEFAULT_PGA_GAIN) -> float:
    """How much of the converter's range the *signal* needs, 0..100.

    ``get_railed_percentage(data, gain)`` reconstructs ADC full scale from the
    amplifier gain, so it must be the gain the device is running at. It reports
    ``max|x|`` against that scale, which on a real head is almost entirely the
    electrode's half-cell potential: measured at 15.7 % offset against 0.3 %
    signal, with 84 % of the range still free. That potential drifts by tens of
    millivolts over a session, is removed completely by the 3-45 Hz chain every
    feature is computed from, and says nothing about whether the recording is
    usable. Scoring it as though it did aborts good sessions.

    So the offset is subtracted first and the answer describes the signal. Two
    failures still have to register, and do: an electrode that floats and swings
    is large after subtraction, and one pinned against a rail is caught by
    ``SATURATION_PCT`` on the uncorrected trace, where subtracting the offset
    would otherwise hide it as a flat line.
    """
    try:
        x = np.ascontiguousarray(np.asarray(channel, dtype=np.float64))
        gain = int(pga_gain)
        # Median, not mean: a run of clipped samples must not drag the
        # reference point towards the rail it is meant to detect.
        ac = np.ascontiguousarray(x - np.median(x))
        signal = float(DataFilter.get_railed_percentage(ac, gain))
        headroom = float(DataFilter.get_railed_percentage(x.copy(), gain))
        if headroom >= SATURATION_PCT:
            return max(signal, headroom)
        return signal
    except Exception:
        # Fallback is scale-free and reads near zero on any signal, so a
        # silent fall-through here would disable every railed check
        # downstream. Log it rather than substituting a plausible number.
        logger.warning("get_railed_percentage unavailable - railed checks "
                       "are running on a scale-free approximation",
                       exc_info=True)
        amax = float(np.abs(channel).max() + 1e-12)
        clipped = np.sum(np.abs(channel) > 0.95 * amax)
        return 100.0 * clipped / len(channel)


def feature_vector(snap: FeatureSnapshot) -> np.ndarray:
    """
    Convert FeatureSnapshot -> fixed-length numeric vector for the RL agent
    observation. Order is stable for the lifetime of the project - if you
    add a feature, append it; the order never changes.
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
