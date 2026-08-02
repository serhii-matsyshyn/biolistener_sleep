"""
Tests for the stimulus-artifact diagnostic.

The diagnostic answers "how much of the measured theta/alpha rise is the
strobe rather than the brain?". Two properties must hold for it to be
usable as evidence:

1. It must be *additive* - turning it on cannot change the observation
   vector, the reward, or therefore the policy's actions. Otherwise a
   notched rollout is a different experiment, not a re-measurement.
2. It must actually remove stimulus power, and the decoy control must
   remove a matched number of bins *without* removing stimulus power -
   otherwise the drop it reports is band loss, not contamination.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.eeg.features import (
    BAND_ALPHA,
    BAND_THETA,
    _decoy_centres,
    _excise_psd_bins,
    extract_features,
    feature_vector,
    stim_artifact_diagnostic,
)
from app.eeg.montage import OCCIPITAL_ALPHA

FS = 500


def _tone(freq_hz: float, amp: float = 3.0, seconds: float = 4.0,
          noise: float = 0.5, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * freq_hz * t) + noise * rng.normal(size=t.size)


# --- property 1: additivity -------------------------------------------------

def test_diagnostic_does_not_change_features_or_observation():
    """Same window, with and without the diagnostic -> identical features."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(8, 4 * FS)) * 10.0
    data[0] += _tone(6.0, amp=20.0, seed=1)

    plain = extract_features(data.copy(), FS, OCCIPITAL_ALPHA, t_ms=0)
    diag = extract_features(data.copy(), FS, OCCIPITAL_ALPHA, t_ms=0,
                            stim_notch_hz=6.0)

    assert plain.diagnostic is None
    assert diag.diagnostic is not None
    # Every production field is bit-identical.
    for field in ("alpha_pwr", "theta_pwr", "beta_pwr",
                  "hypnagogic_theta_pwr", "sigma_pwr", "theta_alpha_ratio",
                  "emg_rms", "eog_energy", "railed_pct"):
        assert getattr(plain, field) == getattr(diag, field), field
    np.testing.assert_array_equal(feature_vector(plain), feature_vector(diag))


def test_diagnostic_raw_channel_tracks_production_theta_alpha():
    """
    The diagnostic re-transforms at nfft=1024, and absolute band power is
    resolution-dependent for narrowband content, so ``_raw`` is NOT numerically
    equal to the production value. What must hold is that it is the same
    quantity - i.e. it moves with the production biomarker across windows.

    Consequence for the analysis: only compare diagnostic channels to each
    other (notch/raw, decoy/raw). Never mix a diagnostic absolute level with
    a production absolute level.
    """
    prod, diag = [], []
    for i, f_tone in enumerate(np.linspace(4.0, 12.0, 20)):
        rng = np.random.default_rng(100 + i)
        data = rng.normal(size=(8, 4 * FS)) * 10.0
        data[0] += _tone(float(f_tone), amp=15.0, seed=i)
        snap = extract_features(data, FS, OCCIPITAL_ALPHA, stim_notch_hz=6.0)
        prod.append(snap.theta_alpha_ratio)
        diag.append(snap.diagnostic["theta_alpha_raw"])
        assert snap.diagnostic["diag_nfft"] == 1024

    r = float(np.corrcoef(np.log(prod), np.log(diag))[0, 1])
    assert r > 0.95, r


# --- property 2: it removes the stimulus, the decoy removes only bins -------

def _diag(signal: np.ndarray, stim_hz: float) -> dict:
    return stim_artifact_diagnostic(signal.astype(np.float64), FS, stim_hz)


def test_notch_removes_tone_power_and_decoy_does_not():
    """A 6 Hz tone must lose theta power under the notch, not under the decoy."""
    d = _diag(_tone(6.0, amp=5.0, noise=0.3, seed=3), 6.0)

    assert d["theta_notch_narrow"] < d["theta_decoy_narrow"]
    assert d["theta_notch_wide"] < d["theta_decoy_wide"]
    # +/-1 bin captures ~94% of a pure tone, so most of theta must go.
    assert d["theta_notch_narrow"] < 0.4 * d["theta_raw"]
    assert d["theta_notch_wide"] < 0.1 * d["theta_raw"]


def test_decoy_is_a_clean_band_loss_control():
    """
    The decoy must remove bins without removing stimulus, so band power has
    to come out essentially unchanged. Its centres are therefore placed off
    the real peak's leakage skirt; interpolation anchors landing on that
    skirt make the control report large *negative* band loss.
    """
    d = _diag(_tone(6.0, amp=5.0, noise=0.3, seed=3), 6.0)
    for width in ("narrow", "wide"):
        band_loss = (d["theta_raw"] - d[f"theta_decoy_{width}"]) / d["theta_raw"]
        assert abs(band_loss) < 0.15, (width, band_loss)


def test_decoy_excises_same_number_of_bins_as_notch():
    """Band-loss control is only valid if the bin counts match."""
    d = _diag(_tone(5.0, seed=4), 5.0)
    assert d["n_bins_notch_narrow"] == d["n_bins_decoy_narrow"]
    assert d["n_bins_notch_wide"] == d["n_bins_decoy_wide"]
    assert d["n_bins_notch_wide"] > d["n_bins_notch_narrow"]


def test_decoy_centres_are_separated_from_every_harmonic():
    df = 500.0 / 1024.0
    true_centres = [5.0, 10.0, 15.0]
    for hw in (1, 2):
        decoys = _decoy_centres(true_centres, df, hw)
        assert len(decoys) == len(true_centres)
        sep = (2 * hw + 2) * df
        for dc in decoys:
            assert all(abs(dc - t) > sep for t in true_centres)
            assert 3.0 <= dc <= 45.0


def test_second_harmonic_contaminates_alpha_and_is_removed():
    """2f of a 5 Hz strobe lands at 10 Hz - inside the alpha denominator."""
    d = _diag(_tone(10.0, amp=5.0, noise=0.3, seed=5), 5.0)  # harmonic 2
    assert d["alpha_notch_narrow"] < d["alpha_decoy_narrow"]
    assert d["alpha_notch_narrow"] < 0.4 * d["alpha_raw"]


def test_notch_preserves_bands_holding_no_stimulus_harmonic():
    """Excision must not gut a band that holds no stimulus harmonic."""
    # 5 Hz strobe -> harmonics 5, 10, 15. Beta is 13-30 Hz; a 22 Hz tone
    # sits there untouched by any harmonic.
    d = _diag(_tone(22.0, amp=5.0, noise=0.3, seed=6), 5.0)
    assert d["beta_notch_narrow"] == pytest.approx(d["beta_raw"], rel=0.15)


# --- excision mechanics -----------------------------------------------------

def test_excise_interpolates_across_removed_bins():
    freq = np.arange(0, 50, 1.0)
    amp = np.ones_like(freq)
    amp[10] = 100.0                       # spike at 10 Hz
    out, n = _excise_psd_bins(amp, freq, [10.0], halfwidth_bins=0)
    assert n == 1
    assert out[10] == pytest.approx(1.0)  # interpolated from neighbours
    assert amp[10] == 100.0               # input untouched


def test_excise_out_of_range_centres_are_skipped():
    freq = np.arange(0, 50, 1.0)
    amp = np.ones_like(freq)
    out, n = _excise_psd_bins(amp, freq, [500.0, -3.0, np.nan], 1)
    assert n == 0
    np.testing.assert_array_equal(out, amp)


def test_diagnostic_resolution_gives_bands_enough_bins():
    """
    Documents why the diagnostic re-transforms instead of reusing the
    production PSD: at nfft=1024 theta spans ~8 bins, so excising +/-1 bin
    removes the stimulus without removing the band.
    """
    df = FS / 1024.0
    n_theta_bins = (BAND_THETA[1] - BAND_THETA[0]) / df
    n_alpha_bins = (BAND_ALPHA[1] - BAND_ALPHA[0]) / df
    assert n_theta_bins >= 8
    assert n_alpha_bins >= 10
    # Production resolution would give theta only ~4 bins.
    assert (BAND_THETA[1] - BAND_THETA[0]) / (FS / 512.0) < 5
