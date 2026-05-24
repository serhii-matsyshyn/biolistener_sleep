"""
Map strobe parameters -> cortical photic-drive current for the NMM.

Models three effects:
  1. Eyelid spectral attenuation - closed-eye transmission is highly
     wavelength-dependent. Bierman (2008) and Robinson & Kertesz (1974)
     report ~98% attenuation at red wavelengths, ~70% at blue, ~85% at green.
  2. Strobe envelope -> effective cortical input. A square strobe at
     frequency f with duty cycle d delivers a periodic input whose
     amplitude scales with brightness and (1 - eyelid_attenuation).
  3. **State-dependent resonance** - in the awake α-dominant state the
     cortex entrains best near individual α (~10 Hz). As drowsiness rises
     (θ/α ratio grows), thalamocortical neurons shift from tonic into a
     burst-firing mode whose preferred entrainment frequency drops into
     θ/δ bands (Steriade et al., Bal & McCormick 1996, Hindriks & van
     Putten 2013). We model this as a downward shift of the resonance
     peak proportional to current drowsiness, clipped to θ-low.

This is the simplest physiologically defensible mapping: amplitude-
modulated input current to the EI population with a state-dependent
frequency-response curve. SSVEP / photic-driving response then emerges
from the Jansen-Rit dynamics naturally (Spiegler et al. 2011).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Bierman 2008 closed-eyelid spectral transmission (approximate)
EYELID_TRANSMISSION = {
    "red":   0.02,   # ~98% attenuated
    "green": 0.15,
    "blue":  0.30,   # ~70% attenuated
    "white": 0.10,   # broadband, dominated by red attenuation
}


@dataclass
class StrobeParams:
    """Mirror of the mask's UDP/JSON CMD_CHANGE_PARAMS fields."""
    enabled: bool = False
    frequency_hz: float = 10.0
    fill: float = 0.2          # duty cycle 0..1
    brightness: float = 0.3    # 0..1
    color: str = "white"       # "red" | "green" | "blue" | "white"

    def effective_amplitude(self) -> float:
        """Brightness * (light reaching retina through closed eyelid)."""
        if not self.enabled:
            return 0.0
        transmission = EYELID_TRANSMISSION.get(self.color, 0.10)
        return self.brightness * transmission


@dataclass
class PhoticInputConfig:
    """
    Per-subject responsiveness to photic stimulation.

    ``cortical_gain`` is the multiplier from effective_amplitude -> input
    current on the EI population. Real inter-individual variability is
    large - some people are strong SSVEP responders, others are
    non-responders (proposal §4.B "Subjective Variability"). Domain
    randomization samples this widely.

    The **frequency response** is a Gaussian centered on the effective
    resonance peak, which is a function of:

    * ``base_resonance_hz`` - the awake-state peak (typically the
      subject's individual α frequency).
    * ``current_drowsiness`` - mutable; the env updates it each control
      tick to the current θ/α ratio. As drowsiness rises, the peak
      shifts down per ``shift_per_unit_ratio`` (clamped to
      [``min_resonance_hz``, ``base_resonance_hz``]).

    This implements the thalamocortical mode-switch physiology - the
    frequency the cortex entrains to most strongly drops as the brain
    transitions wake -> N1 (≈10 -> 4 Hz, the trajectory the RL agent must
    learn to track).
    """
    cortical_gain: float = 400.0
    base_resonance_hz: float = 10.0          # awake-state peak
    resonance_width_hz: float = 4.0          # σ (widened from 3 -> 4)
    # State-dependent shift parameters
    shift_per_unit_ratio: float = 2.0        # Hz per unit (θ/α − baseline)
    shift_baseline_ratio: float = 0.5        # θ/α below which no shift
    min_resonance_hz: float = 4.0            # floor on shifted peak
    # Mutable state - updated by env after each control-tick feature extraction
    current_drowsiness: float = 0.0          # current θ/α ratio

    @property
    def effective_resonance_hz(self) -> float:
        """Resonance peak this *tick*, after state-dependent shift."""
        excess = max(0.0, self.current_drowsiness - self.shift_baseline_ratio)
        shift = excess * self.shift_per_unit_ratio
        peak = self.base_resonance_hz - shift
        return max(self.min_resonance_hz, peak)


def strobe_to_cortical_input(
    t: float,
    strobe: StrobeParams,
    config: PhoticInputConfig,
) -> float:
    """Instantaneous cortical drive at time ``t`` for a square-wave strobe."""
    if not strobe.enabled or strobe.brightness <= 0.0:
        return 0.0
    period = 1.0 / max(strobe.frequency_hz, 1e-6)
    phase = (t % period) / period
    on = 1.0 if phase < strobe.fill else 0.0
    amp = strobe.effective_amplitude()
    # Frequency-dependent gain - peaks near the *current* effective resonance,
    # which moves toward θ as the brain becomes drowsy.
    peak = config.effective_resonance_hz
    df = strobe.frequency_hz - peak
    resonance = np.exp(-(df ** 2) / (2.0 * config.resonance_width_hz ** 2))
    return on * amp * config.cortical_gain * float(resonance)


def strobe_envelope(
    duration_s: float,
    dt: float,
    strobe: StrobeParams,
) -> np.ndarray:
    """
    Generate the strobe ON/OFF envelope (0 or 1) sampled at dt.

    Used by PLV computation in app/eeg/plv.py - the EEG must be phase-locked
    to THIS envelope (not the raw frequency) to count as entrainment.
    """
    n = int(round(duration_s / dt))
    if not strobe.enabled or strobe.brightness <= 0.0:
        return np.zeros(n, dtype=np.float64)
    t = np.arange(n) * dt
    period = 1.0 / max(strobe.frequency_hz, 1e-6)
    phase = (t % period) / period
    return (phase < strobe.fill).astype(np.float64)


def vigilance_drift(
    t: float,
    duration_s: float,
    start_arousal: float = 1.0,
    end_arousal: float = 0.3,
) -> Tuple[float, float]:
    """
    Background drift mimicking falling-asleep over a session.

    Returns (p_mean_multiplier, p_std_multiplier) for the NMM. Arousal drops
    monotonically over the session, modeling reduced ascending reticular
    activation (Hindriks & van Putten 2013-style sleep transition).

    Wired in `InProcessBrainSim.step()` so the brain's background drive
    decays over the session, organically raising θ/α. This is the
    "natural drowsiness drift" the agent both fights against and leverages -
    its job is to accelerate the drift via well-timed entrainment.
    """
    progress = min(1.0, max(0.0, t / max(duration_s, 1e-6)))
    arousal = start_arousal + (end_arousal - start_arousal) * progress
    return arousal, arousal
