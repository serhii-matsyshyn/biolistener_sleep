"""
Map strobe parameters -> cortical photic-drive current for the NMM.

Models three effects:
  1. Eyelid attenuation - only a fraction of the mask's light reaches the
     retina through a closed lid. Modelled as a single broadband scalar
     (see ``EYELID_TRANSMISSION``); the mask is driven white-only, so no
     per-wavelength model is needed.
  2. Strobe envelope -> effective cortical input. A square strobe at
     frequency f with duty cycle d delivers a periodic input whose
     amplitude scales with brightness and the eyelid transmission.
  3. **State-dependent resonance** - in the awake α-dominant state the
     cortex entrains best near individual α (~10 Hz). As drowsiness rises
     (θ/α ratio grows), thalamocortical neurons shift from tonic into a
     burst-firing mode. We model this as a downward shift of the photic
     resonance peak proportional to current drowsiness, clipped to θ-low.

     The tonic/burst switch is well established:
       Steriade, McCormick & Sejnowski (1993) Science 262:679-685,
         doi:10.1126/science.8235588
       Bal & McCormick (1996) Neuron 17:297-308,
         doi:10.1016/S0896-6273(00)80161-0
       Hindriks & van Putten (2013) NeuroImage 70:150-163,
         doi:10.1016/j.neuroimage.2012.12.018
       Schellenberger Costa et al. (2016) PLoS Comput Biol 12(9):e1005022,
         doi:10.1371/journal.pcbi.1005022
     Coupling that switch to the *photic entrainment* peak is a modelling
     choice of this simulator, not a measured frequency-response curve.
     ``shift_per_unit_ratio`` is therefore a free parameter; set it to 0
     to disable the state-dependent shift entirely.

This is the simplest physiologically defensible mapping: amplitude-
modulated input current to the EI population with a state-dependent
frequency-response curve. SSVEP / photic-driving response then emerges
from the Jansen-Rit dynamics naturally (Spiegler et al. 2011).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Nominal broadband transmission of the closed eyelid: the fraction of
# commanded mask light assumed to reach the retina.
#
# This is a modelling placeholder, not a measured constant. It enters the
# drive only as `brightness * EYELID_TRANSMISSION * cortical_gain`, so it is
# mathematically degenerate with `PhoticInputConfig.cortical_gain` - and
# cortical_gain is domain-randomized over a 7x range (see
# `RandomizationRanges.cortical_gain_range`), which subsumes any plausible
# value here. Changing this number rescales the subject distribution; it does
# not add information the agent can exploit.
#
# A per-wavelength model would need real calibration (the published eyelid
# transmittance curves are strongly wavelength-dependent) and is pointless
# while the mask is driven white-only.
EYELID_TRANSMISSION = 0.10


@dataclass
class StrobeParams:
    """Mirror of the mask's UDP/JSON CMD_CHANGE_PARAMS fields."""
    enabled: bool = False
    frequency_hz: float = 10.0
    fill: float = 0.2          # duty cycle 0..1
    brightness: float = 0.3    # 0..1

    def effective_amplitude(self) -> float:
        """Brightness * (light reaching retina through closed eyelid).

        The mask's RGBW colour is operator-set hardware configuration
        (``mask.strobe.*``), not a policy knob, so it plays no part here.
        """
        if not self.enabled:
            return 0.0
        return self.brightness * EYELID_TRANSMISSION


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
    monotonically over the session, modeling reduced ascending arousal drive:

      Phillips & Robinson (2007) J Biol Rhythms 22(2):167-179,
        doi:10.1177/0748730406297512 - mean-field model of the brainstem
        ascending arousal system.
      Robinson, Rennie & Rowe (2002) Phys Rev E 65:041924,
        doi:10.1103/PhysRevE.65.041924 - arousal as a trajectory through a
        low-dimensional parameter space, passing the δ/θ/α resonances.
      Weigenand et al. (2014) PLoS Comput Biol 10(11):e1003923,
        doi:10.1371/journal.pcbi.1003923 - wake -> deep sleep as a
        parameter change in a neural mass cortex.
      Hindriks & van Putten (2013), doi:10.1016/j.neuroimage.2012.12.018 -
        the resulting α amplitude / frequency changes.

    SIMPLIFICATION: the same scalar is returned for both the mean-drive and
    the noise multiplier. This is a deliberate simplification - Weigenand
    et al. (2014) treats cortical noise as independent of the slow
    modulatory drift, so the two could be decoupled if needed.

    Wired in `InProcessBrainSim.step()` so the brain's background drive
    decays over the session, organically raising θ/α. This is the
    "natural drowsiness drift" the agent both fights against and leverages -
    its job is to accelerate the drift via well-timed entrainment.
    """
    progress = min(1.0, max(0.0, t / max(duration_s, 1e-6)))
    arousal = start_arousal + (end_arousal - start_arousal) * progress
    return arousal, arousal
