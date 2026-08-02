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

# Endpoints of the session-long arousal decay in :func:`vigilance_drift`.
# Named here so the exogenous-drift contrast condition can normalise that
# same scalar to [0, 1] instead of re-deriving a parallel schedule.
_VIGILANCE_START = 1.0
_VIGILANCE_END = 0.3


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
    # State-dependent shift parameters.
    #
    # ``shift_per_unit_ratio`` is the GAIN of a state -> plant coupling, not
    # a drift rate: the peak is a function of the current measured state,
    # never of elapsed time. Because a controller that drives θ/α thereby
    # moves the peak, this closes a *positive feedback* loop - the optimum
    # moves in response to the controller's own success.
    shift_per_unit_ratio: float = 2.0        # Hz per unit (θ/α − baseline)
    shift_baseline_ratio: float = 0.5        # θ/α below which no shift
    min_resonance_hz: float = 4.0            # floor on shifted peak
    # Mutable state - the drive that the shift is computed from.
    # Endogenous mode (default): the env writes the smoothed θ/α ratio here
    # after each control-tick feature extraction, so the loop is closed.
    # Exogenous mode: ``InProcessBrainSim.step`` writes a purely
    # time-driven schedule here instead (see ``exogenous_shift``), so the
    # peak drifts on a fixed course the controller cannot influence.
    current_drowsiness: float = 0.0

    # --- exogenous-drift contrast condition ------------------------------
    # When True the shift drive is a function of session progress only,
    # keyed to the same ascending-arousal decay as ``vigilance_drift``
    # (Phillips & Robinson 2007). This turns the state -> plant coupling
    # into a classical exogenous disturbance, removing the feedback path
    # while leaving the *magnitude* of the excursion unchanged, so the same
    # gain axis is directly comparable between the two conditions.
    exogenous_shift: bool = False

    # --- per-subject arousal dynamics (see vigilance_drift) --------------
    # sol_scale_s: this subject's own transition timescale, in simulated
    # seconds. Sampled lognormally per subject so the episode is a window on
    # a longer process rather than the whole process. arousal_floor: how deep
    # this subject's background drive falls once the transition completes.
    sol_scale_s: float = 300.0
    arousal_floor: float = _VIGILANCE_END
    # Drive excess at end of the subject's transition in exogenous mode.
    # Set from the plant's own geometry - the span that carries the peak from
    # base_resonance_hz down to min_resonance_hz at unit coupling gain - so
    # the contrast condition's magnitude is independent of any policy.
    exogenous_span: float = 6.0

    @property
    def effective_resonance_hz(self) -> float:
        """Resonance peak this *tick*, after state-dependent shift."""
        excess = max(0.0, self.current_drowsiness - self.shift_baseline_ratio)
        shift = excess * self.shift_per_unit_ratio
        peak = self.base_resonance_hz - shift
        return max(self.min_resonance_hz, peak)

    def exogenous_drive(self, t: float) -> float:
        """Time-driven value for ``current_drowsiness`` in exogenous mode.

        Derived from :func:`vigilance_drift` - the same ascending-arousal
        decay that already modulates the NMM's background drive, and the
        one carrying the Phillips & Robinson (2007) citation - rather than
        from a second schedule that could drift away from it. The arousal
        scalar is normalised to [0, 1] and scaled by ``exogenous_span``, so
        the peak descends from ``base_resonance_hz`` to
        ``base_resonance_hz − shift_per_unit_ratio × exogenous_span``
        on a course the controller cannot influence.
        """
        arousal, _ = vigilance_drift(
            t, self.sol_scale_s,
            start_arousal=_VIGILANCE_START, end_arousal=self.arousal_floor,
        )
        span = _VIGILANCE_START - self.arousal_floor
        frac = (_VIGILANCE_START - arousal) / span if span else 0.0
        frac = min(1.0, max(0.0, frac))
        return self.shift_baseline_ratio + self.exogenous_span * frac


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
    sol_scale_s: float,
    start_arousal: float = _VIGILANCE_START,
    end_arousal: float = _VIGILANCE_END,
) -> Tuple[float, float]:
    """
    Background drift mimicking falling-asleep over a session.

    ``sol_scale_s`` is the subject's OWN transition timescale, in simulated
    seconds, and is sampled per subject (``RandomizationRanges``). Arousal
    reaches ``end_arousal`` at ``t = sol_scale_s`` and is clamped thereafter.

    This is an ABSOLUTE clock, not a fraction of the episode. With a
    per-subject scale the episode is a *window* on a longer process - fast
    subjects reach the floor inside it, slow ones barely move - which is the
    heterogeneity a closed loop is supposed to exploit. Normalising to episode
    length instead would collapse fast and slow sleepers onto one subject.

    NOTE ON UNITS: the distribution is calibrated so the episode spans the
    population's transition range (median subject completes at about one
    episode length, ~10x spread). It is NOT calibrated to clock-time
    sleep-onset latency, and no result derived from it may be described in
    minutes-to-sleep-onset.

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
    decays over the session, organically raising θ/α.

    THE AGENT CANNOT INFLUENCE THIS. Arousal is a pure function of simulated
    time; no action reaches ``brain.p_mean`` / ``p_std``. The agent's only
    causal path into the plant is the additive photic drive, so *accelerated
    sleep onset* is not merely unmeasured in this simulator, it is
    structurally unmeasurable.
    """
    progress = min(1.0, max(0.0, t / max(sol_scale_s, 1e-6)))
    arousal = start_arousal + (end_arousal - start_arousal) * progress
    return arousal, arousal
