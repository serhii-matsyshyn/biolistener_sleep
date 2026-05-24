"""
Deterministic perceptual shaping for the entrainment audio.

The RL agent's reward only scores EEG response (band-weighted PLV, theta/alpha
rise, descent). The NMM simulator has no audio input model, so the agent
cannot learn perceptual choices like "darken the timbre as the subject becomes
drowsy" or "fade master volume in the last quarter of the session". Those are
open-loop deterministic decisions made at deployment.

This shaper owns two of the seven audio knobs:

- ``warmth_cutoff_pct`` - lowpass filter cutoff (lower = darker, more lullaby).
  Mapped from smoothed theta/alpha so the music darkens as the subject descends
  toward N1.
- ``master_vol_pct``    - replaces the agent's action[4] outright. Combines a
  drowsiness-driven reduction with a final-fraction-of-session fade.

The remaining audio knobs (binaural, brown noise, reverb, strobe depth, sync
frequency) keep their existing wiring: sync_freq tracks the strobe, strobe
depth is the agent's action[3], and the rest stay at their AudioConfig
defaults for the whole session.

All thresholds are documented + config-driven via ``AudioConfig.shaper``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioShaper:
    """Mapping coefficients. Defaults match ``AudioConfig.shaper`` in the
    schema; the daemon constructs one of these from the resolved config."""

    # Drowsiness window. theta/alpha at or below ``baseline`` => "alert",
    # no shaping applied. theta/alpha at or above ``full`` => "fully drowsy",
    # warmth/volume at their endpoint values. Values picked to match the
    # same drowsiness window the NMM resonance shift uses (see
    # PhoticInputConfig.shift_baseline_ratio).
    theta_alpha_baseline: float = 0.5
    theta_alpha_full: float = 3.0

    # Warmth curve (lowpass cutoff %): bright at baseline, dark at full.
    warmth_at_baseline_pct: float = 80.0
    warmth_at_full_pct: float = 25.0

    # Master volume curve. The agent's action[4] is ignored; this is the
    # commanded value every tick.
    # - Drowsiness-driven reduction: full-drowsy multiplies initial by
    #   (1 - drowsy_taper).
    # - End-of-session fade: in the last ``end_fade_frac`` of the
    #   session, volume linearly drops to ``end_fade_min_factor`` of its
    #   pre-fade value.
    initial_master_vol_pct: float = 50.0
    drowsy_vol_taper: float = 0.4          # 50% * (1 - 0.4) = 30% at full drowsy
    end_fade_starts_at_frac: float = 0.75  # fade begins at 75% of session
    end_fade_min_factor: float = 0.4       # tail of session: 50% * 0.4 = 20%

    def drowsiness(self, theta_alpha_smoothed: float) -> float:
        """Saturating drowsiness signal in [0, 1] from the smoothed θ/α."""
        if theta_alpha_smoothed <= self.theta_alpha_baseline:
            return 0.0
        if theta_alpha_smoothed >= self.theta_alpha_full:
            return 1.0
        span = self.theta_alpha_full - self.theta_alpha_baseline
        return (theta_alpha_smoothed - self.theta_alpha_baseline) / span

    def warmth_pct(self, theta_alpha_smoothed: float) -> float:
        d = self.drowsiness(theta_alpha_smoothed)
        return (self.warmth_at_baseline_pct
                + d * (self.warmth_at_full_pct - self.warmth_at_baseline_pct))

    def master_vol_pct(self, theta_alpha_smoothed: float,
                       time_into_session_s: float,
                       session_duration_s: float) -> float:
        d = self.drowsiness(theta_alpha_smoothed)
        vol = self.initial_master_vol_pct * (1.0 - self.drowsy_vol_taper * d)
        if session_duration_s > 0:
            t_frac = max(0.0, min(1.0, time_into_session_s / session_duration_s))
            if t_frac > self.end_fade_starts_at_frac:
                fade_progress = ((t_frac - self.end_fade_starts_at_frac)
                                  / (1.0 - self.end_fade_starts_at_frac))
                fade_factor = (1.0 - fade_progress
                               * (1.0 - self.end_fade_min_factor))
                vol *= fade_factor
        return vol
