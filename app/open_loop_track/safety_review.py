"""
Pre-flight safety review of an open-loop-track activity.

Runs the playback's analytic-evaluator across the full duration in
coarse steps and checks that no point would fall outside the
SafetySupervisor's safe-operating envelope. Used by
``biolistener-sleep open-loop-track-check`` and by the daemon's session-start path
when the user selects this protocol.

If any sample falls inside the PSE band or above the brightness/fill
cap, we still allow the session to proceed (the live SafetySupervisor
will clamp), but we surface a warning to the UI / CLI so the researcher
can pick a different activity if they prefer not to have continuous
clamping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config import BioListenerSleepConfig
from app.open_loop_track.playback import OpenLoopTrackPlayback


@dataclass
class SafetyReviewResult:
    activity: str
    duration_min: int
    peak_freq_hz: float
    peak_fill: float
    samples_in_pse_band: int
    samples_over_brightness_cap: int
    samples_total: int
    warnings: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return (
            self.samples_in_pse_band == 0
            and self.peak_fill <= 1.0
            and not any(w.startswith("ERROR:") for w in self.warnings)
        )


def review(
    playback: OpenLoopTrackPlayback,
    config: BioListenerSleepConfig,
    activity: str,
    duration_min: int,
    step_s: float = 1.0,
) -> SafetyReviewResult:
    """Scan the activity's full duration and report safety findings."""
    safety = config.safety
    n_pse = 0
    n_over_bri = 0   # brightness is a config constant; we just check the cap
    peak_freq = 0.0
    peak_fill = 0.0
    warnings: List[str] = []
    samples_total = 0

    t = 0.0
    end_t = min(duration_min * 60.0, playback.duration_s)
    while t < end_t:
        params = playback.params_at(t)
        t += step_s
        if params is None:
            continue
        freq_hz, fill = params
        samples_total += 1
        peak_freq = max(peak_freq, freq_hz)
        peak_fill = max(peak_fill, fill)
        if safety.pse_band_lo <= freq_hz <= safety.pse_band_hi:
            n_pse += 1
        if fill > safety.fill_max:
            n_over_bri += 1

    if peak_freq > safety.freq_max_hz:
        warnings.append(
            f"peak freq {peak_freq:.2f} Hz exceeds safety.freq_max_hz "
            f"({safety.freq_max_hz}) - will be clamped during the session"
        )
    if n_pse > 0:
        warnings.append(
            f"{n_pse} sample(s) fall inside the PSE band "
            f"[{safety.pse_band_lo}, {safety.pse_band_hi}] Hz - will be clamped"
        )
    if peak_fill > safety.fill_max:
        warnings.append(
            f"peak fill {peak_fill:.2%} exceeds safety.fill_max "
            f"({safety.fill_max:.0%}) - will be clamped"
        )

    return SafetyReviewResult(
        activity=activity,
        duration_min=duration_min,
        peak_freq_hz=peak_freq,
        peak_fill=peak_fill,
        samples_in_pse_band=n_pse,
        samples_over_brightness_cap=n_over_bri,
        samples_total=samples_total,
        warnings=warnings,
    )
