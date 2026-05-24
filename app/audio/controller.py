"""
AudioController - the agent's audio action surface.

Wraps :class:`app.audio.engine.BiofeedbackEngine` with:
- Clipping to safe ranges (no -inf volumes, no >100% depths).
- Time-stamped event log captured for replay (mirrors the strobe log).
- A no-op mode for headless training (when no audio device is available).

The agent only ever talks to this controller - never directly to pyo. That
keeps the RL training environment trivially mockable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("audio.controller")


@dataclass
class AudioEvent:
    """One entry in the audio parameter log (persisted to Parquet per session)."""
    t_ms: int
    param: str
    value: float


@dataclass
class AudioParams:
    """Snapshot of agent-controllable audio parameters."""
    sync_freq_hz: float = 0.0
    strobe_depth_pct: float = 10.0
    binaural_vol_pct: float = 10.0
    brown_noise_pct: float = 80.0
    master_vol_pct: float = 50.0
    warmth_cutoff_pct: float = 80.0
    reverb_space_pct: float = 80.0
    preset_name: str = "sunlight_pads"


class AudioController:
    """
    Façade between the agent and the pyo engine.

    Use ``AudioController(engine=None)`` in unit/training contexts where no
    audio device is available - every setter becomes a no-op except for
    event logging.
    """

    def __init__(self, engine: Optional["object"] = None,
                 initial: Optional[AudioParams] = None):
        self.engine = engine
        self.params = initial or AudioParams()
        self.events: List[AudioEvent] = []
        self._start = time.time()
        if engine is not None:
            self._push_all()

    def reset_clock(self) -> None:
        self._start = time.time()

    def _now_ms(self) -> int:
        return int((time.time() - self._start) * 1000)

    def _log(self, param: str, value: float) -> None:
        self.events.append(AudioEvent(self._now_ms(), param, float(value)))

    def set_sync_freq(self, hz: float) -> None:
        hz = float(max(0.0, min(40.0, hz)))
        self.params.sync_freq_hz = hz
        self._log("sync_freq_hz", hz)
        if self.engine is not None:
            self.engine.set_sync_frequency(hz)

    def set_strobe_depth(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.strobe_depth_pct = percent
        self._log("strobe_depth_pct", percent)
        if self.engine is not None:
            self.engine.set_strobe_depth_pct(percent)

    def set_binaural_vol(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.binaural_vol_pct = percent
        self._log("binaural_vol_pct", percent)
        if self.engine is not None:
            self.engine.set_binaural_volume_pct(percent)

    def set_brown_noise(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.brown_noise_pct = percent
        self._log("brown_noise_pct", percent)
        if self.engine is not None:
            self.engine.set_brown_noise_volume_pct(percent)

    def set_master_vol(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.master_vol_pct = percent
        self._log("master_vol_pct", percent)
        if self.engine is not None:
            self.engine.set_master_volume_pct(percent)

    def set_warmth(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.warmth_cutoff_pct = percent
        self._log("warmth_cutoff_pct", percent)
        if self.engine is not None:
            self.engine.set_warmth_cutoff_pct(percent)

    def set_reverb(self, percent: float) -> None:
        percent = float(max(0.0, min(100.0, percent)))
        self.params.reverb_space_pct = percent
        self._log("reverb_space_pct", percent)
        if self.engine is not None:
            self.engine.set_reverb_space_pct(percent)

    def set_test_mode(self, on: bool) -> None:
        """Ping-pong stereo check tone - for soundcheck before a session."""
        self._log("test_mode", 1.0 if on else 0.0)
        if self.engine is not None:
            try:
                self.engine.set_test_mode(bool(on))
            except AttributeError:
                pass

    def fadeout(self, duration_s: float = 2.0) -> None:
        """Smoothly drop master volume to 0 - used by the panic stop."""
        steps = max(1, int(duration_s * 10))
        start = self.params.master_vol_pct
        for i in range(steps):
            frac = 1.0 - (i + 1) / steps
            self.set_master_vol(start * frac)
            time.sleep(duration_s / steps)

    def _push_all(self) -> None:
        """Push the current AudioParams into a freshly-created engine."""
        if self.engine is None:
            return
        self.engine.set_sync_frequency(self.params.sync_freq_hz)
        self.engine.set_strobe_depth_pct(self.params.strobe_depth_pct)
        self.engine.set_binaural_volume_pct(self.params.binaural_vol_pct)
        self.engine.set_brown_noise_volume_pct(self.params.brown_noise_pct)
        self.engine.set_master_volume_pct(self.params.master_vol_pct)
        self.engine.set_warmth_cutoff_pct(self.params.warmth_cutoff_pct)
        self.engine.set_reverb_space_pct(self.params.reverb_space_pct)

    def event_log(self) -> List[AudioEvent]:
        return list(self.events)

    def clear_event_log(self) -> None:
        self.events.clear()
