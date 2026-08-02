"""
OpenLoopTrackPolicy - wraps OpenLoopTrackPlayback in the policy interface.

The daemon's control loop calls ``predict(observation)`` every tick;
the policy reads its own elapsed time, asks the playback for the active
segment's (frequency, fill), and emits a normalized action. The
SafetySupervisor still clamps the output (so even this protocol stays
within the project's safe-operating envelope).

Brightness comes from config (not the recorded segments) so we can tune
intensity per subject without re-recording. The audio file is played
through ``OpenLoopTrackAudioPlayer``, started separately by the daemon's
``_setup_audio`` branch.

**Clock.** ``tick_hz=None`` (default) reads the wall clock, which is
correct for a live session driven in real time. In simulation the control
loop runs far faster than real time, so callers driving a simulated clock
set ``tick_hz`` to the control rate and each ``predict()`` call advances
exactly 1 / tick_hz of *simulated* time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.open_loop_track.playback import OpenLoopTrackPlayback

logger = logging.getLogger("open_loop_track.policy")


@dataclass
class OpenLoopTrackPolicy:
    playback: OpenLoopTrackPlayback
    brightness: float = 0.30          # not in recorded data - config-level
    audio_master_pct: float = 50.0
    audio_strobe_depth_pct: float = 20.0
    # None -> wall clock (live sessions). Set to the control-loop rate
    # (e.g. 4.0) to count predict() calls as simulated time instead.
    tick_hz: Optional[float] = None
    _start_t: float = field(default_factory=time.time)
    _ticks: int = 0
    # Set True the first time params_at() reports past-end. The daemon's
    # control loop reads this on each tick and closes the session so it
    # doesn't sit forever emitting "off" frames after the CSV's last cycle.
    track_ended: bool = False

    def reset(self) -> None:
        self._start_t = time.time()
        self._ticks = 0
        self.track_ended = False

    def elapsed_s(self) -> float:
        """Seconds since ``reset()`` on whichever clock is configured."""
        if self.tick_hz is not None:
            return self._ticks / max(1e-9, float(self.tick_hz))
        return time.time() - self._start_t

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ

        elapsed = self.elapsed_s()
        self._ticks += 1
        params = self.playback.params_at(elapsed)
        if params is None:
            # Track ended - emit OFF and latch the end-of-track flag.
            self.track_ended = True
            return self._encode_off(), None

        freq_hz, fill = params
        # Clamp the parts of the action that map outside safe range BEFORE
        # encoding (the SafetySupervisor will catch anything we miss).
        freq_norm = 2.0 * (float(freq_hz) - FREQ_MIN_HZ) \
                    / (FREQ_MAX_HZ - FREQ_MIN_HZ) - 1.0
        freq_norm = max(-1.0, min(1.0, freq_norm))
        fill_norm = max(-1.0, min(1.0, 2.0 * float(fill) - 1.0))
        bri_norm = max(-1.0, min(1.0, 2.0 * self.brightness - 1.0))
        depth_norm = 2.0 * (self.audio_strobe_depth_pct / 100.0) - 1.0
        vol_norm = 2.0 * (self.audio_master_pct / 100.0) - 1.0
        action = np.array(
            [freq_norm, fill_norm, bri_norm, depth_norm, vol_norm],
            dtype=np.float32,
        )
        return action, None

    def _encode_off(self) -> np.ndarray:
        return np.array([-1.0, -1.0, -1.0, -1.0,
                         2.0 * (self.audio_master_pct / 100.0) - 1.0],
                        dtype=np.float32)
