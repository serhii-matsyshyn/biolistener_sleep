"""
Open-loop baseline: linear ramp from start_hz down to end_hz over the
session.

Mimics the natural sleep transition (relaxed wakefulness α ~10 Hz ->
hypnagogic θ ~4–6 Hz). Self-authored - no proprietary content. The
policy holds state internally (start time) so it can compute its current
output from the elapsed time at each ``predict`` call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearRampDownPolicy:
    start_hz: float = 11.0
    end_hz: float = 4.0
    duration_s: float = 60.0     # match EnvConfig.episode_s for short tests
    fill: float = 0.20
    brightness: float = 0.30
    audio_depth_pct: float = 20.0
    audio_master_pct: float = 50.0
    _start_t: float = field(default_factory=time.time)

    def reset(self) -> None:
        self._start_t = time.time()

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ
        elapsed = time.time() - self._start_t
        frac = max(0.0, min(1.0, elapsed / max(1e-3, self.duration_s)))
        target_hz = self.start_hz + (self.end_hz - self.start_hz) * frac
        freq_norm = 2.0 * (target_hz - FREQ_MIN_HZ) \
                    / (FREQ_MAX_HZ - FREQ_MIN_HZ) - 1.0
        fill_norm = 2.0 * self.fill - 1.0
        bri_norm = 2.0 * self.brightness - 1.0
        depth_norm = 2.0 * (self.audio_depth_pct / 100.0) - 1.0
        vol_norm = 2.0 * (self.audio_master_pct / 100.0) - 1.0
        action = np.array(
            [freq_norm, fill_norm, bri_norm, depth_norm, vol_norm],
            dtype=np.float32,
        )
        return action, None
