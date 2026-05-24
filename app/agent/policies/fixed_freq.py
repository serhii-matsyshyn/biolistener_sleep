"""
Open-loop baseline: constant strobe at the canonical α frequency.

Defensible reference baseline - represents "what would happen if you
just locked the strobe at α throughout the session?". Compared against
the SAC policy's learning curve in MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedFrequencyPolicy:
    """Always emits the same physical action. Ignores observation."""
    frequency_hz: float = 10.0
    fill: float = 0.20
    brightness: float = 0.30
    audio_depth_pct: float = 20.0
    audio_master_pct: float = 50.0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        # Encode physical -> normalized action expected by SleepOnsetEnv
        from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ
        freq_norm = 2.0 * (self.frequency_hz - FREQ_MIN_HZ) \
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
