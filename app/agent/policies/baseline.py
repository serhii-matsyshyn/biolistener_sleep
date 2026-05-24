"""
BaselineNoStimPolicy - no-treatment control protocol.

Always commands the strobe OFF. The session is EEG-only; brown-noise
audio (or silence) is chosen by the daemon's `_setup_audio` branch
based on `SessionProtocol.baseline_no_stim`. Used as the "no-stimulation"
arm of the three-protocol experimental design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineNoStimPolicy:
    """Always emits the normalized action that decodes to enabled=False."""

    audio_master_pct: float = 30.0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        # Encode physical params (freq, fill, brightness, depth, vol) -> normalized
        # We send min-frequency + zero brightness + zero fill so that even
        # without the SafetySupervisor disabling outright, the strobe is dark.
        from app.agent.safety import FREQ_MIN_HZ, FREQ_MAX_HZ
        freq_norm = 2.0 * (FREQ_MIN_HZ - FREQ_MIN_HZ) \
                    / (FREQ_MAX_HZ - FREQ_MIN_HZ) - 1.0  # = -1.0
        fill_norm = -1.0
        bri_norm = -1.0
        depth_norm = -1.0    # audio strobe-tremolo also off
        vol_norm = 2.0 * (self.audio_master_pct / 100.0) - 1.0
        action = np.array(
            [freq_norm, fill_norm, bri_norm, depth_norm, vol_norm],
            dtype=np.float32,
        )
        return action, None

    def reset(self) -> None:
        pass
