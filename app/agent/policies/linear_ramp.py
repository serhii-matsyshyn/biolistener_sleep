"""
Open-loop baseline: linear ramp from start_hz down to end_hz over the
session.

The schedule mimics the natural sleep transition (relaxed wakefulness
α ~10 Hz -> hypnagogic θ ~4–6 Hz) and is self-authored, containing no
proprietary content. The policy holds state internally so it can compute
its current output from the elapsed time at each ``predict`` call.

**Clock.** ``tick_hz=None`` (default) derives elapsed time from the wall
clock, which is correct for a live session driven in real time. In
simulation the control loop runs orders of magnitude faster than real
time, so a wall clock would collapse a 300 s ramp into a couple of
seconds. Callers driving a simulated clock set ``tick_hz`` to the control
rate; each ``predict()`` call then advances exactly 1 / tick_hz of
*simulated* time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

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
    # None -> wall clock (live sessions). Set to the control-loop rate
    # (e.g. 4.0) to count predict() calls as simulated time instead.
    tick_hz: Optional[float] = None
    _start_t: float = field(default_factory=time.time)
    _ticks: int = 0

    def reset(self) -> None:
        self._start_t = time.time()
        self._ticks = 0

    def elapsed_s(self) -> float:
        """Seconds since ``reset()`` on whichever clock is configured."""
        if self.tick_hz is not None:
            return self._ticks / max(1e-9, float(self.tick_hz))
        return time.time() - self._start_t

    def target_hz(self, elapsed_s: float) -> float:
        frac = max(0.0, min(1.0, elapsed_s / max(1e-3, self.duration_s)))
        return self.start_hz + (self.end_hz - self.start_hz) * frac

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ
        elapsed = self.elapsed_s()
        self._ticks += 1
        target_hz = self.target_hz(elapsed)
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
