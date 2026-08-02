"""
Open-loop schedule family used as the scripted control arms.

A :class:`ScheduledRampPolicy` emits a piecewise-linear frequency
schedule: hold at ``start_hz`` for the first ``ramp_start_frac`` of the
session, ramp linearly to ``end_hz`` by ``ramp_end_frac``, hold there for
the remainder. ``ramp_start_frac=0, ramp_end_frac=1`` reproduces
:class:`LinearRampDownPolicy`.

Two arms are built from it:

* **generic open-loop** - anchored at a population value (11 Hz), the
  protocol a real non-adaptive device would ship.
* **oracle open-loop** - anchored at the subject's *true* individual α
  frequency, which the simulator knows and no real system does. This is
  the upper bound on what a non-adaptive protocol could achieve given
  perfect subject identification, and it separates the two sources of
  closed-loop benefit:

      oracle-OL − generic-OL  =  value of subject identification
      SAC       − oracle-OL   =  value of target tracking

The oracle is deliberately restricted to *static* subject parameters. It
may not see the realised θ/α trajectory, because anticipating how the
resonance peak moves during the episode is precisely the tracking value
that belongs to the closed-loop arm rather than to the oracle.

Clock handling matches :class:`LinearRampDownPolicy`: ``tick_hz=None``
reads the wall clock (live sessions), a number counts ``predict()`` calls
as simulated time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ScheduledRampPolicy:
    """Piecewise-linear open-loop frequency schedule. Ignores observations."""
    start_hz: float = 11.0
    end_hz: float = 4.0
    duration_s: float = 300.0
    ramp_start_frac: float = 0.0    # hold start_hz until this fraction
    ramp_end_frac: float = 1.0      # reach end_hz at this fraction
    fill: float = 0.20
    brightness: float = 0.30
    audio_depth_pct: float = 20.0
    audio_master_pct: float = 50.0
    tick_hz: Optional[float] = None
    _start_t: float = field(default_factory=time.time)
    _ticks: int = 0

    def reset(self) -> None:
        self._start_t = time.time()
        self._ticks = 0

    def elapsed_s(self) -> float:
        if self.tick_hz is not None:
            return self._ticks / max(1e-9, float(self.tick_hz))
        return time.time() - self._start_t

    def target_hz(self, elapsed_s: float) -> float:
        frac = max(0.0, min(1.0, elapsed_s / max(1e-3, self.duration_s)))
        lo = max(0.0, min(1.0, self.ramp_start_frac))
        hi = max(lo + 1e-9, min(1.0, self.ramp_end_frac))
        if frac <= lo:
            return self.start_hz
        if frac >= hi:
            return self.end_hz
        u = (frac - lo) / (hi - lo)
        return self.start_hz + (self.end_hz - self.start_hz) * u

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ
        elapsed = self.elapsed_s()
        self._ticks += 1
        target_hz = self.target_hz(elapsed)
        freq_norm = 2.0 * (target_hz - FREQ_MIN_HZ) \
                    / (FREQ_MAX_HZ - FREQ_MIN_HZ) - 1.0
        freq_norm = max(-1.0, min(1.0, freq_norm))
        action = np.array([
            freq_norm,
            2.0 * self.fill - 1.0,
            2.0 * self.brightness - 1.0,
            2.0 * (self.audio_depth_pct / 100.0) - 1.0,
            2.0 * (self.audio_master_pct / 100.0) - 1.0,
        ], dtype=np.float32)
        return action, None


def oracle_ramp_factory(
    *,
    end_hz: float = 4.0,
    duration_s: float = 300.0,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
    fill: float = 0.20,
    brightness: float = 0.30,
    audio_depth_pct: float = 20.0,
    audio_master_pct: float = 50.0,
):
    """Build a ``policy_factory`` anchoring the ramp at the subject's true α.

    Usable with :func:`app.agent.eval_rollout.rollout_episodes`; the
    factory receives ``(episode_index, subject_info)`` and reads
    ``subject_info["alpha_hz"]`` - the one piece of privileged knowledge
    this arm is granted.
    """
    def _factory(episode: int, subject: dict) -> ScheduledRampPolicy:
        alpha_hz = float(subject.get("alpha_hz", 11.0))
        return ScheduledRampPolicy(
            start_hz=alpha_hz,
            end_hz=end_hz,
            duration_s=duration_s,
            ramp_start_frac=ramp_start_frac,
            ramp_end_frac=ramp_end_frac,
            fill=fill,
            brightness=brightness,
            audio_depth_pct=audio_depth_pct,
            audio_master_pct=audio_master_pct,
        )
    return _factory
