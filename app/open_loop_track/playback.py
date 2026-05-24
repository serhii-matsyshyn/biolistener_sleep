"""
OpenLoopTrackPlayback - parametric playback of a pre-recorded open-loop strobe
sequence.

Each "activity" is a list of time-bounded segments. Each segment is a
piecewise-linear ramp on frequency (Hz) and duty cycle (%) over time,
parameterized by 6 numbers per segment::

    segment = [start_us, duration_us, offset, divisor, m_duty, a_duty, m_freq, a_freq]

The math (derived from a decompiled native library and ported as pure
Python) is::

    raw_val   = current_time_us * 100
    norm_val  = (float(raw_val) - offset) / divisor
    duty_raw  = norm_val * m_duty + a_duty            # -> integer duty (per-mille)
    freq_raw  = norm_val * m_freq + a_freq            # -> integer (Hz*1000)
    period_us = 100_000 / freq_raw                    # period of one ON+OFF cycle
    on_us     = duty_raw * period_us                  # ON portion

We expose ``params_at(t_s)`` returning (frequency_hz, fill_0_to_1) so the
daemon's 4 Hz tick can sample without busy-waiting (busy-wait was the
original playback approach; here we just evaluate the ramp analytically).

Brightness is NOT in the recorded data - it's a config-level value
supplied by ``OpenLoopTrackPolicy``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger("open_loop_track.playback")


@dataclass
class _Segment:
    start_us: int
    duration_us: int
    offset: float
    divisor: float
    m_duty: float
    a_duty: float
    m_freq: float
    a_freq: float

    @property
    def end_us(self) -> int:
        return self.start_us + self.duration_us


class OpenLoopTrackPlayback:
    """
    A loaded activity's segment list, with analytic-evaluation per tick.

    Usage::

        playback = OpenLoopTrackPlayback.from_config_yaml(path, "befriendingsleep", "10")
        freq, fill = playback.params_at(t_s=120.0)
    """

    def __init__(self, segments: List[_Segment]):
        self.segments = segments
        self._total_us = max((s.end_us for s in segments), default=0)

    @property
    def duration_s(self) -> float:
        return self._total_us / 1_000_000.0

    @classmethod
    def from_segments(cls, raw: List[list]) -> "OpenLoopTrackPlayback":
        """Build from a list of [start_us, dur_us, offset, divisor, m_duty, a_duty, m_freq, a_freq]."""
        segs = []
        for row in raw:
            if len(row) < 8:
                raise ValueError(
                    f"Open-loop-track segment must have 8 values, got {len(row)}"
                )
            segs.append(_Segment(
                start_us=int(row[0]), duration_us=int(row[1]),
                offset=float(row[2]), divisor=float(row[3]),
                m_duty=float(row[4]), a_duty=float(row[5]),
                m_freq=float(row[6]), a_freq=float(row[7]),
            ))
        return cls(segs)

    @classmethod
    def from_config_yaml(cls, path: Path, library: str, preset: str
                          ) -> "OpenLoopTrackPlayback":
        """Load segments from a strobe_config.yaml file (external project format)."""
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level must be a mapping")
        if library not in raw:
            raise KeyError(f"library {library!r} not in {path}")
        if preset not in raw[library]:
            raise KeyError(f"preset {preset!r} not in {library} (keys: {list(raw[library])})")
        return cls.from_segments(raw[library][preset])

    def params_at(self, t_s: float) -> Optional[tuple[float, float]]:
        """
        Return (frequency_hz, fill_0_to_1) at elapsed time t_s.

        Returns None if t_s is past the end of all segments (track over).
        """
        t_us = int(t_s * 1_000_000)
        if t_us < 0 or t_us >= self._total_us:
            return None
        seg = self._active_segment(t_us)
        if seg is None:
            return None
        return _evaluate_segment(seg, t_us)

    def _active_segment(self, t_us: int) -> Optional[_Segment]:
        for s in self.segments:
            if s.start_us <= t_us < s.end_us:
                return s
        return None

    def maximum_freq_hz(self) -> float:
        """Conservative ceiling - checks both segment endpoints."""
        peak = 0.0
        for seg in self.segments:
            for t_us in (seg.start_us, seg.end_us - 1):
                params = _evaluate_segment(seg, t_us)
                if params is not None:
                    peak = max(peak, params[0])
        return peak

    def maximum_fill(self) -> float:
        peak = 0.0
        for seg in self.segments:
            for t_us in (seg.start_us, seg.end_us - 1):
                params = _evaluate_segment(seg, t_us)
                if params is not None:
                    peak = max(peak, params[1])
        return peak


def _evaluate_segment(seg: _Segment, t_us: int) -> Optional[tuple[float, float]]:
    """Run the parametric ramp math at one instant and return (Hz, fill)."""
    raw_val = t_us * 100
    if seg.divisor == 0:
        return None
    norm_val = (float(raw_val) - seg.offset) / seg.divisor
    duty_raw = int(norm_val * seg.m_duty + seg.a_duty)
    freq_raw = int(norm_val * seg.m_freq + seg.a_freq)
    if freq_raw <= 0:
        return None
    # freq_raw is in (Hz × 1000); duty_raw is in per-mille of period
    period_us = 100_000 / freq_raw
    on_us = duty_raw * period_us
    if period_us <= 0:
        return None
    freq_hz = freq_raw / 1000.0
    # full cycle is period_us * 1000 µs (since freq_raw was scaled), but
    # both `period_us` and `on_us` use the same scaling so the ratio is correct.
    fill = max(0.0, min(1.0, on_us / (period_us * 1000.0)))
    return freq_hz, fill
