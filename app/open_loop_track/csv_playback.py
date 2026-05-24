"""
CsvEventPlayback - read a raw ``(Timestamp_NS, Torch_State)`` strobe-event
CSV and answer ``params_at(t_s) -> (freq_hz, fill)`` for any wall-clock time.

The catalog YAML maps each activity/variant to one of these CSVs. They
contain the rising/falling edges actually emitted by the external
replication source at runtime (recorded by hooking the LED-toggle call).

Sample CSV (header + nanosecond timestamps):

    Timestamp_NS,Torch_State
    300000,0
    45400300000,1
    45401000000,0
    45470500000,1
    45471200000,0
    ...

Algorithm:
  1. Walk the events once, collecting (rise_ns, on_ns) pairs.
  2. Build cycles from consecutive rises: each cycle's period is the gap
     to the next rise; its on-duration was captured in step 1.

At query time, binary-search the cycle list by start_ns and read off
(freq_hz = 1e9 / period_ns, fill = on_ns / period_ns).

Interface intentionally mirrors :class:`OpenLoopTrackPlayback` so
:class:`OpenLoopTrackPolicy` accepts either via duck typing.
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("open_loop_track.csv_playback")


@dataclass
class _Cycle:
    """One full ON cycle: start at a rising edge, span to the NEXT rising edge."""
    start_ns: int
    period_ns: int
    on_ns: int


class CsvEventPlayback:
    """Event-log playback over a (Timestamp_NS, Torch_State) CSV."""

    def __init__(self, cycles: List[_Cycle], total_ns: int):
        self._cycles = cycles
        self._starts = [c.start_ns for c in cycles]
        self._total_ns = total_ns

    @property
    def duration_s(self) -> float:
        return self._total_ns / 1_000_000_000.0

    @property
    def n_cycles(self) -> int:
        return len(self._cycles)

    @classmethod
    def from_csv(cls, path: Path) -> "CsvEventPlayback":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if "Timestamp_NS" not in df.columns or "Torch_State" not in df.columns:
            raise ValueError(
                f"{path}: expected columns 'Timestamp_NS, Torch_State', "
                f"got {list(df.columns)}"
            )
        df = df.sort_values("Timestamp_NS").reset_index(drop=True)
        ts = df["Timestamp_NS"].astype("int64").to_numpy()
        st = df["Torch_State"].astype("int64").to_numpy()

        # Pass 1: collect (rise_ns, on_ns) pairs from the event stream.
        rises: List[Tuple[int, int]] = []
        current_rise: Optional[int] = None
        prev_state = 0
        for i in range(len(ts)):
            t = int(ts[i])
            s = int(st[i])
            if prev_state == 0 and s == 1:
                current_rise = t
            elif prev_state == 1 and s == 0 and current_rise is not None:
                rises.append((current_rise, t - current_rise))
                current_rise = None
            prev_state = s

        # Pass 2: build cycles from consecutive rises.
        cycles: List[_Cycle] = []
        for k in range(len(rises) - 1):
            rise_ns, on_ns = rises[k]
            next_rise_ns = rises[k + 1][0]
            period_ns = max(1, next_rise_ns - rise_ns)
            cycles.append(_Cycle(
                start_ns=rise_ns, period_ns=period_ns,
                on_ns=max(0, min(on_ns, period_ns)),
            ))
        # The final rising edge can't be paired (no next rise); we
        # intentionally drop it - its ON span alone doesn't pin down a
        # period. Real catalogs have >> 1 cycle so this is negligible.

        total_ns = int(ts[-1]) if len(ts) else 0
        return cls(cycles=cycles, total_ns=total_ns)

    def params_at(self, t_s: float) -> Optional[Tuple[float, float]]:
        """
        Return (frequency_hz, fill_0_to_1) at elapsed time ``t_s``.

        - In a quiet stretch before the first rising edge -> ``(0.0, 0.0)``.
        - Inside a cycle -> that cycle's freq + duty.
        - In a gap between bursts -> ``(0.0, 0.0)``.
        - After the last cycle's projected end -> ``None`` (track over).
        """
        if t_s < 0:
            return None
        if not self._cycles:
            return None
        t_ns = int(t_s * 1_000_000_000)
        if t_ns < self._starts[0]:
            return 0.0, 0.0
        idx = bisect.bisect_right(self._starts, t_ns) - 1
        if idx < 0:
            return 0.0, 0.0
        cyc = self._cycles[idx]
        if t_ns >= cyc.start_ns + cyc.period_ns:
            if idx + 1 < len(self._cycles):
                return 0.0, 0.0
            return None
        freq_hz = 1_000_000_000.0 / cyc.period_ns
        fill = cyc.on_ns / cyc.period_ns
        return freq_hz, max(0.0, min(1.0, fill))
