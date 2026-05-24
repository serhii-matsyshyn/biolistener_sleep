"""
Per-tick observation state shared between training env and runtime daemon.

The training env (:class:`app.agent.env.SleepOnsetEnv`) builds an observation
vector that includes temporal context - EMA-smoothed PLV and Δ(smoothed θ/α)
over the last 4 s. The daemon must build the *identical* vector at deployment
or every SAC checkpoint will hit a shape mismatch on the first call to
``model.predict``.

Rather than duplicate the smoothing logic across env + daemon (a recipe for
drift), this module owns the state machine. Both consumers instantiate one
``ObsState`` per session, call ``update()`` each control tick, and read back
the smoothed θ/α (overwrites the raw feature) plus the temporal-context
suffix used in the obs vector.

Constants here MUST match :class:`SleepOnsetEnv` exactly:
- EMA τ = 2 s (suppresses single-tick noise spikes ~78% over 4 ticks).
- Δ window = 4 s (16 ticks at 4 Hz tick_hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_TICK_HZ: float = 4.0
DEFAULT_EMA_TAU_S: float = 2.0
DEFAULT_DELTA_WINDOW_S: float = 4.0


@dataclass
class ObsState:
    """
    Mutable smoothing state for one session.

    Call ``update(raw_theta_alpha, raw_plv)`` every control tick (before
    building the obs vector). The instance then exposes:

    - ``smoothed_theta_alpha`` - EMA of θ/α; the env overwrites the raw
      feature with this so downstream consumers (reward, resonance shift)
      see the smoothed value. The daemon does the same.
    - ``smoothed_plv``         - EMA of PLV; used directly in the obs suffix.
    - ``delta_theta_alpha``    - smoothed θ/α minus its value
      ``delta_window_ticks`` ago. The 4-s trend signal that lets the policy
      distinguish a real drowsiness rise from a single-tick noise spike.
    """

    tick_hz: float = DEFAULT_TICK_HZ
    ema_tau_s: float = DEFAULT_EMA_TAU_S
    delta_window_s: float = DEFAULT_DELTA_WINDOW_S

    smoothed_theta_alpha: float = 0.0
    smoothed_plv: float = 0.0
    delta_theta_alpha: float = 0.0
    _initialized: bool = False
    _theta_alpha_history: List[float] = field(default_factory=list)

    @property
    def ema_alpha_decay(self) -> float:
        return 1.0 - float(np.exp(-1.0 / self.tick_hz / self.ema_tau_s))

    @property
    def delta_window_ticks(self) -> int:
        return int(round(self.delta_window_s * self.tick_hz))

    def update(self, raw_theta_alpha: float, raw_plv: float) -> None:
        """
        Ingest one tick of raw features. Updates smoothed_* + delta_* in place.

        First call seeds the EMAs with the raw values (no smoothing yet) so
        the policy doesn't see a multi-tick warmup of zeros.
        """
        a = self.ema_alpha_decay
        if not self._initialized:
            self.smoothed_theta_alpha = float(raw_theta_alpha)
            self.smoothed_plv = float(raw_plv)
            self._initialized = True
        else:
            self.smoothed_theta_alpha = (
                (1.0 - a) * self.smoothed_theta_alpha + a * float(raw_theta_alpha)
            )
            self.smoothed_plv = (
                (1.0 - a) * self.smoothed_plv + a * float(raw_plv)
            )

        # Ring buffer of smoothed θ/α; Δ = current − value 16 ticks ago.
        # Until the buffer is fully primed, Δ uses the oldest available
        # value (gracefully degrades to 0 on the very first tick).
        self._theta_alpha_history.append(self.smoothed_theta_alpha)
        if len(self._theta_alpha_history) > self.delta_window_ticks:
            self._theta_alpha_history.pop(0)
        self.delta_theta_alpha = (
            self.smoothed_theta_alpha - self._theta_alpha_history[0]
        )

    def temporal_context(self) -> Tuple[float, float]:
        """Return the 2-tuple appended to the obs vector: (smoothed_plv, Δθ/α)."""
        return (float(self.smoothed_plv), float(self.delta_theta_alpha))
