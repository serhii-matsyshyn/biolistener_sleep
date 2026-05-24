"""
Jansen-Rit neural mass model - single cortical column EEG generator.

Reference: Jansen & Rit (1995), "Electroencephalogram and visual evoked
potential generation in a mathematical model of coupled cortical columns",
Biol. Cybern. 73:357-366. Spiegler et al. (2011) for photic-driving extension.

Three populations per column:
  - PY  pyramidal cells (output)
  - EI  excitatory interneurons
  - II  inhibitory interneurons

State vector per column (6 dims):
  y0 = PSP from PY                y3 = dy0/dt
  y1 = excitatory PSP onto PY     y4 = dy1/dt
  y2 = inhibitory PSP onto PY     y5 = dy2/dt

Output (EEG-proxy) per column = y1 - y2.

Photic stimulation enters as an additive input current p(t) on the EI
population - the same way TVB models visual cortex photic drive.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class JansenRitParams:
    # Default Jansen-Rit 1995 values producing ~10 Hz alpha
    A: float = 3.25       # mV, excitatory PSP amplitude
    B: float = 22.0       # mV, inhibitory PSP amplitude
    a: float = 100.0      # 1/s, excitatory time constant inverse
    b: float = 50.0       # 1/s, inhibitory time constant inverse
    C: float = 135.0      # connectivity constant (Jansen-Rit "C")
    C1: float = 1.0       # PY -> EI weight (× C)
    C2: float = 0.8       # EI -> PY weight (× C)
    C3: float = 0.25      # PY -> II weight (× C)
    C4: float = 0.25      # II -> PY weight (× C)
    v0: float = 6.0       # mV, sigmoid threshold
    e0: float = 2.5       # 1/s, max firing rate / 2
    r:  float = 0.56      # 1/mV, sigmoid steepness
    p_mean: float = 220.0  # baseline cortical input mean (Hz-equivalent)
    p_std: float = 22.0   # cortical input noise std

    def with_alpha_hz(self, alpha_hz: float) -> "JansenRitParams":
        """
        Return params nudged to peak near a target alpha frequency.

        Alpha frequency in Jansen-Rit scales roughly with sqrt(a*b); we tune `a`.
        Empirical: with default b=50, a=100 gives ~10 Hz; we scale `a` linearly.
        """
        scale = alpha_hz / 10.0
        return JansenRitParams(
            A=self.A, B=self.B,
            a=self.a * scale, b=self.b,
            C=self.C, C1=self.C1, C2=self.C2, C3=self.C3, C4=self.C4,
            v0=self.v0, e0=self.e0, r=self.r,
            p_mean=self.p_mean, p_std=self.p_std,
        )


def _sigmoid(v: np.ndarray, e0: float, r: float, v0: float) -> np.ndarray:
    """Wave-to-pulse sigmoid (Jansen-Rit eq. 8)."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@dataclass
class JansenRitColumn:
    """
    Single cortical column. Numerical integration via fixed-step RK4 for
    determinism and reproducible replay.
    """
    params: JansenRitParams = field(default_factory=JansenRitParams)
    dt: float = 1.0 / 500.0  # 500 SPS to match BioListener default
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        # State: y0, y1, y2, y3, y4, y5
        self.y = np.zeros(6, dtype=np.float64)

    def _deriv(self, y: np.ndarray, photic_input: float) -> np.ndarray:
        p = self.params
        y0, y1, y2, y3, y4, y5 = y
        # Background cortical input + photic drive on EI population
        bg = p.p_mean + p.p_std * self._rng.standard_normal() + photic_input
        s_py = _sigmoid(np.array([y1 - y2]), p.e0, p.r, p.v0)[0]
        s_ei = _sigmoid(np.array([p.C1 * p.C * y0]), p.e0, p.r, p.v0)[0]
        s_ii = _sigmoid(np.array([p.C3 * p.C * y0]), p.e0, p.r, p.v0)[0]

        dy0 = y3
        dy3 = p.A * p.a * s_py - 2.0 * p.a * y3 - (p.a ** 2) * y0
        dy1 = y4
        dy4 = p.A * p.a * (bg + p.C2 * p.C * s_ei) - 2.0 * p.a * y4 - (p.a ** 2) * y1
        dy2 = y5
        dy5 = p.B * p.b * p.C4 * p.C * s_ii - 2.0 * p.b * y5 - (p.b ** 2) * y2
        return np.array([dy0, dy1, dy2, dy3, dy4, dy5])

    def step(self, photic_input: float = 0.0) -> float:
        """
        Advance one dt and return EEG-proxy sample (y1 - y2, microvolts-ish).

        `photic_input` is the instantaneous cortical drive from the strobe
        (in Hz-equivalent units that scale the EI input). Use
        `app.brain_sim.photic_input.strobe_to_cortical_input` to map
        (freq, duty, brightness) -> photic_input.
        """
        # Fixed background-noise sample per RK4 evaluation would over-noise the
        # signal; instead we draw once per step (the dt is small enough that
        # this is a reasonable Euler-Maruyama-ish approximation for the noise
        # term, while the deterministic parts get RK4).
        k1 = self._deriv(self.y, photic_input)
        k2 = self._deriv(self.y + 0.5 * self.dt * k1, photic_input)
        k3 = self._deriv(self.y + 0.5 * self.dt * k2, photic_input)
        k4 = self._deriv(self.y + self.dt * k3, photic_input)
        self.y = self.y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return float(self.y[1] - self.y[2])

    def run(
        self,
        duration_s: float,
        photic_fn: Optional[Callable[[float], float]] = None,
    ) -> np.ndarray:
        """Generate `duration_s` of EEG. `photic_fn(t)` returns drive at time t."""
        n = int(round(duration_s / self.dt))
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            t = i * self.dt
            pi = photic_fn(t) if photic_fn is not None else 0.0
            out[i] = self.step(pi)
        return out


class MultiChannelBrain:
    """
    8-channel EEG generator matching BioListener's channel count.

    All channels are integrated in a single vectorized RK4 step over a
    (n_channels, 6) state tensor, which is ~10x faster than a per-channel
    Python loop and fast enough to sustain 500 SPS for the BrainFlow
    integration path.

    Channels share most parameters but have small (±channel_jitter) variation
    in `a` and `b` to mimic spatial heterogeneity. Future: replace with a
    coupled network via TVB connectome.
    """

    def __init__(
        self,
        n_channels: int = 8,
        base_params: Optional[JansenRitParams] = None,
        alpha_hz: float = 10.0,
        sampling_rate: int = 500,
        seed: int = 0,
        channel_jitter: float = 0.05,
    ):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        base = (base_params or JansenRitParams()).with_alpha_hz(alpha_hz)
        self._base = base
        self._rng = np.random.default_rng(seed)

        # Per-channel parameter vectors (shape n_channels)
        jitter = 1.0 + channel_jitter * self._rng.standard_normal(n_channels)
        self.A = np.full(n_channels, base.A)
        self.B = np.full(n_channels, base.B)
        self.a = base.a * jitter
        self.b = base.b * jitter
        self.C = np.full(n_channels, base.C)
        self.C1 = np.full(n_channels, base.C1)
        self.C2 = np.full(n_channels, base.C2)
        self.C3 = np.full(n_channels, base.C3)
        self.C4 = np.full(n_channels, base.C4)
        self.v0 = base.v0
        self.e0 = base.e0
        self.r = base.r
        self.p_mean = base.p_mean
        self.p_std = base.p_std

        # State: (n_channels, 6) - y0..y5 per Jansen-Rit
        self.y = np.zeros((n_channels, 6), dtype=np.float64)

    def _sig(self, v: np.ndarray) -> np.ndarray:
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - v)))

    def _deriv(self, y: np.ndarray, photic_input: float,
               noise: np.ndarray) -> np.ndarray:
        # y shape (n_channels, 6)
        y0, y1, y2, y3, y4, y5 = y.T  # each (n_channels,)
        bg = self.p_mean + self.p_std * noise + photic_input
        s_py = self._sig(y1 - y2)
        s_ei = self._sig(self.C1 * self.C * y0)
        s_ii = self._sig(self.C3 * self.C * y0)
        dy0 = y3
        dy3 = self.A * self.a * s_py - 2.0 * self.a * y3 - (self.a ** 2) * y0
        dy1 = y4
        dy4 = (self.A * self.a * (bg + self.C2 * self.C * s_ei)
               - 2.0 * self.a * y4 - (self.a ** 2) * y1)
        dy2 = y5
        dy5 = (self.B * self.b * self.C4 * self.C * s_ii
               - 2.0 * self.b * y5 - (self.b ** 2) * y2)
        return np.stack([dy0, dy1, dy2, dy3, dy4, dy5], axis=1)

    def step(self, photic_input: float = 0.0) -> np.ndarray:
        """One dt for all channels - vectorized RK4."""
        noise = self._rng.standard_normal(self.n_channels)
        k1 = self._deriv(self.y, photic_input, noise)
        k2 = self._deriv(self.y + 0.5 * self.dt * k1, photic_input, noise)
        k3 = self._deriv(self.y + 0.5 * self.dt * k2, photic_input, noise)
        k4 = self._deriv(self.y + self.dt * k3, photic_input, noise)
        self.y = self.y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.y[:, 1] - self.y[:, 2]

    def run(
        self,
        duration_s: float,
        photic_fn: Optional[Callable[[float], float]] = None,
    ) -> np.ndarray:
        """Generate (n_channels, n_samples) EEG."""
        n = int(round(duration_s / self.dt))
        out = np.empty((self.n_channels, n), dtype=np.float64)
        for i in range(n):
            t = i * self.dt
            pi = photic_fn(t) if photic_fn is not None else 0.0
            out[:, i] = self.step(pi)
        return out
