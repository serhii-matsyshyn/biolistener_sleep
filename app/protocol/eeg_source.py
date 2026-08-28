"""
One window-shaped interface over the two EEG paths.

The runner asks for ``latest_window`` and pushes the delivered strobe back with
``update_strobe``; whether the samples came from an amplifier or from the neural
mass model is not its concern. The simulated path has to be told about the
strobe, otherwise the loop is not closed and a rehearsal proves nothing.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from app.brain_sim.photic_input import StrobeParams

logger = logging.getLogger("protocol.eeg_source")


class EegSource(Protocol):
    def start(self) -> None: ...
    def latest_window(self, window_s: float) -> np.ndarray: ...
    def update_strobe(self, strobe: StrobeParams) -> None: ...
    def stop(self) -> None: ...

    @property
    def samples_emitted(self) -> Optional[int]:
        """
        Samples written to the raw file so far, or ``None`` if unknown.

        Recorded each tick so the analysis can index into the raw recording
        exactly, instead of assuming the stream started when the session did.
        Real recordings fall back to BrainFlow's timestamp channel.
        """

    def set_drowsiness(self, value: float) -> None:
        """
        Feed the measured drowsiness back into the plant.

        The simulator moves its resonance peak as a function of this, and the
        training env does exactly this each step. Without it the peak never
        moves, the loop is only half closed, and a rehearsal cannot verify a
        shift that never happens. A no-op on hardware.
        """

    def raw_eeg_rows(self) -> Optional[list]:
        """
        Which rows of the raw recording hold EEG, in montage order.

        The two paths write different files: BrainFlow's streamer writes every
        board row, so EEG starts at row 1 behind the packet counter, while the
        simulated path writes the eight channels bare. Analysis that assumes
        one layout reads a sawtooth counter on the other.
        """
        return None

    def ground_truth(self) -> dict:
        """
        Plant state that is knowable only in simulation, empty on hardware.

        This is what makes a full-length rehearsal a verification rather than a
        smoke test: the fitted resonance peak can be compared against the value
        the simulator actually used, so the fit's bias and interval coverage are
        measured before a night is spent collecting data.
        """
        return {}


class RealSource:
    """BrainFlow device or CSV playback, via :class:`EEGIngestor`."""

    def __init__(self, config) -> None:
        from app.eeg.ingest import EEGIngestor
        self._ing = EEGIngestor(config)

    @property
    def sampling_rate(self) -> int:
        return self._ing.sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        """Write through, so a measured rate resizes the windows as well."""
        self._ing.sampling_rate = int(value)

    def start(self) -> None:
        self._ing.start()

    def latest_window(self, window_s: float) -> np.ndarray:
        w = self._ing.latest_window(window_s=window_s)
        if w.size == 0:
            return np.zeros((8, 1), dtype=np.float64)
        return w

    def update_strobe(self, strobe: StrobeParams) -> None:
        """The real brain gets the strobe through the mask, not through here."""

    @property
    def samples_emitted(self) -> Optional[int]:
        return None          # BrainFlow owns the file; ticks anchor by time

    def stream_count(self) -> Optional[int]:
        """Running sample count, for measuring the delivered rate."""
        return self._ing.sample_count()

    def raw_eeg_rows(self) -> Optional[list]:
        from brainflow.board_shim import BoardShim
        try:
            return [int(c) for c in BoardShim.get_eeg_channels(self._ing.board_id)]
        except Exception:
            logger.warning("could not read board row layout", exc_info=True)
            return None

    def set_drowsiness(self, value: float) -> None:
        """The real brain gets there on its own."""

    def ground_truth(self) -> dict:
        """No such thing on a real head - the columns stay blank."""
        return {}

    def stop(self) -> None:
        self._ing.stop()


class SimSource:
    """
    In-process neural mass model behind a ring buffer.

    Steps in wall-clock time so a rehearsal has the same timing behaviour as a
    real session. If the loop falls behind, the catch-up is capped: better to
    return a slightly short window than to spend the next tick's budget
    simulating the last one.
    """

    MAX_CATCHUP_S = 2.0

    def __init__(
        self,
        sampling_rate: int = 500,
        window_capacity_s: float = 10.0,
        seed: Optional[int] = None,
        record_to: Optional[Path] = None,
        exogenous_shift: bool = False,
    ) -> None:
        from app.brain_sim.brainflow_board import InProcessBrainSim
        from app.brain_sim.domain_random import RandomizationRanges

        # ``exogenous_shift`` drives the resonance peak from a fixed time course
        # the controller cannot influence. Rehearsal only, and the reason it
        # exists here: under normal stimulation the measured drowsiness stays
        # well below the shift threshold, so the peak never moves and a
        # rehearsal has nothing for the fit to recover.
        subject = RandomizationRanges(exogenous_shift=exogenous_shift).sample(seed)
        self._sim = InProcessBrainSim(photic_config=subject.photic_config)
        self.sampling_rate = int(round(1.0 / self._sim.brain.dt))
        self._cap = int(window_capacity_s * self.sampling_rate)
        self._buf: Optional[np.ndarray] = None
        self._n = 0
        self._t0 = 0.0
        self._emitted = 0
        self._fh = record_to.open("w", buffering=1 << 16) if record_to else None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._emitted = 0

    def _advance(self) -> None:
        due = int((time.perf_counter() - self._t0) * self.sampling_rate)
        want = due - self._emitted
        cap = int(self.MAX_CATCHUP_S * self.sampling_rate)
        if want > cap:
            logger.debug("sim behind by %d samples, capping", want)
            want = cap
        for _ in range(max(0, want)):
            s = np.asarray(self._sim.step(), dtype=np.float64).ravel()
            if self._buf is None:
                self._buf = np.zeros((s.size, self._cap), dtype=np.float64)
            self._buf[:, self._n % self._cap] = s
            self._n += 1
            self._emitted += 1
            if self._fh is not None:
                self._fh.write("\t".join(f"{v:.6f}" for v in s) + "\n")

    def latest_window(self, window_s: float) -> np.ndarray:
        self._advance()
        if self._buf is None or self._n == 0:
            return np.zeros((8, 1), dtype=np.float64)
        n = min(self._n, self._cap, int(window_s * self.sampling_rate))
        if self._n < self._cap:
            return self._buf[:, max(0, self._n - n):self._n]
        pos = self._n % self._cap
        rolled = np.concatenate([self._buf[:, pos:], self._buf[:, :pos]], axis=1)
        return rolled[:, -n:]

    def update_strobe(self, strobe: StrobeParams) -> None:
        self._sim.update_strobe(strobe)

    def stream_count(self) -> Optional[int]:
        """Running sample count, for measuring the delivered rate."""
        return self.samples_emitted

    @property
    def samples_emitted(self) -> Optional[int]:
        return self._n

    def raw_eeg_rows(self) -> Optional[list]:
        return list(range(8))          # written bare, one column per channel

    def set_drowsiness(self, value: float) -> None:
        cfg = self._sim.photic_config
        # In the exogenous-drift condition the peak follows a fixed time course
        # instead and must not be overwritten - the same guard the env applies.
        if not cfg.exogenous_shift:
            cfg.current_drowsiness = float(value)

    def ground_truth(self) -> dict:
        cfg = self._sim.photic_config
        return {
            "true_resonance_hz": float(cfg.effective_resonance_hz),
            "true_drowsiness": float(cfg.current_drowsiness),
            "shift_per_unit_ratio": float(cfg.shift_per_unit_ratio),
        }

    def stop(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                logger.debug("sim csv close failed", exc_info=True)


def build_source(
    kind: str,
    sampling_rate: int,
    record_to: Optional[Path] = None,
    seed: Optional[int] = None,
    ip_address: str = "0.0.0.0",
    ip_port: int = 12345,
    playback_file: Optional[Path] = None,
    exogenous_shift: bool = False,
    active_channels: Optional[list] = None,
    pga_gain: int = 8,
    requested_data_rate: Optional[int] = None,
) -> EegSource:
    if kind == "sim":
        return SimSource(sampling_rate=sampling_rate, seed=seed,
                         record_to=record_to, exogenous_shift=exogenous_shift)
    from app.eeg.ingest import IngestConfig
    cfg = IngestConfig(
        kind=kind, ip_address=ip_address, ip_port=ip_port,
        sampling_rate=sampling_rate, playback_file=playback_file,
        record_to=record_to, pga_gain=pga_gain,
        requested_data_rate=requested_data_rate)
    if active_channels is not None:
        # Inputs with no electrode on them float and rail, so the device is
        # told which ones are wired rather than left streaming all eight.
        cfg.active_channels = list(active_channels)
    return RealSource(cfg)
