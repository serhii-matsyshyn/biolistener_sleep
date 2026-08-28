"""
EEG ingestion - unified BrainFlow ``BoardShim`` wrapper that consumes
samples from either the synthetic NMM TCP server, the real BioListener
device, or a playback CSV.

The agent code never knows which source it is - the swap from sim ->
real is a config-only change.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
    BrainFlowPresets,
)
from brainflow.exit_codes import BrainFlowError

logger = logging.getLogger("eeg.ingest")


@dataclass
class IngestConfig:
    """One config for all three sources - `kind` selects the path."""
    kind: str = "sim"                     # "sim" | "biolistener" | "playback"
    ip_address: str = "0.0.0.0"
    ip_port: int = 12345
    # Per-attempt prepare_session timeout. BIOLISTENER waits for the device
    # to dial in; if it isn't on yet the call blocks for `timeout` seconds
    # before failing; the total budget comes from `connect_attempts`.
    timeout: int = 10
    # What the device actually streams, which is not always what it was asked
    # for: a BIOLISTENER told 500 rounds to a power-of-two divider and streams
    # 512, while ``BoardShim.get_sampling_rate`` keeps reporting 500. Every
    # frequency in the analysis is scaled by the ratio, so this must be the
    # measured rate. ``app.protocol.analysis.measure_sampling_rate`` reads it
    # back off a recording's own timestamps.
    # What the device delivers, used to size windows and to analyse. Not
    # necessarily what it was asked for - see ``requested_data_rate``.
    sampling_rate: int = 500
    # What ``data_rate`` the device is commanded with. ``None`` sends
    # ``sampling_rate``. These are separate because this board answers a
    # commanded 500 with 511.995 Hz, so the value that works and the value
    # that describes the stream are different numbers; sending the measured
    # one back as a command would be asking for a rate never tested on it.
    requested_data_rate: Optional[int] = None
    # Must match `configs/daemon_prod.yaml` and the gain the device is running.
    # The AD7771's PGA accepts 1, 2, 4, 8 or 16; the railed detector
    # reconstructs ADC full scale from this, so a wrong value mis-scales every
    # rail check.
    pga_gain: int = 8
    # Channels physically connected to electrodes. Any of the 8 BIOLISTENER
    # channels NOT in this list gets {"command":2,"channel":N,"enable":0}
    # sent at session start so the device stops streaming on that input
    # (otherwise the floating ADC rails and pollutes railed_pct).
    active_channels: List[int] = field(
        default_factory=lambda: list(range(8))
    )
    # Total connection window = connect_attempts * (timeout + connect_retry_delay_s).
    # 6 attempts * ~10.5 s ≈ 63 s, enough time for the operator to power
    # the BIOLISTENER on after starting the session, but bounded so a
    # truly absent device fails loudly instead of hanging the loop.
    connect_attempts: int = 6
    connect_retry_delay_s: float = 0.5
    # Playback-only
    playback_file: Optional[Path] = None
    master_board: int = BoardIds.BIOLISTENER_BOARD.value
    # Optional: extra config_board JSON lines to send on prepare
    extra_config_board: List[str] = field(default_factory=list)
    # Optional CSV streamer destination
    record_to: Optional[Path] = None


class EEGIngestor:
    """
    Manages BoardShim lifecycle + a background thread that exposes the
    most recent window via :meth:`latest_window`.

    Caller pattern::

        ing = EEGIngestor(config)
        ing.start()
        # ... later, every control tick:
        window = ing.latest_window(window_s=4.0)
        # ... shut down:
        ing.stop()
    """

    def __init__(self, config: IngestConfig):
        self.config = config
        self.board_id = self._select_board_id()
        self._board: Optional[BoardShim] = None
        self._sampling_rate = config.sampling_rate
        self._eeg_channels: List[int] = []
        self._lock = threading.Lock()

    def _build_params(self) -> BrainFlowInputParams:
        """Fresh InputParams per attempt: a failed prepare_session() leaves
        the BoardShim unusable, so both are rebuilt together."""
        params = BrainFlowInputParams()
        params.ip_address = self.config.ip_address
        params.ip_port = self.config.ip_port
        params.timeout = self.config.timeout
        if self.config.kind == "playback":
            if self.config.playback_file is None:
                raise ValueError("playback_file required for kind='playback'")
            params.file = str(self.config.playback_file)
            params.master_board = self.config.master_board
        return params

    def _select_board_id(self) -> int:
        if self.config.kind == "sim":
            # sim driver = same BioListener wire format via NMM TCP server.
            # Caller is expected to spin up NMMBioListenerDevice separately.
            return BoardIds.BIOLISTENER_BOARD.value
        if self.config.kind == "biolistener":
            return BoardIds.BIOLISTENER_BOARD.value
        if self.config.kind == "playback":
            return BoardIds.PLAYBACK_FILE_BOARD.value
        raise ValueError(f"Unknown ingest kind: {self.config.kind!r}")

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        """Adopt a measured rate, so window lengths follow it too."""
        self._sampling_rate = int(value)

    @property
    def eeg_channels(self) -> List[int]:
        return list(self._eeg_channels)

    def _release_quietly(self, board: Optional[BoardShim]) -> None:
        """Drop a half-initialized BoardShim. After a failed prepare_session()
        the instance is unusable but may still hold the listening socket;
        a best-effort release frees it so the next attempt can bind."""
        if board is None:
            return
        try:
            if board.is_prepared():
                board.release_session()
        except Exception:
            pass

    def _prepare_with_retries(self) -> BoardShim:
        """Open a fresh BoardShim, retrying prepare_session() until the
        device dials in or the connection budget runs out. The total window
        is ``connect_attempts * (timeout + connect_retry_delay_s)``. For
        'sim' and 'playback' the first attempt always succeeds."""
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.config.connect_attempts + 1):
            board = BoardShim(self.board_id, self._build_params())
            try:
                board.prepare_session()
                if attempt > 1:
                    logger.info("EEGIngestor connected on attempt %d/%d",
                                attempt, self.config.connect_attempts)
                return board
            except BrainFlowError as e:
                last_err = e
                self._release_quietly(board)
                logger.warning(
                    "EEGIngestor prepare_session attempt %d/%d failed: %s "
                    "(device not pinging at %s:%d yet). Retrying in %.1fs.",
                    attempt, self.config.connect_attempts, e,
                    self.config.ip_address, self.config.ip_port,
                    self.config.connect_retry_delay_s,
                )
                time.sleep(self.config.connect_retry_delay_s)
        raise RuntimeError(
            f"EEGIngestor could not connect to device at "
            f"{self.config.ip_address}:{self.config.ip_port} after "
            f"{self.config.connect_attempts} attempts (last error: {last_err}). "
            "Power-cycle the BIOLISTENER and re-start the session."
        )

    def start(self) -> None:
        self._board = self._prepare_with_retries()

        # BIOLISTENER-specific runtime config. Each command gets its own
        # guard: sharing one meant a rejected data_rate skipped the gain and
        # channel-disable calls that follow it, leaving floating inputs
        # streaming and the railed detector reading against the wrong gain,
        # with a single warning line as the only trace.
        if self.config.kind in {"sim", "biolistener"}:
            rate = self.config.requested_data_rate or self.config.sampling_rate
            try:
                self._board.config_board(
                    f'{{"command": 1, "data_rate": {int(rate)}}}\n')
            except Exception:
                logger.warning("data_rate %s was refused; the device keeps its "
                               "current rate, which pre-flight measures anyway",
                               rate, exc_info=True)
            active = set(int(c) for c in self.config.active_channels)
            for ch in range(8):
                try:
                    if ch in active:
                        # Set gain on connected channels.
                        self._board.config_board(
                            f'{{"command": 3, "channel": {ch}, '
                            f'"pga": {self.config.pga_gain}}}\n'
                        )
                    else:
                        # Disable disconnected channels so the device stops
                        # streaming on floating inputs (railed ADC noise).
                        self._board.config_board(
                            f'{{"command": 2, "channel": {ch}, "enable": 0}}\n'
                        )
                except Exception:
                    logger.warning("channel %d config failed", ch, exc_info=True)
        for extra in self.config.extra_config_board:
            try:
                self._board.config_board(extra)
            except Exception:
                logger.warning("extra config_board %r failed", extra,
                               exc_info=True)

        if self.config.record_to is not None:
            self.config.record_to.parent.mkdir(parents=True, exist_ok=True)
            self._board.add_streamer(f"file://{self.config.record_to}:w")
            logger.info("Recording EEG to %s", self.config.record_to)

        self._board.start_stream(450_000)
        # A multicast streamer lets a second process subscribe to the same
        # samples. Optional - the primary path does not depend on it.
        if self.config.kind in {"sim", "biolistener"}:
            try:
                self._board.add_streamer(
                    "streaming_board://225.1.1.1:6677",
                    BrainFlowPresets.DEFAULT_PRESET,
                )
            except Exception:
                logger.warning("add_streamer(multicast) failed (ignored)",
                               exc_info=True)
        # Cache derived metadata. The configured rate wins: BoardShim reports
        # the board's nominal figure, which the device does not have to honour,
        # and taking it here silently discarded the measured value.
        board_rate = BoardShim.get_sampling_rate(self.board_id)
        self._sampling_rate = int(self.config.sampling_rate or board_rate)
        if board_rate != self._sampling_rate:
            logger.warning(
                "board reports %d Hz but config says %d Hz; using %d. Every "
                "measured frequency scales by this ratio, so confirm it against "
                "a recording's timestamps before trusting a session.",
                board_rate, self._sampling_rate, self._sampling_rate)
        self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        logger.info("EEGIngestor started (board=%d, sr=%d, channels=%s)",
                    self.board_id, self._sampling_rate, self._eeg_channels)

    def stop(self) -> None:
        if self._board is not None:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            try:
                if self._board.is_prepared():
                    self._board.release_session()
            except Exception:
                pass
            self._board = None
            logger.info("EEGIngestor stopped")

    def sample_count(self) -> Optional[int]:
        """Samples the board has buffered, for measuring the real stream rate.

        Non-destructive and monotonic until the ring fills, which at 512 Hz
        takes about fifteen minutes - far longer than the pre-flight window
        this is used in.
        """
        if self._board is None:
            return None
        try:
            return int(self._board.get_board_data_count())
        except Exception:
            logger.debug("get_board_data_count unavailable", exc_info=True)
            return None

    def latest_window(self, window_s: float = 4.0) -> np.ndarray:
        """
        Return last ``window_s`` of data as ``(n_eeg_channels, n_samples)``
        in µV. Returns empty array if no samples available yet.
        """
        if self._board is None:
            return np.empty((0, 0))
        n = int(window_s * self._sampling_rate)
        with self._lock:
            data = self._board.get_current_board_data(n)
        if data.size == 0:
            return np.empty((0, 0))
        eeg = data[self._eeg_channels, :]
        return eeg
