"""
Integration test - true plug-and-play verification.

Spawns the NMM-driven BioListener TCP server and connects BrainFlow's
`BoardShim(BoardIds.BIOLISTENER_BOARD)` to it. Verifies that:

1. BrainFlow successfully receives 8-channel EEG samples.
2. The dominant frequency in the received signal is in the α band - meaning
   the wire-format conversion preserves the NMM dynamics end-to-end.

If this test passes, swapping the real ESP32 BioListener for our simulator
(or vice-versa) is just a config flag, not a code change.
"""
from __future__ import annotations

import logging
import socket
import time

import numpy as np
from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
)
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    WindowOperations,
)

from app.brain_sim.brainflow_board import NMMBioListenerDevice
from app.brain_sim.nmm import MultiChannelBrain
from app.brain_sim.photic_input import PhoticInputConfig

logger = logging.getLogger(__name__)


def _free_port() -> int:
    """Reserve a free TCP port (close immediately and reuse)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_brainflow_receives_alpha_from_nmm_server():
    """
    Start BrainFlow FIRST (it binds the listening port), then launch the NMM
    device which connects outbound - mirrors the real BioListener's behavior.
    """
    port = _free_port()
    brain = MultiChannelBrain(
        n_channels=8, alpha_hz=10.0, sampling_rate=500, seed=42,
    )

    params = BrainFlowInputParams()
    params.ip_address = "127.0.0.1"
    params.ip_port = port
    params.timeout = 15

    device = NMMBioListenerDevice(
        brain=brain,
        photic_config=PhoticInputConfig(),
        host="127.0.0.1",
        port=port,
        connect_timeout=20.0,
    )

    # Start the device FIRST - it'll spin on connect-retry every 100ms until
    # BrainFlow's prepare_session() opens the listening socket and accepts.
    device.start()

    board = BoardShim(BoardIds.BIOLISTENER_BOARD.value, params)
    try:
        board.prepare_session()
        assert device.wait_until_connected(timeout=5.0), \
            "Device did not establish connection"
        board.start_stream(450_000)
        time.sleep(10.0)  # gather enough samples for a clean PSD
        data = board.get_board_data()
        board.stop_stream()
    finally:
        device.stop()
        device.join(timeout=5)
        if board.is_prepared():
            board.release_session()

    sr = BoardShim.get_sampling_rate(BoardIds.BIOLISTENER_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.BIOLISTENER_BOARD.value)
    assert data.shape[1] > 0, "BrainFlow received no samples"
    # Python NMM + TCP serialization runs slower than the device's nominal
    # 500 SPS; require enough samples for a clean ~2-second PSD window.
    assert data.shape[1] >= 1024, (
        f"Only got {data.shape[1]} samples - insufficient for PSD"
    )

    # Use the EFFECTIVE sampling rate from packet timestamps rather than the
    # nominal one, since the Python emitter paces below 500 SPS.
    ts_chan = BoardShim.get_timestamp_channel(BoardIds.BIOLISTENER_BOARD.value)
    timestamps = data[ts_chan]
    elapsed = float(timestamps[-1] - timestamps[0])
    effective_sr = (len(timestamps) - 1) / elapsed if elapsed > 0 else sr
    logger.info("Effective SR: %.1f Hz (nominal %d Hz, %d samples in %.2fs)",
                effective_sr, sr, data.shape[1], elapsed)

    # Take the first EEG channel, run BrainFlow's own PSD pipeline at the
    # effective sample rate.
    eeg = data[eeg_channels[0]].copy()
    eeg = eeg[len(eeg) // 4:]  # drop transient quarter
    DataFilter.detrend(eeg, DetrendOperations.CONSTANT.value)
    effective_sr_int = max(64, int(round(effective_sr)))
    nfft = DataFilter.get_nearest_power_of_two(effective_sr_int)
    psd = DataFilter.get_psd_welch(
        eeg, nfft, nfft // 2, effective_sr_int,
        WindowOperations.BLACKMAN_HARRIS.value,
    )

    # Locate dominant freq in 4..30 Hz
    psd_values = np.asarray(psd[0])
    psd_freqs = np.asarray(psd[1])
    band_mask = (psd_freqs >= 4.0) & (psd_freqs <= 30.0)
    peak_freq = float(psd_freqs[band_mask][np.argmax(psd_values[band_mask])])
    logger.info("Round-trip peak frequency: %.2f Hz", peak_freq)
    assert 7.0 <= peak_freq <= 13.5, (
        f"End-to-end peak {peak_freq:.2f} Hz is outside α band - "
        "wire-format conversion may be wrong"
    )
