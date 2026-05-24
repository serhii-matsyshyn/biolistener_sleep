"""
M6 plug-and-play verification - EEGIngestor (the unified abstraction used
by the daemon for real hardware) successfully sources samples from the
NMM TCP device through BrainFlow's BIOLISTENER driver.

Same code path will work against a real ESP32 BioListener - only the
IP/port differs. This test is the canary for that swap.
"""
from __future__ import annotations

import socket
import time

from app.brain_sim.brainflow_board import NMMBioListenerDevice
from app.brain_sim.nmm import MultiChannelBrain
from app.eeg.ingest import EEGIngestor, IngestConfig


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_eeg_ingestor_reads_from_nmm_device():
    port = _free_port()
    brain = MultiChannelBrain(n_channels=8, alpha_hz=10.0,
                              sampling_rate=500, seed=42)
    device = NMMBioListenerDevice(
        brain=brain, host="127.0.0.1", port=port, connect_timeout=20.0,
    )
    device.start()

    ing = EEGIngestor(IngestConfig(
        kind="sim", ip_address="127.0.0.1", ip_port=port,
        timeout=15, sampling_rate=500,
    ))
    try:
        ing.start()
        time.sleep(4.0)
        window = ing.latest_window(window_s=2.0)
        assert window.size > 0, "EEGIngestor returned no data"
        # 8 EEG channels expected
        assert window.shape[0] == 8
        # At least some samples - Python NMM is slow but should yield ≥100
        assert window.shape[1] >= 100, \
            f"only got {window.shape[1]} samples"
    finally:
        ing.stop()
        device.stop()
        device.join(timeout=5)
