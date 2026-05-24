"""
Daemon "real-hardware path" verification - uses NMMBioListenerDevice as a
stand-in for the real ESP32 BioListener (same wire protocol).

Verifies the daemon's real-mode branch:
- ``eeg_source="biolistener:127.0.0.1:<port>"`` -> EEGIngestor via BoardShim
- ``mask_target="esp32:127.0.0.1:<port>"`` -> MaskClient against a VirtualMaskServer
  bound on that exact port (impersonating the real ESP32)
- Audio: defaults to silent in tests (BIOLISTENER_SLEEP_ENABLE_AUDIO=0 set below)

If this test passes, swapping the NMMBioListenerDevice + VirtualMaskServer
for a real ESP32 BioListener + real ESP32 mask is purely a config change.
"""
from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["MONGO_URL"] = "mongodb://127.0.0.1:1/"
os.environ["BIOLISTENER_SLEEP_ENABLE_AUDIO"] = "0"  # no audio device in CI


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_daemon_real_path_with_simulated_hardware(tmp_path):
    """
    Stand up the daemon in real mode, pointed at:
    - NMMBioListenerDevice (BioListener-wire-format TCP server)
    - VirtualMaskServer impersonating a real ESP32 mask on a TCP-style
      LAN address (we use loopback for the test)
    """
    from app.brain_sim.brainflow_board import NMMBioListenerDevice
    from app.brain_sim.nmm import MultiChannelBrain
    from app.mask.virtual import VirtualMaskServer

    eeg_port = _free_port()
    mask_port = _free_udp_port()

    # Stand up the EEG "real device" - speaks the same wire format as ESP32
    brain = MultiChannelBrain(n_channels=8, alpha_hz=10.0,
                              sampling_rate=500, seed=42)
    eeg_device = NMMBioListenerDevice(
        brain=brain, host="127.0.0.1", port=eeg_port, connect_timeout=20.0,
    )
    eeg_device.start()

    # Stand up the "real ESP32 mask" - same UDP/JSON wire format, on the LAN
    # port the daemon will try to talk to.
    mask_device = VirtualMaskServer(
        listen_ip="127.0.0.1",
        listen_port=mask_port,
        controller_addr=("127.0.0.1", mask_port),
        ping_interval_s=1.0,
    )
    mask_device.start()

    with patch("app.runtime.daemon.get_db") as mock_get_db, \
         patch("app.runtime.daemon.init_indexes",
               AsyncMock(return_value=None)):
        mock_db = MagicMock()
        mock_db.sessions.insert_one = AsyncMock()
        mock_db.sessions.update_one = AsyncMock()
        mock_db.safety_events.insert_one = AsyncMock()
        mock_get_db.return_value = mock_db

        from app.runtime import daemon
        daemon.STUDIES_ROOT = tmp_path / "studies"
        daemon.STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
        daemon.SESSIONS.clear()
        daemon.TICK_TASKS.clear()
        daemon.WEBSOCKETS.clear()

        try:
            resp = await daemon.create_session({
                "subject_code": "REAL01",
                "audio_enabled": False,
                "eeg_source": f"biolistener:127.0.0.1:{eeg_port}",
                "mask_target": f"esp32:127.0.0.1:{mask_port}",
            })
            sid = resp["id"]
            folder = Path(resp["folder"])
            assert folder.exists()

            await daemon.start_session(sid)
            # Give the real path time to: connect EEG, ingest some samples,
            # apply at least one stim command to the "real" mask
            await asyncio.sleep(3.0)

            snap = await daemon.session_status(sid)
            assert snap["tick"] >= 2, f"too few ticks: {snap['tick']}"

            await daemon.end_session(sid)

            # The "real" mask should have received at least one parameter
            # update (proves UDP path works end-to-end against an external IP)
            history = mask_device.history()
            assert len(history) > 0, "real mask received no commands"
            # State should reflect the last command from the daemon
            state = mask_device.snapshot()
            # FixedFrequencyPolicy emits ~10 Hz - should be clamped to safe range
            assert 3.0 <= state.frequency <= 14.0
        finally:
            mask_device.stop()
            mask_device.join(timeout=2)
            eeg_device.stop()
            eeg_device.join(timeout=5)
