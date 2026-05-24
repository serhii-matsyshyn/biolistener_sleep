"""
Verify the virtual mask is wire-compatible with MaskClient.
"""
from __future__ import annotations

import time

from app.mask.client import MaskClient
from app.mask.virtual import VirtualMaskServer


def test_client_discovers_virtual_mask_via_ping():
    """Controller listens on 20020; virtual mask pings it; client discovers."""
    controller_port = 20020
    device_port = 20021
    client = MaskClient(bind_ip="127.0.0.1", port=controller_port)
    server = VirtualMaskServer(
        listen_ip="127.0.0.1",
        listen_port=device_port,
        controller_addr=("127.0.0.1", controller_port),
        ping_interval_s=0.2,
    )
    try:
        server.start()
        # Wait for discovery
        deadline = time.time() + 5
        while time.time() < deadline and client.board_ip is None:
            time.sleep(0.05)
        assert client.board_ip == "127.0.0.1"
        assert client.is_connected()
    finally:
        server.stop()
        server.join(timeout=2)
        client.stop()


def test_change_params_roundtrip():
    """Client sends CMD_CHANGE_PARAMS; virtual mask state updates."""
    ctrl_port, dev_port = 20030, 20031
    client = MaskClient(bind_ip="127.0.0.1", port=ctrl_port)
    server = VirtualMaskServer(
        listen_ip="127.0.0.1",
        listen_port=dev_port,
        controller_addr=("127.0.0.1", ctrl_port),
        ping_interval_s=0.2,
    )
    try:
        server.start()
        # Wait for discovery + pin client target so send_command knows where
        deadline = time.time() + 5
        while time.time() < deadline and client.board_ip is None:
            time.sleep(0.05)
        assert client.board_ip is not None

        # Tell the client where to send (force the right device port)
        client.board_ip = "127.0.0.1"
        client.port = dev_port

        assert client.update_params(
            enabled=True, frequency=8.5, fill=0.18,
            brightness=0.25, color=[0, 0, 0, 200],
            mask=[True] * 8,
        )
        time.sleep(0.2)
        state = server.snapshot()
        assert state.enabled is True
        assert abs(state.frequency - 8.5) < 1e-6
        assert abs(state.fill - 0.18) < 1e-6
        assert abs(state.brightness - 0.25) < 1e-6
        assert state.color == (0, 0, 0, 200)
        assert state.mask == 0xFF
    finally:
        server.stop()
        server.join(timeout=2)
        client.stop()


def test_get_params_reply():
    """Client requests params; virtual mask replies with current state."""
    ctrl_port, dev_port = 20040, 20041

    received = []

    def on_params(msg):
        received.append(msg)

    client = MaskClient(bind_ip="127.0.0.1", port=ctrl_port, on_params=on_params)
    server = VirtualMaskServer(
        listen_ip="127.0.0.1",
        listen_port=dev_port,
        controller_addr=("127.0.0.1", ctrl_port),
        ping_interval_s=0.2,
    )
    try:
        server.start()
        deadline = time.time() + 5
        while time.time() < deadline and client.board_ip is None:
            time.sleep(0.05)
        client.board_ip = "127.0.0.1"
        client.port = dev_port

        client.update_params(frequency=12.0)
        time.sleep(0.1)
        client.get_params()
        time.sleep(0.3)
        assert any(m.get("type") == "params" and abs(m.get("frequency", 0) - 12.0) < 1e-6
                   for m in received), f"got: {received}"
    finally:
        server.stop()
        server.join(timeout=2)
        client.stop()


def test_pse_band_acceptance_at_protocol_level():
    """
    Virtual mask is as-original hardware - it ACCEPTS any frequency including PSE
    band (15-25 Hz). The SafetySupervisor (added in M3) is what blocks
    dangerous commands BEFORE they leave the agent. This test pins that
    contract: device-side safety isn't where we enforce it.
    """
    ctrl_port, dev_port = 20050, 20051
    client = MaskClient(bind_ip="127.0.0.1", port=ctrl_port)
    server = VirtualMaskServer(
        listen_ip="127.0.0.1",
        listen_port=dev_port,
        controller_addr=("127.0.0.1", ctrl_port),
        ping_interval_s=0.2,
    )
    try:
        server.start()
        deadline = time.time() + 5
        while time.time() < deadline and client.board_ip is None:
            time.sleep(0.05)
        client.board_ip = "127.0.0.1"
        client.port = dev_port

        # 22 Hz is firmly in the PSE-risk band - device accepts; safety is upstream.
        client.update_params(enabled=True, frequency=22.0, brightness=0.9)
        time.sleep(0.2)
        state = server.snapshot()
        assert state.enabled is True
        assert abs(state.frequency - 22.0) < 1e-6
    finally:
        server.stop()
        server.join(timeout=2)
        client.stop()
