"""
UDP/JSON client for the strobe mask - vendored & adapted from
``mask_strobe_controller/mask_controller.py`` in the mask hardware.

Wire protocol is unchanged. Same client works against either:
- the real ESP32 mask discovered via inbound pings
- the virtual mask emulator in ``app.mask.virtual`` (listens on 127.0.0.1:20000
  and emits pings as well so this client thinks it's talking to a real device)
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("mask.client")


class MaskClient:
    """
    Discover the mask via inbound UDP pings, then send CMD_CHANGE_PARAMS /
    CMD_GET_PARAMS messages back to it.

    Parameters
    ----------
    bind_ip:
        Local interface to bind to. ``"0.0.0.0"`` accepts pings from any
        network adapter (use for real ESP32); ``"127.0.0.1"`` to scope to
        the virtual mask running on loopback.
    port:
        UDP port. ESP32 mask defaults to 20000; the virtual mask matches.
    target_ip:
        Optionally pin the target IP instead of waiting for ping discovery
        (useful when both ends are on localhost).
    on_params:
        Optional callback fired when the device sends ``{"type":"params"}``.
    """

    def __init__(
        self,
        bind_ip: str = "0.0.0.0",
        port: int = 20000,
        target_ip: Optional[str] = None,
        on_params: Optional[Callable[[dict], None]] = None,
    ):
        self.bind_ip = bind_ip
        self.port = port
        self.target_ip = target_ip
        self.board_ip = target_ip
        self.last_ping = 0.0
        self.on_params = on_params

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_ip, port))
        self.sock.settimeout(1.0)
        self.running = True

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def _listen(self) -> None:
        logger.info("MaskClient listening on %s:%d", self.bind_ip, self.port)
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            kind = msg.get("type")
            if kind == "ping":
                if self.board_ip != addr[0]:
                    logger.info("Mask discovered at %s", addr[0])
                self.board_ip = addr[0]
                self.last_ping = time.time()
            elif kind == "params":
                if self.on_params is not None:
                    try:
                        self.on_params(msg)
                    except Exception:
                        logger.exception("on_params callback failed")

    def _send(self, command: dict) -> bool:
        if not self.board_ip:
            logger.debug("Mask IP not discovered yet - dropping command")
            return False
        try:
            payload = json.dumps(command).encode("utf-8")
            self.sock.sendto(payload, (self.board_ip, self.port))
            return True
        except OSError as err:
            logger.error("send failed: %s", err)
            return False

    def update_params(
        self,
        enabled: Optional[bool] = None,
        frequency: Optional[float] = None,
        fill: Optional[float] = None,
        brightness: Optional[float] = None,
        color: Optional[List[int]] = None,
        mask: Optional[List[bool] | int] = None,
    ) -> bool:
        cmd: dict = {"command": 0}  # CMD_CHANGE_PARAMS
        if isinstance(mask, list):
            bits = 0
            for i, on in enumerate(mask):
                if on:
                    bits |= 1 << i
            mask = bits
        for key, val in dict(
            enabled=enabled, frequency=frequency, fill=fill,
            brightness=brightness, color=color, mask=mask,
        ).items():
            if val is not None:
                cmd[key] = val
        return self._send(cmd)

    def get_params(self) -> bool:
        return self._send({"command": 1})  # CMD_GET_PARAMS

    def is_connected(self) -> bool:
        if self.target_ip is not None:
            return True  # pinned target - always considered reachable
        return self.board_ip is not None and (time.time() - self.last_ping < 10)

    def stop(self) -> None:
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> "MaskClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
