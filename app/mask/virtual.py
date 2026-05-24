"""
Virtual strobe mask - UDP/JSON server protocol-compatible with the real
ESP32 BioListener mask.

Wire protocol mirrored verbatim:
- Listens on UDP port 20000 by default.
- Periodically emits ``{"type":"ping","ip":"<self_ip>"}`` to the controller
  so MaskClient can discover us, same way the real mask does.
- Accepts ``CMD_CHANGE_PARAMS`` (command:0) and updates internal state.
- Accepts ``CMD_GET_PARAMS`` (command:1) and replies with the current state
  as ``{"type":"params", ...}``.

State is exposed via :meth:`snapshot` for the desktop visualizer (or web
monitor) and for the closed-loop agent's strobe-envelope log.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import socket
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger("mask.virtual")

CMD_CHANGE_PARAMS = 0
CMD_GET_PARAMS = 1


@dataclasses.dataclass
class MaskState:
    enabled: bool = False
    frequency: float = 10.0       # Hz
    fill: float = 0.2             # duty cycle 0..1
    brightness: float = 0.3       # 0..1
    color: Tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBW 0..255
    mask: int = 0xFF              # bitmask, default all 8 LEDs on
    battery: float = 4.05         # volts, mock for a healthy LiPo

    def to_params_message(self) -> dict:
        return {
            "type": "params",
            "enabled": self.enabled,
            "frequency": self.frequency,
            "fill": self.fill,
            "brightness": self.brightness,
            "color": list(self.color),
            "mask": self.mask,
            "battery": self.battery,
        }


class VirtualMaskServer(threading.Thread):
    """
    Drop-in virtual ESP32 strobe mask.

    Usage::

        vm = VirtualMaskServer(controller_addr=("127.0.0.1", 20000))
        vm.start()
        ...
        vm.stop()
    """

    def __init__(
        self,
        listen_ip: str = "127.0.0.1",
        listen_port: int = 20001,            # different default from controller
        controller_addr: Tuple[str, int] = ("127.0.0.1", 20000),
        ping_interval_s: float = 2.0,
        device_ip_advertise: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.controller_addr = controller_addr
        self.ping_interval_s = ping_interval_s
        self.device_ip_advertise = device_ip_advertise or listen_ip

        self._state = MaskState()
        self._lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_ip, listen_port))
        self.sock.settimeout(0.5)

        self._keep_alive = True
        self._last_ping = 0.0
        self._param_history: list[tuple[float, MaskState]] = []
        self._history_lock = threading.Lock()

    def snapshot(self) -> MaskState:
        with self._lock:
            return dataclasses.replace(self._state)

    def history(self) -> list[tuple[float, MaskState]]:
        """Return (timestamp, state) entries for every CMD_CHANGE_PARAMS."""
        with self._history_lock:
            return list(self._param_history)

    def stop(self) -> None:
        self._keep_alive = False

    def run(self) -> None:
        logger.info("VirtualMaskServer listening %s:%d, controller @ %s:%d",
                    self.listen_ip, self.listen_port,
                    self.controller_addr[0], self.controller_addr[1])
        # Emit a couple of pings up-front so the controller discovers us fast
        self._send_ping()
        while self._keep_alive:
            now = time.time()
            if now - self._last_ping >= self.ping_interval_s:
                self._send_ping()
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_packet(data, addr)
        try:
            self.sock.close()
        except OSError:
            pass
        logger.info("VirtualMaskServer stopped")

    def _send_ping(self) -> None:
        ping = json.dumps({"type": "ping", "ip": self.device_ip_advertise})
        try:
            self.sock.sendto(ping.encode("utf-8"), self.controller_addr)
            self._last_ping = time.time()
        except OSError as err:
            logger.debug("ping send failed: %s", err)

    def _handle_packet(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Bad packet from %s", addr)
            return
        cmd = msg.get("command")
        if cmd == CMD_CHANGE_PARAMS:
            self._apply_change(msg)
        elif cmd == CMD_GET_PARAMS:
            self._reply_params(addr)
        else:
            logger.warning("Unknown command %r from %s", cmd, addr)

    def _apply_change(self, msg: dict) -> None:
        with self._lock:
            s = self._state
            if "enabled" in msg:
                s.enabled = bool(msg["enabled"])
            if "frequency" in msg:
                s.frequency = float(msg["frequency"])
            if "fill" in msg:
                s.fill = max(0.0, min(1.0, float(msg["fill"])))
            if "brightness" in msg:
                s.brightness = max(0.0, min(1.0, float(msg["brightness"])))
            if "color" in msg:
                col = msg["color"]
                if isinstance(col, list) and len(col) in (3, 4):
                    rgbw = list(col) + ([0] if len(col) == 3 else [])
                    s.color = (int(rgbw[0]), int(rgbw[1]),
                               int(rgbw[2]), int(rgbw[3]))
            if "mask" in msg:
                s.mask = int(msg["mask"]) & 0xFFFFFFFFFFFFFFFF
            snap = dataclasses.replace(s)
        with self._history_lock:
            self._param_history.append((time.time(), snap))

    def _reply_params(self, addr: Tuple[str, int]) -> None:
        payload = json.dumps(self.snapshot().to_params_message()).encode("utf-8")
        try:
            self.sock.sendto(payload, self.controller_addr)
        except OSError as err:
            logger.debug("params reply failed: %s", err)
