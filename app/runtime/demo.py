"""
Demo Open Loop - direct replication of the source mask_strobe_controller
Flask app, exposed as /api/demo/* endpoints. No session/Mongo writes.

  - VLC plays the chosen audio file (handles .m4a/.mp3 natively, supports
    seek + duration).
  - CsvEventPlayback parses the matched strobe-event CSV into a (time ->
    freq, fill) table.
  - A background sync thread runs at 20 Hz, queries VLC's current position,
    looks up (freq, fill) in the CSV, and pushes the merged parameters
    (freq, fill, brightness, color, bitmask, enabled) to the MaskClient.
  - Strobe is enabled iff (UI toggle ON) AND (audio playing) AND
    (audio_pos ≥ first_on_time).

Added on top of the source app:
  - ``generative_audio_enabled``: when on, the BiofeedbackEngine plays
    synth audio with its tremolo locked to the strobe Hz coming out of
    the CSV. Mutually-exclusive with the VLC track audio at the device
    level (PulseAudio mixes them, but we stop VLC when generative engages).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from app.agent.safety import SafetySupervisor, StimAction
from app.audio.controller import AudioController
from app.mask.client import MaskClient
from app.mask.virtual import VirtualMaskServer
from app.open_loop_track.csv_playback import CsvEventPlayback

logger = logging.getLogger("runtime.demo")

_SYNC_HZ = 20.0
_SYNC_PERIOD_S = 1.0 / _SYNC_HZ


@dataclass
class DemoMaskState:
    """User-controlled strobe parameters (freq + fill come from CSV)."""
    enabled: bool = True         # "Strobe Enabled" UI toggle
    brightness: float = 0.10     # 0..1 (UI shows 0..100%)
    color: List[int] = field(default_factory=lambda: [0, 0, 0, 255])  # RGBW
    bitmask: int = 0xFF          # 8-LED mask, all on by default


@dataclass
class DemoTrackState:
    """VLC playback state."""
    sound_path: Optional[str] = None
    strobe_path: Optional[str] = None
    duration_s: float = 0.0
    current_pos_s: float = 0.0
    is_playing: bool = False
    first_on_s: float = 0.0      # first rising edge in the strobe CSV


@dataclass
class DemoAudioState:
    """Generative-audio (BiofeedbackEngine) state. Opt-in."""
    generative_enabled: bool = False
    preset: str = "sunlight_pads"
    master_vol_pct: float = 50.0
    binaural_vol_pct: float = 10.0
    warmth_cutoff_pct: float = 80.0
    brown_noise_pct: float = 80.0
    reverb_space_pct: float = 80.0
    test_mode_on: bool = False    # "Stereo ping-pong" soundcheck


# Defaults restored by the UI's "Reset to defaults" button - these mirror
# the original biofeedback_sound_engine Tk app.
GENERATIVE_DEFAULTS = {
    "preset": "sunlight_pads",
    "master_vol_pct": 50.0,
    "binaural_vol_pct": 10.0,
    "warmth_cutoff_pct": 80.0,
    "brown_noise_pct": 80.0,
    "reverb_space_pct": 80.0,
    "test_mode_on": False,
}


class DemoController:
    """Singleton - boots resources on first access, kept alive for the
    daemon's lifetime, stopped on shutdown."""

    def __init__(self):
        self._lock = threading.RLock()
        self._initialised = False

        # Mask
        self._mask_client: Optional[MaskClient] = None
        self._mask_virtual: Optional[VirtualMaskServer] = None
        self._mask_kind: str = "virtual"   # set by _build_mask_resources
        # Same safety bounds the real-session daemon uses, pulled from
        # cfg.safety.*. Falls back to module-constant defaults when no
        # daemon config has been resolved (e.g. unit-test imports).
        from app.runtime import daemon as daemon_mod
        _cfg = getattr(daemon_mod, "RESOLVED_CONFIG", None)
        self._supervisor = (
            SafetySupervisor.from_config(_cfg.safety)
            if _cfg is not None else SafetySupervisor()
        )

        # VLC track player
        self._vlc_instance = None
        self._vlc_player = None
        self._strobe_csv: Optional[CsvEventPlayback] = None

        # Generative engine (lazy)
        self._audio_server = None
        self._audio_engine = None
        self._audio_controller: Optional[AudioController] = None

        # State
        self._mask_state = DemoMaskState()
        self._track_state = DemoTrackState()
        self._audio_state = DemoAudioState()

        # Sync thread
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_stop = threading.Event()
        self._last_mask_params: Optional[dict] = None


    def _teardown_mask_resources(self) -> None:
        if self._mask_client is not None:
            try:
                self._mask_client.update_params(enabled=False, brightness=0.0)
            except Exception:
                pass
            try:
                self._mask_client.stop()
            except Exception:
                pass
            self._mask_client = None
        if self._mask_virtual is not None:
            try:
                self._mask_virtual.stop()
                self._mask_virtual.join(timeout=2)
            except Exception:
                pass
            self._mask_virtual = None
        self._last_mask_params = None

    def _build_mask_resources(self, kind: str) -> None:
        from app.runtime import daemon
        cfg = getattr(daemon, "RESOLVED_CONFIG", None)
        port = cfg.mask.port if cfg is not None else 20000

        if kind == "esp32":
            # Source-app behaviour: bind 0.0.0.0:port, DO NOT pin target_ip -
            # the mask is the UDP client; it pings the controller every ~2s
            # and MaskClient sets ``board_ip`` from those pings. No IP entry
            # required; same UX as mask_strobe_controller.
            self._mask_client = MaskClient(
                bind_ip="0.0.0.0", port=port, target_ip=None,
            )
            logger.info("Demo: real ESP32 - listening for mask pings on 0.0.0.0:%d",
                        port)
        else:
            ctrl_port, dev_port = 23000, 23001
            self._mask_client = MaskClient(
                bind_ip="127.0.0.1", port=ctrl_port, target_ip="127.0.0.1",
            )
            self._mask_client.port = dev_port
            self._mask_virtual = VirtualMaskServer(
                listen_ip="127.0.0.1", listen_port=dev_port,
                controller_addr=("127.0.0.1", ctrl_port),
                ping_interval_s=1.0,
            )
            self._mask_virtual.start()
            logger.info("Demo: virtual mask on 127.0.0.1:%d", dev_port)
        self._mask_kind = kind

    def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        from app.runtime import daemon
        cfg = getattr(daemon, "RESOLVED_CONFIG", None)
        kind = cfg.mask.kind if cfg is not None else "virtual"
        self._build_mask_resources(kind)

        # VLC - match the source mask_strobe_controller flags exactly.
        # On Linux, --aout=alsa bypasses PulseAudio (which is prone to
        # underrun-driven choppiness when other clients are active).
        try:
            import platform
            import vlc
            flags = ["--no-video", "--quiet",
                     "--file-caching=1000", "--clock-jitter=0"]
            if platform.system() == "Linux":
                flags.append("--aout=alsa")
            self._vlc_instance = vlc.Instance(*flags)
            self._vlc_player = self._vlc_instance.media_player_new()
            logger.info("Demo: VLC instance ready (flags=%s)", flags)
        except Exception:
            logger.warning("Demo: VLC init failed - track audio disabled",
                           exc_info=True)
            self._vlc_instance = None
            self._vlc_player = None

        # Sync thread - always running; sends nothing until a session is loaded.
        self._sync_stop.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_worker, name="demo-sync", daemon=True,
        )
        self._sync_thread.start()
        self._initialised = True

    def shutdown(self) -> None:
        with self._lock:
            self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=1.0)
            self._sync_thread = None
        with self._lock:
            if self._vlc_player is not None:
                try:
                    self._vlc_player.stop()
                except Exception:
                    pass
                self._vlc_player = None
            self._vlc_instance = None
            self._stop_generative()
            self._teardown_mask_resources()
            self._initialised = False


    def _ensure_pyo_server(self) -> bool:
        if self._audio_server is not None:
            return True
        try:
            from app.audio.engine import AudioServer
            self._audio_server = AudioServer()
            self._audio_server.start()
            logger.info("Demo: pyo Server booted (generative)")
            return True
        except Exception:
            logger.warning("Demo: pyo Server boot failed - generative silent",
                           exc_info=True)
            self._audio_server = None
            return False

    def _build_generative_engine(self) -> bool:
        if self._audio_engine is not None:
            return True
        if not self._ensure_pyo_server():
            return False
        try:
            from app.audio.engine import BiofeedbackEngine
            # Same gain + safety caps the real-session daemon uses, so
            # the Demo page's generative audio isn't ~14 dB quieter than
            # VLC's pre-recorded .m4a playback AND can't exceed the
            # operator-set absolute val_vol / strobe-depth / binaural caps.
            from app.runtime import daemon as daemon_mod
            cfg = getattr(daemon_mod, "RESOLVED_CONFIG", None)
            if cfg is not None:
                ac = cfg.audio
                self._audio_engine = BiofeedbackEngine(
                    preset_name=self._audio_state.preset,
                    output_gain=float(ac.output_gain),
                    max_val_vol=float(ac.safety.max_val_vol),
                    max_strobe_depth_pct=float(ac.safety.max_strobe_depth_pct),
                    max_binaural_vol_pct=float(ac.safety.max_binaural_vol_pct),
                )
            else:
                self._audio_engine = BiofeedbackEngine(
                    preset_name=self._audio_state.preset,
                )
            self._audio_controller = AudioController(engine=self._audio_engine)
            self._push_generative_params()
            return True
        except Exception:
            logger.warning("Demo: generative engine build failed",
                           exc_info=True)
            self._audio_engine = None
            self._audio_controller = None
            return False

    def _stop_generative(self) -> None:
        if self._audio_controller is not None:
            try:
                self._audio_controller.set_master_vol(0.0)
            except Exception:
                pass
        self._audio_engine = None
        self._audio_controller = None

    def _push_generative_params(self) -> None:
        ctl = self._audio_controller
        if ctl is None:
            return
        a = self._audio_state
        try:
            ctl.set_master_vol(a.master_vol_pct)
            ctl.set_binaural_vol(a.binaural_vol_pct)
            ctl.set_warmth(a.warmth_cutoff_pct)
            ctl.set_brown_noise(a.brown_noise_pct)
            ctl.set_reverb(a.reverb_space_pct)
        except Exception:
            logger.warning("Demo: pushing generative params failed",
                           exc_info=True)


    def _sync_worker(self) -> None:
        while not self._sync_stop.is_set():
            try:
                self._sync_tick()
            except Exception:
                logger.exception("Demo sync tick raised")
            time.sleep(_SYNC_PERIOD_S)

    def _sync_tick(self) -> None:
        # Read VLC playhead under the lock-free side (VLC is internally safe).
        pos_s, is_playing, ended = self._read_vlc_position()

        with self._lock:
            t = self._track_state
            t.current_pos_s = pos_s
            if ended:
                t.is_playing = False
            else:
                t.is_playing = is_playing

            # Strobe params from CSV at the current playhead.
            freq_hz, fill = 0.0, 0.0
            if self._strobe_csv is not None and is_playing:
                got = self._strobe_csv.params_at(pos_s)
                if got is not None:
                    freq_hz, fill = got

            # Strobe is "active" only while audio plays and past first_on time.
            active_time = pos_s >= t.first_on_s
            send_enabled = bool(
                self._mask_state.enabled
                and (is_playing or self._audio_state.generative_enabled)
                and active_time
                and (self._strobe_csv is not None)
            )

            # Idle path - strobe not active for any reason (no CSV, paused,
            # pre-first-on, mask toggle off). We push a single "disabled"
            # frame and bail. Crucially, we NEVER run SafetySupervisor.clamp
            # on a zero-Hz "off" command - that produces one freq_bounds
            # event per tick (20/sec), drowning the safety-event log and
            # burning CPU. The clamp is for *validating actual stim*, and
            # there's nothing to stim here. This applies whether or not
            # generative audio is playing - if audio is on but strobe is
            # off, we still skip the clamp and just stop tremolo-modulating
            # by parking the engine's sync_freq at 0.
            if not send_enabled:
                params = {
                    "enabled": False,
                    "frequency": 0.0,
                    "fill": 0.0,
                    "brightness": self._mask_state.brightness,
                    "color": list(self._mask_state.color),
                    "mask": int(self._mask_state.bitmask) & 0xFF,
                }
                if params != self._last_mask_params and self._mask_client is not None:
                    try:
                        self._mask_client.update_params(**params)
                    except Exception:
                        logger.warning("Demo: mask update failed", exc_info=True)
                    self._last_mask_params = params
                if (self._audio_state.generative_enabled
                        and self._audio_controller is not None):
                    try:
                        self._audio_controller.set_sync_freq(0.0)
                    except Exception:
                        pass
                return

            # Active path - strobe driven by CSV, clamped via SafetySupervisor.
            requested = StimAction(
                enabled=True,
                frequency_hz=freq_hz,
                fill=fill,
                brightness=self._mask_state.brightness,
                color_white=self._mask_state.color[3] if len(self._mask_state.color) >= 4 else 255,
            )
            safe = self._supervisor.clamp(requested)

            params = {
                "enabled": safe.enabled,
                "frequency": safe.frequency_hz,
                "fill": safe.fill,
                "brightness": safe.brightness,
                "color": [self._mask_state.color[0],
                          self._mask_state.color[1],
                          self._mask_state.color[2],
                          safe.color_white],
                "mask": int(self._mask_state.bitmask) & 0xFF,
            }
            if params != self._last_mask_params and self._mask_client is not None:
                try:
                    self._mask_client.update_params(**params)
                except Exception:
                    logger.warning("Demo: mask update failed", exc_info=True)
                self._last_mask_params = params

            # Generative tremolo retracks to CSV freq.
            if (self._audio_state.generative_enabled
                    and self._audio_controller is not None):
                try:
                    self._audio_controller.set_sync_freq(safe.frequency_hz)
                except Exception:
                    pass

    def _read_vlc_position(self) -> tuple[float, bool, bool]:
        if self._vlc_player is None:
            return 0.0, False, False
        try:
            import vlc as _vlc
            state = self._vlc_player.get_state()
            ended = (state == _vlc.State.Ended)
            t_ms = self._vlc_player.get_time()
            pos_s = max(0.0, t_ms / 1000.0) if t_ms is not None else 0.0
            is_playing = state == _vlc.State.Playing
            return pos_s, is_playing, ended
        except Exception:
            return 0.0, False, False


    def get_state(self) -> dict:
        with self._lock:
            self._ensure_initialised()
            # Refresh VLC position once for the snapshot
            pos_s, is_playing, ended = self._read_vlc_position()
            self._track_state.current_pos_s = pos_s
            if ended:
                self._track_state.is_playing = False
            else:
                self._track_state.is_playing = is_playing
            mask_connected = (self._mask_client is not None
                              and self._mask_client.is_connected())
            return {
                "mask":   asdict(self._mask_state),
                "track":  asdict(self._track_state),
                "audio":  asdict(self._audio_state),
                "mask_kind": self._mask_kind,
                "mask_connected": mask_connected,
                "mask_board_ip":  (self._mask_client.board_ip
                                    if self._mask_client else None),
                "safety_events": [
                    {"t_ms": e.t_ms, "kind": e.kind, "field": e.field,
                     "before": e.before, "after": e.after, "reason": e.reason}
                    for e in self._supervisor.events[-10:]
                ],
            }

    def set_mask_kind(self, kind: str) -> dict:
        """Runtime swap between 'virtual' (loopback emulator) and 'esp32'
        (real mask via ping discovery on 0.0.0.0:20000)."""
        if kind not in {"virtual", "esp32"}:
            raise ValueError(f"unknown mask kind: {kind!r}")
        with self._lock:
            self._ensure_initialised()
            if kind == self._mask_kind:
                return self.get_state()
            self._teardown_mask_resources()
            self._build_mask_resources(kind)
        return self.get_state()

    def select_session(self, *, sound_path: Optional[str],
                       strobe_path: Optional[str]) -> dict:
        """Load a new track (VLC) + matching strobe CSV. Replicates the
        Flask app's POST /select_session."""
        with self._lock:
            self._ensure_initialised()
            # Stop any current playback and clear strobe data.
            self._stop_track_locked()
            self._strobe_csv = None
            self._track_state = DemoTrackState()

            if sound_path:
                ok = self._load_vlc_locked(Path(sound_path))
                if not ok:
                    return self.get_state()
                self._track_state.sound_path = str(sound_path)

            if strobe_path:
                try:
                    self._strobe_csv = CsvEventPlayback.from_csv(Path(strobe_path))
                    self._track_state.strobe_path = str(strobe_path)
                    # first_on_s ≈ first cycle start (in seconds)
                    if self._strobe_csv.n_cycles > 0:
                        first_ns = self._strobe_csv._starts[0]
                        self._track_state.first_on_s = first_ns / 1e9
                    logger.info("Demo: strobe CSV loaded (%d cycles, %.1fs)",
                                self._strobe_csv.n_cycles,
                                self._strobe_csv.duration_s)
                except Exception:
                    logger.warning("Demo: strobe CSV load failed",
                                   exc_info=True)
                    self._strobe_csv = None
        return self.get_state()

    def _load_vlc_locked(self, sound_path: Path) -> bool:
        if not sound_path.exists():
            logger.warning("Demo: audio file not found: %s", sound_path)
            return False
        if self._vlc_player is None:
            return False
        try:
            media = self._vlc_instance.media_new(str(sound_path))
            self._vlc_player.set_media(media)
            media.parse()
            self._track_state.duration_s = (media.get_duration() or 0) / 1000.0
            logger.info("Demo: VLC loaded %s (%.1fs)",
                        sound_path.name, self._track_state.duration_s)
            return True
        except Exception:
            logger.warning("Demo: VLC load failed for %s",
                           sound_path, exc_info=True)
            return False

    def play(self) -> dict:
        with self._lock:
            self._ensure_initialised()
            if self._vlc_player is not None and self._track_state.sound_path:
                try:
                    self._vlc_player.play()
                    self._track_state.is_playing = True
                except Exception:
                    logger.warning("Demo: VLC play failed", exc_info=True)
            # Generative tremolo (if enabled) is driven by the sync thread.
        return self.get_state()

    def pause(self) -> dict:
        with self._lock:
            self._ensure_initialised()
            if self._vlc_player is not None:
                try:
                    self._vlc_player.pause()
                    self._track_state.is_playing = False
                except Exception:
                    logger.warning("Demo: VLC pause failed", exc_info=True)
        return self.get_state()

    def seek(self, pos_s: float) -> dict:
        with self._lock:
            self._ensure_initialised()
            if self._vlc_player is not None:
                try:
                    self._vlc_player.set_time(int(max(0.0, pos_s) * 1000))
                except Exception:
                    logger.warning("Demo: VLC seek failed", exc_info=True)
        return self.get_state()

    def _stop_track_locked(self) -> None:
        if self._vlc_player is not None:
            try:
                self._vlc_player.stop()
            except Exception:
                pass
        self._track_state.is_playing = False
        self._track_state.current_pos_s = 0.0

    def update_params(self, *, brightness: Optional[float] = None,
                      color: Optional[List[int]] = None,
                      enabled: Optional[bool] = None,
                      bitmask: Optional[int] = None) -> dict:
        """Replicates the Flask app's POST /update_params."""
        with self._lock:
            self._ensure_initialised()
            s = self._mask_state
            if brightness is not None:
                s.brightness = max(0.0, min(1.0, float(brightness)))
            if color is not None:
                # [R, G, B, W] in 0..255.
                s.color = [int(c) & 0xFF for c in color][:4]
                while len(s.color) < 4:
                    s.color.append(255)
            if enabled is not None:
                s.enabled = bool(enabled)
            if bitmask is not None:
                s.bitmask = int(bitmask) & 0xFF
        return self.get_state()

    def _set_vlc_volume(self, vol_0_100: int) -> None:
        if self._vlc_player is None:
            return
        try:
            self._vlc_player.audio_set_volume(int(vol_0_100))
        except Exception:
            pass

    def update_audio(self, *, generative_enabled: Optional[bool] = None,
                     preset: Optional[str] = None,
                     master_vol_pct: Optional[float] = None,
                     binaural_vol_pct: Optional[float] = None,
                     warmth_cutoff_pct: Optional[float] = None,
                     brown_noise_pct: Optional[float] = None,
                     reverb_space_pct: Optional[float] = None,
                     test_mode_on: Optional[bool] = None,
                     reset: Optional[bool] = None) -> dict:
        """Generative audio knobs. When ``generative_enabled`` flips true the
        BiofeedbackEngine boots and VLC's volume is muted (its playhead keeps
        running so the CSV-driven strobe pattern stays on its original
        timeline - no audio overlay, but mask blinking unchanged)."""
        with self._lock:
            self._ensure_initialised()
            a = self._audio_state

            # Reset-to-defaults short-circuit (matches the original Tk app).
            if reset:
                self._stop_generative()
                self._set_vlc_volume(100)
                a.generative_enabled = False
                for k, v in GENERATIVE_DEFAULTS.items():
                    setattr(a, k, v)
                return self.get_state()

            if preset is not None and preset != a.preset:
                a.preset = str(preset)
                if a.generative_enabled and self._audio_engine is not None:
                    # Hot-swap preset by rebuilding engine.
                    self._stop_generative()
                    if not self._build_generative_engine():
                        a.generative_enabled = False

            if generative_enabled is not None:
                a.generative_enabled = bool(generative_enabled)
                if a.generative_enabled:
                    if not self._build_generative_engine():
                        a.generative_enabled = False
                    else:
                        # MUTE VLC so the two audio sources don't overlay;
                        # we keep VLC playing (if it was playing) so the
                        # playhead - and hence the CSV-driven strobe - keeps
                        # advancing on its original timeline.
                        self._set_vlc_volume(0)
                else:
                    self._stop_generative()
                    self._set_vlc_volume(100)

            # Slider updates - write into state always, push to engine if live.
            if master_vol_pct is not None:
                a.master_vol_pct = float(master_vol_pct)
            if binaural_vol_pct is not None:
                a.binaural_vol_pct = float(binaural_vol_pct)
            if warmth_cutoff_pct is not None:
                a.warmth_cutoff_pct = float(warmth_cutoff_pct)
            if brown_noise_pct is not None:
                a.brown_noise_pct = float(brown_noise_pct)
            if reverb_space_pct is not None:
                a.reverb_space_pct = float(reverb_space_pct)
            if test_mode_on is not None:
                a.test_mode_on = bool(test_mode_on)
                if a.generative_enabled and self._audio_controller is not None:
                    try:
                        self._audio_controller.set_test_mode(a.test_mode_on)
                    except Exception:
                        pass

            if a.generative_enabled and self._audio_controller is not None:
                self._push_generative_params()
                try:
                    self._audio_controller.set_test_mode(a.test_mode_on)
                except Exception:
                    pass
        return self.get_state()

    def stop_all(self) -> dict:
        """Emergency stop - pauses VLC, silences mask, kills generative."""
        with self._lock:
            self._ensure_initialised()
            self._stop_track_locked()
            self._mask_state.enabled = False
            self._mask_state.brightness = 0.0
            self._stop_generative()
            self._audio_state.generative_enabled = False
            self._audio_state.master_vol_pct = 0.0
            if self._mask_client is not None:
                try:
                    self._mask_client.update_params(enabled=False, brightness=0.0)
                except Exception:
                    pass
            self._last_mask_params = None
        return self.get_state()


# Module-level singleton
_DEMO: Optional[DemoController] = None


def get_demo() -> DemoController:
    global _DEMO
    if _DEMO is None:
        _DEMO = DemoController()
    return _DEMO


def shutdown_demo() -> None:
    global _DEMO
    if _DEMO is not None:
        _DEMO.shutdown()
        _DEMO = None
