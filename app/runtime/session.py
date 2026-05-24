"""
Session lifecycle - one of the daemon's longest-lived state objects.

States: ``idle -> baseline -> running -> completed | aborted``

Holds references to the EEG ingestor, mask client, audio controller,
agent policy, safety supervisor, session writer, and Mongo handles.
The daemon's control tick coroutine drives state transitions.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from app.agent.safety import SafetySupervisor, StimAction
from app.audio.controller import AudioController
from app.eeg.features import FeatureSnapshot
from app.mask.client import MaskClient
from app.persistence.schema import SessionProtocol, SessionStatus
from app.persistence.timeseries import SessionWriter, session_folder

logger = logging.getLogger("runtime.session")


class SessionState(str, Enum):
    idle = "idle"
    baseline = "baseline"
    running = "running"
    completed = "completed"
    aborted = "aborted"


@dataclass
class Session:
    subject_code: str
    studies_root: Path
    protocol: SessionProtocol = SessionProtocol.rl_agent
    protocol_params: Dict[str, Any] = field(default_factory=dict)
    eeg_source: str = "sim"
    mask_target: str = "virtual"
    audio_enabled: Optional[bool] = None
    """True/False to force; None -> daemon auto-derives from eeg_source."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=datetime.utcnow)
    state: SessionState = SessionState.idle
    folder: Optional[Path] = None
    writer: Optional[SessionWriter] = None
    supervisor: SafetySupervisor = field(default_factory=SafetySupervisor)
    audio: Optional[AudioController] = None
    mask: Optional[MaskClient] = None
    audio_seed: Dict[str, Any] = field(default_factory=dict)
    last_features: Optional[FeatureSnapshot] = None
    last_safe_action: Optional[StimAction] = None
    last_reward: float = 0.0
    last_plv: float = 0.0
    last_restfulness: float = 0.0
    tick_count: int = 0
    # Tick-budget probe: control_loop measures perf_counter for the compute
    # half of each tick (everything except asyncio.sleep) and exposes the
    # running max + last + overrun count so the UI / tests can detect when
    # the 4 Hz tick falls behind. tick_last_compute_s is reset every tick;
    # tick_max_compute_s is monotonic over the session.
    tick_last_compute_s: float = 0.0
    tick_max_compute_s: float = 0.0
    tick_overruns: int = 0
    # Last output of the deterministic audio shaper (warmth + master vol),
    # mirrored to telemetry so the operator can see the perceptual shaping
    # respond to the subject's drowsiness in real time. Populated by the
    # control loop; None for sessions without audio shaping (e.g. demo).
    last_audio_shaper: Optional[Dict[str, float]] = None

    def open(self) -> None:
        self.folder = session_folder(self.studies_root, self.subject_code,
                                     self.session_id)
        self.writer = SessionWriter(self.folder)
        logger.info("Session %s opened at %s", self.session_id, self.folder)

    def close(self, status: SessionStatus = SessionStatus.completed) -> None:
        if self.writer is not None:
            self.writer.flush_all()
        if self.audio is not None:
            try:
                self.audio.fadeout(1.0)
            except Exception:
                pass
        if self.mask is not None:
            try:
                self.mask.update_params(enabled=False, brightness=0.0)
            except Exception:
                pass
        self.state = (SessionState.aborted
                      if status == SessionStatus.aborted
                      else SessionState.completed)
        logger.info("Session %s closed (%s, %d ticks)",
                    self.session_id, status, self.tick_count)

    def panic(self, reason: str = "user_panic") -> None:
        self.supervisor.panic(reason)
        if self.mask is not None:
            try:
                self.mask.update_params(enabled=False, brightness=0.0)
            except Exception:
                pass
        if self.audio is not None:
            try:
                self.audio.fadeout(0.5)
            except Exception:
                pass
        self.state = SessionState.aborted

    def telemetry_snapshot(self) -> Dict[str, Any]:
        """Compact dict pushed to live monitor over WebSocket every tick.

        Includes the full set of creation params (eeg_source, mask_target,
        protocol_params, …) so the Live page can render a "session details"
        block without a separate REST round-trip to Mongo. These fields are
        stable per session - but it's cheap to ship them on every frame
        (~150 bytes JSON), and the alternative is teaching the frontend a
        second fetch that races the WS subscribe.
        """
        return {
            "session_id": self.session_id,
            "subject_code": self.subject_code,
            "state": self.state.value,
            "audio_enabled": self.audio_enabled,
            "protocol": self.protocol.value,
            "protocol_params": self.protocol_params,
            "eeg_source": self.eeg_source,
            "mask_target": self.mask_target,
            "started_at": self.started_at.isoformat() + "Z",
            "folder_path": str(self.folder) if self.folder else None,
            "tick": self.tick_count,
            "elapsed_s": (datetime.utcnow() - self.started_at).total_seconds(),
            "features": (None if self.last_features is None
                         else {
                             "alpha_pwr": self.last_features.alpha_pwr,
                             "theta_pwr": self.last_features.theta_pwr,
                             "beta_pwr": self.last_features.beta_pwr,
                             "hypnagogic_theta_pwr": self.last_features.hypnagogic_theta_pwr,
                             "sigma_pwr": self.last_features.sigma_pwr,
                             "theta_alpha_ratio": self.last_features.theta_alpha_ratio,
                             "emg_rms": self.last_features.emg_rms,
                             "eog_energy": self.last_features.eog_energy,
                             "railed_pct": self.last_features.railed_pct,
                         }),
            "plv": self.last_plv,
            "restfulness": self.last_restfulness,
            "stim": (None if self.last_safe_action is None
                     else {
                         "enabled": self.last_safe_action.enabled,
                         "frequency_hz": self.last_safe_action.frequency_hz,
                         "fill": self.last_safe_action.fill,
                         "brightness": self.last_safe_action.brightness,
                     }),
            "reward": self.last_reward,
            "audio_shaper": self.last_audio_shaper,
            "tick_budget": {
                "last_compute_ms": self.tick_last_compute_s * 1000.0,
                "max_compute_ms":  self.tick_max_compute_s * 1000.0,
                "overruns":        self.tick_overruns,
                "budget_ms":       250.0,   # CONTROL_TICK_S = 0.25s
            },
            "safety_events": [
                {"t_ms": e.t_ms, "kind": e.kind, "field": e.field,
                 "before": e.before, "after": e.after, "reason": e.reason}
                for e in self.supervisor.events[-10:]
            ],
        }
