"""
Pydantic schemas for every document persisted to MongoDB.

Schema-evolution is via
Pydantic-v2 ConfigDict(extra='allow') - Mongo will accept extra fields
written by older daemon versions without losing them on read-back.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class _Doc(BaseModel):
    """Base - allows extra fields for forward compatibility."""
    model_config = ConfigDict(extra="allow")


class SessionProtocol(str, Enum):
    """
    Experimental protocol - the policy dimension of a session.

    The EEG/mask source dimension is captured separately by the
    ``eeg_source`` and ``mask_target`` fields (e.g. ``"sim"`` vs
    ``"biolistener:192.168.1.42:12345"``). There is no longer a
    redundant ``mode`` enum on top of those.
    """
    baseline_no_stim = "baseline_no_stim"   # no strobe, no entrainment audio
    open_loop_track = "open_loop_track"     # pre-recorded scripted strobe + audio
    rl_agent = "rl_agent"                   # closed-loop SAC (or any RL policy)


class SessionStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    aborted = "aborted"
    archived = "archived"


class Subject(_Doc):
    code: str
    dob_year: Optional[int] = None
    contraindication_checklist: dict = Field(default_factory=dict)
    baseline_alpha_hz: Optional[float] = None
    baseline_session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = ""


class Session(_Doc):
    subject_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    protocol: SessionProtocol = SessionProtocol.rl_agent
    protocol_params: dict = Field(default_factory=dict)
    eeg_source: str        # e.g. "biolistener:192.168.1.42:12345" | "sim"
    mask_target: str       # e.g. "esp32:192.168.1.43:20000" | "virtual"
    audio_enabled: Optional[bool] = None
    """True/False to force; None -> daemon auto-derives from eeg_source
    (any real hardware -> audio on, sim/playback -> off)."""
    policy_checkpoint_id: Optional[str] = None
    folder_path: str
    audio_seed: dict = Field(default_factory=dict)
    brain_sim_params: dict = Field(default_factory=dict)
    status: SessionStatus = SessionStatus.idle


class AgentDecision(_Doc):
    session_id: str
    t_ms: int
    action: List[float]       # raw normalized action from agent
    safe_action: dict         # post-safety physical action
    policy_value: Optional[float] = None
    exploration_noise: Optional[List[float]] = None


class StimEvent(_Doc):
    session_id: str
    t_ms: int
    mask_freq_hz: float
    duty: float
    brightness: float
    color: Tuple[int, int, int, int]
    audio_strobe_depth: float
    audio_binaural_vol: float
    audio_master_vol: float
    audio_preset: str


class SafetyEventDoc(_Doc):
    session_id: str
    t_ms: int
    kind: Literal[
        "clamp", "panic", "pse_guard", "ramp_limit",
        "session_timeout", "electrode_disconnect", "baseline_revert",
    ]
    field: str
    before: float
    after: float
    reason: str


class FeatureSnapshotDoc(_Doc):
    session_id: str
    t_ms: int
    alpha_pwr: float
    theta_pwr: float
    beta_pwr: float
    hypnagogic_theta_pwr: float
    sigma_pwr: float
    theta_alpha_ratio: float
    emg_rms: float
    eog_energy: float
    railed_pct: float
    plv_strobe: float
    restfulness: Optional[float] = None


class PolicyCheckpoint(_Doc):
    algo: str                 # "SAC" | "DQN" | "FixedFreq" | "LinearRamp"
    version: str
    trained_on_subject_id: Optional[str] = None
    episodes: int
    mean_reward: Optional[float] = None
    artifact_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = ""


class Study(_Doc):
    name: str
    description: str = ""
    session_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


CONTRAINDICATION_QUESTIONS: List[str] = [
    "photosensitive_epilepsy",
    "generalized_seizures",
    "stroke",
    "recent_traumatic_brain_injury",
    "severe_migraines",
    "psychotic_disorders",
    "dissociative_episodes",
    "alcohol_in_last_24h",
    "psychoactive_substances",
    "seizure_threshold_lowering_medications",
]


def has_contraindications(answers: dict) -> List[str]:
    """Return list of YES answers that disqualify session participation."""
    return [q for q in CONTRAINDICATION_QUESTIONS if answers.get(q) is True]
