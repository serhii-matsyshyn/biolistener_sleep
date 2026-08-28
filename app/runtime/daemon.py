"""
FastAPI daemon - long-running closed-loop process.

All REST + WS endpoints are namespaced under ``/api/`` so they cannot
collide with the React SPA's client-side routes.
Anything outside ``/api/`` is served by
the static mount at ``/`` and falls back to ``index.html`` for unknown
paths so React Router can handle the URL.

REST + WS endpoints:
- POST /api/subjects                       create subject (safety checklist)
- GET  /api/subjects                       list subjects
- POST /api/sessions                       create a new session
- GET  /api/sessions                       list sessions
- POST /api/sessions/{id}/start            begin streaming + control tick
- POST /api/sessions/{id}/panic            latch panic, stop strobe/audio
- POST /api/sessions/{id}/end              close session, flush, archive
- GET  /api/sessions/{id}/status           session telemetry snapshot
- GET  /api/open-loop-track/activities     catalog browser
- GET  /api/open-loop-track/activities/{name}/durations
- WS   /api/live/{session_id}              per-tick telemetry stream
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # noqa: E402  (deferred import for grouping)

from time import perf_counter

from app.agent.env import EnvConfig
from app.agent.obs_state import ObsState
from app.agent.policies import (
    BaselineNoStimPolicy, FixedFrequencyPolicy, LinearRampDownPolicy,
)
from app.agent.reward import compute_reward
from app.agent.safety import (
    SafetySupervisor, StimAction, decode_action_for_deployment,
)
from app.audio.controller import AudioController
from app.audio.engine import AudioServer, BiofeedbackEngine
from app.audio.shaper import AudioShaper
from app.brain_sim.brainflow_board import (
    InProcessBrainSim, NMMBioListenerDevice,
)
from app.brain_sim.domain_random import RandomizationRanges
from app.brain_sim.nmm import MultiChannelBrain
from app.brain_sim.photic_input import StrobeParams, strobe_envelope
from app.eeg.features import extract_features, feature_vector
from app.eeg.ingest import EEGIngestor, IngestConfig
from app.eeg.ml_metrics import RestfulnessMetric
from app.eeg.montage import OCCIPITAL_ALPHA
from app.eeg.plv import plv_strobe_to_eeg
from app.mask.client import MaskClient
from app.mask.virtual import VirtualMaskServer
from app.open_loop_track.catalog import OpenLoopTrackCatalog
from app.open_loop_track.csv_playback import CsvEventPlayback
from app.open_loop_track.playback import OpenLoopTrackPlayback
from app.open_loop_track.policy import OpenLoopTrackPolicy
from app.persistence.mongo import get_db, init_indexes
from app.persistence.timeseries import write_audio_seed
from app.persistence.schema import (
    AgentDecision, Session as SessionDoc, SessionProtocol,
    SessionStatus, Subject, has_contraindications,
)
from app.runtime.demo import get_demo, shutdown_demo
from app.runtime.session import Session, SessionState

logger = logging.getLogger("runtime.daemon")


# In-memory session registry - daemon process owns active sessions
SESSIONS: Dict[str, Session] = {}
TICK_TASKS: Dict[str, asyncio.Task] = {}
WEBSOCKETS: Dict[str, list[WebSocket]] = {}

CONTROL_TICK_S = 0.25         # 4 Hz - matches EnvConfig.tick_hz
STUDIES_ROOT = Path("studies")

# Set by app.cli._runner.serve_daemon when launched via `biolistener-sleep daemon`.
# When None, the daemon falls back to schema defaults + env vars.
RESOLVED_CONFIG: Optional["BioListenerSleepConfig"] = None  # type: ignore[name-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        db = get_db()
        await init_indexes(db)
        # Reap orphans - sessions stuck in 'running'/'idle' from a previous
        # daemon process whose control loops were killed before completing.
        # Without this, the Sessions table accumulates stuck rows that the
        # End button can clear individually but the user shouldn't have to.
        try:
            result = await db.sessions.update_many(
                {"status": {"$in": ["running", "idle"]}},
                {"$set": {"status": SessionStatus.aborted.value}},
            )
            if result.modified_count:
                logger.info("Reaped %d orphan session(s) on startup",
                            result.modified_count)
        except Exception:
            logger.exception("Orphan-reap on startup failed")
    except Exception:
        logger.exception("Mongo not reachable on startup - continuing without")
    yield
    for sid in list(TICK_TASKS):
        task = TICK_TASKS.pop(sid)
        task.cancel()
    for sid in list(SESSIONS):
        sess = SESSIONS.pop(sid)
        if sess.state == SessionState.running:
            sess.close(status=SessionStatus.aborted)
    try:
        shutdown_demo()
    except Exception:
        logger.exception("Demo shutdown failed")


app = FastAPI(title="BioListener Sleep - Daemon", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict:
    """Dependencies the UI needs to know about before a request hangs on them.

    Sessions, subjects and every recorded artefact live in Mongo; the hardware
    protocol runner does not touch it. Reported per dependency so the banner can
    say which page is affected rather than just that something is wrong.
    """
    from app.persistence.mongo import MONGO_DB, MONGO_URL, ping
    ok = await ping()
    return {
        "mongo": {"ok": ok, "url": MONGO_URL, "db": MONGO_DB},
        "protocol_runner": {"ok": True},
    }

# Hardware-session protocols (validation + experimental). Separate router so the
# session machinery above is untouched; it refuses to start while one is live.
from app.runtime.protocol_api import router as _protocol_router  # noqa: E402
app.include_router(_protocol_router)


@app.post("/api/subjects")
async def create_subject(payload: dict):
    code = payload.get("code")
    if not code:
        raise HTTPException(400, "code is required")
    checklist = payload.get("contraindication_checklist", {})
    blocks = has_contraindications(checklist)
    if blocks:
        raise HTTPException(
            400, f"Subject {code} has disqualifying contraindications: {blocks}",
        )
    subj = Subject(code=code, **{
        k: v for k, v in payload.items()
        if k in {"dob_year", "notes", "contraindication_checklist"}
    })
    db = get_db()
    res = await db.subjects.insert_one(subj.model_dump())
    return {"id": str(res.inserted_id), "code": code}


@app.get("/api/subjects")
async def list_subjects():
    db = get_db()
    cursor = db.subjects.find({}).sort([("created_at", 1)])
    out = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        out.append(doc)
    return out


@app.post("/api/sessions")
async def create_session(payload: dict):
    code = payload.get("subject_code")
    if not code:
        raise HTTPException(400, "subject_code is required")

    # Legacy clients may still send `mode` - we accept and silently ignore
    # (it's been removed from the schema; eeg_source + mask_target +
    # audio_enabled are the dimensions that matter now).

    protocol_str = payload.get("protocol", "rl_agent")
    try:
        protocol = SessionProtocol(protocol_str)
    except ValueError:
        raise HTTPException(400, f"invalid protocol: {protocol_str}")
    protocol_params = payload.get("protocol_params") or {}
    audio_enabled = payload.get("audio_enabled")   # True | False | None (auto)

    # Wire all 12 safety bounds from `cfg.safety.*` into the supervisor.
    # When RESOLVED_CONFIG is None (smoke tests without a daemon config)
    # we fall back to module-constant defaults baked into SafetySupervisor.
    supervisor = (
        SafetySupervisor.from_config(RESOLVED_CONFIG.safety)
        if RESOLVED_CONFIG is not None
        else SafetySupervisor()
    )
    sess = Session(
        subject_code=code,
        studies_root=STUDIES_ROOT,
        protocol=protocol,
        protocol_params=protocol_params,
        eeg_source=payload.get("eeg_source", "sim"),
        mask_target=payload.get("mask_target", "virtual"),
        audio_enabled=audio_enabled,
        supervisor=supervisor,
    )
    sess.open()
    SESSIONS[sess.session_id] = sess

    # Per-session protocol.json snapshot - alongside config.yaml (written
    # separately by biolistener-sleep run). Lets analysis notebooks classify
    # sessions without loading Mongo.
    if sess.folder is not None:
        (sess.folder / "protocol.json").write_text(
            json.dumps(
                {
                    "protocol": protocol.value,
                    "protocol_params": protocol_params,
                    "eeg_source": sess.eeg_source,
                    "mask_target": sess.mask_target,
                    "audio_enabled": audio_enabled,
                },
                indent=2,
            )
        )

    db = get_db()
    doc = SessionDoc(
        subject_id=code,
        protocol=protocol,
        protocol_params=protocol_params,
        eeg_source=sess.eeg_source,
        mask_target=sess.mask_target,
        audio_enabled=audio_enabled,
        folder_path=str(sess.folder),
    )
    await db.sessions.insert_one(
        {**doc.model_dump(mode="json"), "_id": sess.session_id}
    )
    return {"id": sess.session_id, "folder": str(sess.folder)}


@app.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(404, "session not found")
    if sess.state != SessionState.idle:
        raise HTTPException(400, f"session in state {sess.state}")

    sess.state = SessionState.running
    db = get_db()
    try:
        await db.sessions.update_one(
            {"_id": session_id},
            {"$set": {"status": SessionStatus.running.value}},
        )
    except Exception:
        logger.warning("start_session: failed to update mongo status",
                       exc_info=True)
    # If this session wants the real mask, free the demo controller's
    # 0.0.0.0:20000 socket so the session-side MaskClient can bind it.
    # pyo/VLC stay up; demo will re-init on next visit.
    if sess.mask_target.startswith("esp32"):
        try:
            shutdown_demo()
        except Exception:
            logger.warning("start_session: demo shutdown failed",
                           exc_info=True)
    task = asyncio.create_task(_control_loop(sess))
    TICK_TASKS[session_id] = task
    return {"status": "running"}


@app.post("/api/sessions/{session_id}/panic")
async def panic_session(session_id: str):
    """Mark the session aborted. Like end_session, this works on orphans
    too - sessions stuck in 'running' status after a daemon restart.
    Also silences the Demo controller (process-global pyo + VLC) so the
    button is a true emergency stop regardless of which UI surface kicked
    it."""
    db = get_db()
    sess = SESSIONS.get(session_id)
    if sess is not None:
        # Already terminal - no-op so a double-click doesn't double-notify
        # or re-stop the demo controller. Idempotent by design.
        if sess.state in (SessionState.aborted, SessionState.completed):
            return {"status": "aborted", "in_memory": True, "noop": True}
        sess.panic("user_panic_endpoint")
    else:
        doc = await db.sessions.find_one({"_id": session_id})
        if doc is None:
            raise HTTPException(404, "session not found")
        if doc.get("status") in ("aborted", "completed"):
            return {"status": "aborted", "in_memory": False, "noop": True}
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"status": SessionStatus.aborted.value}},
    )
    try:
        get_demo().stop_all()
    except Exception:
        logger.exception("PANIC: failed to stop demo audio")
    return {"status": "aborted", "in_memory": sess is not None}


@app.post("/api/panic")
async def panic_all():
    """Global emergency stop - silences every session + the demo controller.

    Exposed for the dedicated PANIC button so it works even when the user
    isn't focused on a specific session (e.g. they're on the demo page or
    the live page is showing an already-ended session).
    """
    aborted: list[str] = []
    for sid, sess in list(SESSIONS.items()):
        try:
            sess.panic("global_panic_endpoint")
            aborted.append(sid)
        except Exception:
            logger.exception("PANIC: session %s could not be aborted", sid)
    try:
        get_demo().stop_all()
    except Exception:
        logger.exception("PANIC: failed to stop demo audio")
    return {"status": "panicked", "sessions_aborted": aborted}


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """Mark the session completed. Works for in-memory sessions AND for
    orphans - i.e. sessions whose Mongo status is still 'running' from
    a previous daemon run but are no longer in the SESSIONS dict (the
    common case after a daemon restart). Without this, the Sessions
    table would have stuck 'running' rows that 'End' can never clear."""
    db = get_db()
    sess = SESSIONS.get(session_id)
    if sess is not None:
        task = TICK_TASKS.pop(session_id, None)
        if task is not None:
            task.cancel()
        sess.close(status=SessionStatus.completed)
    else:
        # Orphan - only mongo update. Verify the session exists in Mongo
        # so we don't silently 200-OK a typo'd id.
        doc = await db.sessions.find_one({"_id": session_id})
        if doc is None:
            raise HTTPException(404, "session not found")
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"status": SessionStatus.completed.value}},
    )
    return {"status": "completed", "in_memory": sess is not None}


@app.get("/api/sessions/{session_id}/status")
async def session_status(session_id: str):
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess.telemetry_snapshot()


@app.get("/api/sessions")
async def list_sessions(subject_code: Optional[str] = None, limit: int = 50):
    db = get_db()
    filt = {}
    if subject_code:
        filt["subject_id"] = subject_code
    out = []
    async for doc in db.sessions.find(filt).sort(
        [("started_at", -1)]
    ).limit(limit):
        doc["_id"] = str(doc["_id"])
        out.append(doc)
    return out


@app.websocket("/api/live/{session_id}")
async def live_socket(ws: WebSocket, session_id: str):
    await ws.accept()
    # Push current state immediately so the UI doesn't sit on the
    # "Waiting for telemetry…" banner indefinitely - handles 3 cases:
    #   1. Session is running           -> snapshot now, real ticks follow.
    #   2. Session ended but in memory  -> snapshot reflects terminal state.
    #   3. Session not in memory at all -> emit a single "unknown" frame so
    #      the React UI can render "Session ended" instead of hanging.
    sess = SESSIONS.get(session_id)
    try:
        if sess is not None:
            await ws.send_json(sess.telemetry_snapshot())
        else:
            # Not in memory - try Mongo so the Live page can still render
            # the "Session details" block (creation params survive daemon
            # restarts; live metrics don't). Falls back to a minimal frame
            # if Mongo isn't reachable or the id is bogus.
            frame: dict = {
                "session_id": session_id,
                "state": "unknown",
                "reason": "not_in_daemon_memory",
                "tick": 0,
                "elapsed_s": 0.0,
                "features": None,
                "plv": 0.0,
                "restfulness": 0.0,
                "stim": None,
                "reward": 0.0,
                "safety_events": [],
            }
            try:
                doc = await get_db().sessions.find_one({"_id": session_id})
            except Exception:
                doc = None
            if doc is not None:
                started = doc.get("started_at")
                if hasattr(started, "isoformat"):
                    started = started.isoformat() + "Z"
                frame.update({
                    "state": str(doc.get("status", "unknown")),
                    "subject_code": doc.get("subject_id"),
                    "protocol": doc.get("protocol"),
                    "protocol_params": doc.get("protocol_params", {}),
                    "eeg_source": doc.get("eeg_source"),
                    "mask_target": doc.get("mask_target"),
                    "audio_enabled": doc.get("audio_enabled"),
                    "started_at": started,
                    "folder_path": doc.get("folder_path"),
                })
            await ws.send_json(frame)
    except Exception:
        logger.exception("Initial WS snapshot failed for %s", session_id)
    WEBSOCKETS.setdefault(session_id, []).append(ws)
    try:
        while True:
            # Just keep alive - daemon broadcasts telemetry on each tick.
            await ws.receive_text()
    except WebSocketDisconnect:
        WEBSOCKETS.get(session_id, []).remove(ws)


async def _broadcast(session_id: str, payload: dict):
    sockets = WEBSOCKETS.get(session_id, [])
    if not sockets:
        return
    for ws in list(sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            try:
                sockets.remove(ws)
            except ValueError:
                pass


@dataclass
class _EegHandle:
    kind: str                              # "sim" | "real"
    sim: Optional[InProcessBrainSim] = None
    ingestor: Optional[EEGIngestor] = None
    nmm_device: Optional[NMMBioListenerDevice] = None
    # Optional sim raw-EEG recorder. Mirrors BrainFlow's add_streamer
    # tab-separated layout (rows=samples, cols=channels) so the post-
    # session notebook reads sim and real recordings with the same code.
    # Opened in _setup_eeg when the session has a folder; closed in the
    # control_loop's `finally`.
    sim_csv_fh: Optional[Any] = None


def _parse_eeg_source(spec: str) -> tuple[str, str, int]:
    """
    ``"sim"``                              -> ("sim", "127.0.0.1", 0)
    ``"biolistener"``                      -> ("biolistener", <daemon eeg.ip_address>, <daemon eeg.ip_port>)
                                              (ping-discovery on the YAML-configured interface; default 0.0.0.0:12345)
    ``"biolistener:192.0.2.10:12345"``   -> ("biolistener", "192.0.2.10", 12345)
    """
    if spec == "sim":
        return "sim", "127.0.0.1", 0
    if spec == "biolistener":
        # Bare token = use daemon-wide YAML defaults. Falling back to sim
        # here would be the worst failure mode available: NMM samples stream
        # while the operator believes the device is live.
        if RESOLVED_CONFIG is not None:
            return ("biolistener",
                    RESOLVED_CONFIG.eeg.ip_address,
                    int(RESOLVED_CONFIG.eeg.ip_port))
        return "biolistener", "0.0.0.0", 12345
    parts = spec.split(":")
    if len(parts) == 3 and parts[0] == "biolistener":
        return "biolistener", parts[1], int(parts[2])
    logger.warning(
        "Unrecognized eeg_source %r - falling back to sim. Accepted forms: "
        "'sim', 'biolistener', 'biolistener:<ip>:<port>'.", spec,
    )
    return "sim", "127.0.0.1", 0


def _parse_mask_target(spec: str) -> tuple[str, str, int]:
    """
    ``"virtual"``                       -> ("virtual", "127.0.0.1", 20000)
    ``"esp32"``                         -> ("esp32",   "",          20000)  (ping discovery - no IP needed)
    ``"esp32:192.0.2.11:20000"``      -> ("esp32",   "192.0.2.11", 20000)
    """
    if spec == "virtual":
        return "virtual", "127.0.0.1", 20000
    if spec == "esp32":
        # Real mask, ping discovery (controller binds 0.0.0.0:20000 and
        # waits for the device to ping). Same UX as source mask_strobe_controller.
        return "esp32", "", 20000
    parts = spec.split(":")
    if len(parts) == 3 and parts[0] == "esp32":
        return "esp32", parts[1], int(parts[2])
    return "virtual", "127.0.0.1", 20000


def _setup_eeg(sess: Session, cfg: EnvConfig
               ) -> tuple[_EegHandle, dict]:
    """Build the EEG source per ``sess.eeg_source``."""
    kind, ip, port = _parse_eeg_source(sess.eeg_source)
    sim_meta = {}
    if kind == "biolistener":
        # PGA gain comes from the daemon YAML (eeg.pga_gain) - leaving it
        # at the IngestConfig default of 128 would silently re-program the
        # ADC and change the signal scaling between training and deployment.
        pga = (RESOLVED_CONFIG.eeg.pga_gain
               if RESOLVED_CONFIG is not None else 8)
        active = (list(RESOLVED_CONFIG.eeg.active_channels)
                  if RESOLVED_CONFIG is not None else list(range(8)))
        # The montage selects which rows of the streamed window the feature
        # pipeline reads. If the operator wired (say) channels [0, 1] but
        # the montage points at channel 5, every PSD comes out of a disabled
        # ADC and the run silently produces zeros. Fail loudly instead.
        needed = set(cfg.montage.primary_eeg_channels) \
                  | set(cfg.montage.emg_channels) \
                  | set(cfg.montage.eog_channels)
        missing = needed - set(active)
        if missing:
            raise HTTPException(
                400,
                f"eeg.active_channels {sorted(active)} does not cover the "
                f"montage's required channels {sorted(needed)} "
                f"(missing: {sorted(missing)}). Plug them in or pick a "
                "compatible montage.",
            )
        ing = EEGIngestor(IngestConfig(
            kind="biolistener", ip_address=ip, ip_port=port or 12345,
            sampling_rate=cfg.sampling_rate,
            pga_gain=int(pga),
            active_channels=active,
            record_to=(sess.folder / "raw_eeg.csv") if sess.folder else None,
        ))
        ing.start()
        return _EegHandle(kind="real", ingestor=ing), {"source": sess.eeg_source}

    # Default: sim. Domain-randomized subject, in-process for speed.
    subject = RandomizationRanges().sample(seed=None)
    sim_meta = {
        "alpha_hz": subject.alpha_hz,
        "responder": subject.responder_type,
        "cortical_gain": subject.photic_config.cortical_gain,
    }
    brain = MultiChannelBrain(
        n_channels=8, base_params=subject.nmm_params,
        alpha_hz=subject.alpha_hz, sampling_rate=cfg.sampling_rate,
        seed=subject.seed,
    )
    sim = InProcessBrainSim(brain=brain, photic_config=subject.photic_config)
    # Mirror BrainFlow's `add_streamer("file://...")` for the real path
    # so the post-session notebook treats sim and real runs identically.
    # Layout: tab-separated, no header, rows=samples, one column per
    # channel (col 0 = timestamp µs for parity, cols 1..8 = EEG).
    sim_fh = None
    if sess.folder is not None:
        try:
            sim_fh = (sess.folder / "raw_eeg.csv").open("w", buffering=1 << 16)
        except Exception:
            logger.warning("raw_eeg.csv open failed for sim session",
                           exc_info=True)
            sim_fh = None
    return _EegHandle(kind="sim", sim=sim, sim_csv_fh=sim_fh), sim_meta


def _setup_mask(sess: Session) -> tuple[MaskClient, Optional[VirtualMaskServer]]:
    """Build the mask sink per ``sess.mask_target``."""
    kind, ip, port = _parse_mask_target(sess.mask_target)
    if kind == "esp32":
        # Real ESP32 on LAN - bind controller socket on 0.0.0.0:port. If the
        # UI sent just "esp32" (no IP), do ping discovery (the mask is the
        # UDP client and self-announces); otherwise pin target_ip so we can
        # send immediately even before the first ping.
        target = ip if ip else None
        client = MaskClient(bind_ip="0.0.0.0", port=port, target_ip=target)
        client.port = port
        return client, None

    # Virtual mask on local loopback - per-session port pair to allow
    # multi-session concurrency.
    base_port = 21000 + (len(SESSIONS) * 4)
    ctrl_port, dev_port = base_port, base_port + 1
    client = MaskClient(bind_ip="127.0.0.1", port=ctrl_port,
                        target_ip="127.0.0.1")
    client.port = dev_port
    vm = VirtualMaskServer(
        listen_ip="127.0.0.1", listen_port=dev_port,
        controller_addr=("127.0.0.1", ctrl_port),
        ping_interval_s=1.0,
    )
    vm.start()
    return client, vm


def _setup_audio(sess: Session) -> tuple[AudioController, Optional["object"], Optional["object"]]:
    """
    Build the audio stack for the chosen protocol.

    Returns (controller, pyo_server_or_None, file_player_or_None).

    Protocol-conditioned behavior:
      - rl_agent          -> generative BiofeedbackEngine (current default).
      - baseline_no_stim  -> controller logs events but engine boots only if
                            audio is enabled (brown-noise mask).
      - open_loop_track   -> boots a pyo Server but instead of the generative
                            graph it spins up OpenLoopTrackAudioPlayer for the
                            matched pre-recorded file.

    Audio resolution order (later wins):
      1. ``BIOLISTENER_SLEEP_ENABLE_AUDIO`` env var (any "0/false" disables).
      2. ``sess.audio_enabled`` - explicit True/False from the create_session
         payload (UI's Audio selector / CLI's session.audio_enabled config).
      3. Auto-derive from ``sess.eeg_source`` - anything resolving to the
         real device (bare ``"biolistener"`` or ``"biolistener:<ip>:<port>"``)
         -> audio on; ``sim`` / ``playback`` -> audio off.
    """
    env_audio = os.environ.get("BIOLISTENER_SLEEP_ENABLE_AUDIO")
    if env_audio is not None:
        want_audio = env_audio not in {"0", "false", "False", ""}
    elif sess.audio_enabled is not None:
        want_audio = bool(sess.audio_enabled)
    else:
        want_audio = (sess.eeg_source == "biolistener"
                       or sess.eeg_source.startswith("biolistener:"))

    if not want_audio:
        return AudioController(engine=None), None, None

    # open_loop_track: pygame.mixer plays the pre-recorded file directly.
    # No pyo Server needed - saves boot time + avoids device-collision corner
    # cases between pyo and SDL_mixer.
    if sess.protocol == SessionProtocol.open_loop_track:
        try:
            from app.open_loop_track.audio_player import OpenLoopTrackAudioPlayer
            audio_path = sess.protocol_params.get("audio_path")
            player: Optional[OpenLoopTrackAudioPlayer] = None
            if audio_path:
                player = OpenLoopTrackAudioPlayer(Path(audio_path),
                                                    master_vol=0.5)
                player.start()
            else:
                logger.warning("open_loop_track session missing audio_path; "
                               "audio running silent")
            return AudioController(engine=None), None, player
        except Exception:
            logger.warning("open_loop_track audio boot failed - silent fallback",
                            exc_info=True)
            return AudioController(engine=None), None, None

    # rl_agent / baseline_no_stim: generative engine
    try:
        server = AudioServer()
        server.start()
        if RESOLVED_CONFIG is not None:
            ac = RESOLVED_CONFIG.audio
            engine = BiofeedbackEngine(
                preset_name=ac.preset,
                output_gain=float(ac.output_gain),
                max_val_vol=float(ac.safety.max_val_vol),
                max_strobe_depth_pct=float(ac.safety.max_strobe_depth_pct),
                max_binaural_vol_pct=float(ac.safety.max_binaural_vol_pct),
            )
        else:
            engine = BiofeedbackEngine(preset_name="sunlight_pads")
        return AudioController(engine=engine), server, None
    except Exception:
        logger.warning(
            "Audio engine boot failed (no audio device or pyo error) - "
            "falling back to silent controller. Set BIOLISTENER_SLEEP_ENABLE_AUDIO=0 "
            "to suppress this warning.",
            exc_info=True,
        )
        return AudioController(engine=None), None, None


def _setup_policy(sess: Session) -> "object":
    """
    Build the control policy for the chosen protocol.

    - baseline_no_stim -> BaselineNoStimPolicy (never emits stim)
    - open_loop_track  -> OpenLoopTrackPolicy reading pre-recorded segments
    - rl_agent         -> FixedFrequencyPolicy (placeholder; SAC checkpoint
                         loading is wired via sess.protocol_params or
                         RESOLVED_CONFIG.policy.checkpoint_path)
    """

    if sess.protocol == SessionProtocol.baseline_no_stim:
        return BaselineNoStimPolicy()

    if sess.protocol == SessionProtocol.open_loop_track:
        params = sess.protocol_params
        # Path A: UI-selected catalog activity -> ``strobe_path`` is an
        # event-log CSV from the external folder.
        if params.get("strobe_path"):
            playback: object = CsvEventPlayback.from_csv(Path(params["strobe_path"]))
        # Path B: hand-authored parametric segment YAML (used by tests +
        # power users) via ``strobe_config_path`` + library + preset.
        elif (params.get("strobe_config_path")
              and params.get("library")
              and params.get("preset")):
            playback = OpenLoopTrackPlayback.from_config_yaml(
                Path(params["strobe_config_path"]),
                params["library"], params["preset"],
            )
        else:
            raise HTTPException(
                400,
                "open_loop_track protocol requires protocol_params with "
                "either {strobe_path: <event-log CSV>} OR "
                "{strobe_config_path, library, preset}; "
                f"got {sess.protocol_params}",
            )
        return OpenLoopTrackPolicy(
            playback=playback,
            brightness=float(params.get("brightness", 0.30)),
        )

    # Default: rl_agent -> either SAC (if checkpoint provided) or a baseline.
    if RESOLVED_CONFIG is not None:
        pol = RESOLVED_CONFIG.policy
        if pol.name == "sac" and pol.checkpoint_path is not None:
            ckpt = Path(pol.checkpoint_path)
            if not ckpt.exists():
                raise HTTPException(
                    400,
                    f"SAC checkpoint not found: {ckpt}. "
                    "Provide an absolute path or one relative to the daemon CWD.",
                )
            # Imported lazily so the daemon doesn't pay sb3/torch startup
            # cost when running a baseline-only session.
            from stable_baselines3 import SAC
            logger.info("Loading SAC checkpoint: %s", ckpt)
            model = SAC.load(str(ckpt), device="cpu")
            # The model's `predict()` already matches the FixedFrequencyPolicy
            # contract (returns ``(action, state)``) so no wrapper is needed.
            return model
        if pol.name == "linear_ramp":
            return LinearRampDownPolicy(
                start_hz=pol.ramp_start_hz, end_hz=pol.ramp_end_hz,
                duration_s=RESOLVED_CONFIG.session.duration_s,
            )
        # ``policy.name=='sac'`` without a checkpoint falls through to the
        # baseline so a misconfigured session doesn't silently train from
        # scratch in the daemon.
        return FixedFrequencyPolicy(
            frequency_hz=pol.fixed_freq_hz,
            brightness=pol.fixed_brightness, fill=pol.fixed_fill,
        )
    return FixedFrequencyPolicy(frequency_hz=10.0, brightness=0.30, fill=0.20)


async def _read_window(handle: _EegHandle, buf: np.ndarray,
                        buf_pos: int, buf_filled: bool, cfg: EnvConfig,
                        n_samples_per_tick: int
                        ) -> tuple[np.ndarray, int, bool]:
    """
    Advance one control tick worth of samples into the circular buffer.

    For sim: step the NMM ``n_samples_per_tick`` times.
    For real: grab the freshest window from EEGIngestor (BrainFlow has
    been buffering samples on its own background thread).
    """
    if handle.kind == "sim":
        assert handle.sim is not None
        new_samples = np.empty((8, n_samples_per_tick), dtype=np.float64)
        for i in range(n_samples_per_tick):
            sample = handle.sim.step()
            buf[:, buf_pos] = sample
            new_samples[:, i] = sample
            buf_pos = (buf_pos + 1) % buf.shape[1]
            if buf_pos == 0:
                buf_filled = True
        # Stream the tick's new samples to raw_eeg.csv when recording.
        # Format mirrors BrainFlow's add_streamer: tab-separated, no header,
        # col 0 = monotonic µs timestamp, cols 1..8 = EEG channels.
        if handle.sim_csv_fh is not None:
            try:
                # InProcessBrainSim's `_t` is the sim's monotonic clock in s.
                base_us = int(getattr(handle.sim, "_t", 0.0) * 1_000_000)
                step_us = int(1_000_000 / cfg.sampling_rate)
                lines = []
                for i in range(n_samples_per_tick):
                    ts = base_us + i * step_us
                    cols = [str(ts)] + [f"{v:.6g}" for v in new_samples[:, i]]
                    lines.append("\t".join(cols))
                handle.sim_csv_fh.write("\n".join(lines) + "\n")
            except Exception:
                # Don't let recording faults stop the control loop.
                pass
        if buf_filled:
            window = np.concatenate(
                [buf[:, buf_pos:], buf[:, :buf_pos]], axis=1,
            )
        else:
            window = buf[:, :max(buf_pos, 1)]
        return window, buf_pos, buf_filled

    assert handle.ingestor is not None
    window = handle.ingestor.latest_window(window_s=cfg.window_s)
    if window.size == 0:
        # No samples yet - emit a single zero frame so feature extraction
        # is bypassed cleanly.
        window = np.zeros((8, 1), dtype=np.float64)
    return window, buf_pos, True


async def _control_loop(sess: Session) -> None:
    """
    Closed-loop control tick (4 Hz). Branches on session config:

    - EEG: sim -> InProcessBrainSim; real -> BrainFlow BIOLISTENER via EEGIngestor
    - Mask: virtual -> local UDP VirtualMaskServer; esp32 -> real device on LAN
    - Audio: enabled for real sessions (boots pyo Server + BiofeedbackEngine);
      silent (event-log only) for sim/training. Override via ``BIOLISTENER_SLEEP_ENABLE_AUDIO``.
    """
    cfg = EnvConfig(sampling_rate=500)

    # EEG setup can block for up to ~70 s while EEGIngestor retries
    # prepare_session() against an absent BIOLISTENER (see app/eeg/ingest.py
    # connect_attempts). Run it in a thread so the FastAPI event loop
    # stays responsive - the operator can still hit /panic during the wait.
    eeg_handle, sim_meta = await asyncio.to_thread(_setup_eeg, sess, cfg)
    sess.audio, audio_server, audio_file_player = _setup_audio(sess)
    sess.mask, vm = _setup_mask(sess)

    if sim_meta:
        sess.brain_sim_params = sim_meta  # persisted on session document

    # Dump every RNG seed we know about into audio_seed.json. Used by the
    # replay path and surfaced in the post-session notebook so the operator
    # can prove the run is reproducible. Best-effort - never block the loop.
    if sess.folder is not None:
        try:
            seed_doc = {
                "session_id": sess.session_id,
                "subject_code": sess.subject_code,
                "eeg_source": sess.eeg_source,
                "sim_meta": sim_meta,
                "server_start_ts": datetime.utcnow().isoformat() + "Z",
            }
            write_audio_seed(sess.folder, seed_doc)
        except Exception:
            logger.warning("write_audio_seed failed", exc_info=True)

    logger.info(
        "Session %s control loop: protocol=%s eeg=%s mask=%s audio=%s",
        sess.session_id,
        sess.protocol.value,
        eeg_handle.kind,
        "esp32" if vm is None else "virtual",
        "file_player" if audio_file_player is not None
        else ("engine" if audio_server is not None else "silent"),
    )

    try:
        policy = _setup_policy(sess)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Policy setup failed; falling back to FixedFrequencyPolicy")
        policy = FixedFrequencyPolicy(frequency_hz=10.0,
                                        brightness=0.30, fill=0.20)

    # Auxiliary RESTFULNESS metric - populated per-tick and surfaced on the
    # live monitor. Lazy-prepared so a failure (e.g. missing MLModel binary)
    # degrades gracefully to a constant 0.
    restfulness_metric = None
    try:
        restfulness_metric = RestfulnessMetric()
        restfulness_metric.prepare()
    except Exception:
        logger.warning("RESTFULNESS metric unavailable (continuing without)",
                       exc_info=True)
        restfulness_metric = None

    db = get_db()
    buf = np.zeros((8, int(cfg.window_s * cfg.sampling_rate)),
                   dtype=np.float64)
    buf_pos = 0
    buf_filled = False
    n_samples_per_tick = int(cfg.sampling_rate / cfg.tick_hz)
    prev_features = None
    prev_safe_freq_hz: Optional[float] = None
    # Shared smoothing state - matches the training env's observation
    # pipeline exactly so SAC checkpoints get the obs vector they were
    # trained on (EMA-smoothed PLV + Δ(smoothed θ/α) appended).
    obs_state = ObsState(tick_hz=cfg.tick_hz)
    # Deterministic perceptual audio shaping (warmth + master volume).
    # Replaces the agent's action[4] (master vol) with a drowsiness+time
    # taper because the sim has no audio input model and the agent can't
    # be trained to make perceptual choices. See app/audio/shaper.py.
    if RESOLVED_CONFIG is not None:
        sh = RESOLVED_CONFIG.audio.shaper
        audio_shaper = AudioShaper(
            theta_alpha_baseline=sh.theta_alpha_baseline,
            theta_alpha_full=sh.theta_alpha_full,
            warmth_at_baseline_pct=sh.warmth_at_baseline_pct,
            warmth_at_full_pct=sh.warmth_at_full_pct,
            initial_master_vol_pct=sh.initial_master_vol_pct,
            drowsy_vol_taper=sh.drowsy_vol_taper,
            end_fade_starts_at_frac=sh.end_fade_starts_at_frac,
            end_fade_min_factor=sh.end_fade_min_factor,
        )
        shaper_session_duration_s = float(RESOLVED_CONFIG.session.duration_s)
    else:
        audio_shaper = AudioShaper()
        shaper_session_duration_s = 600.0   # fallback only used when RESOLVED_CONFIG is missing
    # Per-session mask presentation params (LED colour, white channel,
    # which LEDs fire). These are NOT in the policy's action space -
    # the operator sets them per study in `mask.strobe.*`. Parsed once
    # before the loop so we're not re-parsing the hex every tick.
    if RESOLVED_CONFIG is not None:
        sc = RESOLVED_CONFIG.mask.strobe
        rgb_hex = sc.color_hex.lstrip("#")
        strobe_rgb = [int(rgb_hex[i:i+2], 16) for i in (0, 2, 4)]
        strobe_color = strobe_rgb + [int(sc.white_channel)]
        strobe_bitmask = int(sc.led_bitmask)
    else:
        strobe_color = [0, 0, 0, 200]
        strobe_bitmask = 0xFF
    # Rolling tick-budget probe: at 4 Hz we have 250 ms per tick. We log
    # the moving max so the UI can surface latency regressions before they
    # turn into missed ticks.
    tick_budget_s = CONTROL_TICK_S
    tick_max_compute_s = 0.0
    tick_overruns = 0

    try:
        while sess.state == SessionState.running:
            tick_t0 = perf_counter()
            window, buf_pos, buf_filled = await _read_window(
                eeg_handle, buf, buf_pos, buf_filled, cfg, n_samples_per_tick,
            )

            features = extract_features(
                window, cfg.sampling_rate, cfg.montage,
                t_ms=int(sess.tick_count * 1000 * CONTROL_TICK_S),
            )
            sess.last_features = features

            # RESTFULNESS auxiliary score (needs ≥2 s window per BrainFlow's
            # internal heuristics; returns 0 if window too short).
            if restfulness_metric is not None and buf_filled:
                try:
                    sess.last_restfulness = restfulness_metric.score(
                        window,
                        cfg.montage.primary_eeg_channels or [0],
                        cfg.sampling_rate,
                    )
                except Exception:
                    sess.last_restfulness = 0.0

            # PLV must be computed before the obs vector so the temporal
            # context (smoothed PLV) and the obs's `plv` slot agree with
            # what training uses. We compute it inline here; the post-action
            # PLV recompute below still happens for telemetry on the *next*
            # tick's last_strobe (kept for parity with prior behaviour).
            plv_now = 0.0
            if buf_filled and (sess.last_safe_action is not None
                                and sess.last_safe_action.enabled):
                last_sa = sess.last_safe_action
                env_wave = strobe_envelope(
                    duration_s=window.shape[1] / cfg.sampling_rate,
                    dt=1.0 / cfg.sampling_rate,
                    strobe=StrobeParams(
                        enabled=last_sa.enabled,
                        frequency_hz=last_sa.frequency_hz,
                        fill=last_sa.fill, brightness=last_sa.brightness,
                    ),
                )
                primary_ch = cfg.montage.primary_eeg_channels[0]
                if primary_ch < window.shape[0]:
                    plv_now = plv_strobe_to_eeg(
                        window[primary_ch], env_wave, cfg.sampling_rate,
                        last_sa.frequency_hz,
                    )

            # Update shared smoothing state, then overwrite the raw θ/α
            # with the smoothed value so reward + persistence + photic
            # resonance shift all see the same signal as during training.
            obs_state.update(
                raw_theta_alpha=float(features.theta_alpha_ratio),
                raw_plv=float(plv_now),
            )
            features.theta_alpha_ratio = obs_state.smoothed_theta_alpha
            if eeg_handle.kind == "sim" and eeg_handle.sim is not None:
                eeg_handle.sim.photic_config.current_drowsiness = float(
                    obs_state.smoothed_theta_alpha
                )

            obs_features = feature_vector(features)
            last_strobe = sess.last_safe_action or StimAction()
            smoothed_plv, delta_tar = obs_state.temporal_context()
            # Order MUST match SleepOnsetEnv._observe(): 9 features + plv
            # + (optional restfulness) + 3 last-action + 2 temporal.
            # The cohort SAC checkpoints were trained with
            # include_restfulness=False, so the deployment obs drops
            # restfulness from the vector (telemetry still surfaces
            # sess.last_restfulness - that's a display field, not RL input).
            obs = np.concatenate([
                obs_features,
                np.array([plv_now,
                          last_strobe.frequency_hz, last_strobe.fill,
                          last_strobe.brightness,
                          smoothed_plv, delta_tar], dtype=np.float32),
            ])

            action, _ = policy.predict(obs, deterministic=True)
            # Use deployment-mode decoder: agent's [-1, 1] maps directly
            # into the supervisor's safe [min, max] envelope per axis. The
            # post-hoc clamp() still runs (PSE guard, ramp limits, etc.)
            # but freq/fill/brightness are pre-bounded, so we no longer
            # spam `*_bounds` clamp events for every tick.
            stim_action = decode_action_for_deployment(action, sess.supervisor)
            # Bypass the railed-electrode check on sim EEG: there's no
            # electrode to disconnect, and BrainFlow's get_railed_percentage
            # operates on the BoardShim ADC scale (µV after PGA gain) whereas
            # our NMM emits mV-scale y1-y2 values, which look "railed" by
            # ~1000× to the BrainFlow detector and spuriously kill the strobe.
            railed_for_safety = (
                0.0 if eeg_handle.kind == "sim" else features.railed_pct
            )
            safe = sess.supervisor.clamp(
                stim_action, railed_pct=railed_for_safety,
            )
            sess.last_safe_action = safe

            # Loop termination conditions
            # The SafetySupervisor only clamps the next command (it can't
            # end the loop on its own); the OpenLoopTrackPolicy only emits
            # OFF after its CSV ends. Without explicit checks the loop
            # runs forever on either trigger, spamming the safety-event
            # log. Convert both into real session termination.
            end_reason: Optional[str] = None
            if sess.supervisor.session_elapsed_s() > sess.supervisor.session_timeout_s:
                end_reason = "session_timeout"
            elif (sess.protocol == SessionProtocol.open_loop_track
                  and getattr(policy, "track_ended", False)):
                end_reason = "open_loop_track ended"
            if end_reason is not None:
                logger.info("Session %s ending: %s",
                            sess.session_id, end_reason)
                sess.close(status=SessionStatus.completed)
                try:
                    await db.sessions.update_one(
                        {"_id": sess.session_id},
                        {"$set": {"status": SessionStatus.completed.value}},
                    )
                except Exception:
                    logger.warning("session-end: mongo update failed",
                                   exc_info=True)
                TICK_TASKS.pop(sess.session_id, None)
                break

            # Apply to whichever brain we're driving (sim only - real human
            # brain doesn't take UDP commands)
            if eeg_handle.kind == "sim" and eeg_handle.sim is not None:
                eeg_handle.sim.update_strobe(StrobeParams(
                    enabled=safe.enabled, frequency_hz=safe.frequency_hz,
                    fill=safe.fill, brightness=safe.brightness,
                ))
            # Apply to the mask (virtual or real ESP32). RGB+W colour and
            # LED bitmask are operator-set in `mask.strobe.*` (parsed
            # before the loop); frequency / fill / brightness are the
            # policy-controlled knobs the agent + safety supervisor decide.
            sess.mask.update_params(
                enabled=safe.enabled, frequency=safe.frequency_hz,
                fill=safe.fill, brightness=safe.brightness,
                color=strobe_color,
                mask=strobe_bitmask,
            )
            # Apply to the audio controller (silent in headless sessions).
            # sync_freq + strobe_depth follow the agent. master_vol + warmth
            # are deterministically shaped from smoothed θ/α + time-into-session
            # (the agent's commanded master_vol is ignored — the sim has no
            # perceptual model so the agent can't learn this knob).
            sess.audio.set_sync_freq(safe.frequency_hz)
            sess.audio.set_strobe_depth(safe.audio_strobe_depth_pct)
            shaped_warmth_pct = audio_shaper.warmth_pct(
                obs_state.smoothed_theta_alpha,
            )
            shaped_master_vol_pct = audio_shaper.master_vol_pct(
                obs_state.smoothed_theta_alpha,
                time_into_session_s=sess.supervisor.session_elapsed_s(),
                session_duration_s=shaper_session_duration_s,
            )
            sess.audio.set_warmth(shaped_warmth_pct)
            sess.audio.set_master_vol(shaped_master_vol_pct)
            sess.last_audio_shaper = {
                "drowsiness": audio_shaper.drowsiness(obs_state.smoothed_theta_alpha),
                "warmth_pct": shaped_warmth_pct,
                "master_vol_pct": shaped_master_vol_pct,
            }

            # PLV telemetry: surface the value already computed above
            # (which was based on the previous tick's actual strobe -> this
            # tick's EEG response). The strobe we just emitted will be
            # PLV'd on the next tick.
            sess.last_plv = float(plv_now)

            sess.last_reward = compute_reward(
                prev=prev_features,
                curr=features,
                plv_strobe=sess.last_plv,
                strobe_freq_hz=safe.frequency_hz,
                prev_strobe_freq_hz=prev_safe_freq_hz,
            )
            prev_features = features
            prev_safe_freq_hz = safe.frequency_hz

            # Persist
            sess.tick_count += 1
            t_ms = int(sess.tick_count * 1000 * CONTROL_TICK_S)
            sess.writer.add_features({
                "session_id": sess.session_id, "t_ms": t_ms,
                "alpha_pwr": features.alpha_pwr,
                "theta_pwr": features.theta_pwr,
                "beta_pwr": features.beta_pwr,
                "hypnagogic_theta_pwr": features.hypnagogic_theta_pwr,
                "sigma_pwr": features.sigma_pwr,
                "theta_alpha_ratio": features.theta_alpha_ratio,
                "emg_rms": features.emg_rms,
                "eog_energy": features.eog_energy,
                "railed_pct": features.railed_pct,
                "plv_strobe": sess.last_plv,
                "restfulness": sess.last_restfulness,
            })
            sess.writer.add_stim({
                "session_id": sess.session_id, "t_ms": t_ms,
                "mask_freq_hz": safe.frequency_hz,
                "duty": safe.fill,
                "brightness": safe.brightness,
                "color_white": safe.color_white,
            })
            # Raw policy output + decoded request + post-clamp value.
            # `action` is the SAC's [-1, 1] vector before any decode/clamp.
            # `stim_action` is the deployment-decoder's pre-clamp request.
            # `safe` is what physically went to the mask. The diff between
            # stim_action.* and safe.* is exactly the safety-supervisor's
            # contribution per tick - that's the data the safety-overlay
            # figure in the post-session notebook draws from.
            sess.writer.add_action({
                "session_id": sess.session_id, "t_ms": t_ms,
                "action_freq_raw": float(action[0]),
                "action_fill_raw": float(action[1]),
                "action_brightness_raw": float(action[2]),
                "request_freq_hz": float(stim_action.frequency_hz),
                "request_fill": float(stim_action.fill),
                "request_brightness": float(stim_action.brightness),
                "safe_freq_hz": float(safe.frequency_hz),
                "safe_fill": float(safe.fill),
                "safe_brightness": float(safe.brightness),
                "reward": float(sess.last_reward),
            })

            await _broadcast(sess.session_id, sess.telemetry_snapshot())

            new_evts = sess.supervisor.events[-3:]
            for e in new_evts:
                evt_row = {
                    "session_id": sess.session_id, "t_ms": e.t_ms,
                    "kind": e.kind, "field": e.field,
                    "before": e.before, "after": e.after,
                    "reason": e.reason,
                }
                # Filesystem first - it works even without Mongo, and the
                # post-session notebook reads it as the canonical source
                # for the safety-event timeline figure.
                try:
                    sess.writer.append_safety_event(evt_row)
                except Exception:
                    logger.warning("safety_events.jsonl append failed",
                                   exc_info=True)
                try:
                    await db.safety_events.insert_one(evt_row)
                except Exception:
                    pass

            # Tick-budget probe: log the slowest compute spent in this tick
            # (excludes the sleep), and short-sleep by exactly the remainder
            # of the 250 ms window. Without subtracting compute time the
            # real tick rate would dip below 4 Hz under any load.
            tick_compute_s = perf_counter() - tick_t0
            if tick_compute_s > tick_max_compute_s:
                tick_max_compute_s = tick_compute_s
                sess.tick_max_compute_s = tick_max_compute_s
            sess.tick_last_compute_s = tick_compute_s
            sleep_left = tick_budget_s - tick_compute_s
            if sleep_left <= 0:
                tick_overruns += 1
                sess.tick_overruns = tick_overruns
                # Logging at WARNING so deployments see it without DEBUG;
                # the first overrun is the interesting one.
                if tick_overruns <= 5 or tick_overruns % 50 == 0:
                    logger.warning(
                        "Session %s tick overrun #%d: compute=%.0fms > budget=%.0fms",
                        sess.session_id, tick_overruns,
                        tick_compute_s * 1000.0, tick_budget_s * 1000.0,
                    )
                # Still yield to the event loop so other coroutines (WS,
                # mongo) can make progress even if we're saturated.
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(sleep_left)
    except asyncio.CancelledError:
        logger.info("Session %s loop cancelled", sess.session_id)
        raise
    except Exception:
        logger.exception("Session %s loop crashed", sess.session_id)
        sess.panic("loop_exception")
    finally:
        # Send one final telemetry frame so the UI sees the terminal state
        # immediately (otherwise the last broadcast carried state=running
        # and the next-tick broadcast that would've reported `aborted` never
        # fires, since the loop has exited).
        try:
            await _broadcast(sess.session_id, sess.telemetry_snapshot())
        except Exception:
            pass
        # Shut down hardware-side resources in reverse order
        if vm is not None:
            vm.stop()
            vm.join(timeout=2)
        if eeg_handle.ingestor is not None:
            eeg_handle.ingestor.stop()
        if eeg_handle.nmm_device is not None:
            eeg_handle.nmm_device.stop()
            eeg_handle.nmm_device.join(timeout=2)
        if eeg_handle.sim_csv_fh is not None:
            try:
                eeg_handle.sim_csv_fh.close()
            except Exception:
                pass
        if audio_file_player is not None:
            try:
                audio_file_player.stop()
            except Exception:
                pass
        if audio_server is not None:
            try:
                audio_server.stop()
            except Exception:
                pass
        if restfulness_metric is not None:
            try:
                restfulness_metric.release()
            except Exception:
                pass
        sess.writer.flush_all()


def _resolve_open_loop_track_root() -> Optional[Path]:
    """Return the configured external folder, or None if disabled."""
    if RESOLVED_CONFIG is not None:
        root = RESOLVED_CONFIG.open_loop_track.data_root
        if root is not None:
            return Path(root)
    return None


@app.get("/api/open-loop-track/activities")
async def list_open_loop_track_activities():
    """List activities + variants from the external catalog YAML."""
    root = _resolve_open_loop_track_root()
    if root is None:
        raise HTTPException(404, "open_loop_track.data_root unset - protocol disabled")
    cfg = RESOLVED_CONFIG.open_loop_track if RESOLVED_CONFIG is not None else None
    catalog_path = root / (cfg.catalog_filename if cfg else "sessions.yaml")
    try:
        cat = OpenLoopTrackCatalog.from_yaml(catalog_path)
    except Exception as err:
        raise HTTPException(500, f"catalog parse failed: {err}")
    out = []
    for act in cat.activities():
        out.append({
            "name": act.name,
            "display_name": act.display_name,
            "block": act.block,
            "variants": [
                {"key": v.variant_key, "duration_min": v.duration_min,
                 "has_strobe": v.strobe_filename is not None}
                for v in act.variants.values()
            ],
        })
    return out


@app.get("/api/open-loop-track/activities/{name}/durations")
async def list_open_loop_track_durations(name: str):
    root = _resolve_open_loop_track_root()
    if root is None:
        raise HTTPException(404, "open_loop_track.data_root unset")
    cfg = RESOLVED_CONFIG.open_loop_track if RESOLVED_CONFIG is not None else None
    catalog_path = root / (cfg.catalog_filename if cfg else "sessions.yaml")
    cat = OpenLoopTrackCatalog.from_yaml(catalog_path)
    act = cat.get(name)
    if act is None:
        raise HTTPException(404, f"activity {name!r} not found")
    return [
        {"variant_key": v.variant_key, "duration_min": v.duration_min,
         "audio_path": str(v.resolved_audio_path(root, cat.audio_subdir)),
         "strobe_path": (str(v.resolved_strobe_path(root, cat.strobe_subdir))
                          if v.strobe_filename else None)}
        for v in act.variants.values()
    ]


#
# These endpoints DO NOT create sessions, do not write to Mongo, do not write
# session folders. They surface direct control of the mask + audio engine
# for stage demos and ad-hoc impact experiments. The DemoController owns one
# MaskClient (virtual or real ESP32 per config) and one BiofeedbackEngine.
# SafetySupervisor is still in the path - same clamps as session protocols.


@app.get("/api/demo/state")
async def demo_state():
    return get_demo().get_state()


@app.post("/api/demo/set_mask_kind")
async def demo_set_mask_kind(payload: dict):
    """Runtime swap between virtual emulator and real ESP32 mask."""
    return get_demo().set_mask_kind(kind=str(payload.get("kind", "virtual")))


@app.post("/api/demo/select_session")
async def demo_select_session(payload: dict):
    """Load a track + matching strobe CSV (mirrors source app)."""
    return get_demo().select_session(
        sound_path=payload.get("sound_path"),
        strobe_path=payload.get("strobe_path"),
    )


@app.post("/api/demo/play")
async def demo_play():
    return get_demo().play()


@app.post("/api/demo/pause")
async def demo_pause():
    return get_demo().pause()


@app.post("/api/demo/seek")
async def demo_seek(payload: dict):
    return get_demo().seek(pos_s=float(payload.get("pos_s", 0.0)))


@app.post("/api/demo/update_params")
async def demo_update_params(payload: dict):
    """Update user-controlled strobe params (brightness/color/bitmask/enabled).
    Strobe frequency + fill come from the loaded CSV, not from the UI."""
    return get_demo().update_params(
        brightness=payload.get("brightness"),
        color=payload.get("color"),
        enabled=payload.get("enabled"),
        bitmask=payload.get("bitmask"),
    )


@app.post("/api/demo/update_audio")
async def demo_update_audio(payload: dict):
    """Generative-audio toggle + knobs (BiofeedbackEngine)."""
    return get_demo().update_audio(
        generative_enabled=payload.get("generative_enabled"),
        preset=payload.get("preset"),
        master_vol_pct=payload.get("master_vol_pct"),
        binaural_vol_pct=payload.get("binaural_vol_pct"),
        warmth_cutoff_pct=payload.get("warmth_cutoff_pct"),
        brown_noise_pct=payload.get("brown_noise_pct"),
        reverb_space_pct=payload.get("reverb_space_pct"),
        test_mode_on=payload.get("test_mode_on"),
        reset=payload.get("reset"),
    )


@app.post("/api/demo/stop")
async def demo_stop():
    return get_demo().stop_all()


#
# Routing model:
#   /api/...          - all REST + WS endpoints (declared above)
#   /assets/...       - Vite-emitted JS/CSS bundles, served by static mount
#   anything else     - served by the SPA catch-all below as index.html
#                       (React Router takes over client-side)


_static_dir = Path(__file__).parent / "static"
_assets_dir = _static_dir / "assets"
_index_html = _static_dir / "index.html"

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def _spa_catch_all(full_path: str):
    """
    Serve the React SPA's index.html for any path not matched by an /api/
    route or the /assets/ mount. The React Router takes over on the client.
    """
    # Don't intercept API miscalls - return a real 404
    if full_path.startswith("api/") or full_path.startswith("assets/"):
        raise HTTPException(404, "Not Found")
    if _index_html.exists():
        return FileResponse(_index_html)
    return JSONResponse({
        "message": (
            "UI not built. Run 'cd app/ui && npm install && npm run build' "
            "to populate app/runtime/static/, or 'npm run dev' on port 5173 "
            "with the Vite proxy."
        )
    })
