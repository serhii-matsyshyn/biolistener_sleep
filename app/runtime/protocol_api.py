"""
HTTP + WebSocket surface for the hardware-session protocols.

Mounted on the daemon so the browser can do the parts a terminal is bad at:
check electrode quality before lying down, start a session without retyping a
command at 2 a.m., watch the block timeline, and read the gate verdict
afterwards.

One session at a time, process-wide. The runner owns the mask and the EEG
device, so a second one would fight the first for both; ``/start`` refuses while
another is live rather than letting two loops interleave commands.

The runner is a blocking loop, so it runs on a worker thread. Its observer
callbacks only ever mutate a small snapshot under a lock - no awaiting, no I/O -
because they execute inside the control loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.protocol import cues as cue_mod
from app.protocol.runner import RunnerConfig, SessionRunner
from app.protocol.spec import (
    BlockKind,
    DEFAULT_CHECKPOINT,
    build_full,
    build_validation,
    compress,
    order_history,
    settling_report,
    suggest_order,
)

logger = logging.getLogger("runtime.protocol_api")

router = APIRouter(prefix="/api/protocol", tags=["protocol"])

HISTORY_TICKS = 1200        # ~5 min at 4 Hz, enough for the live plots
SESSION_ROOT = Path("data/sessions")


class StartRequest(BaseModel):
    protocol: str = Field("validation", pattern="^(validation|full)$")
    order: str = Field("auto", pattern="^(auto|abba|baab|ramp_first|agent_first)$")
    seed: int = 0
    montage: str = "occipital_alpha"
    eeg: str = Field("sim", pattern="^(sim|biolistener|playback)$")
    eeg_ip: str = "0.0.0.0"
    eeg_port: int = 12345
    mask: str = "virtual"
    # The device's address, when it should not be waited for by ping.
    mask_target_ip: Optional[str] = None
    # Microvolts per training unit, from tools/calibrate_amplitude.py on the
    # Session A recording. None is correct only for the simulated path.
    amplitude_scale: Optional[float] = Field(None, gt=0)
    checkpoint: Optional[str] = None
    lang: str = "en"
    audio_cues: bool = True
    tick_hz: float = Field(4.0, gt=0, le=32)
    # Rehearsal only.
    compress_factor: float = Field(1.0, gt=0, le=1)


class KssRequest(BaseModel):
    value: int = Field(..., ge=1, le=9)
    block: Optional[str] = None


@dataclass
class _State:
    """Everything the UI needs, updated from the control loop under a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    runner: Optional[SessionRunner] = None
    thread: Optional[threading.Thread] = None
    running: bool = False
    status: str = "idle"
    request: Optional[dict] = None
    session_dir: Optional[str] = None
    started_at: Optional[float] = None
    error: Optional[str] = None
    abort_reason: Optional[str] = None
    protocol_name: Optional[str] = None
    blocks: List[dict] = field(default_factory=list)
    current: Optional[dict] = None
    last_tick: Optional[dict] = None
    history: Deque[dict] = field(default_factory=lambda: deque(maxlen=HISTORY_TICKS))
    events: Deque[dict] = field(default_factory=lambda: deque(maxlen=200))
    kss: List[dict] = field(default_factory=list)
    kss_pending: Optional[str] = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "protocol": self.protocol_name,
                "session_dir": self.session_dir,
                "elapsed_s": (None if self.started_at is None
                              else round(time.time() - self.started_at, 1)),
                "request": self.request,
                "blocks": list(self.blocks),
                "current": self.current,
                "tick": self.last_tick,
                "kss": list(self.kss),
                "kss_pending": self.kss_pending,
                "events": list(self.events)[-20:],
                "error": self.error,
                "abort_reason": self.abort_reason,
            }


_state = _State()


def _build_protocol(req: StartRequest):
    if req.protocol == "validation":
        proto = build_validation(seed=req.seed, montage=req.montage)
    else:
        order = (suggest_order(SESSION_ROOT) if req.order == "auto" else req.order)
        proto = build_full(order=order, seed=req.seed, montage=req.montage)
    if req.compress_factor < 1.0:
        proto = compress(proto, req.compress_factor)
    return proto


def _preflight(proto, req: StartRequest) -> dict:
    """The same checks the CLI prints, as data."""
    problems, warnings = [], list(settling_report(proto))
    if warnings and req.compress_factor >= 1.0:
        problems.extend(warnings)

    if (any(b.kind == BlockKind.DESCENT_AGENT for b in proto.blocks)
            and req.eeg != "sim" and not req.amplitude_scale):
        problems.append("amplitude_scale is unset for a hardware session with "
                        "an agent block - run tools/calibrate_amplitude.py on "
                        "the Session A recording first")

    if any(b.kind == BlockKind.DESCENT_AGENT for b in proto.blocks):
        if not req.checkpoint:
            problems.append("protocol has an agent block but no checkpoint was given")
        elif not Path(req.checkpoint).exists():
            problems.append(f"checkpoint not found: {req.checkpoint}")

    missing = ()
    if req.audio_cues:
        missing = cue_mod.CuePlayer(lang=req.lang).missing_cues()
        if missing:
            problems.append(f"{len(missing)} cue file(s) missing: "
                            f"{', '.join(missing[:4])}")
    with _state.lock:
        if _state.running:
            problems.append("another protocol session is already running")
    return {
        "ok": not problems,
        "problems": problems,
        "suggested_order": suggest_order(SESSION_ROOT),
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "order_history": order_history(SESSION_ROOT),
        "warnings": warnings,
        "missing_cues": list(missing),
        "protocol": proto.name,
        "total_duration_s": proto.total_duration_s,
        "blocks": [
            {"name": b.name, "kind": b.kind.value, "duration_s": b.duration_s,
             "eyes": b.eyes.value, "kss_at_end": b.kss_at_end, "note": b.note,
             "t_start_s": s, "t_end_s": e}
            for s, e, b in proto.boundaries()
        ],
    }


@router.get("/plan")
def get_plan(protocol: str = "validation", order: str = "auto",
             seed: int = 0, montage: str = "occipital_alpha",
             compress_factor: float = 1.0, lang: str = "en",
             checkpoint: Optional[str] = None, eeg: str = "sim",
             audio_cues: bool = True,
             amplitude_scale: Optional[float] = None) -> dict:
    """Block plan plus pre-flight, without starting anything."""
    try:
        req = StartRequest(protocol=protocol, order=order, seed=seed,
                           montage=montage, compress_factor=compress_factor,
                           lang=lang, checkpoint=checkpoint, eeg=eeg,
                           audio_cues=audio_cues,
                           amplitude_scale=amplitude_scale)
        proto = _build_protocol(req)
    except Exception as err:
        raise HTTPException(400, str(err)) from err
    return _preflight(proto, req)


@router.get("/status")
def get_status() -> dict:
    return _state.snapshot()


@router.get("/history")
def get_history() -> dict:
    with _state.lock:
        return {"ticks": list(_state.history)}


def _on_tick(row: dict) -> None:
    with _state.lock:
        _state.last_tick = row
        _state.history.append(row)
        _state.current = {
            "block": row.get("block"), "kind": row.get("block_kind"),
            "elapsed_s": row.get("block_elapsed_s"),
            "duration_s": row.get("block_duration_s"),
            "phase": row.get("phase"), "probe_hz": row.get("probe_hz"),
        }


def _on_event(kind: str, payload: dict) -> None:
    with _state.lock:
        _state.events.append({"t": round(time.time(), 3), "kind": kind, **payload})
        if kind == "kss_prompt":
            _state.kss_pending = payload.get("block")
        elif kind == "abort":
            _state.abort_reason = payload.get("reason")


def _run(runner: SessionRunner) -> None:
    try:
        out = runner.run()
        with _state.lock:
            _state.session_dir = str(out)
            _state.status = "aborted" if _state.abort_reason else "completed"
    except Exception as err:                        # noqa: BLE001
        logger.exception("protocol session failed")
        with _state.lock:
            _state.status = "error"
            _state.error = repr(err)
            if runner.session_dir:
                _state.session_dir = str(runner.session_dir)
    finally:
        with _state.lock:
            _state.running = False
            _state.current = None


@router.post("/start")
def start(req: StartRequest) -> dict:
    with _state.lock:
        if _state.running:
            raise HTTPException(409, "a protocol session is already running")

    try:
        proto = _build_protocol(req)
    except Exception as err:
        raise HTTPException(400, str(err)) from err

    pre = _preflight(proto, req)
    if not pre["ok"]:
        raise HTTPException(400, {"message": "pre-flight failed",
                                  "problems": pre["problems"]})

    cfg = RunnerConfig(
        protocol=proto, lang=req.lang, tick_hz=req.tick_hz,
        eeg_kind=req.eeg, eeg_ip=req.eeg_ip, eeg_port=req.eeg_port,
        # Bind local, target the device: `req.mask` names the mask, it is
        # not an address of this machine.
        mask_ip="127.0.0.1" if req.mask == "virtual" else "0.0.0.0",
        mask_target_ip=("127.0.0.1" if req.mask == "virtual"
                        else req.mask_target_ip),
        virtual_mask=req.mask == "virtual",
        amplitude_scale=req.amplitude_scale,
        checkpoint=Path(req.checkpoint) if req.checkpoint else None,
        audio_cues=req.audio_cues, session_root=SESSION_ROOT,
        watch_stdin=False,
    )
    # The demo controller holds 0.0.0.0:20000 for its own mask. Two clients on
    # one port both bind under SO_REUSEADDR and the kernel splits the device's
    # pings between them, so the session sees a mask that never answers while
    # the demo page works fine. Release it first, as the daemon's own sessions
    # already do.
    if req.mask != "virtual":
        try:
            from app.runtime.demo import shutdown_demo
            shutdown_demo()
        except Exception:
            logger.warning("could not release the demo mask", exc_info=True)

    runner = SessionRunner(cfg, on_tick=_on_tick, on_event=_on_event)

    with _state.lock:
        _state.runner = runner
        _state.running = True
        _state.status = "running"
        _state.request = req.model_dump()
        _state.protocol_name = proto.name
        _state.blocks = pre["blocks"]
        _state.session_dir = None
        _state.started_at = time.time()
        _state.error = None
        _state.abort_reason = None
        _state.current = None
        _state.last_tick = None
        _state.history.clear()
        _state.events.clear()
        _state.kss = []
        _state.kss_pending = None
        t = threading.Thread(target=_run, args=(runner,), daemon=True,
                             name="protocol-session")
        _state.thread = t
    t.start()
    return {"started": True, "protocol": proto.name,
            "total_duration_s": proto.total_duration_s}


@router.post("/stop")
def stop(reason: str = "stopped_from_ui") -> dict:
    with _state.lock:
        runner, running = _state.runner, _state.running
    if not running or runner is None:
        raise HTTPException(409, "no protocol session is running")
    runner.request_stop(reason)
    return {"stopping": True, "reason": reason}


@router.post("/kss")
def record_kss(req: KssRequest) -> dict:
    """
    Log a sleepiness rating.

    Spoken aloud during a solo session and transcribed afterwards; this endpoint
    is for when someone is at the screen, so the rating is timestamped by the
    same clock as the ticks and needs no transcription.
    """
    with _state.lock:
        if not _state.running:
            raise HTTPException(409, "no protocol session is running")
        entry = {
            "t_s": (round(time.time() - _state.started_at, 2)
                    if _state.started_at else None),
            "block": req.block or _state.kss_pending
            or (_state.current or {}).get("block"),
            "value": req.value,
        }
        _state.kss.append(entry)
        _state.kss_pending = None
        if _state.runner is not None:
            _state.runner.submit_rating(req.value)
        session_dir = _state.session_dir
        rows = list(_state.kss)

    # Written eagerly so a crash later does not lose the ratings.
    target = Path(session_dir) if session_dir else _find_open_session()
    if target is not None:
        try:
            lines = ["t_s,block,value"] + [
                f"{r['t_s']},{r['block']},{r['value']}" for r in rows]
            (target / "kss.csv").write_text("\n".join(lines) + "\n", encoding="utf8")
        except Exception:
            logger.warning("could not write kss.csv", exc_info=True)
    return {"recorded": entry}


def _find_open_session() -> Optional[Path]:
    """The runner creates its directory after /start returns; find the newest."""
    with _state.lock:
        name = _state.protocol_name
    if not name or not SESSION_ROOT.exists():
        return None
    dirs = sorted(p for p in SESSION_ROOT.glob(f"*_{name}") if p.is_dir())
    return dirs[-1] if dirs else None


@router.get("/sessions")
def list_sessions(limit: int = 25) -> dict:
    import json
    out = []
    if SESSION_ROOT.exists():
        for d in sorted((p for p in SESSION_ROOT.iterdir() if p.is_dir()),
                        reverse=True)[:limit]:
            man = d / "manifest.json"
            if not man.exists():
                continue
            try:
                m = json.loads(man.read_text(encoding="utf8"))
            except Exception:
                continue
            out.append({"dir": str(d), "name": d.name,
                        "protocol": m.get("protocol"), "status": m.get("status"),
                        "duration_s": m.get("duration_s"),
                        "n_ticks": m.get("n_ticks"),
                        "started_utc": m.get("started_utc")})
    return {"sessions": out}


@router.get("/sessions/{name}/gates")
def session_gates(name: str) -> dict:
    """Gate verdict for a recorded validation session."""
    from app.protocol import analysis as an
    from app.protocol import quality as q

    d = SESSION_ROOT / name
    if not d.is_dir():
        raise HTTPException(404, f"no session {name!r}")
    try:
        sess = an.load(d)
        gates = an.validation_gates(sess)
    except Exception as err:
        raise HTTPException(400, f"could not evaluate: {err}") from err
    return q.summarise(gates)


@router.websocket("/live")
async def live(ws: WebSocket) -> None:
    """Push the snapshot at ~4 Hz for as long as the socket is open."""
    await ws.accept()
    try:
        while True:
            await ws.send_json(_state.snapshot())
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.debug("protocol live socket closed", exc_info=True)
        return
