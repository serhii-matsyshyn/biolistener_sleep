"""
Integration tests for the four daemon additions in §12 (M8.3):

- /api/demo/* endpoints (Demo Open Loop direct control surface)
- WS /api/live/{session_id} pushes an initial telemetry snapshot on connect
- /api/sessions accepts create_session payload WITHOUT the legacy `mode` field
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("MONGO_URL", "mongodb://127.0.0.1:1/")
os.environ.setdefault("BIOLISTENER_SLEEP_ENABLE_AUDIO", "0")


@pytest.fixture
def fresh_daemon(tmp_path):
    """Patch mongo + reset daemon registries between tests."""
    with patch("app.runtime.daemon.get_db") as mock_get_db, \
         patch("app.runtime.daemon.init_indexes", AsyncMock(return_value=None)):
        mock_db = MagicMock()
        mock_db.sessions.insert_one = AsyncMock()
        mock_db.sessions.update_one = AsyncMock()
        mock_db.safety_events.insert_one = AsyncMock()
        mock_get_db.return_value = mock_db
        from app.runtime import daemon
        from app.runtime import demo
        daemon.STUDIES_ROOT = tmp_path / "studies"
        daemon.STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
        daemon.SESSIONS.clear()
        daemon.TICK_TASKS.clear()
        daemon.WEBSOCKETS.clear()
        # Reset Demo singleton
        demo.shutdown_demo()
        yield daemon
        demo.shutdown_demo()


@pytest.mark.asyncio
async def test_create_session_without_mode(fresh_daemon):
    """Legacy `mode` field is no longer required; absence must work."""
    resp = await fresh_daemon.create_session({"subject_code": "NO_MODE"})
    sid = resp["id"]
    snap = await fresh_daemon.session_status(sid)
    # The telemetry snapshot no longer carries `mode`
    assert "mode" not in snap


@pytest.mark.asyncio
async def test_create_session_ignores_legacy_mode_field(fresh_daemon):
    """A client that still sends `mode` must not blow up."""
    resp = await fresh_daemon.create_session({
        "subject_code": "LEGACY",
        "mode": "real",   # ignored
        "audio_enabled": True,
    })
    sid = resp["id"]
    snap = await fresh_daemon.session_status(sid)
    assert snap["audio_enabled"] is True


@pytest.mark.asyncio
async def test_create_session_audio_enabled_explicit(fresh_daemon):
    resp = await fresh_daemon.create_session({
        "subject_code": "AUDIO_OFF",
        "audio_enabled": False,
    })
    sid = resp["id"]
    snap = await fresh_daemon.session_status(sid)
    assert snap["audio_enabled"] is False


@pytest.mark.asyncio
async def test_demo_state_initial(fresh_daemon):
    state = await fresh_daemon.demo_state()
    # Strobe is enabled by default (matching the source app's UI), but
    # nothing is sent to the mask until a session is loaded + played.
    assert state["mask"]["enabled"] is True
    assert state["track"]["sound_path"] is None
    assert state["track"]["is_playing"] is False
    assert state["audio"]["generative_enabled"] is False


@pytest.mark.asyncio
async def test_demo_update_params_brightness_color_bitmask(fresh_daemon):
    state = await fresh_daemon.demo_update_params({
        "brightness": 0.45,
        "color": [200, 100, 50, 180],
        "bitmask": 0b00001111,
    })
    assert state["mask"]["brightness"] == pytest.approx(0.45)
    assert state["mask"]["color"] == [200, 100, 50, 180]
    assert state["mask"]["bitmask"] == 0b00001111


@pytest.mark.asyncio
async def test_demo_update_params_strobe_toggle(fresh_daemon):
    state = await fresh_daemon.demo_update_params({"enabled": False})
    assert state["mask"]["enabled"] is False


@pytest.mark.asyncio
async def test_demo_select_session_with_missing_files(fresh_daemon):
    """Selecting a session with no real files must not crash - just
    leave sound_path None and is_playing False."""
    state = await fresh_daemon.demo_select_session({
        "sound_path": "/no/such/file.mp3",
        "strobe_path": None,
    })
    assert state["track"]["sound_path"] is None
    assert state["track"]["is_playing"] is False


@pytest.mark.asyncio
async def test_demo_stop_disables_all(fresh_daemon):
    await fresh_daemon.demo_update_params({"enabled": True, "brightness": 0.3})
    await fresh_daemon.demo_update_audio({"master_vol_pct": 70.0})
    state = await fresh_daemon.demo_stop()
    assert state["mask"]["enabled"] is False
    assert state["mask"]["brightness"] == 0.0
    assert state["audio"]["master_vol_pct"] == 0.0


@pytest.mark.asyncio
async def test_ws_initial_snapshot_for_known_session(fresh_daemon, tmp_path):
    """Connecting to an existing session's WS must immediately push state."""
    from fastapi.testclient import TestClient

    resp = await fresh_daemon.create_session({"subject_code": "WSTEST"})
    sid = resp["id"]

    with TestClient(fresh_daemon.app) as client:
        with client.websocket_connect(f"/api/live/{sid}") as ws:
            # First frame must arrive without waiting on the control loop
            first = ws.receive_json()
            assert first["session_id"] == sid
            # state is "idle" (we never called start_session)
            assert first["state"] in {"idle", "running", "aborted", "completed"}


@pytest.mark.asyncio
async def test_ws_initial_snapshot_for_unknown_session(fresh_daemon):
    """Connecting to a session id not in memory must emit 'unknown' frame."""
    from fastapi.testclient import TestClient

    with TestClient(fresh_daemon.app) as client:
        with client.websocket_connect("/api/live/nonexistent") as ws:
            first = ws.receive_json()
            assert first["state"] == "unknown"
            assert first["reason"] == "not_in_daemon_memory"
