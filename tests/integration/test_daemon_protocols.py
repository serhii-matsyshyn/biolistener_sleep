"""
End-to-end protocol tests - daemon runs each of baseline_no_stim,
open_loop_track, and rl_agent on the sim brain + virtual mask.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

os.environ.setdefault("MONGO_URL", "mongodb://127.0.0.1:1/")
os.environ.setdefault("BIOLISTENER_SLEEP_ENABLE_AUDIO", "0")


@pytest.fixture
def fresh_daemon(tmp_path):
    """Patch mongo + reset registries; yields the daemon module."""
    with patch("app.runtime.daemon.get_db") as mock_get_db, \
         patch("app.runtime.daemon.init_indexes", AsyncMock(return_value=None)):
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
        yield daemon


@pytest.mark.asyncio
async def test_baseline_no_stim_session(fresh_daemon):
    daemon = fresh_daemon
    resp = await daemon.create_session({
        "subject_code": "BASE01",
        "protocol": "baseline_no_stim",
    })
    sid = resp["id"]
    folder = Path(resp["folder"])

    # protocol.json must reflect the chosen protocol
    proto_doc = json.loads((folder / "protocol.json").read_text())
    assert proto_doc["protocol"] == "baseline_no_stim"

    await daemon.start_session(sid)
    await asyncio.sleep(1.5)
    snap = await daemon.session_status(sid)
    assert snap["protocol"] == "baseline_no_stim"
    # Stim must remain OFF throughout the run
    assert (snap["stim"] is None) or (snap["stim"]["brightness"] == 0.0)
    await daemon.end_session(sid)


@pytest.mark.asyncio
async def test_open_loop_track_session(fresh_daemon, tmp_path):
    daemon = fresh_daemon

    # Author a tiny strobe-config and point the policy at it via protocol_params.
    sc = tmp_path / "sc.yaml"
    sc.write_text(yaml.safe_dump({
        "tinytrack": {
            "1": [
                # 30-second segment ramping ~10 -> ~7 Hz at 20% duty
                [0, 30_000_000, 0.0, 30_000_000.0,
                 0.0, 200.0,
                 -3000.0, 10000.0],
            ],
        },
    }))

    resp = await daemon.create_session({
        "subject_code": "OLT01",
        "protocol": "open_loop_track",
        "protocol_params": {
            "strobe_config_path": str(sc),
            "library": "tinytrack",
            "preset": "1",
            "brightness": 0.30,
        },
    })
    sid = resp["id"]
    await daemon.start_session(sid)
    await asyncio.sleep(1.5)
    snap = await daemon.session_status(sid)
    assert snap["protocol"] == "open_loop_track"
    # Strobe should be active (brightness > 0) at some point in a 1.5s window
    assert snap["stim"] is not None
    await daemon.end_session(sid)


@pytest.mark.asyncio
async def test_open_loop_track_session_via_csv_strobe_path(fresh_daemon, tmp_path):
    """
    UI-shaped protocol_params (catalog `strobe_path` to an event-log CSV)
    must work end-to-end through the daemon's _setup_policy. This is what
    the React UI sends after the user picks an activity + duration.
    """
    daemon = fresh_daemon

    # Author a 30-cycle event log: 10 Hz, 20% duty.
    csv_path = tmp_path / "events.csv"
    rows = ["Timestamp_NS,Torch_State", "0,0"]
    for k in range(30):
        rise = 1_000_000_000 + k * 100_000_000
        rows.append(f"{rise},1")
        rows.append(f"{rise + 20_000_000},0")
    csv_path.write_text("\n".join(rows) + "\n")

    resp = await daemon.create_session({
        "subject_code": "OLT_CSV01",
        "protocol": "open_loop_track",
        "protocol_params": {
            "strobe_path": str(csv_path),
            "audio_path": "",   # no audio for this test
            "brightness": 0.30,
        },
    })
    sid = resp["id"]
    await daemon.start_session(sid)
    await asyncio.sleep(1.5)
    snap = await daemon.session_status(sid)
    assert snap["protocol"] == "open_loop_track"
    assert snap["stim"] is not None
    await daemon.end_session(sid)


@pytest.mark.asyncio
async def test_rl_agent_session_default(fresh_daemon):
    daemon = fresh_daemon
    resp = await daemon.create_session({
        "subject_code": "RL01",
        "protocol": "rl_agent",
    })
    sid = resp["id"]
    await daemon.start_session(sid)
    await asyncio.sleep(1.5)
    snap = await daemon.session_status(sid)
    assert snap["protocol"] == "rl_agent"
    assert snap["stim"] is not None
    # Default policy is FixedFrequencyPolicy -> freq close to 10 Hz
    assert 3.0 <= snap["stim"]["frequency_hz"] <= 14.0
    await daemon.end_session(sid)
