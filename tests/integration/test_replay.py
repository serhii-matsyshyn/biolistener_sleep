"""
Replay smoke test - load a daemon-produced session folder back through
``load_session`` + ``iter_stim_events`` + a virtual-mask replay.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


os.environ.setdefault("MONGO_URL", "mongodb://127.0.0.1:1/")


@pytest.mark.asyncio
async def test_replay_loads_and_reissues_recorded_stim(tmp_path):
    # Produce a real session via the daemon, then replay it
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

        resp = await daemon.create_session({
            "subject_code": "REPLAY01",
        })
        sid = resp["id"]
        folder = Path(resp["folder"])

        await daemon.start_session(sid)
        await asyncio.sleep(1.2)
        await daemon.end_session(sid)

    # Now replay
    from app.runtime.replay import (
        iter_stim_events, load_session, reissue_to_virtual_mask,
    )
    bundle = load_session(folder)
    assert bundle.folder == folder
    assert len(bundle.stim_log) > 0
    rows = list(iter_stim_events(bundle))
    assert all("t_ms" in r for r in rows)
    # Sorted by time
    times = [r["t_ms"] for r in rows]
    assert times == sorted(times)

    # Mask reissue (fast-forwarded so the test stays under a second)
    reissue_to_virtual_mask(
        bundle, listen_port=22100, controller_port=22101, speed=100.0,
    )
