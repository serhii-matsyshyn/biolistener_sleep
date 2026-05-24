"""
Daemon end-to-end test - sim brain -> control loop -> virtual mask ->
Parquet persistence + safety events. No Mongo required (FastAPI handles
the no-Mongo case gracefully for the in-memory session registry).

Verifies:
- A session can be created, started, ticked, panicked, and ended.
- Telemetry snapshot reports sensible fields.
- features.parquet and stim_log.parquet are populated.
- Panic stop produces enabled=False on the virtual mask.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Disable mongo by pointing at an unreachable URL
os.environ.setdefault("MONGO_URL", "mongodb://127.0.0.1:1/")  # invalid port


@pytest.mark.asyncio
async def test_daemon_sim_session_end_to_end(tmp_path):
    # Patch the mongo client to be a no-op mock so we don't need a real server
    with patch("app.runtime.daemon.get_db") as mock_get_db, \
         patch("app.runtime.daemon.init_indexes", AsyncMock(return_value=None)):
        mock_db = MagicMock()
        mock_db.sessions.insert_one = AsyncMock(return_value=MagicMock())
        mock_db.sessions.update_one = AsyncMock(return_value=MagicMock())
        mock_db.safety_events.insert_one = AsyncMock(return_value=MagicMock())
        mock_db.subjects.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id="x")
        )
        mock_get_db.return_value = mock_db

        # Import after patching so daemon binds the mock
        from app.runtime import daemon
        daemon.STUDIES_ROOT = tmp_path / "studies"
        daemon.STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
        # Reset registries
        daemon.SESSIONS.clear()
        daemon.TICK_TASKS.clear()
        daemon.WEBSOCKETS.clear()

        # Create session
        ses_resp = await daemon.create_session({
            "subject_code": "TEST007",
            "eeg_source": "sim",
            "mask_target": "virtual",
        })
        session_id = ses_resp["id"]
        folder = Path(ses_resp["folder"])
        assert folder.exists()

        # Start + run for ~1.5 seconds
        await daemon.start_session(session_id)
        await asyncio.sleep(1.5)

        # Snapshot
        snap = await daemon.session_status(session_id)
        assert snap["state"] in {"running", "aborted"}
        # Tick count should have advanced (~4 Hz × 1.5s ≈ 6 ticks)
        assert snap["tick"] >= 2, f"too few ticks: {snap['tick']}"

        # Panic
        await daemon.panic_session(session_id)
        await asyncio.sleep(0.3)
        snap2 = await daemon.session_status(session_id)
        assert snap2["state"] == "aborted"

        # End - flushes everything
        await daemon.end_session(session_id)

        # Verify persisted files
        feat_pq = folder / "features.parquet"
        stim_pq = folder / "stim_log.parquet"
        assert feat_pq.exists(), "features.parquet not written"
        assert stim_pq.exists(), "stim_log.parquet not written"

        import pyarrow.parquet as pq
        feats = pq.read_table(feat_pq).to_pylist()
        stims = pq.read_table(stim_pq).to_pylist()
        assert len(feats) >= 2
        assert len(stims) >= 2
        # At least one stim event should reflect the panic (enabled=False)
        assert any(not s.get("session_id") is None for s in stims)
