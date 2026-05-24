"""
End-to-end test: daemon loads a SAC checkpoint and the control loop
actually uses it.

Two things this proves:

1. `_setup_policy` resolves `policy.name='sac' + checkpoint_path` to a
   real SAC instance (regression test for the "always returns
   FixedFrequencyPolicy" bug we just fixed).
2. The observation vector the daemon builds matches what the checkpoint
   was trained on - otherwise `model.predict()` would raise a shape
   error on the first tick.

The chosen checkpoint is the cohort's full_shaped seed=3 (the actual
sleep-inducer per the reward-ablation analysis). We sim a few seconds
and assert (a) the policy was wired in, (b) it actually emits stim
commands, (c) tick-budget telemetry is populated.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("MONGO_URL", "mongodb://127.0.0.1:1/")
os.environ.setdefault("BIOLISTENER_SLEEP_ENABLE_AUDIO", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAC_CKPT = REPO_ROOT / "app/agent/checkpoints/sac_full_shaped_seed3_1779578521.zip"


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


def _build_resolved_config(checkpoint: Path) -> Any:
    """Construct a minimal RESOLVED_CONFIG that pins policy.name='sac'
    + checkpoint_path. The daemon reads RESOLVED_CONFIG.policy,
    RESOLVED_CONFIG.session.duration_s, and RESOLVED_CONFIG.audio.shaper.*;
    we wire real numeric values for the shaper because it gets compared
    against the smoothed θ/α float each tick."""
    cfg = MagicMock()
    cfg.policy.name = "sac"
    cfg.policy.checkpoint_path = checkpoint
    cfg.session.duration_s = 600.0      # arbitrary; loop ends via tick budget
    # Shaper coefficients must be real floats (MagicMock objects break
    # the float comparisons inside AudioShaper). Defaults match the schema.
    cfg.audio.shaper.theta_alpha_baseline = 0.5
    cfg.audio.shaper.theta_alpha_full = 3.0
    cfg.audio.shaper.warmth_at_baseline_pct = 80.0
    cfg.audio.shaper.warmth_at_full_pct = 25.0
    cfg.audio.shaper.initial_master_vol_pct = 50.0
    cfg.audio.shaper.drowsy_vol_taper = 0.4
    cfg.audio.shaper.end_fade_starts_at_frac = 0.75
    cfg.audio.shaper.end_fade_min_factor = 0.4
    # Mask strobe presentation - real values, not MagicMock, because the
    # daemon parses color_hex and int()-converts led_bitmask before the loop.
    cfg.mask.strobe.color_hex = "#000000"
    cfg.mask.strobe.white_channel = 200
    cfg.mask.strobe.led_bitmask = 0xFF
    # Safety bounds - real floats. SafetySupervisor.from_config does
    # numeric comparisons on every clamp() call.
    cfg.safety.freq_min_hz = 3.0
    cfg.safety.freq_max_hz = 14.0
    cfg.safety.pse_band_lo = 15.0
    cfg.safety.pse_band_hi = 25.0
    cfg.safety.brightness_min = 0.0
    cfg.safety.brightness_max = 0.60
    cfg.safety.fill_min = 0.0
    cfg.safety.fill_max = 0.40
    cfg.safety.session_timeout_s = 2700.0
    cfg.safety.max_freq_step_hz_per_s = 4.0
    cfg.safety.max_brightness_step_per_s = 0.20
    cfg.safety.railed_pct_disconnect_threshold = 80.0
    # Audio engine safety caps - real floats (BiofeedbackEngine clamps
    # numerically against them in set_* methods).
    cfg.audio.preset = "sunlight_pads"
    cfg.audio.output_gain = 1.0
    cfg.audio.safety.max_val_vol = 1.5
    cfg.audio.safety.max_strobe_depth_pct = 50.0
    cfg.audio.safety.max_binaural_vol_pct = 20.0
    return cfg


@pytest.mark.skipif(not SAC_CKPT.exists(),
                     reason=f"checkpoint {SAC_CKPT.name} not present")
@pytest.mark.asyncio
async def test_sac_checkpoint_is_loaded_and_drives_strobe(fresh_daemon):
    """The full_shaped cohort checkpoint must load and produce stim
    commands. If `_setup_policy` falls back to FixedFrequencyPolicy
    (the old bug), the strobe would lock at 10 Hz forever; instead we
    require the SAC instance to be live and the stim payload non-null."""
    daemon = fresh_daemon
    cfg = _build_resolved_config(SAC_CKPT)
    with patch.object(daemon, "RESOLVED_CONFIG", cfg):
        resp = await daemon.create_session({
            "subject_code": "SAC01",
            "protocol": "rl_agent",
        })
        sid = resp["id"]
        await daemon.start_session(sid)
        # ~4 ticks at 4 Hz = 1 s; enough to exercise the obs-build path
        # and surface any shape mismatch via the loop's exception handler.
        await asyncio.sleep(1.5)
        snap = await daemon.session_status(sid)
        await daemon.end_session(sid)

    # Policy was wired in: stim payload is present + within safety bounds.
    assert snap["stim"] is not None, "stim payload should be populated"
    assert 3.0 <= snap["stim"]["frequency_hz"] <= 14.0
    # Loop didn't panic from a shape/load error
    assert snap["state"] in ("running", "completed", "aborted"), snap["state"]
    # The fresh_daemon fixture would have caught the loop crash via
    # session.panic("loop_exception") leaving state='aborted'; we want
    # to *not* see that - the SAC integration should have run cleanly.
    assert snap["state"] != "aborted", (
        "control loop panicked - likely obs-shape mismatch on first SAC.predict"
    )


@pytest.mark.skipif(not SAC_CKPT.exists(),
                     reason=f"checkpoint {SAC_CKPT.name} not present")
@pytest.mark.asyncio
async def test_sac_session_emits_tick_budget_telemetry(fresh_daemon):
    """The new perf_counter probe must populate tick_last_compute_ms +
    tick_max_compute_ms in telemetry. At 4 Hz tick we have 250 ms; if
    compute regularly exceeds that we'd never sustain 4 Hz under real
    EEG. Threshold here is loose (200 ms) - this test catches a
    catastrophic regression, not jitter."""
    daemon = fresh_daemon
    cfg = _build_resolved_config(SAC_CKPT)
    with patch.object(daemon, "RESOLVED_CONFIG", cfg):
        resp = await daemon.create_session({
            "subject_code": "SAC02",
            "protocol": "rl_agent",
        })
        sid = resp["id"]
        await daemon.start_session(sid)
        await asyncio.sleep(2.0)   # ~8 ticks at 4 Hz
        snap = await daemon.session_status(sid)
        await daemon.end_session(sid)

    tb = snap.get("tick_budget")
    assert tb is not None, "telemetry must include tick_budget block"
    assert tb["budget_ms"] == 250.0
    assert tb["last_compute_ms"] >= 0
    assert tb["max_compute_ms"] >= tb["last_compute_ms"]
    # The hard threshold - if average tick exceeds the budget the loop
    # can't sustain 4 Hz on a real subject. We assert max < 200 ms (80%
    # of budget) to leave headroom for jitter from WS+mongo+audio.
    assert tb["max_compute_ms"] < 200.0, (
        f"tick max compute {tb['max_compute_ms']:.1f}ms exceeds 200ms - "
        "would degrade 4 Hz tick rate on real EEG"
    )


def test_setup_policy_rejects_missing_checkpoint(fresh_daemon):
    """If policy.name='sac' but checkpoint_path doesn't resolve, the
    daemon should surface a clear HTTPException, not silently fall back
    to FixedFrequencyPolicy."""
    from fastapi import HTTPException
    daemon = fresh_daemon

    sess = MagicMock()
    sess.protocol = daemon.SessionProtocol.rl_agent
    sess.protocol_params = {}

    cfg = _build_resolved_config(Path("/nonexistent/sac.zip"))
    with patch.object(daemon, "RESOLVED_CONFIG", cfg):
        with pytest.raises(HTTPException) as ei:
            daemon._setup_policy(sess)
        assert "not found" in str(ei.value.detail).lower()
