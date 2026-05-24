"""
Verify AudioController works in headless (no-audio-device) mode and produces
a clean event log for replay. We do NOT boot the pyo Server in tests.
"""
from __future__ import annotations

from app.audio.controller import AudioController, AudioParams


def test_controller_no_engine_logs_events():
    c = AudioController(engine=None)
    c.set_sync_freq(9.5)
    c.set_strobe_depth(15)
    c.set_master_vol(40)
    assert c.params.sync_freq_hz == 9.5
    assert c.params.strobe_depth_pct == 15
    assert c.params.master_vol_pct == 40
    log = c.event_log()
    assert [e.param for e in log] == [
        "sync_freq_hz", "strobe_depth_pct", "master_vol_pct"
    ]


def test_controller_clips_out_of_range():
    c = AudioController(engine=None)
    c.set_sync_freq(-5)              # below 0
    assert c.params.sync_freq_hz == 0.0
    c.set_sync_freq(999)             # above 40 Hz upper limit
    assert c.params.sync_freq_hz == 40.0
    c.set_strobe_depth(150)          # above 100%
    assert c.params.strobe_depth_pct == 100.0
    c.set_master_vol(-30)            # below 0
    assert c.params.master_vol_pct == 0.0


def test_initial_params_are_logged_into_event_log_only_on_setters():
    """Construction shouldn't flood the log; only explicit setters log."""
    c = AudioController(engine=None, initial=AudioParams(sync_freq_hz=10.0))
    assert c.event_log() == []
    c.set_sync_freq(10.0)
    assert len(c.event_log()) == 1
