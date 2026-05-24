"""
DemoController - replicates the source mask_strobe_controller Flask app
(VLC audio + CSV-driven strobe) with a generative-audio toggle on top.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.runtime.demo import DemoController, get_demo, shutdown_demo


@pytest.fixture(autouse=True)
def _isolated_singleton():
    shutdown_demo()
    yield
    shutdown_demo()


def test_lazy_boot_on_first_request():
    d = DemoController()
    assert d._initialised is False
    state = d.get_state()
    assert d._initialised is True
    assert "mask" in state
    assert "track" in state
    assert "audio" in state
    assert "safety_events" in state


def test_get_demo_returns_same_singleton():
    a = get_demo()
    b = get_demo()
    assert a is b


def test_initial_state_defaults():
    state = DemoController().get_state()
    assert state["mask"]["enabled"] is True
    assert state["mask"]["brightness"] == 0.10
    assert state["mask"]["color"] == [0, 0, 0, 255]
    assert state["mask"]["bitmask"] == 0xFF
    assert state["track"]["sound_path"] is None
    assert state["track"]["is_playing"] is False
    assert state["audio"]["generative_enabled"] is False


def test_update_params_brightness_clip():
    d = DemoController()
    s = d.update_params(brightness=1.5)["mask"]
    assert s["brightness"] == 1.0
    s = d.update_params(brightness=-0.2)["mask"]
    assert s["brightness"] == 0.0


def test_update_params_color_round_trip():
    s = DemoController().update_params(color=[255, 0, 128, 200])["mask"]
    assert s["color"] == [255, 0, 128, 200]


def test_update_params_bitmask_masked_to_8_bits():
    s = DemoController().update_params(bitmask=0xABCD)["mask"]
    assert s["bitmask"] == 0xCD


def test_update_params_strobe_toggle():
    d = DemoController()
    assert d.update_params(enabled=False)["mask"]["enabled"] is False
    assert d.update_params(enabled=True)["mask"]["enabled"] is True


def test_select_session_with_strobe_csv_loads_first_on(tmp_path: Path):
    """The CSV's first rising edge sets ``track.first_on_s``."""
    csv = tmp_path / "s.csv"
    # First rise at 2.0 s. Then two more cycles so n_cycles >= 1.
    csv.write_text(textwrap.dedent("""\
        Timestamp_NS,Torch_State
        100000,0
        2000000000,1
        2050000000,0
        2100000000,1
        2150000000,0
        2200000000,1
        2250000000,0
    """))
    state = DemoController().select_session(
        sound_path=None, strobe_path=str(csv),
    )
    assert state["track"]["strobe_path"] == str(csv)
    assert state["track"]["first_on_s"] == pytest.approx(2.0, abs=1e-3)


def test_select_session_missing_sound_keeps_state_consistent():
    d = DemoController()
    state = d.select_session(
        sound_path="/no/such/file.mp3", strobe_path=None,
    )
    # Sound load failed -> sound_path stays None, no crash.
    assert state["track"]["sound_path"] is None
    assert state["track"]["is_playing"] is False


def test_play_pause_are_safe_without_track():
    """Calling play/pause before select_session must not raise."""
    d = DemoController()
    assert d.play()["track"]["is_playing"] is False
    assert d.pause()["track"]["is_playing"] is False
    assert d.seek(10.0)["track"]["current_pos_s"] == 0.0


def test_update_audio_records_slider_state_without_pyo(monkeypatch):
    """If pyo can't boot, generative_enabled bounces back to False but
    slider state is still recorded."""
    import builtins
    real_import = builtins.__import__

    def fail_pyo(name, *a, **kw):
        if name == "app.audio.engine":
            raise RuntimeError("simulated pyo failure")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fail_pyo)
    state = DemoController().update_audio(
        generative_enabled=True, master_vol_pct=70.0,
    )
    assert state["audio"]["generative_enabled"] is False
    assert state["audio"]["master_vol_pct"] == 70.0


def test_update_audio_preset_switch():
    s = DemoController().update_audio(preset="warm_blanket")["audio"]
    assert s["preset"] == "warm_blanket"


def test_stop_all_silences_everything():
    d = DemoController()
    d.update_params(enabled=True, brightness=0.5)
    state = d.stop_all()
    assert state["mask"]["enabled"] is False
    assert state["mask"]["brightness"] == 0.0
    assert state["audio"]["generative_enabled"] is False
    assert state["audio"]["master_vol_pct"] == 0.0
