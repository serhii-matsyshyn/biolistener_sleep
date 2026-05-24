"""
Tests for the three experimental protocols.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import yaml

from app.agent.policies import BaselineNoStimPolicy, FixedFrequencyPolicy
from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ, SafetySupervisor
from app.config import BioListenerSleepConfig
from app.open_loop_track.catalog import OpenLoopTrackCatalog
from app.open_loop_track.csv_playback import CsvEventPlayback
from app.open_loop_track.playback import OpenLoopTrackPlayback
from app.open_loop_track.policy import OpenLoopTrackPolicy
from app.open_loop_track.safety_review import review


def test_baseline_no_stim_decodes_to_disabled_after_safety():
    pol = BaselineNoStimPolicy()
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    # action[0] encodes freq norm = -1 -> decoded freq = FREQ_MIN_HZ
    decoded_freq = FREQ_MIN_HZ + (action[0] + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ)
    assert abs(decoded_freq - FREQ_MIN_HZ) < 1e-6
    # brightness norm = -1 -> 0
    decoded_bri = (action[2] + 1.0) / 2.0
    assert decoded_bri == 0.0
    decoded_fill = (action[1] + 1.0) / 2.0
    assert decoded_fill == 0.0


def test_baseline_passes_through_safety_clean():
    """No safety events fire because the encoded action is already safe."""
    from app.agent.env import SleepOnsetEnv
    pol = BaselineNoStimPolicy()
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    decoder = SleepOnsetEnv.__dict__["_decode_action"]
    stim = decoder(None, action)
    sup = SafetySupervisor()
    safe = sup.clamp(stim)
    # Brightness and fill should still be 0
    assert safe.brightness == 0.0
    assert safe.fill == 0.0
    # No clamp events
    assert all(e.kind != "pse_guard" for e in sup.events)


@pytest.fixture
def sample_strobe_config(tmp_path: Path) -> Path:
    """
    A minimal segment config we author ourselves - no external dependency.
    Single 60-second segment: linear ramp from ~10 Hz down to a slower
    frequency, modest duty cycle.
    """
    cfg = {
        "myactivity": {
            "10": [
                # start_us, duration_us, offset, divisor, m_duty, a_duty, m_freq, a_freq
                [0, 60_000_000, 0.0, 60_000_000.0,
                 0.0, 200.0,    # duty_raw = 200 (per-mille -> 20% fill)
                 -5.0, 11000.0], # freq_raw 11000 -> 10000 over the segment
            ],
        },
    }
    out = tmp_path / "sc.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def test_open_loop_track_playback_evaluates_segments(sample_strobe_config: Path):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    # Mid-segment params: at t=30s the freq ramp is halfway through
    params = pb.params_at(t_s=30.0)
    assert params is not None
    freq, fill = params
    # freq_raw at t=30 = (30e6 * 100 - 0) / 60e6 * -5 + 11000 = -25 + 11000 = ≈10975
    # -> freq_hz ≈ 10.975
    assert 10.0 < freq < 11.5
    assert 0.0 < fill < 1.0


def test_open_loop_track_playback_returns_none_past_end(sample_strobe_config: Path):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    assert pb.params_at(t_s=60.1) is None
    assert pb.params_at(t_s=-1.0) is None


def test_open_loop_track_playback_duration(sample_strobe_config: Path):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    assert pb.duration_s == pytest.approx(60.0, abs=0.01)


def _write_csv(path: Path, rows: list[tuple[int, int]]) -> Path:
    """Helper - write a (Timestamp_NS, Torch_State) CSV from a list of rows."""
    lines = ["Timestamp_NS,Torch_State"]
    for ts, state in rows:
        lines.append(f"{ts},{state}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_csv_event_playback_recovers_freq_and_duty(tmp_path: Path):
    """
    Author a tiny event log with a known schedule:
      - Three rising edges 100 ms apart -> period 100 ms -> 10 Hz
      - Each rising edge stays ON for 20 ms -> duty = 20%
    """
    period_ns = 100_000_000   # 100 ms
    on_ns = 20_000_000        # 20 ms ON per cycle
    base = 1_000_000_000       # start at 1 s
    rows: list[tuple[int, int]] = [(0, 0)]   # baseline OFF at t=0
    for k in range(3):
        rise = base + k * period_ns
        rows.append((rise, 1))
        rows.append((rise + on_ns, 0))
    csv_path = _write_csv(tmp_path / "evt.csv", rows)

    pb = CsvEventPlayback.from_csv(csv_path)
    # Two full cycles can be characterized (third rising edge has no successor)
    assert pb.n_cycles == 2

    # In the quiet zone (before first rising edge): (0, 0)
    assert pb.params_at(t_s=0.5) == (0.0, 0.0)

    # Inside the first cycle (1.0..1.1 s): freq ≈ 10 Hz, duty ≈ 0.20
    freq, fill = pb.params_at(t_s=1.05)
    assert 9.5 < freq < 10.5
    assert 0.18 < fill < 0.22


def test_csv_event_playback_returns_none_past_end(tmp_path: Path):
    rows = [(0, 0), (1_000_000_000, 1), (1_010_000_000, 0),
            (1_100_000_000, 1), (1_110_000_000, 0)]
    csv_path = _write_csv(tmp_path / "short.csv", rows)
    pb = CsvEventPlayback.from_csv(csv_path)
    # Past the last cycle's projected end -> None
    assert pb.params_at(t_s=2.0) is None
    # Negative time -> None
    assert pb.params_at(t_s=-0.1) is None


def test_csv_event_playback_powers_open_loop_track_policy(tmp_path: Path):
    """End-to-end: CsvEventPlayback duck-types into OpenLoopTrackPolicy."""
    rows = [(0, 0)]
    period_ns = 100_000_000
    on_ns = 20_000_000
    for k in range(5):
        rise = 1_000_000_000 + k * period_ns
        rows.append((rise, 1))
        rows.append((rise + on_ns, 0))
    csv_path = _write_csv(tmp_path / "pol.csv", rows)
    pb = CsvEventPlayback.from_csv(csv_path)
    pol = OpenLoopTrackPolicy(playback=pb, brightness=0.30)
    pol.reset()
    # Jump ahead so we're inside the first cycle window
    pol._start_t -= 1.05
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    assert action.shape == (5,)
    decoded = FREQ_MIN_HZ + (action[0] + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ)
    # CSV -> 10 Hz, decoded should match (within encoding rounding)
    assert 9.0 < decoded < 11.0


def test_open_loop_track_policy_emits_clamped_action(sample_strobe_config: Path):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    pol = OpenLoopTrackPolicy(playback=pb, brightness=0.30)
    pol.reset()
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    assert action.shape == (5,)
    # Decoded freq must be in safe range (clamped during encoding)
    decoded = FREQ_MIN_HZ + (action[0] + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ)
    assert FREQ_MIN_HZ <= decoded <= FREQ_MAX_HZ


def test_open_loop_track_policy_emits_off_past_end(sample_strobe_config: Path,
                                                     monkeypatch):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    pol = OpenLoopTrackPolicy(playback=pb)
    pol.reset()
    # Jump the policy's clock 70s forward to simulate end of track
    pol._start_t -= 70.0
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    # First three normalized fields (freq, fill, brightness) should be -1 (OFF)
    assert action[0] == -1.0
    assert action[1] == -1.0
    assert action[2] == -1.0


def test_safety_review_passes_for_in_range_activity(sample_strobe_config: Path):
    pb = OpenLoopTrackPlayback.from_config_yaml(
        sample_strobe_config, "myactivity", "10",
    )
    cfg = BioListenerSleepConfig()
    result = review(pb, cfg, "myactivity", duration_min=1)
    assert result.peak_freq_hz < cfg.safety.freq_max_hz + 5  # within ramp tail
    assert result.samples_in_pse_band == 0


def test_safety_review_flags_pse_band(tmp_path: Path):
    """Segment intentionally pinned at 18 Hz (PSE band)."""
    cfg_path = tmp_path / "pse.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "bad": {
            "1": [
                # Constant 18000 freq_raw -> 18 Hz, well inside PSE band
                [0, 30_000_000, 0.0, 30_000_000.0, 0.0, 100.0, 0.0, 18000.0],
            ],
        },
    }))
    pb = OpenLoopTrackPlayback.from_config_yaml(cfg_path, "bad", "1")
    cfg = BioListenerSleepConfig()
    result = review(pb, cfg, "bad", duration_min=1)
    assert result.samples_in_pse_band > 0
    assert any("PSE band" in w for w in result.warnings)


def test_catalog_parses_yaml_with_yaml1_1_quirky_keys(tmp_path: Path):
    """
    Real-world external catalogs have unquoted YAML keys that PyYAML 1.1
    parses as Python booleans or ints.
    """
    yml = tmp_path / "sessions.yaml"
    yml.write_text(dedent("""
        configuration:
          sounds_dir: snd
          strobes_dir: strb
        blocks:
          Mind Games:
            Yes:                     # YAML 1.1 -> Python True
              default:
                sound: "a.mp3"
                strobe: "x.csv"
          Sleep Better:
            "Befriending Sleep":
              10:                    # parses as Python int
                sound: "b.mp3"
                strobe: "y.csv"
    """).strip())
    cat = OpenLoopTrackCatalog.from_yaml(yml)
    # Both activities present despite the quirky keys
    assert len(cat.activities()) == 2
    names = {a.display_name for a in cat.activities()}
    assert "True" in names or "Yes" in names  # display_name coerced via str()
    assert "Befriending Sleep" in names
    bs = next(a for a in cat.activities() if a.display_name == "Befriending Sleep")
    assert bs.variant_for_duration(10).duration_min == 10


def test_catalog_parses_minimal_yaml(tmp_path: Path):
    yml = tmp_path / "sessions.yaml"
    yml.write_text(dedent("""
        configuration:
          sounds_dir: snd
          strobes_dir: strb
        blocks:
          MyBlock:
            "My Activity":
              "10":
                sound: "a%20b.mp3"
                strobe: "strobe-foo.csv"
              "15":
                sound: "a%20b-15.mp3"
                strobe: "strobe-foo-15.csv"
    """).strip())
    cat = OpenLoopTrackCatalog.from_yaml(yml)
    acts = cat.activities()
    assert len(acts) == 1
    act = acts[0]
    assert act.display_name == "My Activity"
    assert act.block == "MyBlock"
    assert set(act.variants) == {"10", "15"}
    # Files in the external folder are stored as a flat directory with
    # URL-encoded filenames (literal "%20" characters); the path resolver
    # must NOT decode - see app/open_loop_track/catalog.py.
    p = act.variants["10"].resolved_audio_path(tmp_path, "snd")
    assert p.name == "a%20b.mp3"
    assert act.variant_for_duration(10).duration_min == 10
