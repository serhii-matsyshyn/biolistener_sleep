"""A session drives the mask, records everything, and stays inside the envelope."""
import json

import pytest

from app.agent.safety import BRIGHTNESS_MAX_V1, FILL_MAX, FREQ_MAX_HZ, FREQ_MIN_HZ
from app.protocol.spec import Block, BlockKind, Eyes, Protocol
from app.protocol.runner import RunnerConfig, SessionRunner


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One short run against the emulator and the in-process brain."""
    proto = Protocol(
        name="itest", seed=0, montage="occipital_alpha",
        description="integration test",
        blocks=(
            Block("rest", BlockKind.REST, 2.0, Eyes.CLOSED, "settle"),
            Block("steady", BlockKind.REST, 2.0, Eyes.OPEN, "eyes_open",
                  steady_brightness=0.05),
            Block("drive", BlockKind.FIXED, 3.0, Eyes.CLOSED, "strobe_start",
                  fixed_hz=10.0, kss_at_end=True),
            Block("sweep", BlockKind.SWEEP, 26.0, Eyes.CLOSED, "sweep_start",
                  probe_hz=(8.0, 10.0), dwell_s=2.0),
        ),
    )
    out = tmp_path_factory.mktemp("sessions")
    cfg = RunnerConfig(protocol=proto, eeg_kind="sim", virtual_mask=True,
                       mask_ip="127.0.0.1", audio_cues=False,
                       session_root=out, tick_hz=4.0)
    return SessionRunner(cfg).run()


def _ticks(d):
    import pandas as pd
    return pd.read_csv(d / "ticks.csv")


def test_session_completes_and_writes_every_artifact(recorded):
    for name in ("manifest.json", "ticks.csv", "markers.jsonl",
                 "safety_events.jsonl", "raw_eeg.csv", "notes.md"):
        assert (recorded / name).exists(), name
    assert json.loads((recorded / "manifest.json").read_text())["status"] == "completed"


def test_manifest_records_what_is_needed_to_reproduce_the_run(recorded):
    m = json.loads((recorded / "manifest.json").read_text())
    for key in ("protocol", "seed", "montage", "tick_hz", "git_commit",
                "sweeps", "blocks", "eeg_kind"):
        assert key in m, key
    assert m["sweeps"]["sweep"]["order"]


def test_every_block_produced_ticks(recorded):
    tk = _ticks(recorded)
    assert set(tk["block"]) == {"rest", "steady", "drive", "sweep"}


def test_delivered_commands_never_leave_the_safety_envelope(recorded):
    tk = _ticks(recorded)
    # The frequency floor and the duty ceiling both bound flicker, so both
    # apply to flickering light. Continuous light is bounded by brightness,
    # which every row answers to.
    flicker = tk[tk["steady"] == 0]
    lit = flicker[flicker["act_brightness"] > 0]
    assert (lit["act_freq_hz"] >= FREQ_MIN_HZ).all()
    assert (lit["act_freq_hz"] <= FREQ_MAX_HZ).all()
    assert (flicker["act_fill"] <= FILL_MAX + 1e-9).all()
    assert (tk["act_brightness"] <= BRIGHTNESS_MAX_V1 + 1e-9).all()


def test_continuous_light_stays_under_its_own_ceiling(recorded):
    from app.agent.safety import STEADY_BRIGHTNESS_MAX
    tk = _ticks(recorded)
    st = tk[tk["steady"] == 1]
    assert not st.empty
    assert (st["act_brightness"] <= STEADY_BRIGHTNESS_MAX + 1e-9).all()


def test_rest_blocks_deliver_no_light(recorded):
    tk = _ticks(recorded)
    assert (tk[tk["block"] == "rest"]["act_brightness"] == 0).all()


def test_a_steady_block_delivers_light_with_no_flicker(recorded):
    tk = _ticks(recorded)
    st = tk[tk["block"] == "steady"]
    assert (st["steady"] == 1).all()
    assert (st["act_brightness"] > 0).all()
    # The firmware holds light continuously when commanded 0 Hz; a non-zero
    # frequency here would put harmonics into the band the alpha check reads.
    assert (st["act_freq_hz"] == 0.0).all()
    assert (st["act_fill"] == 1.0).all()


def test_a_steady_block_offers_no_phase_to_lock_to(recorded):
    """
    The PLV reference is a rolling buffer, so a steady block that follows a
    lit one still holds the previous block's flicker for one buffer length.
    What must hold is that steady light contributes no phase of its own:
    once the buffer has turned over, there is nothing to lock to.
    """
    tk = _ticks(recorded)
    st = tk[tk["steady"] == 1]
    assert not st.empty
    settled = st.iloc[len(st) // 2:]
    assert (settled["plv"] == 0.0).all()


def test_fixed_block_settles_on_its_commanded_frequency(recorded):
    tk = _ticks(recorded)
    assert tk[tk["block"] == "drive"]["cmd_freq_hz"].eq(10.0).all()


def test_sweep_commands_only_its_own_probes(recorded):
    tk = _ticks(recorded)
    sw = tk[(tk["block"] == "sweep") & tk["probe_hz"].notna()]
    assert set(sw["probe_hz"].unique()) <= {8.0, 10.0}


def test_markers_bracket_every_block(recorded):
    ms = [json.loads(l) for l in (recorded / "markers.jsonl").read_text().splitlines()]
    starts = [m["block"] for m in ms if m["event"] == "block_start"]
    ends = [m["block"] for m in ms if m["event"] == "block_end"]
    assert starts == ends == ["rest", "steady", "drive", "sweep"]
    assert ms[0]["event"] == "session_start" and ms[-1]["event"] == "session_end"


def test_a_rating_is_prompted_where_the_protocol_asks_for_one(recorded):
    ms = [json.loads(l) for l in (recorded / "markers.jsonl").read_text().splitlines()]
    assert [m["block"] for m in ms if m["event"] == "kss_prompt"] == ["drive"]


def test_raw_samples_are_indexable_from_the_tick_log(recorded):
    from app.protocol import analysis as an
    sess = an.load(recorded)
    assert sess.raw is not None and sess.raw.shape[1] > 0
    tk = _ticks(recorded)
    assert tk["raw_sample"].notna().all()
    assert tk["raw_sample"].is_monotonic_increasing
    # The index must land inside the recording, not past its end.
    assert sess.raw_index_at(float(tk["t_s"].max())) <= sess.raw.shape[1]


def test_the_recorded_session_loads_back(recorded):
    from app.protocol import analysis as an
    sess = an.load(recorded)
    assert sess.sampling_rate > 0
    assert an.sweep_blocks(sess) == ["sweep"]
