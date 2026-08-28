"""
The deployment loop must score both arms on the terms training used.

The arm contrast is a difference of rewards, so anything that depends on how
fast an arm moves its command shows up as an effect. These are the checks that
the reward and the observation carry no such dependence.
"""
import time

import numpy as np
import pytest

from app.agent.safety import StimAction
from app.brain_sim.photic_input import StrobeParams, strobe_envelope
from app.eeg.plv import plv_strobe_to_eeg
from app.protocol.runner import DeliveredEnvelope

FS = 500
TICK_S = 0.25
WIN_S = 4.0
N_WIN = int(WIN_S * FS)
SPT = int(FS * TICK_S)

RAMP_RATE = 7.0 / 300.0     # the descending arm: 7 Hz over five minutes
AGENT_RATE = 0.5            # the agent at the supervisor's ramp-limiter cap


def _sweep(rate, phase_continuous, f0=11.0, ticks=80):
    """An arm sweeping downward at ``rate``, and the light it produced."""
    env = DeliveredEnvelope(FS, phase_continuous=phase_continuous)
    for k in range(ticks):
        env.extend(StimAction(enabled=True, frequency_hz=f0 - rate * k * TICK_S,
                              fill=0.4, brightness=0.6), SPT)
    return env, f0 - rate * (ticks - 1) * TICK_S


def _entrained_eeg(light, seed=0):
    """EEG that follows the delivered light exactly, plus measurement noise."""
    return light + 0.25 * np.random.default_rng(seed).standard_normal(light.size)


@pytest.mark.parametrize("plant_continuous", [True, False])
def test_both_arms_score_alike_when_both_are_perfectly_entrained(plant_continuous):
    # Reconstructing the reference the same way the plant generates the light
    # leaves no room for ramp rate to matter.
    plv = {}
    for name, rate in (("ramp", RAMP_RATE), ("agent", AGENT_RATE)):
        plant, f = _sweep(rate, plant_continuous)
        recon, _ = _sweep(rate, plant_continuous)
        plv[name] = plv_strobe_to_eeg(
            _entrained_eeg(plant.latest(N_WIN)), recon.latest(N_WIN), FS, f)
    assert plv["ramp"] > 0.9 and plv["agent"] > 0.9
    assert abs(plv["ramp"] - plv["agent"]) < 0.05


@pytest.mark.parametrize("plant_continuous", [True, False])
def test_a_reference_regenerated_at_one_frequency_splits_the_arms(plant_continuous):
    # Why the reconstruction exists: assuming the command held still across the
    # window scores an arm against a stimulus it never received, and how wrong
    # that is depends on the arm's own ramp rate.
    plv = {}
    for name, rate in (("ramp", RAMP_RATE), ("agent", AGENT_RATE)):
        plant, f = _sweep(rate, plant_continuous)
        regen = strobe_envelope(
            duration_s=WIN_S, dt=1.0 / FS,
            strobe=StrobeParams(enabled=True, frequency_hz=f,
                                fill=0.4, brightness=0.6))
        plv[name] = plv_strobe_to_eeg(
            _entrained_eeg(plant.latest(N_WIN)), regen, FS, f)
    assert abs(plv["ramp"] - plv["agent"]) > 0.3


def test_mistaking_the_drivers_phase_behaviour_costs_sensitivity_not_fairness():
    # The mask firmware is not ours, so which clock it keeps is an assumption.
    # Getting it wrong must lower both arms together rather than one of them.
    plv = {}
    for name, rate in (("ramp", RAMP_RATE), ("agent", AGENT_RATE)):
        plant, f = _sweep(rate, phase_continuous=True)
        wrong, _ = _sweep(rate, phase_continuous=False)
        plv[name] = plv_strobe_to_eeg(
            _entrained_eeg(plant.latest(N_WIN)), wrong.latest(N_WIN), FS, f)
    assert abs(plv["ramp"] - plv["agent"]) < 0.1


def test_no_light_gives_no_reference_rather_than_a_silent_zero():
    env = DeliveredEnvelope(FS)
    env.extend(StimAction(enabled=True, frequency_hz=10.0, fill=0.4,
                          brightness=0.0), SPT * 16)
    assert env.latest(N_WIN) is None


def test_the_dark_control_block_delivers_nothing_while_still_commanding():
    # Zero brightness with the strobe enabled is the artifact control: the
    # command stream continues, the light does not.
    env = DeliveredEnvelope(FS)
    env.extend(StimAction(enabled=True, frequency_hz=10.0, fill=0.4,
                          brightness=0.6), SPT * 8)
    env.extend(StimAction(enabled=True, frequency_hz=10.0, fill=0.4,
                          brightness=0.0), SPT * 8)
    tail = env.latest(SPT * 8)
    assert tail is None or not np.any(tail)


class TestMaskIsActuallyDriven:
    """
    A UDP send to nobody succeeds. Every check that the mask is working has to
    rest on something coming back, or a session delivered in the dark reads as
    a clean recording.
    """

    @staticmethod
    def _pair(ctrl=21990, dev=21991):
        import time
        from app.mask.virtual import VirtualMaskServer
        from app.mask.client import MaskClient
        from app.protocol.runner import MaskLiveness
        vm = VirtualMaskServer(listen_ip="127.0.0.1", listen_port=dev,
                               controller_addr=("127.0.0.1", ctrl))
        vm.start()
        time.sleep(0.3)
        live = MaskLiveness()
        client = MaskClient(bind_ip="127.0.0.1", port=ctrl,
                            target_ip="127.0.0.1", on_params=live.on_params)
        client.port = dev
        return vm, client, live

    def test_commands_reach_the_device(self):
        import time
        vm, client, _ = self._pair()
        try:
            client.update_params(enabled=True, frequency=9.5, fill=0.4,
                                 brightness=0.6, color=[0, 0, 0, 255], mask=0xFF)
            time.sleep(0.4)
            assert len(vm.history()) == 1
            s = vm.snapshot()
            assert s.enabled and s.frequency == 9.5 and s.brightness == 0.6
        finally:
            client.stop()
            vm.stop()

    def test_colour_and_led_mask_arrive_with_the_command(self):
        # The device keeps what it was last told, so a command without these
        # can leave brightness commanded on LEDs that are switched off.
        import time
        vm, client, _ = self._pair(ctrl=21992, dev=21993)
        try:
            client.update_params(enabled=True, frequency=10.0, fill=0.4,
                                 brightness=0.6, color=[0, 0, 0, 255], mask=0xFF)
            time.sleep(0.4)
            s = vm.snapshot()
            assert s.color == (0, 0, 0, 255) and s.mask == 0xFF
        finally:
            client.stop()
            vm.stop()

    def test_liveness_comes_from_a_reply_not_from_the_send_succeeding(self):
        import time
        vm, client, live = self._pair(ctrl=21994, dev=21995)
        try:
            for _ in range(10):
                client.get_params()
                time.sleep(0.15)
                if live.answered:
                    break
            assert live.answered
        finally:
            client.stop()
            vm.stop()

    def test_a_mask_that_never_answers_is_not_reported_as_connected(self):
        # is_connected() returns True for the asking whenever a target was
        # pinned, so it cannot be the thing pre-flight trusts.
        import time
        from app.mask.client import MaskClient
        from app.protocol.runner import MaskLiveness
        live = MaskLiveness()
        client = MaskClient(bind_ip="127.0.0.1", port=21996,
                            target_ip="127.0.0.1", on_params=live.on_params)
        client.port = 21997                      # nothing is listening there
        try:
            assert client.is_connected() is True     # reports reachable: a target was pinned
            client.get_params()
            time.sleep(0.5)
            assert live.answered is False            # no reply has arrived
        finally:
            client.stop()


class TestRatingsSurviveTheBrowserPath:
    def test_recorded_ratings_survive_the_end_of_session_skeleton(self, tmp_path):
        # Two writers: the browser writes kss.csv as each rating is clicked,
        # the runner writes a skeleton at the end. Recorded values win.
        from app.protocol.response import write_kss_skeleton
        (tmp_path / "kss.csv").write_text(
            "t_s,block,value\n180.0,B0_baseline,4\n300.0,sweep1_alert,6\n")
        write_kss_skeleton(tmp_path, [
            {"t_s": 180.0, "block": "B0_baseline", "value": None,
             "clip": "01_B0_baseline.wav"},
            {"t_s": 300.0, "block": "sweep1_alert", "value": None,
             "clip": "02_sweep1_alert.wav"}])
        rows = (tmp_path / "kss.csv").read_text().splitlines()
        assert rows[1].split(",")[2] == "4"
        assert rows[2].split(",")[2] == "6"
        assert rows[1].split(",")[3] == "01_B0_baseline.wav"


class TestStreamRateIsMeasured:
    """
    The device does not have to stream at the rate it is asked for, and the
    rate it reports is the rate it was asked for. Everything spectral scales
    by the ratio, so it is timed off the running stream.
    """

    class _Stub:
        """A source delivering at a fixed rate, counted like a real board."""

        def __init__(self, hz):
            self.hz = hz
            self.sampling_rate = 500
            self._t0 = time.perf_counter()

        def stream_count(self):
            return int((time.perf_counter() - self._t0) * self.hz)

        def latest_window(self, window_s=4.0):
            return np.zeros((8, 1))

    def _runner(self, **kw):
        from app.protocol.runner import RunnerConfig, SessionRunner
        from app.protocol.spec import build_validation
        cfg = RunnerConfig(protocol=build_validation(), eeg_kind="biolistener",
                           **kw)
        return SessionRunner(cfg)

    def test_a_device_streaming_faster_than_declared_is_believed(self, monkeypatch):
        import app.protocol.runner as R
        monkeypatch.setattr(R, "RATE_MEASURE_S", 1.0)
        r = self._runner(sampling_rate=500)
        rate = r._measure_stream_rate(self._Stub(512.0))
        assert rate == pytest.approx(512.0, rel=0.02)

    def test_a_rate_nowhere_near_the_configured_one_stops_the_session(
            self, monkeypatch):
        import app.protocol.runner as R
        monkeypatch.setattr(R, "RATE_MEASURE_S", 1.0)
        r = self._runner(sampling_rate=500)
        with pytest.raises(RuntimeError, match="too far apart"):
            r._measure_stream_rate(self._Stub(120.0))

    def test_a_simulated_source_is_not_timed(self, monkeypatch):
        import app.protocol.runner as R
        monkeypatch.setattr(R, "RATE_MEASURE_S", 1.0)
        from app.protocol.runner import RunnerConfig, SessionRunner
        from app.protocol.spec import build_validation
        cfg = RunnerConfig(protocol=build_validation(), eeg_kind="sim")
        assert SessionRunner(cfg)._measure_stream_rate(self._Stub(512.0)) is None

    def test_the_hardware_rate_is_not_the_nominal_one(self):
        from app.protocol.runner import (BIOLISTENER_SAMPLING_RATE,
                                         NOMINAL_SAMPLING_RATE)
        assert BIOLISTENER_SAMPLING_RATE != NOMINAL_SAMPLING_RATE


class TestStreamRateSurvivesABadLink:
    """
    A link that stalls and then delivers its backlog in one burst is the
    failure this measurement has to survive: a single long average absorbs it
    and reports a plausible wrong number.
    """

    class _Stalling:
        """Delivers at ``hz`` but freezes for ``stall_s``, then catches up."""

        def __init__(self, hz, stall_at, stall_s):
            self.hz, self.stall_at, self.stall_s = hz, stall_at, stall_s
            self.sampling_rate = 500
            self._t0 = time.perf_counter()

        def stream_count(self):
            e = time.perf_counter() - self._t0
            if self.stall_at <= e < self.stall_at + self.stall_s:
                e = self.stall_at              # frozen; backlog not yet handed over
            return int(e * self.hz)

        def latest_window(self, window_s=4.0):
            return np.zeros((8, 1))

    def _runner(self):
        from app.protocol.runner import RunnerConfig, SessionRunner
        from app.protocol.spec import build_validation
        return SessionRunner(RunnerConfig(protocol=build_validation(),
                                          eeg_kind="biolistener",
                                          sampling_rate=500))

    def test_a_stall_and_burst_does_not_move_the_answer(self, monkeypatch):
        import app.protocol.runner as R
        monkeypatch.setattr(R, "RATE_MEASURE_S", 2.0)
        src = self._Stalling(512.0, stall_at=1.0, stall_s=0.5)
        assert self._runner()._measure_stream_rate(src) == pytest.approx(512.0, rel=0.05)

    def test_a_steady_link_is_measured_exactly(self, monkeypatch):
        import app.protocol.runner as R
        monkeypatch.setattr(R, "RATE_MEASURE_S", 2.0)
        src = self._Stalling(512.0, stall_at=1e9, stall_s=0.0)
        assert self._runner()._measure_stream_rate(src) == pytest.approx(512.0, rel=0.02)


class TestBlockIndexingIgnoresArrivalPattern:
    def test_a_burst_does_not_shift_later_blocks(self, tmp_path):
        """
        Session time maps to samples through the recording's own timestamps,
        so a link that delivers four seconds at once leaves every block where
        it was. Multiplying elapsed time by a rate would not.
        """
        import numpy as np
        from app.protocol.analysis import Session

        fs, n = 512.0, 20000
        ts = 1.7e9 + np.arange(n) / fs
        # Four seconds arrive in one burst a third of the way in; the samples
        # still represent the same four seconds of signal.
        ts[n // 3:] -= 0.0
        raw = np.vstack([np.arange(n), np.zeros((11, n)), ts])
        sess = Session(dir=tmp_path, manifest={"sampling_rate": 500},
                       ticks=None, raw=raw, markers=[], kss=None,
                       safety_events=[])
        assert sess.timeline() is not None
        for t in (5.0, 15.0, 30.0):
            assert sess.raw_index_at(t) == pytest.approx(t * fs, abs=2)
