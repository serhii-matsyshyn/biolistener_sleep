"""Ratings are captured without making the subject search a long recording."""
import io

import sys

import pytest

from app.protocol.response import ClipRecorder, StdinInput, write_kss_skeleton


class TestStdinInput:
    @staticmethod
    def _fed(text, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
        inp = StdinInput().start()
        if inp._t is not None:
            inp._t.join(timeout=2)
        return inp

    def test_a_digit_is_a_rating_not_a_stop(self, monkeypatch):
        inp = self._fed("6\n", monkeypatch)
        assert inp.take_rating() == 6
        assert inp.stop_reason is None

    def test_anything_else_stops_the_session(self, monkeypatch):
        inp = self._fed("\n", monkeypatch)
        assert inp.stop_reason == "manual_stop"

    def test_the_last_rating_before_the_window_closes_wins(self, monkeypatch):
        inp = self._fed("4\n7\n", monkeypatch)
        assert inp.take_rating() == 7

    def test_a_rating_is_consumed_once(self, monkeypatch):
        inp = self._fed("5\n", monkeypatch)
        assert inp.take_rating() == 5
        assert inp.take_rating() is None

    def test_clearing_discards_a_stale_answer(self, monkeypatch):
        # Otherwise a digit typed during the previous block would be recorded
        # against this one.
        inp = self._fed("5\n", monkeypatch)
        inp.clear_rating()
        assert inp.take_rating() is None

    @pytest.mark.parametrize("text", ["0\n", "10\n", "12\n"])
    def test_values_off_the_scale_stop_instead_of_scoring(self, text, monkeypatch):
        inp = self._fed(text, monkeypatch)
        assert inp.take_rating() is None
        assert inp.stop_reason == "manual_stop"


class TestClipRecorder:
    def test_disabled_records_nothing(self, tmp_path):
        r = ClipRecorder(session_dir=tmp_path, enabled=False)
        assert r.capture("01_x") is None
        assert not r.clip_dir.exists()

    def test_clips_land_in_their_own_directory(self, tmp_path):
        r = ClipRecorder(session_dir=tmp_path, seconds=0.2)
        got = r.capture("01_x")
        r.stop()
        if got is not None:                    # no capture tool on this host
            assert got.parent == tmp_path / "kss_clips"
            assert got.name == "01_x.wav"

    def test_stopping_twice_is_harmless(self, tmp_path):
        r = ClipRecorder(session_dir=tmp_path, seconds=0.2)
        r.capture("01_x")
        r.stop()
        r.stop()


class TestKssSkeleton:
    def test_writes_one_row_per_prompt_with_blank_values(self, tmp_path):
        # Transcription should be typing numbers into rows that already exist,
        # not reconstructing when each prompt happened.
        p = write_kss_skeleton(tmp_path, [
            {"t_s": 12.0, "block": "B0", "value": None, "clip": "01_B0.wav"},
            {"t_s": 90.0, "block": "s1", "value": None, "clip": "02_s1.wav"},
        ])
        lines = p.read_text().strip().splitlines()
        assert lines[0] == "t_s,block,value,clip"
        assert lines[1] == "12.0,B0,,01_B0.wav"
        assert len(lines) == 3

    def test_a_typed_rating_is_already_filled_in(self, tmp_path):
        p = write_kss_skeleton(tmp_path, [
            {"t_s": 12.0, "block": "B0", "value": 7, "clip": "01_B0.wav"}])
        assert p.read_text().strip().splitlines()[1] == "12.0,B0,7,01_B0.wav"

    def test_it_is_readable_as_the_analysis_expects(self, tmp_path):
        import pandas as pd
        write_kss_skeleton(tmp_path, [
            {"t_s": 12.0, "block": "B0", "value": 7, "clip": "a.wav"}])
        df = pd.read_csv(tmp_path / "kss.csv")
        assert list(df.columns) == ["t_s", "block", "value", "clip"]
        assert df.loc[0, "value"] == 7


class TestCaptureDeviceSelection:
    """
    A laptop's headphone-jack microphone enumerates ahead of the built-in one
    and opens happily with nothing plugged in, returning frames of silence.
    Taking the first device SDL lists therefore recorded six empty clips and
    lost a session's spoken ratings.
    """

    def _with_devices(self, monkeypatch, devices, peaks):
        import app.protocol.response as R

        class _Audio:
            @staticmethod
            def get_audio_device_names(capture):
                return list(devices)

        class _Mixer:
            @staticmethod
            def get_init():
                return True

        monkeypatch.setitem(sys.modules, "pygame",
                            type("pg", (), {"mixer": _Mixer, "AUDIO_S16": 0x8010}))
        monkeypatch.setitem(sys.modules, "pygame._sdl2",
                            type("s", (), {"audio": _Audio}))
        monkeypatch.setitem(sys.modules, "pygame._sdl2.audio", _Audio)
        monkeypatch.setattr(R.ClipRecorder, "_probe",
                            staticmethod(lambda name: peaks[name]))
        return R.ClipRecorder

    def test_the_loudest_device_is_chosen_not_the_first(self, monkeypatch):
        C = self._with_devices(
            monkeypatch, ["Headphones Stereo Microphone", "Digital Microphone"],
            {"Headphones Stereo Microphone": 5, "Digital Microphone": 10873})
        assert C.input_device() == "Digital Microphone"

    def test_a_silent_room_still_records_rather_than_refusing(self, monkeypatch):
        C = self._with_devices(monkeypatch, ["A", "B"], {"A": 2, "B": 1})
        assert C.input_device() == "A"

    def test_a_device_that_cannot_be_probed_is_skipped(self, monkeypatch):
        import app.protocol.response as R
        C = self._with_devices(monkeypatch, ["bad", "good"],
                               {"bad": 0, "good": 900})

        def _probe(name):
            if name == "bad":
                raise OSError("cannot open")
            return 900
        monkeypatch.setattr(R.ClipRecorder, "_probe", staticmethod(_probe))
        assert C.input_device() == "good"

    def test_no_devices_reports_none(self, monkeypatch):
        C = self._with_devices(monkeypatch, [], {})
        assert C.input_device() is None
