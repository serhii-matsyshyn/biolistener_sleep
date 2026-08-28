"""Quality gates separate real signals from noise, and say so with numbers."""
import numpy as np
import pytest

from app.protocol import quality as q

FS = 500


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def pink(n, rng, amp=8.0):
    """1/f background, the shape a resting EEG spectrum actually has."""
    x = rng.normal(0, 1, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    return np.fft.irfft(spec / np.sqrt(f), n)[:n] * amp


def sine(n, hz, amp):
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * hz * t)


def alpha_bump(n, rng, centre=10.3, width=1.4, amp=45.0):
    """
    A resting occipital spectrum's defining feature: a broad alpha peak.

    Band-limited noise rather than a sine, so it has the finite bandwidth and
    waxing envelope of a real rhythm. Pink noise alone does not exercise the
    estimator the way real EEG does - the peak is what a noise floor measured
    a few Hz from the probe lands on.
    """
    x = rng.normal(0, 1, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec *= np.exp(-0.5 * ((f - centre) / width) ** 2)
    return np.fft.irfft(spec, n)[:n] * amp


SECONDS = 60
N = SECONDS * FS


class TestAlphaTopography:
    """The placement check: alpha is larger occipitally than frontally."""

    def test_passes_when_alpha_is_occipital(self, rng):
        occ = pink(N, rng) + alpha_bump(N, rng)
        fro = [pink(N, rng) + alpha_bump(N, rng, amp=10.0) for _ in range(2)]
        g = q.gate_alpha_topography([occ], fro, FS)
        assert g.passed and g.value > q.ALPHA_TOPOGRAPHY_MIN_RATIO

    def test_fails_when_every_channel_sees_the_same_thing(self, rng):
        segs = [pink(N, rng) + alpha_bump(N, rng) for _ in range(3)]
        assert not q.gate_alpha_topography([segs[0]], segs[1:], FS).passed

    def test_fails_when_the_electrode_is_frontal(self, rng):
        weak = pink(N, rng) + alpha_bump(N, rng, amp=10.0)
        strong = [pink(N, rng) + alpha_bump(N, rng) for _ in range(2)]
        assert not q.gate_alpha_topography([weak], strong, FS).passed

    def test_the_weakest_segment_decides(self, rng):
        good = pink(N, rng) + alpha_bump(N, rng)
        bad = pink(N, rng) + alpha_bump(N, rng, amp=10.0)
        fro = [pink(N, rng) + alpha_bump(N, rng, amp=10.0)]
        assert not q.gate_alpha_topography([good, bad], fro, FS).passed

    def test_it_needs_a_frontal_channel(self, rng):
        with pytest.raises(ValueError):
            q.gate_alpha_topography([pink(N, rng)], [], FS)


class TestAlphaPresence:
    """The rhythm check: a peak standing above the aperiodic background."""

    def test_a_real_rhythm_clears_the_bar(self, rng):
        g = q.gate_alpha_present([pink(N, rng) + alpha_bump(N, rng)], FS)
        assert g.passed and g.value > q.ALPHA_PROMINENCE_MIN_DB

    @pytest.mark.parametrize("slope", [0.5, 1.0, 1.5])
    def test_a_bare_background_does_not(self, rng, slope):
        x = rng.normal(0, 1, N)
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(N, 1 / FS)
        f[0] = f[1]
        seg = np.fft.irfft(spec / f ** slope, N)[:N] * 20
        assert not q.gate_alpha_present([seg], FS).passed

    def test_a_steeper_background_is_not_mistaken_for_a_rhythm(self, rng):
        """Band power alone would rise with the slope; prominence must not."""
        x = rng.normal(0, 1, N)
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(N, 1 / FS)
        f[0] = f[1]
        steep = np.fft.irfft(spec / f ** 1.8, N)[:N] * 20
        assert q.alpha_prominence_db(steep, FS) < q.ALPHA_PROMINENCE_MIN_DB

    def test_the_weakest_segment_decides(self, rng):
        good = pink(N, rng) + alpha_bump(N, rng)
        flat = pink(N, rng)
        assert not q.gate_alpha_present([good, flat], FS).passed


class TestAlphaBlocking:
    def test_passes_when_alpha_rises_on_eye_closure(self, rng):
        g = q.gate_alpha_blocking([pink(N, rng)],
                                  [pink(N, rng) + sine(N, 10.2, 6.0)], FS)
        assert g.passed and g.value > q.ALPHA_BLOCK_MIN_RATIO

    def test_fails_when_both_conditions_are_noise(self, rng):
        assert not q.gate_alpha_blocking([pink(N, rng)], [pink(N, rng)], FS).passed

    def test_the_weakest_repeat_decides(self, rng):
        # One strong repeat must not carry a second that shows nothing.
        g = q.gate_alpha_blocking(
            [pink(N, rng), pink(N, rng)],
            [pink(N, rng) + sine(N, 10.0, 8.0), pink(N, rng)], FS)
        assert not g.passed

    def test_needs_both_conditions(self, rng):
        with pytest.raises(ValueError):
            q.gate_alpha_blocking([], [pink(N, rng)], FS)


class TestPhoticDriving:
    def test_detects_a_driven_response(self, rng):
        assert q.gate_ssvep(pink(N, rng) + sine(N, 10.0, 5.0), FS, 10.0).passed

    def test_rejects_undriven_noise(self, rng):
        assert not q.gate_ssvep(pink(N, rng), FS, 10.0).passed

    def test_does_not_credit_a_response_at_another_frequency(self, rng):
        assert not q.gate_ssvep(pink(N, rng) + sine(N, 6.0, 5.0), FS, 10.0).passed


class TestArtifactControl:
    """
    The zero-brightness rung is compared against blocks with the strobe off
    entirely, so what it reports is coupling from the drive electronics rather
    than whatever the subject's own spectrum happens to hold at the probe.
    """

    def test_passes_when_zero_brightness_looks_like_no_strobe_at_all(self, rng):
        unlit = [pink(N, rng) for _ in range(4)]
        assert q.gate_artifact(pink(N, rng), unlit, FS, 6.0).passed

    def test_fails_when_a_peak_survives_with_the_light_off(self, rng):
        # The data line still clocks at zero brightness, so a peak here is the
        # amplifier picking it up and every spectral number inherits it.
        leaking = pink(N, rng) + sine(N, 6.0, 6.0)
        unlit = [pink(N, rng) for _ in range(4)]
        assert not q.gate_artifact(leaking, unlit, FS, 6.0).passed

    def test_an_endogenous_rhythm_at_the_probe_is_not_called_pickup(self, rng):
        """
        The subject's alpha is present with the strobe off as well as at zero
        brightness. Comparing the two rungs against each other reported that
        as instrumentation; comparing against unlit blocks does not.
        """
        unlit = [pink(N, rng) + alpha_bump(N, rng) for _ in range(4)]
        zero = pink(N, rng) + alpha_bump(N, rng)
        assert q.gate_artifact(zero, unlit, FS, 10.0).passed

    def test_it_needs_something_to_compare_against(self, rng):
        with pytest.raises(ValueError):
            q.gate_artifact(pink(N, rng), [], FS, 6.0)


class TestDoseResponse:
    """
    What separates a response from pickup is that it depends on commanded
    brightness at all - the data line clocks the same whatever the LED current.
    """

    def _ladder(self, rng, amps):
        return {b: pink(N, rng) + (sine(N, 6.0, a) if a else 0.0)
                for b, a in amps.items()}

    def test_passes_when_every_lit_rung_stands_clear_of_zero(self, rng):
        g = q.gate_dose_response(
            self._ladder(rng, {0.0: 0.0, 0.10: 6.0, 0.20: 8.0}), FS, 6.0)
        assert g.passed

    def test_passes_when_the_response_is_not_monotone(self, rng):
        # Measured on hardware the peak falls at the top rung. That is a fact
        # about the stimulus, not a fault in the rig, and it must not block.
        g = q.gate_dose_response(
            self._ladder(rng, {0.0: 0.0, 0.10: 9.0, 0.20: 6.0}), FS, 6.0)
        assert g.passed

    def test_fails_when_a_rung_does_not_rise_above_zero_brightness(self, rng):
        g = q.gate_dose_response(
            self._ladder(rng, {0.0: 0.0, 0.10: 8.0, 0.20: 0.0}), FS, 6.0)
        assert not g.passed

    def test_fails_when_nothing_depends_on_brightness(self, rng):
        # Pickup: the same peak at every rung, including zero.
        g = q.gate_dose_response(
            self._ladder(rng, {0.0: 7.0, 0.10: 7.0, 0.20: 7.0}), FS, 6.0)
        assert not g.passed

    def test_it_needs_a_zero_rung_to_compare_against(self, rng):
        with pytest.raises(ValueError, match="zero-brightness rung"):
            q.gate_dose_response(self._ladder(rng, {0.10: 6.0, 0.20: 8.0}), FS, 6.0)


class TestLineNoise:
    def test_flags_mains_interference(self, rng):
        assert not q.gate_line_noise(pink(N, rng) + sine(N, 50.0, 30.0), FS).passed

    def test_accepts_a_clean_recording(self, rng):
        assert q.gate_line_noise(pink(N, rng), FS).passed

    def test_measures_before_the_notch_filter_removes_it(self, rng):
        # The standard chain bandstops 48-52 Hz, so a filtered measurement
        # would report the notch rather than the interference.
        noisy = pink(N, rng) + sine(N, 50.0, 30.0)
        assert (q.band_power(q.psd_welch(noisy, FS, filtered=False), 48.5, 51.5)
                > q.band_power(q.psd_welch(noisy, FS, filtered=True), 48.5, 51.5))


class TestPeakMeasurement:
    def test_reports_a_value_when_resolution_is_coarser_than_the_window(self, rng):
        # A short segment cannot resolve +/-0.5 Hz; the nearest bin is used
        # rather than reporting nothing for a peak that is plainly there.
        short = pink(300, rng) + sine(300, 13.0, 20.0)
        assert np.isfinite(q.peak_amplitude(q.psd_welch(short, FS), 13.0))
        assert np.isfinite(q.peak_snr_db(q.psd_welch(short, FS), 13.0))

    def test_welch_averages_the_whole_segment(self, rng):
        # Truncating to one window would leave the estimate as noisy at 60 s
        # as at 5 s, which is what makes a long recording worth taking.
        spread = lambda n: np.std([                      # noqa: E731
            q.band_power(q.psd_welch(pink(n, rng), FS), *q.ALPHA_BAND)
            for _ in range(30)])
        assert spread(60 * FS) < spread(5 * FS)


def test_summarise_reports_blocking_failures_separately(rng):
    gates = [q.gate_railed(0.0), q.gate_ssvep(pink(N, rng), FS, 10.0)]
    s = q.summarise(gates)
    assert not s["passed"]
    assert "G4" in s["blocking_failures"]


class TestLiveMonitor:
    def test_ignores_a_brief_disturbance(self):
        m = q.QualityMonitor(sustain_ticks=5)
        assert all(m.update(v) is None for v in [0, 0, 99, 0, 0])

    def test_aborts_on_a_sustained_one(self):
        m = q.QualityMonitor(sustain_ticks=3)
        results = [m.update(99) for _ in range(3)]
        assert results[-1] is not None

    def test_remembers_the_worst_value_seen(self):
        m = q.QualityMonitor()
        for v in (1.0, 42.0, 3.0):
            m.update(v)
        assert m.worst_railed_pct == 42.0


class TestCognitiveAlpha:
    """Second alpha check, needed because the mask is never removed and the
    eyes-open blocks therefore happen in darkness."""

    def test_passes_when_load_suppresses_alpha(self, rng):
        rest = pink(N, rng) + sine(N, 10.2, 6.0)
        task = pink(N, rng)
        assert q.gate_cognitive_alpha([rest], [task], FS).passed

    def test_fails_when_alpha_is_unchanged(self, rng):
        seg = pink(N, rng) + sine(N, 10.2, 6.0)
        assert not q.gate_cognitive_alpha([seg], [seg.copy()], FS).passed

    def test_is_reported_but_does_not_block_on_its_own(self, rng):
        g = q.gate_cognitive_alpha([pink(N, rng)], [pink(N, rng)], FS)
        assert g.blocking is False

    def test_needs_both_conditions(self, rng):
        with pytest.raises(ValueError):
            q.gate_cognitive_alpha([], [pink(N, rng)], FS)


class TestAlphaGateRule:
    """
    G3 and G8, both required. They answer different halves of one question -
    the signal is a cortical rhythm, and it comes from where it is supposed
    to. Neither depends on a manipulation now, so neither needs the other to
    cover for it.
    """

    @staticmethod
    def _gate(name, passed, blocking=True):
        return q.GateResult(name, name, 2.0 if passed else 0.5, 1.5, passed,
                            blocking=blocking)

    def test_a_missing_rhythm_blocks_even_with_good_topography(self):
        s = q.summarise([self._gate("G3", True), self._gate("G8", False)])
        assert s["passed"] is False and "G8" in s["blocking_failures"]

    def test_wrong_placement_blocks_even_with_a_clear_rhythm(self):
        s = q.summarise([self._gate("G3", False), self._gate("G8", True)])
        assert s["passed"] is False and "G3" in s["blocking_failures"]

    def test_both_passing_clears_the_alpha_check(self):
        s = q.summarise([self._gate("G3", True), self._gate("G8", True)])
        assert s["passed"] is True and not s["blocking_failures"]

    def test_the_retired_measurements_are_reported_but_never_block(self):
        s = q.summarise([self._gate("G3", True), self._gate("G8", True),
                         self._gate("D1", False, blocking=False),
                         self._gate("D2", False, blocking=False)])
        assert s["passed"] is True
        assert set(s["warnings"]) >= {"D1", "D2"}

    def test_an_unrelated_blocking_gate_still_blocks(self):
        s = q.summarise([self._gate("G3", True),
                         self._gate("G8", True, blocking=False),
                         self._gate("G4", False)])
        assert s["passed"] is False and s["blocking_failures"] == ["G4"]



class TestEstimatorIsUnbiasedAgainstAlpha:
    """
    An occipital recording always contains alpha. A peak measure whose noise
    floor is read far enough from the probe to include it reports a large
    response where there is none, or hides one that is there, depending only
    on which side of the probe the alpha sits.
    """

    @pytest.mark.parametrize("probe", [4.0, 5.0, 6.0, 7.0, 13.0, 14.0])
    def test_no_response_never_reaches_the_threshold_beside_alpha(self, rng, probe):
        """
        Only an upward bias can manufacture a response. A reading below zero
        means the floor beside the probe is hotter than the probe itself,
        which costs sensitivity but cannot pass a gate on nothing.
        """
        seg = pink(N, rng) + alpha_bump(N, rng)
        snr = q.peak_snr_db(q.psd_welch(seg, FS), probe)
        assert snr < q.SSVEP_MIN_SNR_DB, (
            f"{probe} Hz reads {snr:.2f} dB with nothing driving it")

    def test_a_real_drive_is_still_found_beside_an_alpha_peak(self, rng):
        seg = pink(N, rng) + alpha_bump(N, rng) + sine(N, 6.0, 3.0)
        snr = q.peak_snr_db(q.psd_welch(seg, FS), 6.0)
        assert snr >= q.SSVEP_MIN_SNR_DB

    def test_the_gate_does_not_pass_on_alpha_alone(self, rng):
        seg = pink(N, rng) + alpha_bump(N, rng)
        assert not q.gate_ssvep(seg, FS, 10.0).passed

    def test_a_long_block_is_resolved_more_finely_than_a_short_one(self):
        assert q._nfft_for(60 * FS) > q._nfft_for(2 * FS)
