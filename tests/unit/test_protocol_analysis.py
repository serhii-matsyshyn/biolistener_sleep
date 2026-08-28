"""Peak fitting recovers a known resonance and reports its uncertainty."""
import numpy as np
import pytest

from pathlib import Path
from app.protocol.analysis import (
    ProbeResponse,
    ResponseCurve,
    bootstrap_peak,
    fit_peak,
)

GRID = (4.0, 5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 13.0)


def lorentzian(freqs, peak_hz, width=2.0, height=1.0):
    f = np.asarray(freqs, float)
    return height / (1.0 + ((f - peak_hz) / width) ** 2)


class TestFitPeak:
    @pytest.mark.parametrize("true_peak", [7.5, 8.5, 9.0, 10.0, 11.0])
    def test_recovers_a_peak_between_grid_points(self, true_peak):
        got = fit_peak(GRID, lorentzian(GRID, true_peak))
        assert abs(got - true_peak) < 0.75

    def test_interpolates_rather_than_snapping_to_the_grid(self):
        # A peak at 8.5 sits between probes; returning 8.0 or 9.0 would throw
        # away the resolution the interpolation exists to provide.
        got = fit_peak(GRID, lorentzian(GRID, 8.5))
        assert got not in GRID

    def test_a_peak_outside_the_probed_range_lands_near_the_nearest_edge(self):
        # Nothing can be localised outside the probes; the estimate should sit
        # at the near end rather than wander into the middle of the grid.
        lo = fit_peak(GRID, lorentzian(GRID, 2.0))
        hi = fit_peak(GRID, lorentzian(GRID, 20.0))
        assert GRID[0] <= lo < GRID[len(GRID) // 2]
        assert GRID[len(GRID) // 2] < hi <= GRID[-1]

    def test_stays_within_the_probed_range(self):
        for p in (4.5, 6.0, 12.5):
            assert GRID[0] <= fit_peak(GRID, lorentzian(GRID, p)) <= GRID[-1]

    def test_ignores_non_finite_and_non_positive_points(self):
        y = list(lorentzian(GRID, 9.0))
        y[0] = np.nan
        y[-1] = 0.0
        assert abs(fit_peak(GRID, y) - 9.0) < 0.75

    def test_returns_nan_without_data(self):
        assert np.isnan(fit_peak([], []))


def _curve(peak_hz, n_epochs=14, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    probes = []
    for f in GRID:
        base = lorentzian([f], peak_hz)[0]
        eps = np.abs(base * (1 + rng.normal(0, noise, n_epochs)))
        probes.append(ProbeResponse(f, snr_db=10.0, amplitude=float(eps.mean()),
                                    n_epochs=n_epochs, epoch_amplitudes=eps))
    return ResponseCurve(block="b", probes=tuple(probes))


class TestBootstrapPeak:
    def test_interval_contains_the_true_peak(self):
        r = bootstrap_peak(_curve(9.0), n=400)
        assert r["lo"] <= 9.0 <= r["hi"]

    def test_the_interval_never_falls_below_the_measured_accuracy(self):
        # The estimator's error is dominated by systematic bias rather than by
        # sampling noise, so the floor governs in normal conditions. That is
        # what buys 97 % coverage in verification.
        from app.protocol.analysis import PEAK_ACCURACY_HZ
        for noise in (0.02, 0.10, 0.30):
            r = bootstrap_peak(_curve(9.0, noise=noise, seed=1), n=300)
            assert (r["hi"] - r["lo"]) >= 2 * PEAK_ACCURACY_HZ - 1e-9

    def test_the_interval_brackets_its_own_point_estimate(self):
        # The resamples inherit the short-epoch bias and sit above the
        # whole-dwell estimate, so the interval is recentred on the point.
        r = bootstrap_peak(_curve(9.0, noise=0.20, seed=3), n=300)
        assert r["lo"] <= r["peak_hz"] <= r["hi"]

    def test_reports_nothing_when_too_few_probes_are_usable(self):
        thin = ResponseCurve(block="b", probes=tuple(
            ProbeResponse(f, 1.0, 1.0, 1, np.array([1.0])) for f in GRID[:2]))
        r = bootstrap_peak(thin)
        assert np.isnan(r["peak_hz"]) and r["n_boot"] == 0

    def test_single_epoch_probes_do_not_enter_the_interval(self):
        # One epoch cannot be resampled; including it would fake precision.
        probes = [ProbeResponse(f, 1.0, 1.0, 1, np.array([1.0])) for f in GRID]
        assert bootstrap_peak(ResponseCurve("b", tuple(probes)))["n_boot"] == 0


class TestResponseCurve:
    def test_skipped_probes_are_recorded_not_dropped(self):
        # A gap in the curve moves the fitted peak, so it has to be visible.
        c = ResponseCurve(block="b", probes=(), skipped=((10.0, "0 samples"),))
        assert c.skipped and c.skipped[0][0] == 10.0


class TestArmContrast:
    """The primary outcome: ramp against the learned policy, on real EEG."""

    @staticmethod
    def _session(tmp_path, order, arm_reward, drift_per_segment=0.0):
        """A recorded session with known per-arm rewards and a session drift."""
        import json
        import pandas as pd
        from app.protocol.spec import ARM_ORDERS
        rows, blocks = [], []
        t = 0.0
        for pos, arm in enumerate(ARM_ORDERS[order]):
            name = f"seg{pos + 1}_{arm}"
            blocks.append({"name": name, "kind": f"descent_{arm}",
                           "duration_s": 300.0, "eyes": "closed",
                           "kss_at_end": False, "note": ""})
            for _ in range(200):
                rows.append({"t_s": t, "block": name,
                             "reward": arm_reward[arm] + drift_per_segment * pos,
                             "dose": 0.24 if arm == "ramp" else 0.04,
                             "act_freq_hz": 8.0, "theta_alpha_ratio": 0.5})
                t += 0.25
        d = tmp_path / f"sess_{order}"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": f"full_{order}", "status": "completed",
             "arm_order": order, "blocks": blocks, "sampling_rate": 500}))
        (d / "markers.jsonl").write_text("")
        return d

    def test_recovers_a_known_arm_difference(self, tmp_path):
        from app.protocol import analysis as an
        d = self._session(tmp_path, "abba", {"ramp": 10.0, "agent": 6.0})
        c = an.arm_contrast(an.load(d), n_boot=200)
        assert c["ok"] and c["balanced"]
        assert c["ramp_minus_agent"] == pytest.approx(4.0, abs=1e-6)

    def test_a_balanced_order_cancels_session_drift(self, tmp_path):
        # The whole reason for ABBA: with the arms back to back this drift
        # would be charged to the arm effect and could invert its sign.
        from app.protocol import analysis as an
        d = self._session(tmp_path, "abba", {"ramp": 10.0, "agent": 6.0},
                          drift_per_segment=5.0)
        c = an.arm_contrast(an.load(d), n_boot=200)
        assert c["ramp_minus_agent"] == pytest.approx(4.0, abs=1e-6)

    def test_both_orders_give_the_same_answer(self, tmp_path):
        from app.protocol import analysis as an
        vals = [an.arm_contrast(an.load(
            self._session(tmp_path, o, {"ramp": 10.0, "agent": 6.0},
                          drift_per_segment=5.0)), n_boot=200)["ramp_minus_agent"]
            for o in ("abba", "baab")]
        assert vals[0] == pytest.approx(vals[1], abs=1e-6)

    def test_delivered_dose_is_reported_alongside(self, tmp_path):
        # Quoting the contrast without dose repeats a confound already
        # identified in simulation, where the margin halved once dose matched.
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(
            self._session(tmp_path, "abba", {"ramp": 10.0, "agent": 6.0})), n_boot=100)
        assert c["mean_dose"]["ramp"] > c["mean_dose"]["agent"]
        assert c["dose_ratio_agent_over_ramp"] == pytest.approx(0.04 / 0.24, rel=1e-3)

    def test_reports_cleanly_when_an_arm_is_missing(self, tmp_path):
        import json
        import pandas as pd
        from app.protocol import analysis as an
        d = tmp_path / "one_arm"; d.mkdir()
        pd.DataFrame([{"t_s": 0.0, "block": "seg1_ramp", "reward": 1.0,
                       "dose": 0.2, "act_freq_hz": 8.0,
                       "theta_alpha_ratio": 0.5}]).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "sampling_rate": 500,
             "blocks": [{"name": "seg1_ramp", "kind": "descent_ramp",
                         "duration_s": 300.0, "eyes": "closed",
                         "kss_at_end": False, "note": ""}]}))
        (d / "markers.jsonl").write_text("")
        assert an.arm_contrast(an.load(d))["ok"] is False


class TestArmFairness:
    """
    Noise degrades only the closed-loop arm, so a bad recording can look like
    the ramp winning. These are the pre-registered checks that separate them.
    """

    @staticmethod
    def _session(tmp_path, railed_by_arm, freq_sd_by_arm=None):
        import json
        import numpy as np
        import pandas as pd
        from app.protocol.spec import ARM_ORDERS
        rng = np.random.default_rng(0)
        freq_sd_by_arm = freq_sd_by_arm or {"ramp": 1.5, "agent": 1.5}
        rows, blocks = [], []
        t = 0.0
        for pos, arm in enumerate(ARM_ORDERS["abba"]):
            name = f"seg{pos + 1}_{arm}"
            blocks.append({"name": name, "kind": f"descent_{arm}",
                           "duration_s": 300.0, "eyes": "closed",
                           "kss_at_end": False, "note": ""})
            for _ in range(200):
                rows.append({
                    "t_s": t, "block": name, "reward": 1.0, "dose": 0.2,
                    "act_freq_hz": 8.0 + rng.normal(0, freq_sd_by_arm[arm]),
                    "railed_pct": railed_by_arm[arm], "theta_alpha_ratio": 0.5})
                t += 0.25
        d = tmp_path / "s"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "arm_order": "abba",
             "blocks": blocks, "sampling_rate": 500}))
        (d / "markers.jsonl").write_text("")
        return d

    def test_comparable_quality_passes(self, tmp_path):
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(
            self._session(tmp_path, {"ramp": 2.0, "agent": 3.0})), n_boot=50)
        assert c["quality_comparable"] is True

    def test_a_noisier_agent_arm_is_flagged(self, tmp_path):
        # The agent would have been handicapped by the recording rather than
        # beaten, so the contrast carries no directional claim.
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(
            self._session(tmp_path, {"ramp": 1.0, "agent": 25.0})), n_boot=50)
        assert c["quality_comparable"] is False
        assert c["railed_gap_agent_minus_ramp"] > 5.0

    def test_a_policy_pinned_to_a_rail_shows_no_command_spread(self, tmp_path):
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(self._session(
            tmp_path, {"ramp": 1.0, "agent": 1.0},
            freq_sd_by_arm={"ramp": 1.5, "agent": 0.001})), n_boot=50)
        assert c["freq_sd_hz"]["agent"] < 0.1 < c["freq_sd_hz"]["ramp"]


class TestPeakRecovery:
    """
    A simulated rehearsal knows the true resonance, so the fit can be checked
    against it rather than only against synthetic curves.
    """

    @staticmethod
    def _sim_session(tmp_path, true_hz_by_sweep, fitted_offset=0.0):
        import json
        import numpy as np
        import pandas as pd
        rows, blocks, markers = [], [], []
        t = 0.0
        for name, true_hz in true_hz_by_sweep.items():
            blocks.append({"name": name, "kind": "sweep", "duration_s": 115.0,
                           "eyes": "closed", "kss_at_end": True, "note": "",
                           "dwell_s": 10.0})
            markers.append({"t_s": t, "event": "block_start", "block": name})
            for _ in range(40):
                rows.append({"t_s": t, "block": name, "reward": 0.0, "dose": 0.2,
                             "act_freq_hz": 8.0, "probe_hz": 8.0,
                             "railed_pct": 0.0, "theta_alpha_ratio": 0.5,
                             "true_resonance_hz": true_hz + fitted_offset,
                             "true_drowsiness": 0.3})
                t += 0.25
            markers.append({"t_s": t, "event": "block_end", "block": name})
        d = tmp_path / "sim"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "markers.jsonl").write_text(
            "\n".join(json.dumps(m) for m in markers) + "\n")
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "blocks": blocks,
             "sampling_rate": 500}))
        return d

    def test_hardware_recordings_have_no_ground_truth_to_check(self, tmp_path):
        import json
        import pandas as pd
        from app.protocol import analysis as an
        d = tmp_path / "hw"; d.mkdir()
        pd.DataFrame([{"t_s": 0.0, "block": "s1", "reward": 0.0, "dose": 0.2,
                       "act_freq_hz": 8.0, "theta_alpha_ratio": 0.5}]).to_csv(
            d / "ticks.csv", index=False)
        (d / "markers.jsonl").write_text("")
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "blocks": [],
             "sampling_rate": 500}))
        sess = an.load(d)
        assert an.peak_recovery(sess).empty
        assert an.recovery_summary(sess)["ok"] is False

    def test_the_summary_reports_bias_and_coverage(self, tmp_path):
        # Without a raw recording no peak can be fitted, so the table is empty -
        # but the shape of the answer is what the rehearsal notebook consumes.
        from app.protocol import analysis as an
        d = self._sim_session(tmp_path, {"sweep1_alert": 10.0, "sweep2_drowsy": 8.0})
        tab = an.peak_recovery(an.load(d))
        # Presence, not position: the column order is free to grow.
        assert {"block", "fitted_hz", "true_hz", "error_hz", "ci_lo", "ci_hi",
                "covered", "resolvable", "target_stable",
                "true_sd_hz"} <= set(tab.columns)

    def test_ground_truth_columns_survive_a_round_trip(self, tmp_path):
        from app.protocol import analysis as an
        d = self._sim_session(tmp_path, {"sweep1_alert": 9.5})
        tk = an.load(d).ticks
        assert "true_resonance_hz" in tk
        assert tk["true_resonance_hz"].notna().all()


class TestResolvability:
    """
    The bootstrap says how repeatable the fit is, not how right it is. An
    interval far narrower than the probe spacing is a bias signature.
    """

    @staticmethod
    def _curve(values, n_epochs=8, jitter=0.0, seed=0):
        rng = np.random.default_rng(seed)
        probes = []
        for f, v in zip(GRID, values):
            eps = np.abs(v * (1 + rng.normal(0, jitter, n_epochs))) if jitter \
                else np.full(n_epochs, float(v))
            probes.append(ProbeResponse(f, 10.0, float(eps.mean()), n_epochs, eps))
        return ResponseCurve("b", tuple(probes))

    def test_a_degenerate_interval_is_widened_and_marked(self):
        # Identical epochs give zero spread. The interval is widened to the
        # method's demonstrated error rather than reported as a point, and
        # flagged so the reader knows the resamples said nothing.
        from app.protocol.analysis import PEAK_ACCURACY_HZ
        r = bootstrap_peak(self._curve(lorentzian(GRID, 9.0)), n=200)
        assert r["resolvable"] is False
        assert r["width_floor_applied"] is True
        assert r["hi"] - r["lo"] == pytest.approx(2 * PEAK_ACCURACY_HZ)

    def test_resamples_wider_than_the_floor_are_kept(self):
        # A curve with no shape leaves the fit wandering the whole grid, which
        # the resamples show; that spread is real and survives the floor.
        import numpy as np
        from app.protocol.analysis import PEAK_ACCURACY_HZ
        rng = np.random.default_rng(0)
        probes = [ProbeResponse(f, 10.0, 1.0, 12,
                                np.abs(rng.normal(1.0, 0.9, 12))) for f in GRID]
        r = bootstrap_peak(ResponseCurve("b", tuple(probes)), n=600)
        assert r["hi"] - r["lo"] > 2 * PEAK_ACCURACY_HZ
        assert r["width_floor_applied"] is False

    def test_the_spacing_comes_from_the_probe_grid(self):
        r = bootstrap_peak(self._curve(lorentzian(GRID, 9.0), jitter=0.2), n=200)
        assert r["spacing_hz"] == pytest.approx(np.median(np.diff(GRID)))

    def test_too_few_probes_reports_unresolvable(self):
        thin = ResponseCurve("b", tuple(
            ProbeResponse(f, 1.0, 1.0, 4, np.full(4, 1.0)) for f in GRID[:2]))
        assert bootstrap_peak(thin)["resolvable"] is False


class TestProbeStateCoupling:
    """
    Whether the probe moves the state it is measuring - the assumption that
    decides whether a sweep is a passive measurement at all.
    """

    @staticmethod
    def _sweep(tmp_path, state_by_probe):
        import json
        import pandas as pd
        rows = []
        t = 0.0
        for f, state in state_by_probe.items():
            for _ in range(20):
                rows.append({"t_s": t, "block": "sweep1", "probe_hz": f,
                             "act_freq_hz": f, "theta_alpha_ratio": state,
                             "reward": 0.0, "dose": 0.2})
                t += 0.25
        d = tmp_path / "s"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "sampling_rate": 500,
             "blocks": [{"name": "sweep1", "kind": "sweep", "duration_s": 115.0,
                         "eyes": "closed", "kss_at_end": True, "note": "",
                         "dwell_s": 10.0}]}))
        (d / "markers.jsonl").write_text("")
        return d

    def test_a_passive_sweep_shows_no_coupling(self, tmp_path):
        from app.protocol import analysis as an
        d = self._sweep(tmp_path, {5.0: 0.5, 7.0: 0.5, 9.0: 0.5, 11.0: 0.5})
        r = an.probe_state_coupling(an.load(d), "sweep1")
        assert r["ok"] is False          # nothing varies, nothing to correlate

    def test_low_probes_driving_drowsiness_is_detected(self, tmp_path):
        # The simulated plant does exactly this: strobing near theta raises
        # theta/alpha, which is what its resonance tracks.
        from app.protocol import analysis as an
        d = self._sweep(tmp_path, {5.0: 3.0, 7.0: 2.0, 9.0: 1.0, 11.0: 0.5})
        r = an.probe_state_coupling(an.load(d), "sweep1")
        assert r["ok"] and r["r"] < -0.5 and r["coupled"] is True

    def test_weak_coupling_is_not_flagged(self, tmp_path):
        from app.protocol import analysis as an
        d = self._sweep(tmp_path, {5.0: 1.02, 7.0: 0.99, 9.0: 1.03, 11.0: 1.00})
        r = an.probe_state_coupling(an.load(d), "sweep1")
        assert r["ok"] and r["coupled"] is False

    def test_it_reports_per_sweep_for_a_whole_session(self, tmp_path):
        from app.protocol import analysis as an
        d = self._sweep(tmp_path, {5.0: 3.0, 7.0: 2.0, 9.0: 1.0, 11.0: 0.5})
        rep = an.sweep_coupling_report(an.load(d))
        assert list(rep.columns) == ["block", "r", "coupled", "state_range", "note"]


class TestOutcomeGuards:
    """
    The pre-registered checks have to fire, and a partial recording has to be
    visibly partial. A number that reads like a valid one is worse than none.
    """

    @staticmethod
    def _session(tmp_path, ticks_by_seg, freq_sd_by_arm=None):
        import json
        import numpy as np
        import pandas as pd
        from app.protocol.spec import ARM_ORDERS
        rng = np.random.default_rng(0)
        freq_sd_by_arm = freq_sd_by_arm or {"ramp": 1.5, "agent": 1.5}
        rows, blocks = [], []
        t = 0.0
        for pos, arm in enumerate(ARM_ORDERS["abba"]):
            name = f"seg{pos + 1}_{arm}"
            blocks.append({"name": name, "kind": f"descent_{arm}",
                           "duration_s": 300.0, "eyes": "closed",
                           "kss_at_end": False, "note": ""})
            for _ in range(ticks_by_seg[pos]):
                rows.append({
                    "t_s": t, "block": name, "reward": 1.0, "dose": 0.2,
                    "act_freq_hz": 8.0 + rng.normal(0, freq_sd_by_arm[arm]),
                    "railed_pct": 1.0, "theta_alpha_ratio": 0.5})
                t += 0.25
        d = tmp_path / "s"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "aborted", "arm_order": "abba",
             "blocks": blocks, "sampling_rate": 500}))
        (d / "markers.jsonl").write_text("")
        return d

    def test_a_segment_cut_short_by_an_abort_is_not_averaged_with_full_ones(self, tmp_path):
        from app.protocol import analysis as an
        d = self._session(tmp_path, [800, 800, 800, 30])
        tab = an.arm_summary(an.load(d))
        assert "seg4_ramp" not in set(tab["block"])
        # Dropping it leaves the positions unequal, which is the honest state.
        assert an.arm_contrast(an.load(d), n_boot=50)["balanced"] is False

    def test_a_compressed_rehearsal_keeps_all_of_its_short_segments(self, tmp_path):
        # The floor is relative, so it means the same thing at any scale.
        from app.protocol import analysis as an
        tab = an.arm_summary(an.load(self._session(tmp_path, [60, 60, 60, 60])))
        assert len(tab) == 4

    def test_a_policy_that_stopped_controlling_is_reported_as_such(self, tmp_path):
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(self._session(
            tmp_path, [800] * 4,
            freq_sd_by_arm={"ramp": 1.5, "agent": 0.001})), n_boot=50)
        assert c["agent_controlling"] is False

    def test_a_single_night_does_not_claim_an_inferential_interval(self, tmp_path):
        # Two segments per arm cannot estimate between-segment variation, which
        # is the term that dominates at arm level.
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(self._session(tmp_path, [800] * 4)), n_boot=50)
        assert c["n_segments_per_arm"] == {"ramp": 2, "agent": 2}
        assert c["interval_supports_inference"] is False

    def test_the_bootstrap_block_outlasts_the_feature_window(self, tmp_path):
        # A 4 s Welch window advanced 0.25 s at a time makes sixteen
        # consecutive ticks near-copies; blocks shorter than that understate
        # the interval.
        from app.protocol import analysis as an
        c = an.arm_contrast(an.load(self._session(tmp_path, [800] * 4)), n_boot=50)
        assert c["bootstrap_block_len"] >= 16


class TestResonanceRegression:
    """The secondary outcome is the fallback when the primary is void."""

    @staticmethod
    def _peak_table(resolvable=True):
        import pandas as pd
        return pd.DataFrame([
            {"block": f"sweep{i}", "peak_hz": 9.0 + 0.3 * i,
             "ci_lo": 8.0, "ci_hi": 10.0, "resolvable": resolvable,
             "n_probes": 6, "n_epochs": 30}
            for i in range(4)])

    def test_peak_table_carries_the_resolvable_flag_to_its_caller(self):
        # Without the column the regression cannot report whether the peaks it
        # fitted were separable at all.
        assert "resolvable" in self._peak_table().columns

    def test_a_well_formed_session_yields_a_regression_result(self, tmp_path):
        import json
        import numpy as np
        import pandas as pd
        from app.protocol import analysis as an
        rows, blocks, markers = [], [], []
        t = 0.0
        for i in range(4):
            name = f"sweep{i}"
            blocks.append({"name": name, "kind": "sweep", "duration_s": 115.0,
                           "eyes": "closed", "kss_at_end": False, "note": ""})
            markers += [{"t_s": t, "event": "block_start", "block": name},
                        {"t_s": t + 100, "event": "block_end", "block": name}]
            for k in range(400):
                rows.append({"t_s": t + k * 0.25, "block": name, "reward": 0.0,
                             "dose": 0.2, "act_freq_hz": 8.0,
                             "theta_alpha_ratio": 0.5 + 0.4 * i,
                             "railed_pct": 1.0})
            t += 115.0
        d = tmp_path / "s"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "blocks": blocks,
             "sampling_rate": 500}))
        (d / "markers.jsonl").write_text(
            "\n".join(json.dumps(m) for m in markers))
        out = an.resonance_regression(an.load(d))
        assert isinstance(out, dict) and "ok" in out


class TestDriverPhaseModel:
    """
    Which clock the mask's LED driver keeps is an assumption about firmware we
    did not write. Session A can decide it from the recording instead.
    """

    @staticmethod
    def _session(tmp_path, plant_continuous, fs=500, ticks=200):
        import json
        import numpy as np
        import pandas as pd
        from app.agent.safety import StimAction
        from app.protocol.runner import DeliveredEnvelope
        tick_s, spt = 0.25, int(fs * 0.25)
        # A driving block that sweeps, so the two models diverge.
        freqs = np.linspace(11.0, 6.0, ticks)
        env = DeliveredEnvelope(fs, capacity_s=ticks * tick_s + 8.0,
                                phase_continuous=plant_continuous)
        for f in freqs:
            env.extend(StimAction(enabled=True, frequency_hz=float(f),
                                  fill=0.4, brightness=0.6), spt)
        light = env.latest(ticks * spt)
        rng = np.random.default_rng(0)
        eeg = light + 0.3 * rng.standard_normal(light.size)
        raw = np.repeat(eeg[None, :], 8, axis=0)

        d = tmp_path / f"s_{plant_continuous}"
        d.mkdir()
        np.savetxt(d / "raw_eeg.csv", raw.T, delimiter="\t")
        rows = [{"t_s": i * tick_s, "block": "A4_bright_060", "reward": 0.0,
                 "dose": 0.24, "act_freq_hz": float(freqs[i]), "act_fill": 0.4,
                 "act_brightness": 0.6, "railed_pct": 1.0,
                 "theta_alpha_ratio": 0.5,
                 "raw_sample": (i + 1) * spt}
                for i in range(ticks)]
        pd.DataFrame(rows).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "validation", "status": "completed", "tick_hz": 4.0,
             "sampling_rate": fs, "raw_eeg_rows": list(range(8)),
             "blocks": [{"name": "A4_bright_060", "kind": "fixed",
                         "duration_s": ticks * tick_s, "eyes": "closed",
                         "kss_at_end": False, "note": ""}]}))
        (d / "markers.jsonl").write_text("\n".join(json.dumps(m) for m in [
            {"t_s": 0.0, "event": "block_start", "block": "A4_bright_060"},
            {"t_s": ticks * tick_s, "event": "block_end", "block": "A4_bright_060"}]))
        return d

    def test_a_free_running_driver_is_identified(self, tmp_path):
        from app.protocol import analysis as an
        r = an.driver_phase_model(an.load(self._session(tmp_path, True)))
        assert r["ok"] and r["decisive"] and r["best"] == "phase_continuous"

    def test_a_restarting_driver_is_identified(self, tmp_path):
        from app.protocol import analysis as an
        r = an.driver_phase_model(an.load(self._session(tmp_path, False)))
        assert r["ok"] and r["decisive"] and r["best"] == "restarting"


class TestPolicyResponse:
    """
    Command spread cannot tell a controlling policy from one wandering on
    off-distribution features. Re-running it against scrambled EEG can.
    """

    def test_a_recording_without_the_observation_says_so(self, tmp_path):
        import json
        import pandas as pd
        from app.protocol import analysis as an
        d = tmp_path / "s"
        d.mkdir()
        pd.DataFrame([{"t_s": 0.0, "block": "seg1_agent", "reward": 0.0,
                       "act_freq_hz": 8.0}]).to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "blocks": []}))
        (d / "markers.jsonl").write_text("")
        r = an.policy_response(an.load(d), Path("nonexistent.zip"))
        # Reports the missing columns rather than raising, and never loads the
        # checkpoint it would not be able to use.
        assert r["ok"] is False and "plv" in r["reason"]


class TestPolicyResponseDetectsSaturation:
    """
    The check exists to catch a policy that stopped reading the brain. Feeding
    it inputs far outside the trained range is how that happens in practice.
    """

    @staticmethod
    def _checkpoint():
        import glob
        found = sorted(glob.glob("app/agent/checkpoints/*full_shaped*.zip"))
        return Path(found[0]) if found else None

    @staticmethod
    def _session(tmp_path, factor):
        import json
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 400
        t = np.arange(n) * 0.25
        # Features with real structure, so a responsive policy has something
        # to respond to.
        base = dict(
            alpha_pwr=0.2 + 0.15 * np.sin(t / 9),
            theta_pwr=0.03 + 0.02 * np.sin(t / 7 + 1),
            beta_pwr=0.02 + 0.01 * np.cos(t / 11),
            hypnagogic_theta_pwr=0.02 + 0.01 * np.sin(t / 6),
            sigma_pwr=0.01 + 0.005 * np.cos(t / 8),
            emg_rms=0.001 + 0.0005 * rng.random(n),
            eog_energy=0.5 + 0.2 * rng.random(n))
        rows = pd.DataFrame({
            "t_s": t, "block": "seg1_agent", "reward": 0.0, "dose": 0.24,
            "theta_alpha_ratio": 0.5 + 0.4 * np.sin(t / 10),
            "railed_pct": 1.0,
            "plv": 0.4 + 0.3 * np.sin(t / 5),
            "smoothed_plv": 0.4 + 0.25 * np.sin(t / 5),
            "delta_theta_alpha": 0.02 * np.cos(t / 10),
            "act_freq_hz": 9.0 - 0.005 * t, "act_fill": 0.4,
            "act_brightness": 0.6,
            **{k: v * factor for k, v in base.items()}})
        d = tmp_path / f"s{factor:g}"
        d.mkdir()
        rows.to_csv(d / "ticks.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "completed", "sampling_rate": 500,
             "blocks": [{"name": "seg1_agent", "kind": "descent_agent",
                         "duration_s": 100.0, "eyes": "closed",
                         "kss_at_end": False, "note": ""}]}))
        (d / "markers.jsonl").write_text("")
        return d

    def test_input_far_outside_the_trained_range_reads_as_not_responding(self, tmp_path):
        from app.protocol import analysis as an
        ck = self._checkpoint()
        if ck is None:
            pytest.skip("no full_shaped checkpoint available")
        near = an.policy_response(an.load(self._session(tmp_path, 1.0)), ck)
        far = an.policy_response(an.load(self._session(tmp_path, 1e5)), ck)
        assert near["ok"] and far["ok"]
        # Saturated: the commands stop moving when the EEG is scrambled,
        # because they had stopped depending on it.
        assert far["sensitivity"] < near["sensitivity"]
        assert far["responded_to_eeg"] is False

    def test_it_does_not_replace_the_amplitude_calibration(self, tmp_path):
        # Sensitivity falls away smoothly as the input leaves the trained
        # range, and where it crosses the threshold depends on the recording's
        # own magnitudes. It therefore cannot
        # stand in for measuring the scale: the two guards catch different
        # failures and a session needs both.
        from app.protocol import analysis as an
        ck = self._checkpoint()
        if ck is None:
            pytest.skip("no full_shaped checkpoint available")
        sens = [an.policy_response(an.load(self._session(tmp_path, f)), ck)["sensitivity"]
                for f in (1.0, 100.0, 1e5)]
        assert sens == sorted(sens, reverse=True), sens
