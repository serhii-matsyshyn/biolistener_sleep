"""Protocol definitions stay inside the safety envelope and fit their budgets."""
import pytest

from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ, PSE_BAND_HI, PSE_BAND_LO
from app.protocol.spec import (
    DESCENT_S,
    BlockKind,
    build_full,
    build_validation,
    compress,
    settling_report,
    sweep_min_duration_s,
    validate,
)


@pytest.fixture(params=["validation", "abba", "baab"])
def protocol(request):
    if request.param == "validation":
        return build_validation()
    return build_full(order=request.param)


def test_every_commanded_frequency_is_inside_the_safety_envelope(protocol):
    for b in protocol.blocks:
        for f in b.commanded_frequencies():
            assert FREQ_MIN_HZ <= f <= FREQ_MAX_HZ
            assert not (PSE_BAND_LO <= f <= PSE_BAND_HI)


def test_blocks_tile_the_session_without_gaps_or_overlap(protocol):
    bounds = protocol.boundaries()
    assert bounds[0][0] == 0.0
    for (_, end_a, _), (start_b, _, _) in zip(bounds, bounds[1:]):
        assert end_a == start_b
    assert bounds[-1][1] == pytest.approx(protocol.total_duration_s)


def test_block_at_resolves_times_to_the_right_block(protocol):
    for start, end, blk in protocol.boundaries():
        assert protocol.block_at(start) is blk
        assert protocol.block_at((start + end) / 2) is blk
    assert protocol.block_at(protocol.total_duration_s) is None


def test_full_session_fits_the_supervisor_timeout():
    from app.agent.safety import DEFAULT_SESSION_TIMEOUT_S
    from app.protocol.spec import ARM_ORDERS
    for order in ARM_ORDERS:
        assert build_full(order=order).total_duration_s <= DEFAULT_SESSION_TIMEOUT_S


def test_both_arms_get_equal_total_time():
    from app.protocol.spec import ARM_ORDERS, SEGMENT_S
    for order in ARM_ORDERS:
        segs = [b for b in build_full(order=order).blocks
                if b.kind in (BlockKind.DESCENT_RAMP, BlockKind.DESCENT_AGENT)]
        per_arm = {}
        for b in segs:
            per_arm.setdefault(b.kind, []).append(b.duration_s)
        assert len(per_arm) == 2
        assert all(sum(v) == DESCENT_S for v in per_arm.values())
        assert all(d == SEGMENT_S for d in (b.duration_s for b in segs))


class TestArmBalance:
    """The property that makes the arm contrast unbiased under session drift."""

    def test_position_sums_are_equal_for_both_arms(self):
        from app.protocol.spec import ARM_ORDERS, arm_positions
        for order in ARM_ORDERS:
            pos = arm_positions(build_full(order=order))
            assert len(pos) == 2
            assert len(set(sum(v) for v in pos.values())) == 1, pos

    def test_each_arm_appears_once_before_and_once_after_the_realert(self):
        # The re-alerting block is a step change, not a linear trend, so the
        # arms must straddle it too.
        for order in ("abba", "baab"):
            p = build_full(order=order)
            names = [b.name for b in p.blocks]
            r = names.index("B4_realert")
            for arm in ("ramp", "agent"):
                segs = [i for i, n in enumerate(names) if n.endswith(f"_{arm}")]
                assert sum(i < r for i in segs) == 1, (order, arm)
                assert sum(i > r for i in segs) == 1, (order, arm)

    def test_an_unbalanced_order_is_rejected(self):
        from app.protocol.spec import (Block, BlockKind as BK, Eyes, Protocol,
                                       check_arm_balance)
        seq = ("ramp", "ramp", "agent", "agent")     # AABB inverts the contrast
        bad = Protocol(name="aabb", seed=0, blocks=tuple(
            Block(f"seg{i}_{a}",
                  BK.DESCENT_RAMP if a == "ramp" else BK.DESCENT_AGENT,
                  300.0, Eyes.CLOSED, "descent_start")
            for i, a in enumerate(seq)))
        with pytest.raises(ValueError, match="not balanced"):
            check_arm_balance(bad)

    def test_alternating_abab_is_also_rejected(self):
        # ABAB looks balanced and is not: positions 0,2 against 1,3.
        from app.protocol.spec import (Block, BlockKind as BK, Eyes, Protocol,
                                       check_arm_balance)
        seq = ("ramp", "agent", "ramp", "agent")
        bad = Protocol(name="abab", seed=0, blocks=tuple(
            Block(f"seg{i}_{a}",
                  BK.DESCENT_RAMP if a == "ramp" else BK.DESCENT_AGENT,
                  300.0, Eyes.CLOSED, "descent_start")
            for i, a in enumerate(seq)))
        with pytest.raises(ValueError, match="not balanced"):
            check_arm_balance(bad)


class TestArmOrderSelection:
    def test_the_two_orders_are_mirror_images(self):
        from app.protocol.spec import ARM_ORDERS
        assert ARM_ORDERS["abba"] == tuple(
            "ramp" if a == "agent" else "agent" for a in ARM_ORDERS["baab"])

    def test_legacy_names_still_resolve(self):
        assert build_full(order="ramp_first").name == "full_abba"
        assert build_full(order="agent_first").name == "full_baab"

    def test_a_fresh_setup_starts_on_abba(self, tmp_path):
        from app.protocol.spec import suggest_order
        assert suggest_order(tmp_path) == "abba"

    def test_the_suggestion_alternates_as_sessions_accumulate(self, tmp_path):
        import json
        from app.protocol.spec import suggest_order
        def record(order):
            d = tmp_path / f"s{len(list(tmp_path.iterdir()))}_{order}"
            d.mkdir()
            (d / "manifest.json").write_text(json.dumps(
                {"protocol": f"full_{order}", "status": "completed"}))
        assert suggest_order(tmp_path) == "abba"
        record("abba")
        assert suggest_order(tmp_path) == "baab"
        record("baab")
        assert suggest_order(tmp_path) == "abba"

    def test_the_order_is_read_from_the_manifest_not_the_protocol_name(
            self, tmp_path):
        # A real session records arm_order explicitly. Matching on the name
        # alone is the fallback, and it has to keep working.
        import json
        from app.protocol.spec import suggest_order
        d = tmp_path / "run"; d.mkdir()
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "something_unmatchable", "status": "completed",
             "arm_order": "abba", "eeg_kind": "biolistener"}))
        assert suggest_order(tmp_path) == "baab"

    def test_a_rehearsal_does_not_advance_the_alternation(self, tmp_path):
        """
        Alternating the arm order exists to cancel drift across real nights.
        A compressed or simulated run delivers no subject data and balances
        nothing, so counting it would set the real session's order from how
        many times the machinery happened to be exercised.
        """
        import json
        from app.protocol.spec import arm_order_of, build_full, compress, suggest_order
        for i, extra in enumerate((
                {"protocol": "full_abba_x0.05", "eeg_kind": "biolistener",
                 "arm_order": arm_order_of(compress(build_full("abba"), 0.05))},
                {"protocol": "full_abba", "eeg_kind": "sim",
                 "arm_order": "abba"})):
            d = tmp_path / f"rehearsal{i}"; d.mkdir()
            (d / "manifest.json").write_text(
                json.dumps({"status": "completed", **extra}))
        assert suggest_order(tmp_path) == "abba"

    def test_order_is_derived_from_the_segments_not_the_name(self):
        from app.protocol.spec import arm_order_of, build_full, compress
        for order in ("abba", "baab"):
            p = build_full(order)
            assert arm_order_of(p) == order
            assert arm_order_of(compress(p, 0.05)) == order

    def test_an_aborted_session_does_not_count(self, tmp_path):
        import json
        from app.protocol.spec import suggest_order
        d = tmp_path / "aborted"; d.mkdir()
        (d / "manifest.json").write_text(json.dumps(
            {"protocol": "full_abba", "status": "aborted"}))
        assert suggest_order(tmp_path) == "abba"


def test_every_sweep_has_room_for_its_probes(protocol):
    for b in protocol.blocks:
        if b.kind == BlockKind.SWEEP:
            assert sweep_min_duration_s(b.probe_hz, b.dwell_s) <= b.duration_s


def test_one_rating_is_collected_per_sweep():
    p = build_full()
    sweeps = [b.name for b in p.blocks if b.kind == BlockKind.SWEEP]
    rated = set(p.kss_points)
    assert set(sweeps) <= rated


def test_real_protocols_leave_time_to_reach_each_fixed_frequency(protocol):
    assert settling_report(protocol) == ()


def _one_fixed_block(hz):
    from app.protocol.spec import Block, Eyes, Protocol
    return Protocol(name="bad", seed=0, blocks=(
        Block("x", BlockKind.FIXED, 60.0, Eyes.CLOSED, "strobe_start",
              fixed_hz=hz),))


def test_validate_rejects_a_frequency_in_the_pse_band():
    with pytest.raises(ValueError):
        validate(_one_fixed_block((PSE_BAND_LO + PSE_BAND_HI) / 2))


def test_pse_guard_still_fires_if_the_frequency_envelope_is_widened(monkeypatch):
    # The PSE band sits above FREQ_MAX_HZ, so today the envelope check catches
    # these first and the guard never binds. It is kept because it is what
    # holds if the action space is ever widened.
    import app.protocol.spec as spec
    monkeypatch.setattr(spec, "FREQ_MAX_HZ", 30.0)
    with pytest.raises(ValueError, match="PSE band"):
        spec.validate(_one_fixed_block((PSE_BAND_LO + PSE_BAND_HI) / 2))


def test_validate_rejects_a_sweep_that_cannot_hold_its_probes():
    from app.protocol.spec import Block, Eyes, Protocol
    bad = Protocol(name="bad", seed=0, blocks=(
        Block("s", BlockKind.SWEEP, 10.0, Eyes.CLOSED, "sweep_start",
              probe_hz=(4.0, 13.0), dwell_s=15.0),))
    with pytest.raises(ValueError, match="at least"):
        validate(bad)


def test_compression_keeps_sweeps_long_enough_to_walk_the_grid():
    # The rate limiter is hardware, not schedule: a sweep cannot shrink below
    # the time it physically takes to move between probes.
    small = compress(build_full(), 0.01)
    for b in small.blocks:
        if b.kind == BlockKind.SWEEP:
            assert b.duration_s >= sweep_min_duration_s(b.probe_hz, b.dwell_s)


def test_compression_shortens_the_session_overall():
    full = build_full()
    assert compress(full, 0.25).total_duration_s < full.total_duration_s


@pytest.mark.parametrize("factor", [0.0, -1.0, 1.5])
def test_compression_rejects_factors_outside_the_unit_interval(factor):
    with pytest.raises(ValueError):
        compress(build_validation(), factor)


def test_validation_session_pairs_every_eyes_open_block_with_a_closed_one():
    names = [b.name for b in build_validation().blocks]
    assert names.count("A2a_eyes_open") == 1
    assert sum(n.endswith("_eyes_open") for n in names) == \
        sum(n.endswith("_eyes_closed") for n in names)


def test_the_brightness_ladder_varies_only_brightness():
    # Frequency must be identical across rungs, or a difference in response
    # could be frequency rather than light.
    from app.protocol.spec import BRIGHTNESS_LADDER, SSVEP_CHECK_HZ
    rungs = [b for b in build_validation().blocks
             if b.name.startswith("A4_bright_")]
    assert len(rungs) == len(BRIGHTNESS_LADDER)
    assert {b.fixed_hz for b in rungs} == {SSVEP_CHECK_HZ}
    assert sorted(b.brightness for b in rungs) == sorted(BRIGHTNESS_LADDER)
    assert {b.duration_s for b in rungs} == {60.0}


def test_the_ladder_includes_a_zero_brightness_control():
    rungs = [b for b in build_validation().blocks if b.name.startswith("A4_bright_")]
    assert any(b.brightness == 0.0 for b in rungs)


def test_nothing_in_either_protocol_requires_touching_the_hardware():
    # Mask and electrodes go on once and stay. Every condition is reached by
    # changing what is commanded.
    for p in (build_validation(), build_full("abba")):
        for b in p.blocks:
            assert b.kind is not BlockKind.BENCH, b.name
            assert "insert" not in b.name and "blocker" not in b.cue_start


def test_session_a_exercises_a_moving_command_not_only_fixed_ones():
    # Session B spends twenty minutes on a command that moves. If every
    # Session A block holds one frequency, nothing validates that case before
    # the experiment depends on it.
    from app.protocol.spec import build_validation, BlockKind
    kinds = {b.kind for b in build_validation().blocks}
    assert BlockKind.DESCENT_RAMP in kinds or BlockKind.SWEEP in kinds


def test_session_a_is_never_mistaken_for_a_night_of_the_experiment():
    # It contains a descent block, and arm bookkeeping keys on block kind, so
    # this is the check that the two cannot be confused.
    from app.protocol.spec import build_validation, arm_order_of
    assert arm_order_of(build_validation()) is None


def test_each_arm_meets_the_same_entry_conditions():
    # A segment inherits the ramp limiter's state from whatever ran before it,
    # so a segment entered straight after the other arm starts rate-limited
    # while one entered after a sweep does not. If that were lopsided, one arm
    # would spend more time climbing back and the contrast could not see it.
    from app.protocol.spec import build_full, BlockKind
    descents = (BlockKind.DESCENT_RAMP, BlockKind.DESCENT_AGENT)
    for order in ("abba", "baab"):
        blocks = list(build_full(order).blocks)
        preceded = {}
        for i, b in enumerate(blocks):
            if b.kind in descents:
                arm = "ramp" if b.kind == BlockKind.DESCENT_RAMP else "agent"
                preceded.setdefault(arm, []).append(
                    "descent" if blocks[i - 1].kind in descents else "other")
        assert sorted(preceded["ramp"]) == sorted(preceded["agent"]), \
            f"{order}: {preceded}"


def test_the_probe_grid_reaches_both_ends_of_the_legal_band():
    # The descent aims at 4 Hz and a drowsy peak moves down, so the grid has
    # to reach the region the intervention targets; the alert anchor sweep
    # needs the top, where a subject's own alpha may sit.
    import numpy as np
    from app.protocol.spec import BRIEF_PROBE_HZ
    from app.agent.safety import FREQ_MIN_HZ, FREQ_MAX_HZ
    span = max(BRIEF_PROBE_HZ) - min(BRIEF_PROBE_HZ)
    legal = FREQ_MAX_HZ - FREQ_MIN_HZ
    assert min(BRIEF_PROBE_HZ) <= FREQ_MIN_HZ
    assert max(BRIEF_PROBE_HZ) <= FREQ_MAX_HZ
    assert span >= legal - np.median(np.diff(sorted(BRIEF_PROBE_HZ)))


def test_probes_are_no_finer_than_the_estimator_can_resolve():
    # Peak error has a measured p95 of 1.45 Hz; probes closer than that cost a
    # dwell each and add no information.
    import numpy as np
    from app.protocol.spec import BRIEF_PROBE_HZ
    from app.protocol.analysis import PEAK_ACCURACY_HZ
    gaps = np.diff(sorted(BRIEF_PROBE_HZ))
    assert gaps.min() >= PEAK_ACCURACY_HZ * 0.99, gaps


def test_every_sweep_ordering_fits_its_block():
    # plan_sweep randomises probe order, and the ramp limiter charges for the
    # transit between them, so the worst ordering has to fit - not the best.
    from app.protocol.spec import BRIEF_PROBE_HZ, BRIEF_DWELL_S, BRIEF_SWEEP_S
    from app.protocol.sweep import plan_sweep
    for seed in range(200):
        plan_sweep(BRIEF_PROBE_HZ, BRIEF_DWELL_S, BRIEF_SWEEP_S,
                   seed=seed, start_hz=11.0)


# ---- steady light in the eyes-open blocks ---------------------------------

def test_eyes_open_blocks_are_lit():
    from app.protocol.spec import EYES_OPEN_BRIGHTNESS
    p = build_validation()
    open_blocks = [b for b in p.blocks
                   if b.eyes.value == "open" and b.name.startswith("A2")]
    assert open_blocks
    for b in open_blocks:
        assert b.steady_brightness == EYES_OPEN_BRIGHTNESS
        assert b.lit


def test_eyes_closed_blocks_stay_dark():
    p = build_validation()
    for b in p.blocks:
        if b.eyes.value == "closed" and b.kind == BlockKind.REST:
            assert b.steady_brightness is None
            assert not b.lit


def test_steady_brightness_above_the_ceiling_is_rejected():
    from app.agent.safety import STEADY_BRIGHTNESS_MAX
    from app.protocol.spec import Block, Eyes, Protocol
    bad = Protocol(
        name="bad",
        blocks=(Block("x", BlockKind.REST, 10.0, Eyes.OPEN, "eyes_open",
                      steady_brightness=STEADY_BRIGHTNESS_MAX + 0.01),),
        seed=0,
    )
    with pytest.raises(ValueError, match="steady brightness"):
        validate(bad)


def test_the_check_frequency_avoids_the_alpha_band():
    from app.protocol.quality import ALPHA_BAND
    from app.protocol.spec import SSVEP_CHECK_HZ
    lo, hi = ALPHA_BAND
    assert not lo <= SSVEP_CHECK_HZ <= hi


def test_the_brightness_ladder_spans_dark_to_the_working_ceiling():
    from app.protocol.runner import RunnerConfig
    from app.protocol.spec import BRIGHTNESS_LADDER
    rungs = sorted(BRIGHTNESS_LADDER)
    ceiling = RunnerConfig(protocol=build_validation()).brightness_max
    # Dark at the bottom is the artifact control; the top rung is the level the
    # rest of the session runs at, so the ladder brackets what is delivered.
    assert rungs[0] == 0.0
    assert rungs[-1] == ceiling
    assert len(rungs) >= 3
    assert rungs == sorted(set(rungs))
