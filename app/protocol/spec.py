"""
Session protocols as inspectable data.

Two protocols, built by :func:`build_validation` and :func:`build_full`:

``validation``
    Short recording that checks the loop end to end - electrodes on cortex,
    light reaching the retina, and the strobe not leaking into the amplifier.
    Movement is allowed, but the mask stays on.

``full``
    The experimental session. Four frequency sweeps bracket two equal-length
    descent blocks, with a re-alerting block between them so a downward shift
    in the response peak can be separated from electrode drift.

    Motionless start to finish and the mask is never re-seated: sweep-to-sweep
    amplitude comparisons are only meaningful if the optical coupling holds
    constant.

Every commanded frequency here sits inside the supervisor's envelope
(``FREQ_MIN_HZ``..``FREQ_MAX_HZ``, PSE band excluded). :func:`validate` asserts
that rather than trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from app.agent.safety import (
    FREQ_MAX_HZ,
    FREQ_MIN_HZ,
    MAX_FREQ_STEP_HZ_PER_S,
    PSE_BAND_HI,
    PSE_BAND_LO,
    STEADY_BRIGHTNESS_MAX,
)

# Probe grid for every sweep. Spaced more densely through 7-11 Hz, where the
# response peak is expected to sit and move.
PROBE_HZ: Tuple[float, ...] = (4.0, 5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 13.0)

DWELL_S = 15.0          # settled time held at each probe
SWEEP_BLOCK_S = 180.0   # budget per sweep: 8 dwells + rate-limited transitions
# Arm segments. Four of them, ABBA-ordered, so the sum of segment positions is
# equal for both arms and any linear drift over the session cancels exactly.
# Running the arms back to back instead biases the contrast badly enough to
# invert its sign.
# The checkpoint a session runs unless told otherwise. Chosen on the held-out
# cohort by paired contrast against the scripted ramp, before any session; three
# archives carry the seed-2 label and only this one holds the full budget.
DEFAULT_CHECKPOINT = "app/agent/checkpoints/sac_full_shaped_seed2_1785433475.zip"

SEGMENT_S = 300.0
N_SEGMENTS = 4

# Both patterns are position-balanced. Alternating them across nights cancels
# what a single balanced ordering cannot: a drift that is not purely linear.
ARM_ORDERS = {
    "abba": ("ramp", "agent", "agent", "ramp"),
    "baab": ("agent", "ramp", "ramp", "agent"),
}
ARM_ORDER_ALIASES = {"ramp_first": "abba", "agent_first": "baab"}
DESCENT_S = SEGMENT_S * (N_SEGMENTS // 2)   # total per arm
RAMP_START_HZ = 11.0
RAMP_END_HZ = 4.0

# Sweeps inside the experimental session are secondary - they characterise the
# simulator's resonance assumption, not the agent.
#
# Uniform 1.5 Hz spacing across the whole legal band. Both ends matter and for
# different reasons: the descent drives towards 4 Hz and a drowsy peak moves
# *down*, so the grid has to reach the region the intervention aims at. The
# top stays near 13.5 Hz because the alert anchor sweep has to find a subject
# whose own alpha sits high.
#
# The spacing is the estimator's own resolution: peak error has a p95 of
# 1.45 Hz (analysis.PEAK_ACCURACY_HZ), so probes closer together than that
# spend a dwell each and buy nothing.
BRIEF_PROBE_HZ: Tuple[float, ...] = (3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5)
BRIEF_DWELL_S = 10.0
# Sized so every randomised probe order fits once the ramp limiter's transit
# between probes is counted.
BRIEF_SWEEP_S = 125.0

# Fixed-frequency checks in the validation protocol.
#
# Deliberately away from the alpha band. A check at 10 Hz cannot separate a
# driven response from the subject's own alpha, which sits at the same
# frequency, is present with the light off, and is the largest thing in an
# occipital recording - so the SNR gate reads as a pass on alpha alone and the
# zero-brightness contrast reads as a fail, both for the same reason. 6 Hz has
# no endogenous rhythm competing with it.
SSVEP_CHECK_HZ = 6.0
MULTI_CHECK_HZ: Tuple[float, ...] = (6.0, 10.0, 13.0)
# Brightness ladder at SSVEP_CHECK_HZ, driven entirely by command. A genuine
# photic response grows with brightness; pickup from the drive electronics does
# not have to, so the ladder discriminates them without touching the hardware.
#
# The rungs span the working range up to the ceiling the session runs at.
BRIGHTNESS_LADDER: Tuple[float, ...] = (0.0, 0.10, 0.20)

# Continuous light held during the eyes-open blocks, well under
# ``STEADY_BRIGHTNESS_MAX``. Without it those blocks are as dark as the
# eyes-closed ones - the mask does not stop being a blackout mask when the
# eyelids open - and alpha has nothing to desynchronise to, so the blocking
# check measures nothing. Unmodulated on purpose: a flicker here would drive
# the very band the check reads.
#
# Set on the device rather than reasoned about: 3 % is comfortable to hold the
# eyes open against, and enough to be plainly visible.
EYES_OPEN_BRIGHTNESS = 0.03

# Visible flash at the head of each settle block, long enough to be unmistakable
# through closed eyelids or open eyes.
MASK_CHECK_S = 2.0
MASK_CHECK_HZ = 10.0


class BlockKind(str, Enum):
    REST = "rest"                    # no stimulation
    FIXED = "fixed"                  # constant frequency
    SWEEP = "sweep"                  # probe grid
    DESCENT_RAMP = "descent_ramp"    # scripted linear descent
    DESCENT_AGENT = "descent_agent"  # trained policy
    BENCH = "bench"                  # mask exercised off the head


class Eyes(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    FREE = "free"                    # no subject, or unconstrained


@dataclass(frozen=True)
class Block:
    """One timed step. ``cue_start`` names a phrase in the cue catalogue."""

    name: str
    kind: BlockKind
    duration_s: float
    eyes: Eyes
    cue_start: str
    kss_at_end: bool = False
    note: str = ""

    # SWEEP
    probe_hz: Tuple[float, ...] = ()
    dwell_s: float = DWELL_S

    # Continuous light for the whole block, ``None`` for dark. Independent of
    # ``kind``: a rest block emits no strobe either way, and this only decides
    # whether the eyes have anything to see.
    steady_brightness: Optional[float] = None

    # FIXED
    fixed_hz: Optional[float] = None
    # Brightness override, ``None`` for the scripted default. The artifact
    # control commands zero brightness with the strobe still enabled: the
    # command stream, the frequency and the mask position are unchanged, and
    # only the light goes away. Nothing is touched by hand.
    brightness: Optional[float] = None

    # Seconds of visible flashing at the very start of the block, before it
    # does whatever else it does. The mask answering a status request proves
    # the protocol works, not that any light came out; this is the only check
    # that the subject can make, and it has to happen while the eyes are still
    # open and before half an hour of lying still begins.
    mask_check_s: float = 0.0

    # DESCENT_RAMP
    ramp_start_hz: float = RAMP_START_HZ
    ramp_end_hz: float = RAMP_END_HZ

    @property
    def stimulates(self) -> bool:
        return self.kind not in (BlockKind.REST,) or self.steady_brightness is not None

    @property
    def lit(self) -> bool:
        """Whether any light is commanded, flickering or not."""
        return self.steady_brightness is not None or (
            self.stimulates and self.brightness != 0.0)

    def commanded_frequencies(self) -> Tuple[float, ...]:
        """Every frequency this block can ask for - used by :func:`validate`."""
        if self.kind == BlockKind.SWEEP:
            return tuple(self.probe_hz)
        if self.kind == BlockKind.FIXED:
            return (self.fixed_hz,) if self.fixed_hz is not None else ()
        if self.kind == BlockKind.DESCENT_RAMP:
            return (self.ramp_start_hz, self.ramp_end_hz)
        if self.kind == BlockKind.BENCH:
            return (FREQ_MIN_HZ, FREQ_MAX_HZ)
        return ()


@dataclass(frozen=True)
class Protocol:
    name: str
    blocks: Tuple[Block, ...]
    seed: int
    montage: str = "occipital_alpha"
    description: str = ""

    @property
    def total_duration_s(self) -> float:
        return sum(b.duration_s for b in self.blocks)

    @property
    def kss_points(self) -> Tuple[str, ...]:
        return tuple(b.name for b in self.blocks if b.kss_at_end)

    def boundaries(self) -> Tuple[Tuple[float, float, Block], ...]:
        """``(t_start, t_end, block)`` in session-relative seconds."""
        out, t = [], 0.0
        for b in self.blocks:
            out.append((t, t + b.duration_s, b))
            t += b.duration_s
        return tuple(out)

    def block_at(self, t_s: float) -> Optional[Block]:
        for start, end, blk in self.boundaries():
            if start <= t_s < end:
                return blk
        return None


def validate(protocol: Protocol) -> None:
    """Raise if any block would ask for something the supervisor must clamp."""
    if not protocol.blocks:
        raise ValueError(f"{protocol.name}: no blocks")
    for b in protocol.blocks:
        if b.duration_s <= 0:
            raise ValueError(f"{b.name}: duration must be positive")
        if b.steady_brightness is not None:
            if not 0.0 <= b.steady_brightness <= STEADY_BRIGHTNESS_MAX:
                raise ValueError(
                    f"{b.name}: steady brightness {b.steady_brightness} outside "
                    f"[0, {STEADY_BRIGHTNESS_MAX}] - continuous light has no duty "
                    f"cycle bounding its dose")
        for f in b.commanded_frequencies():
            if not FREQ_MIN_HZ <= f <= FREQ_MAX_HZ:
                raise ValueError(
                    f"{b.name}: {f} Hz outside [{FREQ_MIN_HZ}, {FREQ_MAX_HZ}]")
            if PSE_BAND_LO <= f <= PSE_BAND_HI:
                raise ValueError(f"{b.name}: {f} Hz is inside the PSE band")
        if b.kind == BlockKind.SWEEP:
            if not b.probe_hz:
                raise ValueError(f"{b.name}: sweep with no probes")
            need = sweep_min_duration_s(b.probe_hz, b.dwell_s)
            if need > b.duration_s:
                raise ValueError(
                    f"{b.name}: {b.duration_s:.0f} s cannot hold "
                    f"{len(b.probe_hz)} probes at {b.dwell_s:.0f} s dwell - "
                    f"needs at least {need:.0f} s even in monotonic order")


def sweep_min_duration_s(probe_hz: Sequence[float], dwell_s: float) -> float:
    """
    Shortest a sweep can be: every dwell plus a monotonic pass over the grid.

    A monotonic order pays the span once. Randomised orders pay more, which is
    why :func:`app.protocol.sweep.plan_sweep` only accepts an order whose
    transition cost fits the block budget.
    """
    if not probe_hz:
        return 0.0
    span = max(probe_hz) - min(probe_hz)
    return len(probe_hz) * dwell_s + span / MAX_FREQ_STEP_HZ_PER_S


def sweep_transition_budget_s(
    probe_hz: Sequence[float], dwell_s: float, duration_s: float
) -> float:
    """Seconds left for frequency transitions once every dwell is paid for."""
    return duration_s - len(probe_hz) * dwell_s


def _sweep(name: str, cue: str, kss: bool = True, note: str = "") -> Block:
    return Block(
        name=name, kind=BlockKind.SWEEP, duration_s=SWEEP_BLOCK_S,
        eyes=Eyes.CLOSED, cue_start=cue, kss_at_end=kss,
        probe_hz=PROBE_HZ, dwell_s=DWELL_S, note=note,
    )


def build_validation(seed: int = 0, montage: str = "occipital_alpha") -> Protocol:
    """
    Session A - does the loop measure brain and deliver light?

    Mask and electrodes go on once and are not touched again: every condition is
    reached by changing what is commanded, never by moving hardware. Device
    connectivity is checked by pre-flight before anything is worn.
    """
    ladder = tuple(
        Block(f"A4_bright_{int(b * 100):03d}", BlockKind.FIXED, 60.0,
              Eyes.CLOSED, "strobe_start", fixed_hz=SSVEP_CHECK_HZ, brightness=b,
              note=("strobe enabled at zero brightness - pickup floor"
                    if b == 0.0 else f"{int(b * 100)} % brightness"))
        for b in BRIGHTNESS_LADDER
    )
    blocks = (
        Block("A0_settle", BlockKind.REST, 90.0, Eyes.FREE, "lie_down",
              mask_check_s=MASK_CHECK_S,
              note="lie down, confirm the mask flashes - not analysed"),
        Block("A1_settle", BlockKind.REST, 120.0, Eyes.CLOSED, "settle",
              note="electrode settling; railed %, line noise, amplitude range"),
        Block("A2a_eyes_open", BlockKind.REST, 60.0, Eyes.OPEN, "eyes_open",
              steady_brightness=EYES_OPEN_BRIGHTNESS,
              note="steady light so the open eyes have something to see"),
        Block("A2b_eyes_closed", BlockKind.REST, 60.0, Eyes.CLOSED, "eyes_closed"),
        Block("A2c_eyes_open", BlockKind.REST, 60.0, Eyes.OPEN, "eyes_open",
              steady_brightness=EYES_OPEN_BRIGHTNESS,
              note="repeat - alpha blocking must replicate within the session"),
        Block("A2d_eyes_closed", BlockKind.REST, 60.0, Eyes.CLOSED, "eyes_closed"),
        # Second, independent way to modulate alpha, needing no light and no
        # movement: cognitive load suppresses it. Both blocks are eyes-closed
        # and dark, so nothing is injected into the band being measured.
        Block("A3a_alpha_rest", BlockKind.REST, 60.0, Eyes.CLOSED, "alpha_rest",
              note="eyes closed, resting - alpha should be high"),
        Block("A3b_alpha_task", BlockKind.REST, 60.0, Eyes.CLOSED, "alpha_task",
              note="eyes closed, silent arithmetic - alpha should drop"),
        *ladder,
        *[
            Block(f"A5_multi_{hz:g}hz", BlockKind.FIXED, 60.0, Eyes.CLOSED,
                  "strobe_start", fixed_hz=hz,
                  note="flat across frequency would mean pickup, not driving")
            for hz in MULTI_CHECK_HZ
        ],
        # Every other block holds one frequency, so nothing above tests that
        # the mask tracks a command that *moves* - which is what Session B does
        # for twenty minutes. It also decides how the LED driver handles a
        # retune: a free-running timer carries its phase over, an absolute-clock
        # one restarts it, and the two are indistinguishable at a fixed
        # frequency. The reconstruction PLV is scored against depends on the
        # answer, and PLV is in the agent's observation.
        Block("A6_ramp_check", BlockKind.DESCENT_RAMP, 60.0, Eyes.CLOSED,
              "strobe_start", ramp_start_hz=RAMP_START_HZ,
              ramp_end_hz=RAMP_END_HZ,
              note="does the mask follow a moving command, and on which clock"),
    )
    p = Protocol(
        name="validation", blocks=blocks, seed=seed, montage=montage,
        description="System validation. Gates the full session.",
    )
    validate(p)
    return p


def build_full(
    order: str = "ramp_first",
    seed: int = 0,
    montage: str = "occipital_alpha",
) -> Protocol:
    """
    Session B - does the sim-trained policy hold up against the ramp on a real
    brain?

    The primary outcome is the reward, computed from real EEG, for the scripted
    ramp against the learned policy. Both arms run **inside one session** in
    ABBA order, because a cross-night comparison is worthless and a sequential
    within-session one is biased: arousal falls over the session, so whichever
    arm runs second is scored on a drowsier brain.

    The re-alerting block sits in the middle of the four segments, so each arm
    gets one segment before it and one after - the step is balanced as well as
    the linear trend.

    ``order`` picks which arm takes the outer positions - ``abba`` puts the ramp
    there, ``baab`` the agent. Both are position-balanced on their own; swapping
    between nights also cancels the residual, non-linear part of the drift.
    :func:`suggest_order` picks whichever has been run less, so alternating
    needs no bookkeeping.

    Sweeps are secondary: they measure whether the response peak tracks arousal,
    which is what the simulator assumes. Their arousal trajectory is
    non-monotonic thanks to the re-alert, so a regression of peak on arousal can
    be separated from a monotonic electrode drift.
    """
    order = ARM_ORDER_ALIASES.get(order, order)
    if order not in ARM_ORDERS:
        raise ValueError(f"order must be one of {sorted(ARM_ORDERS)}, got {order!r}")

    outer = BlockKind.DESCENT_RAMP if order == "abba" else BlockKind.DESCENT_AGENT
    inner = BlockKind.DESCENT_AGENT if order == "abba" else BlockKind.DESCENT_RAMP

    def _seg(pos: int, kind: BlockKind) -> Block:
        arm = "ramp" if kind == BlockKind.DESCENT_RAMP else "agent"
        return Block(f"seg{pos}_{arm}", kind, SEGMENT_S, Eyes.CLOSED,
                     "descent_start",
                     ramp_start_hz=RAMP_START_HZ, ramp_end_hz=RAMP_END_HZ,
                     note=f"ABBA position {pos} of {N_SEGMENTS}")

    def _brief(name: str, note: str = "") -> Block:
        return Block(name, BlockKind.SWEEP, BRIEF_SWEEP_S, Eyes.CLOSED,
                     "sweep_start", kss_at_end=True,
                     probe_hz=BRIEF_PROBE_HZ, dwell_s=BRIEF_DWELL_S, note=note)

    blocks = (
        # Settling comes before anything is measured. Getting comfortable and
        # seating the mask takes time, and doing it inside the baseline would
        # put movement artifact into the resting reference every session.
        Block("B_settle", BlockKind.REST, 90.0, Eyes.FREE, "lie_down",
              mask_check_s=MASK_CHECK_S,
              note="lie down, adjust the mask - not analysed"),
        Block("B0_baseline", BlockKind.REST, 90.0, Eyes.CLOSED, "settle",
              kss_at_end=True, note="resting reference, no stimulation"),
        _brief("sweep1_alert", "resonance anchor while alert"),
        _seg(1, outer),
        _seg(2, inner),
        _brief("sweep2_drowsy"),
        Block("B4_realert", BlockKind.REST, 240.0, Eyes.OPEN, "realert_start",
              steady_brightness=EYES_OPEN_BRIGHTNESS,
              note="silent serial subtraction under steady light; "
                   "answer spoken at block end"),
        _brief("sweep3_realerted", "drift control - does the peak return?"),
        _seg(3, inner),
        _seg(4, outer),
        _brief("sweep4_drowsy"),
    )
    p = Protocol(
        name=f"full_{order}", blocks=blocks, seed=seed, montage=montage,
        description="Ramp against the learned policy, ABBA-balanced, one session.",
    )
    validate(p)
    check_arm_balance(p)
    return p


def arm_positions(protocol: Protocol) -> dict:
    """``{arm: [positions]}`` for the descent segments, in run order."""
    out: dict = {}
    pos = 0
    for b in protocol.blocks:
        if b.kind in (BlockKind.DESCENT_RAMP, BlockKind.DESCENT_AGENT):
            arm = "ramp" if b.kind == BlockKind.DESCENT_RAMP else "agent"
            out.setdefault(arm, []).append(pos)
            pos += 1
    return out


def check_arm_balance(protocol: Protocol) -> None:
    """
    Raise unless both arms occupy positions summing to the same value.

    This is the property that makes the contrast unbiased under a linear drift.
    ABBA satisfies it; AABB and ABAB do not, and ABAB is the dangerous one
    because it looks balanced.
    """
    pos = arm_positions(protocol)
    if len(pos) != 2:
        return
    sums = {arm: sum(p) for arm, p in pos.items()}
    counts = {arm: len(p) for arm, p in pos.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"{protocol.name}: arms have unequal segment counts {counts}")
    if len(set(sums.values())) != 1:
        raise ValueError(
            f"{protocol.name}: arm positions are not balanced {sums} - a linear "
            f"drift over the session would be charged to the arm effect")


MIN_SETTLED_S = 20.0    # analysable time a fixed-frequency block must retain

# A sweep needs more transition time than a monotonic pass, or randomised
# orders never fit and the schedule collapses back to ascending. The real
# blocks allow ~3.3x the monotonic cost; compressed rehearsals keep the same
# ratio.
SWEEP_TRANSITION_SLACK = 3.3


def sweep_feasible_duration_s(probe_hz: Sequence[float], dwell_s: float) -> float:
    """Shortest block that still admits a randomised probe order."""
    if not probe_hz:
        return 0.0
    span = max(probe_hz) - min(probe_hz)
    return (len(probe_hz) * dwell_s
            + SWEEP_TRANSITION_SLACK * span / MAX_FREQ_STEP_HZ_PER_S)


def settling_report(protocol: Protocol) -> Tuple[str, ...]:
    """
    Warn where the rate limiter eats a block.

    A fixed-frequency block starting far from the previous frequency spends the
    opening seconds in transit, and those samples are not analysable. Reported
    rather than raised: a compressed rehearsal is expected to trip this, a real
    session is not.
    """
    out, prev = [], None
    for b in protocol.blocks:
        if b.kind == BlockKind.FIXED and b.fixed_hz is not None:
            trans = 0.0 if prev is None else abs(b.fixed_hz - prev) / MAX_FREQ_STEP_HZ_PER_S
            settled = b.duration_s - trans
            if settled < MIN_SETTLED_S:
                out.append(
                    f"{b.name}: {trans:.0f} s of {b.duration_s:.0f} s spent "
                    f"reaching {b.fixed_hz:g} Hz, leaving {settled:.0f} s "
                    f"(want >= {MIN_SETTLED_S:.0f} s)")
            prev = b.fixed_hz
        elif b.kind == BlockKind.SWEEP and b.probe_hz:
            prev = None          # sweep order is chosen at run time
        elif b.kind == BlockKind.DESCENT_RAMP:
            prev = b.ramp_end_hz
        elif b.kind == BlockKind.REST:
            pass                 # frequency is held while the strobe is off
    return tuple(out)


def compress(protocol: Protocol, factor: float) -> Protocol:
    """
    Shorten every block by ``factor`` for smoke tests.

    Dwell shrinks with duration so sweeps stay feasible. The result is only for
    exercising the machinery - it is not a shorter experiment.
    """
    if not 0 < factor <= 1:
        raise ValueError("factor must be in (0, 1]")
    blocks = []
    for b in protocol.blocks:
        dwell = max(0.5, b.dwell_s * factor)
        dur = max(2.0, b.duration_s * factor)
        if b.kind == BlockKind.SWEEP:
            # The rate limiter is a property of the hardware, not of the
            # schedule, so a sweep cannot shrink past the time it physically
            # takes to walk the grid however short the dwells become.
            dur = max(dur, sweep_feasible_duration_s(b.probe_hz, dwell))
        blocks.append(Block(**{**b.__dict__, "duration_s": dur, "dwell_s": dwell}))
    blocks = tuple(blocks)
    p = Protocol(name=f"{protocol.name}_x{factor:g}", blocks=blocks,
                 seed=protocol.seed, montage=protocol.montage,
                 description=f"COMPRESSED x{factor:g} - smoke test only")
    validate(p)
    return p


ALL_BUILDERS = {"validation": build_validation, "full": build_full}


def arm_order_of(protocol: Protocol) -> Optional[str]:
    """
    Which named order a protocol actually uses, read from the segment sequence.

    Derived rather than parsed from the name: a compressed rehearsal is called
    ``full_abba_x0.05``, and matching on the name would miss it.
    """
    seq = tuple("ramp" if b.kind == BlockKind.DESCENT_RAMP else "agent"
                for b in protocol.blocks
                if b.kind in (BlockKind.DESCENT_RAMP, BlockKind.DESCENT_AGENT))
    for name, pattern in ARM_ORDERS.items():
        if seq == pattern:
            return name
    return None


def _completed_orders(session_root: "Path | str") -> dict:
    """Count completed sessions per arm order, from the recorded manifests."""
    import json
    from pathlib import Path as _Path

    counts = {k: 0 for k in ARM_ORDERS}
    root = _Path(session_root)
    if not root.is_dir():
        return counts
    for d in sorted(root.iterdir()):
        man = d / "manifest.json"
        if not (d.is_dir() and man.is_file()):
            continue
        try:
            m = json.loads(man.read_text(encoding="utf8"))
        except Exception:
            continue
        if m.get("status") != "completed":
            continue
        # A rehearsal is not a night. Compressed runs and simulated ones both
        # complete and both record an arm order, so counting them would flip
        # the real session's order and break the alternation they exist to
        # protect. The protocol name carries the compression factor.
        if m.get("eeg_kind") not in (None, "biolistener"):
            continue
        if "_x" in str(m.get("protocol", "")):
            continue
        # Prefer the recorded arm order; fall back to the protocol name.
        key = m.get("arm_order")
        if key not in counts:
            key = next((k for k in ARM_ORDERS
                        if str(m.get("protocol", "")).startswith(f"full_{k}")), None)
        if key in counts:
            counts[key] += 1
    return counts


def suggest_order(session_root: "Path | str" = "data/sessions") -> str:
    """
    The arm order run least so far, so nights alternate with no bookkeeping.

    Only completed sessions count: an aborted run delivered no segments and so
    has nothing to balance against.
    """
    counts = _completed_orders(session_root)
    # Ties resolve to declaration order, so a fresh setup starts on abba.
    return min(ARM_ORDERS, key=lambda k: (counts[k], list(ARM_ORDERS).index(k)))


def order_history(session_root: "Path | str" = "data/sessions") -> dict:
    """How many completed sessions used each arm order."""
    return _completed_orders(session_root)
