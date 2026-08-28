"""
Headless protocol runner.

Walks the blocks of a :class:`~app.protocol.spec.Protocol` on a fixed tick,
decides a stimulation command per block kind, puts every command through the
safety supervisor, and records everything.

The agent block reproduces the observation the checkpoints were trained on,
element for element and in the same order as ``SleepOnsetEnv._observe()``. The
smoothing step matters: the raw theta/alpha ratio is replaced by its smoothed
value *before* the feature vector is built, exactly as in training.

Timing runs off absolute deadlines rather than accumulated sleeps, so block
edges stay where the protocol says they are over a 41-minute session.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.agent.obs_state import ObsState
from app.agent.safety import (
    SafetySupervisor,
    StimAction,
    decode_action_for_deployment,
)
from app.agent.reward import RewardWeights, compute_reward
from app.brain_sim.photic_input import StrobeParams
from app.eeg.features import extract_features, feature_vector
from app.eeg.montage import ALL as MONTAGES
from app.eeg.plv import plv_strobe_to_eeg
from app.mask.client import MaskClient
from app.protocol import cues as cue_mod
from app.protocol.eeg_source import build_source
from app.protocol.quality import QualityMonitor
from app.protocol.recorder import SessionRecorder
from app.protocol.response import ClipRecorder, StdinInput, write_kss_skeleton
from app.protocol.spec import (
    Block, BlockKind, MASK_CHECK_HZ, Protocol, arm_order_of)
from app.protocol.spec import SSVEP_CHECK_HZ
from app.protocol.sweep import SweepPlan, plan_sweep

logger = logging.getLogger("protocol.runner")

# Amplitude used by every scripted block. The agent sets its own within the
# supervisor's envelope.
SCRIPTED_FILL = 0.40
SCRIPTED_BRIGHTNESS = 0.20

# Time held open *after the prompt finishes speaking*, long enough to answer
# while drowsy. The prompt's own length is measured rather than assumed: the
# anchored version runs to nearly twelve seconds.
KSS_ANSWER_S = 7.0

# What the BIOLISTENER actually streams. Commanding ``data_rate 500`` makes it
# round to a power-of-two divider and deliver 511.995 Hz, while
# ``BoardShim.get_sampling_rate`` keeps reporting the nominal 500. Analysing at
# 500 puts a commanded 6 Hz drive at 5.86 Hz and walks block boundaries tens of
# seconds off over a session. The simulated and playback sources really do run
# at 500, so the rate follows the source rather than being one global constant.
BIOLISTENER_SAMPLING_RATE = 512
NOMINAL_SAMPLING_RATE = 500
# What the device is actually commanded with. Both hardware sessions were
# recorded after asking for 500 and measured 511.995 Hz, so 500 is the only
# request this board is known to answer that way. Sending 512 back at it would
# be commanding a value never tested on the device, and a firmware that
# refused it would leave the rate wherever it happened to be.
BIOLISTENER_REQUESTED_DATA_RATE = 500

# White-LED level and per-LED enable used when no daemon config is loaded.
# The mask keeps whatever it was last told, so a runner that sends only
# frequency/fill/brightness can command 60 % brightness and emit no light.
DEFAULT_MASK_COLOR = [0, 0, 0, 255]        # R, G, B, white
DEFAULT_MASK_BITMASK = 0xFF                # all eight LEDs

# Pre-flight budgets. Discovery is by inbound ping, so the mask answers within
# a ping interval or it is not on this network.
MASK_DISCOVERY_S = 15.0
EEG_SETTLE_S = 20.0
# How often the mask is asked to confirm it is still there, and how long it may
# stay silent before the session aborts. The loop pauses for several seconds at
# each block transition while a prompt is spoken and answered, so the poll
# interval sits well inside the abort window. A UDP send reports success
# regardless of delivery, so a reply is the only evidence.
MASK_POLL_S = 5.0
MASK_SILENT_ABORT_S = 60.0

# How long the delivered sample rate is timed for during pre-flight, and how
# far it may sit from the configured figure before the session is refused.
# 8 s averages out WiFi burstiness and resolves the rate to well under 0.1 %.
RATE_MEASURE_S = 8.0
RATE_REJECT_FRAC = 0.20
# Sub-intervals the measurement is split into. A stalled link that then hands
# over its backlog puts one interval near zero and the next one far too high;
# the median ignores both, where a single long average would absorb them and
# report a plausible wrong number.
RATE_SUBWINDOWS = 8
# Spread across those sub-intervals, as a fraction of the median, above which
# the stream is too erratic to characterise in one figure.
RATE_JITTER_WARN = 0.10


def _measured_rate(session_dir: Path, nominal: float) -> Optional[float]:
    """True stream rate from the recording just written, or ``None``."""
    try:
        from app.protocol.analysis import load_raw, measure_sampling_rate
        for name in ("raw_eeg.csv", "raw_eeg.txt"):
            f = session_dir / name
            if f.is_file():
                return measure_sampling_rate(load_raw(f), float(nominal))
    except Exception:
        logger.warning("could not measure the recorded sample rate",
                       exc_info=True)
    return None


def _railed_of(window: np.ndarray, montage, pga_gain: int) -> float:
    """Railed percentage on the uncalibrated window, mean over primary EEG."""
    from app.eeg.features import _railed_percentage
    vals = [_railed_percentage(window[c].astype(np.float64), pga_gain)
            for c in montage.primary_eeg_channels if c < window.shape[0]]
    return float(np.mean(vals)) if vals else 0.0


class MaskLiveness:
    """When the mask last proved it was listening."""

    def __init__(self) -> None:
        self.last_reply_t: Optional[float] = None
        self._last_poll_t = 0.0

    def on_params(self, _msg: dict) -> None:
        self.last_reply_t = time.perf_counter()

    @property
    def answered(self) -> bool:
        return self.last_reply_t is not None

    def poll(self, mask) -> Optional[str]:
        """Ask periodically; return an abort reason once it has gone quiet."""
        now = time.perf_counter()
        if now - self._last_poll_t >= MASK_POLL_S:
            self._last_poll_t = now
            mask.get_params()
        if self.last_reply_t is None:
            return None
        silent = now - self.last_reply_t
        if silent > MASK_SILENT_ABORT_S:
            return f"mask silent for {silent:.0f} s"
        return None


class DeliveredEnvelope:
    """
    Per-sample record of the strobe envelope actually commanded.

    PLV is scored against this rather than against a waveform regenerated at
    one frequency: a regenerated reference assumes the command was constant
    across the window, so an arm that moves quickly is compared with a
    stimulus it never received and scores low for moving. The ramp drifts
    0.09 Hz inside a 4 s window and the agent up to 2 Hz, so a regenerated
    reference
    would score the two arms on different terms. Mirrors
    ``InProcessBrainSim``, which keeps the same buffer for the same reason.
    """

    def __init__(self, sampling_rate: int, capacity_s: float = 8.0,
                 phase_continuous: bool = True) -> None:
        from collections import deque
        self.dt = 1.0 / float(sampling_rate)
        self._buf: deque = deque(maxlen=int(capacity_s * sampling_rate))
        self._t = 0.0
        self._phase = 0.0
        self.phase_continuous = phase_continuous

    def extend(self, action: Optional[StimAction], n: int) -> None:
        """
        Append ``n`` samples of the envelope produced by ``action``.

        Two clocks, because the two plants differ. The simulator evaluates
        ``(t % period) / period`` against an absolute clock, so its phase jumps
        whenever the period changes - at 20 s in, a 0.006 Hz step moves it by a
        tenth of a cycle. Matching that is what reproduces training. A real
        driver runs a timer that keeps counting when it is retuned, so its
        phase carries over.
        """
        # Steady light is excluded: this envelope exists to be the reference
        # PLV is measured against, and continuous illumination has no phase to
        # lock to. Counting it would score a rest block against a constant.
        on = (action is not None and action.enabled and action.brightness > 0.0
              and not action.steady)
        freq = max(action.frequency_hz, 1e-6) if on else 0.0
        for _ in range(max(0, n)):
            if self.phase_continuous:
                self._phase = (self._phase + freq * self.dt) % 1.0
                ph = self._phase
            else:
                per = 1.0 / freq if freq else 1.0
                ph = (self._t % per) / per
            self._buf.append(1.0 if on and ph < action.fill else 0.0)
            self._t += self.dt

    def latest(self, n: int) -> Optional[np.ndarray]:
        """Last ``n`` samples, or ``None`` when no light was delivered."""
        if n <= 0 or not self._buf:
            return None
        arr = np.asarray(self._buf, dtype=np.float64)
        if arr.size < n:
            arr = np.concatenate([np.zeros(n - arr.size), arr])
        arr = arr[-n:]
        return arr if np.any(arr) else None


@dataclass
class RunnerConfig:
    protocol: Protocol
    lang: str = "en"
    tick_hz: float = 4.0
    window_s: float = 4.0

    eeg_kind: str = "sim"                 # sim | biolistener | playback
    eeg_ip: str = "0.0.0.0"
    eeg_port: int = 12345
    # None resolves from ``eeg_kind`` - see ``__post_init__``. Set it only to
    # override a device whose rate has been measured to be something else.
    sampling_rate: Optional[int] = None
    pga_gain: int = 8
    # Working ceiling on delivered light for this protocol, well inside the
    # safety envelope, which stays where it is. 60 % is tolerable but far too
    # bright for a session whose point is to let someone become drowsy, and
    # both arms share this ceiling so the dose comparison stays even. 20 % is
    # what a subject can lie under for half an hour, and it is also where the
    # response is largest: measured on hardware, 40 % drives *less* than 20 %,
    # so the higher ceiling cost comfort and bought nothing.
    brightness_max: float = 0.20
    # Inputs that actually have an electrode on them. Everything else is
    # switched off at session start so a floating input does not stream noise
    # into the recording. Must match the wiring table in the session plan.
    active_channels: list = field(default_factory=lambda: [0, 1, 2])
    playback_file: Optional[Path] = None

    # Local interface the controller socket binds to. A device address
    # belongs in ``mask_target_ip``: binding to it fails with EADDRNOTAVAIL,
    # because it is not an address of this machine.
    mask_ip: str = "127.0.0.1"
    # None means "choose": the real mask is always on 20000, but a virtual one
    # is on loopback and must not share a port with another virtual session on
    # this machine - SO_REUSEADDR would let both bind and split the packets.
    mask_port: Optional[int] = None
    mask_target_ip: Optional[str] = None   # pin the device; None = ping discovery
    virtual_mask: bool = True
    mask_color: list = field(default_factory=lambda: list(DEFAULT_MASK_COLOR))
    mask_bitmask: int = DEFAULT_MASK_BITMASK

    # Microvolts per unit of the training signal, measured from a Session A
    # baseline block by ``tools/calibrate_amplitude.py``. Band powers are
    # absolute, so a hardware recording lands far outside the range the
    # policy was trained on until it is divided through by this. None leaves
    # the window untouched, which is correct for the simulated path.
    amplitude_scale: Optional[float] = None
    # Whether the mask's LED driver carries its phase across a retune. None
    # means "simulator plant if simulating, free-running oscillator if not";
    # Session A's driver_phase_model decides it for real hardware.
    phase_continuous: Optional[bool] = None

    checkpoint: Optional[Path] = None     # required if the protocol has an agent block
    audio_cues: bool = True
    cue_root: Path = cue_mod.CUE_ROOT
    session_root: Path = Path("data/sessions")
    # False when something else owns the terminal, e.g. the daemon.
    watch_stdin: bool = True
    # Short microphone clip around each rating prompt, so the spoken
    # answer can be checked without searching a session-long file.
    record_clips: bool = True
    # Rehearsal only: drive the simulated resonance peak from a fixed time
    # course so a full-length run can verify that the fit recovers it.
    sim_exogenous_shift: bool = False
    # Which simulated subject to rehearse against. Defaults to the protocol
    # seed; seed 0 draws a near-non-responder with the cohort's slowest
    # transition timescale, so a rehearsal usually names a different one.
    sim_subject_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sampling_rate is None:
            self.sampling_rate = (
                BIOLISTENER_SAMPLING_RATE if self.eeg_kind == "biolistener"
                else NOMINAL_SAMPLING_RATE)

    @property
    def montage(self):
        return MONTAGES[self.protocol.montage]

    @property
    def tick_s(self) -> float:
        return 1.0 / self.tick_hz


@dataclass
class SessionRunner:
    cfg: RunnerConfig
    # Called once per tick with the recorded row plus block progress. Must not
    # raise and must not block: it runs inside the control loop.
    on_tick: Optional[Callable[[dict], None]] = None
    on_event: Optional[Callable[[str, dict], None]] = None
    _sweeps: Dict[str, SweepPlan] = field(default_factory=dict, init=False)
    _policy: Any = field(default=None, init=False)
    _stop: Optional[str] = field(default=None, init=False)
    _last_plv: float = field(default=0.0, init=False)
    _source: Any = field(default=None, init=False)
    _envelope: Any = field(default=None, init=False)
    _mask: Any = field(default=None, init=False)
    _external_rating: Optional[int] = field(default=None, init=False)
    _prev_features: Any = field(default=None, init=False)
    _prev_freq_hz: Optional[float] = field(default=None, init=False)
    _weights: RewardWeights = field(default_factory=RewardWeights, init=False)
    session_dir: Optional[Path] = field(default=None, init=False)
    kss_prompts: list = field(default_factory=list, init=False)

    def submit_rating(self, value: int) -> None:
        """Record a rating from something other than the terminal.

        The browser panel posts here; stdin is not watched in that mode.
        """
        self._external_rating = int(value)

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the loop to finish at the next tick. Safe from another thread."""
        self._stop = reason

    def _emit_tick(self, row: dict) -> None:
        if self.on_tick is None:
            return
        try:
            self.on_tick(row)
        except Exception:
            logger.debug("on_tick failed", exc_info=True)

    def _emit_event(self, kind: str, payload: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception:
            logger.debug("on_event failed", exc_info=True)

    # ---- setup -----------------------------------------------------------

    def _plan_sweeps(self) -> None:
        """Pre-plan every sweep so the orders live in the manifest."""
        prev_hz: Optional[float] = None
        for i, b in enumerate(self.cfg.protocol.blocks):
            if b.kind != BlockKind.SWEEP:
                if b.kind == BlockKind.DESCENT_RAMP:
                    prev_hz = b.ramp_end_hz
                elif b.kind == BlockKind.FIXED:
                    prev_hz = b.fixed_hz
                continue
            plan = plan_sweep(
                b.probe_hz, b.dwell_s, b.duration_s,
                seed=self.cfg.protocol.seed * 1000 + i, start_hz=prev_hz,
            )
            self._sweeps[b.name] = plan
            prev_hz = plan.order[-1]

    def _load_policy(self) -> None:
        needs = any(b.kind == BlockKind.DESCENT_AGENT
                    for b in self.cfg.protocol.blocks)
        if not needs:
            return
        if not self.cfg.checkpoint:
            raise ValueError("protocol has an agent block but no checkpoint given")
        from stable_baselines3 import SAC
        self._policy = SAC.load(str(self.cfg.checkpoint), device="cpu")
        logger.info("policy loaded from %s", self.cfg.checkpoint)

    # ---- per-block command ----------------------------------------------

    def _scripted_action(
        self, block: Block, t_in_block: float, obs: Optional[np.ndarray]
    ) -> StimAction:
        base = dict(fill=SCRIPTED_FILL, brightness=SCRIPTED_BRIGHTNESS,
                    audio_strobe_depth_pct=0.0, audio_binaural_vol_pct=0.0,
                    audio_master_vol_pct=0.0)

        # Visible confirmation, before anything else the block does. The
        # pre-flight round-trip proves the mask is listening; only this proves
        # light comes out, and it has to happen while the subject can still
        # look at it.
        if block.mask_check_s and t_in_block < block.mask_check_s:
            return StimAction(enabled=True, frequency_hz=MASK_CHECK_HZ, **base)

        if block.kind == BlockKind.REST:
            if block.steady_brightness is not None:
                # Continuous light, no flicker: the eyes have something to see
                # without anything being injected into the measured band. The
                # supervisor forces fill to 1 and holds brightness under its
                # own steady ceiling.
                return StimAction(
                    enabled=True, steady=True, frequency_hz=SSVEP_CHECK_HZ,
                    fill=1.0, brightness=float(block.steady_brightness),
                    audio_strobe_depth_pct=0.0,
                    audio_binaural_vol_pct=0.0,
                    audio_master_vol_pct=0.0)
            # Amplitude to zero, not just disabled: leaving it high would make
            # the next block's first flash arrive at full brightness, and would
            # record a rest block as though it were lit.
            return StimAction(enabled=False, frequency_hz=4.0,
                              fill=0.0, brightness=0.0,
                              audio_strobe_depth_pct=0.0,
                              audio_binaural_vol_pct=0.0,
                              audio_master_vol_pct=0.0)

        if block.kind == BlockKind.FIXED:
            # Zero brightness with the strobe still enabled is the artifact
            # control: same command stream, same frequency, no light.
            if block.brightness is not None:
                base = dict(base, brightness=float(block.brightness))
            return StimAction(enabled=True, frequency_hz=float(block.fixed_hz), **base)

        if block.kind == BlockKind.SWEEP:
            plan = self._sweeps[block.name]
            f = plan.frequency_at(t_in_block)
            if f is None:                      # sweep finished early; hold last
                f = plan.order[-1]
            return StimAction(enabled=True, frequency_hz=float(f), **base)

        if block.kind == BlockKind.DESCENT_RAMP:
            frac = min(1.0, max(0.0, t_in_block / block.duration_s))
            f = block.ramp_start_hz + frac * (block.ramp_end_hz - block.ramp_start_hz)
            return StimAction(enabled=True, frequency_hz=float(f), **base)

        if block.kind == BlockKind.BENCH:
            # Slow triangle across the legal band; confirms the mask tracks.
            lo, hi = 4.0, 13.0
            period = 60.0
            x = (t_in_block % period) / period
            f = lo + (hi - lo) * (2 * x if x < 0.5 else 2 * (1 - x))
            return StimAction(enabled=True, frequency_hz=float(f), **base)

        raise ValueError(f"no scripted command for {block.kind}")

    def _preflight(self, mask, source, live) -> dict:
        """
        Prove the mask answers and the EEG is flowing before the clock starts.

        Liveness has to be a round-trip. ``is_connected()`` returns True for
        the asking whenever a target address was pinned, and a UDP send to
        nobody succeeds just as happily as one to a listening device.
        ``CMD_GET_PARAMS`` is the evidence: the reply arrives on the same
        socket and marks the device live.
        """
        deadline = time.perf_counter() + MASK_DISCOVERY_S
        while not live.answered and time.perf_counter() < deadline:
            mask.get_params()
            time.sleep(0.5)
        if not live.answered:
            raise RuntimeError(
                f"mask did not reply on port {mask.port} within "
                f"{MASK_DISCOVERY_S:g} s"
                + (f" (target {mask.target_ip})" if mask.target_ip
                   else " (waiting for its pings)")
                + ". Power it on and check it is on this network.")
        mask.update_params(enabled=False, brightness=0.0,
                           color=self.cfg.mask_color,
                           mask=self.cfg.mask_bitmask)

        deadline = time.perf_counter() + EEG_SETTLE_S
        window = np.zeros((0, 0))
        while time.perf_counter() < deadline:
            window = source.latest_window(window_s=self.cfg.window_s)
            if window.size and window.shape[1] >= self.cfg.sampling_rate:
                break
            time.sleep(0.25)
        need = self.cfg.montage.primary_eeg_channels[0] + 1
        if window.shape[0] < need or window.shape[1] < self.cfg.sampling_rate:
            raise RuntimeError(
                f"EEG produced {window.shape} in {EEG_SETTLE_S:g} s; need at "
                f"least {need} channels and one second of samples")

        measured = self._measure_stream_rate(source)
        out = {"mask_ip": str(mask.board_ip or mask.target_ip),
               "eeg_channels": int(window.shape[0]),
               "eeg_samples": int(window.shape[1]),
               "configured_rate_hz": int(self.cfg.sampling_rate)}
        if measured is not None:
            out["measured_rate_hz"] = round(measured, 3)
            self.cfg.sampling_rate = int(round(measured))
            source.sampling_rate = self.cfg.sampling_rate
        return out

    def _measure_stream_rate(self, source) -> Optional[float]:
        """
        Samples per second a hardware source is actually delivering.

        Measured rather than configured, because a device need not honour the
        rate it is asked for: this one answers a commanded 500 with 511.995 and
        goes on reporting 500 through ``BoardShim.get_sampling_rate``. Every
        frequency in the analysis scales by the ratio and block boundaries walk
        by it, so trusting the declared figure silently ruins a session. Timing
        the stream costs eight seconds and cannot be wrong about it.

        Only the device is timed. A simulated or played-back source produces
        samples on demand from a rate it holds exactly, so timing it measures
        the polling loop rather than the source, and adopting that would
        introduce an error where there was none.
        """
        if self.cfg.eeg_kind != "biolistener":
            return None
        # Drain whatever accumulated between start-up and here before timing
        # anything: a pull-driven source hands over the whole backlog on its
        # first call, which would be counted as arrivals and inflate the rate.
        if hasattr(source, "stream_count"):
            for _ in range(3):
                source.latest_window(window_s=self.cfg.window_s)
                time.sleep(0.2)
        first = source.stream_count() if hasattr(source, "stream_count") else None
        if first is None:
            logger.info("source does not count samples; keeping the configured "
                        "rate of %d Hz", self.cfg.sampling_rate)
            return None
        # Polled rather than slept through: a simulated source produces on
        # demand, so a blind wait would measure it as delivering nothing. This
        # also matches how the control loop actually reads it.
        each = RATE_MEASURE_S / RATE_SUBWINDOWS
        rates = []
        prev_n, prev_t = first, time.perf_counter()
        for _ in range(RATE_SUBWINDOWS):
            edge = time.perf_counter() + each
            while time.perf_counter() < edge:
                source.latest_window(window_s=self.cfg.window_s)
                time.sleep(0.05)
            n, t = source.stream_count(), time.perf_counter()
            if n is not None and n > prev_n and t > prev_t:
                rates.append((n - prev_n) / (t - prev_t))
            prev_n, prev_t = n, t
        if not rates:
            logger.warning("sample counter did not advance; keeping the "
                           "configured rate of %d Hz", self.cfg.sampling_rate)
            return None
        rate = float(np.median(rates))
        spread = (max(rates) - min(rates)) / rate if rate else 0.0
        if spread > RATE_JITTER_WARN:
            logger.warning(
                "stream rate varied by %.0f %% across %.1f s (%.1f-%.1f Hz). "
                "The median of %.3f Hz is used, and block boundaries are "
                "anchored to sample timestamps rather than to this rate, but a "
                "link this erratic is worth fixing before a long session.",
                100 * spread, RATE_MEASURE_S, min(rates), max(rates), rate)
        nominal = float(self.cfg.sampling_rate)
        drift = abs(rate - nominal) / nominal
        if drift > RATE_REJECT_FRAC:
            raise RuntimeError(
                f"EEG is arriving at {rate:.1f} Hz but the session is "
                f"configured for {nominal:.0f} Hz. That is too far apart to be "
                f"a clock difference - check the device is streaming and that "
                f"nothing else is draining its buffer.")
        if drift > 0.005:
            logger.warning(
                "device streams %.3f Hz, not the configured %.0f Hz - using the "
                "measured value. Every frequency scales by %.3f.",
                rate, nominal, rate / nominal)
        return rate

    def _agent_action(self, obs: np.ndarray, supervisor: SafetySupervisor) -> StimAction:
        action, _ = self._policy.predict(obs, deterministic=True)
        return decode_action_for_deployment(action, supervisor)

    # ---- observation -----------------------------------------------------

    def _build_obs(
        self,
        window: np.ndarray,
        features,
        last_safe: Optional[StimAction],
        obs_state: ObsState,
    ) -> np.ndarray:
        """
        Must match ``SleepOnsetEnv._observe()``: 9 features, plv, 3 last-action,
        2 temporal. The cohort checkpoints were trained without restfulness, so
        that slot is absent here too.
        """
        plv_now = 0.0
        if last_safe is not None and last_safe.enabled:
            env_wave = self._envelope.latest(window.shape[1])
            ch = self.cfg.montage.primary_eeg_channels[0]
            if env_wave is not None and ch < window.shape[0]:
                plv_now = plv_strobe_to_eeg(
                    window[ch], env_wave, self.cfg.sampling_rate,
                    last_safe.frequency_hz)

        obs_state.update(raw_theta_alpha=float(features.theta_alpha_ratio),
                         raw_plv=float(plv_now))
        # Closes the other half of the loop on the simulated path: the plant's
        # resonance peak tracks measured drowsiness, exactly as during training.
        if self._source is not None:
            self._source.set_drowsiness(obs_state.smoothed_theta_alpha)
        # Smoothed value replaces the raw one before the vector is built - the
        # agent never saw the unsmoothed ratio during training.
        features.theta_alpha_ratio = obs_state.smoothed_theta_alpha
        # Training zeroed this at the source, so a live
        # value here is an input the policy has never seen. The measured one
        # still reaches the supervisor, the abort monitor and the tick log.
        features.railed_pct = 0.0

        last = last_safe or StimAction()
        smoothed_plv, delta_tar = obs_state.temporal_context()
        self._last_plv = float(plv_now)
        return np.concatenate([
            feature_vector(features),
            np.array([plv_now, last.frequency_hz, last.fill, last.brightness,
                      smoothed_plv, delta_tar], dtype=np.float32),
        ]).astype(np.float32)

    # ---- main loop -------------------------------------------------------

    def run(self) -> Path:
        cfg = self.cfg
        proto = cfg.protocol
        self._plan_sweeps()
        self._load_policy()

        player = cue_mod.CuePlayer(lang=cfg.lang, root=cfg.cue_root,
                                   enabled=cfg.audio_cues)
        missing = player.missing_cues()
        if cfg.audio_cues and missing:
            logger.warning("%d cues not rendered (%s ...); run tools/render_cues.py",
                           len(missing), ", ".join(missing[:3]))

        rec = SessionRecorder(protocol_name=proto.name, root=cfg.session_root).open()
        self.session_dir = rec.dir
        rec.write_manifest(
            protocol_description=proto.description,
            seed=proto.seed,
            montage=proto.montage,
            arm_order=arm_order_of(proto),
            tick_hz=cfg.tick_hz,
            window_s=cfg.window_s,
            eeg_kind=cfg.eeg_kind,
            sampling_rate=cfg.sampling_rate,
            sim_exogenous_shift=cfg.sim_exogenous_shift,
            sim_subject_seed=cfg.sim_subject_seed,
            checkpoint=str(cfg.checkpoint) if cfg.checkpoint else None,
            lang=cfg.lang,
            cues_missing=list(missing),
            blocks=[{"name": b.name, "kind": b.kind.value,
                     "duration_s": b.duration_s, "eyes": b.eyes.value,
                     "kss_at_end": b.kss_at_end, "note": b.note}
                    for b in proto.blocks],
            sweeps={k: {"order": list(v.order), "seed": v.seed,
                        "transition_cost_s": v.transition_cost_s}
                    for k, v in self._sweeps.items()},
        )

        # Controller and device need two ports on loopback: binding and
        # sending on one addresses this process's own socket. The real mask
        # both listens and pings on the same port, so there dev == ctrl.
        if cfg.mask_port is not None:
            ctrl_port = cfg.mask_port
        elif cfg.virtual_mask:
            ctrl_port = 20000 + 2 * (os.getpid() % 400)      # per-process pair
        else:
            ctrl_port = 20000
        dev_port = ctrl_port + 1 if cfg.virtual_mask else ctrl_port
        virtual = None
        if cfg.virtual_mask:
            from app.mask.virtual import VirtualMaskServer
            virtual = VirtualMaskServer(listen_ip="127.0.0.1",
                                        listen_port=dev_port,
                                        controller_addr=("127.0.0.1", ctrl_port))
            virtual.start()

        source = build_source(
            kind=cfg.eeg_kind, sampling_rate=cfg.sampling_rate,
            record_to=rec.raw_eeg_path,
            seed=(cfg.sim_subject_seed if cfg.sim_subject_seed is not None
                  else proto.seed),
            ip_address=cfg.eeg_ip, ip_port=cfg.eeg_port,
            playback_file=cfg.playback_file,
            exogenous_shift=cfg.sim_exogenous_shift,
            active_channels=cfg.active_channels,
            pga_gain=cfg.pga_gain,
            requested_data_rate=(BIOLISTENER_REQUESTED_DATA_RATE
                                 if cfg.eeg_kind == "biolistener" else None),
        )
        self._source = source
        rec.write_manifest(raw_eeg_rows=source.raw_eeg_rows())
        supervisor = SafetySupervisor(brightness_max=cfg.brightness_max)
        monitor = QualityMonitor()
        obs_state = ObsState(tick_hz=cfg.tick_hz)
        stdin_in = StdinInput(enabled=cfg.watch_stdin).start()
        clips = ClipRecorder(session_dir=rec.dir, enabled=cfg.record_clips)
        n_prompts = 0
        status, abort_reason = "completed", None
        last_safe: Optional[StimAction] = None
        n_safety_seen = 0
        t_shift = 0.0
        # The device's replies to CMD_GET_PARAMS are the only evidence that
        # anything is listening: UDP sends succeed regardless.
        live = MaskLiveness()
        try:
            source.start()
            # Stamped after start() returns, not before: BrainFlow opens the
            # streamer inside it, so the file begins here. Taking it earlier
            # folded the device's connect time into the offset - measured at
            # 1.3 and 1.7 s on the two hardware sessions, and a full 10 s if a
            # prepare_session attempt times out first - which shifts every
            # block window late by that much.
            t_stream0 = time.perf_counter()
            with MaskClient(bind_ip=cfg.mask_ip, port=ctrl_port,
                            target_ip=cfg.mask_target_ip,
                            on_params=live.on_params) as mask:
                mask.port = dev_port          # where commands are sent
                pre = self._preflight(mask, source, live)
                # Both depend on the sample rate, which pre-flight has just
                # measured off the running stream.
                samples_per_tick = int(round(cfg.sampling_rate * cfg.tick_s))
                self._envelope = DeliveredEnvelope(
                    cfg.sampling_rate,
                    phase_continuous=(cfg.phase_continuous
                                      if cfg.phase_continuous is not None
                                      else cfg.eeg_kind != "sim"))
                self._mask = mask
                rec.marker("session_start", protocol=proto.name,
                           total_s=proto.total_duration_s, **pre)
                t_session0 = time.perf_counter()
                # Markers are stamped against the recorder's own wall clock,
                # set when the directory was created. Analysis indexes raw
                # samples by marker time, so the offset between the two
                # origins is written to the manifest.
                rec.set_time_origin()
                # BrainFlow begins writing at source.start(), before
                # pre-flight, and hardware recordings carry no per-tick
                # sample count.
                rec.write_manifest(
                    raw_offset_s=round(t_session0 - t_stream0, 4),
                    amplitude_scale=cfg.amplitude_scale,
                    pga_gain=cfg.pga_gain,
                    active_channels=list(cfg.active_channels),
                    brightness_max=cfg.brightness_max,
                    sampling_rate=cfg.sampling_rate,
                    primary_eeg_channel=cfg.montage.primary_eeg_channels[0],
                    eeg_ip=cfg.eeg_ip,
                    mask_target_ip=cfg.mask_target_ip,
                    phase_continuous=self._envelope.phase_continuous)

                for nominal_start, nominal_end, block in proto.boundaries():
                    # Prompts are spoken and answered between blocks, outside
                    # every block's budget. Protocol boundaries are absolute,
                    # so each block's start and end carry the accumulated
                    # prompt time.
                    b_start = nominal_start + t_shift
                    b_end = nominal_end + t_shift
                    cue = block.cue_start
                    if block.name == "B4_realert":
                        cue = cue_mod.realert_cue_id(proto.seed)
                    player.play(cue)
                    rec.marker("block_start", block=block.name,
                               kind=block.kind.value, cue=cue,
                               eyes=block.eyes.value)
                    self._emit_event("block_start", {
                        "block": block.name, "kind": block.kind.value,
                        "eyes": block.eyes.value, "cue": cue,
                        "t_start_s": b_start, "t_end_s": b_end,
                        "note": block.note})

                    tick = 0
                    while True:
                        t_now = time.perf_counter() - t_session0
                        if t_now >= b_end:
                            break
                        if stdin_in.stop_reason or self._stop:
                            abort_reason = stdin_in.stop_reason or self._stop
                            break

                        window = source.latest_window(window_s=cfg.window_s)
                        # Rails are judged on the amplifier's own scale, so the
                        # check runs before any calibration is applied.
                        railed = (0.0 if cfg.eeg_kind == "sim" else
                                  _railed_of(window, cfg.montage, cfg.pga_gain))
                        if cfg.amplitude_scale:
                            window = window / cfg.amplitude_scale
                        features = extract_features(
                            window, cfg.sampling_rate, cfg.montage,
                            t_ms=int(t_now * 1000), pga_gain=cfg.pga_gain)
                        stop = monitor.update(railed)
                        if stop:
                            abort_reason = f"quality: {stop}"
                            break

                        t_in_block = t_now - b_start
                        obs = self._build_obs(window, features, last_safe, obs_state)
                        if block.kind == BlockKind.DESCENT_AGENT:
                            wanted = self._agent_action(obs, supervisor)
                        else:
                            wanted = self._scripted_action(block, t_in_block, obs)

                        safe = supervisor.clamp(wanted, railed_pct=railed)

                        # The primary outcome. Same reward function the policy
                        # was trained on, computed from real EEG, so the ramp
                        # and the agent are scored on identical terms.
                        reward = compute_reward(
                            prev=self._prev_features, curr=features,
                            plv_strobe=self._last_plv,
                            strobe_freq_hz=safe.frequency_hz,
                            prev_strobe_freq_hz=self._prev_freq_hz,
                            weights=self._weights)
                        # Delivered light per tick; the arm contrast is
                        # reported against it.
                        dose = safe.fill * safe.brightness if safe.enabled else 0.0
                        truth = source.ground_truth()
                        self._prev_features = features
                        self._prev_freq_hz = safe.frequency_hz
                        last_safe = safe

                        # The firmware's always-on mode is a commanded 0 Hz,
                        # which it holds regardless of fill. Steady blocks go
                        # out that way rather than as a full duty cycle, so no
                        # flicker frequency exists whose harmonics could land
                        # in the alpha band the blocking check reads.
                        wire_freq = 0.0 if safe.steady else safe.frequency_hz
                        # Colour and per-LED enable go with every command:
                        # the device keeps whatever it was last told.
                        mask.update_params(
                            enabled=safe.enabled,
                            frequency=wire_freq,
                            fill=safe.fill, brightness=safe.brightness,
                            color=cfg.mask_color, mask=cfg.mask_bitmask)
                        gone = live.poll(mask)
                        if gone:
                            abort_reason = gone
                            break
                        # The envelope the policy is scored against, sample by
                        # sample, advanced by exactly one tick's worth.
                        self._envelope.extend(safe, samples_per_tick)
                        # Closes the loop on the simulated path; a no-op on a
                        # real one, where the light reaches the subject directly.
                        source.update_strobe(StrobeParams(
                            enabled=safe.enabled, frequency_hz=safe.frequency_hz,
                            fill=safe.fill, brightness=safe.brightness))

                        plan = self._sweeps.get(block.name)
                        rec.tick(
                            t_s=round(t_now, 4),
                            raw_sample=source.samples_emitted,
                            block=block.name,
                            phase=plan.phase_at(t_in_block) if plan else block.kind.value,
                            probe_hz=(plan.frequency_at(t_in_block) if plan else ""),
                            cmd_freq_hz=round(wanted.frequency_hz, 4),
                            cmd_fill=round(wanted.fill, 4),
                            cmd_brightness=round(wanted.brightness, 4),
                            act_freq_hz=round(wire_freq, 4),
                            act_fill=round(safe.fill, 4),
                            act_brightness=round(safe.brightness, 4),
                            steady=int(safe.steady),
                            alpha_pwr=features.alpha_pwr,
                            theta_pwr=features.theta_pwr,
                            hypnagogic_theta_pwr=features.hypnagogic_theta_pwr,
                            sigma_pwr=features.sigma_pwr,
                            beta_pwr=features.beta_pwr,
                            theta_alpha_ratio=features.theta_alpha_ratio,
                            emg_rms=features.emg_rms,
                            eog_energy=features.eog_energy,
                            railed_pct=railed,
                            plv=round(self._last_plv, 6),
                            smoothed_plv=round(obs_state.smoothed_plv, 6),
                            delta_theta_alpha=round(
                                obs_state.delta_theta_alpha, 6),
                            reward=round(reward, 6),
                            dose=round(dose, 6),
                            **{k: round(v, 6) for k, v in truth.items()
                               if k in ("true_resonance_hz", "true_drowsiness")},
                        )
                        self._emit_tick({
                            "t_s": round(t_now, 3),
                            "block": block.name,
                            "block_kind": block.kind.value,
                            "block_elapsed_s": round(t_in_block, 2),
                            "block_duration_s": block.duration_s,
                            "session_elapsed_s": round(t_now, 2),
                            "session_duration_s": proto.total_duration_s,
                            "probe_hz": plan.frequency_at(t_in_block) if plan else None,
                            "phase": plan.phase_at(t_in_block) if plan else block.kind.value,
                            "act_freq_hz": round(safe.frequency_hz, 3),
                            "act_brightness": round(safe.brightness, 3),
                            "act_fill": round(safe.fill, 3),
                            "theta_alpha_ratio": features.theta_alpha_ratio,
                            "hypnagogic_theta_pwr": features.hypnagogic_theta_pwr,
                            "sigma_pwr": features.sigma_pwr,
                            "plv": self._last_plv,
                            "alpha_pwr": features.alpha_pwr,
                            "theta_pwr": features.theta_pwr,
                            "beta_pwr": features.beta_pwr,
                            "emg_rms": features.emg_rms,
                            "eog_energy": features.eog_energy,
                            "railed_pct": railed,
                            "reward": reward,
                            "dose": dose,
                            "n_safety_events": n_safety_seen,
                        })
                        events = supervisor.events_since(n_safety_seen)
                        if events:
                            rec.safety_events(events)
                            n_safety_seen += len(events)

                        tick += 1
                        deadline = t_session0 + b_start + tick * cfg.tick_s
                        sleep = deadline - time.perf_counter()
                        if sleep > 0:
                            time.sleep(sleep)

                    if abort_reason:
                        break

                    rec.marker("block_end", block=block.name, ticks=tick)
                    self._emit_event("block_end", {"block": block.name,
                                                   "ticks": tick})
                    prompt_t0 = time.perf_counter()
                    if block.kss_at_end:
                        n_prompts += 1
                        # Anchors in full the first time, so the scale means
                        # something; a short reminder after that.
                        cue_id = ("kss_prompt_anchored" if n_prompts == 1
                                  else "kss_prompt")
                        player.play(cue_id)
                        stdin_in.clear_rating()
                        self._external_rating = None
                        rec.marker("kss_prompt", block=block.name, cue=cue_id)
                        self._emit_event("kss_prompt", {"block": block.name})
                        # Let the prompt finish before recording, so the clip
                        # holds the answer and not the question.
                        time.sleep(player.duration(cue_id))
                        clip = clips.capture(f"{n_prompts:02d}_{block.name}")
                        time.sleep(KSS_ANSWER_S)
                        typed = stdin_in.take_rating()
                        if typed is None and self._external_rating is not None:
                            typed = self._external_rating
                        self._external_rating = None
                        entry = {"t_s": round(time.perf_counter() - t_session0, 2),
                                 "block": block.name, "value": typed,
                                 "clip": (clip.name if clip else "")}
                        self.kss_prompts.append(entry)
                        if typed is not None:
                            rec.marker("kss_value", block=block.name, value=typed,
                                       source="keyboard")
                            self._emit_event("kss_value", entry)
                    if block.name == "B4_realert":
                        player.play("realert_answer")
                        rec.marker("realert_answer_prompt", block=block.name)
                        time.sleep(player.duration("realert_answer"))
                        clips.capture("00_realert_answer")
                        time.sleep(KSS_ANSWER_S)
                    t_shift += time.perf_counter() - prompt_t0

                if abort_reason:
                    status = "aborted"
                    player.play("abort")
                    rec.marker("abort", reason=abort_reason)
                    self._emit_event("abort", {"reason": abort_reason})
                    logger.warning("session aborted: %s", abort_reason)
                else:
                    player.play("session_end")
                    rec.marker("session_end")
                    self._emit_event("session_end", {})
        except KeyboardInterrupt:
            # KeyboardInterrupt is not an Exception subclass; caught here so
            # the session is filed as aborted and `finally` still runs.
            status, abort_reason = "aborted", "keyboard interrupt"
            rec.marker("abort", reason=abort_reason)
            logger.warning("session aborted from the keyboard")
        except Exception as err:                       # noqa: BLE001
            status, abort_reason = "error", repr(err)
            rec.marker("error", detail=repr(err))
            logger.exception("session failed")
            raise
        finally:
            # The strobe is switched off by MaskClient.__exit__, which runs
            # while its socket is still open - doing it here would send on a
            # closed one.
            try:
                clips.stop()
                if self.kss_prompts:
                    write_kss_skeleton(rec.dir, self.kss_prompts)
            except Exception:
                logger.debug("kss finalise failed", exc_info=True)
            try:
                source.stop()
            except Exception:
                logger.debug("eeg source stop failed", exc_info=True)
            if virtual is not None:
                virtual.stop()
            # What the device actually streamed, read back off the recording's
            # own timestamps. The configured rate is what it was asked for, and
            # the two differ on this board; analysis scales every frequency by
            # the ratio, so the measured value is recorded rather than assumed.
            rec.write_manifest(abort_reason=abort_reason,
                               worst_railed_pct=monitor.worst_railed_pct,
                               n_safety_events=n_safety_seen,
                               measured_sampling_rate=_measured_rate(rec.dir, cfg.sampling_rate))
            rec.close(status=status)

        return rec.dir
