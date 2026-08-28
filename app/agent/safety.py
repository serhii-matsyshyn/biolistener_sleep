"""
Safety supervisor - last line of defense between RL agent and actuators.

Runs AFTER the agent produces an action and BEFORE the action reaches the
mask UDP client or audio engine. Every clamp is logged as a
``SafetyEvent`` for the session record so the thesis can defend the
exact actions that ever reached the user.

Enforced rules, from photosensitive-epilepsy (PSE) guidance:

  Fisher et al. (2022) "Visually sensitive seizures: an updated review by
    the Epilepsy Foundation", Epilepsia 63(4):739-768.
    doi:10.1111/epi.17175                      <- current consensus
  Harding et al. (2005) "Photic- and pattern-induced seizures: expert
    consensus of the Epilepsy Foundation of America Working Group",
    Epilepsia 46(9):1423-1425. doi:10.1111/j.1528-1167.2005.31305.x
  Fisher et al. (2005), Epilepsia 46(9):1426-1441 - companion review.
    doi:10.1111/j.1528-1167.2005.31405.x
  Standards envelope: Ofcom Broadcasting Code guidance (UK, successor to
    the ITC), ISO 9241-391, ITU-R BT.1702-3, W3C WCAG 2.x SC 2.3.1.

Limits of this guard:

1. Brightness is uncalibrated. Harding's flash criteria are conjunctive
   (luminance >=20 cd/m^2 AND >=3 Hz AND solid angle >=0.006 sr), and a
   mask on closed eyelids always meets the frequency and solid-angle
   limbs, so luminance carries the margin.

2. Colour is not constrained here. Harding adds a luminance-independent
   criterion - a transition to or from saturated red is itself a risk -
   and the mask's RGBW is operator-set via ``mask.strobe.*``, outside
   this module.

3. The operating band sits inside the hazard range. [15, 25] Hz is the
   peak-risk window this module hard-blocks, but the reported hazard
   range is wider (Fisher 2022: 3-60 Hz). Operating at 3-14 Hz is
   justified on dose grounds - luminance and duty cycle - rather than on
   frequency alone.

- Strobe frequency clipped to [3.0, 14.0] Hz steady-state.
  **15–25 Hz is the highest-risk PSE band** and is hard-blocked
  (`pse_guard`). Transient passes through 14–15 Hz during ramping are
  capped by ``max_freq_step_hz_per_s``.
- Brightness in [0.0, 0.60] for v1; ramp rate ≤ 20 %/s
  (``max_brightness_step_per_s``).
- Duty cycle (fill) in [0.0, 0.40] - high duty × high freq dramatically
  increases retinal load.
- Steady light (``StimAction.steady``) is continuous illumination rather
  than flicker: the duty ceiling does not apply because there is no
  flicker to bound, and brightness is capped far lower instead
  (``STEADY_BRIGHTNESS_MAX``). The flash criteria in the guidance above
  are conjunctive on frequency, so unmodulated light does not meet them
  at any luminance this device reaches; what remains is sustained
  retinal exposure, which the tighter cap bounds. Time-averaged output
  is lower than a flicker block's: 0.05 continuous against 0.40 × 0.40 =
  0.16 for the brightest strobe.
- Session hard timeout = 45 min by default.
- Panic stop (UI button, /panic REST, or ESC) -> enabled=False, brightness=0,
  audio fadeout, session ends in `aborted` state.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("agent.safety")


# Safe operating bounds - change requires explicit thesis-level justification
FREQ_MIN_HZ = 3.0
FREQ_MAX_HZ = 14.0
PSE_BAND_LO = 15.0
PSE_BAND_HI = 25.0
BRIGHTNESS_MIN = 0.0
BRIGHTNESS_MAX_V1 = 0.60
# Ceiling for continuous, unmodulated light. Deliberately far below the
# flicker ceiling: light that never goes off has no duty cycle to limit it,
# so brightness is the only quantity bounding the retinal dose.
STEADY_BRIGHTNESS_MAX = 0.10
FILL_MIN = 0.0
FILL_MAX = 0.40
DEFAULT_SESSION_TIMEOUT_S = 45 * 60
# Ramping stimulation parameters rather than stepping them is standard
# practice in audio-visual entrainment protocols, recommended to reduce
# "startle and fatigue" (Rahmani, Romero Lauro & Pisoni 2025, Brain Sci
# 15(10):1070, doi:10.3390/brainsci15101070). No published rate exists for
# photic frequency transitions, so this value is a conservative engineering
# choice.
MAX_FREQ_STEP_HZ_PER_S = 0.5
MAX_BRIGHTNESS_STEP_PER_S = 0.20  # i.e. 20%/s
RAILED_PCT_DISCONNECT_THRESHOLD = 80.0


@dataclass
class StimAction:
    """Agent's commanded stimulation parameters before clamping."""
    enabled: bool = True
    frequency_hz: float = 10.0
    fill: float = 0.20
    brightness: float = 0.30
    # Continuous light instead of a strobe. Scripted blocks set it; the
    # policy cannot, because ``decode_action_for_deployment`` never does.
    steady: bool = False
    color_white: int = 200       # white channel 0-255
    # audio side - controller already clips its own setters
    audio_strobe_depth_pct: float = 10.0
    audio_binaural_vol_pct: float = 10.0
    audio_master_vol_pct: float = 50.0


@dataclass
class SafetyEvent:
    t_ms: int
    kind: str        # clamp | panic | pse_guard | ramp_limit | session_timeout | electrode_disconnect | baseline_revert
    field: str
    before: float
    after: float
    reason: str


def decode_action_for_deployment(
    action,
    supervisor: "SafetySupervisor",
) -> "StimAction":
    """Map the agent's normalised [-1, 1] action vector directly into the
    supervisor's safe envelope, so the full action resolution is spent
    inside safe bounds: +1 is ``brightness_max``, -1 is ``brightness_min``.

    Audio params (action[3], action[4]) keep the plain [0, 100 %] mapping
    because there are no safety bounds on them - the deterministic audio
    shaper handles their perceptual envelope separately.
    """
    sv = supervisor
    a0 = float(action[0]); a1 = float(action[1]); a2 = float(action[2])
    a3 = float(action[3]); a4 = float(action[4])
    freq = sv.freq_min_hz + (a0 + 1.0) / 2.0 * (sv.freq_max_hz - sv.freq_min_hz)
    fill = sv.fill_min + (a1 + 1.0) / 2.0 * (sv.fill_max - sv.fill_min)
    brightness = sv.brightness_min + (a2 + 1.0) / 2.0 * (sv.brightness_max - sv.brightness_min)
    return StimAction(
        enabled=True,
        frequency_hz=float(freq),
        fill=float(fill),
        brightness=float(brightness),
        color_white=200,
        audio_strobe_depth_pct=float((a3 + 1.0) / 2.0 * 100.0),
        audio_binaural_vol_pct=10.0,
        audio_master_vol_pct=float((a4 + 1.0) / 2.0 * 100.0),
    )


@dataclass
class SafetySupervisor:
    """Stateful clamp - knows last action + last action time for ramp limits.

    Every safety bound is a constructor field with the module-level
    constants as defaults. Use ``SafetySupervisor.from_config(cfg.safety)``
    at session start to pull every bound from YAML. Callers that just
    do ``SafetySupervisor()`` get the conservative v1 defaults baked in
    at the top of this file.
    """
    freq_min_hz: float = FREQ_MIN_HZ
    freq_max_hz: float = FREQ_MAX_HZ
    pse_band_lo: float = PSE_BAND_LO
    pse_band_hi: float = PSE_BAND_HI
    brightness_min: float = BRIGHTNESS_MIN
    brightness_max: float = BRIGHTNESS_MAX_V1
    fill_min: float = FILL_MIN
    fill_max: float = FILL_MAX
    steady_brightness_max: float = STEADY_BRIGHTNESS_MAX
    session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S
    max_freq_step_hz_per_s: float = MAX_FREQ_STEP_HZ_PER_S
    max_brightness_step_per_s: float = MAX_BRIGHTNESS_STEP_PER_S
    railed_pct_disconnect_threshold: float = RAILED_PCT_DISCONNECT_THRESHOLD
    on_event: Optional[Callable[[SafetyEvent], None]] = None
    # Fixed inter-action interval, in seconds, for ramp-rate limiting AND
    # for the session clock.
    # None (default) = derive dt from wall clock, correct for live sessions.
    # Set to 1 / tick_hz when driving a simulated clock so the limiter
    # enforces its documented rate, ``SafetyEvent.t_ms`` timestamps are in
    # *simulated* seconds, and rollouts are reproducible.
    tick_dt_s: Optional[float] = None

    _events: List[SafetyEvent] = field(default_factory=list)
    _last_action: Optional[StimAction] = None
    _last_action_t: float = 0.0
    _session_start_t: float = field(default_factory=time.time)
    # Simulated seconds since session start; advanced by one ``tick_dt_s``
    # per clamp() call and only consulted when tick_dt_s is set.
    _sim_elapsed_s: float = 0.0
    _panic_latched: bool = False

    @classmethod
    def from_config(cls, cfg) -> "SafetySupervisor":
        """Build a SafetySupervisor with every bound taken from a
        :class:`app.config.schema.SafetyConfig`. The caller is the daemon
        / demo controller at session start."""
        return cls(
            freq_min_hz=cfg.freq_min_hz,
            freq_max_hz=cfg.freq_max_hz,
            pse_band_lo=cfg.pse_band_lo,
            pse_band_hi=cfg.pse_band_hi,
            brightness_min=cfg.brightness_min,
            brightness_max=cfg.brightness_max,
            fill_min=cfg.fill_min,
            fill_max=cfg.fill_max,
            session_timeout_s=cfg.session_timeout_s,
            max_freq_step_hz_per_s=cfg.max_freq_step_hz_per_s,
            max_brightness_step_per_s=cfg.max_brightness_step_per_s,
            railed_pct_disconnect_threshold=cfg.railed_pct_disconnect_threshold,
        )

    @property
    def events(self) -> List[SafetyEvent]:
        return list(self._events)

    @property
    def event_count(self) -> int:
        """Number of events logged so far, without copying the list.

        ``events`` copies the list, which per-tick consumers pay for.
        """
        return len(self._events)

    def events_since(self, index: int) -> List[SafetyEvent]:
        """Events logged after ``index`` (pair with ``event_count``)."""
        return self._events[index:]

    @property
    def panic_active(self) -> bool:
        return self._panic_latched

    def reset(self) -> None:
        self._events.clear()
        self._last_action = None
        self._last_action_t = 0.0
        self._session_start_t = time.time()
        self._sim_elapsed_s = 0.0
        self._panic_latched = False

    def panic(self, reason: str = "user_panic") -> StimAction:
        """Latch panic state - subsequent clamp() calls force OFF state."""
        if not self._panic_latched:
            self._panic_latched = True
            self._log(self._now_ms(), "panic", "enabled", 1.0, 0.0, reason)
        return StimAction(
            enabled=False, frequency_hz=0.0, fill=0.0, brightness=0.0,
            color_white=0,
            audio_strobe_depth_pct=0.0,
            audio_binaural_vol_pct=0.0,
            audio_master_vol_pct=0.0,
        )

    def session_elapsed_s(self) -> float:
        """Seconds since session start on whichever clock is configured.

        With ``tick_dt_s`` set this is *simulated* time (one tick per
        ``clamp()`` call), so session-timeout checks and ``SafetyEvent``
        timestamps are meaningful in a simulator that runs ~1000x faster
        than real time. Otherwise it is wall time, correct for a live
        session.
        """
        if self.tick_dt_s is not None:
            return self._sim_elapsed_s
        return time.time() - self._session_start_t

    def clamp(
        self,
        action: StimAction,
        railed_pct: float = 0.0,
    ) -> StimAction:
        """
        Apply all safety rules and return the safe action.

        Records a SafetyEvent for every modification. Advances the
        simulated session clock by one tick on every call (including the
        early-return paths), so event timestamps stay in step with the
        control loop.
        """
        try:
            return self._clamp_impl(action, railed_pct)
        finally:
            if self.tick_dt_s is not None:
                self._sim_elapsed_s += float(self.tick_dt_s)

    def _clamp_impl(
        self,
        action: StimAction,
        railed_pct: float = 0.0,
    ) -> StimAction:
        if self._panic_latched:
            return self.panic("panic_latched")

        t_ms = self._now_ms()
        result = StimAction(
            enabled=action.enabled,
            frequency_hz=action.frequency_hz,
            fill=action.fill,
            brightness=action.brightness,
            steady=action.steady,
            color_white=int(max(0, min(255, action.color_white))),
            audio_strobe_depth_pct=action.audio_strobe_depth_pct,
            audio_binaural_vol_pct=action.audio_binaural_vol_pct,
            audio_master_vol_pct=action.audio_master_vol_pct,
        )

        # 1) Session timeout - soft (just disable, don't latch panic)
        if self.session_elapsed_s() > self.session_timeout_s:
            self._log(t_ms, "session_timeout", "enabled", 1.0, 0.0,
                      f">{self.session_timeout_s:.0f}s")
            result.enabled = False
            result.brightness = 0.0
            return result

        # 2) Electrode disconnect - drop to safe OFF, don't latch
        if railed_pct > self.railed_pct_disconnect_threshold:
            self._log(t_ms, "electrode_disconnect", "enabled", 1.0, 0.0,
                      f"railed={railed_pct:.1f}%")
            result.enabled = False
            result.brightness = 0.0
            return result

        # 3) PSE band guard - REJECT outright if requested freq in [15, 25]
        if self.pse_band_lo <= result.frequency_hz <= self.pse_band_hi:
            self._log(t_ms, "pse_guard", "frequency_hz",
                      result.frequency_hz, self.freq_max_hz,
                      f"requested {result.frequency_hz:.2f} Hz in PSE band")
            result.frequency_hz = self.freq_max_hz

        # 4) Frequency hard clamp
        new_freq = max(self.freq_min_hz, min(self.freq_max_hz, result.frequency_hz))
        if abs(new_freq - result.frequency_hz) > 1e-6:
            self._log(t_ms, "clamp", "frequency_hz",
                      result.frequency_hz, new_freq, "freq_bounds")
        result.frequency_hz = new_freq

        # 5) Fill clamp. Steady light is continuous by definition, so the duty
        #    ceiling - which bounds flicker load - has nothing to bound.
        if result.steady:
            result.fill = 1.0
        else:
            new_fill = max(self.fill_min, min(self.fill_max, result.fill))
            if abs(new_fill - result.fill) > 1e-6:
                self._log(t_ms, "clamp", "fill", result.fill, new_fill,
                          "fill_bounds")
            result.fill = new_fill

        # 6) Brightness clamp. Continuous light carries its whole dose here,
        #    so it answers to the lower of the two ceilings.
        ceiling = min(self.brightness_max, self.steady_brightness_max) \
                  if result.steady else self.brightness_max
        new_bri = max(self.brightness_min, min(ceiling, result.brightness))
        if abs(new_bri - result.brightness) > 1e-6:
            self._log(t_ms, "clamp", "brightness",
                      result.brightness, new_bri,
                      "steady_brightness_bounds" if result.steady
                      else "brightness_bounds")
        result.brightness = new_bri

        # 7) Ramp-rate limits relative to last applied action.
        #
        # dt is wall-clock by default, which is correct for a live session
        # driven in real time. In simulation the loop runs far faster than
        # real time, so wall-clock dt would (a) make the limiter permit only
        # a fraction of its documented rate and (b) make results depend on
        # machine speed. Callers running against a simulated clock pass
        # ``tick_dt_s`` (e.g. 1 / tick_hz) to get the intended rate and
        # reproducible trajectories.
        now = self.session_elapsed_s() if self.tick_dt_s is not None \
              else time.time()
        if self._last_action is not None:
            if self.tick_dt_s is not None:
                dt = float(self.tick_dt_s)
            else:
                dt = max(1e-3, now - self._last_action_t)
            max_df = self.max_freq_step_hz_per_s * dt
            req_df = result.frequency_hz - self._last_action.frequency_hz
            if abs(req_df) > max_df:
                ramp_freq = self._last_action.frequency_hz \
                            + max_df * (1 if req_df > 0 else -1)
                self._log(t_ms, "ramp_limit", "frequency_hz",
                          result.frequency_hz, ramp_freq,
                          f"|Δf|>{self.max_freq_step_hz_per_s} Hz/s")
                result.frequency_hz = ramp_freq

            max_dbri = self.max_brightness_step_per_s * dt
            req_dbri = result.brightness - self._last_action.brightness
            if abs(req_dbri) > max_dbri:
                ramp_bri = self._last_action.brightness \
                           + max_dbri * (1 if req_dbri > 0 else -1)
                self._log(t_ms, "ramp_limit", "brightness",
                          result.brightness, ramp_bri,
                          f"|Δb|>{self.max_brightness_step_per_s}/s")
                result.brightness = ramp_bri

        self._last_action = result
        self._last_action_t = now
        return result

    def _now_ms(self) -> int:
        return int(self.session_elapsed_s() * 1000)

    def _log(self, t_ms: int, kind: str, field_name: str,
             before: float, after: float, reason: str) -> None:
        evt = SafetyEvent(t_ms=t_ms, kind=kind, field=field_name,
                          before=float(before), after=float(after),
                          reason=reason)
        self._events.append(evt)
        if self.on_event is not None:
            try:
                self.on_event(evt)
            except Exception:
                logger.exception("on_event callback failed")
        logger.info("SAFETY %s on %s: %.3f -> %.3f (%s)",
                    kind, field_name, before, after, reason)
