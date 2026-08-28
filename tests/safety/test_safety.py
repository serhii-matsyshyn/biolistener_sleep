"""
Safety supervisor tests - every clamp must trigger an event and the
returned action must always be within safe bounds.
"""
from __future__ import annotations

import time

import pytest

from app.agent.safety import (
    BRIGHTNESS_MAX_V1,
    FILL_MAX,
    FREQ_MAX_HZ,
    PSE_BAND_HI,
    PSE_BAND_LO,
    RAILED_PCT_DISCONNECT_THRESHOLD,
    STEADY_BRIGHTNESS_MAX,
    SafetySupervisor,
    StimAction,
    decode_action_for_deployment,
)


def test_pse_band_request_is_blocked():
    sup = SafetySupervisor()
    for freq in [PSE_BAND_LO, 18.0, 22.5, PSE_BAND_HI]:
        out = sup.clamp(StimAction(frequency_hz=freq))
        assert out.frequency_hz <= FREQ_MAX_HZ
        assert any(e.kind == "pse_guard" for e in sup.events[-5:])


def test_freq_below_min_is_clamped_up():
    sup = SafetySupervisor()
    out = sup.clamp(StimAction(frequency_hz=0.5))
    assert out.frequency_hz >= 3.0
    assert any(e.kind == "clamp" and e.field == "frequency_hz"
               for e in sup.events)


def test_brightness_capped_at_v1_max():
    sup = SafetySupervisor()
    out = sup.clamp(StimAction(brightness=1.0))
    assert out.brightness <= BRIGHTNESS_MAX_V1


def test_fill_capped_at_max():
    sup = SafetySupervisor()
    out = sup.clamp(StimAction(fill=0.95))
    assert out.fill <= FILL_MAX


def test_electrode_disconnect_disables_strobe():
    sup = SafetySupervisor()
    out = sup.clamp(
        StimAction(enabled=True, frequency_hz=10.0, brightness=0.4),
        railed_pct=RAILED_PCT_DISCONNECT_THRESHOLD + 5,
    )
    assert out.enabled is False
    assert out.brightness == 0.0
    assert any(e.kind == "electrode_disconnect" for e in sup.events)


def test_panic_latches_off():
    sup = SafetySupervisor()
    sup.panic("test")
    out = sup.clamp(StimAction(enabled=True, frequency_hz=10.0, brightness=0.3))
    assert out.enabled is False
    assert out.brightness == 0.0
    assert sup.panic_active


def test_session_timeout_disables():
    sup = SafetySupervisor(session_timeout_s=0.0001)
    time.sleep(0.01)
    out = sup.clamp(StimAction(enabled=True, frequency_hz=10.0, brightness=0.3))
    assert out.enabled is False
    assert any(e.kind == "session_timeout" for e in sup.events)


def test_ramp_limit_caps_brightness_step():
    sup = SafetySupervisor()
    # First action: low brightness
    a1 = sup.clamp(StimAction(brightness=0.1))
    assert abs(a1.brightness - 0.1) < 1e-6
    # Wait a tiny moment to keep dt small, then request a big jump
    time.sleep(0.05)  # 50ms -> max Δb = 0.2 × 0.05 = 0.01
    a2 = sup.clamp(StimAction(brightness=0.6))
    # Should be ramp-limited to a1.brightness + max_step
    assert a2.brightness < 0.5
    assert any(e.kind == "ramp_limit" and e.field == "brightness"
               for e in sup.events)


# ---- steady (unmodulated) light ------------------------------------------

def test_steady_light_answers_to_its_own_lower_ceiling():
    sv = SafetySupervisor(brightness_max=0.60)
    out = sv.clamp(StimAction(enabled=True, steady=True, brightness=0.55))
    assert out.brightness == pytest.approx(STEADY_BRIGHTNESS_MAX)
    assert any(e.reason == "steady_brightness_bounds" for e in sv.events)


def test_the_flicker_ceiling_still_applies_when_light_is_not_steady():
    sv = SafetySupervisor(brightness_max=0.40)
    out = sv.clamp(StimAction(enabled=True, steady=False, brightness=0.55))
    assert out.brightness == pytest.approx(0.40)


def test_steady_light_is_continuous_rather_than_duty_limited():
    sv = SafetySupervisor()
    out = sv.clamp(StimAction(enabled=True, steady=True, fill=0.2,
                              brightness=0.05))
    assert out.fill == pytest.approx(1.0)


def test_steady_light_delivers_less_average_output_than_the_brightest_strobe():
    sv = SafetySupervisor(brightness_max=0.40)
    steady = sv.clamp(StimAction(enabled=True, steady=True, brightness=0.05))
    sv.reset()
    strobe = sv.clamp(StimAction(enabled=True, fill=0.40, brightness=0.40))
    assert steady.fill * steady.brightness < strobe.fill * strobe.brightness


def test_the_lower_of_the_two_ceilings_wins():
    sv = SafetySupervisor(brightness_max=0.02)
    out = sv.clamp(StimAction(enabled=True, steady=True, brightness=0.10))
    assert out.brightness == pytest.approx(0.02)


def test_the_policy_cannot_command_steady_light():
    sv = SafetySupervisor()
    out = decode_action_for_deployment([1.0, 1.0, 1.0, 0.0, 0.0], sv)
    assert out.steady is False


def test_panic_extinguishes_steady_light():
    sv = SafetySupervisor()
    sv.clamp(StimAction(enabled=True, steady=True, brightness=0.05))
    out = sv.panic()
    assert out.brightness == 0.0 and out.enabled is False
