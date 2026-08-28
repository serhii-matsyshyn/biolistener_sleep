"""
Amplitude calibration from a Session A baseline block.

Band powers are absolute. Training ran on the neural mass model's own units
and hardware arrives in microvolts, so the seven power features in the
observation land far outside the range the policy was trained on unless the
window is divided through by a constant first. Theta/alpha and PLV are ratios
and are unaffected either way.

The constant is measured, not assumed: this reads the eyes-closed baseline of
a Session A recording, compares its broadband RMS against the training
distribution, and prints the scale to pass to Session B as
``--amplitude-scale``. Session A therefore runs before Session B: it produces
a number Session B requires.

Usage:
    python tools/calibrate_amplitude.py data/sessions/<session-A-dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eeg.features import _filter_chain                    # noqa: E402
from app.protocol import analysis as an                       # noqa: E402

# RMS of the training signal *after* the production filter chain, measured over
# the domain-randomised cohort (median of 8 subjects, 60 s each; range
# 0.57-0.82). It has to be the filtered figure: detrend plus the 3-45 Hz
# bandpass takes the model's output from 7.4 to 0.79, and the band powers the
# policy reads come from the filtered signal. Comparing an unfiltered
# measurement against this would be wrong by an order of magnitude.
TRAINING_RMS = 0.786

# Note this is measured, not derived. The scale comes from the RMS of an actual
# recording, so whatever the amplifier's front end, gain setting and BrainFlow's
# code-to-microvolt conversion do between them is absorbed into one number. No
# datasheet constant enters, and getting the ADC's nominal full scale wrong
# cannot bias it - the plausibility check below is what catches a scale that is
# wrong for some other reason.

# Baseline blocks in preference order, eyes closed and unstimulated, so the
# measurement is of the subject and the montage rather than of the strobe.
# Session A's names come first: it is the session this is meant to be run on,
# and its blocks are not called anything like Session B's.
BASELINE_BLOCKS = (
    "A2b_eyes_closed", "A2d_eyes_closed", "A3a_alpha_rest", "A1_settle",
    "B0_baseline",
)

# Filtered occipital EEG sits around 5-40 uV RMS, so a plausible factor lands
# roughly between 6 and 50. Outside this window the recording is not the kind
# of signal the policy was trained on, and a scale factor would paper over a
# montage, gain or electrode problem rather than correct anything.
PLAUSIBLE_RANGE = (2.0, 100.0)

# Theta/alpha is a ratio, so no amplitude calibration touches it. That makes it
# the one observation element which cannot be corrected if the real spectrum
# differs from the model's - and it does, because the model has almost no 1/f
# background. It can still be *checked*: these are the 1st and 99th percentiles
# of theta/alpha over full-length training episodes, and a real recording that
# falls inside them is in-distribution regardless of the shape mismatch.
TRAINING_THETA_ALPHA_P1 = 0.0023
TRAINING_THETA_ALPHA_P99 = 14.47


def measure(session_dir: Path, block: str | None = None) -> dict:
    sess = an.load(session_dir)
    if sess.raw is None:
        raise SystemExit(f"{session_dir} has no raw recording")

    names = [b.get("name") for b in sess.manifest.get("blocks", [])]
    candidates = [block] if block else [b for b in BASELINE_BLOCKS if b in names]
    if not candidates:
        raise SystemExit(
            f"no baseline block found in {session_dir}; blocks are {names}")

    chosen, seg = None, None
    for name in candidates:
        try:
            seg = sess.segment(name)
            chosen = name
            break
        except Exception as err:
            print(f"  {name}: {err}")
    if seg is None:
        raise SystemExit("no baseline block was usable")

    filtered = seg.copy().astype(np.float64)
    _filter_chain(filtered, sess.sampling_rate)
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    scale = rms / TRAINING_RMS

    # Scale-free, so this is a check rather than a correction.
    tk = sess.ticks
    ta = float("nan")
    if tk is not None and "theta_alpha_ratio" in tk:
        d = tk[tk["block"] == chosen]["theta_alpha_ratio"]
        if len(d):
            ta = float(d.median())
    ta_ok = bool(np.isfinite(ta)
                 and TRAINING_THETA_ALPHA_P1 <= ta <= TRAINING_THETA_ALPHA_P99)
    return {
        "theta_alpha_median": None if not np.isfinite(ta) else round(ta, 4),
        "theta_alpha_in_training_range": ta_ok,
        "theta_alpha_training_range": [TRAINING_THETA_ALPHA_P1,
                                       TRAINING_THETA_ALPHA_P99],
        "session": str(session_dir),
        "block": chosen,
        "n_samples": int(seg.size),
        "seconds": round(seg.size / sess.sampling_rate, 1),
        "raw_row": int(sess.primary_channel),
        "measured_rms_uv": round(rms, 3),
        "training_rms": TRAINING_RMS,
        "amplitude_scale": round(scale, 4),
        "plausible": bool(PLAUSIBLE_RANGE[0] <= scale <= PLAUSIBLE_RANGE[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--block", help="override the baseline block name")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = measure(args.session_dir, args.block)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0 if out["plausible"] else 1

    print(f"\nbaseline block   {out['block']}  ({out['seconds']:g} s, "
          f"raw row {out['raw_row']})")
    print(f"measured RMS     {out['measured_rms_uv']:g} uV")
    print(f"training RMS     {out['training_rms']:g}")
    ta = out["theta_alpha_median"]
    print(f"theta/alpha      {ta if ta is None else f'{ta:g}'}"
          f"   training range {out['theta_alpha_training_range'][0]:g}"
          f"-{out['theta_alpha_training_range'][1]:g}"
          f"   {'in range' if out['theta_alpha_in_training_range'] else 'OUT OF RANGE'}")
    if not out["theta_alpha_in_training_range"]:
        print("  ! theta/alpha is a ratio, so no amplitude calibration moves it.\n"
              "    Outside the trained range the policy is reading a drowsiness\n"
              "    signal it never saw. Report Session B descriptively.")
    print(f"\n  amplitude_scale = {out['amplitude_scale']:g}")
    if out["plausible"]:
        print(f"\nPass to Session B:\n"
              f"  --amplitude-scale {out['amplitude_scale']:g}\n")
        return 0
    print(f"\n! outside {PLAUSIBLE_RANGE[0]:g}-{PLAUSIBLE_RANGE[1]:g}. A scale "
          f"this far off is a montage, gain or electrode problem, and dividing\n"
          f"  it out would hide that rather than fix it. Re-run Session A.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
