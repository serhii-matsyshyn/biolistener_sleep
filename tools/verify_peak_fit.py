"""
Verify that the sweep analysis recovers a resonance peak it is given.

A full-length rehearsal exercises the pipeline, but it cannot verify the *fit*:
the plant's peak is driven by measured drowsiness, and in the operating regime
that saturates almost immediately - it reaches the floor once theta/alpha passes
about 3.5, while training rollouts average 1.7 and reach 42. So a session tends
to park the peak at one end and leave the fit nothing to track.

This does the verification directly. The peak is *set* to a known value, a real
sweep is run through the neural mass model at that setting, and the fitted peak
is compared against the value the plant was actually using. Repeated across the
probe range it gives the fit's bias and its interval coverage - the numbers that
decide whether the interval quoted beside the results figure is honest.

    python3 tools/verify_peak_fit.py                    # default grid
    python3 tools/verify_peak_fit.py --targets 6 8 10 --repeats 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.brain_sim.brainflow_board import InProcessBrainSim          # noqa: E402
from app.brain_sim.domain_random import RandomizationRanges          # noqa: E402
from app.brain_sim.photic_input import StrobeParams                  # noqa: E402
from app.eeg.montage import ALL as MONTAGES                          # noqa: E402
from app.protocol import quality as q                                # noqa: E402
from app.protocol.analysis import (                                  # noqa: E402
    ProbeResponse, ResponseCurve, bootstrap_peak,
)
from app.protocol.spec import BRIEF_DWELL_S, BRIEF_PROBE_HZ          # noqa: E402


def _plant(target_hz: float, subject_seed: int):
    """A subject whose resonance sits at ``target_hz`` and stays there."""
    subj = RandomizationRanges(exogenous_shift=False).sample(subject_seed)
    cfg = subj.photic_config
    # Invert effective_resonance_hz. Held fixed by disabling vigilance drift, so
    # the peak cannot wander while the sweep is running.
    cfg.current_drowsiness = (cfg.shift_baseline_ratio
                              + (cfg.base_resonance_hz - target_hz)
                              / cfg.shift_per_unit_ratio)
    sim = InProcessBrainSim(photic_config=cfg, vigilance_drift_enabled=False)
    return sim, float(cfg.effective_resonance_hz)


def _plv(seg: np.ndarray, fs: int, strobe) -> float:
    from app.brain_sim.photic_input import strobe_envelope
    from app.eeg.plv import plv_strobe_to_eeg
    env = strobe_envelope(duration_s=seg.size / fs, dt=1.0 / fs, strobe=strobe)
    return float(plv_strobe_to_eeg(seg, env[:seg.size], fs, strobe.frequency_hz))


def _metric(seg: np.ndarray, fs: int, f: float, metric: str, strobe=None) -> float:
    """
    How strongly the EEG followed the probe.

    ``amplitude`` is raw spectral power at the probe frequency, which also
    contains whatever the brain was doing there anyway - near the subject's own
    alpha peak that endogenous power dominates and drags the fit toward it.
    ``snr`` divides by the neighbouring bins, so a broad endogenous hump
    cancels and only a narrow driven line survives.
    """
    if metric == "plv" and strobe is not None:
        return _plv(seg, fs, strobe)
    psd = q.psd_welch(seg, fs)
    if metric == "snr":
        v = q.peak_snr_db(psd, f)
        # A fitted peak needs a positive quantity to interpolate through.
        return float(max(v, 0.0)) if np.isfinite(v) else 0.0
    return q.peak_amplitude(psd, f)


def sweep_once(target_hz: float, subject_seed: int, dwell_s: float,
               probes, fs: int, settle_s: float = 1.0, metric: str = "amplitude",
               epoch_s: float = 1.0):
    sim, actual = _plant(target_hz, subject_seed)
    montage = MONTAGES["occipital_alpha"]
    ch = montage.primary_eeg_channels[0]
    responses = []
    for f in probes:
        sim.update_strobe(StrobeParams(enabled=True, frequency_hz=float(f),
                                       fill=0.4, brightness=0.6))
        n_settle = int(settle_s * fs)
        n_keep = int(dwell_s * fs)
        for _ in range(n_settle):
            sim.step()
        seg = np.array([sim.step() for _ in range(n_keep)]).T[ch]
        # PLV is filtered in a ~2 Hz band, which needs about half a second to
        # settle; a one-second epoch is mostly transient, and the leftover
        # biases the curve toward the high probes.
        step = max(64, int(epoch_s * fs))
        st = StrobeParams(enabled=True, frequency_hz=float(f), fill=0.4,
                          brightness=0.6)
        amps = [_metric(seg[a:a + step], fs, float(f), metric, st)
                for a in range(0, seg.size - step + 1, step)]
        if not amps:
            amps = [_metric(seg, fs, float(f), metric, st)]
        psd = q.psd_welch(seg, fs)
        responses.append(ProbeResponse(
            probe_hz=float(f), snr_db=q.peak_snr_db(psd, float(f)),
            amplitude=_metric(seg, fs, float(f), metric, st),
            n_epochs=len(amps), epoch_amplitudes=np.array(amps)))
    return ResponseCurve(block=f"target_{target_hz:g}", probes=tuple(responses)), actual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--subject-seed", type=int, default=2)
    ap.add_argument("--dwell", type=float, default=BRIEF_DWELL_S)
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--probes", type=float, nargs="+",
                    help="probe grid to test (default: the session grid)")
    ap.add_argument("--epoch", type=float, default=1.0,
                    help="seconds per bootstrap epoch")
    ap.add_argument("--metric", choices=("plv", "amplitude", "snr"), default="plv",
                    help="what counts as 'response' at a probe frequency")
    args = ap.parse_args()

    print(f"\n  probes {list(args.probes or BRIEF_PROBE_HZ)}  dwell {args.dwell:g} s  "
          f"subject seed {args.subject_seed}  metric {args.metric}\n")
    print(f"  {'target':>7} {'actual':>7} {'fitted':>7} {'error':>7} "
          f"{'95% interval':>18} {'covered':>8}")
    errors, covered = [], []
    for target in args.targets:
        for rep in range(args.repeats):
            grid = tuple(args.probes) if args.probes else BRIEF_PROBE_HZ
            curve, actual = sweep_once(target, args.subject_seed + rep * 100,
                                       args.dwell, grid, args.fs,
                                       metric=args.metric, epoch_s=args.epoch)
            boot = bootstrap_peak(curve, n=600, seed=rep)
            fitted = boot["peak_hz"]
            if not np.isfinite(fitted):
                print(f"  {target:>7.2f} {actual:>7.2f}   no fit")
                continue
            err = fitted - actual
            hit = boot["lo"] <= actual <= boot["hi"]
            errors.append(err)
            covered.append(hit)
            print(f"  {target:>7.2f} {actual:>7.2f} {fitted:>7.2f} {err:>+7.2f} "
                  f"  [{boot['lo']:>6.2f},{boot['hi']:>6.2f}] {str(hit):>8}")

    if errors:
        e = np.array(errors)
        print(f"\n  n={len(e)}   bias {e.mean():+.3f} Hz   "
              f"|error| {np.abs(e).mean():.3f} Hz   "
              f"coverage {np.mean(covered):.0%}")
        cov = float(np.mean(covered))
        if cov >= 0.95:
            print("\n  Coverage is at nominal: the interval quoted beside the results")
            print("  figure holds up against a plant whose peak is known.")
        else:
            print(f"\n  Coverage {cov:.0%} is below nominal - the interval quoted beside")
            print("  the results figure understates the uncertainty. Raise")
            print("  PEAK_ACCURACY_HZ in app/protocol/analysis.py and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
