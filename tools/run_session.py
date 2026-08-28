"""
Run a hardware session.

    # rehearse the whole thing against the simulator, no hardware, 30 s
    python3 tools/run_session.py --protocol validation --compress 0.03

    # the real validation session
    python3 tools/run_session.py --protocol validation --eeg biolistener \
        --mask 192.0.2.11

    # the experimental session; --order defaults to auto, alternating
    # ABBA and BAAB across nights so you never have to remember
    python3 tools/run_session.py --protocol full \
        --eeg biolistener --mask 192.0.2.11 \
        --checkpoint app/agent/checkpoints/<300k-full_shaped-seed>.zip

Pre-flight runs before anything is recorded: block plan, settling warnings,
missing cues, and whether a checkpoint is needed and present. ``--dry-run``
stops after it.

Press Enter at any time to stop the session cleanly.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.protocol import cues as cue_mod                       # noqa: E402
from app.protocol.runner import RunnerConfig, SessionRunner    # noqa: E402
from app.protocol.spec import (
    DEFAULT_CHECKPOINT,                                # noqa: E402
    ARM_ORDERS, BlockKind, build_full, build_validation, compress,
    order_history, settling_report, suggest_order,
)


def resolve_order(args) -> str:
    """``auto`` alternates ABBA/BAAB by looking at what already completed."""
    if args.order != "auto":
        return args.order
    chosen = suggest_order(args.session_root)
    hist = order_history(args.session_root)
    print(f"\n  arm order : {chosen}  (auto; completed so far "
          f"{', '.join(f'{k}={v}' for k, v in hist.items())})")
    return chosen


def build(args) -> object:
    if args.protocol == "validation":
        proto = build_validation(seed=args.seed, montage=args.montage)
    else:
        proto = build_full(order=resolve_order(args), seed=args.seed,
                           montage=args.montage)
    if args.compress < 1.0:
        proto = compress(proto, args.compress)
    return proto


def preflight(proto, args) -> bool:
    print(f"\n  protocol : {proto.name}")
    print(f"  duration : {proto.total_duration_s/60:.1f} min "
          f"({len(proto.blocks)} blocks)")
    print(f"  montage  : {proto.montage}   seed: {proto.seed}")
    print(f"  eeg      : {args.eeg}        mask: {args.mask}")
    print(f"  cues     : {args.lang}" + ("" if args.cues else "  (disabled)"))
    print()
    for start, end, b in proto.boundaries():
        flag = " KSS" if b.kss_at_end else ""
        print(f"    {start/60:5.1f}-{end/60:5.1f} min  {b.name:22} "
              f"{b.kind.value:14} eyes={b.eyes.value}{flag}")

    ok = True
    warns = settling_report(proto)
    if warns:
        print("\n  settling warnings:")
        for w in warns:
            print(f"    ! {w}")
        if args.compress >= 1.0:
            ok = False

    # An uncalibrated hardware session does not merely handicap the agent: at
    # microvolt scale the absolute-power reward term swamps the scale-free
    # ones, so the primary outcome stops measuring what it claims to.
    if (any(b.kind == BlockKind.DESCENT_AGENT for b in proto.blocks)
            and args.eeg != "sim" and not args.amplitude_scale):
        print("\n  ! --amplitude-scale is unset for a hardware session with an "
              "agent block.\n    Run tools/calibrate_amplitude.py on the "
              "Session A recording first.")
        ok = False

    needs_ckpt = any(b.kind == BlockKind.DESCENT_AGENT for b in proto.blocks)
    if needs_ckpt:
        if not args.checkpoint:
            print("\n  ! this protocol has an agent block but --checkpoint is unset")
            ok = False
        elif not Path(args.checkpoint).exists():
            print(f"\n  ! checkpoint not found: {args.checkpoint}")
            ok = False

    if args.cues:
        player = cue_mod.CuePlayer(lang=args.lang, root=Path(args.cue_root))
        missing = player.missing_cues()
        if missing:
            print(f"\n  ! {len(missing)} cue file(s) missing under "
                  f"{player.path_for('x').parent}: {', '.join(missing[:4])}")
            print("    see: python3 tools/render_cues.py --help")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("validation", "full"), default="validation")
    ap.add_argument("--order", default="auto",
                    choices=("auto", *ARM_ORDERS, "ramp_first", "agent_first"),
                    help="auto alternates ABBA/BAAB across nights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage", default="occipital_alpha")
    ap.add_argument("--eeg", default="sim",
                    choices=("sim", "biolistener", "playback"))
    ap.add_argument("--eeg-ip", default="0.0.0.0")
    ap.add_argument("--eeg-port", type=int, default=12345)
    ap.add_argument("--playback-file", type=Path)
    ap.add_argument("--mask", default="virtual",
                    help="'virtual' for the emulator, or 'esp32' for a real "
                         "mask found by ping discovery")
    ap.add_argument("--mask-target-ip",
                    help="pin the mask's address instead of waiting for its "
                         "pings; this is the device's IP, not a local one")
    ap.add_argument("--mask-port", type=int, default=None,
                    help="controller port; default 20000 for a real "
                         "mask, a per-process pair for a virtual one")
    ap.add_argument("--pga-gain", type=int, default=8,
                    help="amplifier gain the device is running at; the railed "
                         "detector reconstructs ADC full scale from it")
    ap.add_argument("--amplitude-scale", type=float,
                    help="microvolts per training unit, from "
                         "tools/calibrate_amplitude.py on the Session A "
                         "recording. Required for a real Session B: without "
                         "it the policy sees band powers it never trained on")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(DEFAULT_CHECKPOINT),
                    help="policy archive for the agent blocks")
    ap.add_argument("--lang", choices=cue_mod.LANGS, default="en")
    ap.add_argument("--cue-root", default=str(cue_mod.CUE_ROOT))
    ap.add_argument("--no-cues", dest="cues", action="store_false")
    ap.add_argument("--tick-hz", type=float, default=4.0)
    ap.add_argument("--compress", type=float, default=1.0,
                    help="shorten every block; rehearsal only")
    ap.add_argument("--session-root", default="data/sessions")
    ap.add_argument("--sim-subject-seed", type=int,
                    help="which simulated subject to rehearse against "
                         "(default: --seed; note seed 0 is a near-non-responder)")
    ap.add_argument("--sim-exogenous-shift", action="store_true",
                    help="rehearsal only: drive the simulated resonance peak "
                         "from a fixed time course, so peak recovery can be "
                         "verified against a peak that actually moves")
    ap.add_argument("--phase-restarting", dest="phase_continuous",
                    action="store_false", default=None,
                    help="the mask restarts its strobe phase on every retune; "
                         "run driver_phase_model on Session A to find out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    proto = build(args)
    ok = preflight(proto, args)
    if args.dry_run:
        print("\n  dry run - nothing recorded\n")
        return 0 if ok else 1
    if not ok:
        print("\n  pre-flight failed; fix the above or pass --dry-run to inspect\n")
        return 1

    cfg = RunnerConfig(
        protocol=proto, lang=args.lang, tick_hz=args.tick_hz,
        eeg_kind=args.eeg, eeg_ip=args.eeg_ip, eeg_port=args.eeg_port,
        playback_file=args.playback_file,
        # The controller binds a local interface and *targets* the device.
        # Binding the device's own address fails with EADDRNOTAVAIL.
        mask_ip="127.0.0.1" if args.mask == "virtual" else "0.0.0.0",
        mask_port=args.mask_port,
        mask_target_ip=("127.0.0.1" if args.mask == "virtual"
                        else args.mask_target_ip),
        virtual_mask=args.mask == "virtual",
        pga_gain=args.pga_gain,
        amplitude_scale=args.amplitude_scale,
        phase_continuous=args.phase_continuous,
        checkpoint=args.checkpoint, audio_cues=args.cues,
        cue_root=Path(args.cue_root), session_root=Path(args.session_root),
        sim_exogenous_shift=args.sim_exogenous_shift,
        sim_subject_seed=args.sim_subject_seed,
    )

    print("\n  starting - press Enter at any time to stop\n")
    out = SessionRunner(cfg).run()
    print(f"\n  recorded to {out}")
    print(f"  next: transcribe the spoken ratings into {out/'kss.csv'} "
          f"and fill in {out/'notes.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
