"""
Regenerate the spoken cue files. Authoring step - not needed to run a session.

The rendered WAVs are committed under ``data/cues/en/``, so a session needs no
speech engine, no system packages and no network. Run this only after editing a
phrase in ``app.protocol.cues.CUES``.

    python3 tools/render_cues.py --list                  # show the catalogue
    python3 tools/render_cues.py --check                 # what is missing
    python3 tools/render_cues.py --voice <model.onnx>    # re-render

Rendering needs Piper, which is not a project dependency:

    pip install piper-tts
    python3 -m piper.download_voices en_US-lessac-medium --download-dir /tmp/piper
    python3 tools/render_cues.py --voice /tmp/piper/en_US-lessac-medium.onnx

Existing files are kept unless ``--force``, so a phrase recorded by hand in the
subject's own voice is never overwritten by a synthesised one.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.protocol import cues as cue_mod   # noqa: E402


def render(voice_path: Path, out_dir: Path, force: bool) -> dict:
    from piper import PiperVoice

    voice = PiperVoice.load(str(voice_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    made = kept = 0
    for cue_id, text in sorted(cue_mod.CUES.items()):
        dest = out_dir / f"{cue_id}.wav"
        if dest.exists() and not force:
            kept += 1
            continue
        with wave.open(str(dest), "wb") as wav:
            voice.synthesize_wav(text, wav)
        made += 1
    return {"made": made, "kept": kept, "dir": out_dir,
            "voice": voice_path.name}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", type=Path, help="path to a Piper .onnx voice")
    ap.add_argument("--out", default=str(cue_mod.CUE_ROOT / cue_mod.DEFAULT_LANG))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.list:
        for cue_id, text in sorted(cue_mod.CUES.items()):
            print(f"  {cue_id:24} {text}")
        return 0

    if args.check or not args.voice:
        player = cue_mod.CuePlayer()
        missing = player.missing_cues()
        print(f"  {len(cue_mod.CUES)} cues, {len(missing)} missing "
              f"from {player.path_for('x').parent}")
        for c in missing:
            print(f"    missing: {c}")
        if not args.voice:
            print("\n  pass --voice <model.onnx> to render (see --help)")
        return 0 if not missing else 1

    r = render(args.voice, Path(args.out), args.force)
    print(f"  {r['made']} rendered, {r['kept']} kept  ->  {r['dir']}  "
          f"(voice {r['voice']})")
    print("  listen through them once before the first session - they are the "
          "only channel the subject has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
