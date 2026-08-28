"""
Session replay - re-render a recorded session deterministically.

Reads the per-session folder produced by the live daemon and reconstructs:

1. The exact stim timeline that was applied to the mask (replays the
   CSV through a virtual mask's visualizer for video capture).
2. The features the agent observed (re-running BrainFlow's PSD pipeline
   against the recorded raw_eeg.csv via PLAYBACK_FILE_BOARD).
3. Optionally a deterministic re-render of the audio via the recorded
   audio_seed.json - pyo Server in render mode (`recordOptions`) outputs
   audio.wav byte-identical to what played live.

Re-rendering the same session twice produces the same output.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("runtime.replay")


@dataclass
class ReplayBundle:
    """Output of :func:`load_session` - all data needed for re-rendering."""
    folder: Path
    raw_eeg_csv: Optional[Path]
    stim_log: List[dict]
    features: List[dict]
    audio_seed: dict
    sampling_rate: int = 500


def load_session(folder: Path) -> ReplayBundle:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    raw_csv = folder / "raw_eeg.csv"
    raw_csv = raw_csv if raw_csv.exists() else None

    stim = []
    stim_pq = folder / "stim_log.parquet"
    if stim_pq.exists():
        stim = pq.read_table(stim_pq).to_pylist()

    feats = []
    feat_pq = folder / "features.parquet"
    if feat_pq.exists():
        feats = pq.read_table(feat_pq).to_pylist()

    seed_p = folder / "audio_seed.json"
    seed = json.loads(seed_p.read_text()) if seed_p.exists() else {}

    return ReplayBundle(
        folder=folder,
        raw_eeg_csv=raw_csv,
        stim_log=stim,
        features=feats,
        audio_seed=seed,
    )


def iter_stim_events(bundle: ReplayBundle) -> Iterator[dict]:
    """Iterate stim events in time order for replay through a mask."""
    rows = sorted(bundle.stim_log, key=lambda r: r.get("t_ms", 0))
    for r in rows:
        yield r


def reissue_to_virtual_mask(
    bundle: ReplayBundle,
    listen_port: int = 22000,
    controller_port: int = 22001,
    speed: float = 1.0,
) -> None:
    """
    Replay the stim timeline through a virtual mask.

    Useful for: video capture of the strobe pattern offscreen, sanity
    checking the recorded timeline, A/B-ing a different visualizer.
    """
    import time
    from app.mask.client import MaskClient
    from app.mask.virtual import VirtualMaskServer

    vm = VirtualMaskServer(
        listen_ip="127.0.0.1", listen_port=listen_port,
        controller_addr=("127.0.0.1", controller_port),
        ping_interval_s=0.5,
    )
    vm.start()
    client = MaskClient(bind_ip="127.0.0.1", port=controller_port,
                        target_ip="127.0.0.1")
    client.port = listen_port

    try:
        prev_t = None
        for row in iter_stim_events(bundle):
            t_ms = row.get("t_ms", 0)
            if prev_t is not None:
                gap = max(0, (t_ms - prev_t) / 1000.0 / max(0.01, speed))
                time.sleep(gap)
            prev_t = t_ms
            client.update_params(
                enabled=row.get("brightness", 0) > 0,
                frequency=row.get("mask_freq_hz"),
                fill=row.get("duty"),
                brightness=row.get("brightness"),
                color=[0, 0, 0, int(row.get("color_white", 200))],
            )
    finally:
        client.stop()
        vm.stop()
        vm.join(timeout=2)


def replay_eeg_through_playback_board(bundle: ReplayBundle) -> "object":
    """
    Open a BrainFlow PLAYBACK_FILE_BOARD against the recorded raw_eeg.csv.

    The returned BoardShim can be passed directly into the same downstream
    pipeline (EEGIngestor.latest_window, extract_features, policy) used
    in live operation - so "what would policy v4 have done on this
    session?" is a one-liner.
    """
    from brainflow.board_shim import (
        BoardIds, BoardShim, BrainFlowInputParams,
    )
    if bundle.raw_eeg_csv is None:
        raise FileNotFoundError("Bundle has no raw_eeg.csv to replay")
    params = BrainFlowInputParams()
    params.file = str(bundle.raw_eeg_csv)
    params.master_board = BoardIds.BIOLISTENER_BOARD.value
    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD.value, params)
    board.prepare_session()
    board.start_stream(450_000)
    return board
