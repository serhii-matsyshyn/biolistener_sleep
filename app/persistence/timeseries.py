"""
Filesystem time-series persistence.

Per-session folder layout:

    studies/{subject_code}/{session_id}/
        raw_eeg.fif              # MNE-Python Raw (archival format)
        raw_eeg.csv              # BrainFlow add_streamer raw capture (live)
        stim_log.parquet         # high-res strobe + audio param timeline
        features.parquet         # per-tick feature snapshots
        audio_seed.json          # all RNG seeds for deterministic replay
        audio.wav                # optional rendered audio
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("persistence.timeseries")


def session_folder(studies_root: str | Path, subject_code: str,
                   session_id: str) -> Path:
    p = Path(studies_root) / subject_code / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_audio_seed(folder: Path, seed_doc: dict) -> Path:
    out = folder / "audio_seed.json"
    out.write_text(json.dumps(seed_doc, indent=2, default=str))
    return out


def append_parquet(folder: Path, filename: str, rows: List[dict]) -> Path:
    """
    Append rows to a Parquet file. For simplicity we re-write the whole
    file on each call - fine for the <50k rows we produce per session.
    For very long sessions, switch to dataset writes per chunk.
    """
    if not rows:
        return folder / filename
    out = folder / filename
    if out.exists():
        existing = pq.read_table(out).to_pylist()
        rows = existing + rows
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)
    return out


def write_eeg_csv_to_fif(
    csv_path: Path,
    fif_path: Path,
    sampling_rate: int,
    eeg_channel_indices: List[int],
    eeg_channel_names: List[str],
) -> Path:
    """
    Convert BrainFlow's raw CSV (full board layout) into an MNE .fif Raw
    for archival. MNE .fif is the neuroscience-community standard and lets
    the data open in MNE-Python, EEGLAB, Brainstorm, etc.

    The CSV has 'rows = samples, cols = channels' (BrainFlow's format).
    """
    import mne

    # BrainFlow CSV is whitespace-separated and DOES NOT include a header.
    data = np.loadtxt(csv_path, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    arr = data.T   # (channels, samples)
    eeg_only = arr[eeg_channel_indices, :]
    # MNE expects volts; BrainFlow returns µV
    eeg_only_v = eeg_only * 1e-6
    info = mne.create_info(
        ch_names=eeg_channel_names,
        sfreq=float(sampling_rate),
        ch_types=["eeg"] * len(eeg_channel_names),
    )
    raw = mne.io.RawArray(eeg_only_v, info, verbose="WARNING")
    raw.save(str(fif_path), overwrite=True, verbose="WARNING")
    return fif_path


class SessionWriter:
    """
    Buffers stim / feature rows and flushes them to Parquet periodically.

    Designed to be called from the daemon's 4 Hz control tick. ``flush()``
    is also called on session end / panic to guarantee no data loss.
    """

    def __init__(self, folder: Path, flush_every: int = 100):
        self.folder = folder
        self.flush_every = flush_every
        self._stim_buf: List[dict] = []
        self._feat_buf: List[dict] = []
        self._audio_buf: List[dict] = []

    def add_stim(self, row: dict) -> None:
        self._stim_buf.append(row)
        if len(self._stim_buf) >= self.flush_every:
            self.flush_stim()

    def add_features(self, row: dict) -> None:
        self._feat_buf.append(row)
        if len(self._feat_buf) >= self.flush_every:
            self.flush_features()

    def add_audio_event(self, row: dict) -> None:
        self._audio_buf.append(row)
        if len(self._audio_buf) >= self.flush_every:
            self.flush_audio()

    def flush_stim(self) -> None:
        if self._stim_buf:
            append_parquet(self.folder, "stim_log.parquet", self._stim_buf)
            self._stim_buf.clear()

    def flush_features(self) -> None:
        if self._feat_buf:
            append_parquet(self.folder, "features.parquet", self._feat_buf)
            self._feat_buf.clear()

    def flush_audio(self) -> None:
        if self._audio_buf:
            append_parquet(self.folder, "audio_events.parquet", self._audio_buf)
            self._audio_buf.clear()

    def flush_all(self) -> None:
        self.flush_stim()
        self.flush_features()
        self.flush_audio()
