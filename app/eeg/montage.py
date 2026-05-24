"""
Electrode montage definitions for BioListener's 8-channel layout.

Two reference montages backed by the existing work in this lab:

- **OCCIPITAL_ALPHA** (bachelor §5.4.3 validation montage):
    O1 -> CH0 (occipital - strongest endogenous α)
    A1 -> ground (left ear)
    A2 -> reference (right ear)
    Fp2 -> CH1 (EOG / eye-blink EMG)
    Remaining channels free for auxiliary placements.

- **FPZ_MASTOID** (master's-proposal montage):
    Fpz -> CH0 (frontal - captures frontal α + θ)
    Mastoid -> reference
    Plus Fp1/Fp2 for EOG, F3/F4 optional.

Montage selection drives:
- Which channel index is treated as the "primary EEG" for reward computation
- Which channels are EMG/EOG (excluded from α reward)
- Which channels are auxiliary (still logged but not in the agent's observation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Montage:
    name: str
    primary_eeg_channels: List[int]   # indices used for α/θ band power reward
    emg_channels: List[int]           # indices for muscle-tension EMG RMS
    eog_channels: List[int]           # indices for eye-movement EOG energy
    notes: str = ""

    def all_active(self) -> List[int]:
        return sorted(set(self.primary_eeg_channels + self.emg_channels + self.eog_channels))


OCCIPITAL_ALPHA = Montage(
    name="occipital_alpha",
    primary_eeg_channels=[0],       # O1
    emg_channels=[],                # no dedicated EMG channel
    eog_channels=[1],               # Fp2
    notes=("Bachelor validation montage. O1-A1-A2 captures eyes-closed "
           "α dropout reliably; Fp2 picks up eye blinks for EOG energy."),
)

FPZ_MASTOID = Montage(
    name="fpz_mastoid",
    primary_eeg_channels=[0],       # Fpz
    emg_channels=[],
    eog_channels=[1, 2],            # Fp1, Fp2
    notes=("Master's-proposal montage. Fpz-Mastoid captures frontal α + θ "
           "without occipital coverage but is more comfortable for sleep."),
)

ALL: Dict[str, Montage] = {
    OCCIPITAL_ALPHA.name: OCCIPITAL_ALPHA,
    FPZ_MASTOID.name: FPZ_MASTOID,
}


def by_name(name: str) -> Montage:
    if name not in ALL:
        raise KeyError(
            f"Unknown montage {name!r}. Available: {list(ALL)}"
        )
    return ALL[name]
