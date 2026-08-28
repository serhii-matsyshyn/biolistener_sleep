"""
Electrode montage definitions for BioListener's 8-channel layout.

Two reference montages backed by the existing work in this lab:

- **OCCIPITAL_ALPHA** (validation montage):
    O1 -> CH0 (occipital - strongest endogenous α and photic driving)
    Fp2 -> CH1 (EOG / eye-blink)
    M2 -> reference, M1 -> bias. Reference contralateral to the primary, so it
    subtracts less of the signal being measured.
    Remaining channels not connected and disabled at session start.

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
    notes=("O1 referenced to the contralateral mastoid (M2), bias on M1. "
           "Eyes-closed α dropout is largest occipitally, and so is the "
           "photic driving response. Fp2 supplies EOG energy only."),
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
