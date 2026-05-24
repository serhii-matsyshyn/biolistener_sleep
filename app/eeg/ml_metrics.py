"""
Auxiliary BrainFlow MLModel scores - RESTFULNESS only.

We deliberately exclude MINDFULNESS: that classifier targets meditative
β/γ activity, which is the OPPOSITE direction of falling asleep, so it
would add noise to the agent observation without lawful interpretation.

RESTFULNESS empirically tracks α-dominance, which IS the eye-closed-onset
state we care about (verified by feeding the model a pure 10 Hz sine ->
RESTFULNESS = 1.0). It's included as an *auxiliary* observation feature
and shown on the live monitor as a researcher-friendly gauge, but the
reward function never depends on it.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter
from brainflow.ml_model import (
    BrainFlowClassifiers,
    BrainFlowMetrics,
    BrainFlowModelParams,
    MLModel,
)

logger = logging.getLogger("eeg.ml_metrics")


class RestfulnessMetric:
    """
    Wraps BrainFlow's MLModel(RESTFULNESS) with a clean lifecycle.

    Usage::

        with RestfulnessMetric() as r:
            score = r.score(data, eeg_channels, sampling_rate)
    """

    def __init__(self):
        self._params = BrainFlowModelParams(
            BrainFlowMetrics.RESTFULNESS.value,
            BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
        )
        self._model: Optional[MLModel] = None

    def __enter__(self) -> "RestfulnessMetric":
        self.prepare()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def prepare(self) -> None:
        if self._model is None:
            self._model = MLModel(self._params)
            self._model.prepare()

    def release(self) -> None:
        if self._model is not None:
            try:
                self._model.release()
            finally:
                self._model = None

    def score(
        self,
        data: np.ndarray,
        eeg_channels: List[int],
        sampling_rate: int,
    ) -> float:
        """
        Returns RESTFULNESS in [0, 1]. Higher = more α-dominant /
        relaxed-eyes-closed-like. Returns 0.0 if window is too short.
        """
        if self._model is None:
            raise RuntimeError("RestfulnessMetric not prepared")
        if data.shape[1] < sampling_rate * 2:
            # MLModel needs >=2s for stable band-power estimate
            return 0.0
        try:
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True,
            )
            feature_vector = bands[0]
            result = self._model.predict(feature_vector)
            return float(result[0]) if len(result) > 0 else 0.0
        except Exception as err:
            logger.debug("RESTFULNESS predict failed: %s", err)
            return 0.0


def restfulness_score(
    data: np.ndarray,
    eeg_channels: List[int],
    sampling_rate: int,
) -> float:
    """One-shot convenience - opens & closes the model. Slow for repeated use."""
    with RestfulnessMetric() as r:
        return r.score(data, eeg_channels, sampling_rate)
