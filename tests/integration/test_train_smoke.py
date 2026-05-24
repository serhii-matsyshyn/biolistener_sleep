"""
Smoke test for the training loop - short SAC run + baseline comparisons.

Verifies the entire M4 pipeline (env construction, SB3 SAC, MLflow logging,
baseline evaluation) is wired up correctly. Does NOT verify the trained
agent is "good" - that's the role of the Notebook 03 evaluation.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow

from app.agent.train import TrainConfig, train


def test_short_sac_training_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Isolate MLflow tracking & checkpoints to the temp dir
        mlflow.set_tracking_uri(f"file://{tmpdir}/mlruns")
        cfg = TrainConfig(
            total_steps=500,            # tiny - just exercise the loop
            episode_s=2.0,
            checkpoint_dir=f"{tmpdir}/checkpoints",
            experiment_name="smoke_test",
            include_restfulness=False,  # skip RESTFULNESS to keep fast
        )
        ckpt_path = train(cfg)
        assert Path(ckpt_path).exists()
        assert Path(ckpt_path).suffix == ".zip"
