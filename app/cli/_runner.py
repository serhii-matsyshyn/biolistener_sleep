"""
Glue between Typer commands and the existing daemon / training code.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import uvicorn

from app.config import BioListenerSleepConfig, dump_yaml

logger = logging.getLogger("cli.runner")


def _propagate_legacy_env(cfg: BioListenerSleepConfig) -> None:
    """
    Mirror selected config fields into env vars so legacy callsites
    see the resolved values.
    """
    os.environ["MONGO_URL"] = cfg.persistence.mongo_url
    os.environ["MONGO_DB"] = cfg.persistence.mongo_db
    if cfg.audio.enabled is not None:
        os.environ["BIOLISTENER_SLEEP_ENABLE_AUDIO"] = "1" if cfg.audio.enabled else "0"


def _build_eeg_source_spec(cfg: BioListenerSleepConfig) -> str:
    kind = cfg.eeg.kind
    if kind == "sim":
        return "sim"
    if kind == "biolistener":
        return f"biolistener:{cfg.eeg.ip_address}:{cfg.eeg.ip_port}"
    if kind == "playback":
        return f"playback:{cfg.eeg.playback_file}"
    return "sim"


def _build_mask_target_spec(cfg: BioListenerSleepConfig) -> str:
    if cfg.mask.kind == "esp32":
        return f"esp32:{cfg.mask.ip_address}:{cfg.mask.port}"
    return "virtual"


def _build_protocol_params(cfg: BioListenerSleepConfig) -> dict:
    """
    Build the protocol_params dict the daemon's `_setup_policy` expects.

    Priority: explicit cfg.session.protocol_params wins; otherwise we
    derive sensible defaults from the per-protocol config sections.
    """
    params = dict(cfg.session.protocol_params or {})
    if cfg.session.protocol == "rl_agent" and "brightness" not in params:
        params.setdefault("brightness", cfg.policy.fixed_brightness)
    return params


async def run_session_from_config(cfg: BioListenerSleepConfig) -> None:
    """Run one session end-to-end via the existing daemon control loop."""
    _propagate_legacy_env(cfg)
    from app.runtime import daemon

    daemon.STUDIES_ROOT = cfg.persistence.studies_root
    daemon.STUDIES_ROOT.mkdir(parents=True, exist_ok=True)
    # Stash for the daemon's _setup_policy fallback (used when SAC isn't loaded)
    daemon.RESOLVED_CONFIG = cfg

    resp = await daemon.create_session({
        "subject_code": cfg.session.subject_code,
        "protocol": cfg.session.protocol,
        "protocol_params": _build_protocol_params(cfg),
        "eeg_source": _build_eeg_source_spec(cfg),
        "mask_target": _build_mask_target_spec(cfg),
        "audio_enabled": cfg.session.audio_enabled,
    })
    sid = resp["id"]
    folder = Path(resp["folder"])
    logger.info("Created session %s at %s (protocol=%s)",
                sid, folder, cfg.session.protocol)

    # Snapshot resolved config for replay-determinism / audit trail
    (folder / "config.yaml").write_text(dump_yaml(cfg))

    await daemon.start_session(sid)
    try:
        await asyncio.sleep(cfg.session.duration_s)
    finally:
        await daemon.end_session(sid)
    logger.info("Session %s ended", sid)


def train_from_config(cfg: BioListenerSleepConfig) -> str:
    """Run SAC training using values from cfg.training + cfg.env + cfg.mlflow."""
    _propagate_legacy_env(cfg)
    from app.agent.reward import RewardWeights
    from app.agent.train import TrainConfig, train

    if cfg.mlflow.tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Reward shaping. If ``reward.variant`` is set we let train() resolve
    # the registry (a single source of truth, registry tags MLflow and
    # encodes the variant in the checkpoint name). Otherwise we build
    # explicit weights from the individual k_* fields.
    rw_cfg = cfg.reward
    if rw_cfg.variant is None:
        explicit_weights = RewardWeights(
            k_theta_alpha=rw_cfg.k_theta_alpha,
            k_theta_alpha_level=rw_cfg.k_theta_alpha_level,
            k_hypnagogic=rw_cfg.k_hypnagogic,
            k_plv=rw_cfg.k_plv,
            k_descent=rw_cfg.k_descent,
            k_disconnect=rw_cfg.k_disconnect,
            k_alpha_pwr=rw_cfg.k_alpha_pwr,
            flat_plv=rw_cfg.flat_plv,
        )
        reward_variant = None
    else:
        explicit_weights = None       # train() will resolve from variant
        reward_variant = rw_cfg.variant

    tcfg = TrainConfig(
        total_steps=cfg.training.total_steps,
        episode_s=cfg.training.episode_s,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        buffer_size=cfg.training.buffer_size,
        gamma=cfg.training.gamma,
        tau=cfg.training.tau,
        ent_coef=cfg.training.ent_coef,
        seed=cfg.training.seed,
        include_restfulness=cfg.env.include_restfulness,
        checkpoint_dir=str(cfg.training.checkpoint_dir),
        experiment_name=cfg.mlflow.experiment_name,
        reward_variant=reward_variant,
        reward_weights=explicit_weights,
        n_envs=cfg.training.n_envs,
    )
    return train(tcfg)


def serve_daemon(cfg: BioListenerSleepConfig) -> None:
    """Uvicorn boot using cfg.daemon.{host,port}; mongo URL from cfg.persistence."""
    _propagate_legacy_env(cfg)
    # Stash the resolved config so daemon.create_session can read it
    from app.runtime import daemon
    daemon.RESOLVED_CONFIG = cfg

    uvicorn.run(
        "app.runtime.daemon:app",
        host=cfg.daemon.host,
        port=cfg.daemon.port,
        reload=False,
        log_level="info",
    )
