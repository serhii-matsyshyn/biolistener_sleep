"""
Unified `biolistener-sleep` command - single entry point for everything.

Subcommands::

    biolistener-sleep config show     --config <path>   # resolved config dump
    biolistener-sleep config validate --config <path>   # just validate, no print
    biolistener-sleep run             --config <path>   # one closed-loop session
    biolistener-sleep train           --config <path>   # SAC training run
    biolistener-sleep daemon          --config <path>   # FastAPI + UI
    biolistener-sleep replay          --session <id>    # rehydrate + re-render
    biolistener-sleep open-loop-track-check --config <path>   # verify external data

The `--set section.field=value` flag works on every subcommand for one-off
overrides without writing a new YAML profile (parsed by
``app.config.loader.parse_set_override``).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from app.config import BioListenerSleepConfig, dump_yaml, load_config

logger = logging.getLogger("cli")

app = typer.Typer(
    no_args_is_help=True,
    help="BioListener Sleep - closed-loop sleep-onset RL system.",
    pretty_exceptions_show_locals=False,
)


config_app = typer.Typer(help="Inspect / validate resolved config.")
app.add_typer(config_app, name="config")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve(
    config_path: Optional[Path],
    overrides: Optional[List[str]],
) -> BioListenerSleepConfig:
    return load_config(profile_path=config_path, overrides=overrides)


@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(None, "--config", "-c",
                                            help="Profile YAML to layer on top of defaults."),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s",
                                                help="key.path=value override (repeatable)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Print the fully-resolved BioListenerSleepConfig as YAML."""
    _setup_logging(verbose)
    resolved = _resolve(config, set_)
    typer.echo(dump_yaml(resolved))


@config_app.command("validate")
def config_validate(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Validate the resolved config (exits non-zero if invalid)."""
    try:
        _resolve(config, set_)
    except Exception as err:
        typer.echo(f"INVALID: {err}", err=True)
        raise typer.Exit(code=1)
    typer.echo("OK")


@app.command("run")
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run one closed-loop session (sim or real hardware, any protocol)."""
    _setup_logging(verbose)
    cfg = _resolve(config, set_)
    from app.cli._runner import run_session_from_config
    asyncio.run(run_session_from_config(cfg))


@app.command("train")
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train an RL agent against the simulator."""
    _setup_logging(verbose)
    cfg = _resolve(config, set_)
    from app.cli._runner import train_from_config
    train_from_config(cfg)


@app.command("eval")
def eval_(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-k",
                                     help="Path to SAC .zip from `train`."),
    n: int = typer.Option(100, "--n", help="Number of fresh-subject episodes."),
    episode_s: float = typer.Option(60.0, "--episode-s",
                                     help="Seconds per episode."),
    base_seed: int = typer.Option(1000, "--base-seed",
                                   help="Per-episode subject seed base."),
    out: Path = typer.Option(Path("artifacts/rollouts.parquet"), "--out", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Roll a trained checkpoint on N fresh domain-randomised NMM subjects;
    dump per-tick trajectories to a Parquet for the cross-subject
    consistency notebook plot."""
    _setup_logging(verbose)
    from app.agent.eval_rollout import rollout_episodes, write_parquet, summary
    df = rollout_episodes(
        checkpoint_path=checkpoint,
        n_episodes=n,
        episode_s=episode_s,
        base_seed=base_seed,
    )
    write_parquet(df, out)
    typer.echo(summary(df))


@app.command("shield-test")
def shield_test(
    episodes: int = typer.Option(10_000, "--episodes",
                                  help="Number of adversarial requests to fire."),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("artifacts/shield_log.json"), "--out", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Adversarial test of the SafetySupervisor PSE-band shield.

    Pass condition: ``n_pse_leaked == 0``.
    """
    _setup_logging(verbose)
    from app.agent.safety_adversarial import run_shield_attack, write_json
    result = run_shield_attack(n_requests=episodes, seed=seed)
    write_json(result, out)
    typer.echo({
        "n_requested":   result.n_requested,
        "n_pse_blocked": result.n_pse_blocked,
        "n_pse_leaked":  result.n_pse_leaked,
        "passed":        result.n_pse_leaked == 0,
    })


@app.command("daemon")
def daemon(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s"),
    host: Optional[str] = typer.Option(None, "--host",
                                          help="Override daemon.host."),
    port: Optional[int] = typer.Option(None, "--port",
                                         help="Override daemon.port."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the FastAPI daemon (UI + REST + WS)."""
    _setup_logging(verbose)
    cfg = _resolve(config, set_)
    if host is not None:
        cfg.daemon.host = host
    if port is not None:
        cfg.daemon.port = port
    from app.cli._runner import serve_daemon
    serve_daemon(cfg)


@app.command("replay")
def replay(
    session: Path = typer.Option(..., "--session", "-s",
                                    help="Session folder produced by a prior run."),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    mask_replay: bool = typer.Option(False, "--mask-replay",
                                       help="Re-issue stim events through a virtual mask."),
    speed: float = typer.Option(1.0, "--speed",
                                  help="Replay speed multiplier."),
    summary_only: bool = typer.Option(False, "--summary-only"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reload a session and optionally re-issue stim through a virtual mask."""
    _setup_logging(verbose)
    from app.runtime.replay import load_session, reissue_to_virtual_mask
    bundle = load_session(session)
    typer.echo(f"Loaded session: {bundle.folder}")
    typer.echo(f"  stim events:  {len(bundle.stim_log)}")
    typer.echo(f"  feature rows: {len(bundle.features)}")
    typer.echo(f"  raw_eeg.csv:  {bundle.raw_eeg_csv}")
    if summary_only or not mask_replay:
        return
    typer.echo(f"Replaying {len(bundle.stim_log)} stim events at {speed}x...")
    reissue_to_virtual_mask(bundle, speed=speed)


@app.command("open-loop-track-check")
def open_loop_track_check(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    set_: Optional[List[str]] = typer.Option(None, "--set", "-s"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Verify open_loop_track.data_root is reachable and the catalog parses."""
    _setup_logging(verbose)
    cfg = _resolve(config, set_)
    root = cfg.open_loop_track.data_root
    if root is None:
        typer.echo("open_loop_track.data_root is unset - protocol disabled.",
                   err=True)
        raise typer.Exit(code=1)
    if not root.exists():
        typer.echo(f"data_root not found: {root}", err=True)
        raise typer.Exit(code=1)
    catalog_path = root / cfg.open_loop_track.catalog_filename
    if not catalog_path.exists():
        typer.echo(f"catalog missing: {catalog_path}", err=True)
        raise typer.Exit(code=1)
    # Try to parse via the integration module (lazy import - only needed here)
    try:
        from app.open_loop_track.catalog import OpenLoopTrackCatalog
        cat = OpenLoopTrackCatalog.from_yaml(catalog_path)
        n_activities = len(cat.activities())
        typer.echo(f"OK - {n_activities} activities in {catalog_path}")
    except Exception as err:
        typer.echo(f"Catalog parse failed: {err}", err=True)
        raise typer.Exit(code=1)


@app.command("audio-test")
def audio_test() -> None:
    """
    Launch the standalone audio engineer's panel (Tkinter sliders +
    stereo ping-pong test) for tuning the audio engine and verifying
    L/R speaker setup. Boots the pyo Server and the BiofeedbackEngine,
    then hands over to the manual Tkinter UI.

    This is the original `biofeedback_sound_engine.py` UI, preserved
    intact at `app/cli/manual_tk.py`. Use it for one-off audio device
    checks and preset auditioning; closed-loop sessions go through
    `biolistener-sleep run --config <profile>` instead.
    """
    # Spawn as a subprocess so the Tkinter mainloop is in its own process
    # (won't block the Typer CLI; clean SIGINT handling).
    proc = subprocess.run([sys.executable, "-m", "app.cli.manual_tk"])
    raise typer.Exit(code=proc.returncode)


def main() -> None:
    """Entry point for the `biolistener-sleep` console script."""
    app()


if __name__ == "__main__":
    main()
