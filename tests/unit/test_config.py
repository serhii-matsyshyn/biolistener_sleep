"""
Config-resolution tests for M8.1.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from app.config import BioListenerSleepConfig, dump_yaml, load_config
from app.config.loader import parse_set_override


def test_schema_defaults_load():
    cfg = load_config()
    assert cfg.session.protocol == "rl_agent"
    # audio_enabled defaults to None -> daemon auto-derives from eeg_source
    assert cfg.session.audio_enabled is None
    # The configs/defaults.yaml may override or match these
    assert cfg.safety.freq_max_hz == 14.0


def test_profile_overrides_defaults(tmp_path: Path):
    p = tmp_path / "prof.yaml"
    p.write_text(textwrap.dedent("""
        session:
          subject_code: TEST123
          duration_s: 900
        safety:
          brightness_max: 0.50
    """).strip())
    cfg = load_config(profile_path=p)
    assert cfg.session.subject_code == "TEST123"
    assert cfg.session.duration_s == 900.0
    assert cfg.safety.brightness_max == 0.50


def test_env_var_overrides_profile(tmp_path: Path):
    p = tmp_path / "prof.yaml"
    p.write_text("persistence:\n  mongo_db: from_profile\n")
    cfg = load_config(
        profile_path=p,
        extra_env={"BIOLISTENER_SLEEP_PERSISTENCE__MONGO_DB": "from_env"},
    )
    assert cfg.persistence.mongo_db == "from_env"


def test_legacy_env_var_mongo_url():
    cfg = load_config(extra_env={"MONGO_URL": "mongodb://test:1234/"})
    assert cfg.persistence.mongo_url == "mongodb://localhost:27017/"


def test_legacy_env_var_biolistener_sleep_mongo_url():
    cfg = load_config(extra_env={"BIOLISTENER_SLEEP_MONGO_URL": "mongodb://x:1/"})
    assert cfg.persistence.mongo_url == "mongodb://x:1/"


def test_set_override_beats_env(tmp_path: Path):
    cfg = load_config(
        overrides=["safety.freq_max_hz=12.0"],
        extra_env={"BIOLISTENER_SLEEP_SAFETY__FREQ_MAX_HZ": "13.5"},
    )
    assert cfg.safety.freq_max_hz == 12.0


def test_parse_set_override_types():
    assert parse_set_override("a.b=42") == (["a", "b"], 42)
    assert parse_set_override("a.b=4.5") == (["a", "b"], 4.5)
    assert parse_set_override("a.b=true") == (["a", "b"], True)
    assert parse_set_override("a.b=hello") == (["a", "b"], "hello")
    assert parse_set_override("a.b=null") == (["a", "b"], None)


def test_dump_yaml_roundtrip():
    cfg = load_config()
    yaml_text = dump_yaml(cfg)
    data = yaml.safe_load(yaml_text)
    cfg2 = BioListenerSleepConfig.model_validate(data)
    assert cfg2.model_dump() == cfg.model_dump()


def test_invalid_field_raises():
    with pytest.raises(ValueError):
        load_config(overrides=["session.protocol=invalid_value"])


def test_biolistener_sleep_cli_help_smoke():
    """`biolistener-sleep --help` should exit 0 and mention all subcommands."""
    result = subprocess.run(
        ["biolistener-sleep", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    for cmd in ("run", "train", "daemon", "replay", "open-loop-track-check"):
        assert cmd in result.stdout, f"{cmd} missing from --help"


def test_biolistener_sleep_config_show_smoke(tmp_path: Path):
    """`biolistener-sleep config show --config <profile>` returns valid YAML."""
    profile = tmp_path / "p.yaml"
    profile.write_text("session:\n  duration_s: 5\n")
    result = subprocess.run(
        ["biolistener-sleep", "config", "show", "--config", str(profile)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load(result.stdout)
    assert data["session"]["duration_s"] == 5


def test_biolistener_sleep_config_validate_smoke():
    result = subprocess.run(
        ["biolistener-sleep", "config", "validate"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
