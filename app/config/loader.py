"""
5-layer config resolution.

Order (later overrides earlier):
    1. Pydantic schema defaults
    2. configs/defaults.yaml (if it exists in cwd or alongside the loader)
    3. user-selected --config <profile>.yaml
    4. environment variables (BIOLISTENER_SLEEP_SECTION__FIELD)
    5. --set section.field=value CLI overrides

The result is one validated BioListenerSleepConfig that the CLI / daemon / training
loop all consume.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from pydantic import ValidationError

from app.config.schema import BioListenerSleepConfig

logger = logging.getLogger("config.loader")

DEFAULTS_FILENAME = "defaults.yaml"
ENV_PREFIX = "BIOLISTENER_SLEEP_"
ENV_NESTED_SEP = "__"


def load_config(
    profile_path: Optional[Path] = None,
    overrides: Optional[Iterable[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> BioListenerSleepConfig:
    """
    Build a fully-resolved :class:`BioListenerSleepConfig`.

    Parameters
    ----------
    profile_path:
        Optional YAML file. If ``None``, only defaults + env + overrides apply.
    overrides:
        Iterable of ``"section.field=value"`` strings (parsed via
        :func:`parse_set_override`). Useful for one-off CLI tweaks.
    extra_env:
        Optional override of ``os.environ`` (testing).
    """
    # Layer 1: schema defaults
    merged: Dict[str, Any] = BioListenerSleepConfig().model_dump(mode="python")

    # Layer 2: project-wide defaults.yaml
    defaults_yaml = _find_defaults_yaml()
    if defaults_yaml is not None:
        _deep_merge(merged, _load_yaml(defaults_yaml))
        logger.debug("Loaded defaults from %s", defaults_yaml)

    # Layer 3: user-selected profile
    if profile_path is not None:
        profile_path = Path(profile_path)
        if not profile_path.exists():
            raise FileNotFoundError(f"Config profile not found: {profile_path}")
        _deep_merge(merged, _load_yaml(profile_path))
        logger.debug("Loaded profile %s", profile_path)

    # Layer 4: env vars
    env = extra_env if extra_env is not None else os.environ
    for parts, v in _env_overrides(env):
        _set_nested(merged, parts, v)

    # Layer 5: --set CLI overrides
    if overrides:
        for spec in overrides:
            key, val = parse_set_override(spec)
            _set_nested(merged, key, val)

    try:
        return BioListenerSleepConfig.model_validate(merged)
    except ValidationError as err:
        raise ValueError(f"Invalid resolved config:\n{err}") from err


def dump_yaml(config: BioListenerSleepConfig) -> str:
    """Serialize an BioListenerSleepConfig to YAML for snapshotting / inspection."""
    # mode='json' to make Path -> str etc.
    data = config.model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def parse_set_override(spec: str) -> tuple[List[str], Any]:
    """
    Parse a ``--set section.field=value`` string.

    Returns the dotted-key path (split on '.') and the value (parsed as
    JSON if it looks like JSON, otherwise as a string).

    Examples::

        parse_set_override("session.duration_s=900")
            -> (["session", "duration_s"], 900)
        parse_set_override("safety.freq_max_hz=12.5")
            -> (["safety", "freq_max_hz"], 12.5)
        parse_set_override("session.protocol=open_loop_track")
            -> (["session", "protocol"], "open_loop_track")
        parse_set_override("audio.enabled=true")
            -> (["audio", "enabled"], True)
    """
    if "=" not in spec:
        raise ValueError(f"--set must be key=value, got: {spec!r}")
    key, raw = spec.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise ValueError(f"--set has empty key: {spec!r}")
    return key.split("."), _parse_value(raw)


def _find_defaults_yaml() -> Optional[Path]:
    """Search cwd, then alongside this module's project for defaults.yaml."""
    candidates = [
        Path.cwd() / "configs" / DEFAULTS_FILENAME,
        Path(__file__).resolve().parents[2] / "configs" / DEFAULTS_FILENAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return data


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Recursively merge overlay into base. Lists are replaced (not concatenated)."""
    for key, value in overlay.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _env_overrides(env: Dict[str, str]) -> List[tuple[List[str], Any]]:
    """
    Pick env vars like ``BIOLISTENER_SLEEP_PERSISTENCE__MONGO_URL=...`` and produce
    a list of (dotted-key, value) pairs.

    Double underscore is the section/field separator (pydantic-settings
    convention). The legacy ``BIOLISTENER_SLEEP_ENABLE_AUDIO`` is kept for
    backward-compat with the env shim added in M6+.
    """
    out: List[tuple[List[str], Any]] = []
    for k, v in env.items():
        if not k.startswith(ENV_PREFIX):
            continue
        remainder = k[len(ENV_PREFIX):]

        # Backward-compat shim for the M6+ env var
        if remainder == "ENABLE_AUDIO":
            out.append((["audio", "enabled"], _parse_value(v)))
            continue
        if remainder == "MONGO_URL":
            out.append((["persistence", "mongo_url"], v))
            continue
        if remainder == "MONGO_DB":
            out.append((["persistence", "mongo_db"], v))
            continue

        if ENV_NESTED_SEP not in remainder:
            continue  # not a config-targeting var
        parts = remainder.lower().split(ENV_NESTED_SEP)
        out.append((parts, _parse_value(v)))
    return out


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set d[k1][k2]... = value, creating sub-dicts as needed."""
    target = d
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    """JSON-parse if it looks like JSON (numbers, booleans, lists, null), else string."""
    if raw == "":
        return ""
    # Try JSON for unambiguous literals
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Handle a few common cases JSON misses
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    return raw
