"""Unified configuration - single source of truth.

Public entry points::

    from app.config import BioListenerSleepConfig, load_config, resolved_config

The Pydantic schema in ``schema.py`` defines every tunable value in the
project. ``loader.py`` resolves a final config object from the 5-layer chain:

    1. Pydantic schema defaults (today's hardcoded values)
    2. configs/defaults.yaml (committed project-wide overrides)
    3. user-selected --config <profile>.yaml
    4. environment variables (BIOLISTENER_SLEEP_SECTION__FIELD)
    5. --set key=value CLI overrides

The fully-resolved BioListenerSleepConfig is snapshot to studies/<subj>/<sid>/config.yaml
at session start so any session can be re-run byte-identical.
"""
from app.config.schema import BioListenerSleepConfig
from app.config.loader import load_config, dump_yaml, parse_set_override

__all__ = [
    "BioListenerSleepConfig",
    "load_config",
    "dump_yaml",
    "parse_set_override",
]
