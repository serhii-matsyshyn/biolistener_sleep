"""
Parse the external catalog YAML that maps activity name + duration variant
to a (strobe CSV, audio file) pair.

YAML structure expected (catalog format defined by the external
replication project; we just consume it):

    configuration:
      sounds_dir:  <subdir>
      strobes_dir: <subdir>

    blocks:
      "<block name>":
        "<activity name>":
          "<variant key>":   # e.g. "10", "15", "20", "default", "15_no-intro"
            sound:  "<url-encoded relative path under sounds_dir>"
            strobe: "<filename under strobes_dir>"   # optional
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("open_loop_track.catalog")


@dataclass
class OpenLoopTrackVariant:
    """One duration variant of one activity in the catalog."""
    variant_key: str               # "10" | "15" | "20" | "default" | ...
    duration_min: Optional[int]    # parsed from variant_key when numeric
    sound_path_encoded: str        # URL-encoded path as in the YAML
    strobe_filename: Optional[str] # may be missing for sound-only activities

    def resolved_audio_path(self, data_root: Path, audio_subdir: str) -> Path:
        # Files are stored as a FLAT directory with URL-encoded filenames
        # (the external project's chosen layout). Do NOT decode - the actual
        # filename on disk literally contains `%20` / `%2F` characters.
        return data_root / audio_subdir / self.sound_path_encoded

    def resolved_strobe_path(self, data_root: Path, strobe_subdir: str
                              ) -> Optional[Path]:
        if not self.strobe_filename:
            return None
        return data_root / strobe_subdir / self.strobe_filename


@dataclass
class OpenLoopTrackActivity:
    name: str                                  # activity slug used by UI / config
    display_name: str                          # human-readable from the YAML
    block: str                                 # parent block name
    variants: Dict[str, OpenLoopTrackVariant] = field(default_factory=dict)

    def variant_for_duration(self, duration_min: int
                              ) -> Optional[OpenLoopTrackVariant]:
        """Pick the variant with a numeric key matching the requested duration."""
        key = str(duration_min)
        if key in self.variants:
            return self.variants[key]
        return None

    def default_variant(self) -> Optional[OpenLoopTrackVariant]:
        if "default" in self.variants:
            return self.variants["default"]
        # else fall back to first numeric variant
        for k, v in self.variants.items():
            if v.duration_min is not None:
                return v
        return None


@dataclass
class OpenLoopTrackCatalog:
    """All activities parsed from the external catalog YAML."""
    audio_subdir: str
    strobe_subdir: str
    by_name: Dict[str, OpenLoopTrackActivity] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "OpenLoopTrackCatalog":
        if not path.exists():
            raise FileNotFoundError(path)
        raw = yaml.safe_load(path.read_text()) or {}
        configuration = raw.get("configuration", {})
        audio_subdir = configuration.get("sounds_dir", "")
        strobe_subdir = configuration.get("strobes_dir", "")
        cat = cls(audio_subdir=audio_subdir, strobe_subdir=strobe_subdir)
        blocks = raw.get("blocks", {})
        for block_name, activities in blocks.items():
            if not isinstance(activities, dict):
                continue
            # YAML 1.1 quirks: unquoted keys like `yes`/`no`/`on`/`off` parse
            # as Python booleans, and numeric variants (10/15/20) parse as
            # ints. Coerce every key to str up-front so downstream string
            # ops (.lower(), .replace()) never see a non-str.
            block_name_s = str(block_name)
            for display_name, variants_dict in activities.items():
                display_name_s = str(display_name)
                slug = _slugify(display_name_s)
                act = OpenLoopTrackActivity(
                    name=slug, display_name=display_name_s, block=block_name_s,
                )
                if not isinstance(variants_dict, dict):
                    continue
                for variant_key, body in variants_dict.items():
                    if not isinstance(body, dict):
                        continue
                    sound = body.get("sound")
                    strobe = body.get("strobe")
                    if not sound:
                        continue
                    variant_key_s = str(variant_key)
                    dur = _maybe_int(variant_key_s)
                    act.variants[variant_key_s] = OpenLoopTrackVariant(
                        variant_key=variant_key_s,
                        duration_min=dur,
                        sound_path_encoded=sound,
                        strobe_filename=strobe,
                    )
                if act.variants:
                    cat.by_name[slug] = act
        return cat

    def activities(self) -> List[OpenLoopTrackActivity]:
        return list(self.by_name.values())

    def get(self, name: str) -> Optional[OpenLoopTrackActivity]:
        return self.by_name.get(name)


def _slugify(s: str) -> str:
    return (
        s.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("'", "")
        .replace(":", "")
    )


def _maybe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None
