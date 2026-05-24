"""
Open-loop scripted-protocol playback - comparison-arm active control
for the three-protocol experimental design (§10 of the plan).

All public symbols are neutrally named. No third-party product is named
in any module here. The external data (audio + strobe CSV/YAML) is
referenced via the user-supplied ``open_loop_track.data_root`` config
field; only the playback math (parametric ramp equations, ~200 LOC)
lives in-repo.
"""
from app.open_loop_track.catalog import (
    OpenLoopTrackActivity,
    OpenLoopTrackCatalog,
    OpenLoopTrackVariant,
)
from app.open_loop_track.csv_playback import CsvEventPlayback
from app.open_loop_track.playback import OpenLoopTrackPlayback
from app.open_loop_track.policy import OpenLoopTrackPolicy

__all__ = [
    "CsvEventPlayback",
    "OpenLoopTrackActivity",
    "OpenLoopTrackCatalog",
    "OpenLoopTrackVariant",
    "OpenLoopTrackPlayback",
    "OpenLoopTrackPolicy",
]
