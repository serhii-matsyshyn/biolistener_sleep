# `data/external_tracks/` - external open-loop scripted-protocol data

Place a **symlink** named `source` here pointing at your local
replication-data folder. The folder should contain:

```
source/
├── sessions.yaml          # activity catalog (block -> activity -> variant)
├── sounds/       # audio files (subdir name set by the external project)
└── strobe/       # strobe-event CSVs
```

The actual data (audio + CSVs + catalog YAML) is **not** included in this
repository - for both copyright and disk-footprint reasons. Each researcher
creates their own symlink, gitignored, that resolves to wherever they keep
their local replication checkout.

## Setup (one-shot per workstation)

```bash
ln -snf /path/to/your/external/data data/external_tracks/source
```

Then verify:

```bash
biolistener-sleep open-loop-track-check
```

Should print `OK - N activities in data/external_tracks/source/sessions.yaml`.

If you keep your data somewhere unusual, override the path at session time:

```bash
biolistener-sleep run --config configs/real_S007_open_loop_track.yaml \
    --set open_loop_track.data_root=/some/other/path
```

## Why a symlink?

- The repo never carries copyright-protected audio.
- The symlink itself is `.gitignore`d, so it never lands in commits.
- Each researcher's local checkout points at whatever they have locally.
