"""Pin the demo notebook to a specific cohort of MLflow runs.

The reward-ablation MLflow experiment accumulates runs from every sweep over
the lifetime of the project - including old experiments that aren't comparable
with the current sim. Notebook 03 only wants the most recent sweep,
with one variant trained at 50k steps (full_shaped) and three naive variants at 15k.

This script writes `artifacts/cohort_run_ids.json`:

    {
      "naive_alpha_max":   ["<run-id>", ...],
      "naive_plv_only":    ["<run-id>", ...],
      "theta_alpha_delta": ["<run-id>", ...],
      "full_shaped":       ["<run-id>", ...]
    }

Re-run after every `tools/run_ablation.sh` to refresh the pin to the
newest sweep.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

COHORT_BUDGET = {
    "naive_alpha_max":   "30000",
    "naive_plv_only":    "30000",
    "theta_alpha_delta": "30000",
    "full_shaped":       "100000",
}
EXPERIMENT_NAME = "reward_ablation_5min"


def derive(db_path: Path, after_ms: int, experiment: str = EXPERIMENT_NAME) -> dict[str, list[str]]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # Resolve experiment_id
        cur.execute("SELECT experiment_id FROM experiments WHERE name = ?", (experiment,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"MLflow experiment {experiment!r} not found in {db_path}")
        exp_id = row[0]

        out: dict[str, list[str]] = {}
        for variant, budget in COHORT_BUDGET.items():
            cur.execute(
                """
                SELECT r.run_uuid
                FROM runs r
                JOIN tags tv ON r.run_uuid = tv.run_uuid AND tv.key = 'reward_variant' AND tv.value = ?
                JOIN params ts ON r.run_uuid = ts.run_uuid AND ts.key = 'total_steps' AND ts.value = ?
                WHERE r.status = 'FINISHED' AND r.start_time >= ? AND r.experiment_id = ?
                ORDER BY r.start_time
                """,
                (variant, budget, after_ms, exp_id),
            )
            out[variant] = [row[0] for row in cur.fetchall()]
        return out
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("mlflow.db"),
                        help="Path to the MLflow SQLite DB (default: mlflow.db)")
    parser.add_argument("--after-ms", type=int, default=0,
                        help="Lower bound on run start_time in ms epoch. "
                             "Default 0 (no filter) - the experiment name "
                             "filter is normally enough.")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT_NAME,
                        help=f"MLflow experiment name. Default: {EXPERIMENT_NAME!r}")
    parser.add_argument("--out", type=Path, default=Path("artifacts/cohort_run_ids.json"),
                        help="Destination JSON file")
    args = parser.parse_args()

    cohort = derive(args.db, args.after_ms, args.experiment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cohort, indent=2))

    print(f"wrote {args.out}")
    for variant, ids in cohort.items():
        print(f"  {variant:<22s} {len(ids)} runs")
        for rid in ids:
            print(f"    {rid}")


if __name__ == "__main__":
    main()
