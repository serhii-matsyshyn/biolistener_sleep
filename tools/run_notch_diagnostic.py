"""
Stimulus-artifact re-evaluation of the reward-ablation cohort.

Question: how much of the measured theta/alpha rise that separates the four
reward variants is the strobe itself rather than the simulated brain?

Method: replay each of the 12 cohort checkpoints over the same 100
domain-randomised subjects with ``diagnostics=True``. The diagnostic is a
parallel measurement channel - the observation vector and the reward are
untouched - so the actions, and therefore the trajectories, are bit-identical
to a run without it (pinned by
``tests/unit/test_stim_artifact_diagnostic.py``). This is a re-measurement of
the existing policies, not a retraining.

Outputs one parquet per checkpoint in ``artifacts/notch_diag/``. Resumable:
completed cells are skipped.

    python tools/run_notch_diagnostic.py [--n 100] [--workers 6]
"""
from __future__ import annotations

import os

# Must precede numpy import: each worker is a separate process, and BLAS
# threading across 6 of them oversubscribes the box and slows everything down.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

CKPT_DIR = Path("app/agent/checkpoints")
OUT_DIR = Path("artifacts/notch_diag")

# An earlier budget-matched cohort of the three naive variants, retained so the
# diagnostic can be re-run against it. The results reported in the write-up use
# the later common-budget cohort instead.
COHORT_POSTFIX = {
    "naive_alpha_max": [
        "sac_naive_alpha_max_seed1_1785268620",
        "sac_naive_alpha_max_seed2_1785269364",
        "sac_naive_alpha_max_seed3_1785270204",
        "sac_naive_alpha_max_seed4_1785270929",
        "sac_naive_alpha_max_seed5_1785271645"
    ],
    "naive_plv_only": [
        "sac_naive_plv_only_seed1_1785272360",
        "sac_naive_plv_only_seed2_1785273070",
        "sac_naive_plv_only_seed3_1785273791",
        "sac_naive_plv_only_seed4_1785274498",
        "sac_naive_plv_only_seed5_1785275218"
    ],
    "theta_alpha_delta": [
        "sac_theta_alpha_delta_seed1_1785275938",
        "sac_theta_alpha_delta_seed2_1785276647",
        "sac_theta_alpha_delta_seed3_1785277354",
        "sac_theta_alpha_delta_seed4_1785278064",
        "sac_theta_alpha_delta_seed5_1785278781"
    ],
    "full_shaped": [
        "sac_full_shaped_seed1_1785279492",
        "sac_full_shaped_seed2_1785280204",
        "sac_full_shaped_seed3_1785280913",
        "sac_full_shaped_seed4_1785281623",
        "sac_full_shaped_seed5_1785282357"
    ]
}

COHORT_BUDGET100K = {
    "naive_alpha_max": [
        "sac_naive_alpha_max_seed1_1785236909",
        "sac_naive_alpha_max_seed2_1785237652",
        "sac_naive_alpha_max_seed3_1785238387",
    ],
    "naive_plv_only": [
        "sac_naive_plv_only_seed1_1785239139",
        "sac_naive_plv_only_seed2_1785239907",
        "sac_naive_plv_only_seed3_1785240669",
    ],
    "theta_alpha_delta": [
        "sac_theta_alpha_delta_seed1_1785241428",
        "sac_theta_alpha_delta_seed2_1785242160",
        "sac_theta_alpha_delta_seed3_1785242872",
    ],
}

# The original reward-ablation cohort, resolved from
# artifacts/cohort_run_ids.json against the MLflow run names. Pinned here so
# the diagnostic cannot silently pick up a different training batch (there
# are several on disk).
COHORT = {
    "naive_alpha_max": [
        "sac_naive_alpha_max_seed1_1779573924",
        "sac_naive_alpha_max_seed2_1779574277",
        "sac_naive_alpha_max_seed3_1779574629",
    ],
    "naive_plv_only": [
        "sac_naive_plv_only_seed1_1779574989",
        "sac_naive_plv_only_seed2_1779575344",
        "sac_naive_plv_only_seed3_1779575698",
    ],
    "theta_alpha_delta": [
        "sac_theta_alpha_delta_seed1_1779576053",
        "sac_theta_alpha_delta_seed2_1779576409",
        "sac_theta_alpha_delta_seed3_1779576765",
    ],
    "full_shaped": [
        "sac_full_shaped_seed1_1779577121",
        "sac_full_shaped_seed2_1779577823",
        "sac_full_shaped_seed3_1779578521",
    ],
}


def _cell(args) -> dict:
    variant, seed_idx, stem, n_episodes, episode_s, base_seed, out_dir = args
    out = Path(out_dir) / f"{variant}_seed{seed_idx}.parquet"
    if out.exists():
        return {"variant": variant, "seed": seed_idx, "status": "skipped",
                "path": str(out)}

    from app.agent.eval_rollout import rollout_episodes

    t0 = time.time()
    df = rollout_episodes(
        checkpoint_path=CKPT_DIR / f"{stem}.zip",
        n_episodes=n_episodes,
        episode_s=episode_s,
        base_seed=base_seed,          # same 100 subjects for every checkpoint
        deterministic=True,
        diagnostics=True,
        label=f"{variant}/seed{seed_idx}",
    )
    df["variant"] = variant
    df["seed"] = seed_idx
    df["checkpoint"] = stem
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return {"variant": variant, "seed": seed_idx, "status": "ok",
            "rows": int(len(df)), "secs": round(time.time() - t0, 1),
            "path": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="episodes per checkpoint")
    ap.add_argument("--episode-s", type=float, default=300.0)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cohort", choices=["original","budget100k","postfix"],
                    default="original",
                    help="which trained cohort to re-measure")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override output directory")
    args = ap.parse_args()

    cohort = {"original": COHORT, "budget100k": COHORT_BUDGET100K,
              "postfix": COHORT_POSTFIX}[args.cohort]
    out_dir = Path(args.out_dir) if args.out_dir else (
        OUT_DIR if args.cohort == "original"
        else Path(f"artifacts/notch_diag_{args.cohort}")
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("notch_diag")

    jobs = [
        (variant, i + 1, stem, args.n, args.episode_s, args.base_seed, str(out_dir))
        for variant, stems in cohort.items()
        for i, stem in enumerate(stems)
    ]
    missing = [j[2] for j in jobs if not (CKPT_DIR / f"{j[2]}.zip").exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("cohort=%s -> %s | %d cells, %d episodes each, %d workers",
             args.cohort, out_dir, len(jobs), args.n, args.workers)

    t0 = time.time()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_cell, jobs):
            results.append(res)
            log.info("[%2d/%2d] %s", len(results), len(jobs), res)

    summary = {
        "n_episodes": args.n,
        "episode_s": args.episode_s,
        "base_seed": args.base_seed,
        "elapsed_s": round(time.time() - t0, 1),
        "cells": sorted(results, key=lambda r: (r["variant"], r["seed"])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
