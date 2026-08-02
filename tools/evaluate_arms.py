"""Score every comparison arm on the held-out subject cohort.

Writes one parquet per arm to ``artifacts/convergence_eval/``. Every arm sees
the same 100 simulated subjects in the same order, so the cells are paired
subject by subject and can be contrasted directly.

Arms:
  <variant>_seed<N>   a trained policy, one per reward variant and seed
  ramp                open-loop linear descent, 11 Hz to 4 Hz over the episode
  const4              open-loop constant 4 Hz
  ramp_lowdose        the ramp with brightness scaled to the learned arm's
                      mean delivered energy, isolating schedule from dose
  yoked<N>            seed N's own per-tick command trace replayed to a
                      different subject, isolating adaptivity from dose

Checkpoints are selected by the step count stored inside the archive, not by
filename or timestamp: short smoke runs land in the same directory under the
same seed name, and picking the newest file silently selects the wrong one.

  python3 tools/evaluate_arms.py                 # every arm
  python3 tools/evaluate_arms.py --arms ramp yoked
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import re
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

N_EPISODES, EPISODE_S, BASE_SEED, TICK_HZ = 100, 300.0, 1000, 4.0
BUDGET = 300_000
RAMP_START_HZ, RAMP_END_HZ, CONST_HZ = 11.0, 4.0, 4.0
LOWDOSE_ENERGY = 0.1608          # learned arm's mean duty x brightness
VARIANTS = ("full_shaped", "theta_alpha_delta", "naive_alpha_max", "naive_plv_only")
CKPT_DIR = Path("app/agent/checkpoints")
OUT = Path("artifacts/convergence_eval")


def checkpoints_for(variant):
    """Newest checkpoint per seed whose stored step count is the target budget."""
    found = {}
    for p in sorted(CKPT_DIR.glob(f"sac_{variant}_seed*.zip"), key=lambda q: q.stat().st_mtime):
        m = re.search(r"seed(\d+)", p.name)
        if not m:
            continue
        with zipfile.ZipFile(p) as z:
            steps = json.loads(z.read("data").decode()).get("num_timesteps")
        if steps == BUDGET:
            found[m.group(1)] = p
    return found


def one(job):
    kind, name, payload = job
    dest = OUT / f"{name}.parquet"
    if dest.exists():
        return f"{name} (cached)"
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    from app.agent.eval_rollout import rollout_episodes
    from app.agent.policies import (FixedFrequencyPolicy, ScheduledRampPolicy,
                                    build_yoked_factory)
    from app.agent.safety import BRIGHTNESS_MAX_V1, FILL_MAX
    from app.brain_sim.domain_random import RandomizationRanges

    amp = dict(fill=FILL_MAX, brightness=BRIGHTNESS_MAX_V1,
               audio_depth_pct=20.0, audio_master_pct=50.0)
    kw = dict(n_episodes=N_EPISODES, episode_s=EPISODE_S,
              randomization=RandomizationRanges(exogenous_shift=False),
              base_seed=BASE_SEED, tick_hz=TICK_HZ, label=name)

    if kind == "policy":
        df = rollout_episodes(checkpoint_path=payload, **kw)
    elif kind == "ramp":
        df = rollout_episodes(policy=ScheduledRampPolicy(
            start_hz=RAMP_START_HZ, end_hz=RAMP_END_HZ, ramp_end_frac=1.0,
            duration_s=EPISODE_S, **amp), **kw)
    elif kind == "const4":
        df = rollout_episodes(policy=FixedFrequencyPolicy(
            frequency_hz=CONST_HZ, **amp), **kw)
    elif kind == "ramp_lowdose":
        low = dict(amp, brightness=LOWDOSE_ENERGY / FILL_MAX)
        df = rollout_episodes(policy=ScheduledRampPolicy(
            start_hz=RAMP_START_HZ, end_hz=RAMP_END_HZ, ramp_end_frac=1.0,
            duration_s=EPISODE_S, **low), **kw)
    else:                                    # yoked
        from app.agent.policies.yoked import assert_dose_equivalence
        donor = pd.read_parquet(OUT / f"{payload}.parquet")
        df = rollout_episodes(policy_factory=build_yoked_factory(donor, offset=1), **kw)
        assert_dose_equivalence(donor, df, offset=1, tol=1e-9)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest)
    return name


def build_jobs(selected):
    jobs = []
    if "policies" in selected:
        for v in VARIANTS:
            found = checkpoints_for(v)
            if len(found) != 5:
                raise SystemExit(
                    f"{v}: found {len(found)} checkpoints at {BUDGET} steps, expected 5")
            tag = "sac300k" if v == "full_shaped" else f"{v}300k"
            jobs += [("policy", f"{tag}_seed{s}", str(p)) for s, p in sorted(found.items())]
    if "ramp" in selected:
        jobs.append(("ramp", "ramp", None))
    if "const4" in selected:
        jobs.append(("const4", "const4", None))
    if "ramp_lowdose" in selected:
        jobs.append(("ramp_lowdose", "ramp_lowdose", None))
    if "yoked" in selected:
        jobs += [("yoked", f"yoked300k_seed{s}", f"sac300k_seed{s}") for s in range(1, 6)]
    return jobs


if __name__ == "__main__":
    ALL = ["policies", "ramp", "const4", "ramp_lowdose", "yoked"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=ALL, default=ALL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = build_jobs(set(args.arms))
    # yoked replays a donor cell, so those cells must exist first
    if any(j[0] == "yoked" for j in jobs) and any(j[0] == "policy" for j in jobs):
        raise SystemExit("run --arms policies first; yoked replays their output")

    print(f"{len(jobs)} arms, {args.workers} workers", flush=True)
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for name in pool.imap_unordered(one, jobs):
            print("done:", name, flush=True)
    print("EVALUATION COMPLETE", flush=True)
