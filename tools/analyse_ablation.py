"""Score the reward-specification ablation on the held-out cohort.

Scores every variant on the COMMANDED strobe frequency, which is the agent's
action and so cannot be inflated by the stimulus. Uses the delivered-frequency
mask so an arm that stops emitting light cannot score a descent on darkness.
Reports the descent contingency across a threshold sweep, not at one cutoff.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EVAL = Path("artifacts/convergence_eval")
VARIANTS = [("full_shaped", "shaped"), ("theta_alpha_delta", "biomarker only"),
            ("naive_alpha_max", "naive alpha"), ("naive_plv_only", "naive PLV")]
FINAL_S = 30.0        # window over which the "final" command is taken
DEEP_HZ = 6.0         # deep-theta band edge


def per_episode_final(path):
    d = pd.read_parquet(path)
    if "enabled" in d.columns:            # only credit ticks that delivered light
        d = d[d["enabled"].astype(bool)]
    if d.empty:
        return None
    tail = d[d["t_s"] >= d["t_s"].max() - FINAL_S]
    return tail.groupby("episode")["strobe_freq_hz"].median()


def files_for(variant):
    if variant == "full_shaped":
        return sorted(glob.glob(str(EVAL / "sac300k_seed*.parquet")))
    return sorted(glob.glob(str(EVAL / f"{variant}300k_seed*.parquet")))


rows, per_seed = [], []
for key, label in VARIANTS:
    fracs, meds = [], []
    for f in files_for(key):
        s = per_episode_final(f)
        if s is None:
            continue
        seed = re.search(r"seed(\d+)", Path(f).stem).group(1)
        fr = float((s < DEEP_HZ).mean())
        fracs.append(fr); meds.append(float(s.median()))
        per_seed.append(dict(variant=label, seed=seed, median_hz=s.median(), frac_below6=fr))
    if fracs:
        rows.append(dict(variant=label, n=len(fracs), median_hz=np.median(meds),
                         mean_frac=float(np.mean(fracs)), fracs=fracs,
                         seed_range=max(meds) - min(meds)))

df = pd.DataFrame(per_seed)
df.to_csv("artifacts/ablation_per_seed.csv", index=False)

print(f"{'variant':>16} {'median Hz':>10} {'mean frac<6':>12} {'seed range':>11}")
for r in rows:
    print(f"{r['variant']:>16} {r['median_hz']:10.2f} {r['mean_frac']:12.3f} {r['seed_range']:11.2f}")

print("\nthreshold sweep (a seed 'descends' if >T of its episodes end below 6 Hz):")
print(f"{'T':>6} {'shaped':>9} {'others':>9} {'Fisher p':>10}")
shaped = next(r for r in rows if r["variant"] == "shaped")["fracs"]
others = [f for r in rows if r["variant"] != "shaped" for f in r["fracs"]]
for T in (0.05, 0.10, 0.25, 0.50):
    a = sum(1 for f in shaped if f > T); b = sum(1 for f in others if f > T)
    p = stats.fisher_exact([[a, len(shaped) - a], [b, len(others) - b]])[1]
    print(f"{T:6.2f} {a:>4}/{len(shaped):<4} {b:>4}/{len(others):<4} {p:10.4f}")
print("\nper-seed detail -> artifacts/ablation_per_seed.csv")
print(df.to_string(index=False))
