"""
Ablation: does the learned theta-descent depend on the simulator's
state-dependent resonance shift?

Evaluates the SAME policy on bit-identical simulated subjects with the
drowsiness->theta resonance coupling enabled (default 2.0) and disabled
(0.0). The override is applied after all RNG draws, so the two arms differ
only in that coupling.

Resumable: each arm writes its parquet on completion and is skipped on a
rerun if that parquet already exists.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from app.agent.env import EnvConfig, SleepOnsetEnv  # noqa: E402
from app.agent.eval_rollout import summary  # noqa: E402
from app.brain_sim.domain_random import RandomizationRanges  # noqa: E402

OUT = Path("artifacts")


def rollout(ckpt: Path, n_ep: int, ep_s: float, shift, tag: str) -> pd.DataFrame:
    from stable_baselines3 import SAC

    dest = OUT / f"shift_ablation_{tag}.parquet"
    if dest.exists():
        print(f"[{tag}] parquet exists - skipping", flush=True)
        return pd.read_parquet(dest)

    env = SleepOnsetEnv(EnvConfig(
        episode_s=ep_s, include_restfulness=False,
        randomization=RandomizationRanges(shift_per_unit_ratio_override=shift),
    ))
    model = SAC.load(str(ckpt), device="cpu")
    rows, t0 = [], time.time()
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        done = trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            f, s = info["features"], info["stim"]
            rows.append({
                "episode": ep, "t_s": float(info["t"]),
                "strobe_freq_hz": float(s.frequency_hz),
                "fill": float(s.fill), "brightness": float(s.brightness),
                "theta_alpha_ratio": float(f.theta_alpha_ratio),
                "alpha": float(f.alpha_pwr), "theta": float(f.theta_pwr),
                "plv": float(info["plv"]), "plv_2f": float(info.get("plv_2f", 0.0)),
                "reward": float(reward),
            })
        if (ep + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{tag}] {ep+1}/{n_ep} episodes  {el/60:.1f} min elapsed  "
                  f"eta {(el/(ep+1)*(n_ep-ep-1))/60:.1f} min", flush=True)
    env.close()
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"[{tag}] wrote {dest}", flush=True)
    return df


def main() -> None:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "app/agent/checkpoints/sac_full_shaped_seed3_1779578521.zip")
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    print(f"checkpoint: {ckpt.name}   n_episodes={n_ep}   episode_s={ep_s}", flush=True)

    res = {}
    for tag, shift in (("baseline_shift2", None), ("ablation_shift0", 0.0)):
        res[tag] = summary(rollout(ckpt, n_ep, ep_s, shift, tag))
        print(f"\n=== {tag} ===\n"
              + json.dumps({k: round(v, 4) for k, v in res[tag].items()}, indent=1),
              flush=True)

    (OUT / "shift_ablation_summary.json").write_text(json.dumps(res, indent=1))
    b, a = res["baseline_shift2"], res["ablation_shift0"]
    print("\n=== VERDICT: does the descent survive without the resonance shift? ===")
    print(f"  median final strobe : {b['median_final_strobe_hz']:.2f} -> {a['median_final_strobe_hz']:.2f} Hz")
    print(f"  frac < 6 Hz         : {b['frac_final_below_6hz']:.2f} -> {a['frac_final_below_6hz']:.2f}")
    print(f"  frac < 8 Hz         : {b['frac_final_below_8hz']:.2f} -> {a['frac_final_below_8hz']:.2f}")
    print(f"  final theta/alpha   : {b['mean_final_theta_alpha']:.2f} -> {a['mean_final_theta_alpha']:.2f}")
    print(f"  mean PLV (f)        : {b['mean_plv']:.3f} -> {a['mean_plv']:.3f}")


if __name__ == "__main__":
    main()
