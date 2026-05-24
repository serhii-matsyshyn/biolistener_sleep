#!/usr/bin/env bash
# Runs the full reward-ablation sweep: 4 reward variants × 3 seeds = 12 SAC
# training runs. Each run logs to MLflow experiment "reward_ablation" with
# tag reward_variant=<name> so the demo notebook can group + plot.
#
# Defaults to ~15 min total on a CPU.
#
# Usage:
#   bash tools/run_ablation.sh                 # default: seeds 1 2 3
#   SEEDS="1 2 3 4 5" bash tools/run_ablation.sh
#   VARIANTS="full_shaped" bash tools/run_ablation.sh   # one variant only
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

VARIANTS="${VARIANTS:-naive_alpha_max naive_plv_only theta_alpha_delta full_shaped}"
SEEDS="${SEEDS:-1 2 3}"

echo "Reward ablation sweep - variants: $VARIANTS | seeds: $SEEDS"
echo

for variant in $VARIANTS; do
    cfg="configs/ablation_${variant}.yaml"
    if [ ! -f "$cfg" ]; then
        echo "ERROR: missing config $cfg" >&2
        exit 1
    fi
    for seed in $SEEDS; do
        echo ">>> $variant  seed=$seed"
        biolistener-sleep train --config "$cfg" --set "training.seed=$seed"
    done
done

echo
echo "All runs complete. Browse with:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db"
