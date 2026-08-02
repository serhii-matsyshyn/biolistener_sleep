#!/usr/bin/env bash
# Trains reward variants at a common budget, so comparisons between them are
# not confounded by training length. Defaults reproduce the reported cohort:
# all four variants, five seeds, 300k steps.
#
# Runs sequentially: each run already uses n_envs=8, so concurrent runs would
# saturate the machine. Results land in a separate MLflow experiment, leaving
# the original cohort untouched.
#
#   bash tools/run_budget_matched_ablation.sh
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

VARIANTS="${VARIANTS:-full_shaped naive_alpha_max naive_plv_only theta_alpha_delta}"
SEEDS="${SEEDS:-1 2 3 4 5}"
STEPS="${STEPS:-300000}"
EXPERIMENT="${EXPERIMENT:-reward_ablation_300k}"

echo "Budget-matched ablation - variants: $VARIANTS | seeds: $SEEDS | steps: $STEPS"
echo "MLflow experiment: $EXPERIMENT"
echo

for variant in $VARIANTS; do
    cfg="configs/ablation_${variant}.yaml"
    [ -f "$cfg" ] || { echo "ERROR: missing config $cfg" >&2; exit 1; }
    for seed in $SEEDS; do
        echo ">>> $variant seed=$seed steps=$STEPS  ($(date +%H:%M:%S))"
        biolistener-sleep train --config "$cfg" \
            --set "training.seed=$seed" \
            --set "training.total_steps=$STEPS" \
            --set "mlflow.experiment_name=$EXPERIMENT"
    done
done

echo
echo "Done. Compare budgets with:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db"
