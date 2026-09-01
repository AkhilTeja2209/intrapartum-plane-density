#!/usr/bin/env bash
# Runs after the Arm 1 grid: the failed sparse_k5, then Arm 2 both ways.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

echo "=== 1/3 sparse_k5 (re-run: it died on the Windows commit limit) ==="
python -m src.run_experiment --condition sparse_k5 --model frame --seed 0

echo "=== 2/3 Arm 2 temporal, splicing ON (splice_p 0.5) ==="
python -m src.run_experiment --config configs/default.yaml \
    --condition dense_all --model temporal --seed 0

echo "=== 3/3 Arm 2 temporal, splicing OFF (ablation) ==="
python -m src.run_experiment --config configs/no_splice.yaml \
    --condition dense_all --model temporal --seed 0 --tag nosplice

echo "=== phase 2 complete ==="
