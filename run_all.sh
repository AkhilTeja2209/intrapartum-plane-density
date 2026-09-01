#!/usr/bin/env bash
# Full study, start to finish. Run from the repo root.
#
# Rough GPU budget on a single T4 (Colab free tier), 224px, ResNet-18:
#   dense_all  ~6 min/epoch x ~20 epochs           = ~2 h per seed
#   sparse_*   minutes
#   temporal   ~2-3x the frame-wise cost per epoch = ~4-6 h per seed
# Total for 3 seeds x the full grid: roughly 30-40 GPU-hours.
#
# If that is more than you have, cut in this order -- it costs the least:
#   1. --img-size 160 instead of 224     (~2x faster, ~1 point of macro-F1)
#   2. drop dense_stride2 and dense_stride4 (the curve survives with 4 points)
#   3. 3 seeds -> 2 for the stride conditions, keep 3 for the headline pairs
# Do NOT cut: multiple seeds on the headline comparisons, or the smoothed
# baseline. Those are what make the result believable.

set -euo pipefail
CFG=configs/default.yaml
SEEDS="0 1 2"

echo "== 1. extract frames =="
python -m src.extract_frames --dataset-root data/DatasetV3 --out-dir data/frames

echo "== 2. build index =="
python -m src.build_index --dataset-root data/DatasetV3 --frames-dir data/frames --out data/index.csv

echo "== 3. video-level splits =="
python -m src.splits --index data/index.csv --out data/splits.json --seed 0

echo "== 4. compute Protocol B budgets =="
python -m src.make_budgets --config $CFG --write

echo "== 5. ARM 1: frame-wise ResNet-18 across the density grid =="
for c in sparse_k1 sparse_k1_curated sparse_k2 sparse_k5 sparse_k10 sparse_k20 \
         dense_stride8 dense_stride4 dense_stride2 dense_all \
         dense_matched_k5 dense_matched_k20; do
  for s in $SEEDS; do
    echo "--- $c seed $s ---"
    python -m src.run_experiment --config $CFG --condition "$c" --model frame --seed "$s"
  done
done

echo "== 6. ARM 2: temporal, from ImageNet init =="
for s in $SEEDS; do
  python -m src.run_experiment --config $CFG --condition dense_all --model temporal --seed "$s" --tag lstm_bi
done

echo "== 6b. ARM 2 ablations: GRU, and causal (deployable) LSTM =="
for s in $SEEDS; do
  python -m src.run_experiment --config configs/gru.yaml       --condition dense_all --model temporal --seed "$s" --tag gru_bi
  python -m src.run_experiment --config configs/causal.yaml    --condition dense_all --model temporal --seed "$s" --tag lstm_uni
done

echo "== 7. ARM 2 warm-started from the frame-wise encoder =="
for s in $SEEDS; do
  python -m src.run_experiment --config $CFG --condition dense_all --model temporal --seed "$s" \
      --tag warm --warm-start "results/dense_all__frame__seed${s}/best.pt"
done

echo "== 8. anatomy check =="
python -m src.gradcam --checkpoint results/dense_all__frame__seed0/best.pt \
    --mask-dir data/DatasetV3/train/seg --n 128 --out-dir report/gradcam

echo "== 9. tables + figures =="
python -m src.analyze --results-dir results --out-dir report

echo "== 10. the three comparisons the paper turns on =="
# A: does density help at natural size?
python -m src.analyze --results-dir results --out-dir report \
    --compare sparse_k5__frame__seed0 dense_all__frame__seed0
# B: does it still help at a matched frame budget?
python -m src.analyze --results-dir results --out-dir report \
    --compare sparse_k5__frame__seed0 dense_matched_k5__frame__seed0
# C: does the LSTM beat frame-wise + smoothing? (the one that decides Arm 2)
python -m src.analyze --results-dir results --out-dir report \
    --compare dense_all__frame__seed0 dense_all__temporal_lstm_bi__seed0

echo "done -> report/"
