#!/usr/bin/env bash
# Waits for phase 2, then repeats the headline conditions at seeds 1 and 2.
# One seed cannot separate a density effect from run-to-run noise: the seed-0
# spread WITHIN each arm was as large as the gap between arms.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

until grep -q "=== phase 2 complete ===" results/phase2.log 2>/dev/null; do sleep 60; done
echo "phase 2 done, starting extra seeds at $(date +%H:%M:%S)"
python tools/run_grid.py --plan headline --seeds 1,2 --model frame
echo "=== phase 3 complete ==="
