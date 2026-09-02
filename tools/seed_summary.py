"""Aggregate the headline conditions across seeds.

The single-seed tables looked like a result. They were not: the same condition
moved by up to 0.23 macro-F1 between seeds. This script reports mean +/- std
per condition and the per-seed paired deltas for the Protocol B comparisons, so
the variance is impossible to read past.

    python tools/seed_summary.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Protocol B comparisons: (dense arm, its matched sparse counterpart)
PAIRS = [
    ("dense_matched_k5", "sparse_k5"),
    ("dense_prior_matched_k5", "sparse_k5"),
]


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="site/seed_summary.json")
    a = ap.parse_args()

    by_cond: dict[str, dict[int, float]] = defaultdict(dict)
    for rj in sorted((ROOT / a.results).glob("*/results.json")):
        try:
            r = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("model_type") != "frame":
            continue
        cond = (r.get("condition") or {}).get("name")
        by_cond[cond][int(r["seed"])] = float(r["results"]["raw"]["macro_f1"])

    multi = {c: v for c, v in by_cond.items() if len(v) > 1}
    print(f"{'condition':<26} {'seeds':>5} {'mean':>8} {'std':>7} "
          f"{'min':>7} {'max':>7} {'range':>7}")
    rows = []
    for c in sorted(multi, key=lambda k: -len(multi[k])):
        vals = [multi[c][s] for s in sorted(multi[c])]
        m, sd = mean_std(vals)
        rng = max(vals) - min(vals)
        print(f"{c:<26} {len(vals):>5} {m:>8.4f} {sd:>7.4f} "
              f"{min(vals):>7.4f} {max(vals):>7.4f} {rng:>7.4f}")
        rows.append({"condition": c, "n_seeds": len(vals),
                     "per_seed": {str(s): multi[c][s] for s in sorted(multi[c])},
                     "mean": round(m, 4), "std": round(sd, 4),
                     "range": round(rng, 4)})

    print("\nProtocol B, per seed (dense minus its matched sparse arm):")
    pairs_out = []
    for dense, sparse in PAIRS:
        if dense not in multi or sparse not in multi:
            continue
        seeds = sorted(set(multi[dense]) & set(multi[sparse]))
        deltas = [multi[dense][s] - multi[sparse][s] for s in seeds]
        m, sd = mean_std(deltas)
        signs = {"dense wins": sum(d > 0 for d in deltas),
                 "sparse wins": sum(d < 0 for d in deltas)}
        per = "  ".join(f"seed{s} {d:+.4f}" for s, d in zip(seeds, deltas))
        print(f"  {dense} vs {sparse}")
        print(f"    {per}")
        print(f"    mean {m:+.4f} +/- {sd:.4f}   "
              f"({signs['dense wins']} dense / {signs['sparse wins']} sparse)")
        pairs_out.append({"dense": dense, "sparse": sparse,
                          "per_seed": {str(s): round(d, 4)
                                       for s, d in zip(seeds, deltas)},
                          "mean_delta": round(m, 4), "std_delta": round(sd, 4),
                          "dense_wins": signs["dense wins"],
                          "sparse_wins": signs["sparse wins"]})

    if rows:
        worst = max(rows, key=lambda r: r["range"])
        print(f"\nLargest seed-to-seed range: {worst['condition']} "
              f"{worst['range']:.4f} macro-F1 on identical data.")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"conditions": rows, "protocol_b": pairs_out},
                              indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
