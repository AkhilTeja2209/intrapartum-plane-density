"""Compute the exact Protocol B budgets and write them into the config.

Protocol B only works if the dense arm gets exactly as many training frames as
the sparse arm it is being compared against. That number is not known until
the split exists, so it is computed here rather than hardcoded.

    python -m src.make_budgets --config configs/default.yaml --write
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from .sampling import build_condition
from .splits import load_splits
from .utils import get_logger

log = get_logger("budgets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--write", action="store_true",
                    help="Patch the budget values in the config file in place")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    index = pd.read_csv(cfg["paths"]["index_csv"])
    index = index[index.label >= 0]
    splits = load_splits(cfg["paths"]["splits_json"])
    train_full = index[index.video_id.isin(splits["train"])]

    sizes = {}
    for c in cfg["conditions"]:
        if c.get("budget"):
            continue
        sizes[c["name"]] = len(build_condition(train_full, c, seed=0))

    print("\n  condition                frames   videos")
    print("  " + "-" * 42)
    for c in cfg["conditions"]:
        if c.get("budget"):
            continue
        d = build_condition(train_full, c, seed=0)
        print(f"  {c['name']:<22} {len(d):>7}   {d.video_id.nunique():>6}")

    print("\nProtocol B budgets (dense arm capped to the sparse arm's count):")
    patches = {}
    for c in cfg["conditions"]:
        if not c.get("budget"):
            continue
        m = re.search(r"matched_(k\d+)$", c["name"])
        if not m:
            log.warning("cannot infer a source condition for %s", c["name"])
            continue
        src = f"sparse_{m.group(1)}"
        if src not in sizes:
            log.warning("%s references missing condition %s", c["name"], src)
            continue
        patches[c["name"]] = sizes[src]
        print(f"  {c['name']:<22} budget = {sizes[src]}   (matches {src})")

    if args.write and patches:
        with open(args.config) as f:
            text = f.read()
        for name, budget in patches.items():
            # replace the budget line inside that condition's block only
            pat = re.compile(rf"(- name: {re.escape(name)}\n(?:.*\n)*?\s*budget:\s*)\d+")
            text = pat.sub(lambda mo: mo.group(1) + str(budget), text, count=1)
        with open(args.config, "w") as f:
            f.write(text)
        log.info("patched %d budgets into %s", len(patches), args.config)


if __name__ == "__main__":
    main()
