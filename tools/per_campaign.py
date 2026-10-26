"""Score each arm separately on the two acquisition campaigns in the test set.

The leave-one-centre-out split that `docs/SEGMENTATION_AND_ATTENTION.md` asks
for cannot be run: IUGC2024 ships no centre or hospital column, and every video
is re-encoded to an identical 512x512 @ 24 fps single-codec container, so no
device signature survives either. What the release *does* carry is acquisition
date, in the filenames, and it separates the splits completely:

    train   434 videos  20190726T095643_0        Jul 2019 - Jan 2020
    test     93 videos  202012132137152159400I1  Dec 2020 - Jul 2021
    test    207 videos  240422145736             Apr 2024 - Jul 2024

So the published numbers are already measured under acquisition shift, and the
test set holds two campaigns 3 years apart. Scoring them separately asks the
question leave-one-centre-out was meant to ask -- does this model survive a
change of acquisition conditions -- using predictions that already exist.

Each arm keeps the threshold it froze on validation; re-tuning per campaign
would be tuning on test. CIs resample videos, not frames.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import bootstrap_ci, frame_metrics  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("per_campaign")

CAMPAIGNS = {"2021": "Dec 2020 - Jul 2021", "2024": "Apr 2024 - Jul 2024"}


def campaign_of(video_id: str) -> str | None:
    """Which acquisition campaign a test video belongs to, from its name.

    The two conventions are unambiguous and disjoint: the 2021 campaign writes
    a 20-digit stamp then an I-suffix, the 2024 one a 12-digit YYMMDDHHMMSS.
    Match on the date prefix and allow any suffix -- six 2024 videos carry a
    trailing "X", and anchoring the whole string silently dropped them.
    """
    v = str(video_id)
    if re.match(r"^20(2[01])\d{16}", v):
        return "2021"
    if re.match(r"^24\d{10}", v):
        return "2024"
    return None


def score(df: pd.DataFrame, thr: float, seed: int = 0) -> dict:
    m = frame_metrics(df.label.values, df.prob.values, thr)
    lo, hi = bootstrap_ci(df, df.prob.values, thr, "macro_f1", seed=seed)
    return {"macro_f1": m["macro_f1"], "lo": lo, "hi": hi,
            "auprc": m.get("auprc"), "recall": m["recall_standard"],
            "precision": m["precision_standard"], "n_frames": m["n"],
            "prevalence": m["prevalence"], "n_videos": int(df.video_id.nunique())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/per_campaign.json"))
    a = ap.parse_args()

    rows, detail = [], {}
    for d in sorted(a.results.glob("*/test_predictions.csv")):
        run = d.parent.name
        rj = d.parent / "results.json"
        if not rj.exists():
            continue
        thr = json.loads(rj.read_text())["results"]["raw"]["threshold"]
        df = pd.read_csv(d)
        df["campaign"] = df.video_id.map(campaign_of)
        unknown = int(df.campaign.isna().sum())
        if unknown:
            log.warning("%s: %d frames with an unrecognised video id", run, unknown)
        df = df.dropna(subset=["campaign"])

        per = {c: score(g, thr) for c, g in df.groupby("campaign")}
        per["all"] = score(df, thr)
        detail[run] = {"threshold": thr, **per}
        rows.append({"run": run, "thr": thr,
                     **{f"{c}_f1": per[c]["macro_f1"] for c in ("2021", "2024", "all")
                        if c in per},
                     "delta": (per["2024"]["macro_f1"] - per["2021"]["macro_f1"])
                              if {"2021", "2024"} <= set(per) else np.nan})

    t = pd.DataFrame(rows).sort_values("all_f1", ascending=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    d = t.delta.dropna()
    log.info("")
    log.info("2024 minus 2021 macro-F1 over %d arms: mean %+.3f, median %+.3f, "
             "range %+.3f to %+.3f", len(d), d.mean(), d.median(), d.min(), d.max())
    log.info("arms where 2024 scores higher: %d of %d", int((d > 0).sum()), len(d))

    a.out.write_text(json.dumps(detail, indent=2))
    log.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
