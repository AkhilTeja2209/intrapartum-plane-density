"""Dump the label CSV exactly as it is, so the index format can be pinned down.

The label file turned out to be one row per VIDEO with frame-index lists
(`pos_index` / `neg_index`), not one row per frame. Before writing a parser
for those lists we need to see how they are actually written -- comma
separated, bracketed, hyphenated ranges, nested lists, or something else.
Guessing wrong here silently mislabels frames, which is the one failure mode
that never shows up as an error.

    python inspect_labels.py
    python inspect_labels.py --csv data/DatasetV3/train/cls/class_label.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "big5", "cp1252", "latin-1"]


def read_text(p: Path) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return p.read_text(encoding=enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return p.read_text(encoding="latin-1", errors="replace"), "latin-1(forced)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--rows", type=int, default=12)
    a = ap.parse_args()

    if a.csv:
        paths = [Path(a.csv)]
    else:
        paths = sorted(Path("data/DatasetV3").rglob("*.csv"))
    if not paths:
        print("No CSV found under data/DatasetV3. Pass --csv explicitly.")
        return 1

    for p in paths:
        print("\n" + "=" * 70)
        print(f"  {p}")
        print("=" * 70)
        text, enc = read_text(p)
        lines = text.splitlines()
        print(f"encoding: {enc}   |   {len(lines)} lines\n")

        print("--- RAW first lines (exactly as stored) ---")
        for line in lines[:a.rows]:
            print(f"  {line[:300]}")

        if len(lines) > a.rows + 2:
            print("\n--- RAW a few from the middle ---")
            mid = len(lines) // 2
            for line in lines[mid:mid + 4]:
                print(f"  {line[:300]}")

        # Per-column view: the widest value in each column is the most
        # informative, because an index list only shows its real structure
        # once it holds more than one element.
        try:
            import pandas as pd
            df = pd.read_csv(p, encoding=enc)
            print(f"\n--- parsed: {df.shape[0]} rows x {df.shape[1]} cols ---")
            for c in df.columns:
                s = df[c]
                print(f"\n  column {c!r}  (dtype {s.dtype}, "
                      f"{s.isna().sum()} missing)")
                as_str = s.dropna().astype(str)
                if len(as_str):
                    widest = as_str.loc[as_str.str.len().idxmax()]
                    print(f"    longest value ({len(widest)} chars):")
                    print(f"      {widest[:400]}")
                    print(f"    3 sample values: {as_str.head(3).tolist()}")
                    lens = as_str.str.len()
                    print(f"    value length: min={lens.min()} "
                          f"median={int(lens.median())} max={lens.max()}")
        except Exception as e:
            print(f"\n(pandas parse failed: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
