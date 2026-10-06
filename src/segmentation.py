"""Pubic-symphysis / fetal-head masks: indexing, loading, and a UNet dataset.

The dataset ships segmentation masks for a subset of frames, and they are the
only ground truth available for *where* the anatomy is. Two things make them
awkward enough to be worth one module:

1. **Three different file layouts**, one per split::

       train   seg/<video>/mask/<video>_<frame>_<suffix>.png
       val     seg/<video>.png                 (frame from val_info.csv)
       test    seg/<video>_<frame>.png

2. **Two different label encodings.** train masks use ``{0, 7, 8}``; val and
   test masks use ``{0, 127, 255}``. Nothing in the dataset documentation says
   so. Reading a test mask with the train encoding silently yields an empty
   mask -- every area measurement comes out zero and no error is raised, which
   is the same failure mode that cost this project its label join.

Class identity was established by area, not by assumption: across every split
the smaller class covers ~1.5-1.8% of the frame and the larger ~13-15%. The
pubic symphysis is a small hypoechoic oval and the fetal head is a large
shadowing arc, so small = PS and large = FH.

    python -m src.segmentation --dataset-root data/DatasetV3 --out data/masks.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .build_index import canonical_video_id, read_csv_any_encoding
from .utils import get_logger

log = get_logger("seg")

# label value -> class index, per split encoding. 1 = pubic symphysis,
# 2 = fetal head. Verified by area fraction, see the module docstring.
ENCODINGS = {
    "train": {7: 1, 8: 2},
    "val": {127: 1, 255: 2},
    "test": {127: 1, 255: 2},
}
CLASS_NAMES = {0: "background", 1: "pubic_symphysis", 2: "fetal_head"}
N_CLASSES = 3


def decode_mask(arr: np.ndarray, split: str) -> np.ndarray:
    """Map raw mask values onto {0,1,2} using this split's encoding."""
    enc = ENCODINGS.get(split)
    if enc is None:
        raise ValueError(f"no mask encoding known for split {split!r}")
    out = np.zeros(arr.shape, dtype=np.uint8)
    seen = set(np.unique(arr).tolist()) - {0}
    unknown = seen - set(enc)
    if unknown:
        raise ValueError(
            f"mask for split {split!r} contains value(s) {sorted(unknown)} that "
            f"the {split} encoding {enc} does not cover. Mask encodings differ "
            f"between splits; check src/segmentation.ENCODINGS.")
    for raw, cls in enc.items():
        out[arr == raw] = cls
    return out


def _split_video_frame(stem: str, known: set[str]) -> tuple[str, int] | None:
    """Pull (video_id, frame_idx) out of a mask filename.

    Video ids themselves contain underscores and digits
    (``20190726T095643_0``), so splitting on ``_`` is ambiguous. Match against
    the known video ids instead, longest first, and parse what is left.
    """
    for vid in sorted(known, key=len, reverse=True):
        if stem == vid:
            return vid, -1                      # frame comes from the info CSV
        if stem.startswith(vid + "_"):
            rest = stem[len(vid) + 1:]
            nums = re.findall(r"\d+", rest)
            if nums:
                return vid, int(nums[0])
    return None


def build_mask_index(dataset_root: str | Path, index_csv: str | Path) -> pd.DataFrame:
    """One row per mask: video_id, frame_idx, split, mask_path."""
    root = Path(dataset_root)
    frames = pd.read_csv(index_csv)
    known = {s: set(frames[frames.orig_split == s].video_id)
             for s in ("train", "val", "test")}

    rows, unmatched = [], 0
    for split in ("train", "val", "test"):
        seg = root / split / "seg"
        if not seg.exists():
            continue
        files = sorted(seg.glob("*/mask/*.png")) if split == "train" \
            else sorted(seg.glob("*.png"))

        # val masks are named after the video only; the annotated frame index
        # lives in the split's info CSV.
        frame_of = {}
        if split == "val":
            info = read_csv_any_encoding(root / split / f"{split}_info.csv")
            fn = next(c for c in info.columns if c.lower().strip() == "filename")
            fi = next(c for c in info.columns
                      if "labeled_frame_index" in c.lower())
            for _, r in info.iterrows():
                v = canonical_video_id(Path(str(r[fn])).stem)
                try:
                    frame_of[v] = int(str(r[fi]).split(",")[0])
                except (TypeError, ValueError):
                    pass

        for f in files:
            hit = _split_video_frame(canonical_video_id(f.stem), known[split])
            if hit is None:
                unmatched += 1
                continue
            vid, idx = hit
            if idx < 0:
                idx = frame_of.get(vid, -1)
                if idx < 0:
                    unmatched += 1
                    continue
            rows.append({"video_id": vid, "frame_idx": idx, "orig_split": split,
                         "mask_path": str(f)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no masks found under {root}")

    # Every mask must line up with a real extracted frame, or it is useless.
    key = frames.set_index(["video_id", "frame_idx"]).index
    df["has_frame"] = pd.MultiIndex.from_frame(
        df[["video_id", "frame_idx"]]).isin(key)
    bad = int((~df.has_frame).sum())

    log.info("masks found: %d (train %d / val %d / test %d)", len(df),
             *(int((df.orig_split == s).sum()) for s in ("train", "val", "test")))
    if unmatched:
        log.warning("%d mask file(s) could not be matched to a video id", unmatched)
    if bad:
        log.warning("%d mask(s) reference a frame that was not extracted; "
                    "dropping them", bad)
    df = df[df.has_frame].drop(columns=["has_frame"]).reset_index(drop=True)
    log.info("usable masks: %d over %d videos", len(df), df.video_id.nunique())
    return df


class SegDataset(Dataset):
    """Frame + PS/FH mask, for training a segmentation network.

    Augmentation is geometric-only and applied identically to image and mask.
    The same handedness argument as the classifier applies: a mid-sagittal
    transperineal view has the symphysis in the near field and the head behind
    it, so mirroring produces an anatomically impossible image.
    """

    def __init__(self, df: pd.DataFrame, frames_dir: str | Path,
                 img_size: int = 256, train: bool = True, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.root = Path(frames_dir)
        self.size = img_size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r.frame_path).convert("L")
        raw = np.array(Image.open(r.mask_path))
        mask = Image.fromarray(decode_mask(raw, r.orig_split))

        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.train:
            # small rotation + translation, matching the classifier's
            # augmentation budget; nothing that changes handedness
            ang = float(self.rng.uniform(-10, 10))
            dx = int(self.rng.integers(-12, 13))
            dy = int(self.rng.integers(-12, 13))
            img = img.rotate(ang, Image.BILINEAR, translate=(dx, dy))
            mask = mask.rotate(ang, Image.NEAREST, translate=(dx, dy))
            if self.rng.random() < 0.7:
                a = float(self.rng.uniform(0.8, 1.25))
                b = float(self.rng.uniform(-25, 25))
                img = Image.fromarray(
                    np.clip(np.asarray(img, np.float32) * a + b, 0, 255)
                    .astype(np.uint8))

        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
        y = torch.from_numpy(np.asarray(mask, np.int64))
        return (x - 0.449) / 0.226, y          # ImageNet grey mean/std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/DatasetV3")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/masks.csv")
    a = ap.parse_args()

    df = build_mask_index(a.dataset_root, a.index)
    frames = pd.read_csv(a.index)[["video_id", "frame_idx", "frame_path", "label"]]
    df = df.merge(frames, on=["video_id", "frame_idx"], how="left")

    log.info("standard-plane rate among masked frames: %.4f", df.label.mean())
    if df.label.mean() < 0.99:
        log.info("  (masks on non-standard frames exist; the segmenter sees "
                 "both, which is what makes it usable as a plane detector)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
