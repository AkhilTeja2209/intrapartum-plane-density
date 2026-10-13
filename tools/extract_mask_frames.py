"""Re-extract the masked frames at native resolution, losslessly.

`src/extract_frames.py` caches every frame pre-resized to a 256px short side,
which is the right trade for 65,531 frames feeding a classifier. For
segmentation it quietly costs something, because the **masks ship at 512x512
and the source videos are 512x512 too**: training and scoring at 256 means the
pubic symphysis -- about 1.5% of the frame -- is being fitted and measured
against a mask that was downsampled with nearest-neighbour. Published Dice on
this corpus is computed at native resolution, so a 256px number is not
comparable to it.

Only 2,915 frames carry masks, so re-decoding just those at 512 and storing
them as lossless PNG is cheap: one sequential pass per video, keeping the
frames the mask index asks for. Frame indices match `src/extract_frames.py`
exactly -- both count decoded frames from zero in decode order -- which is what
lets the mask index address them.

    python tools/extract_mask_frames.py            # -> data/frames_native/
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_index import canonical_video_id  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("native")


def video_paths(dataset_root: Path) -> dict[str, Path]:
    """video_id -> file, over every split's videos/ directory."""
    out = {}
    for split in ("train", "val", "test"):
        for f in sorted((dataset_root / split / "videos").glob("*")):
            if f.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv"):
                out[canonical_video_id(f.stem)] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="data/masks.csv")
    ap.add_argument("--dataset-root", default="data/DatasetV3")
    ap.add_argument("--out", default="data/frames_native")
    ap.add_argument("--index-out", default="data/masks_native.csv")
    a = ap.parse_args()

    masks = pd.read_csv(ROOT / a.masks)
    out_root = ROOT / a.out
    vids = video_paths(ROOT / a.dataset_root)
    log.info("%d videos on disk, %d masked frames to recover",
             len(vids), len(masks))

    want: dict[str, set[int]] = defaultdict(set)
    for _, r in masks.iterrows():
        want[r.video_id].add(int(r.frame_idx))

    missing_video, written, skipped, sizes = [], 0, 0, defaultdict(int)
    for i, (vid, idxs) in enumerate(sorted(want.items()), 1):
        src = vids.get(vid)
        if src is None:
            missing_video.append(vid)
            continue
        dst_dir = out_root / vid
        todo = {k for k in idxs if not (dst_dir / f"{k:06d}.png").exists()}
        if not todo:
            skipped += len(idxs)
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            log.warning("could not open %s", src)
            missing_video.append(vid)
            continue
        n, last = 0, max(todo)
        while n <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if n in todo:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sizes[grey.shape] += 1
                # cv2.imwrite returns False on a non-ASCII path and raises
                # nothing; several video ids here contain non-ASCII characters,
                # and that silent failure already cost this project 71 videos.
                ok2, buf = cv2.imencode(".png", grey)
                if not ok2:
                    raise RuntimeError(f"PNG encode failed for {vid}:{n}")
                (dst_dir / f"{n:06d}.png").write_bytes(buf.tobytes())
                written += 1
            n += 1
        cap.release()
        if i % 50 == 0:
            log.info("  %d/%d videos, %d frames written", i, len(want), written)

    log.info("wrote %d frames (%d already present) to %s",
             written, skipped, out_root)
    log.info("native frame sizes: %s", dict(sizes))
    if missing_video:
        log.warning("%d video(s) not found on disk: %s",
                    len(missing_video), missing_video[:5])

    # An index pointing at the native frames, with the same columns the
    # 256px mask index has, so SegDataset can take either.
    rows = []
    for _, r in masks.iterrows():
        p = out_root / r.video_id / f"{int(r.frame_idx):06d}.png"
        if p.exists():
            rows.append({**r.to_dict(),
                         "frame_path": str(p.relative_to(out_root))})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / a.index_out, index=False)
    log.info("usable native masked frames: %d / %d (train %d / val %d / test %d)",
             len(df), len(masks),
             *(int((df.orig_split == s).sum()) for s in ("train", "val", "test")))
    if len(df) < len(masks):
        log.warning("%d masked frame(s) could not be recovered at native "
                    "resolution and are absent from %s",
                    len(masks) - len(df), a.index_out)
    log.info("wrote %s", a.index_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
