"""Decode IUGC videos to pre-resized JPEG frames.

Why pre-extract instead of decoding on the fly:
  * 68k frames x ~25 epochs x several conditions means the same frames are
    decoded hundreds of times. mp4 seek/decode is the bottleneck, not the GPU.
  * Pre-resizing to a fixed short side (default 256, we random-crop to 224)
    cuts disk I/O by ~10x versus full-resolution frames.
  * Frame indices become stable and reproducible, which matters because the
    label CSV is keyed by frame index.

Usage:
    python -m src.extract_frames --dataset-root /path/DatasetV3 \
        --out-dir data/frames --short-side 256 --quality 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .utils import get_logger

log = get_logger("extract")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in VIDEO_EXTS)


def assign_ids(videos: list[Path], root: Path) -> dict[Path, str]:
    """Give every video a UNIQUE folder name.

    Using `video.stem` alone is not safe: a dataset can hold two files with
    the same filename in different subfolders. They would then extract into
    the same output folder, and the second would overwrite the first
    frame-by-frame. If the two videos differ in length the survivor is a
    chimera -- the first N frames from one video and the tail from another,
    all under one id, in temporal order. Nothing downstream can detect that,
    and it would corrupt every clip the temporal arm ever sees.

    So: keep the plain stem when it is unique (it matches the label CSV
    directly), and fall back to a path-derived name only for the ones that
    actually clash.
    """
    from collections import Counter

    stems = Counter(v.stem for v in videos)
    ids: dict[Path, str] = {}
    used: set[str] = set()

    for v in videos:
        if stems[v.stem] == 1:
            vid = v.stem
        else:
            # disambiguate with the enclosing folder(s)
            rel = v.relative_to(root)
            parts = list(rel.parts[:-1]) + [v.stem]
            vid = "__".join(parts[-3:])
            n = 1
            while vid in used:
                n += 1
                vid = "__".join(parts[-3:]) + f"__{n}"
        used.add(vid)
        ids[v] = vid
    return ids


def _write_jpeg(path: Path, frame, quality: int) -> None:
    """Encode in memory, then write through Python's file IO.

    cv2.imwrite() passes the path to the C++ layer, which on Windows encodes it
    with the process ANSI codepage. 71 of this dataset's train videos have
    Chinese characters in their filename (..._B_产科_tmp_0), so every write for
    those videos failed -- and imwrite reports failure by RETURNING False, which
    nothing was checking. The extractor counted the decoded frames and recorded
    them in manifest.csv, so the videos looked extracted while their output
    folders were empty, and they vanished from the study without one error.

    imencode does the same JPEG compression but hands back a buffer, and
    Path.write_bytes uses Python's Unicode-aware IO, so the path encoding never
    reaches OpenCV.
    """
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.write_bytes(buf.tobytes())


def extract_one(video: Path, vid_id: str, out_dir: Path, short_side: int,
                quality: int, overwrite: bool = False) -> int:
    """Extract every frame of one video. Returns the frame count written."""
    dst = out_dir / vid_id
    if dst.exists() and not overwrite:
        existing = len(list(dst.glob("*.jpg")))
        if existing > 0:
            return existing
    dst.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning("could not open %s", video)
        return 0

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if min(h, w) != short_side:
            scale = short_side / min(h, w)
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                # INTER_AREA for downscale preserves speckle statistics better
                # than bilinear; ultrasound texture is part of the signal.
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
        # Zero-padded index so lexical sort == temporal sort. The temporal
        # models depend on this.
        _write_jpeg(dst / f"{n:06d}.jpg", frame, quality)
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="Root of the unzipped DatasetV3 (contains train/, val/, test/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be extracted, including filename "
                         "collisions, without decoding anything.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.out_dir)
    videos = find_videos(root)
    log.info("found %d videos under %s", len(videos), root)
    if not videos:
        raise SystemExit(f"No video files under {root}. Check --dataset-root.")

    ids = assign_ids(videos, root)
    n_renamed = sum(1 for v in videos if ids[v] != v.stem)
    if n_renamed:
        log.warning("%d video(s) share a filename with another video and were "
                    "given disambiguated ids:", n_renamed)
        shown = 0
        for v in videos:
            if ids[v] != v.stem and shown < 10:
                log.warning("    %s  ->  %s", v.relative_to(root), ids[v])
                shown += 1
        if n_renamed > 10:
            log.warning("    ... and %d more", n_renamed - 10)
        log.warning("These would have OVERWRITTEN each other under the old "
                    "naming. If you extracted before this fix, delete "
                    "%s and re-extract.", out)

    assert len(set(ids.values())) == len(videos), "id assignment is not unique"

    if args.dry_run:
        from collections import Counter
        by_split = Counter()
        for v in videos:
            try:
                by_split[v.relative_to(root).parts[0]] += 1
            except ValueError:
                by_split["unknown"] += 1
        log.info("dry run -- nothing written")
        log.info("videos per top-level folder: %s", dict(by_split))
        log.info("unique output ids: %d (must equal %d)",
                 len(set(ids.values())), len(videos))
        return

    manifest = []
    total = 0
    for i, v in enumerate(videos, 1):
        # Mirror the split directory (train/val/test) so provenance of each
        # video is recoverable later. We do NOT use the official split for
        # our experiments (see splits.py) but we want to know where it came from.
        try:
            rel_split = v.relative_to(root).parts[0]
        except ValueError:
            rel_split = "unknown"
        n = extract_one(v, ids[v], out / rel_split, args.short_side,
                        args.quality, args.overwrite)
        total += n
        manifest.append({"video_id": ids[v], "source": str(v.relative_to(root)),
                         "split": rel_split, "n_frames": n,
                         "stem": v.stem})
        if i % 25 == 0 or i == len(videos):
            log.info("  %d/%d videos, %d frames so far", i, len(videos), total)

    # The manifest is the record of which source file became which folder.
    # Without it, a disambiguated id is untraceable back to its video.
    import csv
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "stem", "split",
                                          "source", "n_frames"])
        w.writeheader()
        w.writerows(manifest)

    on_disk_videos = sum(1 for p in out.rglob("*") if p.is_dir()
                         and any(p.glob("*.jpg")))
    log.info("done: %d frames from %d videos -> %s", total, len(videos), out)
    log.info("folders on disk containing frames: %d", on_disk_videos)
    log.info("wrote %s", out / "manifest.csv")

    # A video that OpenCV opens but cannot decode yields zero frames with no
    # error. Silent data loss, so name the offenders explicitly.
    empty = [m for m in manifest if m["n_frames"] == 0]
    if empty:
        log.warning("%d video(s) produced ZERO frames:", len(empty))
        for m in empty[:15]:
            log.warning("    %s  (%s)", m["source"], m["split"])
        if len(empty) > 15:
            log.warning("    ... and %d more (all listed in manifest.csv "
                        "with n_frames=0)", len(empty) - 15)
        log.warning("These are unreadable by OpenCV -- usually a codec it was "
                    "not built with. If they are labelled videos you need "
                    "them; try `pip install av` or re-encode with ffmpeg. If "
                    "they are all unlabelled, you can ignore this.")

    if on_disk_videos + len(empty) != len(videos):
        log.error("MISMATCH: %d videos, %d with frames, %d empty. Some "
                  "overwrote each other -- do not proceed.",
                  len(videos), on_disk_videos, len(empty))
    log.info("sanity check: the dataset card says 68,106 frames across 774 "
             "videos. If your total is far off, check for videos that failed "
             "to open (warnings above).")


if __name__ == "__main__":
    main()
