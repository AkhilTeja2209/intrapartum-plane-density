"""Datasets. Two views over the same index: single frames, and clips.

Both arms must see identical pixels for the comparison to mean anything, so
the transform pipeline lives here once and is shared. The only difference
between arms is the shape of what comes out: (3,H,W) vs (T,3,H,W).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Ultrasound B-mode is single-channel greyscale replicated to 3 channels so we
# can use ImageNet-pretrained weights. ImageNet statistics are not ideal for
# greyscale medical images, but they are what the pretrained filters expect,
# and using them consistently across both arms keeps the comparison fair.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True):
    """Augmentations chosen for transperineal ultrasound specifically.

    Deliberately ABSENT:
      * Horizontal flip. The mid-sagittal view has a fixed handedness -- pubic
        symphysis anterior/near-field, fetal head posterior. Mirroring produces
        an anatomically impossible image and teaches the model that the
        left-right arrangement is irrelevant, when it is in fact the main cue.
      * Vertical flip / large rotation, for the same reason. The probe is held
        against the perineum; the depth axis is not arbitrary.
      * Colour jitter beyond brightness/contrast. There is no colour.

    Deliberately PRESENT:
      * Small rotation and translation -- real probe handling varies by a few
        degrees and millimetres.
      * Brightness/contrast -- gain and TGC settings differ across the three
        hospitals and scanner models in this dataset. This is the augmentation
        most likely to matter for cross-centre generalisation.
      * Mild Gaussian blur -- stands in for focus/frequency differences.
    """
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0),
                                         ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FrameDataset(Dataset):
    """One frame -> one label. The frame-wise arm."""

    def __init__(self, df: pd.DataFrame, frames_dir: str | Path,
                 img_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.root = Path(frames_dir)
        self.tf = build_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r.frame_path).convert("L")
        return self.tf(img), int(r.label), i


class ClipDataset(Dataset):
    """A window of T consecutive frames -> T labels. The temporal arm.

    Windows never cross a video boundary. Short videos are padded by repeating
    the edge frame, and the padded positions are masked out of the loss so the
    model is not scored on frames that do not exist.

    train=True  : windows are drawn with `stride` overlap, order preserved.
    train=False : `stride` should equal T (or less) for full coverage; the
                  evaluator averages logits over overlapping windows.
    """

    def __init__(self, df: pd.DataFrame, frames_dir: str | Path,
                 clip_len: int = 16, stride: int = 8, img_size: int = 224,
                 train: bool = True, temporal_stride: int = 1):
        self.root = Path(frames_dir)
        self.T = clip_len
        self.tf = build_transforms(img_size, train)
        self.temporal_stride = temporal_stride
        self.df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

        # Precompute window start offsets into self.df (row positions, so the
        # dataset never has to search at __getitem__ time).
        self.windows: list[np.ndarray] = []
        span = (clip_len - 1) * temporal_stride + 1
        for _, g in self.df.groupby("video_id", sort=True):
            pos = g.index.to_numpy()
            n = len(pos)
            if n <= span:
                idx = np.arange(0, n, temporal_stride)[:clip_len]
                idx = np.pad(idx, (0, clip_len - len(idx)), mode="edge")
                self.windows.append(pos[idx])
                continue
            starts = list(range(0, n - span + 1, stride))
            if starts[-1] != n - span:
                starts.append(n - span)   # always cover the tail
            for s in starts:
                self.windows.append(pos[s:s + span:temporal_stride][:clip_len])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, w):
        rows = self.windows[w]
        imgs, labels = [], []
        for p in rows:
            r = self.df.iloc[p]
            img = Image.open(self.root / r.frame_path).convert("L")
            imgs.append(self.tf(img))
            labels.append(int(r.label))
        return (torch.stack(imgs),                      # (T,3,H,W)
                torch.tensor(labels, dtype=torch.long),  # (T,)
                torch.tensor(rows, dtype=torch.long))    # (T,) row ids for eval


class SplicedClipDataset(ClipDataset):
    """ClipDataset that synthesises label transitions. TRAIN ONLY.

    Why this exists
    ---------------
    The official train split contains 266 trimmed standard-plane clips and 168
    trimmed non-standard clips, and **not one label transition** -- 0.00 per
    video, against 0.93 in val and 1.35 in test (docs/DATA_AUDIT.md S2). A
    temporal model trained there can reach zero training loss by ignoring time
    entirely and emitting one constant per clip, because within any window the
    label never changes. It never observes the only event it exists to model.
    Meanwhile the smoothed frame-wise baseline gets its window and Viterbi
    `p_stay` tuned on validation, which does contain transitions. Arm 2 run on
    the raw split would therefore measure the split, not the architecture.

    What this does
    --------------
    With probability `splice_p`, a training window is built by concatenating a
    contiguous run from a positive video with a contiguous run from a negative
    video. The labels come from the source frames, so the window contains a
    genuine 0->1 or 1->0 boundary at a position the model cannot predict.

    Partners are drawn from the **same acquisition session** where one exists
    (42 of 144 sessions hold both classes, covering 232 of 434 train videos),
    falling back to a random partner otherwise. Same-session splices share
    patient, probe, scanner and gain settings, so the join is a plausible
    off-plane movement rather than a cut between two unrelated recordings.

    Honesty about what this is
    --------------------------
    A spliced boundary is a hard cut. A real probe sweeping off plane passes
    through intermediate frames that are genuinely ambiguous, and no splice
    reproduces those. So this teaches a temporal model that labels change and
    roughly how long runs last -- which is exactly the prior the smoothing
    baseline gets for free -- without teaching it what a real transition looks
    like. It makes Arm 2 measurable, not realistic, and the paper must say so.
    `splice_p` is a reported hyperparameter, and `splice_p=0` recovers the
    unspliced baseline for the ablation.
    """

    def __init__(self, df: pd.DataFrame, frames_dir: str | Path,
                 clip_len: int = 16, stride: int = 8, img_size: int = 224,
                 temporal_stride: int = 1, splice_p: float = 0.5,
                 seed: int = 0, same_session_p: float = 0.8,
                 min_side: float = 0.25):
        super().__init__(df, frames_dir, clip_len, stride, img_size,
                         True, temporal_stride)
        self.splice_p = float(splice_p)
        self.same_session_p = float(same_session_p)
        self.seed = int(seed)
        # at least this fraction of the clip on each side of the boundary, so
        # a spliced window always carries a substantial run of both classes
        self.min_side = float(min_side)
        self._rng_cache: dict[int, np.random.Generator] = {}

        # per-video row positions (already sorted by frame_idx in super()) and
        # the video's single label
        self.vid_rows: dict[str, np.ndarray] = {}
        self.vid_label: dict[str, int] = {}
        for vid, g in self.df.groupby("video_id", sort=True):
            self.vid_rows[vid] = g.index.to_numpy()
            self.vid_label[vid] = int(round(float(g.label.mean())))

        self.pos_vids = [v for v, l in self.vid_label.items() if l == 1]
        self.neg_vids = [v for v, l in self.vid_label.items() if l == 0]

        # session key: the acquisition timestamp prefix shared by videos
        # recorded in one sitting (20190909T155747I1, ...I5, ...I9).
        import re as _re

        def session_of(vid: str) -> str:
            m = _re.match(r"^(\d{8}T\d{6})", str(vid))
            return m.group(1) if m else str(vid)[:15]

        self.session: dict[str, str] = {v: session_of(v) for v in self.vid_rows}
        self.by_session: dict[tuple[str, int], list[str]] = {}
        for v, s in self.session.items():
            self.by_session.setdefault((s, self.vid_label[v]), []).append(v)

        self.window_vid = [str(self.df.iloc[rows[0]].video_id)
                           for rows in self.windows]

        if not self.pos_vids or not self.neg_vids:
            raise ValueError(
                "splicing needs both positive and negative training videos; "
                f"found {len(self.pos_vids)} positive and {len(self.neg_vids)} "
                f"negative. Use ClipDataset (splice_p=0) instead.")

    def _rng(self) -> np.random.Generator:
        """One generator per worker, seeded so runs stay reproducible."""
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else int(info.id)
        if wid not in self._rng_cache:
            self._rng_cache[wid] = np.random.default_rng(
                [self.seed, wid, 0x5911CE])
        return self._rng_cache[wid]

    def _partner(self, vid: str, rng: np.random.Generator) -> str:
        """A video of the opposite class, same session when one exists."""
        want = 1 - self.vid_label[vid]
        same = self.by_session.get((self.session[vid], want), [])
        if same and rng.random() < self.same_session_p:
            return str(rng.choice(same))
        pool = self.pos_vids if want == 1 else self.neg_vids
        return str(rng.choice(pool))

    def _run(self, vid: str, n: int, rng: np.random.Generator,
             from_start: bool) -> np.ndarray:
        """n contiguous row positions from `vid`, edge-padded if it is short.

        from_start=False takes a run ending at a random point (the frames that
        lead INTO the boundary); from_start=True takes one beginning at a
        random point (the frames that follow it).
        """
        rows = self.vid_rows[vid]
        if len(rows) <= n:
            pad = np.full(n - len(rows), rows[-1] if from_start else rows[0])
            return np.concatenate([rows, pad] if from_start else [pad, rows])
        s = int(rng.integers(0, len(rows) - n + 1))
        return rows[s:s + n]

    def __getitem__(self, w):
        rng = self._rng()
        if rng.random() >= self.splice_p:
            return super().__getitem__(w)

        vid_a = self.window_vid[w]
        vid_b = self._partner(vid_a, rng)

        lo = max(1, int(round(self.T * self.min_side)))
        hi = self.T - lo
        cut = int(rng.integers(lo, hi + 1)) if hi >= lo else self.T // 2

        # Randomise which class leads, so the model sees 0->1 and 1->0 equally
        # and cannot learn "the clip starts positive".
        if rng.random() < 0.5:
            first, n_first, second, n_second = vid_a, cut, vid_b, self.T - cut
        else:
            first, n_first, second, n_second = vid_b, cut, vid_a, self.T - cut

        rows = np.concatenate([self._run(first, n_first, rng, from_start=False),
                               self._run(second, n_second, rng, from_start=True)])

        imgs, labels = [], []
        for p in rows:
            r = self.df.iloc[int(p)]
            imgs.append(self.tf(Image.open(self.root / r.frame_path).convert("L")))
            labels.append(int(r.label))
        return (torch.stack(imgs),
                torch.tensor(labels, dtype=torch.long),
                # Row ids are only read during evaluation, and splicing is
                # train-only. -1 makes a leak into eval fail loudly.
                torch.full((self.T,), -1, dtype=torch.long))


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss.

    Standard planes are a minority of frames, and -- crucially -- the ratio
    DIFFERS between the sparse and dense conditions (a curated one-frame-per-
    video set is enriched for standard planes; a dense set is not). Without
    reweighting, part of any measured difference would just be a difference in
    class prior, not in what the model learned. Applying the same
    inverse-frequency rule to every condition removes that confound.
    """
    counts = np.bincount(df.label.values.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)
