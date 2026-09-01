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
