"""Grad-CAM, and a quantitative check that the model looks at anatomy.

Qualitative heatmaps in a paper are close to worthless -- you can always find
five that look convincing. This module turns the check into a number.

The dataset ships pubic-symphysis / fetal-head segmentation masks for standard
planes. That lets us compute an **anatomical attention ratio**: the fraction of
total Grad-CAM activation that falls inside the PS+FH mask, versus what you
would get from a random mask of equal area. A ratio near 1.0 means the model
is not using the anatomy at all -- it is reading depth markers, the scanner
vendor's UI overlay, the sector fan shape, or speckle statistics that happen
to correlate with which hospital the video came from.

This is the single most likely way this project produces a high number that
does not mean anything. Three hospitals and multiple scanner models are
represented, and if standard-plane frames are not uniformly distributed across
centres, a model can score well by learning "this is the Hitachi from centre
2" and never looking at the symphysis.

    python -m src.gradcam --checkpoint results/dense_all__frame__seed0/best.pt \
        --n 64 --out-dir report/gradcam
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .datasets import build_transforms
from .models import build
from .splits import load_splits
from .utils import get_logger, save_json

log = get_logger("gradcam")


class GradCAM:
    """Grad-CAM on the last conv block. Works for any of the ResNet variants."""

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        enc = model.encoder
        layer = target_layer or getattr(enc, "layer4", None) or list(enc.children())[-3]
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls: int = 1) -> np.ndarray:
        """x: (1,3,H,W) -> (H,W) heatmap in [0,1]."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out if out.dim() == 2 else out[:, out.shape[1] // 2, :]
        logits[0, cls].backward()

        w = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP over space
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


def attention_ratio(cam: np.ndarray, mask: np.ndarray,
                    n_random: int = 20, seed: int = 0) -> float:
    """CAM mass inside the anatomy mask / expected mass in a random mask of
    the same area. 1.0 = no anatomical preference; >1 = attends to anatomy."""
    mask = mask.astype(bool)
    if mask.sum() == 0 or cam.sum() == 0:
        return float("nan")
    inside = cam[mask].sum() / cam.sum()

    rng = np.random.default_rng(seed)
    area = int(mask.sum())
    flat = cam.ravel()
    baseline = np.mean([flat[rng.choice(flat.size, area, replace=False)].sum()
                        / flat.sum() for _ in range(n_random)])
    return float(inside / (baseline + 1e-8))


def load_mask(mask_dir: Path, video_id: str, frame_idx: int,
              size) -> np.ndarray | None:
    """Load a PS/FH mask if one exists for this frame. Any nonzero label is
    treated as anatomy (the dataset uses distinct ids for PS and FH)."""
    for pat in (f"{video_id}/{frame_idx:06d}.png", f"{video_id}_{frame_idx:06d}.png",
                f"{video_id}/{frame_idx}.png"):
        p = mask_dir / pat
        if p.exists():
            m = Image.open(p).convert("L").resize(size, Image.NEAREST)
            return (np.array(m) > 0).astype(np.uint8)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--mask-dir", default=None,
                    help="Directory of PS/FH segmentation masks. Without it, "
                         "only qualitative overlays are produced.")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out-dir", default="report/gradcam")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    cam_fn = GradCAM(model)

    df = pd.read_csv(args.index)
    df = df[df.label == 1]
    splits = load_splits(args.splits)
    df = df[df.video_id.isin(splits["test"])]
    df = df.sample(min(args.n, len(df)), random_state=args.seed)

    tf = build_transforms(args.img_size, train=False)
    frames_dir, ratios = Path(args.frames_dir), []
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    for _, r in df.iterrows():
        img = Image.open(frames_dir / r.frame_path).convert("L")
        x = tf(img).unsqueeze(0).to(device)
        cam = cam_fn(x, cls=1)

        if mask_dir is not None:
            m = load_mask(mask_dir, r.video_id, int(r.frame_idx),
                          (args.img_size, args.img_size))
            if m is not None:
                ratios.append(attention_ratio(cam, m))

        base = np.array(img.resize((args.img_size, args.img_size)))
        heat = (cam * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([base, heat], axis=1)).save(
            out / f"{r.video_id}_{int(r.frame_idx):06d}.png")

    res = {"n_overlays": len(df), "n_with_mask": len(ratios)}
    if ratios:
        arr = np.array([x for x in ratios if np.isfinite(x)])
        res.update(attention_ratio_mean=float(arr.mean()),
                   attention_ratio_median=float(np.median(arr)),
                   frac_above_1_5=float((arr > 1.5).mean()))
        log.info("anatomical attention ratio: mean %.2f  median %.2f  "
                 "%.0f%% of frames above 1.5",
                 arr.mean(), np.median(arr), 100 * (arr > 1.5).mean())
        if arr.mean() < 1.3:
            log.warning("Attention is barely above chance. Before reporting "
                        "ANY accuracy number, check for centre/scanner "
                        "shortcuts: train on two hospitals and test on the "
                        "third, and crop the UI overlay region.")
    save_json(res, out / "attention.json")
    log.info("wrote %d overlays to %s", len(df), out)


if __name__ == "__main__":
    main()
