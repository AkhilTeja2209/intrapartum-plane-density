"""Grad-CAM, and a quantitative check that the model looks at anatomy.

Qualitative heatmaps in a paper are close to worthless -- you can always find
five that look convincing. This module turns the check into a number.

The dataset ships pubic-symphysis / fetal-head segmentation masks for standard
planes. That lets us compute an **anatomical attention ratio**: the fraction of
total Grad-CAM activation that falls inside the PS+FH mask, divided by what an
equal-area random region would collect. A ratio near 1.0 means the model is not
using the anatomy at all -- it is reading depth markers, the scanner's UI
overlay, the sector fan shape, or speckle statistics that happen to correlate
with which hospital the video came from.

This is the single most likely way this project produces a high number that
means nothing. Three hospitals and several scanner models are represented, and
if standard planes are not uniformly distributed across centres, a model can
score well by learning "this is the scanner from centre 2" and never looking at
the symphysis.

Both arms are supported. The temporal arm needs more care than the frame arm:
its encoder sees a clip flattened to (B*T, 3, H, W), so the activations and
gradients for one frame are a slice of that batch, and the backward pass has to
target that frame's logit rather than the clip's.

    # frame-wise arm
    python -m src.gradcam --checkpoint results/dense_all__frame__seed0/best.pt

    # temporal CNN-BiLSTM arm
    python -m src.gradcam --checkpoint results/dense_all__temporal__seed0/best.pt
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
from .segmentation import decode_mask
from .splits import load_splits
from .utils import get_logger, save_json

log = get_logger("gradcam")


class GradCAM:
    """Grad-CAM on the encoder's last conv block, for either arm.

    For the temporal model the encoder is applied to a flattened clip, so the
    captured activations have shape (T, C, h, w) for a single clip and the
    frame of interest is one row of that. `frame_in_clip` selects it on both
    the forward activations and the backward gradients.
    """

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

    def __call__(self, x, cls: int = 1, frame_in_clip: int | None = None,
                 out_size=None) -> np.ndarray:
        """x: (1,3,H,W) for the frame arm, or (1,T,3,H,W) for the temporal arm.

        Returns an (H,W) heatmap in [0,1] for the selected frame.
        """
        self.model.zero_grad(set_to_none=True)
        # cuDNN refuses to run an RNN backward pass outside training mode, and
        # switching the model to train() would turn the head's dropout back on
        # -- which would change the very attribution being visualised. Running
        # the recurrent layers without cuDNN keeps the model in eval mode.
        with torch.backends.cudnn.flags(enabled=False):
            out = self.model(x)

            if out.dim() == 3:                   # (B, T, C) -- temporal arm
                t = out.shape[1] // 2 if frame_in_clip is None else frame_in_clip
                target = out[0, t, cls]
            else:                                # (B, C) -- frame arm
                t = None
                target = out[0, cls]
            target.backward()

        acts, grads = self.acts, self.grads
        if t is not None:                        # pick this frame out of the clip
            acts, grads = acts[t:t + 1], grads[t:t + 1]

        w = grads.mean(dim=(2, 3), keepdim=True)         # GAP over space
        cam = F.relu((w * acts).sum(dim=1, keepdim=True))
        size = out_size or x.shape[-2:]
        cam = F.interpolate(cam, size=size, mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


def attention_ratio(cam: np.ndarray, mask: np.ndarray,
                    n_random: int = 20, seed: int = 0) -> float:
    """CAM mass inside the anatomy mask / expected mass in a random region of
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


def load_mask(mask_path: str, split: str, size) -> np.ndarray:
    """Binary PS+FH mask at `size`, decoded with this split's encoding."""
    raw = np.array(Image.open(mask_path))
    dec = decode_mask(raw, split)
    m = Image.fromarray(dec).resize(size, Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def overlay(grey: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heatmap over the frame, so a reader can see what the number measures."""
    import matplotlib
    rgb = np.stack([grey] * 3, axis=-1).astype(np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    blend = (1 - alpha) * rgb + alpha * heat
    return Image.fromarray((np.clip(blend, 0, 1) * 255).astype(np.uint8))


def build_clip(index: pd.DataFrame, video_id: str, frame_idx: int, clip_len: int):
    """A clip of `clip_len` frames centred on frame_idx, clamped to the video."""
    g = index[index.video_id == video_id].sort_values("frame_idx")
    pos = int(np.searchsorted(g.frame_idx.values, frame_idx))
    half = clip_len // 2
    start = max(0, min(pos - half, len(g) - clip_len))
    start = max(0, start)
    rows = g.iloc[start:start + clip_len]
    if len(rows) < clip_len:                     # short video: repeat the edge
        pad = rows.iloc[[-1]].copy()
        rows = pd.concat([rows] + [pad] * (clip_len - len(rows)), ignore_index=True)
    centre = int(np.clip(pos - start, 0, clip_len - 1))
    return rows.reset_index(drop=True), centre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--masks", default="data/masks.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-overlays", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    temporal = cfg.get("model", {}).get("type", "frame") == "temporal"
    out = Path(a.out_dir or f"report/gradcam/{Path(a.checkpoint).parent.name}")
    out.mkdir(parents=True, exist_ok=True)
    log_ = get_logger("gradcam", str(out / "log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    cam_fn = GradCAM(model)
    clip_len = int(cfg["data"].get("clip_len", 16)) if temporal else 1
    log_.info("checkpoint=%s arm=%s clip_len=%d", a.checkpoint,
              "temporal" if temporal else "frame", clip_len)

    index = pd.read_csv(a.index)
    index = index[index.label >= 0]
    splits = load_splits(a.splits)
    masks = pd.read_csv(a.masks)
    masks = masks[masks.video_id.isin(splits[a.split])]
    if masks.empty:
        raise SystemExit(f"no masks for split {a.split!r}")
    sel = masks.sample(min(a.n, len(masks)), random_state=a.seed)
    log_.info("scoring %d masked frames from the %s split", len(sel), a.split)

    tf = build_transforms(a.img_size, train=False)
    frames_dir = Path(a.frames_dir)
    ratios, made = [], 0

    for _, r in sel.iterrows():
        if temporal:
            rows, centre = build_clip(index, r.video_id, int(r.frame_idx), clip_len)
            imgs = [tf(Image.open(frames_dir / p).convert("L"))
                    for p in rows.frame_path]
            x = torch.stack(imgs).unsqueeze(0).to(device)       # (1,T,3,H,W)
            cam = cam_fn(x, cls=1, frame_in_clip=centre,
                         out_size=(a.img_size, a.img_size))
        else:
            img = Image.open(frames_dir / r.frame_path).convert("L")
            x = tf(img).unsqueeze(0).to(device)
            cam = cam_fn(x, cls=1)

        m = load_mask(r.mask_path, r.orig_split, (a.img_size, a.img_size))
        ratios.append(attention_ratio(cam, m, seed=a.seed))

        if made < a.n_overlays:
            grey = np.array(Image.open(frames_dir / r.frame_path).convert("L")
                            .resize((a.img_size, a.img_size)))
            ov = overlay(grey, cam)
            # outline the ground-truth anatomy so the reader can judge the
            # heatmap against it rather than taking the ratio on trust
            arr = np.array(ov)
            b = np.zeros_like(m, dtype=bool)
            b[1:-1, 1:-1] = (m[1:-1, 1:-1] == 1) & (
                (m[:-2, 1:-1] == 0) | (m[2:, 1:-1] == 0) |
                (m[1:-1, :-2] == 0) | (m[1:-1, 2:] == 0))
            arr[b] = [255, 255, 255]
            Image.fromarray(arr).save(
                out / f"{r.video_id}_{int(r.frame_idx):06d}.png")
            made += 1

    arr = np.array([v for v in ratios if np.isfinite(v)])
    res = {"checkpoint": str(a.checkpoint),
           "arm": "temporal" if temporal else "frame",
           "split": a.split, "n_frames": int(len(arr)),
           "n_overlays": made,
           "attention_ratio_mean": float(arr.mean()),
           "attention_ratio_median": float(np.median(arr)),
           "attention_ratio_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
           "frac_above_1_5": float((arr > 1.5).mean()),
           "frac_below_1_1": float((arr < 1.1).mean())}
    log_.info("anatomical attention ratio: mean %.2f  median %.2f  "
              "%.0f%% above 1.5", arr.mean(), np.median(arr),
              100 * (arr > 1.5).mean())
    if arr.mean() < 1.3:
        log_.warning("Attention is barely above chance. Before reporting ANY "
                     "accuracy number, check for centre/scanner shortcuts: "
                     "train on two hospitals and test on the third, and crop "
                     "the UI overlay region.")
    save_json(res, out / "attention.json")
    log_.info("wrote %d overlays and attention.json to %s", made, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
