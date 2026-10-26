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

Ground-truth masks exist only for standard planes, so the check could
originally only be run on positives -- which leaves open the possibility that
the model attends to anatomy when there is anatomy and to scanner furniture
when there is not. `--unet` closes that: the segmenter predicts PS/FH masks for
any frame, including negatives, so attention can be measured on both classes.
Predicted masks are not ground truth, so the two are compared on the frames
that have both before the predicted ones are used anywhere alone.

This is the single most likely way this project produces a high number that
means nothing. Three hospitals and several scanner models are represented, and
if standard planes are not uniformly distributed across centres, a model can
score well by learning "this is the scanner from centre 2" and never looking at
the symphysis. The per-centre split that would test this directly is not
available -- the release carries no centre label and every video is re-encoded
to an identical container -- but the splits are separated by acquisition date
instead, so `tools/per_campaign.py` tests the same shortcut across the 2021 and
2024 test campaigns.

Both arms are supported. The temporal arm needs more care than the frame arm:
its encoder sees a clip flattened to (B*T, 3, H, W), so the activations and
gradients for one frame are a slice of that batch, and the backward pass has to
target that frame's logit rather than the clip's.

    # frame-wise arm
    python -m src.gradcam --checkpoint results/dense_all__frame__seed0/best.pt

    # temporal CNN-BiLSTM arm
    python -m src.gradcam --checkpoint results/dense_all__temporal__seed0/best.pt

    # attention on NON-standard frames too, using the U-Net's predicted masks
    python -m src.gradcam --checkpoint results/sparse_k1__frame__seed0/best.pt         --unet results/unet/base_s0/best.pt --on all
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


def load_unet(path: str, device):
    """The segmenter, for predicting masks on frames that have none."""
    from .unet import ResUNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    size = int(cfg.get("img_size", ck.get("img_size", 256)))
    m = ResUNet(3, pretrained=False, in_ch=1, arch=cfg.get("arch", "resnet18"))
    m.load_state_dict(ck["model"])
    return m.eval().to(device), size


@torch.no_grad()
def predict_mask(unet, size: int, img: Image.Image, out_size, device) -> np.ndarray:
    """Binary PS+FH mask predicted for one frame, at `out_size`."""
    a = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    x = torch.from_numpy((a - 0.449) / 0.226)[None, None].to(device)
    pred = unet(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    m = Image.fromarray(pred).resize(out_size, Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


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
    ap.add_argument("--unet", default=None,
                    help="U-Net checkpoint; enables predicted masks, and with "
                         "--on all lets attention be measured on non-standard "
                         "frames, which carry no ground-truth mask")
    ap.add_argument("--on", default="masked", choices=["masked", "all"],
                    help="'masked': the annotated frames only (all standard "
                         "planes). 'all': a class-balanced sample of the split, "
                         "which requires --unet.")
    a = ap.parse_args()
    if a.on == "all" and not a.unet:
        raise SystemExit("--on all needs --unet: non-standard frames have no "
                         "ground-truth mask to measure attention against.")

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

    unet, unet_size = (None, None)
    if a.unet:
        unet, unet_size = load_unet(a.unet, device)
        log_.info("predicted masks from %s (%dpx)", a.unet, unet_size)

    if a.on == "masked":
        sel = masks.sample(min(a.n, len(masks)), random_state=a.seed)
        log_.info("scoring %d annotated frames from the %s split "
                  "(all of them standard planes)", len(sel), a.split)
    else:
        # A class-balanced sample of the split, so the ratio can be reported
        # per class. Negatives have no ground-truth mask; the segmenter
        # supplies one, which is why --on all requires --unet.
        pool = index[index.video_id.isin(splits[a.split])]
        half = a.n // 2
        parts = []
        for lab in (1, 0):
            g = pool[pool.label == lab]
            parts.append(g.sample(min(half, len(g)), random_state=a.seed))
        sel = pd.concat(parts).sample(frac=1.0, random_state=a.seed)
        sel["mask_path"] = None
        sel["orig_split"] = a.split
        log_.info("scoring %d frames from the %s split (%d standard, %d not)",
                  len(sel), a.split, int((sel.label == 1).sum()),
                  int((sel.label == 0).sum()))

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

        grey_img = Image.open(frames_dir / r.frame_path).convert("L")
        gt = (load_mask(r.mask_path, r.orig_split, (a.img_size, a.img_size))
              if isinstance(r.mask_path, str) else None)
        pm = (predict_mask(unet, unet_size, grey_img,
                           (a.img_size, a.img_size), device)
              if unet is not None else None)
        m = gt if gt is not None else pm
        # A frame with no mask, or with an all-zero Grad-CAM, yields no ratio.
        # Record it anyway rather than dropping it: both are informative, and
        # silently shrinking the denominator hides a selection effect. An
        # all-zero map on a non-standard frame in particular is the model
        # having no positive evidence anywhere, which is reasonable behaviour
        # and exactly the frame an attention mean would otherwise omit.
        no_mask = m is None or m.sum() == 0
        zero_cam = float(cam.sum()) == 0.0
        ratios.append({"label": int(r.label), "no_mask": bool(no_mask),
                       "zero_cam": bool(zero_cam),
                       "gt": attention_ratio(cam, gt, seed=a.seed)
                             if (gt is not None and not no_mask) else np.nan,
                       "pred": attention_ratio(cam, pm, seed=a.seed)
                               if (pm is not None and not no_mask) else np.nan})
        if no_mask:
            continue

        if made < a.n_overlays:
            grey = np.array(grey_img.resize((a.img_size, a.img_size)))
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

    df = pd.DataFrame(ratios)

    def summarise(v) -> dict:
        """Always the same shape, so callers never have to test for presence.

        A frame contributes NaN when its Grad-CAM map is entirely zero -- the
        map is ReLU'd, so an untrained or indifferent model can produce one --
        and a whole class can come back empty that way. Returning {} for that
        case made every reader of this dict a crash waiting for the right
        input, which is how it reached CI.
        """
        v = np.asarray([x for x in np.asarray(v, dtype=float)
                        if np.isfinite(x)], dtype=float)
        if not len(v):
            return {"n": 0, "mean": None, "median": None, "std": None,
                    "frac_above_1_5": None, "frac_below_1_1": None}
        return {"n": int(len(v)), "mean": float(v.mean()),
                "median": float(np.median(v)),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "frac_above_1_5": float((v > 1.5).mean()),
                "frac_below_1_1": float((v < 1.1).mean())}

    res = {"checkpoint": str(a.checkpoint),
           "arm": "temporal" if temporal else "frame",
           "split": a.split, "mask_source": a.on, "n_overlays": made,
           "unet": a.unet,
           "ground_truth_masks": summarise(df.get("gt", pd.Series(dtype=float))),
           "predicted_masks": summarise(df.get("pred", pd.Series(dtype=float)))}

    # Predicted masks are not ground truth. Where a frame has both, compare
    # them, so the predicted-mask numbers on negatives are read with a measured
    # idea of how far they can drift rather than on trust.
    both = df.dropna() if {"gt", "pred"} <= set(df.columns) else df.iloc[:0]
    if len(both) > 2:
        res["gt_vs_pred"] = {
            "n": int(len(both)),
            # both["gt"], not both.gt -- DataFrame.gt is the comparison method
            "mean_abs_diff": float((both["gt"] - both["pred"]).abs().mean()),
            "pearson_r": float(np.corrcoef(both["gt"], both["pred"])[0, 1])}
        log_.info("predicted vs ground-truth masks on %d frames that have "
                  "both: r = %.3f, mean |difference| %.3f", len(both),
                  res["gt_vs_pred"]["pearson_r"],
                  res["gt_vs_pred"]["mean_abs_diff"])

    res["frames_scored"] = int(len(df))
    res["frames_without_mask"] = int(df["no_mask"].sum()) if len(df) else 0
    res["frames_with_empty_cam"] = int(df["zero_cam"].sum()) if len(df) else 0
    log_.info("%d frames scored; %d had no mask, %d had an all-zero Grad-CAM",
              res["frames_scored"], res["frames_without_mask"],
              res["frames_with_empty_cam"])

    # per class, which is the point of --on all
    key = "pred" if a.on == "all" else "gt"
    if "label" in df.columns and df.label.nunique() > 1:
        res["by_class"] = {
            ("standard" if lab == 1 else "non_standard"):
                {**summarise(g[key].values), "n_sampled": int(len(g)),
                 "n_empty_cam": int(g["zero_cam"].sum()),
                 "n_no_mask": int(g["no_mask"].sum())}
            for lab, g in df.groupby("label")}
        for name, v in res["by_class"].items():
            if v["n"]:
                log_.info("  %-12s attention %.2f (median %.2f, n=%d of %d "
                          "sampled; %d empty maps, %d without a mask)",
                          name, v["mean"], v["median"], v["n"], v["n_sampled"],
                          v["n_empty_cam"], v["n_no_mask"])
            else:
                log_.warning("  %-12s no frame produced a usable Grad-CAM map "
                             "(%d sampled)", name, v["n_sampled"])

    head = (res["ground_truth_masks"] if res["ground_truth_masks"]["n"]
            else res["predicted_masks"])
    # keep the original flat keys so existing readers of attention.json
    # (docs tables, the write-up) do not silently lose their numbers
    res.update({"n_frames": head.get("n", 0),
                "attention_ratio_mean": head.get("mean"),
                "attention_ratio_median": head.get("median"),
                "attention_ratio_std": head.get("std"),
                "frac_above_1_5": head.get("frac_above_1_5"),
                "frac_below_1_1": head.get("frac_below_1_1")})
    if not head["n"]:
        log_.warning("no frame produced a usable Grad-CAM map; nothing to "
                     "report. Check that the checkpoint and the frames match.")
        save_json(res, out / "attention.json")
        return 0
    log_.info("anatomical attention ratio (%s masks): mean %.2f  median %.2f  "
              "%.0f%% above 1.5", "ground-truth" if key == "gt" else "predicted",
              head["mean"], head["median"], 100 * head["frac_above_1_5"])
    if head["mean"] < 1.3:
        log_.warning("Attention is barely above chance. Before reporting ANY "
                     "accuracy number, check for acquisition shortcuts: score "
                     "the 2021 and 2024 test campaigns separately "
                     "(tools/per_campaign.py) and crop the UI overlay region. "
                     "A per-hospital split is not possible -- IUGC2024 ships "
                     "no centre label and the videos are uniformly "
                     "re-encoded.")
    save_json(res, out / "attention.json")
    log_.info("wrote %d overlays and attention.json to %s", made, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
