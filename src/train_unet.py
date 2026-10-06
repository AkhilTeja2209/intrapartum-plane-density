"""Train the PS/FH U-Net, and test whether segmentation can classify.

Two things happen here, and the second is the interesting one.

1. **Segmentation.** Train on the masked training frames, select on validation
   Dice, report per-class Dice on the official test masks. This is the stage-two
   model of the cascade the IUGC challenge is built around, and the number it
   produces is directly comparable to published work on this corpus.

2. **Segmentation-derived classification.** Run the trained segmenter over the
   *whole* test set -- 8,665 frames, most of which it has never seen anything
   like, because masks exist only for standard planes -- and turn its output
   into a plane score: how much pubic symphysis and how much fetal head did it
   find? Threshold that on validation, freeze it, and score it on exactly the
   labels and the test set the ResNet classifier was scored on.

   This is the honest way to answer "can a U-Net do the classification task".
   The architecture alone cannot answer it, because the mask supervision
   contains no negatives at all.

    python -m src.train_unet --epochs 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .datasets import build_transforms
from .metrics import frame_metrics, pick_threshold
from .segmentation import N_CLASSES, SegDataset
from .splits import load_splits
from .unet import ResUNet, combined_loss, dice_per_class
from .utils import get_logger, save_json, set_seed

log = get_logger("unet")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = torch.zeros(N_CLASSES)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += dice_per_class(model(x), y, N_CLASSES).cpu()
        n += 1
    return (tot / max(1, n)).tolist()


@torch.no_grad()
def plane_scores(model, df, frames_dir, img_size, device, batch=64):
    """Score every frame by how much anatomy the segmenter finds.

    A standard plane is defined by both structures being resolvable, so the
    score is the smaller of the two foreground areas: a frame showing a fetal
    head and no symphysis is not a standard plane, and taking the minimum says
    so. Areas are soft (summed probability), not thresholded masks, so the
    score degrades smoothly instead of falling off a cliff.
    """
    model.eval()
    tf = build_transforms(img_size, train=False)
    root = Path(frames_dir)
    out = np.zeros(len(df), dtype=np.float32)

    for s in range(0, len(df), batch):
        chunk = df.iloc[s:s + batch]
        xs = []
        for _, r in chunk.iterrows():
            img = Image.open(root / r.frame_path).convert("L") \
                       .resize((img_size, img_size), Image.BILINEAR)
            a = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
            xs.append((a - 0.449) / 0.226)
        prob = torch.softmax(model(torch.stack(xs).to(device)), dim=1)
        area = prob[:, 1:].mean(dim=(2, 3))            # (B, 2) soft area frac
        out[s:s + len(chunk)] = area.min(dim=1).values.cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="data/masks.csv")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/unet_seg__seed0")
    ap.add_argument("--skip-classification", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_ = get_logger("unet", str(out / "log.txt"))
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    masks = pd.read_csv(a.masks)
    tr = masks[masks.orig_split == "train"]
    va = masks[masks.orig_split == "val"]
    te = masks[masks.orig_split == "test"]
    log_.info("masks: train %d, val %d, test %d (videos %d/%d/%d)",
              len(tr), len(va), len(te), tr.video_id.nunique(),
              va.video_id.nunique(), te.video_id.nunique())

    # Windows exhausts its commit limit when several DataLoader workers each
    # hold a copy of this process; the mask set is small enough to load inline.
    nw = 0
    tr_ld = DataLoader(SegDataset(tr, a.frames_dir, a.img_size, True, a.seed),
                       batch_size=a.batch_size, shuffle=True, num_workers=nw,
                       pin_memory=True, persistent_workers=nw > 0)
    va_ld = DataLoader(SegDataset(va, a.frames_dir, a.img_size, False),
                       batch_size=a.batch_size, num_workers=0)
    te_ld = DataLoader(SegDataset(te, a.frames_dir, a.img_size, False),
                       batch_size=a.batch_size, num_workers=0)

    model = ResUNet(N_CLASSES, pretrained=True, in_ch=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # the symphysis is ~1.5% of a frame and the head ~14%, so weight the loss
    # against the background dominating
    cw = torch.tensor([0.2, 2.0, 1.0], device=device)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for x, y in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = combined_loss(model(x), y, cw, N_CLASSES)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = combined_loss(model(x), y, cw, N_CLASSES)
                loss.backward()
                opt.step()
            run += float(loss) * x.size(0)
        sched.step()

        d = evaluate(model, va_ld, device)
        mean_fg = float(np.mean(d[1:]))
        history.append({"epoch": ep, "loss": run / len(tr_ld.dataset),
                        "val_dice_ps": d[1], "val_dice_fh": d[2],
                        "val_dice_fg": mean_fg})
        log_.info("ep %02d loss %.4f | val Dice  PS %.4f  FH %.4f  mean %.4f",
                  ep, run / len(tr_ld.dataset), d[1], d[2], mean_fg)

        if mean_fg > best:
            best, bad = mean_fg, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                log_.info("early stop at epoch %d (best %.4f)", ep, best)
                break

    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "img_size": a.img_size,
                "n_classes": N_CLASSES, "seed": a.seed}, out / "best.pt")

    te_d = evaluate(model, te_ld, device)
    log_.info("TEST Dice  PS %.4f  FH %.4f  mean %.4f",
              te_d[1], te_d[2], float(np.mean(te_d[1:])))
    results = {"val_dice_best_fg": best,
               "test_dice": {"background": te_d[0], "pubic_symphysis": te_d[1],
                             "fetal_head": te_d[2],
                             "mean_foreground": float(np.mean(te_d[1:]))},
               "n_masks": {"train": len(tr), "val": len(va), "test": len(te)},
               "history": history}

    # ---- can the segmenter classify? ------------------------------------
    if not a.skip_classification:
        index = pd.read_csv(a.index)
        index = index[index.label >= 0]
        sp = load_splits(a.splits)
        vdf = index[index.video_id.isin(sp["val"])].reset_index(drop=True)
        tdf = index[index.video_id.isin(sp["test"])].reset_index(drop=True)

        log_.info("scoring %d val and %d test frames by segmented area",
                  len(vdf), len(tdf))
        vs = plane_scores(model, vdf, a.frames_dir, a.img_size, device)
        ts = plane_scores(model, tdf, a.frames_dir, a.img_size, device)

        thr = pick_threshold(vdf.label.values, vs, "macro_f1")
        m = frame_metrics(tdf.label.values, ts, thr)
        log_.info("segmentation-derived classifier: macroF1 %.4f  balAcc %.4f "
                  "auprc %.4f  (threshold %.4f from validation)",
                  m["macro_f1"], m["balanced_accuracy"], m["auprc"], thr)
        results["derived_classifier"] = {"threshold": float(thr), **m}
        pd.DataFrame({"video_id": tdf.video_id, "frame_idx": tdf.frame_idx,
                      "label": tdf.label, "score": ts}) \
            .to_csv(out / "test_plane_scores.csv", index=False)

    save_json(results, out / "results.json")
    log_.info("wrote %s", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
