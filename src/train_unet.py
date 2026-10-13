"""Train the PS/FH U-Net, and test whether segmentation can classify.

Two things happen here, and the second is the interesting one.

1. **Segmentation.** Train on the masked training frames, select on validation
   Dice, report per-class Dice on the official test masks. This is the stage-two
   model of the cascade the IUGC challenge is built around, and the number it
   produces is directly comparable to published work on this corpus.

2. **Segmentation-derived classification.** Run the trained segmenter over the
   *whole* test set -- 8,665 frames, most of which it has never seen anything
   like, because masks exist only for standard planes -- and turn its output
   into a plane score. Threshold that on validation, freeze it, and score it on
   exactly the labels and the test set the ResNet classifier was scored on.

   This is the honest way to answer "can a U-Net do the classification task".
   The architecture alone cannot answer it, because the mask supervision
   contains no negatives at all.

   Several ways of turning two segmented areas into one plane score are
   defensible, so all of them are computed in the same forward pass and
   reported side by side. Picking the winner after seeing test numbers would be
   selection on test, so the threshold *and* the rule are both chosen on
   validation; `tools/unet_ensemble.py` reports the rest for reference only.

Two Dice estimators are reported, because the choice moves the number by more
than some of the differences being compared:

  ``dice_batchmean``  mean over mini-batches of a batch-pooled Dice. This is
                      what the first run of this script reported.
  ``dice_image``      mean over images of each image's own Dice, counting only
                      images where that class is actually annotated. This is
                      the estimator published PSFHS / IUGC numbers use, so it
                      is the one to compare against 0.893-0.930.

    python -m src.train_unet --epochs 40
    python -m src.train_unet --arch resnet34 --img-size 384 --tta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .metrics import frame_metrics, pick_threshold
from .segmentation import N_CLASSES, SegDataset
from .splits import load_splits
from .unet import ResUNet, combined_loss, dice_loss
from .utils import get_logger, save_json, set_seed

log = get_logger("unet")

# Scales for test-time augmentation. Mirroring is deliberately absent: a
# mid-sagittal transperineal view has the symphysis in the near field and the
# head behind it, so a horizontal flip is an anatomically impossible image --
# the same argument that keeps it out of the training augmentation.
TTA_SCALES = (0.875, 1.0, 1.125)

# Ways of collapsing two segmented areas into one standard-plane score. A
# standard plane needs *both* structures resolvable, which is an AND, so the
# rules that behave like an AND (min, geometric mean) are the principled ones;
# the single-class rules are here to show how much of the signal is just "is
# there a fetal head in frame".
SCORE_RULES = ("min", "geomean", "ps", "fh", "sum")


def area_scores(prob: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-frame plane scores from a (B,3,H,W) probability map."""
    a = prob[:, 1:].mean(dim=(2, 3))                   # (B,2) soft area frac
    ps, fh = a[:, 0], a[:, 1]
    return {"min": torch.minimum(ps, fh), "geomean": (ps * fh).clamp_min(0).sqrt(),
            "ps": ps, "fh": fh, "sum": ps + fh}


@torch.no_grad()
def predict_prob(model, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Softmax probabilities at the input resolution, optionally scale-TTA."""
    if not tta:
        return torch.softmax(model(x), dim=1)
    hw = x.shape[-2:]
    acc = None
    for s in TTA_SCALES:
        xi = x if s == 1.0 else F.interpolate(x, scale_factor=s,
                                              mode="bilinear", align_corners=False)
        p = torch.softmax(model(xi), dim=1)
        if p.shape[-2:] != hw:
            p = F.interpolate(p, size=hw, mode="bilinear", align_corners=False)
        acc = p if acc is None else acc + p
    return acc / len(TTA_SCALES)


def _dice_counts(pred: torch.Tensor, y: torch.Tensor, n_classes: int):
    """Per-image (intersection, |pred|, |gt|) for each class, as (B, C) each."""
    b = pred.shape[0]
    inter = torch.zeros(b, n_classes)
    np_ = torch.zeros(b, n_classes)
    ng = torch.zeros(b, n_classes)
    for c in range(n_classes):
        p, t = pred == c, y == c
        inter[:, c] = (p & t).flatten(1).sum(1).float().cpu()
        np_[:, c] = p.flatten(1).sum(1).float().cpu()
        ng[:, c] = t.flatten(1).sum(1).float().cpu()
    return inter, np_, ng


@torch.no_grad()
def evaluate(model, loader, device, tta: bool = False) -> dict:
    """Both Dice estimators over a loader, plus the per-image spread."""
    model.eval()
    batch_dice, per_img = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = predict_prob(model, x, tta).argmax(1)
        inter, np_, ng = _dice_counts(pred, y, N_CLASSES)
        # batch-pooled Dice, averaged over batches (the original estimator)
        bi, bn, bg = inter.sum(0), np_.sum(0), ng.sum(0)
        batch_dice.append(((2 * bi + 1e-6) / (bn + bg + 1e-6)).numpy())
        # per-image Dice, kept only where the class is annotated
        d = (2 * inter / (np_ + ng).clamp_min(1e-6)).numpy()
        d[(ng.numpy() == 0)] = np.nan
        per_img.append(d)

    bm = np.mean(np.stack(batch_dice), axis=0)
    pi = np.concatenate(per_img, axis=0)
    with np.errstate(invalid="ignore"):
        pim = np.nanmean(pi, axis=0)
        pis = np.nanstd(pi, axis=0, ddof=1)
    return {"dice_batchmean": bm.tolist(),
            "dice_image": np.nan_to_num(pim).tolist(),
            "dice_image_std": np.nan_to_num(pis).tolist(),
            "n_images": int(pi.shape[0]),
            "mean_fg_batchmean": float(bm[1:].mean()),
            "mean_fg_image": float(np.nanmean(pim[1:]))}


@torch.no_grad()
def plane_scores(model, df, frames_dir, img_size, device, batch=64,
                 tta: bool = False) -> dict[str, np.ndarray]:
    """Score every frame by how much anatomy the segmenter finds.

    A standard plane is defined by both structures being resolvable, so the
    headline rule is the smaller of the two foreground areas: a frame showing a
    fetal head and no symphysis is not a standard plane, and taking the minimum
    says so. Areas are soft (mean probability), not thresholded masks, so the
    score degrades smoothly instead of falling off a cliff.
    """
    model.eval()
    root = Path(frames_dir)
    out = {k: np.zeros(len(df), dtype=np.float32) for k in SCORE_RULES}

    for s in range(0, len(df), batch):
        chunk = df.iloc[s:s + batch]
        xs = []
        for _, r in chunk.iterrows():
            img = Image.open(root / r.frame_path).convert("L") \
                       .resize((img_size, img_size), Image.BILINEAR)
            a = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
            xs.append((a - 0.449) / 0.226)
        prob = predict_prob(model, torch.stack(xs).to(device), tta)
        for k, v in area_scores(prob).items():
            out[k][s:s + len(chunk)] = v.cpu().numpy()
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
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet34"])
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random encoder init, to size what ImageNet is worth")
    ap.add_argument("--loss", default="ce_dice", choices=["ce_dice", "ce", "dice"])
    ap.add_argument("--no-aug", action="store_true",
                    help="train without geometric/intensity augmentation")
    ap.add_argument("--tta", action="store_true",
                    help="multi-scale test-time augmentation at evaluation")
    ap.add_argument("--out-dir", default="results/unet_seg__seed0")
    ap.add_argument("--skip-classification", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_ = get_logger("unet", str(out / "log.txt"))
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {"arch": a.arch, "img_size": a.img_size, "loss": a.loss,
           "pretrained": not a.no_pretrained, "augment": not a.no_aug,
           "tta": a.tta, "seed": a.seed, "lr": a.lr, "epochs": a.epochs,
           "batch_size": a.batch_size}
    log_.info("config: %s", cfg)

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
    tr_ld = DataLoader(SegDataset(tr, a.frames_dir, a.img_size,
                                  not a.no_aug, a.seed),
                       batch_size=a.batch_size, shuffle=True, num_workers=nw,
                       pin_memory=True, persistent_workers=nw > 0)
    va_ld = DataLoader(SegDataset(va, a.frames_dir, a.img_size, False),
                       batch_size=a.batch_size, num_workers=0)
    te_ld = DataLoader(SegDataset(te, a.frames_dir, a.img_size, False),
                       batch_size=a.batch_size, num_workers=0)

    model = ResUNet(N_CLASSES, pretrained=not a.no_pretrained, in_ch=1,
                    arch=a.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # the symphysis is ~1.5% of a frame and the head ~14%, so weight the loss
    # against the background dominating
    cw = torch.tensor([0.2, 2.0, 1.0], device=device)

    def loss_fn(logits, y):
        if a.loss == "ce":
            return F.cross_entropy(logits, y, weight=cw)
        if a.loss == "dice":
            return dice_loss(logits, y, N_CLASSES)
        return combined_loss(logits, y, cw, N_CLASSES)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for x, y in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
            run += float(loss) * x.size(0)
        sched.step()

        # selection uses the plain forward pass even when --tta is set, so the
        # checkpoint chosen is not a function of the TTA recipe
        d = evaluate(model, va_ld, device)
        mean_fg = d["mean_fg_batchmean"]
        history.append({"epoch": ep, "loss": run / len(tr_ld.dataset),
                        "val_dice_ps": d["dice_batchmean"][1],
                        "val_dice_fh": d["dice_batchmean"][2],
                        "val_dice_fg": mean_fg,
                        "val_dice_fg_image": d["mean_fg_image"]})
        log_.info("ep %02d loss %.4f | val Dice  PS %.4f  FH %.4f  mean %.4f",
                  ep, run / len(tr_ld.dataset), d["dice_batchmean"][1],
                  d["dice_batchmean"][2], mean_fg)

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
                "n_classes": N_CLASSES, "seed": a.seed, "cfg": cfg},
               out / "best.pt")

    te_d = evaluate(model, te_ld, device, tta=a.tta)
    va_d = evaluate(model, va_ld, device, tta=a.tta)
    log_.info("TEST Dice  batchmean: PS %.4f FH %.4f mean %.4f | "
              "per-image: PS %.4f FH %.4f mean %.4f  (n=%d%s)",
              te_d["dice_batchmean"][1], te_d["dice_batchmean"][2],
              te_d["mean_fg_batchmean"], te_d["dice_image"][1],
              te_d["dice_image"][2], te_d["mean_fg_image"], te_d["n_images"],
              ", TTA" if a.tta else "")
    results = {"config": cfg, "val_dice_best_fg": best,
               "test": te_d, "val": va_d,
               "n_masks": {"train": len(tr), "val": len(va), "test": len(te)},
               "n_params": int(sum(p.numel() for p in model.parameters())),
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
        vs = plane_scores(model, vdf, a.frames_dir, a.img_size, device, tta=a.tta)
        ts = plane_scores(model, tdf, a.frames_dir, a.img_size, device, tta=a.tta)

        derived, val_pick = {}, {}
        for rule in SCORE_RULES:
            thr = pick_threshold(vdf.label.values, vs[rule], "macro_f1")
            derived[rule] = {"threshold": float(thr),
                             **frame_metrics(tdf.label.values, ts[rule], thr)}
            val_pick[rule] = frame_metrics(vdf.label.values, vs[rule],
                                           thr)["macro_f1"]
            log_.info("  rule %-8s thr %.4f  VAL macroF1 %.4f  "
                      "TEST macroF1 %.4f  balAcc %.4f  auprc %.4f",
                      rule, thr, val_pick[rule], derived[rule]["macro_f1"],
                      derived[rule]["balanced_accuracy"],
                      derived[rule]["auprc"])

        # the rule is chosen on validation, never on test
        chosen = max(val_pick, key=val_pick.get)
        log_.info("rule chosen on validation: %r -> TEST macroF1 %.4f",
                  chosen, derived[chosen]["macro_f1"])
        results["derived_classifier"] = {
            "rules": derived, "val_macro_f1": val_pick,
            "chosen_rule": chosen, "chosen": derived[chosen]}
        pd.DataFrame({"video_id": tdf.video_id, "frame_idx": tdf.frame_idx,
                      "label": tdf.label,
                      **{f"score_{k}": v for k, v in ts.items()}}) \
            .to_csv(out / "test_plane_scores.csv", index=False)

    save_json(results, out / "results.json")
    log_.info("wrote %s", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
