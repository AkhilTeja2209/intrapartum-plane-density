"""Score every U-Net arm on the same footing, then say which one is best.

Each arm's own ``results.json`` reports Dice at that arm's input resolution,
against a mask that was resized down to match. Those numbers are not
comparable across arms: a 256px arm is scored against a nearest-neighbour
downsampled 512px mask, and a 512px arm is not. Published Dice on this corpus
is computed at the mask's native resolution, which is what makes 0.893-0.930
the number to beat.

So this script re-scores everything one way:

  * the input is the frame re-decoded at the videos' native 512x512
    (``tools/extract_mask_frames.py``), resized to whatever that arm was
    trained at;
  * the output probability map is bilinearly resampled back up to 512x512;
  * Dice is computed against the untouched 512x512 ground-truth mask.

Every arm then differs only by the thing that was changed in it, and every
number is comparable to the literature.

The winner is picked on the 40 validation masks and only then looked up on the
300 test masks. Ensembles average probability maps at 512 before the argmax,
which is why they are computed with their members held in memory together
rather than from cached predictions.

    python tools/unet_compare.py
    python tools/unet_compare.py --no-tta --skip-ensembles
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import frame_metrics, pick_threshold  # noqa: E402
from src.segmentation import N_CLASSES, decode_mask  # noqa: E402
from src.splits import load_splits  # noqa: E402
from src.train_unet import SCORE_RULES, TTA_SCALES, area_scores  # noqa: E402
from src.unet import ResUNet  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

NATIVE = 512
# Common grid for averaging an ensemble's probability maps when its members
# were trained at different input sizes.
SCORE_GRID = 256
GREY_MEAN, GREY_STD = 0.449, 0.226

# Ensembles to try. Averaging over seeds is the cheapest real gain in
# segmentation and what most IUGC entries did; averaging over input sizes asks
# whether the arms make different mistakes.
ENSEMBLES = {
    "ens_seeds": ["base_s0", "base_s1", "base_s2"],
    "ens_native_sizes": ["native256_s0", "native384_s0", "native512_s0"],
    "ens_best_mixed": ["base_s0", "native512_s0", "resnet34_s0"],
    # Once native384 has three seeds, the obvious ensemble is the shipped
    # configuration averaged over its own seeds: it isolates the gain from
    # averaging away seed noise, with the input size held fixed, which
    # ens_native_sizes confounds with resolution.
    "ens_native384_seeds": ["native384_s0", "native384_s1", "native384_s2"],
    "ens_native256_seeds": ["native256_s0", "native256_s1", "native256_s2"],
}


def load_arm(run_dir: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    size = int(cfg.get("img_size", ckpt.get("img_size", 256)))
    arch = cfg.get("arch", "resnet18")
    m = ResUNet(N_CLASSES, pretrained=False, in_ch=1, arch=arch)
    m.load_state_dict(ckpt["model"])
    return m.eval().to(device), {**cfg, "img_size": size, "arch": arch}


def native_frame(frames_native: Path, vid: str, idx: int) -> np.ndarray:
    return np.asarray(Image.open(frames_native / vid / f"{idx:06d}.png")
                      .convert("L"), np.float32) / 255.0


def to_input(grey: np.ndarray, size: int, device) -> torch.Tensor:
    """Native frame -> that arm's input tensor, Pillow bilinear like training."""
    if grey.shape[0] != size:
        img = Image.fromarray((grey * 255).astype(np.uint8)) \
                   .resize((size, size), Image.BILINEAR)
        grey = np.asarray(img, np.float32) / 255.0
    t = torch.from_numpy((grey - GREY_MEAN) / GREY_STD)[None, None]
    return t.to(device)


@torch.no_grad()
def prob_at_native(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """(1,3,512,512) probabilities, whatever resolution the model works at."""
    scales = TTA_SCALES if tta else (1.0,)
    acc = None
    for s in scales:
        xi = x if s == 1.0 else F.interpolate(x, scale_factor=s,
                                              mode="bilinear", align_corners=False)
        p = torch.softmax(model(xi), dim=1)
        if p.shape[-1] != NATIVE:
            p = F.interpolate(p, size=(NATIVE, NATIVE), mode="bilinear",
                              align_corners=False)
        acc = p if acc is None else acc + p
    return acc / len(scales)


def dice_table(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """Per-image Dice (classes present only) plus the pooled estimator."""
    per, inter, card = [], np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    for p, g in zip(preds, gts):
        row = np.full(N_CLASSES, np.nan)
        for c in range(N_CLASSES):
            pc, tc = p == c, g == c
            i, n = (pc & tc).sum(), pc.sum() + tc.sum()
            inter[c] += i
            card[c] += n
            if tc.sum():
                row[c] = 2 * i / max(1, n)
        per.append(row)
    per = np.stack(per)
    with np.errstate(invalid="ignore"):
        pim = np.nanmean(per, axis=0)
    pooled = 2 * inter / np.maximum(card, 1)
    return {"dice_image": np.nan_to_num(pim).tolist(),
            "dice_pooled": pooled.tolist(),
            "mean_fg_image": float(np.nanmean(pim[1:])),
            "mean_fg_pooled": float(pooled[1:].mean()),
            "n_images": int(len(per))}


def mask_rows(masks: pd.DataFrame, split: str) -> pd.DataFrame:
    return masks[masks.orig_split == split].reset_index(drop=True)


def ground_truth(rows: pd.DataFrame) -> list[np.ndarray]:
    out = []
    for _, r in rows.iterrows():
        g = decode_mask(np.array(Image.open(r.mask_path)), r.orig_split)
        if g.shape[0] != NATIVE:                 # every shipped mask is 512
            g = np.asarray(Image.fromarray(g).resize((NATIVE, NATIVE),
                                                     Image.NEAREST))
        out.append(g)
    return out


def derived_classifier(members, index, sp, frames_dir, device, log,
                       label: str) -> dict:
    """Area-derived plane classification for one model or an ensemble.

    Thresholds and the aggregation rule are picked on validation and then
    looked up on test, for every rule, so the *selection* can be audited rather
    than just its outcome. That matters here: the 40-video validation split
    turns out to be too thin to pick a threshold reliably, and reporting only
    the selected rule would hide it.
    """
    got = {}
    for split in ("val", "test"):
        df = index[index.video_id.isin(sp[split])].reset_index(drop=True)
        scores = {k: np.zeros(len(df), np.float32) for k in SCORE_RULES}
        # A U-Net holds full-resolution decoder activations, so memory grows
        # with the square of the input size. Fix the pixel budget instead of
        # the batch count: 32 frames at 256px, 8 at 512px.
        big = max(cfg["img_size"] for _, cfg in members)
        B = max(4, int(32 * (256 / big) ** 2))
        for s0 in range(0, len(df), B):
            chunk = df.iloc[s0:s0 + B]
            acc = None
            for mm, cfg in members:
                xs = []
                for _, r in chunk.iterrows():
                    img = Image.open(frames_dir / r.frame_path).convert("L")                                .resize((cfg["img_size"], cfg["img_size"]),
                                       Image.BILINEAR)
                    g = np.asarray(img, np.float32) / 255.0
                    xs.append(torch.from_numpy((g - GREY_MEAN) / GREY_STD)[None])
                with torch.no_grad():
                    p = torch.softmax(mm(torch.stack(xs).to(device)), 1)
                # members may work at different input sizes, so bring every
                # probability map to a common grid before averaging. The score
                # is a spatial mean, so the grid only has to be consistent.
                if p.shape[-1] != SCORE_GRID:
                    p = F.interpolate(p, size=(SCORE_GRID, SCORE_GRID),
                                      mode="bilinear", align_corners=False)
                acc = p if acc is None else acc + p
            for k, v in area_scores(acc / len(members)).items():
                scores[k][s0:s0 + len(chunk)] = v.cpu().numpy()
        got[split] = (df.label.values, scores)

    rules = {}
    for rule in SCORE_RULES:
        thr = pick_threshold(got["val"][0], got["val"][1][rule], "macro_f1")
        v = frame_metrics(got["val"][0], got["val"][1][rule], thr)
        t = frame_metrics(got["test"][0], got["test"][1][rule], thr)
        rules[rule] = {"threshold": float(thr),
                       "val_macro_f1": v["macro_f1"], "val_auprc": v["auprc"],
                       **t}
    by_f1 = max(rules, key=lambda r: rules[r]["val_macro_f1"])
    by_ap = max(rules, key=lambda r: rules[r]["val_auprc"])
    log.info("  %s: rule on val macro-F1 -> %r (TEST %.4f); on val AUPRC -> "
             "%r (TEST %.4f)", label, by_f1, rules[by_f1]["macro_f1"],
             by_ap, rules[by_ap]["macro_f1"])
    for r in SCORE_RULES:
        log.info("    %-8s thr %.4f | VAL F1 %.4f AP %.4f | TEST F1 %.4f "
                 "AP %.4f", r, rules[r]["threshold"], rules[r]["val_macro_f1"],
                 rules[r]["val_auprc"], rules[r]["macro_f1"], rules[r]["auprc"])
    return {"rules": rules, "chosen_by_val_macro_f1": by_f1,
            "chosen_by_val_auprc": by_ap,
            "test_macro_f1_of_chosen": rules[by_f1]["macro_f1"],
            "test_macro_f1_spread": [min(rules[r]["macro_f1"] for r in rules),
                                     max(rules[r]["macro_f1"] for r in rules)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/unet")
    ap.add_argument("--masks-native", default="data/masks_native.csv")
    ap.add_argument("--frames-native", default="data/frames_native")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--out", default="results/unet/comparison.json")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--skip-ensembles", action="store_true")
    ap.add_argument("--classify", nargs="?", const="key", default=None,
                    choices=["key", "all"],
                    help="also run the area-derived classifier over all 8,665 "
                         "test frames: 'key' for the seeds, the winner and the "
                         "seed ensemble, 'all' for every arm (slow)")
    a = ap.parse_args()

    out_path = ROOT / a.out
    log = get_logger("compare", str(out_path.parent / "comparison.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn = ROOT / a.frames_native

    masks = pd.read_csv(ROOT / a.masks_native)
    rows = {s: mask_rows(masks, s) for s in ("val", "test")}
    gts = {s: ground_truth(rows[s]) for s in rows}
    greys = {s: [native_frame(fn, r.video_id, int(r.frame_idx))
                 for _, r in rows[s].iterrows()] for s in rows}
    log.info("scoring at native %dx%d against %d val / %d test masks",
             NATIVE, NATIVE, len(gts["val"]), len(gts["test"]))

    # best.pt is written before the test and classification passes, so an arm
    # still training has a checkpoint and no results.json. Require both, or a
    # sweep that is still running gets half-scored.
    arms = sorted(d.name for d in (ROOT / a.runs).iterdir()
                  if (d / "best.pt").exists() and (d / "results.json").exists())
    if not arms:
        raise SystemExit(f"no arms with a best.pt under {a.runs}")
    log.info("arms: %s", ", ".join(arms))

    table = {}
    for name in arms:
        t0 = time.time()
        model, cfg = load_arm(ROOT / a.runs / name, device)
        entry = {"arch": cfg["arch"], "img_size": cfg["img_size"],
                 "loss": cfg.get("loss", "ce_dice"),
                 "pretrained": cfg.get("pretrained", True),
                 "augment": cfg.get("augment", True),
                 "seed": cfg.get("seed", 0),
                 "native_input": "native" in name,
                 "n_params": int(sum(p.numel() for p in model.parameters()))}

        for split in ("val", "test"):
            for tta in (False,) if a.no_tta else (False, True):
                preds = [prob_at_native(model, to_input(g, cfg["img_size"], device),
                                        tta).argmax(1)[0].cpu().numpy()
                         for g in greys[split]]
                entry[f"{split}{'_tta' if tta else ''}"] = dice_table(preds, gts[split])

        # the arm's own self-reported number, for the record
        rj = json.loads((ROOT / a.runs / name / "results.json").read_text())
        entry["self_reported_test_mean_fg"] = rj.get("test", {}).get("mean_fg_image")
        entry["epochs_run"] = len(rj.get("history", []))
        dc = rj.get("derived_classifier") or {}
        if dc:
            entry["derived"] = {
                "chosen_rule": dc.get("chosen_rule"),
                "val_macro_f1": dc.get("val_macro_f1"),
                "test_macro_f1": {k: v["macro_f1"]
                                  for k, v in dc.get("rules", {}).items()}}
        table[name] = entry
        log.info("%-14s val %.4f  test %.4f%s  (%.0fs)", name,
                 entry["val"]["mean_fg_image"], entry["test"]["mean_fg_image"],
                 "" if a.no_tta else
                 f"  test+TTA {entry['test_tta']['mean_fg_image']:.4f}",
                 time.time() - t0)
        del model
        torch.cuda.empty_cache()

    # ---- ensembles ------------------------------------------------------
    if not a.skip_ensembles:
        for ens, members in ENSEMBLES.items():
            have = [m for m in members if m in table]
            if len(have) < 2:
                log.info("%s: only %d member(s) present, skipping", ens, len(have))
                continue
            t0 = time.time()
            loaded = [load_arm(ROOT / a.runs / m, device) for m in have]
            entry = {"members": have, "img_size": None, "arch": "ensemble",
                     "n_params": sum(int(sum(p.numel() for p in mm.parameters()))
                                     for mm, _ in loaded)}
            for split in ("val", "test"):
                preds = []
                for g in greys[split]:
                    acc = None
                    for mm, cfg in loaded:
                        p = prob_at_native(mm, to_input(g, cfg["img_size"], device),
                                           False)
                        acc = p if acc is None else acc + p
                    preds.append((acc / len(loaded)).argmax(1)[0].cpu().numpy())
                entry[split] = dice_table(preds, gts[split])
            table[ens] = entry
            log.info("%-14s val %.4f  test %.4f  (%d members, %.0fs)", ens,
                     entry["val"]["mean_fg_image"], entry["test"]["mean_fg_image"],
                     len(have), time.time() - t0)
            del loaded
            torch.cuda.empty_cache()

    # ---- the winner, chosen on validation -------------------------------
    def val_of(k):
        return table[k]["val"]["mean_fg_image"]

    winner = max(table, key=val_of)
    log.info("winner on validation: %s (val %.4f) -> TEST %.4f "
             "(PS %.4f, FH %.4f)", winner, val_of(winner),
             table[winner]["test"]["mean_fg_image"],
             table[winner]["test"]["dice_image"][1],
             table[winner]["test"]["dice_image"][2])

    # does TTA help the winner? decided on validation too
    best_single = max((k for k in table if "val_tta" in table[k]), key=val_of,
                      default=None)
    tta_verdict = None
    if best_single and not a.no_tta:
        e = table[best_single]
        gain_val = e["val_tta"]["mean_fg_image"] - e["val"]["mean_fg_image"]
        tta_verdict = {"arm": best_single, "val_gain": gain_val,
                       "test_gain": e["test_tta"]["mean_fg_image"]
                                    - e["test"]["mean_fg_image"],
                       "use_tta": gain_val > 0}
        log.info("TTA on %s: val %+.4f, test %+.4f -> %s", best_single,
                 tta_verdict["val_gain"], tta_verdict["test_gain"],
                 "use it" if tta_verdict["use_tta"] else "do not use it")

    # ---- is the validation split even able to pick a winner? ------------
    # 40 masks is a thin basis for model selection. If val Dice does not
    # predict test Dice across the arms, then "best on validation" is close to
    # a coin toss, and that has to be said next to the winner rather than
    # discovered later by someone trying to reproduce it.
    singles = [k for k in table if "members" not in k and "members" not in table[k]]
    sel = None
    if len(singles) > 3:
        v = np.array([table[k]["val"]["mean_fg_image"] for k in singles])
        t = np.array([table[k]["test"]["mean_fg_image"] for k in singles])
        rv = np.argsort(np.argsort(v))
        rt = np.argsort(np.argsort(t))
        sel = {"n_arms": len(singles),
               "pearson_r": float(np.corrcoef(v, t)[0, 1]),
               "spearman_r": float(np.corrcoef(rv, rt)[0, 1]),
               "best_on_val": singles[int(np.argmax(v))],
               "best_on_test": singles[int(np.argmax(t))],
               "test_cost_of_selecting_on_val":
                   float(t.max() - t[int(np.argmax(v))]),
               "n_val_masks": table[singles[0]]["val"]["n_images"]}
        log.info("validation as a selector over %d arms: Pearson r = %+.3f, "
                 "Spearman r = %+.3f; best on val is %s, best on test is %s "
                 "(picking on val costs %.4f Dice)", sel["n_arms"],
                 sel["pearson_r"], sel["spearman_r"], sel["best_on_val"],
                 sel["best_on_test"], sel["test_cost_of_selecting_on_val"])

    # ---- seed spread, the bar everything else has to clear --------------
    seeds = [k for k in table if k.startswith("base_s")]
    spread = None
    if len(seeds) > 1:
        v = np.array([table[k]["test"]["mean_fg_image"] for k in seeds])
        spread = {"arms": seeds, "test_mean": float(v.mean()),
                  "test_std": float(v.std(ddof=1)),
                  "test_range": [float(v.min()), float(v.max())]}
        log.info("seed spread at the base config: test %.4f +/- %.4f "
                 "(range %.4f-%.4f)", v.mean(), v.std(ddof=1), v.min(), v.max())

    # ---- the area-derived classifier, uniformly ------------------------
    derived = None
    if a.classify:
        index = pd.read_csv(ROOT / a.index)
        index = index[index.label >= 0]
        sp = load_splits(ROOT / a.splits)
        fd = ROOT / a.frames_dir
        targets = [k for k in table if a.classify == "all"
                   or k in ("base_s0", "base_s1", "base_s2", winner, "ens_seeds")]
        derived = {}
        for k in targets:
            t0 = time.time()
            if "members" in table[k]:
                members = [load_arm(ROOT / a.runs / m, device)
                           for m in table[k]["members"]]
            else:
                members = [load_arm(ROOT / a.runs / k, device)]
            derived[k] = derived_classifier(members, index, sp, fd, device,
                                            log, k)
            log.info("  %s derived classifier in %.0fs", k, time.time() - t0)
            del members
            torch.cuda.empty_cache()

    payload = {"native_resolution": NATIVE, "arms": table, "winner": winner,
               "winner_val": val_of(winner),
               "winner_test": table[winner]["test"],
               "tta": tta_verdict, "seed_spread": spread,
               "validation_as_selector": sel,
               "derived_classifier": derived}
    save_json(payload, out_path)
    log.info("wrote %s", out_path)

    # ---- a table for the write-up ---------------------------------------
    lines = ["| arm | input | params | val Dice | test PS | test FH | test mean |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for k in sorted(table, key=val_of, reverse=True):
        e = table[k]
        src = "native" if e.get("native_input") else "cached"
        inp = (f"{e['img_size']}px {src}" if e.get("img_size")
               else f"{len(e.get('members', []))}x ensemble")
        lines.append(
            f"| `{k}` | {inp} | {e['n_params'] / 1e6:.1f}M | "
            f"{e['val']['mean_fg_image']:.4f} | "
            f"{e['test']['dice_image'][1]:.4f} | "
            f"{e['test']['dice_image'][2]:.4f} | "
            f"**{e['test']['mean_fg_image']:.4f}** |")
    md = out_path.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log.info("wrote %s", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
