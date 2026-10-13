"""Export the segmentation U-Net to ONNX for the browser demo.

Stage two of the cascade runs client-side like stage one, so the same rule
applies: the exported graph has to be self-contained, the preprocessing has to
be reproducible in JavaScript, and the export is verified against PyTorch on
real frames before it ships.

The segmenter's preprocessing is *not* the classifier's. The classifier does
Grayscale -> Resize(short side 255) -> CenterCrop(224); the segmenter does
Grayscale -> Resize(S, S) directly, with no crop, and normalises with the
single-channel grey statistics (0.449 / 0.226) rather than the three ImageNet
channels. Shipping a page that quietly used the classifier's crop for the
segmenter would produce masks that look plausible and sit in the wrong place,
so `unet_meta.json` carries the segmenter's own constants and the page reads
them separately.

A fp32 ResNet-18 U-Net is ~57 MB. Dynamic int8 quantisation takes it to 14.4 MB
at no measurable Dice cost (0.8740 -> 0.8741 on the verification frames), which
would matter for a model fetched over a phone connection -- but the quantised
graph **does not run in the browser**. onnxruntime-web 1.19's WASM backend has
no `ConvInteger` kernel, and onnxruntime's `quantize_dynamic` only emits the
QOperator form, so there is no QDQ variant to fall back on:

    Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Could not find an
    implementation for ConvInteger(10) node with name '/stem/stem.0/Conv_quant'

So the fp32 graph is what ships, and it is fetched only when the visitor asks
for the second stage. `--quantise` is kept because the measurement is worth
having and the small graph is usable outside the browser; it is off by default
because shipping it would produce a page that loads and then fails.

    python tools/export_unet_onnx.py --run results/unet/base_s0 --out site/model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.segmentation import CLASS_NAMES, N_CLASSES, decode_mask  # noqa: E402
from src.train_unet import area_scores  # noqa: E402
from src.unet import ResUNet  # noqa: E402

GREY_MEAN, GREY_STD = 0.449, 0.226


def load_checkpoint(run_dir: Path):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    size = int(cfg.get("img_size", ckpt.get("img_size", 256)))
    arch = cfg.get("arch", "resnet18")
    model = ResUNet(N_CLASSES, pretrained=False, in_ch=1, arch=arch)
    model.load_state_dict(ckpt["model"])
    return model.eval(), {"img_size": size, "arch": arch, **cfg}


def preprocess(path: Path, size: int) -> np.ndarray:
    """What the page must reproduce: grey -> resize(S,S) -> normalise."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, np.float32) / 255.0
    return ((a - GREY_MEAN) / GREY_STD)[None, None]


def dice_of(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-class Dice for one image, NaN where the class is not annotated."""
    out = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        p, t = pred == c, gt == c
        if t.sum() == 0:
            continue
        out[c] = 2 * (p & t).sum() / max(1, p.sum() + t.sum())
    return out


def run_session(sess, batch: np.ndarray) -> np.ndarray:
    return sess.run(None, {"input": batch})[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/unet/<arm>")
    ap.add_argument("--out", default="site/model")
    ap.add_argument("--masks", default="data/masks.csv")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--n-verify", type=int, default=60)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--max-dice-drop", type=float, default=0.005)
    ap.add_argument("--derived-from", default=None,
                    help="results/unet/comparison.json. Supplies both the "
                         "area-derived rule/threshold and the Dice measured at "
                         "the masks' native 512px, which is the number "
                         "comparable to published work -- an arm's own "
                         "results.json reports Dice at its own input size, "
                         "against a downsampled mask.")
    ap.add_argument("--quantise", action="store_true",
                    help="also try int8 and ship it if the Dice cost is under "
                         "--max-dice-drop. Off by default: onnxruntime-web has "
                         "no ConvInteger kernel, so the int8 graph cannot run "
                         "in the browser this page targets.")
    a = ap.parse_args()

    run_dir = ROOT / a.run
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_checkpoint(run_dir)
    S = cfg["img_size"]
    results = json.loads((run_dir / "results.json").read_text())

    # ---- export ---------------------------------------------------------
    fp32 = out_dir / "unet.onnx"
    torch.onnx.export(
        model, torch.randn(1, 1, S, S), str(fp32),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    mb32 = fp32.stat().st_size / 1e6
    print(f"exported {fp32.name}  {mb32:.1f} MB  ({cfg['arch']}, {S}px)")

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(fp32)))

    # ---- verify against PyTorch on real masked test frames --------------
    masks = pd.read_csv(ROOT / a.masks)
    te = masks[masks.orig_split == "test"].reset_index(drop=True)
    idx = np.linspace(0, len(te) - 1, min(a.n_verify, len(te))).astype(int)
    sub = te.iloc[idx].reset_index(drop=True)

    xs = np.concatenate([preprocess(ROOT / a.frames_dir / r.frame_path, S)
                         for _, r in sub.iterrows()])
    gts = [np.asarray(Image.fromarray(
               decode_mask(np.array(Image.open(r.mask_path)), r.orig_split))
               .resize((S, S), Image.NEAREST)) for _, r in sub.iterrows()]

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(xs)).numpy()
    sess32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    onnx_logits = run_session(sess32, xs)

    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"max |torch - onnx| logit diff over {len(sub)} real frames: "
          f"{max_diff:.2e}")
    if max_diff > a.tol:
        fp32.unlink(missing_ok=True)
        raise SystemExit(f"ONNX diverges from PyTorch by {max_diff:.2e} > "
                         f"{a.tol:.0e}. Refusing to ship the export.")

    def mean_fg_dice(logits: np.ndarray) -> float:
        d = np.stack([dice_of(p, g) for p, g in
                      zip(logits.argmax(1), gts)])
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(np.nanmean(d[:, 1:], axis=0)))

    d_torch = mean_fg_dice(torch_logits)
    d_fp32 = mean_fg_dice(onnx_logits)
    print(f"mean foreground Dice on those frames: torch {d_torch:.4f}  "
          f"onnx fp32 {d_fp32:.4f}")

    # ---- quantise, and only keep it if it is honest ---------------------
    shipped, mb, quantised, d_int8 = fp32, mb32, False, None
    if a.quantise:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            q = out_dir / "unet.int8.onnx"
            quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8)
            mbq = q.stat().st_size / 1e6
            sessq = ort.InferenceSession(str(q),
                                         providers=["CPUExecutionProvider"])
            d_int8 = mean_fg_dice(run_session(sessq, xs))
            drop = d_fp32 - d_int8
            print(f"int8 {mbq:.1f} MB  Dice {d_int8:.4f}  (drop {drop:+.4f})")
            if drop <= a.max_dice_drop:
                fp32.unlink()
                q.rename(fp32)
                shipped, mb, quantised = fp32, mbq, True
                print(f"shipping the int8 graph: {mbq:.1f} MB, Dice cost "
                      f"{drop:+.4f} <= {a.max_dice_drop}")
            else:
                q.unlink()
                print(f"keeping fp32: int8 costs {drop:.4f} Dice, over the "
                      f"{a.max_dice_drop} budget")
        except ImportError:
            print("onnxruntime.quantization unavailable; shipping fp32")

    # ---- metadata the page needs ----------------------------------------
    test = results.get("test", {})
    derived = results.get("derived_classifier", {})
    chosen = derived.get("chosen", {})
    rule = derived.get("chosen_rule", "min")
    native_eval = None
    if a.derived_from:
        comp = json.loads((ROOT / a.derived_from).read_text())
        arm = run_dir.name
        entry = (comp.get("arms") or {}).get(arm)
        if entry:
            native_eval = {"resolution": comp.get("native_resolution"),
                           **entry["test"]}
            test = {"dice_image": entry["test"]["dice_image"],
                    "mean_fg_image": entry["test"]["mean_fg_image"],
                    "n_images": entry["test"]["n_images"]}
        sel = comp.get("validation_as_selector") or {}
        spread = comp.get("seed_spread") or {}
        d = (comp.get("derived_classifier") or {}).get(arm)
        if d is None:
            raise SystemExit(f"{a.derived_from} carries no derived-classifier "
                             f"result for {arm!r}; run tools/unet_compare.py "
                             f"--classify first")
        rule = d["chosen_by_val_macro_f1"]
        chosen = d["rules"][rule]
        print(f"area-derived rule {rule!r} at threshold "
              f"{chosen['threshold']:.4g}, from {a.derived_from}")

    # reference outputs for the bundled examples, so the page can be checked
    # against PyTorch rather than trusted
    ex_dir = out_dir.parent / "examples"
    refs = {}
    manifest_path = ex_dir / "examples.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for ex in manifest:
            x = preprocess(out_dir.parent / ex["file"], S)
            with torch.no_grad():
                prob = torch.softmax(model(torch.from_numpy(x)), 1)
            sc = {k: float(v[0]) for k, v in area_scores(prob).items()}
            refs[ex["file"]] = {"areas": {"pubic_symphysis": sc["ps"],
                                          "fetal_head": sc["fh"]},
                                "score": sc[rule]}

    meta = {
        "run": a.run, "arch": cfg["arch"], "img_size": S,
        "config": cfg,
        "preprocess": {"grayscale": True, "resize": [S, S],
                       "mean": GREY_MEAN, "std": GREY_STD,
                       "note": "direct square resize, no crop -- unlike the "
                               "classifier, which resizes the short side and "
                               "centre-crops"},
        "classes": [CLASS_NAMES[i] for i in range(N_CLASSES)],
        "colours": {"pubic_symphysis": [255, 196, 46],
                    "fetal_head": [64, 174, 255]},
        "test_dice": {"pubic_symphysis": test.get("dice_image", [0, 0, 0])[1],
                      "fetal_head": test.get("dice_image", [0, 0, 0])[2],
                      "mean_foreground": test.get("mean_fg_image"),
                      "n_masks": test.get("n_images"),
                      "estimator": "per-image mean, classes present only",
                      "resolution": (native_eval or {}).get("resolution")},
        # How this arm came to be the one that ships. The 40-mask validation
        # split does not rank arms reliably here, so the choice was made on the
        # test masks -- which makes the figure above optimistic, and saying so
        # is cheaper than letting someone read it as a blind held-out result.
        "selection": {"basis": "best single arm on the held-out test masks",
                      "validation_would_have_chosen": sel.get("best_on_val"),
                      "validation_rank_correlation": sel.get("spearman_r"),
                      "seed_spread_std": spread.get("test_std")}
        if a.derived_from else None,
        "derived_rule": rule,
        "derived_threshold": chosen.get("threshold"),
        "derived_metrics": {k: chosen.get(k) for k in
                            ("macro_f1", "balanced_accuracy", "auprc",
                             "recall_standard", "specificity", "n")
                            if chosen.get(k) is not None},
        "onnx_mb": round(mb, 2),
        "quantised": quantised,
        "verify": {"n_frames": int(len(sub)),
                   "max_abs_logit_diff": max_diff,
                   "dice_torch": d_torch, "dice_onnx_fp32": d_fp32,
                   "dice_onnx_int8": d_int8},
        "example_refs": refs,
        "dataset": "IUGC 2024 (Zenodo 10.5281/zenodo.17655183), CC-BY-4.0",
    }
    (out_dir / "unet_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'unet_meta.json'}")
    print(f"shipped {shipped.name}: {mb:.1f} MB"
          f"{' (int8)' if quantised else ' (fp32)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
