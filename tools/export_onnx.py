"""Export a trained frame-wise checkpoint to ONNX for in-browser inference.

The web demo runs the model client-side with onnxruntime-web, so the exported
graph has to be self-contained and the preprocessing has to be reproducible in
JavaScript. This script therefore also writes a metadata JSON carrying the
decision threshold and the exact preprocessing constants, so the page cannot
drift from the model.

    python tools/export_onnx.py --run results/dense_all__frame__seed0 \
        --out site/model

Verification is not optional here. A silent preprocessing mismatch between
torchvision and canvas would produce a demo that looks like it works and is
wrong, which is the same class of failure as the label join in the audit. The
script compares ONNX against PyTorch on real test frames and refuses to write
if they disagree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import IMAGENET_MEAN, IMAGENET_STD, FrameDataset  # noqa: E402
from src.metrics import frame_metrics  # noqa: E402
from src.models import FrameClassifier  # noqa: E402


def load_model(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    m = cfg.get("model", {})
    model = FrameClassifier(arch=m.get("arch", "resnet18"), pretrained=False,
                            dropout=m.get("dropout", 0.2))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run_name>")
    ap.add_argument("--out", default="site/model")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--n-verify", type=int, default=256)
    ap.add_argument("--tol", type=float, default=2e-4)
    a = ap.parse_args()

    run_dir = ROOT / a.run
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(run_dir)
    img_size = int(cfg["data"]["img_size"])
    results = json.loads((run_dir / "results.json").read_text())

    # ---- export ---------------------------------------------------------
    onnx_path = out_dir / "model.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB)")

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # ---- verify against PyTorch on REAL frames --------------------------
    # Random tensors would not catch a preprocessing error, so use the actual
    # evaluation transform over actual test frames.
    index = pd.read_csv(ROOT / a.index)
    index = index[index.label >= 0]
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    test_df = index[index.video_id.isin(splits["test"])] \
        .sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    sub = test_df.iloc[np.linspace(0, len(test_df) - 1,
                                   min(a.n_verify, len(test_df))).astype(int)]
    sub = sub.reset_index(drop=True)

    ds = FrameDataset(sub, ROOT / a.frames_dir, img_size, train=False)
    batch = torch.stack([ds[i][0] for i in range(len(ds))])

    with torch.no_grad():
        torch_logits = model(batch).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input": batch.numpy()})[0]

    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"max |torch - onnx| logit diff over {len(sub)} real frames: {max_diff:.2e}")
    if max_diff > a.tol:
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(
            f"ONNX output diverges from PyTorch by {max_diff:.2e} > {a.tol:.0e}. "
            f"Refusing to ship the export.")

    def sm(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, 1]

    thr = float(results["results"]["raw"].get("threshold", 0.5))
    agree = float((( sm(onnx_logits) >= thr) == (sm(torch_logits) >= thr)).mean())
    print(f"decision agreement at threshold {thr:.3f}: {agree * 100:.2f}%")

    # ---- metadata the page needs ----------------------------------------
    raw = results["results"]["raw"]
    meta = {
        "run": results.get("run"),
        "condition": results.get("condition", {}).get("name"),
        "seed": results.get("seed"),
        "arch": cfg.get("model", {}).get("arch", "resnet18"),
        "img_size": img_size,
        # torchvision Resize(int) scales the SHORT side; the page must match.
        "resize_short_side": int(img_size * 1.14),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "grayscale": True,
        "threshold": thr,
        "onnx_mb": round(size_mb, 2),
        "test_metrics": {k: raw[k] for k in
                         ("macro_f1", "balanced_accuracy", "auprc", "auroc",
                          "mcc", "accuracy", "top1_frame_precision",
                          "recall_standard", "precision_standard",
                          "specificity", "n", "prevalence")
                         if k in raw},
        "test_macro_f1_ci95": raw.get("macro_f1_ci95"),
        "train_stats": results.get("train_stats"),
        "dataset": "IUGC 2024 (Zenodo 10.5281/zenodo.17655183), CC-BY-4.0",
        "split": "official train/val/test",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'model_meta.json'}")

    # A few real test frames for the demo's "try one" buttons, re-encoded as
    # PNG. The frames on disk are JPEG, and browser and PIL JPEG decoders
    # disagree by up to one 8-bit level on some pixels -- enough to move the
    # output probability by ~0.04 on a borderline frame. PNG is lossless, so
    # the built-in examples reproduce the PyTorch reference exactly. Uploaded
    # JPEGs keep the decoder difference; the page says so.
    from PIL import Image

    ex_dir = out_dir.parent / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for stale in ex_dir.glob("*.jpg"):
        stale.unlink()
    manifest = []
    for lab, tag in ((1, "standard"), (0, "nonstandard")):
        pick = test_df[test_df.label == lab]
        for j, i in enumerate(np.linspace(0, len(pick) - 1, 2).astype(int)):
            r = pick.iloc[int(i)]
            src = ROOT / a.frames_dir / r.frame_path
            dst = ex_dir / f"{tag}_{j}.png"
            Image.open(src).convert("L").save(dst, format="PNG", optimize=True)

            # Record the reference probability so the page can be checked
            # against PyTorch rather than trusted.
            with torch.no_grad():
                x = ds.tf(Image.open(dst).convert("L")).unsqueeze(0)
                ref = float(torch.softmax(model(x), 1)[0, 1])
            manifest.append({"file": f"examples/{dst.name}", "label": int(lab),
                             "label_name": tag, "reference_prob": round(ref, 6)})
    (ex_dir / "examples.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} example frames to {ex_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
