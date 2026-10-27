# Intrapartum Standard-Plane Classification

[![CI](https://github.com/AkhilTeja2209/intrapartum-plane-density/actions/workflows/ci.yml/badge.svg)](https://github.com/AkhilTeja2209/intrapartum-plane-density/actions/workflows/ci.yml)
[![Pages](https://github.com/AkhilTeja2209/intrapartum-plane-density/actions/workflows/pages.yml/badge.svg)](https://github.com/AkhilTeja2209/intrapartum-plane-density/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning pipeline that classifies intrapartum transperineal ultrasound
frames as **standard** or **non-standard** — that is, whether the pubic
symphysis and fetal head are resolved well enough to measure the angle of
progression.

It also runs a controlled study of **frame sampling density**: both the sparse
("image dataset") and dense ("video dataset") arms are built from one corpus by
varying only how many frames are drawn per video, with budget-matched and
prior-matched arms to separate density from sample count.

**Live demo:** [akhilteja2209.github.io/intrapartum-plane-density](https://akhilteja2209.github.io/intrapartum-plane-density/)
— drop in a frame. The classifier scores it and shows the pixels it used; the
U-Net then outlines the pubic symphysis and fetal head. Both run in your
browser. Research demo, not a medical device.

---

## Installation

```bash
git clone https://github.com/AkhilTeja2209/intrapartum-plane-density.git
cd intrapartum-plane-density
pip install -r requirements.txt
```

### Dataset

The IUGC 2024 corpus (774 videos, CC-BY-4.0) is not redistributed here.
Download `DatasetV3.zip` from
[Zenodo 10.5281/zenodo.17655183](https://zenodo.org/records/17655183) into
`data/`, then unwrap its nested archives:

```bash
python tools/unpack_dataset.py --zip data/DatasetV3.zip --out data/DatasetV3
```

---

## Quick start

Verify the whole pipeline on synthetic data first — no dataset, no GPU, ~2 minutes:

```bash
python -m src.smoke_test --synthetic
```

Then prepare the real data and train:

```bash
# 1. decode videos to pre-resized frames
python -m src.extract_frames --dataset-root data/DatasetV3 --out-dir data/frames

# 2. join frames to labels (refuses to write unless the join rate is 1.0)
python -m src.build_index --dataset-root data/DatasetV3 --frames-dir data/frames --out data/index.csv

# 3. split by video using the dataset's official folders
python -m src.splits --index data/index.csv --out data/splits.json --scheme official

# 4. compute the matched-budget frame counts and write them into the config
python -m src.make_budgets --config configs/default.yaml --write

# 5. train one condition
python -m src.run_experiment --condition dense_all --model frame --seed 0
```

---

## Commands

### Pipeline

| Command | Description |
|---|---|
| `src.extract_frames` | Decode each video once to pre-resized JPEG frames |
| `src.build_index` | Build the frame-level index; enforces label-join invariants |
| `src.splits` | Video-level split (`--scheme official` or `regrouped`) |
| `src.make_budgets` | Compute Protocol B frame budgets from the split |
| `src.run_experiment` | Train and evaluate one condition × one seed |
| `src.analyze` | Summary tables, density curve, paired bootstrap tests |
| `src.gradcam` | Grad-CAM for both arms; anatomical attention ratio |
| `src.segmentation` | Build the PS/FH mask index |
| `src.train_unet` | Train the U-Net segmenter and score it as a classifier |
| `src.unet` | ResNet-encoder U-Net, Dice + cross-entropy losses |
| `src.smoke_test` | End-to-end synthetic run of every stage |

### Tooling

| Command | Description |
|---|---|
| `tools/unpack_dataset.py` | Unwrap the nested Zenodo archives |
| `tools/run_grid.py` | Run a condition grid, cheapest-first, resumable |
| `tools/seed_summary.py` | Aggregate repeated seeds; per-seed paired deltas |
| `tools/export_onnx.py` | Export the classifier for the browser demo, with verification |
| `tools/export_unet_onnx.py` | Export the segmenter for the browser demo, with verification |
| `tools/extract_mask_frames.py` | Re-decode the masked frames at the videos' native 512px |
| `tools/run_unet_sweep.py` | Train every U-Net variant, sequentially and resumably |
| `tools/unet_compare.py` | Re-score every variant at native resolution; ensembles, TTA, winner |
| `tools/make_site_summary.py` | Collect results into the demo's summary JSON |
| `tools/make_architecture_figure.py` | Regenerate `docs/architecture.png` |

### Common options

```bash
# segmentation: build the mask index, then sweep the U-Net variants
python -m src.segmentation --dataset-root data/DatasetV3 --out data/masks.csv
python tools/extract_mask_frames.py
python tools/run_unet_sweep.py
python tools/unet_compare.py --classify

# attention: Grad-CAM on both classes, using the segmenter's predicted masks
python -m src.gradcam --checkpoint results/sparse_k1__frame__seed0/best.pt     --unet results/unet/native384_s1/best.pt --on all

# cross-campaign transfer: score the 2021 and 2024 test campaigns separately
python -m tools.per_campaign

# temporal arm instead of frame-wise
python -m src.run_experiment --condition dense_all --model temporal --seed 0

# a whole grid across seeds
python tools/run_grid.py --plan arm1 --seeds 0
python tools/run_grid.py --plan headline --seeds 1,2

# ablation: disable transition splicing
python -m src.run_experiment --config configs/no_splice.yaml \
    --condition dense_all --model temporal --seed 0 --tag nosplice
```

---

## Architecture

![System architecture](docs/architecture.png)

Videos, label files and segmentation annotations enter at the top. Frames are
decoded once rather than per epoch, and a sentinel-aware parser reads the label
files. The two streams meet at a **single frame-level index** that every
downstream stage reads, so no condition can accidentally derive different
labels from another. The split is fixed once at video level and reused by every
condition. Both arms share one encoder definition and differ only in the head,
so both emit one probability per frame and the same metrics apply to each.

| Module | Role |
|---|---|
| `src/extract_frames.py` | Video → JPEG frames, Unicode-safe write path |
| `src/build_index.py` | Label parsing and the frame index; integrity assertions |
| `src/splits.py` | Video-level splits, leakage assertions, transition diagnostics |
| `src/sampling.py` | The study's independent variable: k, stride, budget, prior |
| `src/datasets.py` | Frame and clip views, augmentation, transition splicing |
| `src/models.py` | One shared ResNet-18 encoder; frame-wise and BiLSTM heads |
| `src/engine.py` | One trainer for both arms |
| `src/metrics.py` | Imbalance-aware and video-level metrics, video bootstrap |
| `src/smoothing.py` | Post-hoc smoothing — the baseline the temporal arm must beat |
| `src/segmentation.py` | PS/FH mask index; the per-split mask encodings |
| `src/unet.py` | U-Net decoder on the shared ResNet encoder |
| `src/train_unet.py` | Segmentation training; the area-derived classifier |
| `src/gradcam.py` | Grad-CAM for both arms; the anatomical attention ratio |
| `src/analyze.py` | Summary tables and significance tests |
| `site/` | Static browser demo running both exported models client-side |

---

## Experimental conditions

Defined in `configs/default.yaml`. Sampling applies to the training split only;
every condition is scored on the identical complete test set.

| Group | Conditions | What it varies |
|---|---|---|
| Sparse | `sparse_k1`, `k2`, `k5`, `k10`, `k20` | k frames per video |
| Dense | `dense_stride8`, `stride4`, `stride2`, `dense_all` | temporal stride |
| Budget-matched | `dense_matched_k5`, `dense_matched_k20` | dense arm capped to the sparse arm's frame count |
| Prior-matched | `dense_prior_matched_k5`, `dense_prior_matched_k20` | frame count **and** class prior held fixed |

---

## Results

ResNet-18 on the official split, scored on the same 8,665-frame test set.

| Condition | Train frames | macro-F1 (3 seeds) |
|---|---:|---|
| `sparse_k1` | 434 | 0.682 ± 0.053 |
| `sparse_k5` | 2,170 | 0.638 ± 0.050 |
| `dense_matched_k5` | 2,170 | 0.576 ± 0.034 |
| `dense_prior_matched_k5` | 2,170 | 0.608 ± 0.119 |
| `dense_all` | 53,996 | 0.610 ± 0.032 |

More frames did not help — the strongest single result came from 434 training
frames, one per video. The budget-matched comparison **reverses sign across
seeds** (−0.106, +0.013, −0.093), and one condition moves 0.231 macro-F1 on
identical data, so seed-to-seed variance exceeds any density effect at this
corpus size.

Temporal arm on `dense_all`: frame-wise 0.579 raw and 0.545 smoothed, against
0.670 for the BiLSTM with spliced transitions. The ablation without splicing
reaches 0.650, so the temporal gain is not attributable to transition modelling.

### Segmentation

A U-Net on the same ResNet-18 encoder segments the pubic symphysis and fetal
head. Twenty-one arms, all scored one way — predictions resampled back to the
masks' native 512×512 and compared against the untouched ground truth, which is
the protocol the published 0.893–0.930 range uses.

| Configuration | Test Dice (PS / FH / mean) |
|---|---|
| Three-seed base config | 0.8665 ± 0.0034 |
| Shipped arm (`native384_s1`) | 0.863 / 0.917 / **0.8897** |
| Shipped config, 3 seeds ensembled | 0.862 / 0.921 / **0.8913** |

The frames are cached at a 256px short side while the masks and source videos
are 512×512, so re-decoding the 2,915 masked frames at native resolution is
worth **+0.0163 Dice** at 384px — the only effect in the sweep that resolves at
three seeds (t = +3.79). It is sharper than "more resolution helps": *cached*
384px, interpolated up from the 256px cache, instead **loses** 0.010 and
triples the seed spread. Asking for a resolution the pixels do not contain is
worse than not asking.

Arms within ~2σ of base were re-run at three seeds, and two single-seed results
did not survive. ResNet-34 read as −0.005 at one seed and is **+0.0008** at
three — the sign flipped, so the encoder makes no difference. Removing
augmentation does not resolve as a mean shift either, but it triples the spread:
on this corpus augmentation buys stability, not Dice. Dropping the Dice term
(−0.012) and training from scratch (−0.009) remain the two effects that matter.

The 40-mask validation split ranks the arms almost at random — validation and
test Dice correlate at Spearman **r = 0.28** over 21 arms. It happens to pick
the test-best arm this time, so the shipped model is not selected on test; but
with that correlation, that is luck rather than a working selector.

### Attention

Grad-CAM gives an **anatomical attention ratio**: CAM mass inside the PS/FH
mask against what an equal-area random region would collect. The segmenter
supplies masks for non-standard frames, which carry no ground truth — validated
at r ≈ 0.93 against real masks on the frames that have both.

| Model | macro-F1 | standard | non-standard |
|---|---:|---:|---:|
| ResNet-18 `dense_all` | 0.579 | **1.96** | 1.47 |
| ResNet-18 `sparse_k1` | 0.687 | **1.88** | 1.51 |
| CNN-BiLSTM, no splice | 0.650 | 1.20 | 1.45 |
| CNN-BiLSTM, spliced | 0.670 | **1.12** | **1.04** |

The frame models read anatomy on both classes; the temporal models largely do
not. The best-scoring temporal model attends to the anatomy least — a second,
independent reason to doubt that its advantage reflects better frame
understanding. Two counts worth reading alongside: the segmenter drew anatomy
on **every** frame including all 120 non-standard ones, and the temporal models
produce no positive evidence anywhere on about a quarter of negatives against
2–5% for the frame models.

### Nothing transfers between the two acquisition campaigns

The test set holds two campaigns three years apart — 93 videos from Dec 2020 –
Jul 2021 and 207 from Apr–Jul 2024 — and training is entirely a third campaign
from 2019. Scored separately (AUROC, which is prevalence-independent; the
campaigns differ at 0.68 against 0.52):

| Model | overall F1 | AUROC 2021 | AUROC 2024 | gap [95% CI] |
|---|---:|---:|---:|---:|
| ResNet-18 `sparse_k1` | 0.687 | 0.699 | 0.861 | **+0.162** [+0.079, +0.252] |
| ResNet-18 `dense_all` | 0.579 | 0.630 | 0.813 | **+0.183** [+0.098, +0.267] |
| CNN-BiLSTM, spliced | 0.670 | **0.579** | 0.844 | **+0.266** [+0.177, +0.357] |
| CNN-BiLSTM, no splice | 0.650 | 0.592 | 0.787 | **+0.195** [+0.103, +0.294] |

Intervals resample examinations, not clips: the 93 "2021" videos are 33
examinations and the 207 "2024" ones about 84 sessions.

No arm transfers, and the best-scoring one transfers worst: the spliced
CNN-BiLSTM's 2021 interval is [0.495, 0.667], which includes 0.5 — chance, on
33 examinations. It holds across **every** arm in the project: the gap is
positive in 25 of 25, mean AUROC 0.632 on 2021 against 0.810 on 2024. The
overall figures look healthier only because 2024 is 207 of the 300 test videos.
Re-tuning the threshold per campaign does not rescue it, so this is degraded
ranking rather than miscalibration. Nor is it temporal drift — training is
2019, so 2021 is the *nearer* campaign and still the one that fails.

A per-centre split would be the direct test and **is not possible**: the
release carries no centre label and every video is uniformly re-encoded. Details
and limits in [docs/SEGMENTATION_AND_ATTENTION.md](docs/SEGMENTATION_AND_ATTENTION.md) §4.4.

Scored as a classifier by how much anatomy it finds, the segmenter reaches
**0.530 ± 0.039 macro-F1**, below the dense frame classifier's 0.610 ± 0.032.
An earlier version of this README reported 0.607 and called that a match; it
was one seed at the top of a wide distribution. A later version retreated to
"the best arms reach 0.603 and 0.622", and that did not survive either: running
the same configuration at another seed gives **0.456**, 0.147 lower, and that
lower seed is the one with the best Dice in the sweep. Better segmentation does
not buy a better derived classifier.

The threshold-free AUPRC is far more stable at 0.757 ± 0.024, so what is
unreliable is converting a segmented area into a decision — not the segmenter's
ordering of frames. The demo labels its area-derived verdict accordingly.

Full write-up, including why the browser's resampling had to be rewritten in
fixed-point arithmetic:
**[`docs/SEGMENTATION_AND_ATTENTION.md`](docs/SEGMENTATION_AND_ATTENTION.md)**.

---

## Requirements

- Python 3.10 or later
- PyTorch 2.0+ with CUDA for training (CPU is fine for the synthetic smoke test)
- ~10 GB disk for the dataset and extracted frames
- A CUDA GPU with 8 GB VRAM reproduces the full grid in roughly 5 GPU-hours

Dependencies are listed in `requirements.txt`.

---

## License

Code is MIT-licensed — see [LICENSE](LICENSE).

The IUGC 2024 dataset is distributed separately by its authors under CC-BY-4.0
and is not redistributed in this repository.
