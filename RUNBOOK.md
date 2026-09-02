# RUNBOOK — how to actually get results

Follow this top to bottom. Every command is copy-paste. Nothing is assumed.

**Total time: about 3–5 days of wall-clock, of which maybe 4 hours is you
sitting at the keyboard.** The rest is the GPU grinding while you do other
things.

---

## Part 0 — What you are actually building, in plain terms

You have 774 ultrasound videos of women in labour. Each video is a few
thousand frames. Some frames are *standard planes* (you can see the pubic
symphysis and the fetal head clearly enough to take a measurement); most are
not.

You are training a program to look at a frame and say "usable" or "not
usable". Then you are asking two questions:

1. **Does it help to train on lots of frames from each video, instead of one
   or two frames from each video?** (Arm 1)
2. **Does it help to let the model see neighbouring frames when deciding?**
   (Arm 2)

That is the whole project. Everything below is machinery to answer those two
questions in a way a reviewer can't poke holes in.

---

## Part 1 — Pick where you will run it

You need a GPU. Your laptop will not do — this is 30–40 GPU-hours of work.
Three realistic options:

| | Kaggle Notebooks | Google Colab (free) | Colab Pro (~₹1,000/mo) |
|---|---|---|---|
| GPU | P100 or T4 x2 | T4, when available | T4 / L4, usually available |
| Session limit | 12 h | ~4–6 h, disconnects randomly | ~12 h, more stable |
| Weekly quota | 30 GPU-h, resets Saturday | none stated, throttled if heavy | ~100 compute units |
| Disk | 73 GB, wiped after session | ~100 GB, wiped after session | same |
| Storage that persists | Kaggle Datasets (20 GB) | your Google Drive | your Google Drive |

**Recommendation: Kaggle.** 30 GPU-hours a week and 12-hour sessions fit this
project almost exactly, and the dataset can live in a Kaggle Dataset so you
never re-upload it. If your college has any GPU server, use that instead and
skip Part 3's upload dance.

Whatever you pick, the golden rule is:

> **The machine wipes itself when the session ends. Anything you want to keep
> must be written somewhere permanent before the session dies.**

The steps below are built around that.

---

## Part 2 — Test the code on your own laptop first (30 minutes)

Do this **before** downloading 1.1 GB or spending a single GPU-minute. It
catches broken installs and typos while they cost you nothing.

```bash
# on your own laptop
cd ~/Desktop
unzip intrapartum-spc.zip      # or however you got the folder
cd intrapartum-spc

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn Pillow PyYAML opencv-python-headless matplotlib
python -m src.smoke_test --synthetic
```

This invents 20 fake videos of random noise, runs the entire pipeline on them,
and prints a `[PASS]`/`[FAIL]` for each stage.

**What you want to see at the end:**

```
ALL CHECKS PASSED. The pipeline is wired correctly.
```

It will say `[SKIP] torch not installed` for the training steps. That is
expected and fine — you are testing the data plumbing, and torch goes on the
GPU machine.

**If something says `[FAIL]`, stop and fix it here.** Debugging on your laptop
takes minutes; debugging in a Kaggle session burns GPU quota.

---

## Part 3 — Get the dataset onto your GPU machine

### 3a. Download it

Go to <https://zenodo.org/records/17655183> and download **DatasetV3.zip**
(1.1 GB). Do this on a decent connection; it is a single file.

### 3b. Verify it downloaded intact

```bash
md5sum DatasetV3.zip        # macOS: md5 DatasetV3.zip
```

Must print `548cd5d52e3459521522a526d0d25b1b`. If it doesn't, the download is
corrupt — delete and re-download. A truncated zip fails in confusing ways
three steps later.

### 3c. Look inside before you unzip

```bash
unzip -l DatasetV3.zip | head -40
```

Write down what you see. You are looking for:

- folders named something like `train/`, `validation/`, `test/`
- a CSV with "class_label" or similar in the name
- video files (`.mp4`) or already-extracted image files
- a `seg/` folder with segmentation masks

**The layout may not match what I assumed.** That's fine, and it's exactly why
you look first. If the videos are already extracted to images, you skip Part
4a entirely. Tell me what the listing says and I'll adjust the scripts.

### 3d. Get it onto the GPU machine

**Kaggle route (recommended):**

1. Kaggle → Datasets → New Dataset → upload `DatasetV3.zip`. Takes a while;
   do it once and it's there forever.
2. Create a new Notebook, Settings → Accelerator → **GPU T4 x2**.
3. Add Data → your dataset. It appears at `/kaggle/input/<your-dataset-name>/`.
4. Upload the code folder as a second Kaggle Dataset (or `git clone` it if you
   push to GitHub — easier for iterating).

**Colab route:**

1. Upload `DatasetV3.zip` to your Google Drive.
2. New notebook → Runtime → Change runtime type → **T4 GPU**.
3. First cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## Part 4 — One-time data preparation

Run these **once**. The output gets saved permanently in Part 5 so you never
repeat this.

### Setup cell (run at the start of every session)

```bash
cd /kaggle/working            # Colab: cd /content
pip install -q pyyaml opencv-python-headless

# code
cp -r /kaggle/input/<your-code-dataset>/intrapartum-spc .
cd intrapartum-spc

# data
mkdir -p data
unzip -q /kaggle/input/<your-data-dataset>/DatasetV3.zip -d data/
ls data/DatasetV3
```

### 4a. Turn videos into image files

```bash
python -m src.extract_frames \
    --dataset-root data/DatasetV3 \
    --out-dir data/frames \
    --short-side 256
```

**What this does:** opens each `.mp4`, saves every frame as a JPEG shrunk to
256 pixels on the short side. This exists purely for speed — decoding video is
slow, and without this you'd re-decode the same frames hundreds of times
across all your training runs.

**Time:** 20–40 minutes. **Output size:** roughly 3–5 GB.

**Check:** the last line should report a frame count near **68,106**. If it's
far off, scroll up for `could not open` warnings.

```bash
ls data/frames/*/ | head
du -sh data/frames
```

### 4b. Build the index — **THIS IS THE CRITICAL STEP**

```bash
python -m src.build_index \
    --dataset-root data/DatasetV3 \
    --frames-dir data/frames \
    --out data/index.csv
```

**What this does:** builds one table listing every frame, which video it came
from, its position in that video, and whether it's a standard plane. Every
later step reads only this table.

**Now actually read the output.** It prints the column names it guessed and
the resulting statistics. You are checking four things:

| Line it prints | What's healthy | What's wrong |
|---|---|---|
| `using video=... frame=... label=...` | names that match your CSV's real columns | it guessed a wrong column |
| `standard-plane rate` | roughly **0.05 to 0.40** | near 0.0 or near 1.0 |
| `unlabelled` count | only frames from the official `test/` folder | thousands from `train/` |
| `frames per video: min/median/max` | median in the tens or hundreds | median of 1 |

**If the standard-plane rate looks wrong, stop.** A misread label column
produces a table that looks perfectly fine and silently corrupts every number
in your thesis. Open the CSV yourself and compare:

```python
import pandas as pd
df = pd.read_csv('data/DatasetV3/train/cls/class_label.csv')   # adjust path
print(df.columns.tolist())
print(df.head(20))
print(df.iloc[:, -1].value_counts())
```

Then fix `LABEL_COLS` / `VIDEO_COLS` / `FRAME_COLS` near the top of
`src/build_index.py` to match the real names, and re-run.

### 4c. Split into train / validation / test

```bash
python -m src.splits --index data/index.csv --out data/splits.json --seed 0
```

**What this does:** decides which *videos* go into training, which into
validation, which into testing.

**Why by video and not by frame:** frame 340 and frame 341 of the same video
are 1/30th of a second apart and look nearly identical. If one lands in
training and the other in testing, the model is being graded on pictures it
already memorised, and your accuracy comes out 10–20 points too high. Every
number would be wrong in the same direction, so it wouldn't even look like
noise — it would just look like a great result. This is the single most common
way medical-imaging papers get retracted.

**Check:** three lines, roughly 60/20/20 split, with similar pos-rate across
all three. The script asserts no overlap and crashes if there is any.

### 4d. Compute the matched budgets

```bash
python -m src.make_budgets --config configs/default.yaml --write
```

**What this does:** works out exactly how many frames each sparse condition
produces, then sets the dense conditions to use the *same* number. This is
what makes Protocol B a fair test. It prints a table and edits the config for
you.

**Check:** `dense_matched_k5 budget = <some number> (matches sparse_k5)`.

---

## Part 5 — Save your work so you never redo Part 4

Part 4 took an hour. Your session will die. Save it now.

```bash
# Package the prepared data into ONE file.
# One big file, not 68,000 small ones -- Drive and Kaggle both choke on
# thousands of tiny files, and this turns a 40-minute restore into 2 minutes.
cd /kaggle/working/intrapartum-spc
tar -czf /kaggle/working/prepared.tar.gz data/frames data/index.csv data/splits.json
ls -lh /kaggle/working/prepared.tar.gz
```

**Kaggle:** in the notebook sidebar, the file appears under Output. Download
it, then upload it as a new Kaggle Dataset called `iugc-prepared`. From now
on, every session starts by attaching that dataset and running:

```bash
tar -xzf /kaggle/input/iugc-prepared/prepared.tar.gz -C /kaggle/working/intrapartum-spc/
```

**Colab:** `cp /content/prepared.tar.gz /content/drive/MyDrive/`

**Also make results persistent**, or a dead session loses hours of training:

```bash
# Colab
mkdir -p /content/drive/MyDrive/spc_results
ln -sfn /content/drive/MyDrive/spc_results results

# Kaggle: results/ under /kaggle/working already survives as notebook output --
# just remember to Save Version before closing.
```

---

## Part 6 — Prove it works on real data (15 minutes)

```bash
python -m src.smoke_test --real --n-videos 20 --epochs 2
```

Runs both model types on a handful of real videos, briefly. It re-checks your
label mapping against the real data and confirms the frame files are actually
where the index says they are.

**Want:** `REAL-DATA SMOKE TEST PASSED. Safe to launch the full grid.`

**Do not skip this.** Two minutes here versus discovering at hour six that
every frame path was wrong.

---

## Part 7 — Arm 1: does frame density matter?

This is the bulk of the compute. Run one condition at a time so a dead session
costs you one run, not all of them.

### Start with the two extremes

```bash
# the "image dataset" end: one frame per video
python -m src.run_experiment --condition sparse_k1 --model frame --seed 0

# the "video dataset" end: every frame
python -m src.run_experiment --condition dense_all --model frame --seed 0
```

`sparse_k1` takes a few minutes. `dense_all` takes about 2 hours.

While `dense_all` runs you'll see one line per epoch:

```
ep 01 loss 0.5123 | val macroF1 0.7210 balAcc 0.7455 auprc 0.6801 thr 0.45 | 340s
```

Watch `val macroF1`. It should climb for several epochs then flatten. Training
stops automatically when it stops improving.

**Sanity check before continuing:** `dense_all` should clearly beat
`sparse_k1`. If they're identical, something is broken — most likely the
sampling isn't actually varying, so check the `condition ... -> N frames` line
at the top of each log.

### Then fill in the curve

```bash
for c in sparse_k2 sparse_k5 sparse_k10 sparse_k20 dense_stride8 dense_stride4; do
    python -m src.run_experiment --condition $c --model frame --seed 0
done
```

### Then the matched-budget conditions — your actual contribution

```bash
python -m src.run_experiment --condition dense_matched_k5  --model frame --seed 0
python -m src.run_experiment --condition dense_matched_k20 --model frame --seed 0
```

These are quick (small training sets) and they carry the paper.

### Then repeat everything with seeds 1 and 2

```bash
for s in 1 2; do
  for c in sparse_k1 sparse_k5 sparse_k20 dense_all dense_matched_k5 dense_matched_k20; do
    python -m src.run_experiment --condition $c --model frame --seed $s
  done
done
```

**Why three seeds:** neural network training is random. Run the identical
setup twice and you get slightly different numbers. Without repeats you cannot
tell a real 2-point improvement from luck — and a single-seed difference of
1–2 macro-F1 points on this dataset is almost always luck.

---

## Part 8 — Arm 2: does temporal context help?

```bash
for s in 0 1 2; do
  python -m src.run_experiment --condition dense_all --model temporal --seed $s --tag lstm_bi
done
```

~4–6 hours per seed. If you're tight on quota, do seed 0 only and note it.

Then the ablations, which are what make this a real study rather than one
number:

```bash
# GRU instead of LSTM
python -m src.run_experiment --config configs/gru.yaml --condition dense_all \
    --model temporal --seed 0 --tag gru_bi

# causal (can't see future frames) -- the only version deployable during a
# live scan, so the gap between this and bidirectional is a finding in itself
python -m src.run_experiment --config configs/causal.yaml --condition dense_all \
    --model temporal --seed 0 --tag lstm_uni

# start from the already-trained frame-wise encoder instead of ImageNet
python -m src.run_experiment --condition dense_all --model temporal --seed 0 \
    --tag warm --warm-start results/dense_all__frame__seed0/best.pt
```

**The comparison that decides Arm 2 is not LSTM vs plain ResNet.** Every
frame-wise run already computed a *smoothed* score — its own predictions
passed through a temporal filter, no extra training. Look for this line in the
`dense_all` frame log:

```
smoothed (viterbi) test macroF1 0.9xxx vs raw 0.8xxx
```

That smoothed number is what the LSTM has to beat. Standard planes come in
runs, so most of the "temporal information" is just the fact that neighbouring
frames share a label — and a three-line filter captures that for free. In my
simulation the filter alone took macro-F1 from 0.834 to 0.965. If you compare
the LSTM against the *raw* number you will report a huge temporal gain that is
pure post-processing artefact.

---

## Part 9 — Get your tables and figures

```bash
python -m src.analyze --results-dir results --out-dir report
```

Produces:

- `report/summary.csv` — mean ± std of every metric per condition. **This is
  Table 1 of your paper.**
- `report/density_curve.png` — accuracy vs training-set size. **Figure 1.**
- `report/all_runs.csv` — every individual run.

Then the three head-to-head tests:

```bash
# A: does density help at natural size?
python -m src.analyze --results-dir results --out-dir report \
    --compare sparse_k5__frame__seed0 dense_all__frame__seed0

# B: does it still help when both get the SAME number of frames?
python -m src.analyze --results-dir results --out-dir report \
    --compare sparse_k5__frame__seed0 dense_matched_k5__frame__seed0

# C: does the LSTM beat frame-wise?
python -m src.analyze --results-dir results --out-dir report \
    --compare dense_all__frame__seed0 dense_all__temporal_lstm_bi__seed0
```

Each prints something like:

```
diff +0.0421  95% CI [+0.0180, +0.0663]  p = 0.0012
```

**How to read it:** `diff` is how much better B is than A. If the CI does
**not** cross zero, the difference is real. If it does cross zero, the two are
tied — say so plainly rather than reporting the raw difference as if it meant
something.

---

## Part 10 — Check the model isn't cheating

```bash
python -m src.gradcam \
    --checkpoint results/dense_all__frame__seed0/best.pt \
    --mask-dir data/DatasetV3/train/seg \
    --n 128 --out-dir report/gradcam
```

**Why this matters more than it sounds.** The videos come from three different
hospitals with different ultrasound machines. Each machine draws its own depth
markers, menu bars, and fan-shaped image border. If standard-plane frames
aren't evenly spread across the three hospitals, your model can score
brilliantly by learning *"this is the machine at hospital 2"* while never once
looking at the pubic symphysis. It would fail instantly in any real clinic and
the accuracy number would be meaningless.

This script measures where the model looks: what share of its attention lands
inside the actual pubic-symphysis and fetal-head regions, versus a random
patch of the same size.

- **above ~1.5** → good, it's reading anatomy
- **near 1.0** → it's reading something else. Crop the interface region and
  retrain, and add a leave-one-hospital-out split.

Run this **before** you write any accuracy number into your report.

---

## Part 11 — If you run out of GPU time

Minimum set that still supports a defensible paper — about **8 GPU-hours**:

| Run | Seeds | Why it's essential |
|---|---|---|
| `sparse_k1` frame | 3 | the image-dataset end of the curve |
| `sparse_k5` frame | 3 | the sparse arm of Protocol B |
| `dense_matched_k5` frame | 3 | **the contribution** |
| `dense_all` frame | 3 | the dense end + the smoothed baseline |
| `dense_all` temporal | 1 | Arm 2, noted as single-seed |

Cut in this order, cheapest damage first:

1. `--img-size 160` instead of 224 → about 2× faster, costs ~1 macro-F1 point
2. drop `dense_stride2` and `dense_stride4` → the curve survives on 4 points
3. temporal arm at 1 seed instead of 3, stated explicitly in the limitations

**Never cut:** multiple seeds on the four headline conditions, or the smoothed
baseline. Those are what make the result believable rather than a story.

---

## Part 12 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch too big | `batch_size: 32` in the config; temporal arm may need 16 |
| Session died mid-run | Colab/Kaggle timeout | results are per-run; just rerun that one condition |
| Accuracy suspiciously ~99% | frame-level leakage | you skipped 4c or edited it — splits must be by video |
| pos-rate is 0.0 or 1.0 | label column misread | redo 4b, inspect the CSV by hand |
| `no frames under data/frames` | 4a didn't run or wrote elsewhere | check `--out-dir` matches the config |
| Every condition scores the same | sampling not applied | check the `condition ... -> N frames` line differs per run |
| `prediction files are not frame-aligned` | comparing runs from different splits | you regenerated `splits.json` mid-study — regenerate all runs |
| Training loss stuck flat | LR wrong for your setup | try `lr: 3.0e-4`; check pos-rate isn't degenerate first |
