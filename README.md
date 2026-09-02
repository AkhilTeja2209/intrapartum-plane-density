# Standard-plane classification in intrapartum ultrasound

Frame density and temporal modelling in standard-plane classification, on the
IUGC 2024 dataset
([Zenodo 10.5281/zenodo.17655183](https://zenodo.org/records/17655183),
774 videos / 68,106 frames / 1.1 GB, CC-BY-4.0).

> **[▶ Live demo](https://akhilteja2209.github.io/intrapartum-plane-density/)** —
> drop in an ultrasound frame and the trained ResNet-18 scores it in your
> browser. Nothing is uploaded. **Research demo, not a medical device.**

> **Status.** Data preparation is done and verified: 65,531 frames / 774
> videos, label join rate 1.0000, zero unlabelled. The Arm 1 density grid runs
> on the official split. Arm 2 uses spliced transitions, because the training
> split contains none (see below).
>
> Several assumptions in `PROTOCOL.md` are false for this dataset release — the
> official test labels *are* public, the positive rate is ~0.44 rather than
> ~0.17, and the training split has **zero label transitions**. Each changes
> the experiment. See **[`ROADMAP.md`](ROADMAP.md)** for the decisions,
> **[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)** for the evidence, and
> **[`docs/RESULTS.md`](docs/RESULTS.md)** for what the runs show so far.

**Headline.** Frame density buys nothing measurable here: the best single result
in the study comes from **434 training frames**, one per video, against 53,996
for the dense arm. But the sharper finding is that **the effect is not
measurable at this scale at all.** At seed 0 the dense arm lost every
matched-budget comparison; repeated over three seeds, both Protocol B
comparisons **flip sign**, and one condition moves by **0.23 macro-F1** on
identical data. Seed-to-seed variance (±0.12) exceeds any density effect
present, so a single-run comparison on a 434-video corpus cannot tell a real
effect from a reseed — which is what prior work reports.

The temporal arm beats the frame-wise arm including its smoothed baseline
(0.670 vs 0.545), but the ablation shows splicing is not what produced that: the
unspliced model, which cannot have learned transitions, reaches 0.650.

Read **`PROTOCOL.md`** first. It contains the experimental design, one design
problem in the current abstract that needs fixing before you run anything, and
the failure modes most likely to produce a good-looking number that means
nothing.

## Setup

```bash
pip install -r requirements.txt
mkdir -p data && cd data
# download DatasetV3.zip from the Zenodo link, then:
unzip DatasetV3.zip -d DatasetV3 && cd ..
```

## Run order

```bash
python -m src.extract_frames --dataset-root data/DatasetV3 --out-dir data/frames
python -m src.build_index    --dataset-root data/DatasetV3 --frames-dir data/frames --out data/index.csv
python -m src.splits         --index data/index.csv --out data/splits.json --scheme official
python -m src.make_budgets   --config configs/default.yaml --write

python -m src.run_experiment --condition sparse_k1  --model frame    --seed 0
python -m src.run_experiment --condition dense_all  --model frame    --seed 0
python -m src.run_experiment --condition dense_all  --model temporal --seed 0

python -m src.analyze --results-dir results --out-dir report
```

`./run_all.sh` runs the whole grid across three seeds.

**`build_index` now enforces its own invariants** rather than reporting them.
It refuses to write an index if any split has a join rate below 1.0, if an
extracted folder is empty or disagrees with `manifest.csv`, if an `ALL`/`NONE`
sentinel does not partition its video exactly, or if two label files disagree
about the same frame. `--allow-unlabelled` and `--allow-empty-dirs` override
these, but only reach for them once you know why the shortfall is genuine.

It also prints the column mapping it auto-detected and the annotation
granularity per split. Read that output — a silent mis-parse here would poison
every number downstream, and it has already happened once (see the audit).

## Layout

| File | Role |
|---|---|
| `src/extract_frames.py` | mp4 → pre-resized JPEGs (decode once, not once per epoch) |
| `src/build_index.py` | one frame-level index CSV; auto-detects label columns and reports the join |
| `src/splits.py` | **video-level** splits, official by default; reports label-transition structure |
| `src/sampling.py` | the study's independent variable: k-frames-per-video, stride, budget matching |
| `src/datasets.py` | frame and clip datasets; ultrasound-appropriate augmentation |
| `src/models.py` | one shared encoder, two heads |
| `src/engine.py` | one trainer for both arms |
| `src/metrics.py` | imbalance-aware + video-level clinical metrics, video bootstrap |
| `src/smoothing.py` | the baseline the LSTM must beat |
| `src/run_experiment.py` | one condition × one seed, end to end |
| `src/analyze.py` | summary tables, density curve, paired bootstrap tests |
| `src/gradcam.py` | quantitative check that the model looks at anatomy |
| `tools/run_grid.py` | runs a condition grid, cheapest-first, resumable |
| `tools/export_onnx.py` | exports a checkpoint for the browser demo; refuses to ship a divergent graph |
| `tools/make_site_summary.py` | collects results into the table the demo renders |
| `site/` | the static GitHub Pages demo (client-side inference) |

## Compute

Single T4 (Colab free tier), 224px, ResNet-18: `dense_all` is ~6 min/epoch,
so ~2 h per seed. The temporal arm costs 2–3× that. The full grid at 3 seeds
is roughly 30–40 GPU-hours.

To fit a smaller budget, cut in this order: drop to `--img-size 160` (~2×
faster, costs about a point of macro-F1), then drop `dense_stride2` and
`dense_stride4` (the curve survives with four points), then reduce the stride
conditions to two seeds. Do **not** cut multiple seeds on the headline
comparisons or the smoothed baseline — those are what make the result
believable.

## Anatomical features that define a standard plane

You do not hand-engineer these — but you need them to sanity-check labels and
to validate Grad-CAM output.

Mid-sagittal transperineal view:

- **Pubic symphysis** — hypoechoic oval in the near field, long axis fully
  visible. Partial or foreshortened PS makes the frame unmeasurable.
- **Fetal head** — bright echogenic skull contour with posterior acoustic
  shadowing.
- **Both simultaneously present**, with the PS long axis clearly resolvable:
  AoP is measured from that axis to the tangent line touching the head, so a
  frame missing either structure cannot be a standard plane by definition.
- **Urethra / bladder neck** — near-field orientation landmark.

Non-standard signatures: oblique or off-midline angulation, head-only or
PS-only frames, probe-transition blur, shadowing that obscures the skull
contour, gain saturation.
