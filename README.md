# Standard-plane classification in intrapartum ultrasound

BCSE497J Project I — frame density and temporal modelling in standard-plane
classification, on the IUGC 2024 dataset
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
> the experiment. See **[`ROADMAP.md`](ROADMAP.md)** for the decisions and
> **[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)** for the evidence.

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

## Notes on the questions from your professor thread

**"The dataset is first trimester (11–14 weeks)."** The `akhil1.docx` note says
this, but it is wrong, and your professor corrected it in the thread: this is
third-trimester **intrapartum** (during-labour) transperineal imaging. It
matters for more than framing — it determines which anatomy the model should
be looking at (pubic symphysis and fetal head, not nuchal translucency), which
is what the Grad-CAM check validates against.

**"Why is the test folder structured differently?"** Because IUGC withholds
the test labels. `build_index.py` keeps those frames with `label = -1` and
excludes them from every supervised run; your held-out test set is carved from
train+validation at the video level. Say so explicitly in the paper.

**"Would ResNet's lack of memory make it worse than CNN-LSTM?"** Not
necessarily, and this is the most interesting thing in the project. Standard
planes occur in contiguous runs, so most of the usable temporal signal is
label autocorrelation — which a moving average over ResNet's own output
probabilities recovers for free. In simulation, tuned smoothing took macro-F1
from 0.834 to 0.965. So the honest comparison is LSTM versus *smoothed*
ResNet, and if the LSTM only matches it, the finding is that temporal
architecture buys nothing a post-processing filter doesn't. That is a
publishable negative result, and a sharper one than a positive result against
a baseline nobody would deploy.

**"Can I work with images rather than video?"** Yes — and Protocol B is
exactly that framing made rigorous. Every experiment here operates on
extracted frames; the video structure enters only through *which* frames get
sampled and how they are grouped. You never need a video model to answer the
main question.

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
