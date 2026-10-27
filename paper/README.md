# Paper

`main.tex` is an IEEE conference-format (IEEEtran) write-up of the whole
study, including the results that did not survive replication.

## Building

Needs a TeX distribution with `IEEEtran.cls` (TeX Live: `texlive-publishers`;
MiKTeX installs it on first use).

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

Two passes, so the cross-references and table numbers resolve. The figure is
`../docs/demo.png`, so build from inside `paper/`.

## Where the numbers come from

Every figure in the paper is reproducible from artifacts committed to this
repository, none of it from a training run you would have to repeat:

| Paper section | Source |
|---|---|
| Sampling spreads (Table II) | `results/*/results.json` |
| Segmentation seeds (Table III) | `results/unet/comparison.json` |
| Derived classifier (Table IV) | `results/unet/comparison.json`, `derived_classifier` |
| Attention (Table V) | `report/gradcam/*/attention.json` |
| Cross-campaign (Table VI) | `python -m tools.per_campaign`, from saved predictions |
| Browser agreement | the live demo console, on every inference |
