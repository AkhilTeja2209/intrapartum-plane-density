"""Regenerate the system architecture diagram used in the README.

    python tools/make_architecture_figure.py

Writes docs/architecture.png at 200 dpi on white.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
BOX = "#f2f4f7"
EDGE = "#5a6472"
ACC = "#dce6f1"
ACC2 = "#e8f0e4"

FONT = {"family": "Times New Roman", "color": INK}


def box(ax, x, y, w, h, text, fc=BOX, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.0, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True, **FONT)


def arrow(ax, p1, p2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color=EDGE, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def finish(fig, ax, name):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("wrote", OUT / name)


def architecture():
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    box(ax, .03, .86, .22, .10, "IUGC 2024 corpus\n774 videos (.avi)", ACC, 9, True)
    box(ax, .28, .86, .22, .10, "Label CSVs\npos/neg frame indices", ACC, 9, True)
    box(ax, .53, .86, .22, .10, "Segmentation masks\n+ AoP/HSD landmarks", ACC, 9, True)

    box(ax, .03, .70, .47, .09, "Frame extraction  -  decode once to JPEG, short side 256\n"
                                "(Unicode-safe write path)", BOX, 9)
    box(ax, .53, .70, .22, .09, "Sentinel-aware\nlabel parser (ALL / NONE)", BOX, 9)

    box(ax, .18, .555, .44, .085, "Frame index  -  65,531 rows\nvideo_id | frame_idx | label | split",
        ACC2, 9, True)

    box(ax, .03, .40, .30, .095, "Official split by VIDEO\ntrain 434 / val 40 / test 300", BOX, 9)
    box(ax, .37, .40, .30, .095, "Sampling conditions\nk-per-video | stride | budget | prior", BOX, 9)
    box(ax, .71, .40, .26, .095, "Splicing\nsynthesised transitions", BOX, 9)

    box(ax, .10, .235, .34, .095, "Arm 1  -  ResNet-18 frame encoder\none frame in, one probability out",
        ACC, 9, True)
    box(ax, .52, .235, .34, .095, "Arm 2  -  ResNet-18 + BiLSTM(256)\nT frames in, T probabilities out",
        ACC, 9, True)

    box(ax, .06, .095, .26, .085, "Post-hoc smoothing\nmov-avg | median | Viterbi", BOX, 8.5)
    box(ax, .35, .095, .30, .085, "Evaluation\nmacro-F1, bal-acc, AUPRC, MCC\nvideo bootstrap CI", BOX, 8.5)
    box(ax, .68, .095, .26, .085, "ONNX export\nbrowser classifier", ACC2, 8.5)

    for a, b in [((.14, .86), (.14, .79)), ((.39, .86), (.39, .79)),
                 ((.64, .86), (.64, .79)),
                 ((.26, .70), (.34, .64)), ((.60, .70), (.50, .64)),
                 ((.30, .555), (.18, .495)), ((.42, .555), (.52, .495)),
                 ((.18, .40), (.24, .33)), ((.52, .40), (.44, .33)),
                 ((.84, .40), (.70, .33)),
                 ((.24, .235), (.19, .18)), ((.30, .235), (.45, .18)),
                 ((.66, .235), (.55, .18)), ((.72, .235), (.80, .18))]:
        arrow(ax, a, b)
    finish(fig, ax, "architecture.png")


if __name__ == "__main__":
    architecture()
