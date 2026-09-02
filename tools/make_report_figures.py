"""Generate the figures the Project-I report template asks for.

Fig. 1 Gantt chart          (section 2.5 Project Plan)
Fig. 2 System architecture  (section 4.1)
Fig. 3 Data flow diagram    (section 4.2.1, mandatory)
Fig. 4 Use case diagram     (section 4.2.2, mandatory)

Rendered at 200 dpi on white so they sit cleanly in a Word page.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
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


# ---------------------------------------------------------------- Fig 1 ----
def gantt():
    tasks = [
        ("Literature survey and gap identification", 0, 3),
        ("Dataset acquisition and integrity audit", 2, 3),
        ("Frame extraction and label index", 3, 3),
        ("Split design and sampling conditions", 5, 2),
        ("Arm 1: density grid (13 conditions)", 6, 3),
        ("Arm 2: temporal model and splicing", 8, 2),
        ("Repeat seeds and variance analysis", 9, 2),
        ("Browser deployment of the classifier", 10, 2),
        ("Grad-CAM attention validation", 12, 2),
        ("Paired bootstrap and final report", 13, 3),
    ]
    weeks = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8",
             "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for i, (name, start, dur) in enumerate(tasks):
        y = len(tasks) - i - 1
        done = start + dur <= 12
        ax.barh(y, dur, left=start, height=0.55,
                color=ACC if done else ACC2, edgecolor=EDGE, linewidth=0.9)
        ax.text(start + dur / 2, y, f"{dur}w", ha="center", va="center",
                fontsize=7.5, **FONT)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.5,
                       fontfamily="Times New Roman")
    ax.set_xticks(range(16))
    ax.set_xticklabels(weeks, fontsize=8, fontfamily="Times New Roman")
    ax.set_xlim(0, 16)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d8dce2", linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)
    ax.axvline(12, color="#b4432f", linewidth=1.4, linestyle="--")
    ax.text(12.1, len(tasks) - 0.3, " Review 2", fontsize=8.5,
            color="#b4432f", fontfamily="Times New Roman", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_gantt.png", dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("wrote", OUT / "fig1_gantt.png")


# ---------------------------------------------------------------- Fig 2 ----
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
    finish(fig, ax, "fig2_architecture.png")


# ---------------------------------------------------------------- Fig 3 ----
def dfd():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    def store(x, y, w, h, label):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=ACC2,
                                   edgecolor=EDGE, linewidth=1.0))
        ax.plot([x, x + w], [y + h, y + h], color=EDGE, lw=1.0)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, **FONT)

    def proc(x, y, r, label):
        ax.add_patch(Ellipse((x, y), r * 2, r * 1.35, facecolor=ACC,
                             edgecolor=EDGE, linewidth=1.0))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, **FONT)

    ax.add_patch(plt.Rectangle((.02, .58), .15, .10, facecolor=BOX,
                               edgecolor=EDGE, linewidth=1.2))
    ax.text(.095, .63, "Clinician /\nresearcher", ha="center", va="center",
            fontsize=8.5, fontweight="bold", **FONT)

    store(.02, .20, .15, .10, "D1  Video corpus")
    proc(.31, .63, .095, "1.0\nExtract\nframes")
    store(.24, .20, .16, .10, "D2  Frame store")
    proc(.56, .63, .095, "2.0\nBuild\nlabel index")
    store(.48, .20, .16, .10, "D3  Index + splits")
    proc(.80, .63, .095, "3.0\nTrain /\nevaluate")
    store(.73, .20, .18, .10, "D4  Model + metrics")
    proc(.56, .87, .085, "4.0\nClassify\nframe")

    arrow(ax, (.17, .25), (.24, .55))
    arrow(ax, (.31, .55), (.31, .31))
    arrow(ax, (.40, .25), (.48, .55))
    arrow(ax, (.56, .55), (.56, .31))
    arrow(ax, (.64, .25), (.72, .55))
    arrow(ax, (.80, .55), (.82, .31))
    arrow(ax, (.17, .63), (.215, .63))
    arrow(ax, (.80, .72), (.655, .855))          # trained model -> classifier
    # the clinician submits a frame and gets a decision back
    arrow(ax, (.115, .690), (.475, .880))
    arrow(ax, (.470, .850), (.110, .660))

    ax.text(.335, .43, "frames", fontsize=7.5, rotation=90, **FONT)
    ax.text(.585, .43, "labels", fontsize=7.5, rotation=90, **FONT)
    ax.text(.83, .43, "results", fontsize=7.5, rotation=90, **FONT)
    ax.text(.255, .830, "query frame", fontsize=7.5, rotation=27, **FONT)
    ax.text(.230, .700, "probability + decision", fontsize=7.5, rotation=27, **FONT)
    ax.text(.740, .820, "trained model", fontsize=7.5, rotation=-40, **FONT)
    ax.text(.205, .44, "corpus", fontsize=7.5, rotation=62, **FONT)
    finish(fig, ax, "fig3_dfd.png")


# ---------------------------------------------------------------- Fig 4 ----
def usecase():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    def actor(x, y, label):
        ax.plot([x], [y + .055], marker="o", ms=8, mfc="white",
                mec=EDGE, mew=1.2)
        ax.plot([x, x], [y + .035, y - .015], color=EDGE, lw=1.2)
        ax.plot([x - .028, x + .028], [y + .018, y + .018], color=EDGE, lw=1.2)
        ax.plot([x, x - .022], [y - .015, y - .06], color=EDGE, lw=1.2)
        ax.plot([x, x + .022], [y - .015, y - .06], color=EDGE, lw=1.2)
        ax.text(x, y - .10, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", **FONT)

    def uc(x, y, label):
        ax.add_patch(Ellipse((x, y), .26, .10, facecolor=ACC,
                             edgecolor=EDGE, linewidth=1.0))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.2, **FONT)

    ax.add_patch(plt.Rectangle((.26, .06), .48, .88, fill=False,
                               edgecolor=EDGE, linewidth=1.1))
    ax.text(.50, .955, "Standard-plane classification system",
            ha="center", fontsize=9.5, fontweight="bold", **FONT)

    actor(.10, .62, "Researcher")
    actor(.90, .48, "Clinician")

    ys = [.85, .72, .59, .46, .33, .20]
    labels = ["Prepare dataset\nand build index",
              "Configure sampling\ncondition",
              "Train model", "Evaluate on\nheld-out test set",
              "Submit an\nultrasound frame",
              "View plane decision\nand probability"]
    for y, l in zip(ys, labels):
        uc(.50, y, l)

    for y in ys[:4]:
        arrow(ax, (.135, .60), (.375, y), style="-")
    for y in ys[4:]:
        arrow(ax, (.865, .48), (.625, y), style="-")
    # "Submit a frame" includes "View the decision" -- keep the connector in
    # the gap between the two ellipses so it does not cross either of them.
    arrow(ax, (.50, .281), (.50, .252), style="-|>", ls="--")
    ax.text(.525, .266, "«include»", fontsize=7.2, style="italic",
            va="center", **FONT)
    finish(fig, ax, "fig4_usecase.png")




# ---------------------------------------------------------------- Fig 5 ----
def litreview_table():
    """Literature-survey table for the PPT's object placeholder on slide 5."""
    rows = [
        ["Study", "Focus", "Data regime", "Limitation this project addresses"],
        ["Chen et al.\n(2015)", "Transferred CNN features for\nfetal standard plane localisation",
         "Video frames", "Density vs sample count\nnot separated"],
        ["Baumgartner et al.\n(2017) SonoNet", "Real-time detection of 13 fetal\nstandard planes",
         "Freehand sweeps", "Single-run results; no\nvariance reported"],
        ["Burgos-Artizzu et al.\n(2020)", "6-class maternal-fetal plane\nclassification benchmark",
         "Curated images\n(~12,400)", "Compared across corpora,\nso confounded"],
        ["Ghi et al. (2018)\nKalache et al. (2009)", "ISUOG intrapartum guidelines;\nangle of progression",
         "Clinical protocol", "Defines the positive class,\nnot a model"],
        ["This project", "Standard-plane classification +\nsampling-density study",
         "One corpus, k frames\nper video", "Budget- and prior-matched\narms; 3 seeds"],
    ]
    widths = [0.17, 0.28, 0.20, 0.35]
    fig, ax = plt.subplots(figsize=(11.0, 2.5))
    tbl = ax.table(cellText=[r[1:] for r in rows], rowLabels=[r[0] for r in rows],
                   colWidths=widths[1:], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.7)
        cell.get_text().set_fontfamily("Times New Roman")
        if r == 0:
            cell.set_facecolor(ACC)
            cell.get_text().set_fontweight("bold")
        elif r == len(rows) - 1:
            cell.set_facecolor(ACC2)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
        if c == -1:
            cell.get_text().set_fontweight("bold")
    ax.axis("off")
    fig.savefig(OUT / "fig5_litreview.png", dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print("wrote", OUT / "fig5_litreview.png")


if __name__ == "__main__":
    gantt()
    architecture()
    dfd()
    usecase()
    litreview_table()
