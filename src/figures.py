"""Paper figures. Fig. 2: perception boundary (detection vs distance per stratum; d50 day vs night).

Usage: .venv/bin/python src/figures.py --analysis results/analysis/runA --out paper/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# categorical slots 1-5 of the validated palette, fixed order by stratum (never cycled)
STRATA = ["boston_day", "boston_rain", "singapore_day", "singapore_night", "singapore_night_rain"]
LABEL = {"boston_day": "Boston day", "boston_rain": "Boston rain", "singapore_day": "Singapore day",
         "singapore_night": "Singapore night", "singapore_night_rain": "Singapore night + rain"}
COLOR = dict(zip(STRATA, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARK = dict(zip(STRATA, ["o", "s", "^", "D", "v"]))
INK, MUTED, GRID = "#2b2b2a", "#6f6e69", "#e6e5e0"
BIN_MID = {"(0, 15]": 0, "(15, 25]": 1, "(25, 35]": 2, "(35, 50]": 3, "(50, 200]": 4}
CURVE_STRATA = STRATA[:4]  # night+rain has 49 positives: too few per bin, reported in text
NICE = {"Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
        "InternVL3-2B-hf": "InternVL3-2B", "InternVL3-8B-hf": "InternVL3-8B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B", "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "SmolVLM2-2.2B-Instruct": "SmolVLM2-2.2B"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def fig2_curves(an: Path, out: Path,
                rep=("Qwen3-VL-8B-Instruct", "InternVL3-8B-hf", "Qwen2.5-VL-7B-Instruct")):
    """Fig. 2: P(pedestrian detected) vs distance per condition, one panel per model family."""
    cur = pd.read_csv(an / "curves.csv")
    rep = [m for m in rep if m in set(cur.model)]
    fig, axes = plt.subplots(1, len(rep), figsize=(7.0, 2.7), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), rep):
        g = cur[cur.model == m]
        for st in CURVE_STRATA:
            h = g[(g.stratum == st) & (g["size"] >= 15)].copy()
            if h.empty:
                continue
            h["x"] = h.dbin.map(BIN_MID)
            h = h.dropna(subset=["x"]).sort_values("x")
            k = np.round(h["mean"] * h["size"]).astype(int)
            lo, hi = proportion_confint(k, h["size"], method="wilson")
            ax.fill_between(h.x, lo, hi, color=COLOR[st], alpha=0.10, lw=0)
            ax.plot(h.x, h["mean"], color=COLOR[st], lw=1.8, marker=MARK[st], ms=4.5,
                    mec="white", mew=0.6, label=LABEL[st])
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.set_title(NICE[m], color=INK, fontsize=9, loc="left")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.3, 4.3)
        ax.set_xticks(range(5), ["<15", "15–25", "25–35", "35–50", ">50"], fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlabel("Distance to pedestrian (m)")
    np.atleast_1d(axes)[0].set_ylabel("P(pedestrian detected)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig2_curves")


def fig3_d50(an: Path, out: Path):
    """Fig. 3: d50 for all models, dumbbells: (a) Singapore day vs night, (b) Boston day vs rain."""
    d50 = pd.read_csv(an / "d50.csv")
    models = [m for m in NICE if m in set(d50.model)]
    models = sorted(models, key=lambda m: d50[(d50.model == m) & (d50.stratum == "singapore_day")].d50.mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (a, b, title) in zip(axes, (("singapore_day", "singapore_night", "Night (Singapore)"),
                                        ("boston_day", "boston_rain", "Rain (Boston)"))):
        for y, m in enumerate(models):
            ra = d50[(d50.model == m) & (d50.stratum == a)].iloc[0]
            rb = d50[(d50.model == m) & (d50.stratum == b)].iloc[0]
            for r, st, dy in ((ra, a, 0.17), (rb, b, -0.17)):
                ax.plot([r.ci_lo, min(r.ci_hi, 140)], [y + dy, y + dy], color=COLOR[st], lw=1.1, alpha=0.8, zorder=2)
                ax.plot(r.d50, y + dy, MARK[st], color=COLOR[st], ms=6, mec="white", mew=0.7, zorder=3,
                        label=LABEL[st] if y == 0 else None)
            ratio = rb.d50 / ra.d50
            ax.text(145, y, f"×{ratio:.2f}", va="center", fontsize=6.8, color=INK)
        ax.text(145, len(models) - 0.35, "ratio", va="center", fontsize=6.5, color=MUTED)
        ax.set_yticks(range(len(models)), [NICE[m] for m in models], fontsize=7.5)
        ax.set_xlim(0, 160)
        ax.set_xticks([0, 25, 50, 75, 100, 125])
        ax.set_xlabel("Distance of 50% pedestrian detection, $d_{50}$ (m)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_ylim(-0.6, len(models) - 0.1)
    h = sum((list(ax.get_legend_handles_labels()[0]) for ax in axes), [])
    l = sum((list(ax.get_legend_handles_labels()[1]) for ax in axes), [])
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig3_d50")


MR_LABEL = {"paraphrase": "Paraphrase", "mirror": "Mirror", "zoom": "Zoom consistency"}


def fig4_metamorphic(meta: Path, out: Path, focus=("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct")):
    """Fig. 4: per operational cell (stratum x distance bin), reference miss rate vs violation rate of each
    label-free relation; one row per model."""
    import json
    cells = pd.read_csv(meta / "cells.csv")
    res = json.load(open(meta / "metamorphic.json"))
    focus = [m for m in focus if m in res]
    mrs = list(MR_LABEL)
    fig, axes = plt.subplots(len(focus), 3, figsize=(7.0, 2.35 * len(focus)), squeeze=False)
    for r, m in enumerate(focus):
        for c, mr in enumerate(mrs):
            ax = axes[r, c]
            if mr not in res[m]:
                ax.set_axis_off()
                continue
            cc = cells[(cells.base == m) & (cells.mr == mr)]
            for st in STRATA:
                h = cc[cc.stratum == st]
                if len(h):
                    ax.scatter(h.miss, h.viol, s=14 + h.n / 7, color=COLOR[st], marker=MARK[st],
                               edgecolor="white", linewidth=0.6, label=LABEL[st], zorder=3)
            rho = res[m][mr].get("cells_spearman", {})
            rv = rho.get('rho', np.nan); rv = 0.0 if abs(rv) < 0.005 else rv
            ax.set_title(f"{MR_LABEL[mr]}  ($\\rho$ = {rv:.2f})", color=INK, fontsize=8.5, loc="left")
            ax.set_xlim(-0.03, 1.0)
            ax.set_ylim(bottom=0)
            ax.grid(color=GRID, lw=0.5)
            if r == len(focus) - 1:
                ax.set_xlabel("Reference miss rate (labels)")
            if c == 0:
                ax.set_ylabel(f"{NICE[m]}\nViolation rate (no labels)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig4_metamorphic")


def figS1_lift(meta: Path, out: Path):
    """Supplementary Fig. S1: lift = P(error | violation) / P(error) with scene-bootstrap 95% CI."""
    import json
    res = json.load(open(meta / "metamorphic.json"))
    models = [m for m in NICE if m in res]
    mr_color = {"paraphrase": "#2a78d6", "mirror": "#eb6834", "zoom": "#1baf7a"}
    mr_mark = {"paraphrase": "o", "mirror": "s", "zoom": "^"}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    y, yt, yl, seen = 0, [], [], set()
    for m in models:
        for mr in MR_LABEL:
            if mr not in res[m]:
                continue
            s = res[m][mr]
            lo, hi = s["ci_lift"]
            ax.plot([lo, hi], [y, y], color=mr_color[mr], lw=1.2)
            ax.plot(s["lift"], y, mr_mark[mr], color=mr_color[mr], ms=5, mec="white", mew=0.6,
                    label=MR_LABEL[mr] if mr not in seen else None)
            seen.add(mr)
            yt.append(y)
            yl.append(f"{NICE[m]} · {MR_LABEL[mr].split()[0].lower()}")
            y -= 1
        y -= 0.5
    ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_yticks(yt, yl, fontsize=7)
    ax.set_xlabel("Lift: P(error | violation) / P(error), 95% CI")
    ax.set_xlim(0, 8)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "figS1_lift")


def fig1_workflow(out: Path, img="data/check_imgs/box_night.jpg"):
    """Fig. 1: end-to-end workflow of the testing framework (Algorithm 1), with equation and algorithm
    references matching the Methodology section."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from PIL import Image
    BLUE, ORANGE, AQUA, YEL, MAG, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
    PANEL = "#f4f3ef"
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 80)

    def box(x, y, w, h, title, body="", ec=MUTED, fc="white", tfs=6.8, bfs=5.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tfs, color=INK, fontweight="bold")
        if body:
            ax.text(x + w / 2, (y + h - 4.3) if title else (y + h / 2), body, ha="center", va="top" if title else "center", fontsize=bfs, color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1, c=MUTED):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=c, shrinkA=0,
                                                            shrinkB=0, mutation_scale=8))

    def stage(y, label):
        ax.text(0.8, y, label, rotation=90, ha="center", va="center", fontsize=7, color=MUTED, fontweight="bold")

    # ---- Stage 1: data and oracle
    stage(71, "1  DATA & ORACLE")
    axi = fig.add_axes([0.03, 0.80, 0.22, 0.175]); axi.imshow(Image.open(img)); axi.set_axis_off()
    sc = Image.open(img).size[0] / 1600
    for k, x0 in enumerate((0, 500, 1000)):
        axi.add_patch(Rectangle((x0 * sc, 360 * sc), 600 * sc, 300 * sc, fill=False, ec=YEL, lw=0.8))
    box(27, 64, 16, 14, "3D boxes → image", "project every box\n(Eq. 1); keep class,\ndistance, visibility,\nlidar/radar returns", fc=PANEL)
    box(46, 64, 16, 14, "Strata & split", "5 city × condition\nstrata; every 4th\nkeyframe; scene-level\ncalib/test split", fc=PANEL)
    box(65, 64, 16, 14, "Questions", "6 classes × 2 wordings\n= 39,096 questions;\npositive / negative\nlabels (Eq. 2)", fc=PANEL)
    box(84, 64, 15, 14, "Views", "full · mirror ·\n3 zoom tiles ·\nnoisy (VCD)", fc=PANEL)
    for x0, x1 in ((25.5, 27), (43, 46), (62, 65), (81, 84)):
        arrow(x0, 71, x1, 71)
    # ---- Stage 2: model query
    stage(54.5, "2  QUERY")
    box(20, 50, 62, 9, "VLM under test: one forward pass per view",
        "P(yes) from first-token logits (Eq. 3) · image + prompt encoded once, questions batched from the KV cache\n"
        "7 open VLMs (2–8 B), fp16 on NVIDIA T4", ec=BLUE)
    arrow(54, 64, 54, 59.2)
    # ---- Stage 3: four test modules
    stage(34.5, "3  TEST MODULES")
    cols = [(2, AQUA, "T1 Operational boundary", "oracle labels vs answers\n→ detection vs distance\n$d_{50}$ (Eq. 4)\nGEE odds ratios (Eq. 5)"),
            (26.5, ORANGE, "T2 Metamorphic tests", "paraphrase · mirror · zoom\nviolations without labels\n(Eqs. 6–7); validated by\nlift and ρ vs oracle (Eq. 8)"),
            (51, MAG, "T3 Fault injection", "Algorithm 2: 1,000 bit flips\nper setting (Eqs. 9–10)\nweights / activations ·\nvision / language model\noutcomes (Eq. 11), Ranger (Eq. 12)"),
            (75.5, VIO, "T4 Safeguard", "Algorithm 3: score per method\n(Eq. 13); threshold by conformal\nrisk control on calibration\nscenes (Eq. 14); evaluate on\ntest scenes")]
    for x, c, t, b in cols:
        box(x, 26, 22.5, 17, t, b, ec=c, tfs=6.6, bfs=5.9)
        arrow(51, 50, x + 11.25, 43.4, c=c)
    # ---- Stage 4: outputs
    stage(8.5, "4  OUTPUTS")
    outs = [(2, "$d_{50}$ per model and\ncondition; night / rain\neffects (Fig. 2)"),
            (26.5, "which tests reveal the\nboundary without labels\n(Fig. 3)"),
            (51, "SDC / DUE rates, margin\ndependence, Ranger effect\n(Fig. 4)"),
            (75.5, "false alarms at a\nguaranteed miss rate;\ncost (Fig. 5)")]
    for (x, t), (_, c, _, _) in zip(outs, cols):
        box(x, 3, 22.5, 11, "", t, ec=c, fc=PANEL, bfs=6.2)
        arrow(x + 11.25, 26, x + 11.25, 14.4, c=c)
    save(fig, out, "fig1_workflow")


def fig_overview(out: Path, img="data/check_imgs/box_night.jpg"):
    """Fig. 1: framework. (a) oracle on a real nuScenes night frame with projected 3D boxes and the
    three horizon tiles; (b) metamorphic relations; (c) fault injection sites and outcomes;
    (d) risk-controlled safeguard."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from PIL import Image
    BLUE, ORANGE, AQUA, YEL, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
    PANEL = "#f4f3ef"

    def box(ax, x, y, w, h, text, fc="white", ec=MUTED, fs=6.6, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc=fc, ec=ec, lw=0.7, clip_on=False))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(ax, x0, y0, x1, y1, c=MUTED):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c,
                                                            shrinkA=0, shrinkB=0, mutation_scale=7))

    fig = plt.figure(figsize=(7.2, 4.1))
    # (a) oracle image
    axa = fig.add_axes([0.0, 0.40, 0.50, 0.55])
    im = Image.open(img)
    sc = im.size[0] / 1600
    axa.imshow(im)
    for k, x0 in enumerate((0, 500, 1000)):
        axa.add_patch(Rectangle((x0 * sc, 360 * sc), 600 * sc, 300 * sc, fill=False, ec=YEL, lw=1.1,
                                ls=(0, (4, 2)) if k == 1 else "-"))
    axa.text(8, 360 * sc - 8, "zoom tiles (far-field band)", color=YEL, fontsize=6.3, va="bottom")
    axa.set_axis_off()
    fig.text(0.005, 0.965, "a", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.03, 0.965, "Automated oracle from 3D annotations", fontsize=8, color=INK)
    axq = fig.add_axes([0.01, 0.02, 0.48, 0.36]); axq.set_axis_off(); axq.set_xlim(0, 1); axq.set_ylim(0, 1)
    box(axq, 0.00, 0.62, 0.30, 0.30, "nuScenes 3D boxes\n(class, distance,\nvisibility, lidar pts)", fc=PANEL)
    box(axq, 0.35, 0.62, 0.30, 0.30, "\"Is there a pedestrian\nin this image?\"\n→ P(yes) from logits", fc=PANEL)
    box(axq, 0.70, 0.62, 0.30, 0.30, "hit / miss / false alarm\nscored without\nhuman or LLM judge", fc=PANEL)
    arrow(axq, 0.30, 0.77, 0.35, 0.77); arrow(axq, 0.65, 0.77, 0.70, 0.77)
    box(axq, 0.00, 0.08, 1.00, 0.40, "Operational boundary: detection vs distance, visibility, night, rain\n"
        "→ $d_{50}$ per condition  ·  same-city comparisons  ·  scene-clustered statistics", fc="white")
    arrow(axq, 0.85, 0.62, 0.85, 0.48)

    # (b) metamorphic relations
    axb = fig.add_axes([0.53, 0.68, 0.45, 0.27]); axb.set_axis_off(); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    fig.text(0.53, 0.965, "b", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.555, 0.965, "Label-free metamorphic tests", fontsize=8, color=INK)
    box(axb, 0.00, 0.30, 0.31, 0.55, "Paraphrase\ntwo wordings\nmust agree", ec=BLUE)
    box(axb, 0.345, 0.30, 0.31, 0.55, "Mirror\nflipped image\nmust agree", ec=ORANGE)
    box(axb, 0.69, 0.30, 0.31, 0.55, "Zoom\nyes on a tile ⇒\nyes on full image", ec=AQUA)
    axb.text(0.5, 0.06, "violation rate validated against the oracle", ha="center", fontsize=6.4, color=MUTED)

    # (c) fault injection
    axc = fig.add_axes([0.53, 0.36, 0.45, 0.25]); axc.set_axis_off(); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    fig.text(0.53, 0.645, "c", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.555, 0.645, "Statistical fault injection (fp16 bit flips)", fontsize=8, color=INK)
    box(axc, 0.00, 0.50, 0.17, 0.32, "image +\nquestion", fc=PANEL)
    box(axc, 0.22, 0.50, 0.22, 0.32, "vision\nencoder", ec=BLUE)
    box(axc, 0.49, 0.50, 0.22, 0.32, "language\nmodel", ec=BLUE)
    box(axc, 0.76, 0.50, 0.24, 0.32, "P(yes) vs\nfault-free run", fc=PANEL)
    for x0, x1 in ((0.17, 0.22), (0.44, 0.49), (0.71, 0.76)):
        arrow(axc, x0, 0.66, x1, 0.66)
    for xc in (0.33, 0.60):
        axc.text(xc, 0.92, "⚡", ha="center", va="center", fontsize=9, color=MAG)
    axc.text(0.66, 0.92, "bit flip in a weight or activation\n1,000 injections per setting", ha="left",
             fontsize=6.0, color=MAG, va="center")
    axc.text(0.5, 0.24, "benign  ·  DUE (crash/NaN)  ·  SDC-miss  ·  SDC-false alarm", ha="center",
             fontsize=6.4, color=INK)
    axc.text(0.5, 0.04, "with and without Ranger (range restriction to profiled bounds)", ha="center",
             fontsize=6.2, color=MUTED)

    # (d) safeguard
    axd = fig.add_axes([0.53, 0.02, 0.45, 0.30]); axd.set_axis_off(); axd.set_xlim(0, 1); axd.set_ylim(0, 1)
    fig.text(0.53, 0.325, "d", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.555, 0.325, "Runtime safeguard with a guarantee", fontsize=8, color=INK)
    box(axd, 0.00, 0.52, 0.22, 0.30, "full image\n$p_f$", fc=PANEL)
    box(axd, 0.00, 0.12, 0.22, 0.30, "3 zoom tiles\n$p_t$", fc=PANEL, ec=YEL)
    box(axd, 0.30, 0.30, 0.25, 0.34, "score\n(any method)", ec=BLUE)
    box(axd, 0.63, 0.30, 0.37, 0.34, "threshold by conformal\nrisk control on\ncalibration scenes", ec=BLUE)
    arrow(axd, 0.22, 0.67, 0.30, 0.52); arrow(axd, 0.22, 0.27, 0.30, 0.42); arrow(axd, 0.55, 0.47, 0.63, 0.47)
    axd.text(0.815, 0.12, "VRU miss rate ≤ α on new scenes", ha="center", fontsize=6.4, color=INK)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_overview.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", out / "fig1_overview.pdf")


METH = {  # fixed colour per method (never reassigned); vanilla = neutral reference
    "vanilla": ("Unaided model", "#6f6e69", "o"), "fused": ("FZ (ours)", "#eb6834", "D"),
    "fused_g": ("GFZ (ours)", "#eb6834", "d"), "vcd": ("VCD", "#2a78d6", "s"),
    "zoom_or": ("Zoom-OR", "#1baf7a", "^"), "vicrop": ("ViCrop", "#eda100", "v"),
    "see": ("SEE", "#e87ba4", "P"), "ug": ("UG-Search", "#6250d6", "X")}


def fig_safeguard(sg_dir: Path, out: Path, models=("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct")):
    cur = pd.read_csv(sg_dir / "curves.csv")
    models = [m for m in models if m in set(cur.model)]
    fig, axes = plt.subplots(len(models), 3, figsize=(7.2, 2.45 * len(models)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for r, m in enumerate(models):
        for c, (setname, meths, title) in enumerate((
                ("benchmark", ["vanilla", "vcd", "zoom_or", "fused"], "All 3,359 frames"),
                ("baseline_subset", ["vanilla", "vicrop", "see", "ug", "fused"], "700-frame subset, all methods"))):
            ax = axes[r, c]
            g = cur[(cur.model == m) & (cur.set == setname)]
            for k in meths:
                h = g[g.method == k].sort_values("miss")
                if h.empty:
                    continue
                lab, col, mk = METH[k]
                ax.plot(h.miss * 100, h.fa * 100, color=col, lw=2.0 if k.startswith("fused") else 1.3,
                        marker=mk, ms=3.2, markevery=6, mec="white", mew=0.5, label=lab,
                        zorder=4 if k.startswith("fused") else 3)
            ax.set_xlim(2, 40)
            ax.set_ylim(0, None)
            ax.grid(color=GRID, lw=0.5)
            ax.set_xlabel("Missed pedestrians/motorcycles (%)")
            if c == 0:
                ax.set_ylabel(f"{NICE.get(m, m)}\nFalse alarms (%)")
            ax.set_title(title, color=INK, fontsize=8, loc="left")
            if c == 0:
                ax.legend(fontsize=6.2, loc="upper right", handlelength=1.6)
            else:
                ax.legend(fontsize=6.2, loc="upper right", bbox_to_anchor=(1.0, 0.92), handlelength=1.6)
        # cost panel: false alarms at 20% miss vs passes per question (subset; benchmark for gated)
        ax = axes[r, 2]
        g = cur[(cur.model == m) & (cur.set == "baseline_subset") & (cur.miss == 0.2)]
        crc = pd.read_csv(sg_dir / "crc.csv")
        cost = crc[(crc.model == m) & (crc.set == "baseline_subset") & (crc.alpha == 0.2)].set_index("method").passes
        for k in ["vanilla", "fused_g", "fused", "vicrop", "see", "ug"]:
            if k == "fused_g" and k in cost and abs(cost[k] - cost.get("fused", -1)) < 0.05:
                continue  # GFZ identical to FZ for this model (no confident answers to skip)
            if k in set(g.method) and k in cost:
                lab, col, mk = METH[k]
                fa = g[g.method == k].fa.iloc[0] * 100
                ax.plot(cost[k], fa, mk, color=col, ms=6, mec="white", mew=0.6)
                off = {"vanilla": (-4, 9), "fused_g": (-6, -13), "fused": (6, 4), "vicrop": (-8, -13)}.get(k, (4, 3))
                ax.annotate(lab.replace(" (ours)", "*"), (cost[k], fa),
                            xytext=off, textcoords="offset points", fontsize=6, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 10, 20], ["1", "2", "4", "10", "20"])
        ax.set_xlim(0.8, 40)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_xlabel("Model passes per question")
        ax.set_ylabel("False alarms at 20% miss (%)")
        ax.grid(color=GRID, lw=0.5)
        ax.set_title("Cost", color=INK, fontsize=8, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig6_safeguard.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", out / "fig6_safeguard.pdf")


def figS2_curves_all(an: Path, out: Path):
    """Supplementary Fig. S2: pedestrian detection against distance for all seven models."""
    cur = pd.read_csv(an / "curves.csv")
    models = [m for m in NICE if m in set(cur.model)]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, models):
        g = cur[cur.model == m]
        for st in CURVE_STRATA:
            h = g[(g.stratum == st) & (g["size"] >= 15)].copy()
            if h.empty:
                continue
            h["x"] = h.dbin.map(BIN_MID)
            h = h.dropna(subset=["x"]).sort_values("x")
            k = np.round(h["mean"] * h["size"]).astype(int)
            lo, hi = proportion_confint(k, h["size"], method="wilson")
            ax.fill_between(h.x, lo, hi, color=COLOR[st], alpha=0.10, lw=0)
            ax.plot(h.x, h["mean"], color=COLOR[st], lw=1.3, marker=MARK[st], ms=3, mec="white", mew=0.5, label=LABEL[st])
        ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (3, 3)))
        ax.set_title(NICE[m], color=INK, fontsize=7.5, loc="left")
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.3, 4.3)
        ax.set_xticks(range(5), ["<15", "15–25", "25–35", "35–50", ">50"], fontsize=5.5)
        ax.tick_params(labelsize=5.5)
        ax.grid(axis="y", color=GRID, lw=0.4)
    leg = axes.flat[-1]; leg.set_axis_off()
    h, l = axes.flat[0].get_legend_handles_labels()
    leg.legend(h, l, loc="center", fontsize=7)
    fig.supxlabel("Distance to pedestrian (m)", fontsize=7.5); fig.supylabel("P(pedestrian detected)", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "figS2_curves_all")


def fig4_faults(out: Path, fi_dirs=("results/runD/out", "results/runE/out", "results/runG/out"), rates_csv="results/analysis/fi_all/rates.csv"):
    """Fig. 4: (a) SDC and DUE rate per configuration with Wilson CIs; (b) SDC rate by flipped bit;
    (c) decision-flip rate of near-threshold vs other questions; (d) weight-fault SDC-miss vs distance."""
    import ast
    import sys
    sys.path.insert(0, "src")
    import fi_analyze as FA
    rates = pd.read_csv(rates_csv)
    df = FA.load(list(fi_dirs))
    probes = pd.read_parquet("results/probes/probes.parquet").set_index("probe_id")
    df = df.join(probes[["easy_dist"]], on="probe_id")
    inj = FA.injection_level(df)
    SITE = {"vision": "vision", "llm": "language"}
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.75, wspace=0.42)
    # (a)
    ax = fig.add_subplot(gs[:, 0])
    order = []
    for m in ("Qwen3-VL-4B-Instruct", "InternVL3-2B-hf"):
        for kind, bits in (("act", "all"), ("act", "hi"), ("weight", "all")):
            for site in ("vision", "llm"):
                for prot in ("none", "ranger"):
                    cfg = f"{site}:{kind}:{prot}:{bits}"
                    if ((rates.model == m) & (rates.cfg == cfg)).any():
                        order.append((m, cfg))
    y = 0; yt, yl, last_m = [], [], None
    for m, cfg in order:
        if last_m is not None and m != last_m:
            y -= 0.8
        r = rates[(rates.model == m) & (rates.cfg == cfg)].iloc[0]
        site, kind, prot, bits = cfg.split(":")
        for col, key, mk in (("#e87ba4", "SDC", "o"), ("#6f6e69", "DUE", "s")):
            lo, hi = ast.literal_eval(r[f"{key}_ci"])
            ax.plot([100 * lo, 100 * hi], [y, y], color=col, lw=1.0, alpha=0.8)
            ax.plot(100 * r[key], y, mk, color=col, ms=3.6 if key == "SDC" else 3.2, mec="white", mew=0.4,
                    label=(key if (m, cfg) == order[0] else None))
        lab = f"{SITE[site]} · {'weight' if kind == 'weight' else ('act. exp.' if bits == 'hi' else 'activation')}{' · R' if prot == 'ranger' else ''}"
        yt.append(y); yl.append(lab); y -= 1; last_m = m
    ax.set_yticks(yt, yl, fontsize=5.6)
    ax.set_xlabel("Injections (%), Wilson 95% CI", fontsize=7)
    ax.set_xlim(0, 6)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.legend(fontsize=6, loc="lower right")
    n1 = sum(1 for m, _ in order if m == "Qwen3-VL-4B-Instruct")
    ax.text(5.9, yt[0] + 0.6, "Qwen3-VL-4B", ha="right", fontsize=6.5, color=INK, fontweight="bold")
    ax.text(5.9, yt[n1] + 0.6, "InternVL3-2B", ha="right", fontsize=6.5, color=INK, fontweight="bold")
    ax.set_title("a  Outcome per configuration", fontsize=8, loc="left", color=INK)
    # (b) SDC by bit, unprotected, uniform-bit campaigns
    ax = fig.add_subplot(gs[0, 1])
    u = inj[inj.cfg_full.str.contains(":none:all")]
    u = u.assign(kind=u.cfg_full.str.split(":").str[1])
    for kind, col, off in (("act", "#2a78d6", -0.2), ("weight", "#eb6834", 0.2)):
        b = u[u.kind == kind].groupby("bit").sdc.mean().reindex(range(16), fill_value=0) * 100
        ax.bar(np.arange(16) + off, b.values, width=0.4, color=col, label="activation" if kind == "act" else "weight")
    ax.axvspan(9.5, 15.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.text(12.5, ax.get_ylim()[1] * 0.92, "exponent / sign", ha="center", fontsize=5.5, color=MUTED)
    ax.set_xticks([0, 5, 10, 14, 15]); ax.tick_params(labelsize=5.5)
    ax.set_xlabel("Flipped bit (fp16)", fontsize=6.5); ax.set_ylabel("SDC (%)", fontsize=6.5)
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("b  SDC rate by bit", fontsize=8, loc="left", color=INK)
    # (c) near-threshold vs other questions
    ax = fig.add_subplot(gs[1, 1])
    ok = df[df.outcome != "DUE"].copy()
    ok["flip"] = ok.outcome.str.startswith("SDC"); ok["near"] = (ok.p_gold - 0.5).abs() < 0.1
    ok["kind"] = ok.cfg_full.str.split(":").str[1]
    groups = [("Qwen3-VL-4B-Instruct", "act"), ("Qwen3-VL-4B-Instruct", "weight"), ("InternVL3-2B-hf", "act"), ("InternVL3-2B-hf", "weight")]
    xs = np.arange(len(groups))
    for j, (near, col, lab) in enumerate(((True, "#e87ba4", "near-threshold"), (False, "#c3c2b7", "other"))):
        vals = [100 * ok[(ok.model == m) & (ok.kind == k) & (ok.near == near)].flip.mean() for m, k in groups]
        ax.bar(xs + (j - 0.5) * 0.38, vals, width=0.38, color=col, label=lab)
    ax.set_yscale("log"); ax.set_ylim(0.005, 20)
    ax.set_xticks(xs, ["Qwen3\nact.", "Qwen3\nweight", "InternVL3\nact.", "InternVL3\nweight"], fontsize=5.3)
    ax.tick_params(axis="y", labelsize=5.5); ax.set_ylabel("Decision flips (%)", fontsize=6.5)
    ax.legend(fontsize=5.5, loc="upper left", ncol=2)
    ax.set_title("c  Flips by decision margin", fontsize=8, loc="left", color=INK)
    # (d) weight-fault SDC-miss among detected pedestrians vs distance
    ax = fig.add_subplot(gs[2, 1])
    w = ok[(ok.kind == "weight") & (ok.cls == "pedestrian") & (ok.label == 1) & (ok.p_gold >= 0.5)]
    w = w.assign(dbin=pd.cut(w.easy_dist, [0, 15, 25, 35, 50, 200], labels=["<15", "15–25", "25–35", "35–50", ">50"]))
    g = w.groupby("dbin", observed=True).outcome.agg(lambda o: (o == "SDC-miss").sum()).astype(int)
    n = w.groupby("dbin", observed=True).size()
    lo, hi = proportion_confint(g.values, n.values, method="wilson")
    x = np.arange(len(g))
    ax.errorbar(x, 100 * g.values / n.values, yerr=[100 * (g.values / n.values - lo), 100 * (hi - g.values / n.values)],
                fmt="o-", color="#eb6834", ms=3.5, lw=1.2, capsize=1.5, elinewidth=0.7)
    ax.set_xticks(x, list(g.index), fontsize=5.5); ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("Distance to pedestrian (m)", fontsize=6.5); ax.set_ylabel("SDC-miss (%)", fontsize=6.5)
    ax.set_ylim(0, None); ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_title("d  Weight faults: misses by distance", fontsize=8, loc="left", color=INK)
    save(fig, out, "fig5_faults")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis/all")
    ap.add_argument("--meta", default="results/analysis/metamorphic_all")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--only", default="overview,boundary,metamorphic")
    a = ap.parse_args()
    if "boundary" in a.only:
        fig2_curves(Path(a.analysis), Path(a.out))
        fig3_d50(Path(a.analysis), Path(a.out))
    if "metamorphic" in a.only:
        fig4_metamorphic(Path(a.meta), Path(a.out))
        figS1_lift(Path(a.meta), Path(a.out))
    if "overview" in a.only:
        fig1_workflow(Path(a.out))
    if "faults" in a.only:
        fig4_faults(Path(a.out))
    if "supp" in a.only:
        fig2 = Path(a.analysis); figS2_curves_all(fig2, Path(a.out))
    if "safeguard" in a.only:
        fig_safeguard(Path("results/analysis/safeguard"), Path(a.out))
