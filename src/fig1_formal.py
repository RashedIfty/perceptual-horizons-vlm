"""Formal Fig. 1 draft (for approval), following the structure of recent VLM / reliability papers:
numbered stages left to right, one real worked example carried through the figure, branching into the test
modules, little text inside (details in the caption), green = correct / red = wrong.
Example: probe 28066 (calibration split), Singapore night, one fully visible pedestrian
at 43.4 m; all seven VLMs answer "no" on the full image; Qwen3-VL-4B answers "yes" on the zoom tile that
contains the pedestrian. Writes paper/figures/drafts/fig1_formal.{pdf,png}."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

INK, MUTED, GRID, PANEL = "#2b2b2a", "#6f6e69", "#e6e5e0", "#f6f5f1"
BLUE, ORANGE, AQUA, YEL, MAG, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
GOOD, BAD = "#0a8a3a", "#d0352b"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 1.5,
                     "ytick.major.size": 1.5, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
IMG = "data/fig_example/n015-2018-11-21-19-11-29+0800__CAM_FRONT__1542799048862460.jpg"
PED = (774.0, 494.6, 793.1, 546.1)  # projected 3D box of the pedestrian (x1, y1, x2, y2), 43.4 m
TILES = (0, 500, 1000)               # left edges; band y = 360-660
OUT = Path("paper/figures/drafts")
CW, CH = 100, 56                     # canvas units (square: 7.2 in x 4.03 in)


def axes_at(fig, x, y, w, h):
    return fig.add_axes([x / CW, y / CH, w / CW, h / CH])


def rbox(ax, x, y, w, h, ec=MUTED, fc="white", lw=0.7, r=0.7, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, x0, y0, x1, y1, c=MUTED, lw=0.8, style="-|>"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle=style, lw=lw, color=c, shrinkA=0, shrinkB=0,
                                                        mutation_scale=6))


def badge(ax, x, y, n, c=INK):
    ax.add_patch(Circle((x, y), 1.25, fc=c, ec="none"))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")


def tidy(a):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(pad=1)


def main():
    fig = plt.figure(figsize=(7.2, 7.2 * CH / CW))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.set_axis_off()

    # ------------------------------------------------------------------ stage headers
    for x, n, t in ((1.3, 1, "Reference labels"), (37.3, 2, "Query the VLM"), (66.3, 3, "Test modules")):
        badge(ax, x + 1.2, 54.2, n)
        ax.text(x + 3.3, 54.2, t, fontsize=8, fontweight="bold", color=INK, va="center")

    # ------------------------------------------------------------------ (1) oracle
    img = Image.open(IMG).convert("RGB")
    s = 32 / 1600
    a_img = axes_at(fig, 1, 33.5, 32, 18)
    a_img.imshow(img); a_img.set_axis_off()
    for k, x0 in enumerate(TILES):  # zoom tiles; the one containing the pedestrian is solid
        a_img.add_patch(Rectangle((x0 + 4, 360), 592, 300, fill=False, ec=YEL, lw=1.0 if k == 1 else 0.6,
                                  ls="-" if k == 1 else (0, (3, 2))))
    x1, y1, x2, y2 = PED
    a_img.add_patch(Rectangle((x1 - 6, y1 - 6), x2 - x1 + 12, y2 - y1 + 12, fill=False, ec=BAD, lw=1.1))
    a_img.annotate("pedestrian, 43 m", (x2 + 8, y1), (1060, 250), color="white", fontsize=5.6,
                   arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    a_img.text(20, 345, "zoom tiles", color=YEL, fontsize=5.2, va="bottom")
    ax.text(17, 32.1, "nuScenes front camera, Singapore at night", ha="center", fontsize=5.3, color=MUTED)

    rbox(ax, 1, 23.3, 32, 7.2, fc=PANEL, ec=GRID)
    ax.text(2.4, 28.7, "Question", fontsize=6, fontweight="bold", color=INK, va="center")
    ax.text(2.4, 25.4, "“Is there a pedestrian in this image?\nAnswer with yes or no.”",
            fontsize=5.7, color=INK, va="center", linespacing=1.25)
    rbox(ax, 1, 13.6, 32, 8.0, fc=PANEL, ec=GRID)
    ax.text(2.4, 19.7, "Reference label (Eqs. 1–2)", fontsize=6, fontweight="bold", color=INK,
            va="center")
    ax.text(2.4, 16.8, "$y^{\\star}=1$: projected box with a lidar return\nderived from the 3D annotations",
            fontsize=5.6, color=INK, va="center", linespacing=1.25)
    ax.text(17, 10.3, "3,359 frames · 821 scenes · 5 strata (city × condition)\n"
            "6 classes × 2 wordings = 39,096 questions",
            ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.4)
    arrow(ax, 33.6, 42.5, 37.2, 42.5)

    # ------------------------------------------------------------------ (2) query
    rbox(ax, 37.5, 33.5, 25, 18, ec=BLUE, fc="white", lw=0.9)
    ax.text(50, 50.0, "VLM under test (7 models, 2–8 B)", ha="center", fontsize=6, color=INK, va="center")
    rbox(ax, 40, 43.0, 20, 4.4, ec=BLUE, fc=PANEL)
    ax.text(50, 45.2, "vision encoder", ha="center", va="center", fontsize=6.2, color=INK)
    rbox(ax, 40, 36.2, 20, 4.4, ec=BLUE, fc=PANEL)
    ax.text(50, 38.4, "language model", ha="center", va="center", fontsize=6.2, color=INK)
    arrow(ax, 50, 43.0, 50, 40.6, c=BLUE)
    for yy in (45.2, 38.4):
        ax.text(61.2, yy, "⚡", fontsize=7.5, color=MAG, ha="center", va="center")
    ax.text(61.2, 48.3, "fault\nsites", fontsize=4.6, color=MAG, ha="center", va="center", linespacing=1.0)
    ax.text(50, 34.5, "$\\psi=P(\\mathrm{yes})$ from first-token logits (Eq. 3)", ha="center", va="center",
            fontsize=5.3, color=MUTED)
    arrow(ax, 50, 33.4, 50, 30.7, c=BLUE)

    # answers for the example (real values)
    def answer(y, thumb, label, psi, verdict, ok):
        a = axes_at(fig, 37.5, y, 7.6, 4.3 if thumb == "full" else 2.15 * 1.0)
        if thumb == "full":
            a.imshow(img)
        else:
            a.imshow(img.crop((500, 360, 1100, 660)))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(YEL if thumb != "full" else MUTED); sp.set_linewidth(0.7)
        yc = y + (2.15 if thumb == "full" else 1.07)
        ax.text(46.0, yc + 0.9, label, fontsize=5.5, color=INK, va="center")
        ax.text(46.0, yc - 1.0, f"$\\psi$ = {psi}  →  “{verdict}”", fontsize=5.8, color=INK, va="center")
        ax.text(61.8, yc, "✓" if ok else "✗", fontsize=9, color=GOOD if ok else BAD, ha="center",
                va="center", fontweight="bold")

    rbox(ax, 37.2, 13.6, 25.6, 16.9, ec=GRID, fc="white")
    answer(25.4, "full", "full image", "0.06", "no", False)
    answer(20.8, "tile", "zoom tile 2", "1.00", "yes", True)
    ax.text(38.0, 18.2, "fused zoom (Eq. 13): $\\psi^{\\mathrm{FZ}}$ = 0.97 → “yes”", fontsize=5.6,
            color=INK, va="center")
    ax.text(61.8, 18.2, "✓", fontsize=9, color=GOOD, ha="center", va="center", fontweight="bold")
    ax.text(38.0, 15.4, "all 7 VLMs, full image: $\\psi\\leq$0.16 (missed)", fontsize=5.2, color=BAD,
            va="center")
    ax.text(50, 10.3, "one pass per view (full, mirror, 3 tiles, noisy);\nimage and prompt encoded once, KV-cached",
            ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.4)

    # ------------------------------------------------------------------ (3) test modules
    rows = [(40.5, AQUA, "T1  Perceptual horizon", "detection vs distance;\n$d_{50}$, odds ratios\n(Eqs. 4–5)"),
            (28.4, ORANGE, "T2  Metamorphic tests", "paraphrase, mirror,\nzoom; $\\Gamma_r$, $\\kappa_r$\n(Eqs. 6–8)"),
            (16.3, MAG, "T3  Fault injection", "bit flips, outcomes,\nrange guard\n(Eqs. 9–12; Alg. 2)"),
            (4.2, VIO, "T4  Safeguard", "fused zoom +\nconformal threshold\n(Eqs. 13–14; Alg. 3)")]
    for y, c, t, b in rows:
        rbox(ax, 66.3, y, 32.7, 11.0, ec=GRID, fc="white")
        ax.add_patch(Rectangle((66.3, y + 0.35), 0.8, 10.3, color=c, lw=0))
        ax.text(68.0, y + 9.6, t, fontsize=6.3, fontweight="bold", color=INK, va="top")
        ax.text(68.0, y + 6.6, b, fontsize=5.4, color=INK, va="top", linespacing=1.3)
        arrow(ax, 63.0, 22.0, 66.1, y + 5.5, c=c, lw=0.7)
    ax.add_patch(Circle((63.0, 22.0), 0.35, fc=MUTED, ec="none"))

    # insets from the study's results
    cur = pd.read_csv("results/analysis/all/curves.csv")
    order = ["(0, 15]", "(15, 25]", "(25, 35]", "(35, 50]", "(50, 200]"]
    a1 = axes_at(fig, 87.0, 41.6, 11.2, 6.3); tidy(a1)
    for st, col, lab in (("singapore_day", AQUA, "day"), ("singapore_night", YEL, "night")):
        g = cur[(cur.model == "Qwen3-VL-8B-Instruct") & (cur.stratum == st)].set_index("dbin").reindex(order)
        a1.plot(range(5), g["mean"], color=col, lw=1.0, marker="o", ms=1.5, label=lab)
    a1.axhline(0.5, color=MUTED, lw=0.4, ls=(0, (2, 2)))
    a1.set_ylim(0, 1.05); a1.set_xticks([0, 2, 4], ["10", "30", "65 m"]); a1.set_yticks([0, 0.5, 1])
    a1.legend(fontsize=4.3, frameon=False, loc="lower left", handlelength=1, borderpad=0.1, labelspacing=0.15)

    cells = pd.read_csv("results/analysis/metamorphic_all/cells.csv")
    g = cells[(cells.base == "Qwen3-VL-4B-Instruct") & (cells.mr == "zoom")]
    a2 = axes_at(fig, 87.0, 30.2, 11.2, 5.8); tidy(a2)
    a2.scatter(g.miss, g.viol, s=2.2, color=ORANGE, lw=0)
    a2.set_xlim(-0.03, 1); a2.set_xticks([0, 0.5, 1]); a2.set_yticks([0, 0.25, 0.5])
    a2.set_xlabel("reference miss", fontsize=4.5, labelpad=0.5); a2.set_ylabel("violation", fontsize=4.5, labelpad=0.5)
    a2.text(0.04, 0.96, "$\\kappa_Z$ = 0.85", transform=a2.transAxes, fontsize=4.6, va="top", color=INK)

    a3 = axes_at(fig, 87.0, 18.3, 11.2, 5.6); tidy(a3)
    a3.bar([0, 1], [3.75, 0.075], color=[MAG, GRID], width=0.55)
    a3.set_yscale("log"); a3.set_ylim(0.01, 10)
    a3.set_xticks([0, 1], ["near $\\tau$", "confident"], fontsize=4.4)
    a3.set_ylabel("flip rate (%)", fontsize=4.5, labelpad=0.5)
    a3.text(0.5, 0.97, "50×", transform=a3.transAxes, fontsize=4.8, ha="center", va="top", color=INK)

    sg = pd.read_csv("results/analysis/safeguard/curves.csv")
    a4 = axes_at(fig, 87.0, 6.0, 11.2, 5.8); tidy(a4)
    for mth, col, lab in (("vanilla", MUTED, "unaided"), ("fused", ORANGE, "fused")):
        h = sg[(sg.model == "Qwen3-VL-4B-Instruct") & (sg.set == "benchmark") & (sg.method == mth)].sort_values("miss")
        a4.plot(h.miss * 100, h.fa * 100, color=col, lw=1.0, label=lab)
    a4.set_xlim(5, 35); a4.set_ylim(0, 40); a4.set_xticks([10, 20, 30]); a4.set_yticks([0, 20, 40])
    a4.set_xlabel("miss (%)", fontsize=4.5, labelpad=0.5); a4.set_ylabel("false alarm (%)", fontsize=4.5, labelpad=0.5)
    a4.legend(fontsize=4.3, frameon=False, loc="upper right", handlelength=1, borderpad=0.1, labelspacing=0.15)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_formal.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig1_formal.png")


if __name__ == "__main__":
    main()
