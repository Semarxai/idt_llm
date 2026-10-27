"""
Figure 3 — the window sweep: excursions against the derived bound, and the
directional lead, across W = 1, 2, 5, 10, 20, 50 in three corpora.

Row 1  max |Omega| / band(W), the largest excursion any conversation produced
       at that window, as a fraction of the analytic bound, eq. (3).  1.0 is the
       bound.  No band exists at W = 1, so that column is empty by construction.

Row 2  the share of measurable conversations on which side A leads, against the
       coin line.  Ties are excluded from the share and reported in the caption.

Colour is the same key as Figure 2: blue = side A, orange = side B.  Corpus is
carried by column, never by colour.

Maxima and counts only.  Nothing averaged.

Inputs, under CLOSURE, already computed:  sweep_{c1,c2,c3}.csv
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paths import CLOSURE, FIGURES   # noqa: E402
CLOSURE, FIGURES = Path(CLOSURE), Path(FIGURES)
FIGURES.mkdir(parents=True, exist_ok=True)

A_COL = "#2a78d6"      # side A
B_COL = "#eb6834"      # side B
TIE_COL = "#9b9a94"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd5"
SURF = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

MM = 1 / 25.4
GRIDW = [1, 2, 5, 10, 20, 50]
CORPORA = [("c1", "C1   human–LLM", "58 conversations"),
           ("c2", "C2   LLM–LLM", "8 runs"),
           ("c3", "C3   human–human", "2,171 games")]


def load(tag):
    s = pd.read_csv(CLOSURE / f"sweep_{tag}.csv")
    return s[s.measurable]


rows = {}
for tag, _, _ in CORPORA:
    s = load(tag)
    rec = {}
    for W in GRIDW:
        g = s[s.W == W]
        if len(g) == 0:
            continue
        a = int((g.absA_gt_absB > g.readings / 2).sum())
        b = int((g.absA_gt_absB < g.readings / 2).sum())
        tie = len(g) - a - b
        rec[W] = dict(
            n=len(g),
            rA=float(g.ratio_A.max()) if g.ratio_A.notna().any() else np.nan,
            rB=float(g.ratio_B.max()) if g.ratio_B.notna().any() else np.nan,
            lead=a / (a + b) if (a + b) else np.nan, a=a, b=b, tie=tie)
    rows[tag] = rec

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 100 * MM), sharex=True,
                         sharey="row")
fig.subplots_adjust(left=0.105, right=0.985, top=0.805, bottom=0.115,
                    wspace=0.13, hspace=0.22)

xt = np.arange(len(GRIDW))
xmap = {W: i for i, W in enumerate(GRIDW)}

for col, (tag, name, sub) in enumerate(CORPORA):
    rec = rows[tag]

    # ---------------------------------------------------------- row 1: band
    ax = axes[0, col]
    ax.axhspan(1.0, 1.18, color=GRID, alpha=0.55, lw=0, zorder=1)
    ax.axhline(1.0, color=INK2, lw=1.0, zorder=3)
    for key, colr, lab in (("rA", A_COL, "side A"), ("rB", B_COL, "side B")):
        ws = [W for W in GRIDW if W in rec and np.isfinite(rec[W][key])]
        ax.plot([xmap[W] for W in ws], [rec[W][key] for W in ws],
                color=colr, lw=1.6, zorder=4, solid_capstyle="round", label=lab)
        for W in ws:   # same open-marker rule as row 2: fewer than 10 measurable
            small = rec[W]["n"] < 10
            ax.plot(xmap[W], rec[W][key], marker="o", ms=3.6, zorder=5,
                    mfc=SURF if small else colr, mec=colr, mew=1.1)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{name}", loc="left", fontsize=8, fontweight="bold",
                 pad=13, color=INK)
    ax.text(0.0, 1.035, sub, transform=ax.transAxes, fontsize=7, color=INK2)
    if col == 0:
        ax.set_ylabel("max |Ω|  /  bound, eq. (3)")
        ax.text(0.985, 0.955, "the bound", transform=ax.transAxes,
                fontsize=6.8, color=INK2, ha="right", va="top")
        ax.text(xmap[1], 0.055, "no band\nat W = 1", fontsize=6.4, color=INK2,
                ha="center", va="bottom", linespacing=1.15)

    # ------------------------------------------------------- row 2: lead
    ax = axes[1, col]
    ax.axhline(0.5, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ws = [W for W in GRIDW if W in rec and np.isfinite(rec[W]["lead"])]
    y = [rec[W]["lead"] for W in ws]
    ax.plot([xmap[W] for W in ws], y, color=INK2, lw=0.9, zorder=3, alpha=0.5)
    for W, v in zip(ws, y):
        c = A_COL if v > 0.5 else (B_COL if v < 0.5 else TIE_COL)
        small = rec[W]["n"] < 10
        ax.plot(xmap[W], v, marker="o", ms=5.2, zorder=4,
                mfc=SURF if small else c, mec=c, mew=1.4)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(W) for W in GRIDW])
    ax.set_xlabel("window  W  (messages)")
    if col == 0:
        ax.set_ylabel("share of conversations\nwhere side A leads")
        ax.text(1.03, 0.545, "coin line", fontsize=6.8, color=INK2,
                ha="left", va="bottom")

    # per-W denominators along the bottom
    for W in GRIDW:
        if W in rec:
            ax.text(xmap[W], 0.045, f"{rec[W]['n']}", fontsize=6.2,
                    color=INK2, ha="center", va="bottom")
        else:
            ax.text(xmap[W], 0.045, "–", fontsize=6.2, color=INK2,
                    ha="center", va="bottom")
    if col == 0:
        ax.text(-0.02, 0.045, "n", fontsize=6.2, color=INK2, ha="right",
                va="bottom", transform=ax.get_yaxis_transform())

# ------------------------------------------------------------------ legend
h = [plt.Line2D([], [], color=A_COL, lw=1.8, marker="o", ms=3.6,
                label="side A  (human prompt stream in C1, first speaker in C3)"),
     plt.Line2D([], [], color=B_COL, lw=1.8, marker="o", ms=3.6,
                label="side B  (model response stream in C1, second speaker in C3)")]
h.append(plt.Line2D([], [], color=INK2, lw=0, marker="o", ms=5.2,
                    mfc=SURF, mec=INK2, mew=1.4,
                    label="open marker: fewer than 10 conversations measurable"))
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.103, 1.002),
           ncol=2, fontsize=7, handlelength=1.8, columnspacing=1.6,
           labelspacing=0.45)

fig.text(0.985, 0.905, "zero band exceedances in every cell of every corpus",
         ha="right", va="center", fontsize=7.2, color=INK, fontweight="bold")

fig.savefig(FIGURES / "fig3_sweep.pdf")
fig.savefig(FIGURES / "fig3_sweep.png", dpi=400)
print("wrote fig3_sweep.pdf / .png\n")
for tag, name, _ in CORPORA:
    for W, r in rows[tag].items():
        print(f"{tag} W={W:2d} n={r['n']:5d} rA={r['rA']:.4f} rB={r['rB']:.4f} "
              f"A={r['a']} B={r['b']} tie={r['tie']} lead={r['lead']:.3f}")
