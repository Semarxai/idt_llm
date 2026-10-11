"""
Figure 2 — Omega per turn at W = 1, and the directional lead across corpora.

(a) One exemplar human-LLM conversation, both streams, same y-scale.
    Exemplar selection rule, declared: the LONGEST C1 conversation below 150
    paired turns -> claude_5f70ab01, 144 turns.  No conversation was chosen on
    the strength of its result.

(b) Every conversation in C1 and C3 as one point: the fraction of its turns on
    which the acting side's excursion is the larger of the two.  0.5 is the
    coin line.  Colour carries the same meaning as in (a).

No averaging anywhere: (a) is a raw series, (b) is a per-conversation count
divided by that conversation's own number of readings.

Inputs, all under CLOSURE, already computed:
    series_c1.npz      per-turn Omega, both streams, 58 conversations
    omega_w1_c1.csv      per-conversation lead counts, C1
    sweep_c3.csv             per-conversation lead counts, C3 (W = 1 rows)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paths import CLOSURE, FIGURES   # noqa: E402
CLOSURE, FIGURES = Path(CLOSURE), Path(FIGURES)
FIGURES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- palette
# dataviz reference palette, categorical slots 1 and 2; validated light mode:
# CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1, all checks PASS.
A_COL = "#2a78d6"      # slot 1 blue   — side A / human prompt stream
B_COL = "#eb6834"      # slot 2 orange — side B / model response stream
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

EXEMPLAR = "claude_5f70ab01"
MM = 1 / 25.4

# ----------------------------------------------------------------- data
z = np.load(CLOSURE / "series_c1.npz")
oA, oB, _ = z[EXEMPLAR]                      # rows: Omega_A, Omega_B, gap
mask = np.isfinite(oA) & np.isfinite(oB)
t = np.arange(len(oA))

c1 = pd.read_csv(CLOSURE / "omega_w1_c1.csv")
c1_frac = (c1.absA_gt_absB / c1.n_compared).to_numpy()

c3 = pd.read_csv(CLOSURE / "sweep_c3.csv")
c3 = c3[c3.W == 1]
c3_frac = (c3.absA_gt_absB / c3.readings).to_numpy()

# ----------------------------------------------------------------- figure
fig = plt.figure(figsize=(183 * MM, 88 * MM))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1.0],
              height_ratios=[1, 1], wspace=0.28, hspace=0.18,
              left=0.065, right=0.985, top=0.90, bottom=0.115)

ylim = 4.2
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a1, sharey=ax_a1)

for ax, series, col, lab in ((ax_a1, oA, A_COL, "human prompt stream"),
                             (ax_a2, oB, B_COL, "model response stream")):
    ax.axhline(0, color=GRID, lw=0.7, zorder=1)
    ax.fill_between(t, 0, np.nan_to_num(series), color=col, alpha=0.22, lw=0,
                    zorder=2)
    ax.plot(t, series, color=col, lw=1.0, zorder=3, solid_capstyle="round")
    ax.set_ylim(-ylim, ylim)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlim(0, len(oA) - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.012, 0.93, lab, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, color=col, fontweight="bold")
    mx = np.nanmax(np.abs(series))
    ax.text(0.988, 0.93, f"max |Ω| = {mx:.2f}", transform=ax.transAxes,
            va="top", ha="right", fontsize=7, color=INK2)

ax_a1.tick_params(labelbottom=False)
ax_a2.set_xlabel("paired turn")
ax_a1.set_ylabel("Ω  (bits)")
ax_a2.set_ylabel("Ω  (bits)")

ax_a1.set_title("a   One conversation, every turn", loc="left", fontsize=8.5,
                fontweight="bold", pad=6, color=INK)
ax_a1.text(0.0, 1.10, "", transform=ax_a1.transAxes)
fig.text(0.065, 0.935,
         "", fontsize=7, color=INK2)

# ------------------------------------------------- panel b: lead fractions
ax_b = fig.add_subplot(gs[:, 1])
rng = np.random.default_rng(7)

rows = [("C3   human–human   2,171 games", c3_frac, 0.0,
         "1,025 above  ·  1,015 below  ·  131 tied      p = 0.84"),
        ("C1   human–LLM   58 conversations", c1_frac, 1.0,
         "57 of 58 above the coin line      p = 4 × 10⁻¹⁶")]

for lab, frac, y0, stat in rows:
    n = len(frac)
    jit = rng.uniform(-0.155, 0.155, n)
    cols = np.where(frac > 0.5, A_COL, np.where(frac < 0.5, B_COL, TIE_COL))
    size = 9 if n < 100 else 2.6
    alpha = 0.95 if n < 100 else 0.28
    ax_b.scatter(frac, y0 + jit, s=size, c=cols, alpha=alpha, linewidths=0,
                 zorder=3)
    ax_b.text(0.012, y0 + 0.40, lab, fontsize=7.5, color=INK,
              fontweight="bold", ha="left", va="bottom")
    ax_b.text(0.012, y0 + 0.255, stat, fontsize=7, color=INK2,
              ha="left", va="bottom")

ax_b.axvline(0.5, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(-0.30, 1.72)
ax_b.set_yticks([])
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xlabel("fraction of turns on which side A's |Ω| is the larger")
for s in ("top", "right", "left"):
    ax_b.spines[s].set_visible(False)
ax_b.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax_b.set_axisbelow(True)

ax_b.text(0.513, 1.63, "coin line", ha="left", va="bottom", fontsize=6.8,
          color=INK2)

ax_b.set_title("b   Every conversation, both corpora", loc="left",
               fontsize=8.5, fontweight="bold", pad=6, color=INK)

fig.savefig(FIGURES / "fig2_omega_w1.pdf")
fig.savefig(FIGURES / "fig2_omega_w1.png", dpi=400)
print("wrote fig2_omega_w1.pdf / .png")
print(f"exemplar {EXEMPLAR}  turns={len(oA)}  "
      f"max|A|={np.nanmax(np.abs(oA)):.4f}  max|B|={np.nanmax(np.abs(oB)):.4f}")
print(f"C1 above 0.5: {(c1_frac > 0.5).sum()} of {len(c1_frac)}")
print(f"C3 above {(c3_frac > 0.5).sum()}  below {(c3_frac < 0.5).sum()}  "
      f"tied {(c3_frac == 0.5).sum()}  of {len(c3_frac)}")
