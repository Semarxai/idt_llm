"""
sweep_summary.txt — the window-sweep summary, written from sweep_{c1,c2,c3}.csv.

RECONSTRUCTED.  In the original session this summary was produced by an inline
step that was never saved as a file.  This script reproduces it from the
committed CSVs; it reads them only and recomputes no Omega value.  It was
checked byte-for-byte against the committed sweep_summary.txt (see the
provenance note, docs/closure_provenance.md).

Per (corpus, W):
  convs        records with at least one reading at this W (rows in the CSV)
  measurable   records with readings >= 10 (column `measurable`)
  max|O| A/B   largest maxabs_A / maxabs_B over MEASURABLE records only
  max ratio    largest ratio_A / ratio_B over measurable records; "-" at W = 1
  exceed band  total band exceedances, both sides, measurable records; "-" at W=1
  |A|>|B|      side-A lead count of non-tied measurable records.  A record
               leads for A when absA_gt_absB > readings / 2, for B when
               < readings / 2; exactly half is a tie, EXCLUDED from the
               denominator.  p is the two-sided exact binomial test, p = 0.5.
A W with records but no measurable record prints a single "below the minimum"
line; a W with no records at all is omitted.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from closure_inputs import CLOSURE                  # noqa: E402
from omega_core import BANDS, GRID, MIN_READINGS   # noqa: E402

SEP = "=" * 100
LABEL = {"c1": "C1 human-LLM", "c2": "C2 LLM-LLM", "c3": "C3 PhotoBook"}
HEAD = ("    W  convs  measurable   max|O| A   max|O| B  max ratio A  "
        "max ratio B  exceed band  |A|>|B| convs")


def block(tag, s):
    out = [SEP, f"{LABEL[tag]}   {s.conv.nunique()} conversations", SEP, HEAD]
    for W in GRID:
        g = s[s.W == W]
        if not len(g):
            continue
        m = g[g.measurable]
        if not len(m):
            out.append(f"{W:5d}{len(g):7d}{0:12d}   -- all cells below the "
                       f"{MIN_READINGS}-reading minimum --")
            continue
        a = int((m.absA_gt_absB > m.readings / 2).sum())
        b = int((m.absA_gt_absB < m.readings / 2).sum())
        p = binomtest(a, a + b, 0.5).pvalue
        if W == 1:
            rA = rB = ex = "-"
            row_r = f"{rA:>13s}{rB:>13s}{ex:>13s}"
        else:
            ex = int(m.exceed_band_A.sum() + m.exceed_band_B.sum())
            row_r = f"{m.ratio_A.max():13.4f}{m.ratio_B.max():13.4f}{ex:13d}"
        lead = f"{a} of {a + b} p={p:.2g}"
        out.append(f"{W:5d}{len(g):7d}{len(m):12d}"
                   f"{m.maxabs_A.max():11.4f}{m.maxabs_B.max():11.4f}"
                   f"{row_r} {lead:>14s}")
    return out


def main(src=None, out=None):
    src = Path(src or CLOSURE)
    out = Path(out or CLOSURE) / "sweep_summary.txt"
    bands = "   ".join(f"W={W} {BANDS[W]:.4f}" for W in GRID if W >= 2)
    lines = [SEP,
             "OMEGA WINDOW SWEEP — do excursions contract RELATIVE to the derived bound,",
             "and does the side asymmetry persist across scale?",
             SEP, "",
             f"bands:  {bands}",
             "        W=1 has no band (the closure premise needs W>=2)",
             "null expectation: excursion and band both scale ~1/W, so the RATIO "
             "should be flat.",
             ""]
    for tag in ("c1", "c2", "c3"):
        lines += block(tag, pd.read_csv(src / f"sweep_{tag}.csv"))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
