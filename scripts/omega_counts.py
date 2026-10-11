"""
omega_counts.txt — the W = 1 summary, written from the three omega_w1_*.csv files.

RECONSTRUCTED.  In the original session this summary was produced by an inline
step that was never saved as a file.  This script reproduces it from the
committed CSVs; it reads them only and recomputes no Omega value.  It was
checked byte-for-byte against the committed omega_counts.txt (see the
provenance note, docs/closure_provenance.md).

Counting rules
  gap line    a record counts as "positive" when gap_positive > gap_negative,
              "negative" when fewer; equal counts are ties and are EXCLUDED
              from the denominator.
  |A|>|B|     a record counts for side A when absA_gt_absB > n_compared / 2,
              for side B when < n_compared / 2; exactly half is a tie and is
              EXCLUDED from the denominator.
  sign p      two-sided exact binomial test, p = 0.5, on the non-tied records.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from closure_inputs import CLOSURE   # noqa: E402

SEP = "=" * 92
LABEL = {"c1": "C1 human-LLM", "c2": "C2 LLM-LLM", "c3": "C3 PhotoBook"}


def block(tag, d):
    gp = int((d.gap_positive > d.gap_negative).sum())
    gn = int((d.gap_positive < d.gap_negative).sum())
    la = int((d.absA_gt_absB > d.n_compared / 2).sum())
    lb = int((d.absA_gt_absB < d.n_compared / 2).sum())
    p_gap = binomtest(gp, gp + gn, 0.5).pvalue
    p_lead = binomtest(la, la + lb, 0.5).pvalue
    lab = "  {:<26s}   ".format
    return [
        f"{LABEL[tag]}   {len(d)} conversations   "
        f"turns {d.n_turns.min()}–{d.n_turns.max()}",
        lab("per-message entropy H(q_t)")
        + f"side A {d.H_A_min.min():.3f} to {d.H_A_max.max():.3f} bits   "
          f"side B {d.H_B_min.min():.3f} to {d.H_B_max.max():.3f}",
        lab("Omega, single turn")
        + f"side A {d.omega_A_min.min():+.3f} to {d.omega_A_max.max():+.3f}   "
          f"side B {d.omega_B_min.min():+.3f} to {d.omega_B_max.max():+.3f}",
        lab("gap = Omega_A - Omega_B")
        + f"{d.gap_min.min():+.3f} to {d.gap_max.max():+.3f}",
        f"  conversations where gap is positive on more turns than negative: "
        f"{gp} of {gp + gn}   sign p = {p_gap:.3g}",
        f"  conversations where |Omega_A| exceeds |Omega_B| on more turns:    "
        f"{la} of {la + lb}   sign p = {p_lead:.3g}",
        f"  turns compared, total: {int(d.n_compared.sum()):,}",
        f"  empty messages: side A {int(d.empty_A.sum())}   side B {int(d.empty_B.sum())}",
        "",
    ]


def main(src=None, out=None):
    src = Path(src or CLOSURE)
    out = Path(out or CLOSURE) / "omega_counts.txt"
    lines = [SEP,
             "OMEGA at W = 1, per turn, both sides — Llama-2 tokens, K=32,000, "
             "no alphabet, no window",
             SEP, ""]
    for tag in ("c1", "c2", "c3"):
        lines += block(tag, pd.read_csv(src / f"omega_w1_{tag}.csv"))
    lines.append("For scale: the D1 band at W=50 is 0.4408 bits.  "
                 "At W=1 it is 14.97 bits, so no band applies.")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
