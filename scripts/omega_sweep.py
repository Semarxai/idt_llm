"""
Omega across a window sweep, both sides, on C1, C2 and C3.

QUESTION (frozen before running):
  As W increases, do observed closure excursions contract RELATIVE to the
  derived finite-window bound, and does the side asymmetry persist across scale?

  Null expectation: the excursion on intact traffic is a single message swap
  diluted by 1/W, and the band is also ~1/W, so max|Omega|/band should be
  roughly FLAT in W.  Contraction or expansion is the finding.

Estimator: equal message weighting, rolling window — see src/omega_core.py.
Empty messages are DROPPED before windowing (unlike scripts/omega_w1.py).
Grid W = 1, 2, 5, 10, 20, 50.  Maxima and counts only.

Measurable: a (record, W) cell is measurable when it has at least
MIN_READINGS = 10 Omega values on BOTH sides (readings = the common length of
the two series).  Unmeasurable cells are written with measurable=False and
excluded from every summary; a record shorter than W + 1 messages produces no
row at that W at all.

Writes CLOSURE/sweep_{c1,c2,c3}.csv, one row per (record, W).
Original session name: omega_sweep.py.  Run as
    python scripts/omega_sweep.py c1|c2|c3
"""
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from closure_inputs import C1_CSV, C2_CSV, C3_LOGS, CLOSURE, TOKENIZER  # noqa: E402
from omega_core import (BANDS, GRID, K, MIN_READINGS, message_dists,     # noqa: E402
                        omega_series, tokenizer)
from photobook_pairs import parse_game                                   # noqa: E402

OUT = Path(CLOSURE)
OUT.mkdir(parents=True, exist_ok=True)
tok = tokenizer(TOKENIZER)


def sweep_conversation(corpus, name, texts_A, texts_B, n_turns, extra=None):
    qA, qB = message_dists(tok, texts_A), message_dists(tok, texts_B)
    rows = []
    for W in GRID:
        oA, oB = omega_series(qA, W), omega_series(qB, W)
        m = min(len(oA), len(oB))
        if m == 0:
            continue
        gap = oA[:m] - oB[:m]
        b = BANDS[W]
        rec = dict(corpus=corpus, conv=name, n_turns=n_turns, W=W,
                   readings=m, band=(None if np.isnan(b) else round(b, 4)),
                   measurable=bool(m >= MIN_READINGS),
                   maxabs_A=round(float(np.max(np.abs(oA[:m]))), 5),
                   maxabs_B=round(float(np.max(np.abs(oB[:m]))), 5),
                   maxabs_gap=round(float(np.max(np.abs(gap))), 5),
                   omega_A_min=round(float(oA[:m].min()), 5),
                   omega_A_max=round(float(oA[:m].max()), 5),
                   omega_B_min=round(float(oB[:m].min()), 5),
                   omega_B_max=round(float(oB[:m].max()), 5),
                   absA_gt_absB=int((np.abs(oA[:m]) > np.abs(oB[:m])).sum()),
                   exceed_band_A=(None if np.isnan(b) else int((np.abs(oA[:m]) > b).sum())),
                   exceed_band_B=(None if np.isnan(b) else int((np.abs(oB[:m]) > b).sum())))
        if not np.isnan(b):
            rec["ratio_A"] = round(rec["maxabs_A"] / b, 5)
            rec["ratio_B"] = round(rec["maxabs_B"] / b, 5)
        else:
            rec["ratio_A"] = rec["ratio_B"] = None
        if extra:
            rec.update(extra)
        rows.append(rec)
    return rows


def run_c1():
    df = pd.read_csv(C1_CSV)
    df["Prompt"] = df["Prompt"].fillna("").astype(str)
    df["Response"] = df["Response"].fillna("").astype(str)
    sz = df.groupby("conv_id").size()
    rows = []
    for i, cid in enumerate(sorted(sz[sz >= 25].index, key=lambda c: -sz[c]), 1):
        d = df[df.conv_id == cid].sort_values("Turn")
        rows += sweep_conversation("C1", cid, d["Prompt"].tolist(),
                                   d["Response"].tolist(), len(d),
                                   extra=dict(platform=d["platform"].iloc[0]))
        print(f"  C1 [{i}] {cid} n={len(d)}", flush=True)
    return rows


def run_c2():
    a = pd.read_csv(C2_CSV)
    a["prompt"] = a["prompt"].fillna("").astype(str)
    a["response"] = a["response"].fillna("").astype(str)
    rows = []
    for rid in sorted(a.run_id.unique()):
        d = a[a.run_id == rid].sort_values("turn")
        rows += sweep_conversation("C2", rid.split("_")[1], d["prompt"].tolist(),
                                   d["response"].tolist(), len(d))
    return rows


def run_c3(limit=None):
    fs = sorted(glob.glob(str(Path(C3_LOGS) / "*.json")))
    if limit:
        fs = sorted(fs, key=lambda f: hashlib.md5(
            os.path.basename(f).encode()).hexdigest())[:limit]
    rows = []
    for i, f in enumerate(fs, 1):
        p = parse_game(f)
        if p is None:
            continue
        A, B, score, nr, flag4 = p
        if flag4:
            continue
        n = min(len(A), len(B))
        if n < 25:
            continue
        rows += sweep_conversation("C3", os.path.basename(f)[:-5], A[:n], B[:n], n,
                                   extra=dict(score=score))
        if i % 200 == 0:
            print(f"  C3 {i}/{len(fs)}", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"tokenizer {TOKENIZER}  K={K}  grid {GRID}  min readings {MIN_READINGS}")
    rows = {"c1": run_c1, "c2": run_c2}.get(which, lambda: run_c3(lim))()
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"sweep_{which}.csv", index=False)
    print(f"wrote sweep_{which}.csv  ({len(t)} rows)")
