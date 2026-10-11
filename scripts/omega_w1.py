"""
Omega at W = 1, per turn, both sides, on all three corpora.

    Omega(t) = H(q_t) - H(q_{t-1})            bits
    gap(t)   = Omega_sideA(t) - Omega_sideB(t)

Sides
    C1  side A = human prompt stream (Prompt),  side B = model response (Response)
    C2  side A = asker stream (prompt),         side B = answerer (response)
    C3  side A = first speaker in the game,     side B = second speaker
    The two streams are index-aligned by turn: message t of A and message t of
    B belong to the same paired turn.

Records
    C1  every conv_id with >= 25 turns (58), rows in descending order of length
    C2  every run_id (8 runs, 25 turns each)
    C3  every eligible PhotoBook game (2,171) — see src/photobook_pairs.py

Empty messages keep their slot and give NaN for that turn and the next.

Writes, under CLOSURE:
    omega_w1_c1.csv, omega_w1_c2.csv, omega_w1_c3.csv   one row per record
    series_c1.npz, series_c2.npz                         per-turn series, keyed
        by record id; each value is a 3 x n array: [Omega_A, Omega_B, gap]
        (no series file for C3)

Original session name: omega_w1_all.py.  Run as
    python scripts/omega_w1.py c1|c2|c3|all
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
from omega_core import K, omega_stream_w1, tokenizer                   # noqa: E402
from photobook_pairs import parse_game                                 # noqa: E402

OUT = Path(CLOSURE)
OUT.mkdir(parents=True, exist_ok=True)
tok = tokenizer(TOKENIZER)


def summarise(name, corpus, n_turns, texts_A, texts_B, extra=None):
    omA, hA, tokA = omega_stream_w1(tok, texts_A)
    omB, hB, tokB = omega_stream_w1(tok, texts_B)
    n = min(len(omA), len(omB))
    gap = omA[:n] - omB[:n]
    mA, mB, mg = np.isfinite(omA), np.isfinite(omB), np.isfinite(gap)
    rec = dict(corpus=corpus, conv=name, n_turns=n_turns,
               readings_A=int(mA.sum()), readings_B=int(mB.sum()),
               H_A_min=round(np.nanmin(hA), 4), H_A_max=round(np.nanmax(hA), 4),
               H_B_min=round(np.nanmin(hB), 4), H_B_max=round(np.nanmax(hB), 4),
               omega_A_min=round(np.nanmin(omA), 4), omega_A_max=round(np.nanmax(omA), 4),
               omega_B_min=round(np.nanmin(omB), 4), omega_B_max=round(np.nanmax(omB), 4),
               gap_min=round(np.nanmin(gap), 4), gap_max=round(np.nanmax(gap), 4),
               gap_positive=int((gap[mg] > 0).sum()), gap_negative=int((gap[mg] < 0).sum()),
               absA_gt_absB=int((np.abs(omA[:n][mg]) > np.abs(omB[:n][mg])).sum()),
               n_compared=int(mg.sum()),
               empty_A=int((tokA == 0).sum()), empty_B=int((tokB == 0).sum()))
    if extra:
        rec.update(extra)
    return rec, omA, omB, gap


def run_c1():
    df = pd.read_csv(C1_CSV)
    df["Prompt"] = df["Prompt"].fillna("").astype(str)
    df["Response"] = df["Response"].fillna("").astype(str)
    sz = df.groupby("conv_id").size()
    rows, series = [], {}
    for cid in sorted(sz[sz >= 25].index, key=lambda c: -sz[c]):
        d = df[df.conv_id == cid].sort_values("Turn")
        rec, oA, oB, g = summarise(cid, "C1", len(d),
                                   d["Prompt"].tolist(), d["Response"].tolist(),
                                   extra=dict(platform=d["platform"].iloc[0]))
        rows.append(rec)
        series[cid] = np.vstack([oA[:len(g)], oB[:len(g)], g])
        print(f"  C1 {cid:20s} n={len(d):5d}", flush=True)
    np.savez_compressed(OUT / "series_c1.npz", **series)
    return rows


def run_c2():
    a = pd.read_csv(C2_CSV)
    a["prompt"] = a["prompt"].fillna("").astype(str)
    a["response"] = a["response"].fillna("").astype(str)
    rows, series = [], {}
    for rid in sorted(a.run_id.unique()):
        d = a[a.run_id == rid].sort_values("turn")
        tag = rid.split("_")[1]                      # e.g. "s11"
        rec, oA, oB, g = summarise(tag, "C2", len(d),
                                   d["prompt"].tolist(), d["response"].tolist())
        rows.append(rec)
        series[tag] = np.vstack([oA[:len(g)], oB[:len(g)], g])
        print(f"  C2 {tag} n={len(d)}", flush=True)
    np.savez_compressed(OUT / "series_c2.npz", **series)
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
        rec, _, _, _ = summarise(os.path.basename(f)[:-5], "C3", n, A[:n], B[:n],
                                 extra=dict(score=score))
        rows.append(rec)
        if i % 200 == 0:
            print(f"  C3 {i}/{len(fs)}", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"tokenizer {TOKENIZER}  K = {K}  W = 1  (no band at W = 1)\n")
    todo = ["c1", "c2", "c3"] if which == "all" else [which]
    for w in todo:
        rows = {"c1": run_c1, "c2": run_c2}.get(w, lambda: run_c3(lim))()
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"omega_w1_{w}.csv", index=False)
        print(f"wrote omega_w1_{w}.csv  ({len(t)} records)")
