"""
Control substitutions under an alternative symbolization  (Reviewer 2 pt 4, Reviewer 4 pt 7)

Repeats the three control substitutions reported in the paper (A, B, C) with one
declared choice changed: the embedding model is mpnet instead of MiniLM.
Everything else is identical -- same three records, same declared symbol count,
same 100 replicates, same seeds.

Put this file in the SAME folder as run_sensitivity.py and Run it.
It reuses cache/mpnet/*.npz, so no re-embedding happens.

Writes, next to this script:
   controls_mpnet.csv          one row per record per side per cell
   controls_mpnet_summary.csv  the reduction percentages   <-- send me this
"""
import os, hashlib, time, numpy as np, pandas as pd
from sklearn.cluster import KMeans

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
TURNS = os.environ.get("SCC_TURNS", os.path.join(ROOT, "data", "c1", "source", "turns_all_ts.csv"))
CACHE = os.environ.get("SCC_CACHE", os.path.join(HERE, "cache", "mpnet"))

NREP, NPERM, COS_MAX, BLOCK, FLOOR = 100, 500, 0.50, 0.15, 25


def mm_entropy(counts):
    c = counts[counts > 0]; N = c.sum()
    if N <= 0: return 0.0
    p = c / N
    return float(-(p * np.log2(p)).sum() + (len(c) - 1) / (2 * N * np.log(2)))


def eps_from(S, A, Sp, k):
    hS   = mm_entropy(np.bincount(S, minlength=k))
    hSS  = mm_entropy(np.bincount(S * k + Sp, minlength=k * k))
    hSA  = mm_entropy(np.bincount(S * k + A,  minlength=k * k))
    hSAS = mm_entropy(np.bincount((S * k + A) * k + Sp, minlength=k ** 3))
    room, resid = hSS - hS, hSAS - hSA
    return ((room - resid) / room if room > 0 else np.nan), room


def seed_of(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


def kfor(n):
    return 2 if n < 150 else (3 if n < 400 else 4)


def q95(S, A, Sp, k, rng):
    v = [eps_from(S, np.random.default_rng(rng.integers(1 << 30)).permutation(A), Sp, k)[0]
         for _ in range(NPERM)]
    v = np.array(v, float); v = v[np.isfinite(v)]
    return float(np.quantile(v, 0.95))


df = pd.read_csv(TURNS)
g = df.groupby("conv_id").size()
TOP = list(g[g >= FLOOR].sort_values(ascending=False).index[:3])
print("records:", TOP, flush=True)

rows = []
for cid in TOP:
    d = df[df.conv_id == cid].sort_values("Turn")
    P = d["Prompt"].fillna("").astype(str).tolist()
    R = d["Response"].fillna("").astype(str).tolist()
    n = len(d); k = kfor(n)
    z = np.load(os.path.join(CACHE, f"{cid}.npz"))
    Ep, Er = z["p"], z["r"]
    cp = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Ep)
    cr = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Er)
    Lp = np.array([len(x) for x in P]); Lr = np.array([len(x) for x in R])

    for side in ("model", "user"):
        if side == "model":
            S, A, Sp = cp[:-1], cr[:-1], cp[1:]; EA, LA = Er[:-1], Lr[:-1]
        else:
            S, A, Sp = cr[:-1], cp[1:], cr[1:];  EA, LA = Ep[1:],  Lp[1:]
        m = len(S); blk = max(3, int(BLOCK * m))
        base = eps_from(S, A, Sp, k)[0]
        ref  = q95(S, A, Sp, k, np.random.default_rng(seed_of(cid, side, "ref")))
        rows.append(dict(record=cid, side=side, k=k, N=m, cell="C0_observed",
                         eps_median=base, clears=bool(base > ref), ref=ref,
                         rep_clear_rate=np.nan, reduction_pct=0.0))
        cos = EA @ EA.T
        for cell in ("C1_compat_nonadj", "C2_lengthA", "C3_dissimB"):
            rng = np.random.default_rng(seed_of(cid, side, cell))
            vals = []; clears = 0
            for _ in range(NREP):
                A2 = A.copy()
                for t in range(m):
                    if cell == "C1_compat_nonadj":
                        cand = np.where((S == S[t]) & (np.abs(np.arange(m) - t) >= 2)
                                        & (np.abs(np.arange(m) - t) <= blk))[0]
                        if len(cand) == 0:
                            cand = np.where((S == S[t]) & (np.abs(np.arange(m) - t) >= 2))[0]
                    elif cell == "C2_lengthA":
                        ok  = np.abs(np.arange(m) - t) >= 2
                        rel = np.abs(LA - LA[t]) / np.maximum(LA[t], 1)
                        cand = np.where(ok & (rel <= 0.05))[0]
                        if len(cand) == 0:
                            cand = np.where(ok)[0][np.argsort(rel[ok])[:10]]
                    else:
                        ok = np.abs(np.arange(m) - t) >= 2
                        for band in (0.10, 0.20, 0.50, 1.00, 1e9):
                            rel = np.abs(LA - LA[t]) / np.maximum(LA[t], 1)
                            cand = np.where(ok & (rel <= band) & (cos[t] < COS_MAX))[0]
                            if len(cand) > 0: break
                        if len(cand) == 0:
                            cand = np.where(ok)[0][np.argsort(cos[t][ok])[:10]]
                    if len(cand): A2[t] = A[rng.choice(cand)]
                e = eps_from(S, A2, Sp, k)[0]
                vals.append(e); clears += int(np.isfinite(e) and e > ref)
            v = np.array(vals, float); v = v[np.isfinite(v)]
            med = float(np.median(v))
            red = 100.0 * (1.0 - med / base) if base and np.isfinite(base) else np.nan
            rows.append(dict(record=cid, side=side, k=k, N=m, cell=cell,
                             eps_median=med, clears=np.nan, ref=ref,
                             rep_clear_rate=clears / NREP, reduction_pct=red))
            print(f"  {str(cid)[:16]} {side:5} {cell:18} med={med:.4f} obs={base:.4f} "
                  f"reduction={red:5.1f}%  clears={clears}/100", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "controls_mpnet.csv"), index=False)

s = (out[out.cell != "C0_observed"]
     .groupby("cell")["reduction_pct"]
     .agg(mean_reduction_pct="mean", min_reduction_pct="min", max_reduction_pct="max")
     .reset_index())
s["label"] = s.cell.map({"C2_lengthA": "Control A (length-matched)",
                         "C3_dissimB": "Control B (length + cosine ceiling)",
                         "C1_compat_nonadj": "Control C (same symbol, local block)"})
s.to_csv(os.path.join(HERE, "controls_mpnet_summary.csv"), index=False)

print("\nDONE -- send me controls_mpnet_summary.csv")
print(s.to_string(index=False))
