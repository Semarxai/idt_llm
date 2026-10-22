"""
C2 control substitutions under the alternative symbolization  (Reviewer 2 pt 4)

Companion to run_controls_sensitivity.py, which covered the six C1 cells.
This covers the four C2 cells (pooled, k=3 and k=2, both sides), so the
comparison with the published medians runs over all ten cells.

Only one declared choice changes: mpnet embeddings in place of MiniLM.
C2 is eight runs of 25 turns, so embedding takes under a minute.

Run it. Writes, next to this script:
   controls_c2_mpnet.csv
   controls_ten_cell_summary.csv     <-- send me this
"""
import os, hashlib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
C2CSV = os.environ.get("SCC_C2", os.path.join(ROOT, "data", "c2", "omega_advancement_test_v1_turns.csv"))
MPNET = os.environ.get("SCC_MPNET", "sentence-transformers/all-mpnet-base-v2")

NREP, NPERM, COS_MAX, BLOCK = 100, 500, 0.50, 0.15

_tok = AutoTokenizer.from_pretrained(MPNET)
_mod = AutoModel.from_pretrained(MPNET).eval()


def embed(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = _tok([str(t) for t in texts[i:i+bs]], padding=True, truncation=True,
                 max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = _mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=1).numpy())
    return np.vstack(out)


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


d = pd.read_csv(C2CSV)
d = d[d.turn >= 2].reset_index(drop=True)
runs = sorted(d.run_id.unique())
P = d.prompt.fillna("").astype(str).tolist()
R = d.response.fillna("").astype(str).tolist()
print(f"{len(runs)} runs, {len(d)} turns -- embedding with mpnet", flush=True)
Ep, Er = embed(P), embed(R)
Lp = np.array([len(x) for x in P]); Lr = np.array([len(x) for x in R])

rows = []
for k in (3, 2):
    cp = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Ep)
    cr = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Er)
    for side in ("model", "user"):
        S = []; A = []; Sp = []; EAi = []; LAi = []; RUN = []
        for r in runs:
            idx = d.index[d.run_id == r].tolist()
            a, b = cp[idx], cr[idx]
            if side == "model":
                S += list(a[:-1]); A += list(b[:-1]); Sp += list(a[1:])
                EAi += idx[:-1];   LAi += list(Lr[idx[:-1]])
            else:
                S += list(b[:-1]); A += list(a[1:]);  Sp += list(b[1:])
                EAi += idx[1:];    LAi += list(Lp[idx[1:]])
            RUN += [r] * (len(idx) - 1)
        S = np.array(S); A = np.array(A); Sp = np.array(Sp)
        RUN = np.array(RUN); LA = np.array(LAi)
        EA = (Er if side == "model" else Ep)[np.array(EAi)]
        m = len(S); pos = np.arange(m); blk = max(3, int(BLOCK * m / len(runs)))

        base = eps_from(S, A, Sp, k)[0]
        rng0 = np.random.default_rng(seed_of("C2", k, side, "ref"))
        ref = float(np.quantile([eps_from(S, rng0.permutation(A), Sp, k)[0]
                                 for _ in range(NPERM)], 0.95))
        rows.append(dict(record=f"C2_pooled_k{k}", side=side, k=k, N=m,
                         cell="C0_observed", eps_median=base, ref=ref,
                         rep_clear_rate=np.nan, reduction_pct=0.0))
        cos = EA @ EA.T
        for cell in ("C1_compat_nonadj", "C2_lengthA", "C3_dissimB"):
            rng = np.random.default_rng(seed_of("C2", k, side, cell))
            vals = []; clears = 0
            for _ in range(NREP):
                A2 = A.copy()
                for t in range(m):
                    same = (RUN == RUN[t]) & (np.abs(pos - t) >= 2)
                    if cell == "C1_compat_nonadj":
                        cand = np.where(same & (S == S[t]) & (np.abs(pos - t) <= blk))[0]
                        if len(cand) == 0:
                            cand = np.where(same & (S == S[t]))[0]
                    elif cell == "C2_lengthA":
                        rel = np.abs(LA - LA[t]) / np.maximum(LA[t], 1)
                        cand = np.where(same & (rel <= 0.05))[0]
                        if len(cand) == 0:
                            f = np.where(same)[0]
                            cand = f[np.argsort(rel[f])[:10]] if len(f) else np.array([], int)
                    else:
                        cand = np.array([], int)
                        for band in (0.10, 0.20, 0.50, 1.00, 1e9):
                            rel = np.abs(LA - LA[t]) / np.maximum(LA[t], 1)
                            cand = np.where(same & (rel <= band) & (cos[t] < COS_MAX))[0]
                            if len(cand) > 0: break
                        if len(cand) == 0:
                            f = np.where(same)[0]
                            cand = f[np.argsort(cos[t][f])[:10]] if len(f) else np.array([], int)
                    if len(cand): A2[t] = A[rng.choice(cand)]
                e = eps_from(S, A2, Sp, k)[0]
                vals.append(e); clears += int(np.isfinite(e) and e > ref)
            v = np.array(vals, float); v = v[np.isfinite(v)]
            med = float(np.median(v)); red = 100.0 * (1.0 - med / base)
            rows.append(dict(record=f"C2_pooled_k{k}", side=side, k=k, N=m, cell=cell,
                             eps_median=med, ref=ref, rep_clear_rate=clears / NREP,
                             reduction_pct=red))
            print(f"  k={k} {side:5} {cell:18} med={med:.4f} obs={base:.4f} "
                  f"reduction={red:5.1f}%  clears={clears}/100", flush=True)

c2 = pd.DataFrame(rows)
c2.to_csv(os.path.join(HERE, "controls_c2_mpnet.csv"), index=False)

c1path = os.path.join(HERE, "controls_mpnet.csv")
label = {"C2_lengthA": "Control A (length-matched)",
         "C3_dissimB": "Control B (length + cosine ceiling)",
         "C1_compat_nonadj": "Control C (same symbol, local block)"}
if os.path.exists(c1path):
    allc = pd.concat([pd.read_csv(c1path), c2])
    s = (allc[allc.cell != "C0_observed"].groupby("cell").reduction_pct
         .agg(median_reduction_pct="median", n_cells="count").reset_index())
    s["label"] = s.cell.map(label)
    s.to_csv(os.path.join(HERE, "controls_ten_cell_summary.csv"), index=False)
    print("\nDONE -- all ten cells, mpnet")
    print(s.to_string(index=False))
else:
    print("\nDONE -- C2 only; controls_mpnet.csv not found for the ten-cell summary")
