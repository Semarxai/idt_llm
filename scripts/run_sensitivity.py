"""
Symbolization sensitivity analysis  (Reviewer 2 pt 4, Reviewer 4 pts 3 and 7)

Edit the four paths below, then Run.
Needs: torch, transformers, scikit-learn, pandas, numpy, scipy
   pip install torch transformers scikit-learn pandas numpy scipy

Writes, next to this script:
   cache/<model>/<conv>.npz      embeddings, so re-runs are instant
   grid_<config>.csv             one row per record per side
   grid_summary.csv              one row per configuration   <-- send me this
"""
import os, sys, time, hashlib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from transformers import AutoTokenizer, AutoModel
import torch

# ---------------------------------------------------------------- PATHS
# Resolved from the repository root; override with environment variables.
#   SCC_TURNS   path to the C1 source (not released - personal correspondence)
#   SCC_MINILM  local folder holding sentence-transformers/all-MiniLM-L6-v2
#   SCC_MPNET   local folder holding sentence-transformers/all-mpnet-base-v2
# The model folders need: config.json, model.safetensors, tokenizer.json,
# tokenizer_config.json, vocab.txt
HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
TURNS  = os.environ.get("SCC_TURNS",  os.path.join(ROOT, "data", "c1", "source", "turns_all_ts.csv"))
MINILM = os.environ.get("SCC_MINILM", "sentence-transformers/all-MiniLM-L6-v2")
MPNET  = os.environ.get("SCC_MPNET",  "sentence-transformers/all-mpnet-base-v2")
OUT    = os.path.join(ROOT, "results", "sensitivity")
os.makedirs(OUT, exist_ok=True)
# ------------------------------------------------------------------------

NPERM, MINN, COVER, FLOOR = 2000, 5, 0.70, 25
_tok = _mod = None

def load(model_path):
    global _tok, _mod
    _tok = AutoTokenizer.from_pretrained(model_path)
    _mod = AutoModel.from_pretrained(model_path).eval()

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

def null_strat(A, S, rng, minn=MINN):
    A2 = A.copy(); mask = np.zeros(len(A), bool)
    for s in np.unique(S):
        idx = np.where(S == s)[0]
        if len(idx) >= minn:
            A2[idx] = rng.permutation(A[idx]); mask[idx] = True
    return A2, mask

def seed_of(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

def kfor(n):  # the declared rule
    return 2 if n < 150 else (3 if n < 400 else 4)

df = pd.read_csv(TURNS)
g  = df.groupby("conv_id").size()
convs = sorted(g[g >= FLOOR].index)
print(f"{len(convs)} eligible conversations", flush=True)

def embeddings(name, path):
    D = os.path.join(ROOT, "results", "sensitivity", "cache", name); os.makedirs(D, exist_ok=True)
    todo = [c for c in convs if not os.path.exists(f"{D}/{c}.npz")]
    if todo:
        print(f"[{name}] embedding {len(todo)} conversations", flush=True)
        load(path); t0 = time.time()
        for i, cid in enumerate(todo, 1):
            d = df[df.conv_id == cid].sort_values("Turn")
            np.savez_compressed(f"{D}/{cid}.npz",
                                p=embed(d["Prompt"].fillna("").tolist()),
                                r=embed(d["Response"].fillna("").tolist()))
            print(f"   {i}/{len(todo)}  {cid}  n={len(d)}  {time.time()-t0:.0f}s", flush=True)
    E = {}
    for c in convs:
        z = np.load(f"{D}/{c}.npz"); E[c] = (z["p"], z["r"])
    return E

def run(E, kmode, shared, tag):
    rows = []
    for cid in convs:
        Ep, Er = E[cid]; n = len(Ep); base = kfor(n)
        k = {"base": base, "minus": max(2, base - 1), "plus": base + 1, "fix2": 2}[kmode]
        if shared:
            lab = KMeans(k, random_state=0, n_init=10).fit_predict(np.vstack([Ep, Er]))
            cp, cr = lab[:n], lab[n:]
        else:
            cp = KMeans(k, random_state=0, n_init=10).fit_predict(Ep)
            cr = KMeans(k, random_state=0, n_init=10).fit_predict(Er)
        for side in ("model", "user"):
            S, A, Sp = (cp[:-1], cr[:-1], cp[1:]) if side == "model" else (cr[:-1], cp[1:], cr[1:])
            eps, room = eps_from(S, A, Sp, k)
            sym = all(len(np.unique(x)) == k for x in (S, A, Sp))
            cells = int((np.bincount((S * k + A) * k + Sp, minlength=k ** 3) > 0).sum())
            ok = sym and cells and len(S) / cells >= 3 and room > 0.05 and eps == eps
            clear = False
            if ok:
                rng = np.random.default_rng(seed_of(cid, side, "strat"))
                _, mask = null_strat(A, S, rng)
                if mask.mean() >= COVER:
                    Sm, Am, Spm = S[mask], A[mask], Sp[mask]
                    obs = eps_from(Sm, Am, Spm, k)[0]
                    vals = [eps_from(Sm, null_strat(Am, Sm, rng)[0], Spm, k)[0] for _ in range(NPERM)]
                    clear = bool(obs > np.nanpercentile(vals, 95)); eps = obs
            rows.append(dict(tag=tag, record=cid, side=side, n=n, k=k, eps=eps,
                             screen_ok=bool(ok), clear=clear))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, f"grid_{tag}.csv"), index=False)
    return r

def direction(r, tag):
    w = r.pivot_table(index="record", columns="side", values="clear", aggfunc="first")
    e = r.pivot_table(index="record", columns="side", values="eps",   aggfunc="first")
    both = w.index[(w["model"] == True) & (w["user"] == True)]
    d = (e.loc[both, "model"] - e.loc[both, "user"]).dropna()
    neg = int((d < 0).sum()); n = len(d)
    p = stats.binomtest(neg, n, 0.5).pvalue if n else np.nan
    med = float(np.median(d)) if n else np.nan
    print(f"RESULT {tag:28s} both={n:3d}  negative={neg:3d}  median={med:+.4f}  p={p:.3g}", flush=True)
    return dict(config=tag, both_clear=n, negative=neg, median=med, p=p)

summary = []
for mname, mpath in (("minilm", MINILM), ("mpnet", MPNET)):
    E = embeddings(mname, mpath)
    for kmode, shared, tag in (("base",  False, f"{mname}_k-declared_separate"),
                               ("minus", False, f"{mname}_k-minus1_separate"),
                               ("plus",  False, f"{mname}_k-plus1_separate"),
                               ("base",  True,  f"{mname}_k-declared_shared"),
                               ("fix2",  False, f"{mname}_k-2fixed_separate")):
        f = os.path.join(OUT, f"grid_{tag}.csv")
        r = pd.read_csv(f) if os.path.exists(f) else run(E, kmode, shared, tag)
        summary.append(direction(r, tag))
        pd.DataFrame(summary).to_csv(os.path.join(OUT, "grid_summary.csv"), index=False)

print("\nDONE")
print(pd.DataFrame(summary).to_string(index=False))
