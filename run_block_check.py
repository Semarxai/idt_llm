"""
C1 centred block-bootstrap check  (Reviewer 2 pt 3; Supplementary Table 3)

Uses Chenan Wei's block_ci_test (LLM_Reviewer_Response_Support/serial_ci.py)
on the ORIGINAL C1 symbols: MiniLM, declared k, separate clustering,
random_state=0, n_init=10, rebuilt from cache/minilm.

Pre-declared settings (agreed before running):
  block lengths 8, 16, 32, each reported separately
  a length is used only if N >= 5 x length, N = number of transitions
  1,999 draws, p = (1 + #{Q* >= Q}) / 2000, a side clears at p < 0.05
  both sides must clear at the same length to enter that length's comparison
  effect summary: the manuscript's eps_A (grid_minilm_k-declared_separate.csv)
  screens: symbol occupancy, cell occupancy, room > 0.05 (unchanged)

Interruption-safe: every finished test is appended to block_check_results.csv
and flushed to disk at once. Re-running the script skips finished tests and
continues where it stopped.

Writes, to results/sensitivity/:
  block_check_manifest.json   settings, versions, checksums
  block_check_results.csv     one row per conversation x side x block length
  block_check_summary.csv     one row per block length (Supplementary Table 3)
"""
import os, sys, csv, json, time, hashlib, platform
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import sklearn

# ---------------------------------------------------------------- PATHS
HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUT     = os.path.join(ROOT, "results", "sensitivity")
TURNS   = os.environ.get("SCC_TURNS", os.path.join(ROOT, "data", "c1", "source", "turns_all_ts.csv"))
SUPPORT = os.environ.get("SCC_SUPPORT", os.path.join(ROOT, "supplementary", "serial_ci"))
CACHE   = os.environ.get("SCC_CACHE", os.path.join(HERE, "cache", "minilm"))
GRID    = os.path.join(OUT, "grid_minilm_k-declared_separate.csv")
RESULTS = os.path.join(OUT, "block_check_results.csv")
SUMMARY = os.path.join(OUT, "block_check_summary.csv")
MANIFEST= os.path.join(OUT, "block_check_manifest.json")
# ----------------------------------------------------------------------

BLOCKS, MIN_BLOCKS, DRAWS, ALPHA = (8, 16, 32), 5, 1999, 0.05
MINN, COVER, FLOOR = 5, 0.70, 25

sys.path.insert(0, SUPPORT)
from serial_ci import block_ci_test   # noqa: E402

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""): h.update(chunk)
    return h.hexdigest()

# ---- guard: Chenan's code must be exactly the version he sent ----------
expected = {}
with open(os.path.join(SUPPORT, "SHA256SUMS")) as f:
    for line in f:
        digest, name = line.split(maxsplit=1); expected[name.strip()] = digest
if sha256(os.path.join(SUPPORT, "serial_ci.py")) != expected["serial_ci.py"]:
    sys.exit("STOP: serial_ci.py differs from the version Chenan sent.")

# ---- same estimator helpers as run_sensitivity.py ----------------------
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

def strat_mask(S):
    mask = np.zeros(len(S), bool)
    for s in np.unique(S):
        idx = np.where(S == s)[0]
        if len(idx) >= MINN: mask[idx] = True
    return mask

def kfor(n): return 2 if n < 150 else (3 if n < 400 else 4)

def seed_of(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

# ---- load ---------------------------------------------------------------
df = pd.read_csv(TURNS)
g = df.groupby("conv_id").size()
convs = sorted(g[g >= FLOOR].index)
grid = pd.read_csv(GRID)
paper_eps = {(r.record, r.side): r.eps for r in grid.itertuples()}
gw = grid.pivot_table(index="record", columns="side", values="clear", aggfunc="first")
orig24 = set(gw.index[(gw["model"] == True) & (gw["user"] == True)])
print(f"{len(convs)} eligible conversations; original both-clear set: {len(orig24)}", flush=True)

# ---- rebuild original symbols and VERIFY against the published grid ----
streams, mism = {}, 0
for cid in convs:
    z = np.load(os.path.join(CACHE, f"{cid}.npz")); Ep, Er = z["p"], z["r"]
    k = kfor(len(Ep))
    cp = KMeans(k, random_state=0, n_init=10).fit_predict(Ep)
    cr = KMeans(k, random_state=0, n_init=10).fit_predict(Er)
    for side in ("model", "user"):
        S, A, Sp = (cp[:-1], cr[:-1], cp[1:]) if side == "model" else (cr[:-1], cp[1:], cr[1:])
        eps, room = eps_from(S, A, Sp, k)
        sym = all(len(np.unique(x)) == k for x in (S, A, Sp))
        cells = int((np.bincount((S * k + A) * k + Sp, minlength=k ** 3) > 0).sum())
        ok = bool(sym and cells and len(S) / cells >= 3 and room > 0.05 and eps == eps)
        if ok:
            m = strat_mask(S)
            if m.mean() >= COVER: eps = eps_from(S[m], A[m], Sp[m], k)[0]
        pe = paper_eps[(cid, side)]
        if not ((np.isnan(eps) and np.isnan(pe)) or np.isclose(eps, pe, atol=1e-12)): mism += 1
        streams[(cid, side)] = dict(S=S, A=A, Y=Sp, k=k, N=len(S), screen_ok=ok)

if mism:
    sys.exit(f"STOP: {mism} readings differ from the published grid; symbols not reproduced.")
d0 = pd.Series({c: paper_eps[(c, "model")] - paper_eps[(c, "user")] for c in orig24})
print(f"VERIFIED: all {2*len(convs)} readings match the published grid; "
      f"original comparison {int((d0 < 0).sum())} of {len(d0)} negative, median {d0.median():+.4f}", flush=True)

# ---- manifest (written once) -------------------------------------------
if not os.path.exists(MANIFEST):
    json.dump(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), blocks=BLOCKS, min_blocks=MIN_BLOCKS,
                   draws=DRAWS, alpha=ALPHA, p_rule="(1 + #{Q* >= Q}) / (draws + 1)",
                   seeds="md5(record|side|block|length)", symbols="MiniLM, declared k, separate, random_state=0, n_init=10",
                   effect_summary="manuscript eps_A from grid_minilm_k-declared_separate.csv",
                   serial_ci_sha256=expected["serial_ci.py"], python=platform.python_version(),
                   numpy=np.__version__, sklearn=sklearn.__version__),
              open(MANIFEST, "w"), indent=2)

# ---- run, checkpointing every finished test -----------------------------
FIELDS = ["record", "side", "block_length", "N", "k", "screen_ok", "status",
          "p_block", "clear", "statistic", "eps_MM_full", "warnings", "seed"]
done = set()
if os.path.exists(RESULTS):
    for r in csv.DictReader(open(RESULTS)):
        done.add((r["record"], r["side"], int(r["block_length"])))
    print(f"resuming: {len(done)} tests already on disk", flush=True)

fh = open(RESULTS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if not done and os.path.getsize(RESULTS) == 0: w.writeheader(); fh.flush()

todo = [(c, s, l) for c in convs for s in ("model", "user") for l in BLOCKS if (c, s, l) not in done]
t0 = time.time()
for i, (cid, side, ell) in enumerate(todo, 1):
    st = streams[(cid, side)]
    row = dict(record=cid, side=side, block_length=ell, N=st["N"], k=st["k"],
               screen_ok=st["screen_ok"], seed=seed_of(cid, side, "block", ell))
    if not st["screen_ok"]:
        row.update(status="screen_failed", clear=False)
    elif st["N"] < MIN_BLOCKS * ell:
        row.update(status="too_short", clear=False)
    else:
        try:
            res = block_ci_test(st["S"], st["A"], st["Y"], block_length=ell, draws=DRAWS, seed=row["seed"])
            p = res.get("p_block", np.nan)
            row.update(status=res["status"], p_block=p, clear=bool(p == p and p < ALPHA),
                       statistic=res.get("statistic", ""), eps_MM_full=res.get("eps_MM", ""),
                       warnings=res.get("warnings", ""))
        except Exception as e:
            row.update(status=f"error: {e}", clear=False)
    w.writerow(row); fh.flush(); os.fsync(fh.fileno())
    if i % 10 == 0 or i == len(todo):
        print(f"   {i}/{len(todo)}  {cid}  {side:5}  l={ell:2d}  {row['status']}  {time.time()-t0:.0f}s", flush=True)
fh.close()

# ---- summary --------------------------------------------------------------
r = pd.read_csv(RESULTS)
out = []
for ell in BLOCKS:
    x = r[r.block_length == ell]
    usable_m = set(x[(x.side == "model") & x.status.isin(["computed", "warnings"])].record)
    usable_u = set(x[(x.side == "user") & x.status.isin(["computed", "warnings"])].record)
    usable = usable_m & usable_u
    cm = set(x[(x.side == "model") & (x.clear == True)].record)
    cu = set(x[(x.side == "user") & (x.clear == True)].record)
    both = cm & cu
    dd = pd.Series({c: paper_eps[(c, "model")] - paper_eps[(c, "user")] for c in both}, dtype=float)
    neg, n = int((dd < 0).sum()), len(dd)
    out.append(dict(block_length=ell, usable_records=len(usable),
                    original24_usable=len(orig24 & usable), both_sides_clear=n,
                    overlap_with_original24=len(both & orig24), new_entrants=len(both - orig24),
                    inconclusive=len(usable) - n, negative=neg,
                    median_difference=float(dd.median()) if n else np.nan,
                    sign_test_p=stats.binomtest(neg, n, 0.5).pvalue if n else np.nan))
s = pd.DataFrame(out); s.to_csv(SUMMARY, index=False)
print("\nDONE (saved as results/sensitivity/block_check_summary.csv)")
print(s.to_string(index=False))
