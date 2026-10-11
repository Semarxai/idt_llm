"""
Omega (closure) core — shared by scripts/omega_w1.py and scripts/omega_sweep.py.

Representation
    Every message is tokenized with the Llama-2 tokenizer (K = 32,000),
    `add_special_tokens=False`.  A message is represented by its normalised
    token-frequency vector q_i (token id -> count / total tokens in message).
    No symbol alphabet, no embedder, no clustering, no permutation null.

Two estimators, and the one difference between them
    W = 1 (scripts/omega_w1.py) works on per-message entropies directly:
        Omega(t) = H(q_t) - H(q_{t-1})
        An EMPTY message (zero tokens) gives H = NaN, so Omega is NaN at that
        turn and at the next.  The message keeps its slot; alignment between
        the two streams is preserved.

    W >= 1 (scripts/omega_sweep.py) works on equal-message-weighted mixtures:
        qbar_Z(t) = (1/W) sum_{i=t-W+1..t}   q_i
        qbar_S(t) = (1/W) sum_{i=t-W..t-1}   q_i
        Omega(t)  = H(qbar_Z(t)) - H(qbar_S(t))
        Here an EMPTY message is DROPPED from its stream before windowing.
        The two streams are then truncated to their common number of
        readings.  In the corpora as used this affects exactly one C1 message
        (one empty prompt, side A); C2 and C3 have none.  So the W = 1 rows of
        sweep_c1.csv and omega_w1_c1.csv can differ for that one conversation.

    Equal message weighting is required: pooling raw token counts across a
    window breaks TV <= 1/W and with it the D1 band.

D1 band
    band(W) = (1/W) * log2(K - 1) + h2(1/W),   K = 32,000,   W >= 2
        W=2  8.4829   W=5  3.7151   W=10  1.9656   W=20  1.0347   W=50  0.4408
    At W = 1 it evaluates to log2(K-1) = 14.97 bits, wider than any entropy in
    the data, and the closure premise needs W >= 2: no band is defined there.

All entropies in bits, plug-in (no Miller-Madow on the Omega side).
"""
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer

K = 32000
GRID = [1, 2, 5, 10, 20, 50]
MIN_READINGS = 10          # a (record, W) cell below this is unmeasurable

_TOK = None


def tokenizer(name):
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(name, use_fast=True)
    return _TOK


def token_ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------ entropy
def H_bits(tok, text):
    """Per-message plug-in entropy of the token distribution, and token count.
    Empty message -> (NaN, 0)."""
    ids = token_ids(tok, text)
    if not ids:
        return np.nan, 0
    c = np.fromiter(Counter(ids).values(), dtype=np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum()), len(ids)


def h2(x):
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def band(W):
    """D1 finite-window band.  NaN at W = 1 (not defined)."""
    if W < 2:
        return np.nan
    return (1.0 / W) * np.log2(K - 1) + h2(1.0 / W)


BANDS = {W: band(W) for W in GRID}


# ------------------------------------------------------------- W = 1 path
def omega_stream_w1(tok, texts):
    """Omega at W = 1 for one stream.  Returns (omega, H, n_tokens).
    omega[0] is NaN; an empty message makes its own and the next Omega NaN."""
    h, n = [], []
    for t in texts:
        a, b = H_bits(tok, t)
        h.append(a); n.append(b)
    h = np.array(h, dtype=float)
    om = np.full(len(h), np.nan)
    om[1:] = h[1:] - h[:-1]
    return om, h, np.array(n)


# ---------------------------------------------------------- windowed path
def message_dists(tok, texts):
    """q_i per message as {token_id: probability}.  Empty messages DROPPED."""
    out = []
    for t in texts:
        ids = token_ids(tok, t)
        if not ids:
            continue
        c = Counter(ids)
        n = sum(c.values())
        out.append({k: v / n for k, v in c.items()})
    return out


def _H_mix(acc, W):
    """Entropy of the equal-weight mixture whose summed q_i are held in acc."""
    s = 0.0
    for v in acc.values():
        p = v / W
        if p > 0:
            s -= p * np.log2(p)
    return s


def omega_series(qs, W):
    """Rolling-window Omega(t), equal message weighting.
    Returns len(qs) - W values; empty array if the stream is too short."""
    n = len(qs)
    if n < W + 1:
        return np.array([])
    acc = defaultdict(float)
    for q in qs[:W]:
        for k, v in q.items():
            acc[k] += v
    prev = _H_mix(acc, W)                     # qbar over messages 0 .. W-1
    out = []
    for t in range(W, n):
        for k, v in qs[t - W].items():        # drop the oldest
            acc[k] -= v
            if acc[k] <= 1e-15:
                del acc[k]
        for k, v in qs[t].items():            # add the newest
            acc[k] += v
        cur = _H_mix(acc, W)
        out.append(cur - prev)
        prev = cur
    return np.array(out)
