# Closure (Ω̂) — code provenance, column dictionary, run order

This package regenerates every file under `results/closure/` that the paper reports,
plus Figures 2 and 3. It contains no new computation: the two analysis scripts are the
ones that produced the published files, ported only for paths; the two summary writers
are reconstructions verified byte-for-byte against the published text files.

## 1. File → script

| output (under `CLOSURE`) | produced by | status |
|---|---|---|
| `omega_w1_c1.csv`, `omega_w1_c2.csv`, `omega_w1_c3.csv` | `scripts/omega_w1.py` | original logic, paths ported |
| `series_c1.npz`, `series_c2.npz` | `scripts/omega_w1.py` | original logic, paths ported |
| `omega_counts.txt` | `scripts/omega_counts.py` | **reconstructed**, byte-identical to the published file |
| `sweep_c1.csv`, `sweep_c2.csv`, `sweep_c3.csv` | `scripts/omega_sweep.py` | original logic, paths ported |
| `sweep_summary.txt` | `scripts/sweep_summary.py` | **reconstructed**, byte-identical to the published file |
| `FIGURES/fig2_omega_w1.{pdf,png}` | `scripts/fig2_omega_w1.py` | the final version, paths ported |
| `FIGURES/fig3_sweep.{pdf,png}` | `scripts/fig3_sweep.py` | the final version, paths ported |
| `omega_w1_claude_fc552e23.csv`, `omega_w1_claude_90b6161d.csv`, `omega_w1_gemini_68bac569.csv` | **not produced by this code** | see §6 |

Shared modules: `src/omega_core.py` (tokenizer, entropies, Ω̂ at W = 1 and windowed,
D1 band), `src/photobook_pairs.py` (C3 pairing), `src/closure_inputs.py` (the one place
that maps `paths.py` onto the raw corpora).

**`src/omega.py` and `src/msg_entropy.py` are not the basis.** The scripts here are
self-contained and import nothing from the earlier closure work. Whether those modules
compute the same quantity has not been checked.

## 2. Run order

```
python scripts/omega_w1.py all       # omega_w1_{c1,c2,c3}.csv, series_{c1,c2}.npz
python scripts/omega_counts.py       # omega_counts.txt   (reads the three CSVs only)
python scripts/omega_sweep.py c1     # sweep_c1.csv
python scripts/omega_sweep.py c2     # sweep_c2.csv
python scripts/omega_sweep.py c3     # sweep_c3.csv
python scripts/sweep_summary.py      # sweep_summary.txt  (reads the three CSVs only)
python scripts/fig2_omega_w1.py      # Figure 2
python scripts/fig3_sweep.py         # Figure 3
```

`omega_counts.py` and `sweep_summary.py` also accept `src_dir out_dir` arguments, so they
can be pointed at the published CSVs and diffed without touching `results/`.

## 3. Definitions

**Tokenizer.** `NousResearch/Llama-2-7b-hf`, fast tokenizer, `add_special_tokens=False`,
vocabulary K = 32,000. In the original session it was loaded from a local copy of that
tokenizer's files; the files are the same.

**Message representation.** q_i = token counts of message i / total tokens in message i.

**Sides.** Side A and side B are two index-aligned streams; message t of A and message t
of B belong to the same paired turn.

| corpus | records | side A | side B | record id |
|---|---|---|---|---|
| C1 | `conv_id` with ≥ 25 turns (58 of 65) | `Prompt` (human) | `Response` (model) | `conv_id` |
| C2 | every `run_id` (8 × 25 turns) | `prompt` (asker) | `response` (answerer) | `run_id.split("_")[1]`, e.g. `s11` |
| C3 | eligible PhotoBook games (2,171 of 2,502) | first speaker | second speaker | log filename without `.json` |

C1 and C2 messages are ordered by `Turn` / `turn`; missing text becomes the empty string.
C3 pairing and eligibility are in `src/photobook_pairs.py` — selections are removed
*before* same-speaker collapse; 6 score-anomaly games and 325 games under 25 pairs are
excluded.

**Ω̂ at W = 1** (`omega_w1.py`). Ω̂(t) = H(q_t) − H(q_{t−1}), plug-in entropy in bits.
gap(t) = Ω̂_A(t) − Ω̂_B(t).

**Ω̂ at window W** (`omega_sweep.py`). Equal-message-weighted mixtures,
q̄_Z(t) = (1/W) Σ_{i=t−W+1..t} q_i and q̄_S(t) = (1/W) Σ_{i=t−W..t−1} q_i;
Ω̂(t) = H(q̄_Z(t)) − H(q̄_S(t)). Computed on a rolling window, one value per step;
a stream of n messages gives n − W values. The two streams are truncated to their
common length, `readings`.

**Empty messages — the one place the two scripts differ.** At W = 1 an empty message
keeps its slot and makes Ω̂ NaN at that turn and the next. In the sweep, empty messages
are dropped from their stream before windowing. The corpora contain exactly one empty
message (one C1 prompt); C2 and C3 contain none. So the W = 1 rows of `sweep_c1.csv`
and the corresponding record in `omega_w1_c1.csv` can differ for that one conversation.
The paper's W = 1 counts (57 of 58; 1,025 / 1,015 / 131) are the same from either file.

**D1 band.** band(W) = (1/W)·log₂(K − 1) + h₂(1/W), h₂ the binary entropy, K = 32,000.
This is the Fannes–Audenaert bound h₂(δ) + δ·log₂(m − 1) at δ = 1/W, m = K.
W=2 8.4829 · W=5 3.7151 · W=10 1.9656 · W=20 1.0347 · W=50 0.4408.
Undefined at W = 1 (the closure premise needs W ≥ 2; formally it would be 14.97 bits,
wider than any entropy in the data). Stored as empty in the `band` column at W = 1.

**Measurable.** A (record, W) cell is measurable when `readings ≥ 10`. A record shorter
than W + 1 messages produces no row at that W. Unmeasurable rows are kept in the CSV with
`measurable = False` and excluded from every summary and from Figure 3.

**Directional counts and ties.** A record leads for side A when `absA_gt_absB` exceeds
half its readings (`readings / 2` in the sweep, `n_compared / 2` at W = 1), for side B
when it is below half. Exactly half is a tie. **Ties are excluded from the denominator**
and the sign test is the two-sided exact binomial at p = 0.5 on the non-tied records.
Examples from the published files: C1 W = 10 is "53 of 57" (58 measurable, 1 tie);
C3 W = 1 is "1025 of 2040" (2,171 records, 131 ties).
The gap line in `omega_counts.txt` uses `gap_positive` vs `gap_negative` the same way.

**Summary maxima** (`sweep_summary.txt`, Figure 3) are taken over measurable records only.

## 4. Column dictionary

### `omega_w1_{c1,c2,c3}.csv` — one row per record

| column | meaning |
|---|---|
| `corpus` | C1, C2 or C3 |
| `conv` | record id (see §3) |
| `n_turns` | paired turns in the record |
| `readings_A`, `readings_B` | finite Ω̂ values on each side (n_turns − 1, less any NaN from empty messages) |
| `H_A_min`, `H_A_max`, `H_B_min`, `H_B_max` | min / max per-message entropy H(q_t), bits, 4 dp |
| `omega_A_min`, `omega_A_max`, `omega_B_min`, `omega_B_max` | min / max Ω̂ at W = 1, bits, 4 dp |
| `gap_min`, `gap_max` | min / max of Ω̂_A − Ω̂_B |
| `gap_positive`, `gap_negative` | turns with gap > 0 and gap < 0 |
| `absA_gt_absB` | turns with \|Ω̂_A\| > \|Ω̂_B\| |
| `n_compared` | turns where both sides are finite |
| `empty_A`, `empty_B` | zero-token messages per side |
| `platform` | C1 only: ChatGPT, Claude or Gemini |
| `score` | C3 only: summed round score of the game |

### `series_{c1,c2}.npz`

One array per record, keyed by record id, shape 3 × n_turns:
row 0 Ω̂_A(t), row 1 Ω̂_B(t), row 2 gap(t). Index 0 is NaN by construction. No C3 file.

### `sweep_{c1,c2,c3}.csv` — one row per (record, W)

| column | meaning |
|---|---|
| `corpus`, `conv`, `n_turns` | as above |
| `W` | window, one of 1, 2, 5, 10, 20, 50 |
| `readings` | common number of Ω̂ values on the two sides at this W |
| `band` | D1 band at this W, 4 dp; empty at W = 1 |
| `measurable` | `readings ≥ 10` |
| `maxabs_A`, `maxabs_B` | max \|Ω̂\| on each side, 5 dp |
| `maxabs_gap` | max \|Ω̂_A − Ω̂_B\| |
| `omega_A_min`, `omega_A_max`, `omega_B_min`, `omega_B_max` | signed extremes |
| `absA_gt_absB` | readings with \|Ω̂_A\| > \|Ω̂_B\| |
| `exceed_band_A`, `exceed_band_B` | readings with \|Ω̂\| > band; empty at W = 1 |
| `ratio_A`, `ratio_B` | `maxabs / band`, 5 dp; empty at W = 1 |
| `platform` | C1 only |
| `score` | C3 only |

## 5. Figures 2 and 3 — inputs

| figure | reads (all under `CLOSURE`) |
|---|---|
| Figure 2 | `series_c1.npz` (panel a, record `claude_5f70ab01`), `omega_w1_c1.csv` (panel b, C1 lead fraction = `absA_gt_absB / n_compared`), `sweep_c3.csv` W = 1 rows (panel b, C3 lead fraction = `absA_gt_absB / readings`) |
| Figure 3 | `sweep_c1.csv`, `sweep_c2.csv`, `sweep_c3.csv`, measurable rows only |

Figure 2 exemplar rule, declared in the script: the longest C1 conversation below 150
paired turns, which is `claude_5f70ab01` (144 turns). The annotation strings in panel b
("57 of 58 … p = 4 × 10⁻¹⁶", "1,025 above · 1,015 below · 131 tied … p = 0.84") are
literal text in the script, not computed at draw time. Panel b uses a fixed jitter seed (7).

## 6. Not produced by this code

**`omega_w1_claude_fc552e23.csv`, `omega_w1_claude_90b6161d.csv`,
`omega_w1_gemini_68bac569.csv`.** No script in this session wrote per-conversation CSVs,
and these filenames do not appear in any session output. They were produced elsewhere.
Two observations that matter for the repository:

- `claude_fc552e23` (238 turns) and `claude_90b6161d` (1,402 turns) are C1 records. Their
  per-turn W = 1 series are in `series_c1.npz` under those keys, so the files can be
  regenerated from it — but the column layout of the existing files is unknown here, so
  they cannot be certified as matching.
- **`gemini_68bac569` is not a C1 record.** It has 10 turns in `turns_all_ts.csv`, below
  the 25-turn floor, and appears in neither `omega_w1_c1.csv` nor `series_c1.npz`.
  Whatever produced that file ran outside the published C1 set. It should not sit under
  `results/closure/` next to published outputs without a note saying what it is.

**Figure 1.** Not produced in this session. The schematic
(`Figure1_Recurrent_Communication_Loop.png`) was already in the working folder; no
script for it exists here.

## 7. Verification performed

No Ω̂ value was recomputed. The two reconstructed writers were run once against the
session's published CSVs, writing to a scratch directory, and diffed:

```
omega_counts.txt    IDENTICAL
sweep_summary.txt   IDENTICAL
```

Checksums of the session outputs this code reproduces — compare against
`results/closure/` to confirm the repository holds the same files:

```
7655ccd1fff053d50efb15b4d101aea7  omega_w1_c1.csv
6aae61fe9419abe051b01e1dc906eca8  omega_w1_c2.csv
1a257c07cc2e94de555887afae6b3aa4  omega_w1_c3.csv
7271df1c97c3e44193c216768731cf0c  omega_counts.txt
fc720ec9ef0445dad72833cb9cc22cfa  series_c1.npz
30cda30b3eb01b8aa99827ad69b86e97  series_c2.npz
5406771615e572f62748f5aa63e2e606  sweep_c1.csv
ed6e61c25e093b8e5aa7fb55755d1885  sweep_c2.csv
b12a0c6baa1a7ea7f331a8686a5bae7f  sweep_c3.csv
91ef4affda5406194ae61e547ca2f72f  sweep_summary.txt
```

Original session scripts, for the record: `omega_w1_all.py`
(md5 8f3a1bebd02cf47aa27c8cecaea9b3e3) and `omega_sweep.py`
(md5 96ec76c750bc5e7be970ddc6b2399b14), plus `pb_run.parse_game` for C3. Original input
locations: C1 `07_IDT_LLM_v3/Data/turns_all_ts.csv`; C2
`13 IDT_Closure_Sep/Data/advancement/omega_advancement_test_v1_turns.csv`; C3 the
PhotoBook release `logs/` directory.

**Not verified:** that the ported `omega_w1.py` and `omega_sweep.py` reproduce the CSVs
bit-for-bit end to end. They were not re-run, per the instruction. The ports change only
path resolution and the tokenizer source (hub id instead of a local copy of the same
files). The first full run in the repository should diff against the checksums above.
