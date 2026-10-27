# Reviewer 2.3: reference implementation and synthetic checks

The mathematical derivations, reviewer wording and manuscript edits are in the
accompanying PDF. This archive contains code and saved synthetic results only.

## Run the remaining C1 check locally

Use the existing v11 preprocessing in your own analysis environment. Reuse the
original symbol assignments where available, or regenerate them with the original
settings and seeds. Insert the check at the stage where the aligned state, action
and next-state symbols form the contingency table. This package does not assume
how those objects are stored and does not request a data export or transfer.

The array-level interface is:

```python
from serial_ci import block_ci_test

# S, A and Y denote your existing aligned symbol arrays for one uninterrupted
# conversation and direction. Map these mathematical roles to your own objects.
result = block_ci_test(S, A, Y, block_length=ell, draws=1999, seed=seed)
```

The core calculation is: compute conditional-independence cell residuals;
resample consecutive blocks of whole recorded triplets within one sequence;
recompute the residuals and centre each bootstrap replicate at the observed
residual; compare the observed quadratic statistic with this centred reference.
The accompanying note defines d, Q_N and Q_N* explicitly.

First confirm that your original counts and contribution estimates are recovered.
Keep representation, ordering, record/run boundaries and existing screens fixed.
Never create new transitions across resampled block joins or close gaps caused by
screening. The procedure tests the same one-time conditional-independence null;
it does not replace the reported contribution measure or condition on more history.

Prespecify feasible block lengths from record size, support and dependence, not the
directional result. The synthetic grid 8, 16, 32 is a diagnostic grid, not a universal
recommendation. Report all retained lengths and any inconclusive settings.

For each retained length report: usable records; bilateral clearing count; overlap
with the original 24; negative model-minus-human differences; median difference.
Recalculate eligibility instead of testing only the previously selected 24.

The supplied CSV command is an OPTIONAL example adapter. Its field names are not
assumed to exist in the v11 pipeline. Use the function directly or change the adapter:

```bash
python serial_ci.py example_symbols.csv --blocks 8 16 32 --draws 1999 --output serial_ci_results.csv
```

Only this optional adapter uses `record,side,t,S,A,Sp`, with one uninterrupted run
per record/side and consecutive integer time indices. No export to that schema is
needed when using the function directly. The implementation is for separate runs;
it is not a bootstrap for concatenated C2 sequences.

## Conditions and status

The C1 reanalysis has NOT been performed here. No C1 text, ordered symbols or C1
block-reference results are supplied. The current revision changes documentation
and interface guidance, not the numerical procedure or saved simulation results.

The centred block reference is asymptotic, not an exact finite-sample test. It
remains anticonservative in some strongly persistent or short-record synthetic
nulls. Warnings do not validate stationarity or establish admissibility. The code
does not automatically select eligible records or perform the median bootstrap.

## Files and testing

- `serial_ci.py`: array-level test and optional CSV adapter.
- `validate_serial_ci.py`: fixed stress grid, including adverse outcomes.
- `validate_markov_extension.py`: target-sufficient nulls with action memory.
- `test_serial_ci.py`: implementation tests.
- `results/`: all saved synthetic replicates and summaries.
- `environment.json`: environment recorded with the original diagnostic package.
- `provenance.json`: source and preparation history.

Python 3.10+ and NumPy are sufficient:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
```

To regenerate the synthetic experiments:

```bash
python validate_serial_ci.py --replicates 300 --draws 399
python validate_markov_extension.py
```

These commands overwrite result files; use a separate copy for regeneration.
