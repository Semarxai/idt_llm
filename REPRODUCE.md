# Reproduction map

Each manuscript display item and reported number, with the script that produces it
and the data it consumes.

| Display item / number | Script | Data | Status |
|---|---|---|---|
| Table 1 | — editorial | — | n/a |
| Figure 1 — loop schematic | none (drawn) | — | schematic, no script |
| Figure 2a,b — Omega at W=1 | `scripts/fig2_omega_w1.py` | `results/closure/series_c1.npz`, `omega_w1_c1.csv`, `sweep_c3.csv` | complete |
| Figure 3 — band occupancy by W | `scripts/fig3_sweep.py` | `results/closure/sweep_c*.csv` | complete |
| Figure 4a–d — contribution | `scripts/fig4_v5.py` | `results/contribution/c1_nulls.csv`, `c3_nulls.csv` | complete |
| Figure 5a–c — controls | `scripts/fig5.py` | `results/contribution/c1_controls.csv`, `c2_controls.csv`, `c1_cos.csv` | complete |
| C1 clears 25 / 36 / 24 | `scripts/run_c1.py` | `results/contribution/c1_nulls.csv` | complete |
| C2 four pooled cells clear | `scripts/run_c2.py`, `run_c2b.py` | `results/contribution/c2_nulls*.csv` | complete |
| C3 clears 620 / 784 / 143 | `scripts/run_c3.py` | `results/contribution/c3_nulls.csv` | complete |
| C3 eligibility 2,171 of 2,502 | `scripts/prep_c3b.py`, `src/c3_pairing_audit.py` | `data/c3/c3_manifest.csv` | complete |
| Control reductions 92.0 / 86.8 / 90.0% | `scripts/controls.py`, `controls_c2.py` | `results/contribution/c1_controls.csv`, `c2_controls.csv` | complete |
| Control cosine 0.53 / 0.48 / 0.37 | `scripts/cos_c1.py`, `controls_c2.py` | `results/contribution/c1_cos.csv` | complete |
| Calibration long, type-I 0.045 | `scripts/calib.py` | `results/calibration/calibration.csv` | complete |
| Calibration short, type-I 0.028 | `scripts/calib_short.py` | `results/calibration/calibration_short.csv` | complete |
| Length / prefix analysis | `scripts/prefix.py` | `results/contribution/prefix.csv` | complete |
| Omega counts, 57 of 58 and 1,025 vs 1,015 | `scripts/omega_counts.py` | `results/closure/omega_w1_*.csv` | complete |
| Window bands W=2 … W=50 | `scripts/omega_sweep.py`, `scripts/sweep_summary.py` | `results/closure/sweep_c*.csv` | complete |
| D1 bound verification | `src/verify_d1_proof.py`, `src/verify_band_scaling.py` | analytic | complete |
| Estimator verification | `src/verify_estimator.py` | analytic | complete |
| C2 generation pipeline | `src/generate.py`, `seam_log.py`, `seam_run.py`, `pilot_gate.py`, `task_hidden_spec.py`, `task_hidden_incident.py` | — | requires a local model endpoint and API keys |

Closure provenance, column dictionary and run order are in `docs/closure_provenance.md`.
Figure 2 panel a reads `results/closure/series_c1.npz`, the per-turn series for C1; it
contains Omega values only and no message text.

All scripts resolve their inputs and outputs through `src/paths.py`, relative to the
repository root. Run them from anywhere:

```
pip install -r requirements.txt
python scripts/run_c3.py        # writes results/contribution/c3_nulls.csv
python scripts/fig4_v5.py       # writes figures/fig4_epsilon.png
```

C1 source is not in the repository, so `scripts/run_c1.py` expects
`data/c1/source/turns_all_ts.csv` and will not run without it. The C1 results it would
produce are already in `results/contribution/`.
