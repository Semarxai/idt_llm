"""
Input locations for the closure (Omega) scripts.

This is the ONLY file that maps the repository's path constants onto the three
raw corpora.  If `src/paths.py` defines C1, C2, C3 differently from what is
assumed here, adapt these three lines and nothing else.

Original session locations are recorded in docs/closure_provenance.md.
"""
from paths import C1, C2, C3, CLOSURE, FIGURES   # noqa: F401  (re-exported)

C1_CSV = C1 / "turns_all_ts.csv"
C2_CSV = C2 / "omega_advancement_test_v1_turns.csv"
C3_LOGS = C3 / "logs"                       # directory of <game_id>.json files

TOKENIZER = "NousResearch/Llama-2-7b-hf"    # K = 32,000
