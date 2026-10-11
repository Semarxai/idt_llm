"""Repository-relative paths. Import this instead of hard-coding locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
C1 = DATA / "c1"
C2 = DATA / "c2"
C3 = DATA / "c3"
RESULTS = ROOT / "results"
CLOSURE = RESULTS / "closure"
CONTRIB = RESULTS / "contribution"
CALIB = RESULTS / "calibration"
FIGURES = ROOT / "figures"

for _p in (CLOSURE, CONTRIB, CALIB, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Public model identifier; downloaded on first use.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
