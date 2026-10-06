"""FastAPI backend for the NL -> SQL assistant. It reuses the modules in src/ instead of reimplementing them."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
