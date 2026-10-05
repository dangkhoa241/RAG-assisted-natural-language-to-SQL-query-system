"""FastAPI backend for the NL -> SQL assistant. It reuses the modules in src/ instead of reimplementing them."""
import os
import sys
from pathlib import Path

# src/intent.py uses @st.cache_resource, which logs a "bare mode" warning on every call outside
# `streamlit run`. Streamlit reads its log level from this when its config loads.
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
