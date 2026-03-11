import sys
from pathlib import Path


# Ensure project root is importable when pytest is launched from any interpreter/entrypoint.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
