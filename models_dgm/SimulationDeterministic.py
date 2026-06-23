"""Run the shared deterministic simulator with DGM model classes.

The shared simulator is architecture-agnostic except for its imports.  This
wrapper puts ``models_dgm`` first on ``sys.path`` and then executes the shared
script, so imports such as ``PreDamagePreTech`` resolve to the DGM classes.
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
DGM_DIR = Path(__file__).resolve().parent
SHARED_SCRIPT = ROOT / "models" / "SimulationDeterministic.py"

sys.path = [str(DGM_DIR)] + [
    entry for entry in sys.path if Path(entry or ".").resolve() != DGM_DIR
]

runpy.run_path(str(SHARED_SCRIPT), run_name="__main__")
