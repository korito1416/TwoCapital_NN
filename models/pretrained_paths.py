"""Shared paths for bundled pretrained checkpoints."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = REPO_ROOT / "pretrained"
LEGACY_NBER_RELATIVE = PRETRAINED_ROOT / "nber_legacy"
LEGACY_NBER_ABSOLUTE = Path(
    "/project/lhansen/Cap_NN_oldVersion/November_version_NewParameters/output/"
    "Novem_NewParaters_0.01_LR_piecewiseconstant_10e-5,10e-5,10e-5,10e-5_128_neurons_32_"
    "#HiddenLayer_4_logxi_-3.0_logximax_5.0_num_iterations2000000"
)


def legacy_nber_folder(required=True):
    """Return the legacy NBER warm-start folder, preferring repo-local weights."""

    candidates = []
    env_path = os.environ.get("NBER_PRETRAINED_FOLDER")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([LEGACY_NBER_RELATIVE, LEGACY_NBER_ABSOLUTE])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if not required:
        return None

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Missing legacy NBER warm-start checkpoints. Expected one of:\n"
        f"{searched}\n"
        "Set NBER_PRETRAINED_FOLDER or restore pretrained/nber_legacy."
    )
