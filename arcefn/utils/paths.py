"""
Cross-Platform Project Path Resolution
=======================================
Provides a single, robust path anchor for all scripts and modules.

Uses a sentinel-file walk (looks for ``setup.py``) so that it works
from *any* depth in the repository ― no more brittle
``os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`` chains.

Usage
-----
    from arcefn.utils.paths import get_project_root, ensure_dir

    PROJECT_ROOT = get_project_root()
    DATA_DIR = PROJECT_ROOT / 'data' / 'top_tagging'
    RESULTS_DIR = PROJECT_ROOT / 'experiments'
"""
import os
from pathlib import Path


def get_project_root(marker: str = 'setup.py') -> Path:
    """
    Walk up from this utility file until *marker* is found.

    Parameters
    ----------
    marker : str
        Filename that identifies the project root (default ``setup.py``).

    Returns
    -------
    Path
        Absolute ``Path`` to the project root directory.

    Raises
    ------
    FileNotFoundError
        If *marker* is not found between this file and the filesystem root.
    """
    current = Path(__file__).resolve().parent
    for _ in range(50):                     # safety limit
        if (current / marker).exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Could not locate project root: '{marker}' not found "
        f"in any parent of {Path(__file__).resolve()}"
    )


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory to create.

    Returns
    -------
    Path
        The (now extant) directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Convenience constants ───────────────────────────────────────────
# These are computed once on import so every script that imports from
# this module gets correct paths without extra function calls.
# MODEL_DIR can be overridden via the ARCEFN_MODEL_DIR environment variable.

_PROJECT_ROOT = get_project_root()

DATA_DIR      = _PROJECT_ROOT / 'data' / 'top_tagging'
EXPERIMENTS   = _PROJECT_ROOT / 'experiments'
SCRIPTS_DIR   = _PROJECT_ROOT / 'scripts'

# Allow MODEL_DIR override via environment variable
_model_dir_env = os.environ.get('ARCEFN_MODEL_DIR')
if _model_dir_env:
    MODEL_DIR = Path(_model_dir_env).resolve()
else:
    MODEL_DIR = EXPERIMENTS / 'robustness_s16_m05'
CHECKPOINT    = MODEL_DIR / 'checkpoint.pt'
