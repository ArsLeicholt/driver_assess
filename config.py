"""Configuration for the standalone driver-assessment utility.

This is a stripped-down variant of the analysis-pipeline ``config.py`` that
contains only the constants and paths that ``assess_exact.py`` and
``assess_dual_mode.py`` actually need. It does NOT create any output
directories.

Path resolution order:

1. If ``DRIVER_ASSESS_REFERENCE`` is set, that TSV/CSV/parquet is used as the
   reference ``master_per_protein`` table (recommended for full-Zenodo data).
2. Otherwise, the bundled ``data/master_per_protein_reference.tsv`` in this
   repository is used (installed with the package).
3. If ``PROTEINS_ROOT`` is set (legacy), we also try
   ``$PROTEINS_ROOT/output/tables/final/master_per_protein.tsv`` and
   ``$PROTEINS_ROOT/output/tables/intermediate/``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of the ``src/`` folder that contains this file.
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_BUNDLED_DATA_DIR = _REPO_ROOT / "data"

# --- Reference table (needed by the screening backend) ------------------
# One of: env override, legacy PROTEINS_ROOT layout, bundled reference.
_env_ref = os.environ.get("DRIVER_ASSESS_REFERENCE")
if _env_ref:
    _REFERENCE_TABLE = Path(_env_ref).resolve()
elif os.environ.get("PROTEINS_ROOT"):
    _REFERENCE_TABLE = (
        Path(os.environ["PROTEINS_ROOT"]).resolve()
        / "output" / "tables" / "final" / "master_per_protein.tsv"
    )
else:
    _REFERENCE_TABLE = _BUNDLED_DATA_DIR / "master_per_protein_reference.tsv"

# --- Directories consumed by the two scripts ----------------------------
# ``FINAL_DIR`` is only used as the directory containing the reference
# ``master_per_protein.tsv``. We expose it as the reference table's parent so
# the existing ``from config import FINAL_DIR`` calls resolve without change.
FINAL_DIR = _REFERENCE_TABLE.parent
INTERMEDIATE_DIR = FINAL_DIR  # kept for compatibility with the exact-mode script

# Log directory. Defaults to ``logs/`` under CWD so the tool never writes into
# the installed package directory. Override with ``DRIVER_ASSESS_LOG_DIR``.
_env_logs = os.environ.get("DRIVER_ASSESS_LOG_DIR")
LOG_DIR = Path(_env_logs).resolve() if _env_logs else (Path.cwd() / "logs")

# ``DISORDER_DIRS`` is used by the dual-mode script if the user passes
# ``--class de_novo`` and expects to look up an existing raw PUNCH2 ``.caid``
# file. For the standalone utility we default to ``None`` per class; the user
# must supply ``--disorder-file`` explicitly.
DISORDER_DIRS = {
    "de_novo": None,
    "conserved": None,
    "disprot": None,
    "random": None,
}

# --- DSSP / driver-score constants (identical to the pipeline) ----------
DSSP_BINARY = os.environ.get("DSSP_BINARY", "mkdssp")
DSSP_HELIX_CODES = {"H", "G", "I"}
DSSP_STRAND_CODES = {"E", "B"}
DSSP_COIL_CODES = {"T", "S", " ", "C", "-"}

DRIVER_SCORE_THRESHOLD = 1.0

# --- Class names (order matches the paper) ------------------------------
CLASSES = ["de_novo", "conserved", "disprot", "random"]
CLASS_LABELS = {
    "de_novo": "de novo",
    "conserved": "conserved",
    "disprot": "DisProt",
    "random": "random",
}

RANDOM_SEED = 13073027


def reference_table_path() -> Path:
    """Return the currently resolved reference-table path (for --version banner)."""
    return _REFERENCE_TABLE
