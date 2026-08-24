#!/usr/bin/env python3
"""Exact-mode driver assessment.

Runs only the exact published driver score. Requires structure-derived pLDDT,
DSSP secondary structure and matching PUNCH2 disorder input. Does not fall
back to any surrogate.

Usage:
    ./assess_driver_exact.py --help
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

if __name__ == "__main__":
    runpy.run_path(str(_HERE / "src" / "assess_exact.py"), run_name="__main__")
