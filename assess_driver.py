#!/usr/bin/env python3
"""Recommended entrypoint: dual-mode driver assessment.

Runs the exact published score when disorder input is available, otherwise
falls back to the lightweight AF3-only or sequence-only surrogate.

Usage:
    ./assess_driver.py --help
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

if __name__ == "__main__":
    runpy.run_path(str(_HERE / "src" / "assess_dual_mode.py"), run_name="__main__")
