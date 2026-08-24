
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLASSES, DRIVER_SCORE_THRESHOLD

CLASS_ORDER = list(CLASSES)
AA_FRACTION_COLUMNS = [f"frac_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
STRICT_NONDEF_BASE_FEATURES = [
    "helix_fraction",
    "coil_fraction",
    "tm_fraction",
    "tm_count",
    "sp_fraction",
    "has_signal_peptide",
    "lc_fraction",
    "homorepeat_fraction",
    "simple_repeat_fraction",
    "global_entropy",
    "mean_hydropathy",
    "net_charge",
    "pos_charge_frac",
    "neg_charge_frac",
    "seq_length",
    "helix_frac_n",
    "helix_frac_center",
    "helix_frac_c",
    "coil_frac_n",
    "coil_frac_center",
    "coil_frac_c",
    "tm_frac_n",
    "tm_frac_center",
    "tm_frac_c",
    "hydrophobic_frac",
    "charged_frac",
    "polar_frac",
]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in AA_FRACTION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["hydrophobic_frac"] = out[
        [c for c in ["frac_A", "frac_V", "frac_I", "frac_L", "frac_M", "frac_F", "frac_W", "frac_Y", "frac_C"] if c in out.columns]
    ].sum(axis=1)
    out["charged_frac"] = out[
        [c for c in ["frac_D", "frac_E", "frac_K", "frac_R"] if c in out.columns]
    ].sum(axis=1)
    out["polar_frac"] = out[
        [c for c in ["frac_S", "frac_T", "frac_N", "frac_Q", "frac_H"] if c in out.columns]
    ].sum(axis=1)
    return out


def identify_driver_subset(
    prot: pd.DataFrame,
    class_order: list[str] | None = None,
    score_threshold: float = DRIVER_SCORE_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for cls in class_order or CLASS_ORDER:
        sub = prot[prot["class"] == cls].copy()
        keep = sub.dropna(subset=["mean_plddt", "mean_disorder", "strand_fraction"])
        zp = (keep["mean_plddt"] - keep["mean_plddt"].mean()) / keep["mean_plddt"].std(ddof=0)
        zd = (keep["mean_disorder"] - keep["mean_disorder"].mean()) / keep["mean_disorder"].std(ddof=0)
        zs = (keep["strand_fraction"] - keep["strand_fraction"].mean()) / keep["strand_fraction"].std(ddof=0)
        keep["driver_score"] = zp * zd + zp * (-zs)
        keep["is_driver"] = keep["driver_score"] >= score_threshold
        keep["driver_group"] = np.where(keep["is_driver"], "driver", "non-driver")
        rows.append(keep)
    return pd.concat(rows, ignore_index=True)


def strict_nondef_features(df: pd.DataFrame) -> list[str]:
    strict = STRICT_NONDEF_BASE_FEATURES + AA_FRACTION_COLUMNS
    return [c for c in strict if c in df.columns]


def benchmark_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    strict = strict_nondef_features(df)
    ultra = [
        "tm_fraction",
        "tm_count",
        "sp_fraction",
        "has_signal_peptide",
        "lc_fraction",
        "homorepeat_fraction",
        "simple_repeat_fraction",
        "global_entropy",
        "mean_hydropathy",
        "net_charge",
        "pos_charge_frac",
        "neg_charge_frac",
        "seq_length",
        "hydrophobic_frac",
        "charged_frac",
        "polar_frac",
    ] + AA_FRACTION_COLUMNS
    architecture = [
        "helix_fraction",
        "coil_fraction",
        "tm_fraction",
        "tm_count",
        "sp_fraction",
        "has_signal_peptide",
        "seq_length",
        "helix_frac_n",
        "helix_frac_center",
        "helix_frac_c",
        "coil_frac_n",
        "coil_frac_center",
        "coil_frac_c",
        "tm_frac_n",
        "tm_frac_center",
        "tm_frac_c",
    ]
    composition = [
        "lc_fraction",
        "homorepeat_fraction",
        "simple_repeat_fraction",
        "global_entropy",
        "mean_hydropathy",
        "net_charge",
        "pos_charge_frac",
        "neg_charge_frac",
        "hydrophobic_frac",
        "charged_frac",
        "polar_frac",
    ] + AA_FRACTION_COLUMNS
    return {
        "strict_nondef": strict,
        "ultra_strict": [c for c in ultra if c in df.columns],
        "architecture_only": [c for c in architecture if c in df.columns],
        "composition_only": [c for c in composition if c in df.columns],
    }
