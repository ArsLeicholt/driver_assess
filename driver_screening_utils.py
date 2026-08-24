from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from driver_classifier_utils import (
    AA_FRACTION_COLUMNS,
    add_derived_columns,
    identify_driver_subset,
)

SCREENING_RANDOM_SEED = 20062068
SCREENING_MODEL_NAME = "hist_gb_easy"

SEQUENCE_EASY_FEATURES = [
    "seq_length",
    "mean_hydropathy",
    "net_charge",
    "pos_charge_frac",
    "neg_charge_frac",
    "global_entropy",
    "lc_fraction",
    "homorepeat_fraction",
    "simple_repeat_fraction",
    "hydrophobic_frac",
    "charged_frac",
    "polar_frac",
] + AA_FRACTION_COLUMNS

AF3_EASY_EXTRA_FEATURES = [
    "mean_plddt",
    "helix_fraction",
    "strand_fraction",
    "coil_fraction",
]

SCREENING_THRESHOLDS = {
    "sequence_easy": {
        "de_novo": 0.2680,
        "conserved": 0.1601,
        "disprot": 0.1665,
        "random": 0.2851,
    },
    "af3_easy": {
        "de_novo": 0.3514,
        "conserved": 0.2592,
        "disprot": 0.0572,
        "random": 0.3942,
    },
}


def screening_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    seq = [col for col in SEQUENCE_EASY_FEATURES if col in df.columns]
    af3 = seq + [col for col in AF3_EASY_EXTRA_FEATURES if col in df.columns]
    return {
        "sequence_easy": seq,
        "af3_easy": af3,
    }


def screening_model(seed: int = SCREENING_RANDOM_SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=150,
        min_samples_leaf=20,
        random_state=seed,
    )


def build_screening_reference(master_df: pd.DataFrame) -> pd.DataFrame:
    ref = add_derived_columns(master_df.copy())
    if "is_driver" not in ref.columns or "driver_score" not in ref.columns:
        ref = identify_driver_subset(ref)
    return ref


def fit_screening_models(
    reference_df: pd.DataFrame,
    backend_name: str,
    classes: list[str],
) -> dict[str, dict[str, object]]:
    feature_map = screening_feature_sets(reference_df)
    if backend_name not in feature_map:
        raise ValueError(f"Unknown screening backend: {backend_name}")
    features = feature_map[backend_name]
    models: dict[str, dict[str, object]] = {}
    for cls in classes:
        sub = reference_df[reference_df["class"] == cls].dropna(subset=features).copy()
        if sub.empty:
            continue
        X = sub[features].astype(float).to_numpy()
        y = sub["is_driver"].astype(int).to_numpy()
        model = screening_model()
        model.fit(X, y)
        models[cls] = {
            "model": model,
            "features": features,
            "threshold": SCREENING_THRESHOLDS[backend_name][cls],
            "n_samples": int(len(sub)),
            "driver_fraction": float(sub["is_driver"].mean()),
        }
    return models
