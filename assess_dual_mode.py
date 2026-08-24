#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DISORDER_DIRS, FINAL_DIR
from driver_classifier_utils import add_derived_columns
from driver_screening_utils import (
    SCREENING_THRESHOLDS,
    build_screening_reference,
    fit_screening_models,
    screening_feature_sets,
)
from logging_utils import get_logger

logger = get_logger("33_assess_driver_candidates_dual_mode")

REFERENCE_CLASSES = ["de_novo", "conserved", "disprot", "random"]
SCREEN_BACKENDS = ["sequence_easy", "af3_easy"]


@dataclass
class Candidate:
    candidate_index: int
    protein_id: str
    input_label: str
    source_kind: str
    sequence: str
    structure_target: Optional[object] = None


def _load_module(script_name: str, module_name: str):
    script_path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Structure files, AF3 directories, sequence FASTA files, or directories containing them.",
    )
    parser.add_argument(
        "--mode",
        choices=["exact", "screen", "both"],
        default="both",
        help="Exact published score mode, surrogate screening mode, or both. Default: both.",
    )
    parser.add_argument(
        "--disorder-dir",
        type=Path,
        default=None,
        help="Optional directory containing PUNCH2-style .caid files for exact score mode.",
    )
    parser.add_argument(
        "--dssp-table",
        type=Path,
        default=None,
        help="Optional DSSP-derived table used in exact and AF3-screening mode.",
    )
    parser.add_argument(
        "--reference-classes",
        nargs="+",
        default=REFERENCE_CLASSES,
        choices=REFERENCE_CLASSES,
        help="Reference class contexts used for exact mode and screening probabilities. Default: all four classes.",
    )
    parser.add_argument(
        "--screen-backend",
        choices=["auto", "sequence_easy", "af3_easy"],
        default="auto",
        help="Force the lightweight sequence-only surrogate or the AF3-augmented surrogate. Default: auto.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FINAL_DIR / "driver_candidate_dual_mode_assessment.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: Optional[str] = None
    current_seq: list[str] = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq).upper()))
                current_id = stripped[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(stripped)
    if current_id is not None:
        records.append((current_id, "".join(current_seq).upper()))
    return records


def collect_candidates(inputs: list[str], exact_mod) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    next_index = 0
    for raw in inputs:
        path = Path(raw)
        if exact_mod._looks_like_sequence_file(path):
            for seq_id, sequence in _read_fasta(path):
                key = (seq_id, path.name, "sequence")
                if key in seen:
                    continue
                candidates.append(
                    Candidate(
                        candidate_index=next_index,
                        protein_id=seq_id,
                        input_label=seq_id,
                        source_kind="sequence",
                        sequence=sequence,
                    )
                )
                seen.add(key)
                next_index += 1
            continue
        for target in exact_mod._resolve_structure_target(path):
            sequence, _residue_plddt, _source = exact_mod._mean_b_iso_per_residue(target.structure_path)
            key = (target.protein_id, str(target.structure_path), "structure")
            if key in seen:
                continue
            candidates.append(
                Candidate(
                    candidate_index=next_index,
                    protein_id=target.protein_id,
                    input_label=target.label,
                    source_kind="structure",
                    sequence=sequence,
                    structure_target=target,
                )
            )
            seen.add(key)
            next_index += 1
    if not candidates:
        raise FileNotFoundError("No supported structure or sequence inputs were found.")
    return candidates


def find_project_disorder_dir(target, exact_mod) -> tuple[Optional[Path], Optional[str]]:
    for cls, disorder_dir in DISORDER_DIRS.items():
        disorder_path, note = exact_mod.find_disorder_file(target, disorder_dir)
        if note is None and disorder_path is not None:
            return disorder_dir, cls
    return None, None


def build_sequence_feature_row(candidate: Candidate, complexity_mod) -> dict[str, object]:
    feat = complexity_mod.protein_features(candidate.protein_id, candidate.sequence, "query")
    feat.pop("class", None)
    feat.pop("protein_id", None)
    return feat


def choose_screen_backend(
    row: pd.Series,
    requested_backend: str,
    feature_map: dict[str, list[str]],
) -> tuple[Optional[str], Optional[str]]:
    if requested_backend == "sequence_easy":
        return "sequence_easy", None
    af3_features = feature_map["af3_easy"]
    has_complete_af3 = all(pd.notna(row.get(col)) for col in af3_features)
    if requested_backend == "af3_easy":
        if has_complete_af3:
            return "af3_easy", None
        return None, "Forced af3_easy screening backend but AF3-derived features were incomplete"
    if has_complete_af3:
        return "af3_easy", None
    return "sequence_easy", None


def apply_screening(
    candidate_index: int,
    row: pd.Series,
    backend_name: Optional[str],
    backend_note: Optional[str],
    models_by_backend: dict[str, dict[str, dict[str, object]]],
    reference_classes: list[str],
) -> dict[str, object]:
    out: dict[str, object] = {
        "candidate_index": candidate_index,
        "screening_backend_used": pd.NA,
        "screening_call": "indeterminate",
        "screening_driver_like_class": pd.NA,
        "screening_best_class": pd.NA,
        "screening_best_probability": pd.NA,
        "screening_threshold_for_best_class": pd.NA,
        "screening_status": "screening_unavailable",
        "screening_explanation": backend_note or "Screening backend could not be selected",
    }
    for cls in reference_classes:
        out[f"screen_prob_{cls}"] = pd.NA
    if backend_name is None:
        return out

    backend_models = models_by_backend.get(backend_name, {})
    if not backend_models:
        out["screening_explanation"] = f"No trained screening models were available for backend {backend_name}"
        return out

    probabilities: dict[str, float] = {}
    for cls in reference_classes:
        model_bundle = backend_models.get(cls)
        if model_bundle is None:
            continue
        features = model_bundle["features"]
        if any(pd.isna(row.get(col)) for col in features):
            continue
        X = pd.DataFrame([row])[features].astype(float).to_numpy()
        prob = float(model_bundle["model"].predict_proba(X)[:, 1][0])
        probabilities[cls] = prob
        out[f"screen_prob_{cls}"] = prob

    if not probabilities:
        out["screening_explanation"] = f"Backend {backend_name} was selected but required features were missing"
        return out

    out["screening_backend_used"] = backend_name
    best_class = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_class]
    out["screening_best_class"] = best_class
    out["screening_best_probability"] = best_prob
    out["screening_threshold_for_best_class"] = SCREENING_THRESHOLDS[backend_name][best_class]

    positive_classes = [
        cls
        for cls, prob in probabilities.items()
        if prob >= SCREENING_THRESHOLDS[backend_name][cls]
    ]
    if positive_classes:
        driver_like_class = max(positive_classes, key=lambda cls: probabilities[cls])
        out["screening_call"] = "driver_like"
        out["screening_driver_like_class"] = driver_like_class
        out["screening_status"] = "screening_positive"
        out["screening_explanation"] = (
            f"Surrogate {backend_name} probability crossed the validated {driver_like_class} "
            f"threshold ({probabilities[driver_like_class]:.3f} >= {SCREENING_THRESHOLDS[backend_name][driver_like_class]:.4f})"
        )
    else:
        out["screening_call"] = "not_driver_like"
        out["screening_status"] = "screening_negative"
        out["screening_explanation"] = (
            f"Best surrogate probability remained below the validated class threshold "
            f"({best_prob:.3f} < {SCREENING_THRESHOLDS[backend_name][best_class]:.4f})"
        )
    return out


def add_overall_assessment(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    overall_status = []
    overall_explanation = []
    for _, row in out.iterrows():
        exact_call = row.get("primary_discordance_call", pd.NA)
        screen_call = row.get("screening_call", pd.NA)
        if mode in {"exact", "both"} and exact_call == "discordant":
            overall_status.append("exact_driver")
            overall_explanation.append(row.get("discordance_explanation", "Exact score mode classified the protein as discordant"))
        elif mode in {"exact", "both"} and exact_call == "not_discordant":
            overall_status.append("exact_not_driver")
            overall_explanation.append(row.get("discordance_explanation", "Exact score mode remained below threshold"))
        elif mode in {"screen", "both"} and screen_call == "driver_like":
            overall_status.append("screening_driver_like_only")
            overall_explanation.append(row.get("screening_explanation", "Surrogate screening classified the protein as driver-like"))
        elif mode in {"screen", "both"} and screen_call == "not_driver_like":
            overall_status.append("screening_not_driver_like")
            overall_explanation.append(row.get("screening_explanation", "Surrogate screening classified the protein as not driver-like"))
        else:
            overall_status.append("indeterminate")
            overall_explanation.append("Neither exact mode nor surrogate screening produced a positive driver call")
    out["overall_assessment"] = overall_status
    out["overall_explanation"] = overall_explanation
    return out


def main() -> None:
    args = parse_args()
    exact_mod = _load_module("31_assess_driver_candidates.py", "driver_score_assessment")
    complexity_mod = _load_module("05_compute_complexity.py", "driver_complexity_features")

    logger.info("Collecting sequence and structure inputs")
    candidates = collect_candidates(args.inputs, exact_mod)
    logger.info("Collected %d candidate inputs", len(candidates))

    dssp_summary_df, dssp_source_label = exact_mod.load_dssp_summary_table(args.dssp_table)
    logger.info("DSSP source for dual-mode tool: %s", dssp_source_label)

    screening_feature_map: dict[str, list[str]] = {}
    models_by_backend: dict[str, dict[str, dict[str, object]]] = {}
    if args.mode in {"screen", "both"}:
        reference_master = pd.read_parquet(FINAL_DIR / "master_per_protein.parquet")
        reference_screen = build_screening_reference(reference_master)
        screening_feature_map = screening_feature_sets(reference_screen)
        backends_to_fit = SCREEN_BACKENDS if args.screen_backend == "auto" else [args.screen_backend]
        for backend_name in backends_to_fit:
            models_by_backend[backend_name] = fit_screening_models(reference_screen, backend_name, args.reference_classes)
            logger.info(
                "Fitted %s screening models for %d classes",
                backend_name,
                len(models_by_backend[backend_name]),
            )

    base_rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    screening_inputs: dict[int, dict[str, object]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        empty_disorder_dir = Path(tmp_dir_name)

        for candidate in candidates:
            base_row = {
                "candidate_index": candidate.candidate_index,
                "protein_id": candidate.protein_id,
                "input_label": candidate.input_label,
                "source_kind": candidate.source_kind,
                "seq_length": len(candidate.sequence),
            }
            seq_feat = build_sequence_feature_row(candidate, complexity_mod)
            screening_inputs[candidate.candidate_index] = seq_feat.copy()
            screening_inputs[candidate.candidate_index]["candidate_index"] = candidate.candidate_index

            if candidate.source_kind == "structure":
                if args.disorder_dir is not None:
                    disorder_dir_to_use = args.disorder_dir
                    exact_disorder_mode = "user_disorder_dir"
                    exact_project_class = pd.NA
                else:
                    project_disorder_dir, project_class = find_project_disorder_dir(candidate.structure_target, exact_mod)
                    if project_disorder_dir is not None:
                        disorder_dir_to_use = project_disorder_dir
                        exact_disorder_mode = "project_disorder_lookup"
                        exact_project_class = project_class
                    else:
                        disorder_dir_to_use = empty_disorder_dir
                        exact_disorder_mode = "no_disorder_available"
                        exact_project_class = pd.NA
                core_row = exact_mod.extract_candidate_features(
                    candidate.structure_target,
                    disorder_dir_to_use,
                    dssp_summary_df,
                    dssp_source_label,
                )
                core_row["candidate_index"] = candidate.candidate_index
                core_row["exact_disorder_mode"] = exact_disorder_mode
                core_row["exact_project_disorder_class"] = exact_project_class
                exact_rows.append(core_row)
                screening_inputs[candidate.candidate_index].update(
                    {
                        "mean_plddt": core_row.get("mean_plddt"),
                        "helix_fraction": core_row.get("helix_fraction"),
                        "strand_fraction": core_row.get("strand_fraction"),
                        "coil_fraction": core_row.get("coil_fraction"),
                        "secondary_structure_available": core_row.get("secondary_structure_available"),
                        "secondary_structure_source": core_row.get("secondary_structure_source"),
                        "screening_notes": core_row.get("notes", ""),
                    }
                )
            else:
                screening_inputs[candidate.candidate_index]["screening_notes"] = "Sequence-only input"
            base_rows.append(base_row)

    base_df = pd.DataFrame(base_rows)

    if args.mode in {"exact", "both"} and exact_rows:
        exact_df = pd.DataFrame(exact_rows)
        reference_df = exact_mod.load_reference_master()
        exact_df = exact_mod.add_driver_scores(exact_df, reference_df, args.reference_classes)
        exact_df = exact_mod.add_driver_summary(exact_df, args.reference_classes)
        exact_cols = [
            col
            for col in exact_df.columns
            if col not in {"candidate_index", "input_label", "protein_id", "seq_length"}
        ]
        base_df = base_df.merge(exact_df[["candidate_index"] + exact_cols], on="candidate_index", how="left")
    if args.mode in {"exact", "both"}:
        seq_only_mask = base_df["source_kind"] == "sequence"
        base_df.loc[seq_only_mask, "primary_discordance_call"] = "unavailable_sequence_only"
        base_df.loc[seq_only_mask, "combined_discordance_status"] = "exact_mode_requires_structure"
        base_df.loc[seq_only_mask, "discordance_explanation"] = (
            "Exact published score mode requires AF3-like structure input because mean pLDDT and strand fraction are unavailable from sequence alone"
        )

    if args.mode in {"screen", "both"}:
        screening_rows: list[dict[str, object]] = []
        for candidate_index, payload in screening_inputs.items():
            candidate_df = add_derived_columns(pd.DataFrame([payload]))
            screening_row = candidate_df.iloc[0]
            backend_name, backend_note = choose_screen_backend(
                screening_row,
                args.screen_backend,
                screening_feature_map,
            )
            screening_rows.append(
                apply_screening(
                    candidate_index=candidate_index,
                    row=screening_row,
                    backend_name=backend_name,
                    backend_note=backend_note,
                    models_by_backend=models_by_backend,
                    reference_classes=args.reference_classes,
                )
            )
        screening_df = pd.DataFrame(screening_rows)
        base_df = base_df.merge(screening_df, on="candidate_index", how="left")

    base_df = add_overall_assessment(base_df, args.mode)
    base_df = base_df.sort_values(["source_kind", "protein_id", "input_label"]).reset_index(drop=True)
    base_df = base_df.drop(columns=["candidate_index"], errors="ignore")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(args.out, sep="\t", index=False)
    logger.info("Wrote %s", args.out)

    summary_cols = [
        "protein_id",
        "source_kind",
        "primary_discordance_call",
        "score_defined_class",
        "best_driver_score",
        "screening_backend_used",
        "screening_call",
        "screening_driver_like_class",
        "screening_best_probability",
        "overall_assessment",
    ]
    present = [col for col in summary_cols if col in base_df.columns]
    print(base_df[present].to_string(index=False))


if __name__ == "__main__":
    main()
