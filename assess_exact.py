#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import gemmi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DRIVER_SCORE_THRESHOLD,
    DSSP_BINARY,
    DSSP_HELIX_CODES,
    DSSP_STRAND_CODES,
    FINAL_DIR,
    INTERMEDIATE_DIR,
)
from logging_utils import get_logger  # noqa: E402

logger = get_logger("31_assess_driver_candidates")

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
REFERENCE_CLASSES = ["de_novo", "conserved", "disprot", "random"]
STRUCTURE_SUFFIXES = {".cif", ".mmcif", ".pdb"}
SEQUENCE_SUFFIXES = {".fa", ".faa", ".fasta", ".fsa", ".seq"}


@dataclass
class Target:
    protein_id: str
    label: str
    structure_path: Path
    confidence_json: Optional[Path] = None


def _clean_structure_stem(stem: str) -> str:
    return re.sub(r"_model$", "", stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess whether new structures satisfy the published score-defined driver-protein rule."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Structure files, AF3 protein directories, or directories containing multiple structures.",
    )
    parser.add_argument(
        "--disorder-dir",
        type=Path,
        required=True,
        help="Directory containing PUNCH2-style .caid files matched by protein_id or file stem.",
    )
    parser.add_argument(
        "--reference-classes",
        nargs="+",
        default=REFERENCE_CLASSES,
        choices=REFERENCE_CLASSES,
        help="Reference class contexts used for within-class standardization. Default: all four classes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FINAL_DIR / "driver_candidate_assessment.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--dssp-table",
        type=Path,
        default=None,
        help=(
            "Optional DSSP-derived table to use instead of on-the-fly DSSP assignment. "
            "Accepted formats: parquet, TSV, CSV. The table may be protein-level "
            "(helix_fraction/strand_fraction/...) or residue-level "
            "(protein_id with ss3 or ss_code)."
        ),
    )
    return parser.parse_args()


def _extract_protein_id(name: str) -> str:
    name = re.sub(r"___.*$", "", name)
    name = re.sub(r"^[A-Za-z]+_\d+_", "", name)
    return name


def _resolve_structure_target(path: Path) -> list[Target]:
    if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES:
        protein_id = _extract_protein_id(_clean_structure_stem(path.stem))
        return [Target(protein_id=protein_id, label=path.stem, structure_path=path)]

    if not path.is_dir():
        return []

    if (path / "ranking_scores.csv").exists():
        return [_af3_inner_dir_target(path)]

    inner = path / path.name.lower()
    if inner.is_dir() and (inner / "ranking_scores.csv").exists():
        return [_af3_top_dir_target(path)]
    if inner.is_dir():
        flat_model = _flat_model_target(path, inner)
        if flat_model is not None:
            return [flat_model]

    af3_children = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        child_inner = child / child.name.lower()
        if child_inner.is_dir() and (child_inner / "ranking_scores.csv").exists():
            af3_children.append(_af3_top_dir_target(child))
        elif (child / "ranking_scores.csv").exists():
            af3_children.append(_af3_inner_dir_target(child))
        elif (child / child.name.lower()).is_dir():
            flat_model = _flat_model_target(child, child / child.name.lower())
            if flat_model is not None:
                af3_children.append(flat_model)
    if af3_children:
        return af3_children

    direct_models = sorted(path.glob("*_model.cif"))
    if direct_models:
        targets = []
        for model_path in direct_models:
            conf_path = model_path.with_name(model_path.name.replace("_model.cif", "_confidences.json"))
            protein_id = _extract_protein_id(_clean_structure_stem(model_path.stem))
            targets.append(
                Target(
                    protein_id=protein_id,
                    label=model_path.stem,
                    structure_path=model_path,
                    confidence_json=conf_path if conf_path.exists() else None,
                )
            )
        return targets

    targets = []
    for structure in sorted(path.rglob("*")):
        if structure.is_file() and structure.suffix.lower() in STRUCTURE_SUFFIXES:
            if any(part.startswith("seed-") for part in structure.parts):
                continue
            targets.append(Target(protein_id=structure.stem, label=structure.stem, structure_path=structure))
    return targets


def _looks_like_sequence_file(path: Path) -> bool:
    if path.suffix.lower() in SEQUENCE_SUFFIXES:
        return True
    if not path.is_file() or path.suffix.lower() not in {"", ".txt"}:
        return False
    try:
        with open(path) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    return stripped.startswith(">")
    except OSError:
        return False
    return False


def _choose_best_sample_dir(inner_dir: Path) -> Path:
    ranking_csv = inner_dir / "ranking_scores.csv"
    sample_dirs = sorted(inner_dir.glob("seed-*_sample-*"))
    if not sample_dirs:
        raise FileNotFoundError(f"No AF3 sample directories found in {inner_dir}")
    if ranking_csv.exists():
        with open(ranking_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and {"seed", "sample", "ranking_score"}.issubset(rows[0]):
            best = max(rows, key=lambda row: float(row["ranking_score"]))
            candidate = inner_dir / f"seed-{int(best['seed'])}_sample-{int(best['sample'])}"
            if candidate.exists():
                return candidate
    return sample_dirs[0]


def _af3_top_dir_target(top_dir: Path) -> Target:
    inner_dir = top_dir / top_dir.name.lower()
    sample_dir = _choose_best_sample_dir(inner_dir)
    structure_path = sample_dir / "model.cif"
    confidence_path = sample_dir / "confidences.json"
    protein_id = _extract_protein_id(top_dir.name)
    return Target(
        protein_id=protein_id,
        label=top_dir.name,
        structure_path=structure_path,
        confidence_json=confidence_path if confidence_path.exists() else None,
    )


def _af3_inner_dir_target(inner_dir: Path) -> Target:
    sample_dir = _choose_best_sample_dir(inner_dir)
    structure_path = sample_dir / "model.cif"
    confidence_path = sample_dir / "confidences.json"
    protein_id = _extract_protein_id(inner_dir.name)
    return Target(
        protein_id=protein_id,
        label=inner_dir.name,
        structure_path=structure_path,
        confidence_json=confidence_path if confidence_path.exists() else None,
    )


def _flat_model_target(top_dir: Path, inner_dir: Path) -> Optional[Target]:
    model_files = sorted(inner_dir.glob("*_model.cif"))
    if not model_files:
        return None
    model_path = model_files[0]
    conf_path = model_path.with_name(model_path.name.replace("_model.cif", "_confidences.json"))
    protein_id = _extract_protein_id(top_dir.name)
    return Target(
        protein_id=protein_id,
        label=top_dir.name,
        structure_path=model_path,
        confidence_json=conf_path if conf_path.exists() else None,
    )


def collect_targets(inputs: list[str]) -> list[Target]:
    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    for raw in inputs:
        path = Path(raw)
        if _looks_like_sequence_file(path):
            raise ValueError(
                f"Sequence-only input is not supported by the current driver workflow: {path}. "
                "Provide a structure file or AF3 output directory together with disorder predictions."
            )
        for target in _resolve_structure_target(path):
            key = (target.protein_id, str(target.structure_path))
            if key not in seen:
                targets.append(target)
                seen.add(key)
    if not targets:
        raise FileNotFoundError("No supported structure inputs were found.")
    return targets


def _mean_b_iso_per_residue(structure_path: Path) -> tuple[str, np.ndarray, str]:
    structure = gemmi.read_structure(str(structure_path))
    model = structure[0]
    sequence = []
    residue_plddt = []
    for chain in model:
        for residue in chain:
            aa = AA3TO1.get(residue.name.upper())
            if aa is None:
                continue
            atom_b = [atom.b_iso for atom in residue if not math.isnan(atom.b_iso)]
            if not atom_b:
                continue
            sequence.append(aa)
            residue_plddt.append(float(np.mean(atom_b)))
    if not sequence:
        raise ValueError(f"No standard amino-acid residues found in {structure_path}")
    return "".join(sequence), np.array(residue_plddt, dtype=float), "b_iso"


def _pydssp_available() -> bool:
    try:
        import pydssp  # noqa: F401
        return True
    except Exception:
        return False


def _mkdssp_available() -> bool:
    return shutil.which(DSSP_BINARY) is not None and importlib.util.find_spec("Bio.PDB") is not None


def _ss_state_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "coil"
    text = str(value).strip()
    lowered = text.lower()
    upper = text.upper()
    if lowered in {"helix", "h"} or upper in DSSP_HELIX_CODES:
        return "helix"
    if lowered in {"strand", "sheet", "beta", "e", "b"} or upper in DSSP_STRAND_CODES:
        return "strand"
    return "coil"


def run_pydssp(structure_path: Path) -> Optional[np.ndarray]:
    if not _pydssp_available():
        return None
    try:
        import pydssp

        structure = gemmi.read_structure(str(structure_path))
        pdb_text = structure.make_pdb_string()
        backbone = pydssp.read_pdbtext(pdb_text)
        coords = np.array(backbone)
        if coords.size == 0:
            return None
        ss = pydssp.assign(coords, out_type="c3")
        return np.array(ss)
    except Exception as exc:
        logger.warning("pydssp failed for %s: %s", structure_path, exc)
        return None


def _mkdssp_code_to_c3(code: str) -> str:
    if code in DSSP_HELIX_CODES:
        return "H"
    if code in DSSP_STRAND_CODES:
        return "E"
    return "C"


def run_mkdssp(structure_path: Path) -> Optional[np.ndarray]:
    if not _mkdssp_available():
        return None
    tmp_path: Optional[Path] = None
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        from Bio.PDB.Polypeptide import is_aa

        structure = gemmi.read_structure(str(structure_path))
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            handle.write(structure.make_pdb_string())
            tmp_path = Path(handle.name)

        parser = PDBParser(QUIET=True)
        bio_structure = parser.get_structure("query", str(tmp_path))
        model = bio_structure[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="parse error at line 1: This file does not seem to be an mmCIF file",
            )
            dssp = DSSP(model, str(tmp_path), dssp=DSSP_BINARY)

        c3_codes = []
        missing_keys = 0
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                key = (chain.id, residue.id)
                if key not in dssp:
                    missing_keys += 1
                    continue
                c3_codes.append(_mkdssp_code_to_c3(dssp[key][2]))
        if missing_keys:
            logger.warning(
                "mkdssp missing %d standard residues for %s; result may not align exactly",
                missing_keys,
                structure_path,
            )
        if not c3_codes:
            return None
        return np.array(c3_codes)
    except Exception as exc:
        logger.warning("mkdssp failed for %s: %s", structure_path, exc)
        return None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for DSSP input: {path}")


def summarise_residue_dssp_table(df: pd.DataFrame) -> pd.DataFrame:
    residue_df = df.copy()
    if "protein_id" not in residue_df.columns:
        raise ValueError("Residue-level DSSP table requires a protein_id column.")
    if "ss3" in residue_df.columns:
        residue_df["ss_label"] = residue_df["ss3"].map(_ss_state_label)
    elif "ss_code" in residue_df.columns:
        residue_df["ss_label"] = residue_df["ss_code"].map(_ss_state_label)
    else:
        raise ValueError("Residue-level DSSP table requires either ss3 or ss_code values.")
    if "class" not in residue_df.columns:
        residue_df["class"] = pd.NA
    out = (
        residue_df
        .groupby(["class", "protein_id"])
        .agg(
            helix_fraction=("ss_label", lambda x: (x == "helix").mean()),
            strand_fraction=("ss_label", lambda x: (x == "strand").mean()),
            coil_fraction=("ss_label", lambda x: (x == "coil").mean()),
        )
        .reset_index()
    )
    return out


def load_dssp_summary_table(dssp_table_path: Optional[Path]) -> tuple[pd.DataFrame, str]:
    source_path = dssp_table_path
    source_label = "user_dssp_table"
    if source_path is None:
        protein_default = INTERMEDIATE_DIR / "per_protein_dssp.parquet"
        residue_default = INTERMEDIATE_DIR / "per_residue_dssp.parquet"
        if protein_default.exists():
            source_path = protein_default
            source_label = "project_per_protein_dssp"
        elif residue_default.exists():
            source_path = residue_default
            source_label = "project_per_residue_dssp"
        else:
            return pd.DataFrame(), "none"
    if not source_path.exists():
        raise FileNotFoundError(f"DSSP table not found: {source_path}")

    raw = read_table_auto(source_path)
    if {"protein_id", "helix_fraction", "strand_fraction"}.issubset(raw.columns):
        out = raw.copy()
        if "coil_fraction" not in out.columns:
            out["coil_fraction"] = 1.0 - out["helix_fraction"] - out["strand_fraction"]
        if "class" not in out.columns:
            out["class"] = pd.NA
        return out[["class", "protein_id", "helix_fraction", "strand_fraction", "coil_fraction"]], source_label
    if "protein_id" in raw.columns and ("ss3" in raw.columns or "ss_code" in raw.columns):
        return summarise_residue_dssp_table(raw), source_label
    raise ValueError(
        f"DSSP table {source_path} is not recognized. "
        "Expected protein-level helix/strand fractions or residue-level ss3/ss_code values."
    )


def find_disorder_file(target: Target, disorder_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    if not disorder_dir.exists():
        return None, f"Disorder directory not found: {disorder_dir}"
    candidates = [
        disorder_dir / f"{target.protein_id}.caid",
        disorder_dir / f"{target.label}.caid",
        disorder_dir / f"{target.structure_path.stem}.caid",
        disorder_dir / f"{_clean_structure_stem(target.structure_path.stem)}.caid",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, None
    return None, "No disorder file matched"


def parse_disorder(disorder_path: Optional[Path]) -> tuple[dict[str, float], list[str]]:
    if disorder_path is None:
        return {"mean_disorder": np.nan, "disorder_fraction": np.nan}, []
    scores = []
    binaries = []
    with open(disorder_path) as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            binary = int(parts[3]) if len(parts) >= 4 else int(score >= 0.5)
            scores.append(score)
            binaries.append(binary)
    if not scores:
        return {"mean_disorder": np.nan, "disorder_fraction": np.nan}, [f"Disorder file was empty or unparsable: {disorder_path.name}"]
    return {
        "mean_disorder": float(np.mean(scores)),
        "disorder_fraction": float(np.mean(binaries)),
    }, []


def extract_candidate_features(
    target: Target,
    disorder_dir: Path,
    dssp_summary_df: pd.DataFrame,
    dssp_source_label: str,
) -> dict[str, object]:
    notes: list[str] = []
    sequence, residue_plddt, plddt_source = _mean_b_iso_per_residue(target.structure_path)
    row: dict[str, object] = {
        "protein_id": target.protein_id,
        "input_label": target.label,
        "structure_path": str(target.structure_path),
        "seq_length": len(sequence),
        "plddt_source": plddt_source,
        "mean_plddt": float(np.mean(residue_plddt)),
        "secondary_structure_available": False,
        "secondary_structure_source": pd.NA,
        "disorder_available": False,
        "disorder_source": pd.NA,
    }

    dssp_row = None
    if not dssp_summary_df.empty:
        match = dssp_summary_df[dssp_summary_df["protein_id"] == target.protein_id]
        if len(match) == 1:
            dssp_row = match.iloc[0]
        elif len(match) > 1:
            notes.append(f"Multiple DSSP summary rows matched protein_id={target.protein_id}; falling back from table lookup")

    if dssp_row is not None:
        row.update(
            {
                "helix_fraction": float(dssp_row["helix_fraction"]),
                "strand_fraction": float(dssp_row["strand_fraction"]),
                "coil_fraction": float(dssp_row["coil_fraction"]),
            }
        )
        row["secondary_structure_available"] = True
        row["secondary_structure_source"] = dssp_source_label
    else:
        pydssp_available = _pydssp_available()
        mkdssp_available = _mkdssp_available()
        ss = run_pydssp(target.structure_path) if pydssp_available else None
        ss_source = "pydssp" if ss is not None else None
        if ss is None and mkdssp_available:
            ss = run_mkdssp(target.structure_path)
            if ss is not None:
                ss_source = "mkdssp"
        if ss is None or len(ss) != len(sequence):
            row.update(
                {
                    "helix_fraction": np.nan,
                    "strand_fraction": np.nan,
                    "coil_fraction": np.nan,
                }
            )
            if not dssp_summary_df.empty:
                notes.append(f"No DSSP summary row matched protein_id={target.protein_id}")
            if not pydssp_available:
                notes.append("pydssp unavailable")
            if not mkdssp_available:
                notes.append(f"{DSSP_BINARY} unavailable")
            if pydssp_available and not mkdssp_available:
                notes.append("pydssp failed; DSSP-derived fractions set to NaN")
            elif mkdssp_available and not pydssp_available:
                notes.append(f"{DSSP_BINARY} failed or returned mismatched residues; DSSP-derived fractions set to NaN")
            elif pydssp_available and mkdssp_available:
                notes.append(f"Both pydssp and {DSSP_BINARY} failed or returned mismatched residues; DSSP-derived fractions set to NaN")
            else:
                notes.append("No on-the-fly DSSP backend available; DSSP-derived fractions set to NaN")
            if ss is not None and len(ss) != len(sequence):
                notes.append(
                    f"DSSP residue count mismatch ({len(ss)} residues in DSSP vs {len(sequence)} in structure); "
                    "DSSP-derived fractions set to NaN"
                )
        else:
            states = np.array([_ss_state_label(code) for code in ss], dtype=object)
            row["secondary_structure_available"] = True
            row["secondary_structure_source"] = ss_source
            row.update(
                {
                    "helix_fraction": float(np.mean(states == "helix")),
                    "strand_fraction": float(np.mean(states == "strand")),
                    "coil_fraction": float(np.mean(states == "coil")),
                }
            )

    disorder_path, disorder_note = find_disorder_file(target, disorder_dir)
    if disorder_note is not None:
        notes.append(disorder_note)
    else:
        row["disorder_available"] = True
        row["disorder_source"] = str(disorder_path)
    disorder_feats, disorder_notes = parse_disorder(disorder_path)
    row.update(disorder_feats)
    notes.extend(disorder_notes)
    row["notes"] = "; ".join(notes) if notes else ""
    return row


def load_reference_master() -> pd.DataFrame:
    path = FINAL_DIR / "master_per_protein.parquet"
    df = pd.read_parquet(path)
    required = {"class", "mean_plddt", "mean_disorder", "strand_fraction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Reference table is missing required columns: {missing}")
    return df


def reference_stats(df: pd.DataFrame, cls: str) -> dict[str, tuple[float, float]]:
    sub = df[df["class"] == cls].dropna(subset=["mean_plddt", "mean_disorder", "strand_fraction"]).copy()
    if sub.empty:
        raise ValueError(f"No complete reference rows available for class {cls}")
    stats = {}
    for col in ["mean_plddt", "mean_disorder", "strand_fraction"]:
        stats[col] = (float(sub[col].mean()), float(sub[col].std(ddof=0)))
    return stats


def add_driver_scores(candidate_df: pd.DataFrame, reference_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    out = candidate_df.copy()
    for cls in classes:
        stats = reference_stats(reference_df, cls)
        mu_p, sd_p = stats["mean_plddt"]
        mu_d, sd_d = stats["mean_disorder"]
        mu_s, sd_s = stats["strand_fraction"]
        z_plddt = []
        z_disorder = []
        z_strand = []
        scores = []
        calls = []
        for _, row in out.iterrows():
            if pd.isna(row.get("mean_disorder")) or pd.isna(row.get("strand_fraction")):
                z_plddt.append(np.nan)
                z_disorder.append(np.nan)
                z_strand.append(np.nan)
                scores.append(np.nan)
                calls.append(pd.NA)
                continue
            zp = (float(row["mean_plddt"]) - mu_p) / sd_p if sd_p else np.nan
            zd = (float(row["mean_disorder"]) - mu_d) / sd_d if sd_d else np.nan
            zs = (float(row["strand_fraction"]) - mu_s) / sd_s if sd_s else np.nan
            score = zp * zd + zp * (-zs)
            z_plddt.append(float(zp))
            z_disorder.append(float(zd))
            z_strand.append(float(zs))
            scores.append(float(score))
            calls.append(bool(score >= DRIVER_SCORE_THRESHOLD) if not np.isnan(score) else pd.NA)
        out[f"z_mean_plddt_{cls}"] = z_plddt
        out[f"z_mean_disorder_{cls}"] = z_disorder
        out[f"z_strand_fraction_{cls}"] = z_strand
        out[f"driver_score_{cls}"] = scores
        out[f"is_driver_{cls}"] = calls

    score_cols = [f"driver_score_{cls}" for cls in classes]
    best_classes = []
    best_scores = []
    for _, row in out[score_cols].iterrows():
        valid = row.dropna()
        if valid.empty:
            best_classes.append(pd.NA)
            best_scores.append(np.nan)
        else:
            best_col = valid.idxmax()
            best_classes.append(best_col.replace("driver_score_", ""))
            best_scores.append(float(valid.max()))
    out["best_driver_score_class"] = best_classes
    out["best_driver_score"] = best_scores
    return out


def add_driver_summary(candidate_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    out = candidate_df.copy()
    score_cols = [f"driver_score_{cls}" for cls in classes]
    out["score_evaluable"] = out[score_cols].notna().any(axis=1)
    out["score_defined_class"] = pd.NA
    out["primary_discordance_call"] = "indeterminate"

    combined_status = []
    explanations = []
    for idx, row in out.iterrows():
        positive_classes = [
            cls
            for cls in classes
            if pd.notna(row[f"is_driver_{cls}"]) and bool(row[f"is_driver_{cls}"])
        ]
        if positive_classes:
            best_cls = max(positive_classes, key=lambda cls: float(row[f"driver_score_{cls}"]))
            out.at[idx, "score_defined_class"] = best_cls
            out.at[idx, "primary_discordance_call"] = "discordant"
            combined_status.append("score_defined_discordant")
            explanations.append(
                f"Driver-defined in {best_cls} context "
                f"(score={float(row[f'driver_score_{best_cls}']):.3f} >= {DRIVER_SCORE_THRESHOLD:.1f})"
            )
            continue

        if bool(row["score_evaluable"]):
            out.at[idx, "primary_discordance_call"] = "not_discordant"
            combined_status.append("score_evaluable_below_threshold")
            explanations.append(
                f"Best score stayed below threshold (max={float(row['best_driver_score']):.3f} < {DRIVER_SCORE_THRESHOLD:.1f})"
            )
            continue

        combined_status.append("indeterminate_missing_required_features")
        explanations.append(
            "Driver score could not be evaluated because disorder or strand_fraction was unavailable"
        )

    out["combined_discordance_status"] = combined_status
    out["discordance_explanation"] = explanations
    return out


def main() -> None:
    args = parse_args()
    logger.info("Collecting structure targets")
    targets = collect_targets(args.inputs)
    logger.info("Collected %d targets", len(targets))

    dssp_summary_df, dssp_source_label = load_dssp_summary_table(args.dssp_table)
    if dssp_source_label == "none":
        logger.info("No DSSP summary table available; secondary-structure fallback depends on pydssp or mkdssp")
    else:
        logger.info(
            "Loaded DSSP summary table from %s with %d proteins",
            dssp_source_label,
            0 if dssp_summary_df.empty else dssp_summary_df["protein_id"].nunique(),
        )

    rows = [
        extract_candidate_features(target, args.disorder_dir, dssp_summary_df, dssp_source_label)
        for target in targets
    ]
    candidate_df = pd.DataFrame(rows)

    reference_df = load_reference_master()
    candidate_df = add_driver_scores(candidate_df, reference_df, args.reference_classes)
    candidate_df = add_driver_summary(candidate_df, args.reference_classes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(args.out, sep="\t", index=False)
    logger.info("Wrote %s", args.out)

    summary_cols = [
        "protein_id",
        "primary_discordance_call",
        "score_defined_class",
        "best_driver_score_class",
        "best_driver_score",
        "combined_discordance_status",
        "discordance_explanation",
        "notes",
    ]
    present = [col for col in summary_cols if col in candidate_df.columns]
    print(candidate_df[present].to_string(index=False))


if __name__ == "__main__":
    main()
