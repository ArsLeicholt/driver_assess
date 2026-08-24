# driver_assess

Standalone utility for testing whether a protein satisfies the score-based **driver definition** from Eicholt & Middendorf (2026).

Given AlphaFold3 confidence (pLDDT), PUNCH2 disorder and DSSP-derived strand fraction, the driver score is computed within each class as

```
driver_score = z(mean_pLDDT) * z(mean_disorder) + z(mean_pLDDT) * (-z(strand_fraction))
```

and a protein is called a **driver** when the score is ≥ 1.0. Driver proteins are the ones that concentrate the discordance between AlphaFold3 and PUNCH2 in random and *de novo* sequence sets. See the paper for full context.

This repository provides two entrypoints:

| Entrypoint | Mode | Inputs it needs |
|---|---|---|
| `./assess_driver.py` (recommended) | Dual-mode: exact score when disorder is available, lightweight surrogate otherwise | FASTA, AF3 CIF/PDB, and/or PUNCH2 `.caid` |
| `./assess_driver_exact.py` | Exact published score only | pLDDT + DSSP + PUNCH2 `.caid` |

## Install (macOS)

```bash
git clone https://github.com/ArsLeicholt/driver_assess.git
cd driver_assess
bash setup_macos.sh
source .venv/bin/activate
```

Linux users can adapt `setup_macos.sh` (only difference: `mkdssp` install line — use `apt install dssp` or equivalent). `mkdssp` is optional for the surrogate mode.

## Usage

### Dual-mode (recommended)

```bash
# Sequence-only surrogate (screen mode auto-selected when no structure/disorder):
./assess_driver.py --mode screen examples/example.fasta

# AF3-only surrogate (structure input, no disorder):
./assess_driver.py --mode screen my_prediction.cif

# Exact published score (needs AF3 structure + PUNCH2 disorder directory):
./assess_driver.py --mode exact --disorder-dir path/to/caid_files/ my_prediction.cif

# Both, when possible:
./assess_driver.py --mode both --disorder-dir path/to/caid_files/ my_prediction.cif
```

### Exact mode only

```bash
./assess_driver_exact.py \
    --disorder-dir path/to/caid_files/ \
    my_prediction.cif
```

Both entrypoints accept multiple inputs (single files, AF3 protein directories, or directories containing many). Output is a per-protein TSV. Results and logs go to your current working directory by default (`./driver_candidate_dual_mode_assessment.tsv` and `./logs/`). Override with `--out` and `DRIVER_ASSESS_LOG_DIR`. Run either entrypoint with `--help` for the full flag list.

## Reference data

A copy of the manuscript per-protein reference table is bundled at `data/master_per_protein.tsv` and `data/master_per_protein.parquet` (~9 MB total, 9,303 proteins across four classes). The surrogate backend uses it to train class-specific lightweight histogram-gradient-boosting models at runtime.

To use a different master table (for example if you re-derive it from a newer Zenodo release), point to it with an environment variable:

```bash
export DRIVER_ASSESS_REFERENCE=/path/to/master_per_protein.tsv
```

Logs go under `./logs/` in your CWD by default. Override with `DRIVER_ASSESS_LOG_DIR=/absolute/path`.

## Backends

- **`sequence_easy`** — sequence composition, charge, hydropathy, entropy, low-complexity, repeat features. Useful mainly as a pre-screen.
- **`af3_easy`** — the same sequence features plus mean pLDDT and global DSSP helix/strand/coil fractions. Strong on internal validation.
- **exact score** — the published driver rule from the paper. Requires disorder input; does not use any classifier.

The surrogate backends are driver-likeness support, not a replacement for the exact published score.

## Related repositories

- Analysis pipeline that generated the paper figures and tables: `github.com/ArsLeicholt/protein-driver-discordance-AF3-PUNCH2` (empty placeholder — final URL to be added on submission).
- Underlying data (FASTA, AF3 rank-0 PDBs, PUNCH2, DSSP, Phobius, complexity, integrated tables) on Zenodo (DOI to be assigned).

## Citation

If you use this tool, please cite:

- Eicholt L.A. & Middendorf L. *A discrete protein subset drives persistent structure–disorder predictor discordance on de novo and random sequences.*, 2026.
- The predecessor paper: Middendorf L. & Eicholt L.A. *Proteins* 92(6):757–767 (2024), doi:[10.1002/prot.26652](https://doi.org/10.1002/prot.26652).

## License

MIT (see `LICENSE`). Bundled reference data is derived from the Zenodo deposit and shared under CC-BY 4.0.
