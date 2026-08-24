#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${VENV_DIR:-$repo_root/.venv}"

python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$repo_root/requirements.txt"

if command -v mkdssp >/dev/null 2>&1; then
  printf 'Found mkdssp at %s\n' "$(command -v mkdssp)"
elif command -v brew >/dev/null 2>&1; then
  brew install dssp
else
  printf 'mkdssp is required for on-the-fly DSSP unless you always provide --dssp-table.\n' >&2
  printf 'Install DSSP and ensure mkdssp is on PATH, or set DSSP_BINARY.\n' >&2
fi

printf 'driver_assess environment ready in %s\n' "$venv_dir"
printf 'Activate with: source %s/bin/activate\n' "$venv_dir"
printf 'Optional extra DSSP backend: pip install pydssp\n'
printf 'To point at the full Zenodo reference instead of the bundled subset:\n'
printf '  export DRIVER_ASSESS_REFERENCE=/path/to/zenodo_af/tables/final/master_per_protein.tsv\n'
