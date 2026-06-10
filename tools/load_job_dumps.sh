#!/bin/bash
#SBATCH --job-name=sparta_convert
#SBATCH --account=open
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --mem=950G
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

# Stage 1 of post-processing: convert SPARTA .dat dumps -> Parquet.
# Run AFTER the simulation job has finished writing dumps.
#
# Usage:
#   sbatch tools/load_job_dumps.sh [dumps_dir]
# Default dumps_dir is "dumps".
#
# This stage is the heavy one: it needs many cores (parallel per-file
# conversion) and the himem node to handle large frames. Run
# scripts/analyze_job.sh afterwards once this finishes.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DUMPS_DIR="${1:-dumps}"

# Activate the project virtual environment (created at .venv)
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "WARNING: .venv not found; falling back to system python3" >&2
fi

echo "=== Converting dumps in '$DUMPS_DIR' to Parquet ==="
# Streams each .dat file (memory-safe) and writes one Parquet per timestep.
# Parallelise across the cores SLURM gave us (falls back to 1 if unset).
NJOBS="${SLURM_CPUS_PER_TASK:-1}"
python3 tools/load_dumps.py "$DUMPS_DIR" -j "$NJOBS"

echo "=== Conversion complete. Now run: sbatch scripts/analyze_job.sh $DUMPS_DIR ==="
