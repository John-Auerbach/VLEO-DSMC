#!/bin/bash
#SBATCH --job-name=sparta_convert
#SBATCH --account=read_crch_l1sgb100
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=15:00:00
#SBATCH --mem=0 # Use all memory on the node
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

# Stage 1 of post-processing: convert SPARTA .dat dumps -> Parquet.
# Run AFTER the simulation job has finished writing dumps.
#
# Usage:
#   sbatch tools/load_job_dumps.sh [dumps_dir] [extra load_dumps.py args...]
# Default dumps_dir is "dumps".
#
# Conversion is incremental by default: frames whose Parquet already exists are
# skipped, so you can run this mid-simulation for prelim results then re-run
# later to convert only the new frames. Pass --force to re-convert everything:
#   sbatch tools/load_job_dumps.sh dumps/esasat --force
#
# This stage is the heavy one: it needs many cores (parallel per-file
# conversion) and the himem node to handle large frames. Run
# scripts/analyze_job.sh afterwards once this finishes.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DUMPS_DIR="${1:-dumps}"
# Any further args (e.g. --force) are passed straight through to load_dumps.py.
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")

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
# Particle frames are huge (~25-30 GB each in memory for a ~487M-particle dump),
# so converting many at once OOMs even the himem node. himem has ~1 TB, so ~20
# concurrent particle frames (~600 GB peak) is fast and stays well under memory;
# grid/surf frames are small and use the full NJOBS.
# Override via env: PARTICLE_NJOBS=8 sbatch tools/load_job_dumps.sh
# Particle frames are ~33 GB each and np.loadtxt peaks at ~2-3x that while
# parsing, so each worker needs ~80-100 GB. On a ~1 TB himem node, keep this
# at ~8 to stay well clear of OOM (20 caused OUT_OF_MEMORY / exit 0:125).
PARTICLE_NJOBS="${PARTICLE_NJOBS:-8}"
if (( PARTICLE_NJOBS > NJOBS )); then PARTICLE_NJOBS=$NJOBS; fi
python3 tools/load_dumps.py "$DUMPS_DIR" -j "$NJOBS" --particle-jobs "$PARTICLE_NJOBS" "${EXTRA_ARGS[@]}"

echo "=== Conversion complete. Now run: sbatch scripts/analyze_job.sh $DUMPS_DIR ==="
