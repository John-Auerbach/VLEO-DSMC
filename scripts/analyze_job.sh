#!/bin/bash
#SBATCH --job-name=sparta_plot
#SBATCH --account=open
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=950G
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

# Stage 2 of post-processing: generate plots and animations from Parquet.
# Run AFTER tools/load_job_dumps.sh has produced the Parquet files.
#
# Usage:
#   sbatch scripts/analyze_job.sh [dumps_dir]
# Default dumps_dir is "dumps".
#
# Plotting reads one Parquet frame at a time. Each animation script can
# precompute frames across several workers (-j); peak memory scales with the
# number of workers, so keep this modest for the large grid frames. Comment/
# uncomment the scripts below as needed.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DUMPS_DIR="${1:-dumps}"
# Parallel workers for per-frame precompute (defaults to the allocated cores).
NJOBS="${SLURM_CPUS_PER_TASK:-1}"
# Particle dumps are ~4-4.4 GB each (parquet); loaded as a DataFrame one frame
# is ~8-10 GB, and pyarrow briefly holds both the Arrow table and the pandas
# copy during the read, so peak is ~2x that per worker. Loading many at once
# OOMs. velocity_heatmap now projects to only x,y,z,vx,vy,vz and downcasts to
# float32 (~halves per-frame RAM), but keep the worker count modest so a few
# concurrent frames stay well within node memory.
PART_NJOBS=2
if (( NJOBS < PART_NJOBS )); then PART_NJOBS=$NJOBS; fi

# Activate the project virtual environment (created at .venv) FIRST, so the
# ffmpeg PATH set up below is not clobbered by the activate script.
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "WARNING: .venv not found; falling back to system python3" >&2
fi

# ffmpeg is needed to write the .mp4 animations. Load the module, then make sure
# it is actually on PATH (the module function does not always update PATH in a
# batch shell). Fall back to the known module install location if needed.
module load ffmpeg/4.3.2 2>/dev/null || true
if ! command -v ffmpeg >/dev/null 2>&1; then
    ff_bin="$(ls -d /swst/apps/ffmpeg/*/bin 2>/dev/null | head -1)"
    [[ -n "$ff_bin" ]] && export PATH="$ff_bin:$PATH"
fi
if command -v ffmpeg >/dev/null 2>&1; then
    echo "Using ffmpeg: $(command -v ffmpeg)"
else
    echo "WARNING: ffmpeg not found; mp4 saving will fail" >&2
fi

# Matplotlib must not try to open a display on a compute node
export MPLBACKEND=Agg

echo "=== Generating plots and animations ==="
# Lightweight stats plots
#python3 scripts/plot_drag.py "$DUMPS_DIR" || echo "plot_drag.py failed (non-fatal)"

# Grid heatmaps (precompute frames in parallel across $NJOBS workers)
python3 scripts/grid_density_heatmap.py  "$DUMPS_DIR" -j "$NJOBS" || echo "grid_density_heatmap.py failed (non-fatal)"
#python3 scripts/grid_pressure_heatmap.py "$DUMPS_DIR" -j "$NJOBS" || echo "grid_pressure_heatmap.py failed (non-fatal)"
python3 scripts/grid_temp_heatmap.py     "$DUMPS_DIR" -j "$NJOBS" || echo "grid_temp_heatmap.py failed (non-fatal)"
#python3 scripts/velocity_heatmap.py      "$DUMPS_DIR" -j "$PART_NJOBS" || echo "velocity_heatmap.py failed (non-fatal)"
python3 scripts/streamlines.py --anim    "$DUMPS_DIR"                 || echo "streamlines.py failed (non-fatal)"

# Surface / particle animations
#python3 scripts/surface_temp_heatmap.py  "$DUMPS_DIR" -j "$NJOBS" || echo "surface_temp_heatmap.py failed (non-fatal)"
#python3 scripts/animate_particles.py     "$DUMPS_DIR" -j "$NJOBS" || echo "animate_particles.py failed (non-fatal)"

echo "=== Plotting complete. Outputs written to outputs/ ==="
