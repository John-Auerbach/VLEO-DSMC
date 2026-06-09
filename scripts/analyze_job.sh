#!/bin/bash
#SBATCH --job-name=sparta_plot
#SBATCH --account=open
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=950G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Stage 2 of post-processing: generate plots and animations from Parquet.
# Run AFTER tools/load_job_dumps.sh has produced the Parquet files.
#
# Usage:
#   sbatch scripts/analyze_job.sh [dumps_dir]
# Default dumps_dir is "dumps".
#
# Plotting is mostly serial numpy/pandas/matplotlib reading one Parquet frame
# at a time, so it needs few cores but keeps the himem node for the heatmap
# colour-scale passes. Comment/uncomment the scripts below as needed.

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

# Matplotlib must not try to open a display on a compute node
export MPLBACKEND=Agg

echo "=== Generating plots and animations ==="
# Lightweight stats plots
python3 scripts/plot_drag.py || echo "plot_drag.py failed (non-fatal)"

# Grid heatmaps (read Parquet one frame at a time)
python3 scripts/grid_density_heatmap.py  "$DUMPS_DIR" || echo "grid_density_heatmap.py failed (non-fatal)"
#python3 scripts/grid_pressure_heatmap.py "$DUMPS_DIR" || echo "grid_pressure_heatmap.py failed (non-fatal)"
#python3 scripts/grid_temp_heatmap.py     "$DUMPS_DIR" || echo "grid_temp_heatmap.py failed (non-fatal)"
#python3 scripts/velocity_heatmap.py      "$DUMPS_DIR" || echo "velocity_heatmap.py failed (non-fatal)"
#python3 scripts/streamlines.py --anim                 || echo "streamlines.py failed (non-fatal)"

# Surface / particle animations
#python3 scripts/surface_temp_heatmap.py  "$DUMPS_DIR" || echo "surface_temp_heatmap.py failed (non-fatal)"
#python3 scripts/animate_particles.py     "$DUMPS_DIR" || echo "animate_particles.py failed (non-fatal)"

echo "=== Plotting complete. Outputs written to outputs/ ==="
