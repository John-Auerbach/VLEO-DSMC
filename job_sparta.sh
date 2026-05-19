#!/bin/bash
#SBATCH --job-name=sparta_ampt_himem
#SBATCH --account=open
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=1:00:00
#SBATCH --mem=950G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load required modules (must match what SPARTA was compiled with)
module load gcc/14.2.0
module load openmpi/4.1.1-pmi2

# Move to the project directory
cd $SLURM_SUBMIT_DIR

# Run SPARTA with MPI
mpirun -np $SLURM_NTASKS ./sparta -in in.ampt_box_Roar

# Auto-log the run: parse log.sparta + sacct and update data/ampt_box_log.tsv
# (uses $SLURM_JOB_ID implicitly; safe to skip if python/sacct unavailable)
python3 tools/log_run.py || echo "log_run.py failed (non-fatal)"
