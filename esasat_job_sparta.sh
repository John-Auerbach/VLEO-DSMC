#!/bin/bash
#SBATCH --job-name=sparta_ampt_himem
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=10:00:00
#SBATCH --mem=96G
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err

# Load required modules (must match what SPARTA was compiled with)
module load gcc/14.2.0
module load openmpi/4.1.1-pmi2

# Move to the project directory
cd $SLURM_SUBMIT_DIR

# Run SPARTA with MPI
mpirun -np $SLURM_NTASKS ./sparta -in in.esasat_ROAR
