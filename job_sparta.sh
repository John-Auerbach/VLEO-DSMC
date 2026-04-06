#!/bin/bash
#SBATCH --job-name=sparta_ampt
#SBATCH --account=open
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=04:00:00
#SBATCH --mem=950G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load required modules (must match what SPARTA was compiled with)
module load gcc/14.2.0
module load openmpi/4.1.1-pmi2

# Move to the project directory
cd $SLURM_SUBMIT_DIR

# Run SPARTA with MPI
mpirun -np $SLURM_NTASKS ./sparta -in in.ampt_box_ROAR
