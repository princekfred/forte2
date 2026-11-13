#!/bin/bash
#SBATCH --job-name=H4_DSRG_Heff
#SBATCH --partition=ayush
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

#SBATCH --output=output/dsrg_%j.out
#SBATCH --error=output/dsrg_%j.err

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate forte2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on node: $(hostname)"
echo "Starting DSRG calculation"

srun python -u run_dsrg.py

echo "DSRG calculation finished"
