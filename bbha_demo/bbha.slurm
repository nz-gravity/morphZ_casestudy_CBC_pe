#!/bin/bash
#SBATCH --job-name=bbha_cbc_pe
#SBATCH --output=logs/bbha.out
#SBATCH --error=logs/bbha.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate


echo "Starting job ${SLURM_JOB_ID}"
date

srun python bbha.py

rc=$?
date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}