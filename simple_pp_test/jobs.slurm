#!/bin/bash
#SBATCH --job-name=cbc_pe_array
#SBATCH --output=logs/cbc_pe_%A_%a.out
#SBATCH --error=logs/cbc_pe_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-10

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate


echo "Starting job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Running: ${SCRIPT} ${SLURM_ARRAY_TASK_ID}"
date

srun python cbc_pe.py  ${SLURM_ARRAY_TASK_ID}

rc=$?
date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}