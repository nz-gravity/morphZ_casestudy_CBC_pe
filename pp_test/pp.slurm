#!/bin/bash
#SBATCH --job-name=pp_%a
#SBATCH --output=logs/seed_%a.out
#SBATCH --error=logs/seed_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-100

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate


echo "Starting job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
date

srun python pp.py  ${SLURM_ARRAY_TASK_ID}

rc=$?
date
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
