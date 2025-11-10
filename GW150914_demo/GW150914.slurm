#!/bin/bash
#SBATCH --job-name=GW150914
#SBATCH --output=logs/GW150914.out
#SBATCH --error=logs/GW150914.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate



echo "Starting job ${SLURM_JOB_ID} GW150914 analysis"
srun python GW150914.py

rc=$?
date
echo "Finished job ${SLURM_JOB_ID} with exit code ${rc}"
exit ${rc}