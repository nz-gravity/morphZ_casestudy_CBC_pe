#!/bin/bash
#SBATCH --job-name=pp_q_mcmc
#SBATCH --output=logs/quick_mcmc_%a.out
#SBATCH --error=logs/quick_mcmc_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-100

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting Bilby-MCMC quick job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
date

srun python pp.py ${SLURM_ARRAY_TASK_ID} --quick --samplers bilby-mcmc

rc=$?
date
echo "Finished Bilby-MCMC quick task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
