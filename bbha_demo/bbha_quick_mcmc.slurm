#!/bin/bash
#SBATCH --job-name=bbha_q_mcmc
#SBATCH --output=logs/quick_mcmc_%j.out
#SBATCH --error=logs/quick_mcmc_%j.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting BBHA Bilby-MCMC quick job ${SLURM_JOB_ID}"
date

srun python bbha.py --quick --samplers bilby-mcmc

rc=$?
date
echo "Finished BBHA Bilby-MCMC quick job with exit code ${rc}"
exit ${rc}
