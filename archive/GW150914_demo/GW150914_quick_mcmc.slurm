#!/bin/bash
#SBATCH --job-name=gw150914_q_mcmc
#SBATCH --output=logs/quick_mcmc_%j.out
#SBATCH --error=logs/quick_mcmc_%j.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting GW150914 Bilby-MCMC quick job ${SLURM_JOB_ID}"
date

srun python GW150914.py --quick --samplers bilby_mcmc

rc=$?
date
echo "Finished GW150914 Bilby-MCMC quick job with exit code ${rc}"
exit ${rc}
