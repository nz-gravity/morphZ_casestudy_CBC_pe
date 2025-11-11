#!/bin/bash
#SBATCH --job-name=gw150914_p_mcmc
#SBATCH --output=logs/prod_mcmc_%j.out
#SBATCH --error=logs/prod_mcmc_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting GW150914 Bilby-MCMC prod job ${SLURM_JOB_ID}"
date

srun python GW150914.py --samplers bilby-mcmc

rc=$?
date
echo "Finished GW150914 Bilby-MCMC prod job with exit code ${rc}"
exit ${rc}
