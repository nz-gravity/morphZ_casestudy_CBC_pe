#!/bin/bash
#SBATCH --job-name=bbha_p_dyn
#SBATCH --output=logs/prod_dynesty_%j.out
#SBATCH --error=logs/prod_dynesty_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting BBHA Dynesty prod job ${SLURM_JOB_ID}"
date

srun python bbha.py --samplers dynesty

rc=$?
date
echo "Finished BBHA Dynesty prod job with exit code ${rc}"
exit ${rc}
