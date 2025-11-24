#!/bin/bash
#SBATCH --job-name=gw150914_q_dyn
#SBATCH --output=logs/quick_dynesty_%j.out
#SBATCH --error=logs/quick_dynesty_%j.err
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --dependency=singleton


module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

echo "Starting GW150914 Dynesty quick job ${SLURM_JOB_ID}"
date

srun python GW150914.py --quick --samplers dynesty

rc=$?
date
echo "Finished GW150914 Dynesty quick job with exit code ${rc}"
exit ${rc}
