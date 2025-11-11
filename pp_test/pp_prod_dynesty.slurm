#!/bin/bash
#SBATCH --job-name=pp_p_dyn
#SBATCH --output=logs/prod_dynesty_%a.out
#SBATCH --error=logs/prod_dynesty_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-100

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Starting Dynesty prod job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
date

srun python pp.py ${SLURM_ARRAY_TASK_ID} --samplers dynesty

rc=$?
date
echo "Finished Dynesty prod task ${SLURM_ARRAY_TASK_ID} with exit code ${rc}"
exit ${rc}
