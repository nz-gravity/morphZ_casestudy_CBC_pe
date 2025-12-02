#!/bin/bash
#SBATCH --job-name=gw150914_NS_wave
#SBATCH --output=logs/dynesty_%A_%a.out
#SBATCH --error=logs/dynesty_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=300MB
#SBATCH --array=0-4

module purge
module load python-scientific/3.11.5-foss-2023b
source /fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/.venv/bin/activate

NLIVE_DEFAULT=2000
NLIVE_GRID=(100 500 1000 1500 2500)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID < ${#NLIVE_GRID[@]} )); then
        NLIVE=${NLIVE_GRID[$SLURM_ARRAY_TASK_ID]}
    else
        echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds NLIVE_GRID size ${#NLIVE_GRID[@]}" >&2
        exit 1
    fi
else
    NLIVE=${NLIVE_DEFAULT}
fi

JOBNAME="gw150914_NS_wave_nlive${NLIVE}"
scontrol update JobId=${SLURM_JOB_ID} JobName=${JOBNAME}

echo "Starting GW150914 Dynesty prod job ${SLURM_JOB_ID}"
echo "Using dynesty nlive=${NLIVE}"
date

srun python GW150914_analysis.py --sampler dynesty --nlive ${NLIVE}

rc=$?
date
echo "Finished GW150914 Dynesty prod job with exit code ${rc}"
exit ${rc}
