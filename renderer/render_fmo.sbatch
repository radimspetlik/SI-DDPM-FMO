#!/bin/bash
#
#SBATCH -p cpu # partition (queue)
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=refgrad
#SBATCH --time=1-0:00:00
#SBATCH --array=0-250:1
#SBATCH --exclude=n33
#SBATCH -o /home/spetlrad/data/defmo/slurm_logs/slurm.%A.%a_%N.out # STDOUT
#SBATCH -e /home/spetlrad/data/defmo/slurm_logs/slurm.%A.%a_%N.err # STDERR

cd "/home/spetlrad/defmo/DeFMO/renderer/"

ml purge
ml "libGLU/9.0.0-fosscuda-2018b"

python3 run_render.py ${SLURM_ARRAY_TASK_ID} 1