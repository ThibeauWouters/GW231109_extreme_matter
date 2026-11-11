#!/bin/bash -l
#SBATCH -J et_2l_ce_xasinj_gw231109
#SBATCH -o logs/et_2l_ce_xasinj_gw231109.out
#SBATCH -e logs/et_2l_ce_xasinj_gw231109.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 192
#SBATCH -p cpu

source /work/wouters/projects/19_GW231109_referee/.venv/bin/activate

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun python bns_mb.py
