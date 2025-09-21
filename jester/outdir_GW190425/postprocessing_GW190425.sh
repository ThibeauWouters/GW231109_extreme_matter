#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rome
#SBATCH -t 00:05:00
#SBATCH --output="outdir_GW190425/postprocessing_GW190425.out"
#SBATCH --job-name="postprocessing_GW190425"

now=$(date)
echo "$now"

# Loading modules (if needed)
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jester

# Run the script
python postprocessing.py outdir_GW190425

echo "DONE"
