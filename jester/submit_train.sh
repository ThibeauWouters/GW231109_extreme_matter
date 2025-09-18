#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="train.out"
#SBATCH --job-name="NF_train"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jose

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python train_NF.py ../posterior_samples/S250818k_l10000.npz \
    --output-dir ./NFs/S250818k_l10000/ \
    --learning-rate 0.0005 \
    --epochs 5000 \
    --batch-size 128 \
    --nn-block-dim 16 \
    --nn-depth 6

echo "DONE"