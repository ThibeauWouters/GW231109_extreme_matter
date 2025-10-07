#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G
#SBATCH --output="train_NF.out"
#SBATCH --job-name="train"

now=$(date)
echo "$now"

# Load environment
conda activate /home/twouters2/miniconda3/envs/jester

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

##############################################
# Flexible arguments
##############################################
DATA_PATH=$1     # first argument to script
OUTDIR=$2        # second argument to script (optional)

# If no OUTDIR given, derive from data filename
if [ -z "$OUTDIR" ]; then
    BASENAME=$(basename "$DATA_PATH" .npz)
    OUTDIR="./NFs/${BASENAME}/"
fi

echo "Using data:    $DATA_PATH"
echo "Output folder: $OUTDIR"

##############################################
# Run training
##############################################
python train_NF.py "$DATA_PATH" \
    --output-dir "$OUTDIR" \
    --learning-rate 0.0005 \
    --epochs 5000 \
    --batch-size 64 \
    --nn-block-dim 16 \
    --nn-depth 6

echo "DONE"
