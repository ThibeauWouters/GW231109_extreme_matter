#!/bin/bash

# Directory containing the .npz files
DATA_DIR="../posteriors/data/"

# List of input files (just paste your filenames here)
FILES=(
  prod_BW_XP_s005_l5000_double_gaussian.npz
)

# Loop over files and submit job for each
for f in "${FILES[@]}"; do
    INPUT_PATH="$DATA_DIR/$f"
    BASENAME=$(basename "$f" .npz)
    OUTDIR="./NFs/${BASENAME}/"

    echo "Submitting job for $INPUT_PATH → $OUTDIR"

    sbatch template_train.sh "$INPUT_PATH" "$OUTDIR"
done
