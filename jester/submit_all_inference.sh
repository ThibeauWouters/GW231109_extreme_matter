#!/bin/bash

# Loop over all subdirectories that start with "outdir"
for DIR in outdir*/; do
    # Check if it's actually a directory (in case no matches found)
    if [ -d "$DIR" ]; then
        # Remove trailing slash for cleaner output
        DIR_NAME="${DIR%/}"
        
        # Skip if directory name contains "double_gaussian"
        if [[ "$DIR_NAME" == *double_gaussian* ]]; then
            echo "Skipping $DIR_NAME (contains 'double_gaussian')"
            continue
        fi

        # Check if submit.sh exists in the directory
        if [ -f "$DIR_NAME/submit.sh" ]; then
            echo "Submitting job for $DIR_NAME"
            sbatch "$DIR_NAME/submit.sh"
        else
            echo "Warning: $DIR_NAME/submit.sh not found, skipping"
        fi
    fi
done
