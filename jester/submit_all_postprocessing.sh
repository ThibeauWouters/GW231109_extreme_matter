#!/bin/bash

# Loop over all subdirectories matching outdir_*
for d in outdir_*; do
    if [[ -d "$d" ]]; then
        tag="${d#outdir_}"   # Extract suffix after 'outdir_'

        # Path for job script inside the outdir
        jobscript="${d}/postprocessing_${tag}.sh"

        # Create job script in the outdir
        sed "s/{{TAG}}/${tag}/g; s|{{OUTDIR}}|${d}|g" template_postprocessing.sh > "$jobscript"

        # Submit the job from inside the current working dir (so we can find the Python script)
        sbatch "$d/postprocessing_${tag}.sh"
    fi
done
