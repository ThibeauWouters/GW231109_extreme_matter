#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for subdir in "$SCRIPT_DIR"/*/; do
    # Skip the 3G directory
    [[ "$(basename "$subdir")" == "3G" ]] && continue

    if [[ -f "$subdir/submit.sh" ]]; then
        echo "Submitting: $subdir"
        (cd "$subdir" && sbatch submit.sh)
    fi
done

echo "All jobs submitted."