"""
Check the SNRs of the different GW analyses (different priors) and how they might change.
Loops over all prod_BW_* directories (same setup as marginalize_posterior.py) and saves
SNR samples to a JSON file for postprocessing locally.
"""

import h5py
import numpy as np
import os
import glob
import json
from pathlib import Path

# Mirror the exact same directory setup as marginalize_posterior.py
TOP_LEVEL_DIRS = []
TOP_LEVEL_DIRS += glob.glob(os.path.join("/work/wouters/GW231109/", "prod_BW_*"))
TOP_LEVEL_DIRS += ["/work/puecher/S231109/eos_sampling/prod_BW_XP_s005_leos_default/",
                   "/work/puecher/S231109/eos_sampling/prod_BW_XP_s040_leos_default/",
                   ]

OUTPUT_FILE = "./data/snr_samples.json"

def load_snrs_from_posterior(hdf5_filename):
    """Extract all SNR-related keys from a posterior HDF5 file."""
    with h5py.File(hdf5_filename, 'r') as f:
        posterior = f['posterior']
        snrs_dict = {}
        for k in posterior.keys():
            if 'matched_filter_snr' not in k and 'optimal_snr' not in k:
                continue
            arr = posterior[k][()]
            # matched_filter_snr can be complex; store the absolute value
            if np.iscomplexobj(arr):
                arr = np.abs(arr)
            snrs_dict[k] = arr.tolist()
    return snrs_dict

def find_hdf5_file(top_level_dir):
    """Find the HDF5 posterior file in a top-level directory (same logic as marginalize_posterior.py)."""
    top_level_path = Path(top_level_dir)
    final_result_dir = top_level_path / "outdir" / "final_result"

    if not final_result_dir.exists():
        print(f"Warning: {final_result_dir} does not exist. Skipping {top_level_dir}")
        return None

    print(f"Trying to find HDF5 files in {final_result_dir}")
    hdf5_files = list(final_result_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"Warning: No HDF5 files found in {final_result_dir}. Checking `result` instead of `final_result`")
        final_result_dir = top_level_path / "outdir" / "result"
        hdf5_files = list(final_result_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"Warning: No HDF5 files found in {final_result_dir}. Skipping {top_level_dir}")
            return None

    if len(hdf5_files) > 1:
        print(f"Warning: Multiple HDF5 files found in {final_result_dir}. Using the first one: {hdf5_files[0]}")

    return hdf5_files[0]

def main():
    Path("./data").mkdir(exist_ok=True)

    all_results = {}

    for top_level_dir in TOP_LEVEL_DIRS:
        print(f"\n--- Processing directory: {top_level_dir} ---")

        hdf5_file = find_hdf5_file(top_level_dir)
        if hdf5_file is None:
            continue

        print(f"Processing: {hdf5_file}")
        try:
            snrs = load_snrs_from_posterior(hdf5_file)
        except Exception as e:
            print(f"Failed to load SNRs from {hdf5_file}: {e}")
            continue

        print(f"Found SNR keys: {list(snrs.keys())}")

        dir_name = Path(top_level_dir).name
        all_results[dir_name] = snrs

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f)

    print(f"\n--- Processing complete ---")
    print(f"SNR samples saved to: {OUTPUT_FILE}")
    print(f"Runs saved: {list(all_results.keys())}")

if __name__ == "__main__":
    main()
