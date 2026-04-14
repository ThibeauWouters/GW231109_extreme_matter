"""
Check the log Bayes factors of the different GW analyses (different priors).
Loops over all prod_BW_* directories (same setup as check_snr.py) and reads
log_bayes_factor from each bilby HDF5 result file.

The log Bayes factor reported here is signal vs. noise (Gaussian noise), i.e.
ln B^signal_noise.  A value >> 0 indicates the data strongly prefer the
signal hypothesis over pure noise.

Results are printed as a sorted table and optionally saved to a JSON file.
"""

import h5py
import numpy as np
import os
import glob
import json
from pathlib import Path

# Mirror the exact same directory setup as check_snr.py / marginalize_posterior.py
TOP_LEVEL_DIRS = []
TOP_LEVEL_DIRS += glob.glob(os.path.join("/work/wouters/GW231109/", "prod_BW_*"))
TOP_LEVEL_DIRS += [
    "/work/puecher/S231109/eos_sampling/prod_BW_XP_s005_leos_default/",
    "/work/puecher/S231109/eos_sampling/prod_BW_XP_s040_leos_default/",
]

OUTPUT_FILE = "./data/logbf_results.json"


def load_logbf_from_hdf5(hdf5_filename):
    """Extract log_bayes_factor from a bilby result HDF5 file.

    Bilby stores this as a root-level dataset (scalar) named 'log_bayes_factor'.
    Falls back to checking root-level attributes if the dataset is absent.
    """
    with h5py.File(hdf5_filename, "r") as f:
        # Primary location: root-level dataset
        if "log_bayes_factor" in f:
            val = f["log_bayes_factor"][()]
            return float(val)
        # Secondary: root-level attribute
        if "log_bayes_factor" in f.attrs:
            return float(f.attrs["log_bayes_factor"])
        # Tertiary: some bilby versions nest under 'meta_data'
        if "meta_data" in f and "log_bayes_factor" in f["meta_data"]:
            val = f["meta_data"]["log_bayes_factor"][()]
            return float(val)
    raise KeyError(f"log_bayes_factor not found in {hdf5_filename}")


def find_hdf5_file(top_level_dir):
    """Find the HDF5 posterior file in a top-level directory (same logic as check_snr.py)."""
    top_level_path = Path(top_level_dir)
    final_result_dir = top_level_path / "outdir" / "final_result"

    if not final_result_dir.exists():
        print(f"Warning: {final_result_dir} does not exist. Skipping {top_level_dir}")
        return None

    print(f"Trying to find HDF5 files in {final_result_dir}")
    hdf5_files = list(final_result_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(
            f"Warning: No HDF5 files found in {final_result_dir}. "
            "Checking `result` instead of `final_result`"
        )
        final_result_dir = top_level_path / "outdir" / "result"
        hdf5_files = list(final_result_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(
                f"Warning: No HDF5 files found in {final_result_dir}. "
                f"Skipping {top_level_dir}"
            )
            return None

    if len(hdf5_files) > 1:
        print(
            f"Warning: Multiple HDF5 files found in {final_result_dir}. "
            f"Using the first one: {hdf5_files[0]}"
        )

    return hdf5_files[0]


def print_table(results):
    """Print results sorted by log Bayes factor (descending)."""
    sorted_items = sorted(results.items(), key=lambda kv: kv[1], reverse=True)

    col_width = max(len(k) for k in results) + 2
    header = f"{'Run':<{col_width}}  ln B^signal_noise"
    print(header)
    print("-" * len(header))
    for run_name, lbf in sorted_items:
        print(f"{run_name:<{col_width}}  {lbf:.4f}")
    print()
    values = list(results.values())
    print(f"  N runs : {len(values)}")
    print(f"  Min    : {min(values):.4f}")
    print(f"  Max    : {max(values):.4f}")
    print(f"  Median : {np.median(values):.4f}")
    print(f"  Mean   : {np.mean(values):.4f}")


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
            lbf = load_logbf_from_hdf5(hdf5_file)
        except Exception as e:
            print(f"Failed to load log_bayes_factor from {hdf5_file}: {e}")
            continue

        print(f"  log_bayes_factor = {lbf:.4f}")
        dir_name = Path(top_level_dir).name
        all_results[dir_name] = lbf

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n--- Processing complete ---")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Runs processed: {len(all_results)}\n")

    if all_results:
        print_table(all_results)


if __name__ == "__main__":
    main()
