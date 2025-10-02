"""
A script to marginalize the posterior over parameters, and just obtain mass_i and Lambda_i marginal posteriors.
Extracts posteriors from multiple top-level directories and saves them with standardized naming.
"""

import numpy as np
import h5py
import os
from pathlib import Path
import glob
import json

# Define the top-level directories from which to fetch posteriors -- essentially in this dirs there should be outdir/final_result/ or outdir/*.json for me to find the posterior samples
TOP_LEVEL_DIRS = []

# # Add all the runs that I did
# TOP_LEVEL_DIRS += glob.glob(os.path.join("/work/wouters/GW231109/", "prod_BW_*"))

# # Add Anna's 3G runs as well
# TOP_LEVEL_DIRS += ["/work/puecher/S231109/third_gen_runs/et_run/",
#                    "/work/puecher/S231109/third_gen_runs/et_run_alignedspin/",
#                    "/work/puecher/S231109/third_gen_runs/et_ce_run/",
#                    "/work/puecher/S231109/third_gen_runs/et_ce_run_alignedspin/",
#                    ]

# Add Anna's EOS sampling runs as well
TOP_LEVEL_DIRS += ["/work/puecher/S231109/eos_sampling/rerun_prod_BW_XP_s005_leos_default/",
                   "/work/puecher/S231109/eos_sampling/prod_BW_XP_s040_leos_default/",
                   ]

# # Add my run for GW190425 as well
# TOP_LEVEL_DIRS += ["/work/wouters/neural_priors_paper_runs/GW190425/bns/default"]

# # Add my run for GW230529 as well
# TOP_LEVEL_DIRS += ["/work/wouters/neural_priors_paper_runs/GW230529/nsbh/default"]

# Create output directory if it doesn't exist
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)

def extract_posterior_from_dir(top_level_dir):
    """
    Extract posterior samples from a top-level directory.
    
    Parameters:
    -----------
    top_level_dir : str or Path
        The top-level directory containing outdir/final_result/
    
    Returns:
    --------
    dict or None
        Dictionary containing posterior samples, or None if extraction failed
    """
    top_level_path = Path(top_level_dir)
    
    # Construct path to the final_result directory
    final_result_dir = top_level_path / "outdir" / "final_result"
    
    try:
        if not final_result_dir.exists():
            print(f"Warning: {final_result_dir} does not exist. Skipping {top_level_dir}")
            raise FileNotFoundError(f"{final_result_dir} does not exist")
        
        # Find HDF5 files in the final_result directory
        print(f"Trying to find HDF5 files in {final_result_dir}")
        hdf5_files = list(final_result_dir.glob("*.hdf5"))
        
        if not hdf5_files:
            print(f"Warning: No HDF5 files found in {final_result_dir}. Skipping {top_level_dir}")
            return None
        
        if len(hdf5_files) > 1:
            print(f"Warning: Multiple HDF5 files found in {final_result_dir}. Using the first one: {hdf5_files[0]}")
        
        posterior_filename = hdf5_files[0]
        print(f"Processing: {posterior_filename}")
    
        with h5py.File(posterior_filename, "r") as f:
            posterior = f["posterior"]
            posterior_dict = {key: posterior[key][:] for key in posterior.keys()}
            
        return posterior_dict
    
    except Exception as e:
        print(f"Did not find HDF5 files or something went wrong: {e}")
    
    try:
        # Find HDF5 files in the final_result directory
        json_outdir = final_result_dir = top_level_path / "outdir"
        print(f"Trying to find JSON files in {json_outdir}")
        json_files = list(json_outdir.glob("*.json"))
    
        if not json_files:
            print(f"Warning: No JSON files found in {json_outdir}. Skipping {top_level_dir}")
            return None
        
        if len(json_files) > 1:
            print(f"Warning: Multiple HDF5 files found in {json_outdir}. Using the first one: {json_files[0]}")
        
        posterior_filename = json_files[0]
        print(f"Processing: {posterior_filename}")
    
        with open(posterior_filename, "r") as f:
            data = json.load(f)
            posterior = data["posterior"]["content"]
            posterior_data = {key: np.array(posterior[key]) for key in posterior.keys()}
            
        return posterior_data
    
    except Exception as e:
        return None

def main():
    """Main function to process all directories and save posterior samples."""
    
    keys_to_fetch = ["mass_1_source",
                     "mass_2_source",
                     "lambda_1",
                     "lambda_2",
                     "lambda_tilde",
                     "delta_lambda_tilde",
                     "total_mass",
                     "chirp_mass",
                     "mass_ratio",
                     "luminosity_distance",
                     "chi_eff",
                     "chi_p",
                     ]
    
    for top_level_dir in TOP_LEVEL_DIRS:
        print(f"\n--- Processing directory: {top_level_dir} ---")
        
        # Extract posterior data
        posterior_data = extract_posterior_from_dir(top_level_dir)
        
        if posterior_data is None:
            continue
        
        # Generate output filename using the top-level directory name
        if "GW190425" in top_level_dir:
            output_filename = "./data/GW190425.npz"
        if "GW230529" in top_level_dir:
            output_filename = "./data/GW230529.npz"
        else:
            dir_name = Path(top_level_dir).name
            output_filename = output_dir / f"{dir_name}.npz"
        
        params_dict = {}
        for key in keys_to_fetch:
            if key in posterior_data:
                params_dict[key] = posterior_data[key]
            else:
                print(list(posterior_data.keys()))
                print(f"Key {key} not found in posterior data from {top_level_dir}. Continuing")
                continue
        
        # Save the marginalized posterior
        np.savez(output_filename, **params_dict)
        
        print(f"Saved posterior samples to: {output_filename}")
        print(f"Number of samples: {len(posterior_data['mass_1_source'])}")
    
    print(f"\n--- Processing complete ---")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main()