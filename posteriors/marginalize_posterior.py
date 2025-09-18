"""
A script to marginalize the posterior over parameters, and just obtain mass_i and Lambda_i marginal posteriors.
Extracts posteriors from multiple top-level directories and saves them with standardized naming.
"""

import numpy as np
import h5py
import os
from pathlib import Path
import glob

# Define the top-level directories from which to fetch posteriors
BASE_PATH = "/work/wouters/GW231109/"
TOP_LEVEL_DIRS = glob.glob(os.path.join(BASE_PATH, "prod_BW_*"))

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
    
    if not final_result_dir.exists():
        print(f"Warning: {final_result_dir} does not exist. Skipping {top_level_dir}")
        return None
    
    # Find HDF5 files in the final_result directory
    hdf5_files = list(final_result_dir.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"Warning: No HDF5 files found in {final_result_dir}. Skipping {top_level_dir}")
        return None
    
    if len(hdf5_files) > 1:
        print(f"Warning: Multiple HDF5 files found in {final_result_dir}. Using the first one: {hdf5_files[0]}")
    
    posterior_filename = hdf5_files[0]
    print(f"Processing: {posterior_filename}")
    
    try:
        with h5py.File(posterior_filename, "r") as f:
            posterior = f["posterior"]
            
            # Extract the required parameters
            posterior_data = {
                "mass_1_source": posterior["mass_1_source"][:],
                "mass_2_source": posterior["mass_2_source"][:],
                "lambda_1": posterior["lambda_1"][:],
                "lambda_2": posterior["lambda_2"][:]
            }
            
        return posterior_data
    
    except Exception as e:
        print(f"Error processing {posterior_filename}: {e}")
        return None

def main():
    """Main function to process all directories and save posterior samples."""
    
    for top_level_dir in TOP_LEVEL_DIRS:
        print(f"\n--- Processing directory: {top_level_dir} ---")
        
        # Extract posterior data
        posterior_data = extract_posterior_from_dir(top_level_dir)
        
        if posterior_data is None:
            continue
        
        # Generate output filename using the top-level directory name
        dir_name = Path(top_level_dir).name
        output_filename = output_dir / f"{dir_name}.npz"
        
        # Save the marginalized posterior
        np.savez(output_filename,
                 mass_1_source=posterior_data["mass_1_source"],
                 mass_2_source=posterior_data["mass_2_source"],
                 lambda_1=posterior_data["lambda_1"],
                 lambda_2=posterior_data["lambda_2"])
        
        print(f"Saved posterior samples to: {output_filename}")
        print(f"Number of samples: {len(posterior_data['mass_1_source'])}")
    
    print(f"\n--- Processing complete ---")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main()