"""
Check the SNRs of the different GW analyses (different priors) and how they might change
"""

import numpy as np
import h5py

def load_posterior(hdf5_filename: str):
    
    # Open HDF5 file and show keys in posterior
    with h5py.File(hdf5_filename, 'r') as f:
        posterior = f['posterior']
        print(f"Keys in posterior: {list(posterior.keys())}")
        

def main():
    test_filename = "../posteriors/data/prod_BW_XP_s005_leos_default_no_zeros.npz"
    load_posterior(test_filename)
    
    test_filename = "../posteriors/data/prod_BW_XP_s005_l5000_double_gaussian.npz"
    load_posterior(test_filename)
    
if __name__ == "__main__":
    main()