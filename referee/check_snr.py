"""
Check the SNRs of the different GW analyses (different priors) and how they might change
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16, 
        "figure.titlesize": 16}

plt.rcParams.update(params)

def load_snrs_from_posterior(hdf5_filename: str):
    
    # Open HDF5 file and show keys in posterior
    with h5py.File(hdf5_filename, 'r') as f:
        posterior = f['posterior']
        print(f"Keys in posterior: {list(posterior.keys())}")
        
        snrs_dict = {k: posterior[k][()] for k in posterior.keys() if 'matched_filter_snr' in k or 'optimal_snr' in k}
        
    return snrs_dict

def main():
    
    test_filename = "/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/outdir/final_result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5"
    snrs = load_snrs_from_posterior(test_filename)
    

if __name__ == "__main__":
    main()