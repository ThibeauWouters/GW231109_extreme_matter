"""
A quick script to marginalize the posterior over parameters, and just obtain mass_i and Lambda_i marginal posteriors.
"""

import numpy as np
import h5py

# This is Anna's run dir which used an identical setup to the ROQ Pv2_NRTv2 runs
posterior_filename = "/work/puecher/S250818k/xpnrtv3_lvkcomparison/outdir/final_result/S250818k_XPv3_largespins_smalllambda_data0_1439515224-0351562_analysis_H1L1V1_merge_result.hdf5"
with h5py.File(posterior_filename, "r") as f:
    posterior = f["posterior"]
    mass_1_source = posterior["mass_1_source"][:]
    mass_2_source = posterior["mass_2_source"][:]
    lambda_1 = posterior["lambda_1"][:]
    lambda_2 = posterior["lambda_2"][:]
    
    # Save the marginalized posterior
    np.savez("./posterior_samples/S250818k.npz",
             mass_1_source=mass_1_source,
             mass_2_source=mass_2_source,
             lambda_1=lambda_1,
             lambda_2=lambda_2)
    
    print("DONE")
    
posterior_filename = "/work/wouters/S250818k/XP_NRTv3_LVK_comparison_fixed_sky/outdir/final_result/XP_NRTv3_LVK_comparison_fixed_sky_data0_1439515224-0351562_analysis_H1L1V1_merge_result.hdf5"

with h5py.File(posterior_filename, "r") as f:
    posterior = f["posterior"]
    mass_1_source = posterior["mass_1_source"][:]
    mass_2_source = posterior["mass_2_source"][:]
    lambda_1 = posterior["lambda_1"][:]
    lambda_2 = posterior["lambda_2"][:]
    
    # Save the marginalized posterior
    np.savez("./posterior_samples/S250818k_fixed_sky.npz",
             mass_1_source=mass_1_source,
             mass_2_source=mass_2_source,
             lambda_1=lambda_1,
             lambda_2=lambda_2)
    
    print("DONE")