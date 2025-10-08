"""Shared utilities for loading and processing EOS data.

This module provides centralized functions for loading equation of state (EOS)
samples from jester inference results, used by both collect_table.py and
money_plots_snellius.py.
"""

import numpy as np
import os
import sys

# Add jester directory to path for imports
sys.path.append('../jester')
import jesterTOV.utils as jose_utils


def load_eos_data(outdir: str):
    """Load EOS data from the specified output directory.

    Automatically removes samples containing NaN values in masses_EOS,
    radii_EOS, or Lambdas_EOS arrays.

    Args:
        outdir: Path to output directory containing eos_samples.npz

    Returns:
        dict: Dictionary containing EOS data arrays with NaN samples removed

    Raises:
        FileNotFoundError: If eos_samples.npz does not exist in outdir
    """
    filename = os.path.join(outdir, "eos_samples.npz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EOS samples file not found: {filename}")

    print(f"Loading data from {filename}")
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
    log_prob = data["log_prob"]

    # Get original sample count
    n_samples_original = len(m)

    # Check for NaN values in each sample (row)
    # For each sample, check if there are any NaNs across all points in that sample
    mask_m = ~np.any(np.isnan(m), axis=1)  # True if no NaNs in this sample
    mask_r = ~np.any(np.isnan(r), axis=1)
    mask_l = ~np.any(np.isnan(l), axis=1)

    # Combine masks - keep sample only if all three are NaN-free
    valid_mask = mask_m & mask_r & mask_l

    # Count and report dropped samples
    n_samples_dropped = n_samples_original - np.sum(valid_mask)

    if n_samples_dropped > 0:
        percent_dropped = 100 * n_samples_dropped / n_samples_original
        print(f"  Dropped {n_samples_dropped} samples ({percent_dropped:.2f}%) containing NaN values")

        # Apply mask to all arrays
        m = m[valid_mask]
        r = r[valid_mask]
        l = l[valid_mask]
        n = n[valid_mask]
        p = p[valid_mask]
        e = e[valid_mask]
        cs2 = cs2[valid_mask]
        log_prob = log_prob[valid_mask]
    else:
        print(f"  No NaN values detected in {n_samples_original} samples")

    # Convert units
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric

    return {
        'masses': m,
        'radii': r,
        'lambdas': l,
        'densities': n,
        'pressures': p,
        'energies': e,
        'cs2': cs2,
        'log_prob': log_prob
    }
