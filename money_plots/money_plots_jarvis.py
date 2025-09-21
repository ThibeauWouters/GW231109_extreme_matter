#!/usr/bin/env python3
"""
Final money plots script for GW231109 extreme matter investigations.
This script creates comparison corner plots where multiple analysis runs
are overlaid for direct comparison.
"""

import os
import argparse
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner

from utils import (
    DEFAULT_CORNER_KWARGS, GW231109_COLOR, GW190425_COLOR, PRIOR_COLOR, GW170817_COLOR,
    ORANGE, BLUE, GREEN, identify_person_from_path, load_posterior_samples,
    load_run_metadata, load_priors_for_corner, ensure_directory_exists
)

# Parameter translation dictionary for LaTeX symbols
PARAMETER_LABELS = {
    "chirp_mass": r"$\mathcal{M}_c$ [M$_\odot$]",
    "mass_ratio": r"$q$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "chi_eff": r"$\chi_{\rm eff}$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "mass_1": r"$m_1$ [M$_\odot$]",
    "mass_2": r"$m_2$ [M$_\odot$]",
    "luminosity_distance": r"$d_L$ [Mpc]",
    "theta_jn": r"$\theta_{JN}$",
    "psi": r"$\psi$",
    "phase": r"$\phi$",
    "geocent_time": r"$t_c$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$",
    "phi_12": r"$\Delta\phi$",
    "phi_jl": r"$\phi_{JL}$"
}

def generate_cache_filename(source_dirs: list[str], parameters: list[str]) -> str:
    """
    Generate a unique cache filename based on source directories and parameters.

    Args:
        source_dirs (list[str]): List of source directories
        parameters (list[str]): List of parameters

    Returns:
        str: Cache filename
    """
    # Create a string combining directories and parameters
    cache_key = "|".join(sorted(source_dirs)) + "|" + "|".join(sorted(parameters))

    # Generate MD5 hash for a consistent filename
    hash_obj = hashlib.md5(cache_key.encode())
    hash_hex = hash_obj.hexdigest()

    return f"./data/comparison_cache_{hash_hex}.pkl"

def save_comparison_data(filename: str, data: dict) -> bool:
    """
    Save comparison data to cache file.

    Args:
        filename (str): Cache filename
        data (dict): Data to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory_exists(filename)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved comparison data to cache: {filename}")
        return True
    except Exception as e:
        print(f"Failed to save cache data: {e}")
        return False

def load_comparison_data(filename: str) -> dict:
    """
    Load comparison data from cache file.

    Args:
        filename (str): Cache filename

    Returns:
        dict: Loaded data, or None if failed
    """
    try:
        if not os.path.exists(filename):
            return None

        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded comparison data from cache: {filename}")
        return data
    except Exception as e:
        print(f"Failed to load cache data: {e}")
        return None

def load_and_process_data(source_dirs: list[str],
                         parameters: list[str],
                         labels: list[str] = None,
                         colors: list[str] = None,
                         zorders: list[int] = None) -> tuple:
    """
    Load and process data from source directories.

    Args:
        source_dirs (list[str]): List of directories containing posterior samples
        parameters (list[str]): Parameters to include
        labels (list[str]): Labels for each run (optional)
        colors (list[str]): Colors for each run (optional)
        zorders (list[int]): Z-order for each run (optional)

    Returns:
        tuple: (all_samples, valid_labels, valid_colors, valid_zorders)
    """
    # Default colors if not provided
    if colors is None:
        default_colors = [ORANGE, BLUE, GREEN, GW231109_COLOR, GW190425_COLOR, GW170817_COLOR, PRIOR_COLOR]
        colors = default_colors[:len(source_dirs)]

    # Default labels if not provided
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(source_dirs))]

    # Default z-orders if not provided
    if zorders is None:
        zorders = list(range(len(source_dirs)))  # Default: 0, 1, 2, ...
    elif len(zorders) != len(source_dirs):
        print(f"Warning: zorders length ({len(zorders)}) doesn't match source_dirs length ({len(source_dirs)})")
        zorders = list(range(len(source_dirs)))

    # Load all posterior samples
    all_samples = []
    valid_dirs = []
    valid_labels = []
    valid_colors = []
    valid_zorders = []

    for i, source_dir in enumerate(source_dirs):
        try:
            print(f"Loading samples from: {source_dir}")

            # Load metadata
            metadata = load_run_metadata(source_dir)
            if "log_bayes_factor" in metadata:
                print(f"  Log Bayes factor: {metadata['log_bayes_factor']}")
            if "sampling_time_hrs" in metadata:
                print(f"  Sampling time: {metadata['sampling_time_hrs']:.2f} hours")

            # Load posterior samples
            samples = load_posterior_samples(source_dir, parameters)
            all_samples.append(samples)
            valid_dirs.append(source_dir)
            valid_labels.append(labels[i])
            valid_colors.append(colors[i])
            valid_zorders.append(zorders[i])

            print(f"  Loaded {len(samples)} samples")

        except Exception as e:
            print(f"  Failed to load samples from {source_dir}: {e}")
            continue

    if not all_samples:
        return [], [], [], []

    # Sort all data by z-order (lowest first, so higher z-order plots appear on top)
    sorted_data = sorted(zip(valid_zorders, all_samples, valid_labels, valid_colors),
                       key=lambda x: x[0])
    valid_zorders, all_samples, valid_labels, valid_colors = zip(*sorted_data)
    valid_zorders, all_samples, valid_labels, valid_colors = list(valid_zorders), list(all_samples), list(valid_labels), list(valid_colors)

    return all_samples, valid_labels, valid_colors, valid_zorders

def create_comparison_cornerplot(source_dirs: list[str],
                               parameters: list[str],
                               labels: list[str] = None,
                               colors: list[str] = None,
                               ranges: dict = None,
                               zorders: list[int] = None,
                               save_name: str = "comparison_cornerplot.pdf",
                               overwrite: bool = False) -> bool:
    """
    Create a comparison corner plot with multiple runs overlaid.

    Args:
        source_dirs (list[str]): List of directories containing posterior samples
        parameters (list[str]): Parameters to include in the corner plot
        labels (list[str]): Labels for each run (optional)
        colors (list[str]): Colors for each run (optional)
        ranges (dict): Parameter ranges as {param: (min, max)} (optional)
        zorders (list[int]): Z-order for each run (higher values appear on top, optional)
        save_name (str): Output filename
        overwrite (bool): Whether to overwrite existing plots

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists, skipping...")
            return True

        print(f"Creating comparison corner plot for {len(source_dirs)} runs")
        print(f"Parameters: {parameters}")

        # Check if cached data exists
        cache_filename = generate_cache_filename(source_dirs, parameters)
        cached_data = load_comparison_data(cache_filename)

        if cached_data is not None:
            print("Using cached comparison data")
            all_samples = cached_data['all_samples']
            valid_labels = cached_data['valid_labels']
            valid_colors = cached_data['valid_colors']
            valid_zorders = cached_data['valid_zorders']
        else:
            print("Loading data from source files and caching...")
            # Load data from scratch
            all_samples, valid_labels, valid_colors, valid_zorders = load_and_process_data(
                source_dirs, parameters, labels, colors, zorders
            )

            if not all_samples:
                print("No valid samples loaded!")
                return False

            # Save to cache
            cache_data = {
                'all_samples': all_samples,
                'valid_labels': valid_labels,
                'valid_colors': valid_colors,
                'valid_zorders': valid_zorders
            }
            save_comparison_data(cache_filename, cache_data)

        # Create the corner plot with the first dataset
        print("Creating corner plot...")

        # Set up corner kwargs
        corner_kwargs = DEFAULT_CORNER_KWARGS.copy()
        corner_kwargs["color"] = valid_colors[0]

        # Apply parameter ranges if provided
        if ranges:
            range_list = []
            for param in parameters:
                if param in ranges:
                    range_list.append(ranges[param])
                else:
                    range_list.append(None)
            corner_kwargs["range"] = range_list

        # Create parameter labels using translation dictionary
        parameter_labels = [PARAMETER_LABELS.get(param, param) for param in parameters]

        # Create initial plot
        fig = corner.corner(all_samples[0],
                           labels=parameter_labels,
                           **corner_kwargs)

        # Overlay additional datasets
        for i in range(1, len(all_samples)):
            corner_kwargs_overlay = corner_kwargs.copy()
            corner_kwargs_overlay["color"] = valid_colors[i]
            corner_kwargs_overlay["fig"] = fig

            corner.corner(all_samples[i],
                         labels=parameter_labels,
                         **corner_kwargs_overlay)

        # Add legend
        legend_elements = []
        for i, (label, color) in enumerate(zip(valid_labels, valid_colors)):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='k', label=label)
            )

        fig.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), frameon=True)

        # Save plot
        ensure_directory_exists(save_name)
        print(f"Saving comparison corner plot to {save_name}")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

        return True

    except Exception as e:
        print(f"Failed to create comparison corner plot: {e}")
        return False

def main():
    """
    Main function for creating comparison corner plots.
    Edit this function to specify the directories, parameters, and settings for your comparison.
    """

    # ====== USER CONFIGURATION ======
    # Specify the source directories to compare
    source_dirs = [
        "/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/",  # Replace with actual paths
        "/work/wouters/GW231109/prod_BW_XP_s005_lquniv_default/",
        "/work/wouters/GW231109/prod_BW_XP_s005_l5000_double_gaussian",
    ]

    # Specify the parameters to include in the corner plot
    parameters = [
        "chirp_mass",
        "mass_ratio",
        "lambda_1",
        "lambda_2",
        "chi_eff",
        "lambda_tilde"
    ]

    # Specify labels for each run (optional)
    labels = [
        "Default Prior",
        "Quasi-Universal",
        "Double Gaussian",
    ]

    # Specify colors for each run (optional)
    colors = [
        ORANGE,    # "#de8f07"
        BLUE,      # "#0472b1"
        GREEN,     # "#019e72"
    ]

    # Specify z-orders for each run (higher values appear on top)
    # Making quasi-universal (index 1) have the highest z-order
    zorders = [0, 2, 1]  # Default Prior: 0, Quasi-Universal: 2 (highest), Double Gaussian: 1

    # Specify parameter ranges (optional)
    # Format: {parameter_name: (min_value, max_value)}
    # ranges = {
    #     "chirp_mass": (1.1, 1.5),
    #     "lambda_tilde": (0, 2000),
    #     # "mass_ratio": (0.7, 1.0),
    #     # Add more parameter ranges as needed
    # }
    ranges = None

    # Output filename
    save_name = "comparison_cornerplot.pdf"

    # Whether to overwrite existing plots
    overwrite = True

    # ====== END USER CONFIGURATION ======

    print("Creating comparison corner plot...")
    print(f"Directories: {len(source_dirs)}")
    print(f"Parameters: {parameters}")
    print(f"Labels: {labels}")

    # Create the comparison corner plot
    success = create_comparison_cornerplot(
        source_dirs=source_dirs,
        parameters=parameters,
        labels=labels,
        colors=colors,
        ranges=ranges,
        zorders=zorders,
        save_name=save_name,
        overwrite=overwrite
    )

    if success:
        print(f"✓ Successfully created comparison corner plot: {save_name}")
    else:
        print("✗ Failed to create comparison corner plot")

if __name__ == "__main__":
    main()