#!/usr/bin/env python3
"""
Final money plots script for GW231109 extreme matter investigations.
This script creates comparison corner plots where multiple analysis runs
are overlaid for direct comparison.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner

from utils import (
    DEFAULT_CORNER_KWARGS, GW231109_COLOR, GW190425_COLOR, PRIOR_COLOR, GW170817_COLOR,
    ORANGE, BLUE, GREEN, identify_person_from_path, load_posterior_samples,
    load_run_metadata, load_priors_for_corner, ensure_directory_exists
)

def create_comparison_cornerplot(source_dirs: list[str],
                               parameters: list[str],
                               labels: list[str] = None,
                               colors: list[str] = None,
                               ranges: dict = None,
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

        # Default colors if not provided
        if colors is None:
            default_colors = [ORANGE, BLUE, GREEN, GW231109_COLOR, GW190425_COLOR, GW170817_COLOR, PRIOR_COLOR]
            colors = default_colors[:len(source_dirs)]

        # Default labels if not provided
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(source_dirs))]

        # Load all posterior samples
        all_samples = []
        valid_dirs = []
        valid_labels = []
        valid_colors = []

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

                print(f"  Loaded {len(samples)} samples")

            except Exception as e:
                print(f"  Failed to load samples from {source_dir}: {e}")
                continue

        if not all_samples:
            print("No valid samples loaded!")
            return False

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

        # Create initial plot
        fig = corner.corner(all_samples[0],
                           labels=parameters,
                           **corner_kwargs)

        # Overlay additional datasets
        for i in range(1, len(all_samples)):
            corner_kwargs_overlay = corner_kwargs.copy()
            corner_kwargs_overlay["color"] = valid_colors[i]
            corner_kwargs_overlay["fig"] = fig

            corner.corner(all_samples[i],
                         labels=parameters,
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
        "/work/wouters/GW231109/run1",  # Replace with actual paths
        "/work/wouters/GW231109/run2",
        "/work/wouters/GW231109/run3",
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

    # Specify parameter ranges (optional)
    # Format: {parameter_name: (min_value, max_value)}
    ranges = {
        "chirp_mass": (1.1, 1.5),
        "lambda_tilde": (0, 2000),
        # "mass_ratio": (0.7, 1.0),
        # Add more parameter ranges as needed
    }

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
        save_name=save_name,
        overwrite=overwrite
    )

    if success:
        print(f"✓ Successfully created comparison corner plot: {save_name}")
    else:
        print("✗ Failed to create comparison corner plot")

if __name__ == "__main__":
    main()