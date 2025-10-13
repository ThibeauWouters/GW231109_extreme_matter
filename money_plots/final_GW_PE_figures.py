#!/usr/bin/env python3
"""
Final GW PE figures script for GW231109 extreme matter investigations.
Creates comparison corner plots from .npz posterior files with customizable datasets,
parameters, colors, and ranges.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
import arviz

from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_mass_ratio
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.gw.conversion import generate_spin_parameters, generate_component_spins
from bilby.gw.conversion import luminosity_distance_to_redshift

# If running on Mac, so we can use TeX (not on Jarvis), change some rc params
if "Woute029" in os.getcwd():
    print(f"Updating plotting parameters for TeX")
    fs = 18
    ticks_fs = 20
    label_fs = 24  # Bigger labels
    legend_fs = 24  # Smaller legend
    labelpad = 18  # Bigger labelpad
    rc_params = {"axes.grid": False,
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"],
            "xtick.labelsize": ticks_fs,
            "ytick.labelsize": ticks_fs,
            "axes.labelsize": label_fs,
            "legend.fontsize": legend_fs,
            "legend.title_fontsize": legend_fs,
            "figure.titlesize": fs}
    plt.rcParams.update(rc_params)
    

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
    "mass_1_source": r"$m_1^{\rm src}$ [M$_\odot$]",
    "mass_2_source": r"$m_2^{\rm src}$ [M$_\odot$]",
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
    "phi_jl": r"$\phi_{JL}$",
    "EOS": r"EOS"
}

# Default corner plot settings
DEFAULT_CORNER_KWARGS = {
    "bins": 40,
    "smooth": 0.9,
    "plot_datapoints": False,
    "plot_density": False,
    "fill_contours": True,
    "show_titles": False,
    "title_fmt": ".3f",
    "levels": [0.5, 0.9],  # 50% and 90% credible regions
    "labelpad": 0.10,
    "max_n_ticks": 3,
    "min_n_ticks": 2,
}

# Default colors
ORANGE = "#de8f07"
BLUE = "#0472b1"
GREEN = "#019e72"
RED = "#cc3311"

# Default parameter ranges for plots
DEFAULT_RANGES = {
    "chirp_mass": (1.3055, 1.3080),
    "mass_ratio": (0.3, 1.0),
    "chi_eff": (-0.02, 0.15),
    "lambda_1": (0, 5000),
    "lambda_2": (0, 5000),
    "lambda_tilde": (0, 5000),
    "luminosity_distance": (100, 300),
    "mass_1_source": (1.0, 3.0),
    "mass_2_source": (1.0, 3.0),
}


def ensure_directory_exists(filepath: str):
    """
    Ensure the directory for a filepath exists.

    Args:
        filepath (str): Full file path
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_npz_data(filepath: str, parameters: list[str]) -> np.ndarray:
    """
    Load posterior samples from .npz file.

    Args:
        filepath (str): Path to .npz file
        parameters (list[str]): Parameters to extract

    Returns:
        np.ndarray: Samples array with shape (n_samples, n_parameters)
    """
    data = np.load(filepath)

    # Check that all parameters are available
    available_params = list(data.keys())
    missing_params = [p for p in parameters if p not in available_params]

    if missing_params:
        raise ValueError(f"Parameters {missing_params} not found in {filepath}. "
                        f"Available: {available_params}")

    # Extract samples for each parameter
    samples = []
    for param in parameters:
        samples.append(data[param])

    samples_array = np.column_stack(samples)

    print(f"Loaded {len(samples_array)} samples from {filepath}")
    print(f"  Parameters: {parameters}")

    return samples_array


def create_comparison_cornerplot(
    filepaths: list[str],
    parameters: list[str],
    labels: list[str],
    colors: list[str] = None,
    ranges: dict = None,
    zorders: list[int] = None,
    save_name: str = "./figures/GW_PE/comparison_cornerplot.pdf",
    overwrite: bool = False,
    dummy_normalization_indices: list[int] = None,
    truths: list[float] = None,
    reverse_legend: bool = True,
    show_credible_intervals: bool = True,
    credible_interval_formats: dict = None
) -> bool:
    """
    Create a comparison corner plot with multiple datasets overlaid.

    Args:
        filepaths (list[str]): List of paths to .npz files
        parameters (list[str]): Parameters to include in the corner plot
        labels (list[str]): Labels for each dataset
        colors (list[str]): Colors for each dataset (optional, uses defaults)
        ranges (dict): Parameter ranges as {param: (min, max)} (optional)
        zorders (list[int]): Z-order for each dataset (higher values appear on top, optional)
        save_name (str): Output filename
        overwrite (bool): Whether to overwrite existing plots
        dummy_normalization_indices (list[int]): Indices of datasets to use for each parameter
            to create a dummy normalization dataset (optional)
        truths (list[float]): True values for each parameter to plot as reference lines (optional)
        reverse_legend (bool): Whether to reverse legend order (default True shows highest z-order first)
        show_credible_intervals (bool): Whether to show 90% credible intervals on 1D panels (default True)
        credible_interval_formats (dict): Custom format strings for credible intervals as {param: format_string}
            (e.g., {"chirp_mass": ".5f"}). If not provided, format is auto-determined from parameter range (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists, skipping...")
            return True

        print(f"Creating comparison corner plot for {len(filepaths)} datasets")
        print(f"Parameters: {parameters}")

        # Set default colors if not provided
        if colors is None:
            default_colors = [ORANGE, BLUE, GREEN, RED]
            colors = default_colors[:len(filepaths)]

        # Set default z-orders if not provided
        if zorders is None:
            zorders = list(range(len(filepaths)))

        # Validate inputs
        if len(labels) != len(filepaths):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of filepaths ({len(filepaths)})")

        if len(colors) != len(filepaths):
            raise ValueError(f"Number of colors ({len(colors)}) must match number of filepaths ({len(filepaths)})")

        # Load all datasets
        all_samples = []
        for i, filepath in enumerate(filepaths):
            print(f"\nLoading dataset {i+1}/{len(filepaths)}: {filepath}")
            print(f"  Label: {labels[i]}, Color: {colors[i]}, Z-order: {zorders[i]}")

            samples = load_npz_data(filepath, parameters)
            all_samples.append(samples)

        if not all_samples:
            print("No valid samples loaded!")
            return False

        # Sort all datasets by z-order for proper plotting (lower z-order first)
        sorted_data = sorted(zip(zorders, all_samples, labels, colors), key=lambda x: x[0])
        zorders_sorted, all_samples, labels, colors = zip(*sorted_data)
        zorders = list(zorders_sorted)
        all_samples = list(all_samples)
        labels = list(labels)
        colors = list(colors)

        # Create dummy normalization dataset if requested
        dummy_dataset = None
        if dummy_normalization_indices is not None:
            # Validate dummy_normalization_indices
            n_params = len(parameters)
            if len(dummy_normalization_indices) != n_params:
                raise ValueError(f"dummy_normalization_indices must have length {n_params} "
                               f"(number of parameters), got {len(dummy_normalization_indices)}")

            # Check that all indices are valid dataset indices
            n_datasets = len(all_samples)
            for i, idx in enumerate(dummy_normalization_indices):
                if not (0 <= idx < n_datasets):
                    raise ValueError(f"dummy_normalization_indices[{i}] = {idx} is not a valid "
                                   f"dataset index (0 to {n_datasets-1})")

            # Find minimum number of samples across all datasets
            min_samples = min(len(samples) for samples in all_samples)
            print(f"\nCreating dummy normalization dataset using dataset indices: {dummy_normalization_indices}")
            print(f"Using minimum sample count: {min_samples}")

            # Create dummy dataset by selecting specified dataset for each parameter
            dummy_columns = []
            for param_idx, dataset_idx in enumerate(dummy_normalization_indices):
                samples = all_samples[dataset_idx]
                # Downsample to min_samples by taking first min_samples rows
                downsampled_samples = samples[:min_samples]
                dummy_columns.append(downsampled_samples[:, param_idx])

            dummy_dataset = np.column_stack(dummy_columns)
            print(f"Created dummy dataset with shape: {dummy_dataset.shape}")

        # Create the corner plot with the first dataset
        print("\nCreating corner plot...")

        # Set up corner kwargs
        corner_kwargs = DEFAULT_CORNER_KWARGS.copy()
        corner_kwargs["color"] = colors[0]
        corner_kwargs["density"] = True  # Use density normalization for all histograms
        corner_kwargs["hist_kwargs"] = {"density": True}

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

        # Add truths if provided
        if truths is not None:
            corner_kwargs["truths"] = truths
            corner_kwargs["truth_color"] = "black"

        # Create initial plot
        corner_kwargs["labels"] = parameter_labels
        fig = corner.corner(all_samples[0], **corner_kwargs)

        # Overlay additional datasets
        for i in range(1, len(all_samples)):
            corner_kwargs_overlay = corner_kwargs.copy()
            # Deep copy the range to avoid modification issues
            if "range" in corner_kwargs_overlay:
                corner_kwargs_overlay["range"] = corner_kwargs_overlay["range"].copy()
            corner_kwargs_overlay["color"] = colors[i]
            corner_kwargs_overlay["hist_kwargs"] = {"color": colors[i], "density": True}
            corner_kwargs_overlay["density"] = True
            corner_kwargs_overlay["fig"] = fig
            corner_kwargs_overlay["labels"] = parameter_labels
            # Keep truths in overlay plots
            if truths is not None:
                corner_kwargs_overlay["truths"] = truths
                corner_kwargs_overlay["truth_color"] = "black"

            corner.corner(all_samples[i], **corner_kwargs_overlay)

        # Plot dummy normalization dataset last (invisible, for normalization)
        if dummy_dataset is not None:
            print("Adding dummy normalization dataset...")
            invisible_kwargs = DEFAULT_CORNER_KWARGS.copy()
            invisible_kwargs.update({
                'labels': parameter_labels,
                'hist_kwargs': {'alpha': 0, "density": True},  # Invisible histograms
                'plot_density': False,     # Disable 2D density plots
                'plot_contours': False,    # Disable 2D contours
                'plot_datapoints': False,  # Disable scatter points
                'no_fill_contours': True,  # Disable filled contours
                'color': 'black',
                'density': True,           # Use density normalization
                'fig': fig
            })

            # Apply parameter ranges if provided
            if ranges:
                range_list = []
                for param in parameters:
                    if param in ranges:
                        range_list.append(ranges[param])
                    else:
                        range_list.append(None)
                invisible_kwargs["range"] = range_list

            # Keep truths in dummy plot
            if truths is not None:
                invisible_kwargs["truths"] = truths
                invisible_kwargs["truth_color"] = "black"

            corner.corner(dummy_dataset, **invisible_kwargs)

        # Add credible interval annotations as titles on 1D panels
        if show_credible_intervals:
            print("Adding 90% credible interval annotations as titles...")

            # Get the axes from the figure (corner plot creates a grid of axes)
            axes = np.array(fig.axes).reshape((len(parameters), len(parameters)))

            # For each parameter (diagonal), compute credible intervals for all datasets
            for param_idx, param in enumerate(parameters):
                ax = axes[param_idx, param_idx]  # Get diagonal axis

                # Font size for credible intervals
                fontsize = 18

                # Add credible interval text for each dataset with matching colors
                for dataset_idx in range(len(all_samples)):
                    # Get samples for this parameter from this dataset
                    param_samples = all_samples[dataset_idx][:, param_idx]

                    # Compute 90% HDI credible interval using arviz
                    low, high = arviz.hdi(param_samples, hdi_prob=0.90)
                    med = np.median(param_samples)

                    # Format credible interval string
                    # Check if custom format is provided for this parameter
                    if credible_interval_formats is not None and param in credible_interval_formats:
                        fmt = credible_interval_formats[param]
                    else:
                        # Determine number of decimal places based on parameter range
                        param_range = np.ptp([low, high])
                        if param_range < 0.1:
                            fmt = ".4f"
                        elif param_range < 1:
                            fmt = ".3f"
                        elif param_range < 10:
                            fmt = ".2f"
                        else:
                            fmt = ".1f"

                    textstr = f"${med:{fmt}}^{{+{high - med:{fmt}}}}_{{-{med - low:{fmt}}}}$"

                    # Position text above the plot (stacked vertically for multiple datasets)
                    # Start from y=1.05 and stack downward with spacing
                    y_pos = 1.05 + (len(all_samples) - 1 - dataset_idx) * 0.20

                    # Add colored text
                    ax.text(0.5, y_pos, textstr,
                           transform=ax.transAxes,
                           verticalalignment='bottom',
                           horizontalalignment='center',
                           color=colors[dataset_idx],
                           fontsize=fontsize)

        # Add legend
        legend_elements = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='k', label=label)
            )

        # Reverse to show highest z-order (on top) at top of legend if requested
        if reverse_legend:
            legend_elements = legend_elements[::-1]

        fig.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), frameon=True)

        # Save plot
        ensure_directory_exists(save_name)
        print(f"\nSaving comparison corner plot to {save_name}")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

        return True

    except Exception as e:
        print(f"Failed to create comparison corner plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function for creating comparison corner plots.
    """

    # Base path for data
    base_path = "../posteriors/data/"

    # ====== COMPARISON 1: l5000 with different spin priors ======
    print("=" * 60)
    print("COMPARISON 1: l5000 with different spin priors")
    print("=" * 60)

    # Parameters to include in comparison 1
    parameters_1 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde"
    ]

    filepaths_1 = [
        os.path.join(base_path, "prod_BW_XP_s040_l5000_default.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_l5000_default.npz"),
    ]

    labels_1 = [
        r"$\chi_i \leq 0.4$",
        r"$\chi_i \leq 0.05$",
    ]

    colors_1 = [BLUE, ORANGE]

    zorders_1 = [0, 1]  # Low spin prior (chi<0.05) on top

    # Use default ranges
    ranges_1 = {param: DEFAULT_RANGES[param] for param in parameters_1}

    # Use chi<0.05 dataset (index 1) for normalization on all parameters
    dummy_indices_1 = [1] * len(parameters_1)

    success_1 = create_comparison_cornerplot(
        filepaths=filepaths_1,
        parameters=parameters_1,
        labels=labels_1,
        colors=colors_1,
        ranges=ranges_1,
        zorders=zorders_1,
        save_name="./figures/GW_PE/comparison_l5000_spin.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_1
    )

    if success_1:
        print(" Successfully created comparison 1: comparison_l5000_spin.pdf")
    else:
        print(" Failed to create comparison 1")

    # ====== COMPARISON 2: leos with different spin priors ======
    print("\n" + "=" * 60)
    print("COMPARISON 2: leos with different spin priors")
    print("=" * 60)

    # Parameters to include in comparison 2
    parameters_2 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde"
    ]

    filepaths_2 = [
        os.path.join(base_path, "prod_BW_XP_s040_leos_default_no_zeros.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_leos_default_no_zeros.npz"),
    ]

    labels_2 = [
        r"$\chi_i \leq 0.4$",
        r"$\chi_i \leq 0.05$",
    ]

    colors_2 = ["#2596be", "#ff642c"]

    zorders_2 = [0, 1]  # Low spin prior (chi<0.05) on top

    # Use default ranges but override lambda_tilde for EOS plot
    ranges_2 = {param: DEFAULT_RANGES[param] for param in parameters_2}
    ranges_2["lambda_tilde"] = (0, 1000)  # Tighter range for EOS plot

    # Print Lambda statistics for EOS runs
    print("\nLambda parameter statistics for EOS runs:")
    for i, (filepath, label) in enumerate(zip(filepaths_2, labels_2)):
        data = np.load(filepath)
        print(f"\n  {label}:")

        # Lambda_1
        lambda_1 = data['lambda_1']
        print(f"    Lambda_1:")
        print(f"      Mean: {np.mean(lambda_1):.2f}, Median: {np.median(lambda_1):.2f}, Std: {np.std(lambda_1):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_1, 5):.2f} - {np.percentile(lambda_1, 95):.2f}")

        # Lambda_2
        lambda_2 = data['lambda_2']
        print(f"    Lambda_2:")
        print(f"      Mean: {np.mean(lambda_2):.2f}, Median: {np.median(lambda_2):.2f}, Std: {np.std(lambda_2):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_2, 5):.2f} - {np.percentile(lambda_2, 95):.2f}")

        # Lambda_tilde
        lambda_tilde = data['lambda_tilde']
        print(f"    Lambda_tilde:")
        print(f"      Mean: {np.mean(lambda_tilde):.2f}, Median: {np.median(lambda_tilde):.2f}, Std: {np.std(lambda_tilde):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_tilde, 5):.2f} - {np.percentile(lambda_tilde, 95):.2f}")

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.4 (index 0) for lambda_tilde
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde
    dummy_indices_2 = [1, 1, 1, 0]  # Use high spin (chi<0.4) for lambda_tilde normalization

    success_2 = create_comparison_cornerplot(
        filepaths=filepaths_2,
        parameters=parameters_2,
        labels=labels_2,
        colors=colors_2,
        ranges=ranges_2,
        zorders=zorders_2,
        save_name="./figures/GW_PE/comparison_leos_spin.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_2
    )

    if success_2:
        print(" Successfully created comparison 2: comparison_leos_spin.pdf")
    else:
        print(" Failed to create comparison 2")

    # ====== COMPARISON 2 DEBUG: leos with additional parameters ======
    print("\n" + "=" * 60)
    print("COMPARISON 2 DEBUG: leos with additional parameters")
    print("=" * 60)

    # Parameters to include in comparison 2 debug (extended version)
    parameters_2_debug = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde",
        "mass_1_source",
        "mass_2_source",
        "lambda_1",
        "lambda_2"
    ]

    # Use same filepaths and labels as comparison 2
    filepaths_2_debug = filepaths_2
    labels_2_debug = labels_2
    colors_2_debug = colors_2
    zorders_2_debug = zorders_2

    # Use default ranges but override lambda parameters
    ranges_2_debug = {param: DEFAULT_RANGES.get(param) for param in parameters_2_debug}
    ranges_2_debug["lambda_tilde"] = (0, 1000)  # Same as comparison 2
    ranges_2_debug["lambda_1"] = (0, 500)
    ranges_2_debug["lambda_2"] = (0, 1000)

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.4 (index 0) for lambda parameters
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde, mass_1_source, mass_2_source, lambda_1, lambda_2
    dummy_indices_2_debug = [1, 1, 1, 0, 1, 1, 0, 0]  # Use high spin for lambda params

    success_2_debug = create_comparison_cornerplot(
        filepaths=filepaths_2_debug,
        parameters=parameters_2_debug,
        labels=labels_2_debug,
        colors=colors_2_debug,
        ranges=ranges_2_debug,
        zorders=zorders_2_debug,
        save_name="./figures/GW_PE/comparison_leos_spin_debug.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_2_debug
    )

    if success_2_debug:
        print("✓ Successfully created comparison 2 debug: comparison_leos_spin_debug.pdf")
    else:
        print("✗ Failed to create comparison 2 debug")

    # ====== COMPARISON 2 DEBUG WITH EOS: leos with EOS parameter ======
    print("\n" + "=" * 60)
    print("COMPARISON 2 DEBUG WITH EOS: leos with EOS parameter")
    print("=" * 60)

    # Parameters to include in comparison 2 debug with EOS (extended version)
    parameters_2_debug_eos = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde",
        "EOS"
    ]

    # Use same filepaths and labels as comparison 2
    filepaths_2_debug_eos = filepaths_2
    labels_2_debug_eos = labels_2
    colors_2_debug_eos = colors_2
    zorders_2_debug_eos = zorders_2

    # Use default ranges but override EOS parameter
    ranges_2_debug_eos = {param: DEFAULT_RANGES.get(param) for param in parameters_2_debug_eos}
    ranges_2_debug_eos["lambda_tilde"] = (0, 1000)  # Same as comparison 2
    ranges_2_debug_eos["EOS"] = (0, 5000)

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.4 (index 0) for lambda_tilde and EOS
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde, EOS
    dummy_indices_2_debug_eos = [1, 1, 1, 0, 0]  # Use high spin for lambda_tilde and EOS

    success_2_debug_eos = create_comparison_cornerplot(
        filepaths=filepaths_2_debug_eos,
        parameters=parameters_2_debug_eos,
        labels=labels_2_debug_eos,
        colors=colors_2_debug_eos,
        ranges=ranges_2_debug_eos,
        zorders=zorders_2_debug_eos,
        save_name="./figures/GW_PE/comparison_leos_spin_debug_eos.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_2_debug_eos
    )

    if success_2_debug_eos:
        print("✓ Successfully created comparison 2 debug EOS: comparison_leos_spin_debug_eos.pdf")
    else:
        print("✗ Failed to create comparison 2 debug EOS")

    # ====== COMPARISON 2 WITH ZEROS: leos with different spin priors (including zero Lambdas) ======
    print("\n" + "=" * 60)
    print("COMPARISON 2 WITH ZEROS: leos with different spin priors (including zero Lambdas)")
    print("=" * 60)

    # Parameters to include in comparison 2 with zeros
    parameters_2_zeros = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde"
    ]

    filepaths_2_zeros = [
        os.path.join(base_path, "prod_BW_XP_s040_leos_default.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_leos_default.npz"),
    ]

    labels_2_zeros = [
        r"$\chi_i \leq 0.4$",
        r"$\chi_i \leq 0.05$",
    ]

    colors_2_zeros = ["#2596be", "#ff642c"]

    zorders_2_zeros = [0, 1]  # Low spin prior (chi<0.05) on top

    # Use default ranges but override lambda_tilde for EOS plot
    ranges_2_zeros = {param: DEFAULT_RANGES[param] for param in parameters_2_zeros}
    ranges_2_zeros["lambda_tilde"] = (0, 1000)  # Tighter range for EOS plot

    # Print Lambda statistics for EOS runs with zeros
    print("\nLambda parameter statistics for EOS runs (with zeros):")
    for i, (filepath, label) in enumerate(zip(filepaths_2_zeros, labels_2_zeros)):
        data = np.load(filepath)
        print(f"\n  {label}:")

        # Lambda_1
        lambda_1 = data['lambda_1']
        print(f"    Lambda_1:")
        print(f"      Mean: {np.mean(lambda_1):.2f}, Median: {np.median(lambda_1):.2f}, Std: {np.std(lambda_1):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_1, 5):.2f} - {np.percentile(lambda_1, 95):.2f}")
        print(f"      Number of zeros: {np.sum(lambda_1 == 0)}")

        # Lambda_2
        lambda_2 = data['lambda_2']
        print(f"    Lambda_2:")
        print(f"      Mean: {np.mean(lambda_2):.2f}, Median: {np.median(lambda_2):.2f}, Std: {np.std(lambda_2):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_2, 5):.2f} - {np.percentile(lambda_2, 95):.2f}")
        print(f"      Number of zeros: {np.sum(lambda_2 == 0)}")

        # Lambda_tilde
        lambda_tilde = data['lambda_tilde']
        print(f"    Lambda_tilde:")
        print(f"      Mean: {np.mean(lambda_tilde):.2f}, Median: {np.median(lambda_tilde):.2f}, Std: {np.std(lambda_tilde):.2f}")
        print(f"      5th-95th percentile: {np.percentile(lambda_tilde, 5):.2f} - {np.percentile(lambda_tilde, 95):.2f}")
        print(f"      Number of zeros: {np.sum(lambda_tilde == 0)}")

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.4 (index 0) for lambda_tilde
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde
    dummy_indices_2_zeros = [1, 1, 1, 0]  # Use high spin (chi<0.4) for lambda_tilde normalization

    success_2_zeros = create_comparison_cornerplot(
        filepaths=filepaths_2_zeros,
        parameters=parameters_2_zeros,
        labels=labels_2_zeros,
        colors=colors_2_zeros,
        ranges=ranges_2_zeros,
        zorders=zorders_2_zeros,
        save_name="./figures/GW_PE/comparison_leos_spin_with_zeros.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_2_zeros
    )

    if success_2_zeros:
        print("✓ Successfully created comparison 2 with zeros: comparison_leos_spin_with_zeros.pdf")
    else:
        print("✗ Failed to create comparison 2 with zeros")

    # ====== COMPARISON 2 WITH ZEROS DEBUG WITH EOS: leos with EOS parameter (including zeros) ======
    print("\n" + "=" * 60)
    print("COMPARISON 2 WITH ZEROS DEBUG WITH EOS: leos with EOS parameter (including zeros)")
    print("=" * 60)

    # Parameters to include in comparison 2 with zeros debug with EOS
    parameters_2_zeros_debug_eos = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde",
        "EOS"
    ]

    # Use same filepaths and labels as comparison 2 with zeros
    filepaths_2_zeros_debug_eos = filepaths_2_zeros
    labels_2_zeros_debug_eos = labels_2_zeros
    colors_2_zeros_debug_eos = colors_2_zeros
    zorders_2_zeros_debug_eos = zorders_2_zeros

    # Use default ranges but override EOS parameter
    ranges_2_zeros_debug_eos = {param: DEFAULT_RANGES.get(param) for param in parameters_2_zeros_debug_eos}
    ranges_2_zeros_debug_eos["lambda_tilde"] = (0, 1000)  # Same as comparison 2
    ranges_2_zeros_debug_eos["EOS"] = (0, 5000)

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.4 (index 0) for lambda_tilde and EOS
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde, EOS
    dummy_indices_2_zeros_debug_eos = [1, 1, 1, 0, 0]  # Use high spin for lambda_tilde and EOS

    success_2_zeros_debug_eos = create_comparison_cornerplot(
        filepaths=filepaths_2_zeros_debug_eos,
        parameters=parameters_2_zeros_debug_eos,
        labels=labels_2_zeros_debug_eos,
        colors=colors_2_zeros_debug_eos,
        ranges=ranges_2_zeros_debug_eos,
        zorders=zorders_2_zeros_debug_eos,
        save_name="./figures/GW_PE/comparison_leos_spin_with_zeros_debug_eos.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_2_zeros_debug_eos
    )

    if success_2_zeros_debug_eos:
        print("✓ Successfully created comparison 2 with zeros debug EOS: comparison_leos_spin_with_zeros_debug_eos.pdf")
    else:
        print("✗ Failed to create comparison 2 with zeros debug EOS")


    # # Summary
    # print("\n" + "=" * 60)
    # print("SUMMARY")
    # print("=" * 60)

    # # ====== COMPARISON 3: ET vs ET+CE ======
    # print("\n" + "=" * 60)
    # print("COMPARISON 3: ET vs ET+CE")
    # print("=" * 60)

    # Parameters to include in comparison 3
    parameters_3 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde"
    ]

    filepaths_3 = [
        os.path.join(base_path, "jester_eos_et_run_alignedspin.npz"),
        os.path.join(base_path, "jester_eos_et_ce_run_alignedspin.npz"),
    ]

    labels_3 = [
        "ET",
        "ET+CE",
    ]

    # Colors from money_plots_snellius.py
    ET_COLOR = "#de8f05"
    ET_CE_COLOR = "#d45d01"
    colors_3 = [ET_COLOR, ET_CE_COLOR]

    zorders_3 = [0, 1]  # ET+CE on top

    # Use ET+CE dataset (index 1) for normalization on most parameters, but ET (index 0) for chi_eff
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde
    dummy_indices_3 = [1, 1, 0, 1]  # Use ET (index 0) for chi_eff normalization

    # Put the new injection parameters here
    injection_parameters = {"mass_1": 1.5879187040159342,
                            "mass_2": 1.4188967691574992,
                            "geocent_time": 1383609314.0505133,
                            "a_1": 0.0305182034770227,
                            "a_2": 0.028570123024914268,
                            "phi_12": 0.0,
                            "phi_jl": 0.0,
                            "psi": 1.5591681494817768,
                            "theta_jn": 2.5287713998365304,
                            "ra": 0.1778415509774547,
                            "dec": -0.602827165369817,
                            "tilt_1": 0.0,
                            "tilt_2": 0.0,
                            "phase": 3.139490467696903,
                            "luminosity_distance": 168.3222418883087,
                            'lambda_1': 271.02342967819004,
                            'lambda_2': 553.1640516248044
                            }
    
    # Add chirp mass, mass ratio and lambda_tilde
    chirp_mass = component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters['mass_2'])
    mass_ratio = component_masses_to_mass_ratio(injection_parameters['mass_1'], injection_parameters['mass_2'])
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(injection_parameters['lambda_1'], injection_parameters['lambda_2'],
                                                  injection_parameters['mass_1'], injection_parameters['mass_2'])
    injection_parameters['chirp_mass'] = chirp_mass
    injection_parameters['mass_ratio'] = mass_ratio
    injection_parameters['lambda_tilde'] = lambda_tilde

    # This is also necessary for the spin calculation
    injection_parameters["reference_frequency"] = 5.0
    injection_parameters = generate_spin_parameters(injection_parameters)

    # Compute source frame masses from detector frame masses
    # First compute redshift from luminosity distance using bilby
    z = luminosity_distance_to_redshift(injection_parameters['luminosity_distance'])

    # Convert detector frame masses to source frame
    injection_parameters['mass_1_source'] = injection_parameters['mass_1'] / (1 + z)
    injection_parameters['mass_2_source'] = injection_parameters['mass_2'] / (1 + z)
    
    # If jester was in the EOS, then get
    filename = "../figures/EOS_data/jester_GW170817_maxL_EOS.npz"
    eos_data = np.load(filename)
    
    print(list(eos_data.keys()))
    
    # Interpolate Lambdas again
    lambda_1_interp = np.interp(injection_parameters['mass_1_source'], eos_data['masses'], eos_data['Lambdas'])
    lambda_2_interp = np.interp(injection_parameters['mass_2_source'], eos_data['masses'], eos_data['Lambdas'])
    injection_parameters['lambda_1'] = lambda_1_interp
    injection_parameters['lambda_2'] = lambda_2_interp
    injection_parameters['lambda_tilde'] = lambda_1_lambda_2_to_lambda_tilde(lambda_1_interp, lambda_2_interp,
                                                  injection_parameters['mass_1'], injection_parameters['mass_2'])


    truths_3 = [injection_parameters[param] for param in parameters_3]
    ranges_3 = {
        "chirp_mass": (1.3063+0.1e-5, 1.3063+2.2e-5),
        "mass_ratio": (0.78, 1.0),
        "chi_eff": (0.027, 0.035),
        "lambda_tilde": (250, 450)
    }
    
    success_4 = create_comparison_cornerplot(
        filepaths=filepaths_3,
        parameters=parameters_3,
        labels=labels_3,
        colors=colors_3,
        ranges=ranges_3,
        zorders=zorders_3,
        save_name="./figures/GW_PE/comparison_new_ET_vs_ET_CE.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_3,
        truths=truths_3,
        reverse_legend=False,
        show_credible_intervals=True,
        credible_interval_formats={"chirp_mass": ".5f"}
    )

    if success_4:
        print(" Successfully created comparison 4: comparison_new_ET_vs_ET_CE.pdf")
    else:
        print(" Failed to create comparison 4")

    # ====== COMPARISON 4 DEBUG: ET vs ET+CE with component masses and Lambdas ======
    print("\n" + "=" * 60)
    print("COMPARISON 4 DEBUG: ET vs ET+CE with component masses and Lambdas")
    print("=" * 60)

    # Parameters to include in comparison 4 debug
    parameters_4_debug = [
        "mass_1_source",
        "mass_2_source",
        "lambda_1",
        "lambda_2"
    ]

    # Use same filepaths and labels as comparison 4
    filepaths_4_debug = filepaths_3
    labels_4_debug = labels_3
    colors_4_debug = colors_3
    zorders_4_debug = zorders_3

    # Use None for all ranges
    ranges_4_debug = None

    # Extract truths for the new parameters
    truths_4_debug = [injection_parameters[param] for param in parameters_4_debug]

    # Use ET+CE dataset (index 1) for normalization on all parameters
    dummy_indices_4_debug = [1] * len(parameters_4_debug)

    success_4_debug = create_comparison_cornerplot(
        filepaths=filepaths_4_debug,
        parameters=parameters_4_debug,
        labels=labels_4_debug,
        colors=colors_4_debug,
        ranges=ranges_4_debug,
        zorders=zorders_4_debug,
        save_name="./figures/GW_PE/comparison_new_ET_vs_ET_CE_DEBUG.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_4_debug,
        truths=truths_4_debug,
        reverse_legend=False,
        show_credible_intervals=False
    )

    if success_4_debug:
        print(" Successfully created comparison 4 debug: comparison_new_ET_vs_ET_CE_DEBUG.pdf")
    else:
        print(" Failed to create comparison 4 debug")

    # ====== COMPARISON 5: Low spin prior with default vs Gaussian chirp mass prior ======
    print("\n" + "=" * 60)
    print("COMPARISON 5: Low spin prior with default vs Gaussian chirp mass prior")
    print("=" * 60)

    # Parameters to include in comparison 5
    parameters_5 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_1",
        "lambda_2",
        "lambda_tilde"
    ]

    filepaths_5 = [
        os.path.join(base_path, "prod_BW_XP_s005_l5000_default.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_l5000_gaussian.npz"),
    ]

    labels_5 = [
        r"Default",
        r"Gaussian",
    ]

    colors_5 = [ORANGE, BLUE]

    zorders_5 = [0, 1]  # Gaussian on top

    # Use default ranges
    ranges_5 = {param: DEFAULT_RANGES[param] for param in parameters_5}
    ranges_5["lambda_1"] = (0, 5000)
    ranges_5["lambda_2"] = (0, 5000)

    # Use Gaussian dataset (index 1) for normalization on all parameters
    dummy_indices_5 = [1] * len(parameters_5)

    success_5 = create_comparison_cornerplot(
        filepaths=filepaths_5,
        parameters=parameters_5,
        labels=labels_5,
        colors=colors_5,
        ranges=ranges_5,
        zorders=zorders_5,
        save_name="./figures/GW_PE/comparison_s005_l5000_default_vs_gaussian.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_5
    )

    if success_5:
        print("✓ Successfully created comparison 5: comparison_s005_l5000_default_vs_gaussian.pdf")
    else:
        print("✗ Failed to create comparison 5")

    # ====== COMPARISON 6: Low spin prior with default vs double_gaussian chirp mass prior ======
    print("\n" + "=" * 60)
    print("COMPARISON 6: Low spin prior with default vs double_gaussian chirp mass prior")
    print("=" * 60)

    # Parameters to include in comparison 6
    parameters_6 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_1",
        "lambda_2",
        "lambda_tilde"
    ]

    filepaths_6 = [
        os.path.join(base_path, "prod_BW_XP_s005_l5000_default.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_l5000_double_gaussian.npz"),
    ]

    labels_6 = [
        r"Default",
        r"Double Gaussian",
    ]

    colors_6 = [ORANGE, GREEN]

    zorders_6 = [0, 1]  # Double Gaussian on top

    # Use default ranges
    ranges_6 = {param: DEFAULT_RANGES[param] for param in parameters_6}
    ranges_6["lambda_1"] = (0, 5000)
    ranges_6["lambda_2"] = (0, 5000)

    # Use Double Gaussian dataset (index 1) for normalization on all parameters
    dummy_indices_6 = [1] * len(parameters_6)

    success_6 = create_comparison_cornerplot(
        filepaths=filepaths_6,
        parameters=parameters_6,
        labels=labels_6,
        colors=colors_6,
        ranges=ranges_6,
        zorders=zorders_6,
        save_name="./figures/GW_PE/comparison_s005_l5000_default_vs_double_gaussian.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_6
    )

    if success_6:
        print("✓ Successfully created comparison 6: comparison_s005_l5000_default_vs_double_gaussian.pdf")
    else:
        print("✗ Failed to create comparison 6")


if __name__ == "__main__":
    main()
