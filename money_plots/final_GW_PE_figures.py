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

# If running on Mac, so we can use TeX (not on Jarvis), change some rc params
cwd = os.getcwd()
if "Woute029" in cwd:
    print(f"Updating plotting parameters for TeX")
    fs = 18
    ticks_fs = 20
    label_fs = 22  # Bigger labels
    legend_fs = 22  # Smaller legend
    labelpad = 15  # Bigger labelpad
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
    "phi_jl": r"$\phi_{JL}$"
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
    "labelpad": 0.05,
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
    truths: list[float] = None
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

        # Add legend (reverse order so highest z-order appears first)
        legend_elements = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='k', label=label)
            )

        # Reverse to show highest z-order (on top) at top of legend
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
        r"$|\chi_i| \leq 0.40$",
        r"$|\chi_i| \leq 0.05$",
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
        os.path.join(base_path, "prod_BW_XP_s040_leos_default.npz"),
        os.path.join(base_path, "prod_BW_XP_s005_leos_default.npz"),
    ]

    labels_2 = [
        r"$|\chi_i| \leq 0.40$",
        r"$|\chi_i| \leq 0.05$",
    ]

    colors_2 = [RED, GREEN]

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

    # Use chi<0.05 EOS dataset (index 1) for most parameters, but chi<0.40 (index 0) for lambda_tilde
    # Parameters order: chirp_mass, mass_ratio, chi_eff, lambda_tilde
    dummy_indices_2 = [1, 1, 1, 0]  # Use high spin (chi<0.40) for lambda_tilde normalization

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

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success_1:
        print(" Comparison 1 (l5000 spin): comparison_l5000_spin.pdf")
    else:
        print(" Comparison 1 (l5000 spin): FAILED")

    if success_2:
        print(" Comparison 2 (leos spin): comparison_leos_spin.pdf")
    else:
        print(" Comparison 2 (leos spin): FAILED")
    # ====== COMPARISON 3: ET vs ET+CE ======
    print("\n" + "=" * 60)
    print("COMPARISON 3: ET vs ET+CE")
    print("=" * 60)

    # Parameters to include in comparison 3
    parameters_3 = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_tilde"
    ]

    filepaths_3 = [
        os.path.join(base_path, "et_run_alignedspin.npz"),
        os.path.join(base_path, "et_ce_run_alignedspin.npz"),
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

    # Use ET+CE dataset (index 1) for normalization on all parameters
    dummy_indices_3 = [1] * len(parameters_3)

    # Load injection values for truths
    injection_file = "./data/injection_ETCE_gw231109_injectionxas_a3eaf212_chi_eff_chirp_mass_lambda_1_lambda_2_lambda_tilde_mass_ratio.npz"
    injection_data = np.load(injection_file, allow_pickle=True)
    injection_params = injection_data['injection_params'].item()

    # Extract truths in the same order as parameters_3
    truths_3 = [injection_params[param] for param in parameters_3]
    print(f"Injection values: {dict(zip(parameters_3, truths_3))}")

    # Load data to compute chirp_mass quantiles
    print("\nComputing ranges for ET vs ET+CE comparison...")
    all_chirp_mass = []
    for filepath in filepaths_3:
        data = np.load(filepath)
        all_chirp_mass.append(data['chirp_mass'])
    all_chirp_mass = np.concatenate(all_chirp_mass)
    chirp_mass_min = np.quantile(all_chirp_mass, 0.001)
    chirp_mass_max = np.quantile(all_chirp_mass, 0.99)
    print(f"  Chirp mass range (0.01-0.99 quantiles): ({chirp_mass_min:.6f}, {chirp_mass_max:.6f})")

    # Set custom ranges
    ranges_3 = {
        "chirp_mass": (chirp_mass_min, chirp_mass_max),
        "mass_ratio": (0.82, 1.0),
        "chi_eff": (0.029, 0.035),
        "lambda_tilde": (220, 400)
    }
    print(f"  Final ranges: {ranges_3}")

    success_3 = create_comparison_cornerplot(
        filepaths=filepaths_3,
        parameters=parameters_3,
        labels=labels_3,
        colors=colors_3,
        ranges=ranges_3,
        zorders=zorders_3,
        save_name="./figures/GW_PE/comparison_ET_vs_ET_CE.pdf",
        overwrite=True,
        dummy_normalization_indices=dummy_indices_3,
        truths=truths_3
    )

    if success_3:
        print(" Successfully created comparison 3: comparison_ET_vs_ET_CE.pdf")
    else:
        print(" Failed to create comparison 3")

    # Update summary
    if success_3:
        print(" Comparison 3 (ET vs ET+CE): comparison_ET_vs_ET_CE.pdf")
    else:
        print(" Comparison 3 (ET vs ET+CE): FAILED")


if __name__ == "__main__":
    main()
