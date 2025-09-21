"""Money plots script for jester inference results on Snellius.

This script generates comparison figures from multiple jester inference output directories:
- Comparison histograms of MTOV, R14, and other EOS parameters overlaying all datasets
- Mass-radius comparison plots with different colors for each dataset
- Pressure-density (EOS) comparison plots overlaying all datasets

The script creates single plots comparing all specified directories, rather than
separate plots for each directory. All plots use the 'crest' colormap for consistency.

Usage:
    Modify the main() function to specify directories and colors, then run:
    python money_plots_snellius.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tqdm
import arviz
from scipy.stats import gaussian_kde
import sys

# Add jester directory to path for imports
sys.path.append('../jester')
import jesterTOV.utils as jose_utils

# Matplotlib parameters
mpl_params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)

# Constants
COLORS_DICT = {"prior": "gray",
               "GW170817": "orange",
               "GW231109": "teal",
               "GW231109_only": "red"}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

def load_eos_data(outdir: str):
    """Load EOS data from the specified output directory."""
    filename = os.path.join(outdir, "eos_samples.npz")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"EOS samples file not found: {filename}")

    print(f"Loading data from {filename}")
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]

    # Convert units
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric

    log_prob = data["log_prob"]

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

def report_credible_interval(values: np.array,
                             hdi_prob: float = 0.90,
                             verbose: bool = False) -> tuple:
    """Calculate credible intervals for given values."""
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob=hdi_prob)

    low = med - low
    high = high - med

    if verbose:
        print(f"{med:.2f}-{low:.2f}+{high:.2f} (at {hdi_prob} HDI prob)")
    return low, med, high

def make_parameter_histograms(data_list: list, outdir_names: list, colors: list, save_suffix: str = ""):
    """Create comparison histograms for key EOS parameters across multiple datasets.

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        save_suffix: Optional suffix for filename
    """
    print(f"Creating parameter comparison histograms for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    # Define parameter ranges and labels
    parameter_configs = {
        'MTOV': {'range': (1.75, 2.75), 'xlabel': r"$M_{\rm{TOV}}$ [$M_{\odot}$]"},
        'R14': {'range': (10.0, 16.0), 'xlabel': r"$R_{1.4}$ [km]"},
        'p3nsat': {'range': (0.1, 200.0), 'xlabel': r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]"}
    }

    # Calculate parameters for all datasets
    all_parameters = {}
    for i, data_dict in enumerate(data_list):
        m, r = data_dict['masses'], data_dict['radii']
        n, p = data_dict['densities'], data_dict['pressures']

        # Calculate derived parameters
        MTOV_list = np.array([np.max(mass) for mass in m])
        R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
        p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

        all_parameters[i] = {
            'MTOV': MTOV_list,
            'R14': R14_list,
            'p3nsat': p3nsat_list
        }

    # Create comparison plots for each parameter
    for param_name, config in parameter_configs.items():
        plt.figure(figsize=figsize_horizontal)

        for i, (data_dict, outdir_name, color) in enumerate(zip(data_list, outdir_names, colors)):
            param_values = all_parameters[i][param_name]

            # Create posterior KDE
            kde = gaussian_kde(param_values)
            x = np.linspace(config['range'][0], config['range'][1], 1000)
            y = kde(x)

            # Get directory basename for label
            dir_basename = os.path.basename(outdir_name.rstrip('/'))

            # Add credible interval to label
            low, med, high = report_credible_interval(param_values)
            label = f'{dir_basename}: {med:.2f} -{low:.2f} +{high:.2f}'

            plt.plot(x, y, color=color, lw=3.0, label=label)
            plt.fill_between(x, y, alpha=0.3, color=color)

        plt.xlabel(config['xlabel'])
        plt.ylabel('Density')
        plt.xlim(config['range'])
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.title(f'{param_name} Comparison')

        # Save comparison plot
        save_name = os.path.join("./figures", f"comparison_{param_name}_histogram{save_suffix}.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"  {param_name} comparison histogram saved to {save_name}")

def make_mass_radius_plot(data_list: list, outdir_names: list, colors: list, save_suffix: str = ""):
    """Create comparison mass-radius plot with different colors for each dataset.

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        save_suffix: Optional suffix for filename
    """
    print(f"Creating mass-radius comparison plot for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    plt.figure(figsize=(6, 12))
    m_min, m_max = 0.75, 3.5
    r_min, r_max = 6.0, 18.0

    # Plot each dataset with its assigned color
    for dataset_idx, (data_dict, outdir_name, color) in enumerate(zip(data_list, outdir_names, colors)):
        m, r, l = data_dict['masses'], data_dict['radii'], data_dict['lambdas']
        log_prob = data_dict['log_prob']
        nb_samples = np.shape(m)[0]

        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        print(f"  Dataset {dir_basename}: {nb_samples} samples")

        # Normalize probabilities for alpha blending within this dataset
        log_prob_norm = np.exp(log_prob)
        prob_min, prob_max = np.min(log_prob_norm), np.max(log_prob_norm)

        bad_counter = 0
        for i in tqdm.tqdm(range(len(log_prob)), desc=f"Plotting {dir_basename}"):
            # Skip invalid samples
            if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
                bad_counter += 1
                continue

            if any(l[i] < 0):
                bad_counter += 1
                continue

            if any((m[i] > 1.0) * (r[i] > 20.0)):
                bad_counter += 1
                continue

            # Use probability for alpha and zorder, but fixed color per dataset
            normalized_prob = (log_prob_norm[i] - prob_min) / (prob_max - prob_min)
            alpha = 0.3 + 0.7 * normalized_prob  # Alpha between 0.3 and 1.0

            plt.plot(r[i], m[i],
                    color=color,
                    alpha=alpha,
                    rasterized=True,
                    zorder=1e10 * dataset_idx + normalized_prob,
                    linewidth=0.5)

        print(f"    Excluded {bad_counter} invalid samples from {dir_basename}")

    # Styling
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(r_min, r_max)
    plt.ylim(m_min, m_max)

    # Add legend
    legend_elements = []
    for outdir_name, color in zip(outdir_names, colors):
        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=dir_basename))

    plt.legend(handles=legend_elements, loc='upper right')

    # Save comparison figure
    save_name = os.path.join("./figures", f"comparison_mass_radius_plot{save_suffix}.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Mass-radius comparison plot saved to {save_name}")

def make_pressure_density_plot(data_list: list, outdir_names: list, colors: list, save_suffix: str = ""):
    """Create comparison equation of state plot (pressure vs density).

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        save_suffix: Optional suffix for filename
    """
    print(f"Creating pressure-density comparison plot for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    plt.figure(figsize=(11, 6))

    # Plot each dataset with its assigned color
    for dataset_idx, (data_dict, outdir_name, color) in enumerate(zip(data_list, outdir_names, colors)):
        m, r, l = data_dict['masses'], data_dict['radii'], data_dict['lambdas']
        n, p = data_dict['densities'], data_dict['pressures']
        log_prob = data_dict['log_prob']

        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        print(f"  Dataset {dir_basename}: processing pressure-density curves")

        # Normalize probabilities for alpha blending within this dataset
        log_prob_norm = np.exp(log_prob)
        prob_min, prob_max = np.min(log_prob_norm), np.max(log_prob_norm)

        bad_counter = 0
        for i in tqdm.tqdm(range(len(log_prob)), desc=f"Plotting {dir_basename}"):
            # Skip invalid samples
            if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
                bad_counter += 1
                continue

            if any(l[i] < 0):
                bad_counter += 1
                continue

            if any((m[i] > 1.0) * (r[i] > 20.0)):
                bad_counter += 1
                continue

            # Use probability for alpha and zorder, but fixed color per dataset
            normalized_prob = (log_prob_norm[i] - prob_min) / (prob_max - prob_min)
            alpha = 0.3 + 0.7 * normalized_prob  # Alpha between 0.3 and 1.0

            # Plot pressure-density curve
            mask = (n[i] > 0.5) * (n[i] < 6.0)
            plt.plot(n[i][mask], p[i][mask],
                    color=color,
                    alpha=alpha,
                    rasterized=True,
                    zorder=1e10 * dataset_idx + normalized_prob,
                    linewidth=0.5)

        print(f"    Excluded {bad_counter} invalid samples from {dir_basename}")

    # Styling
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.yscale('log')
    plt.xlim(0.5, 6.0)

    # Add legend
    legend_elements = []
    for outdir_name, color in zip(outdir_names, colors):
        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=dir_basename))

    plt.legend(handles=legend_elements, loc='upper left')

    # Save comparison figure
    save_name = os.path.join("./figures", f"comparison_pressure_density_plot{save_suffix}.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Pressure-density comparison plot saved to {save_name}")

def load_all_data(directories: list):
    """Load EOS data from all specified directories.

    Args:
        directories: List of directory paths

    Returns:
        tuple: (data_list, valid_directories) - loaded data and corresponding valid directories
    """
    data_list = []
    valid_directories = []

    for outdir in directories:
        print(f"\n{'='*60}")
        print(f"Loading data from directory: {outdir}")
        print(f"{'='*60}")

        # Check if directory exists
        if not os.path.exists(outdir):
            print(f"Warning: Directory {outdir} does not exist. Skipping...")
            continue

        # Load data
        try:
            data = load_eos_data(outdir)
            print("Data loaded successfully!")
            data_list.append(data)
            valid_directories.append(outdir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
            continue

    return data_list, valid_directories

def main():
    """Main function - loads all data and creates comparison plots."""

    # =======================================================================
    # USER CONFIGURATION - MODIFY THIS SECTION
    # =======================================================================

    # List of directories to process (relative or absolute paths)
    directories = [
        "outdir",
        "outdir_radio",
        "outdir_GW231109",
        # Add more directories as needed
    ]

    # Colors for comparison plots (one per directory)
    # Options: 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'
    colors = [
        'blue',
        'red',
        'green',
        # Add more colors as needed (should match number of directories)
    ]

    # Optional suffix for all output files
    save_suffix = ""

    # =======================================================================
    # END USER CONFIGURATION
    # =======================================================================

    print("Money Plots Generator for Jester Inference Results")
    print("=" * 60)
    print(f"Processing {len(directories)} directories for comparison plots...")

    # Ensure we have enough colors
    if len(colors) < len(directories):
        print(f"Warning: Only {len(colors)} colors provided for {len(directories)} directories.")
        print("Extending with default colors...")
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        while len(colors) < len(directories):
            colors.append(default_colors[len(colors) % len(default_colors)])

    # Load all data
    data_list, valid_directories = load_all_data(directories)

    if len(data_list) == 0:
        print("Error: No valid data directories found!")
        return

    # Match colors to valid directories
    valid_colors = []
    for valid_dir in valid_directories:
        try:
            idx = directories.index(valid_dir)
            valid_colors.append(colors[idx])
        except ValueError:
            # This shouldn't happen, but fallback to first color
            valid_colors.append(colors[0])

    print(f"\n{'='*60}")
    print(f"Creating comparison plots for {len(data_list)} valid datasets...")
    print(f"{'='*60}")

    # Create all comparison plots
    try:
        make_parameter_histograms(data_list, valid_directories, valid_colors, save_suffix)
        make_mass_radius_plot(data_list, valid_directories, valid_colors, save_suffix)
        make_pressure_density_plot(data_list, valid_directories, valid_colors, save_suffix)
        print(f"\nAll comparison plots generated successfully!")
    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        return

    print(f"\n{'='*60}")
    print(f"Summary: Successfully created comparison plots for {len(data_list)} datasets")
    print(f"Valid datasets: {[os.path.basename(d.rstrip('/')) for d in valid_directories]}")
    print(f"{'='*60}")
    print("All figures saved to ./figures/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()