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

INJECTION_COLOR = "#cb78bd" # avoid importing since bilby not on Snellius

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

LABELS_DICT = {"outdir": "Prior", 
               "outdir_radio": "Radio timing",
               "outdir_GW170817": "+GW170817",
               "outdir_GW231109": "+GW231109",
               "outdir_GW170817_GW231109": "+GW170817+GW231109",
               "outdir_GW170817_GW231109": "+GW170817+GW231109",
               "outdir_GW231109_double_gaussian": "+GW231109 (double Gaussian)",
               "outdir_GW231109_quniv": "+GW231109 (QUR)",
               "outdir_ET_AS": "ET",
               }

COLORS_DICT = {"outdir": "darkgray",
               "outdir_radio": "dimgray",
               "outdir_GW170817": "orange",
               "outdir_GW231109": "green",
               "outdir_GW170817_GW231109": "red",
               "outdir_GW231109_double_gaussian": "purple",
               "outdir_GW231109_quniv": "red",
               "outdir_ET_AS": INJECTION_COLOR
               }

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

def make_parameter_histograms(data_list: list, outdir_names: list, colors: list, filename_prefix: str = ""):
    """Create comparison histograms for key EOS parameters across multiple datasets.

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        filename_prefix: Prefix for filename based on dataset labels
    """
    print(f"Creating parameter comparison histograms for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures/EOS_comparison", exist_ok=True)

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

            # Get directory basename and map to label
            dir_basename = os.path.basename(outdir_name.rstrip('/'))
            label = LABELS_DICT.get(dir_basename, dir_basename)

            plt.plot(x, y, color=color, lw=3.0, label=label)
            plt.fill_between(x, y, alpha=0.3, color=color)

        plt.xlabel(config['xlabel'])
        plt.ylabel('Density')
        plt.xlim(config['range'])
        plt.ylim(bottom=0.0)
        plt.legend()

        # Save comparison plot
        save_name = os.path.join("./figures/EOS_comparison", f"{filename_prefix}_{param_name}_histogram.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"  {param_name} comparison histogram saved to {save_name}")

def make_mass_radius_contour_plot(data_list: list, outdir_names: list, colors: list, filename_prefix: str = "", m_min: float = 0.6, m_max: float = 2.5):
    """Create comparison mass-radius contour plot with credible intervals for each dataset.

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        filename_prefix: Prefix for filename based on dataset labels
        m_min: Minimum mass for contour plot
        m_max: Maximum mass for contour plot
    """
    print(f"Creating mass-radius contour comparison plot for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures/EOS_comparison", exist_ok=True)

    plt.figure(figsize=figsize_vertical)

    masses_array = np.linspace(m_min, m_max, 100)

    # Plot contours for each dataset
    for data_dict, outdir_name, color in zip(data_list, outdir_names, colors):
        m, r = data_dict['masses'], data_dict['radii']

        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        label = LABELS_DICT.get(dir_basename, dir_basename)
        print(f"  Processing contours for {label}")

        radii_low = np.empty_like(masses_array)
        radii_high = np.empty_like(masses_array)

        for i, mass_point in enumerate(masses_array):
            # Find radii at this mass for all EOS samples
            radii_at_mass = []
            for mass_curve, radius_curve in zip(m, r):
                if len(mass_curve) > 0 and np.min(mass_curve) <= mass_point <= np.max(mass_curve):
                    radius_interp = np.interp(mass_point, mass_curve, radius_curve)
                    radii_at_mass.append(radius_interp)

            if len(radii_at_mass) > 10:  # Require at least 10 samples
                low, med, high = report_credible_interval(np.array(radii_at_mass), hdi_prob=0.90)
                radii_low[i] = med - low
                radii_high[i] = med + high
            else:
                radii_low[i] = np.nan
                radii_high[i] = np.nan

        # Remove NaN values
        valid_mask = ~np.isnan(radii_low) & ~np.isnan(radii_high)
        masses_valid = masses_array[valid_mask]
        radii_low_valid = radii_low[valid_mask]
        radii_high_valid = radii_high[valid_mask]

        if len(masses_valid) > 0:
            # Plot credible interval
            plt.fill_betweenx(masses_valid, radii_low_valid, radii_high_valid,
                            alpha=ALPHA, color=color, label=label)
            plt.plot(radii_low_valid, masses_valid, lw=2.0, color=color)
            plt.plot(radii_high_valid, masses_valid, lw=2.0, color=color)

    # Styling
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(8.0, 16.0)
    plt.ylim(m_min, m_max)
    plt.legend()

    # Save comparison figure
    save_name = os.path.join("./figures/EOS_comparison", f"{filename_prefix}_mass_radius_contour.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Mass-radius contour plot saved to {save_name}")

def make_pressure_density_contour_plot(data_list: list, outdir_names: list, colors: list, filename_prefix: str = "", n_min: float = 0.5, n_max: float = 6.0):
    """Create comparison pressure-density contour plot with credible intervals for each dataset.

    Args:
        data_list: List of dictionaries containing EOS data
        outdir_names: List of directory names for labeling
        colors: List of colors to use for each dataset
        filename_prefix: Prefix for filename based on dataset labels
        n_min: Minimum density for contour plot
        n_max: Maximum density for contour plot
    """
    print(f"Creating pressure-density contour comparison plot for {len(data_list)} datasets...")

    # Ensure figures directory exists
    os.makedirs("./figures/EOS_comparison", exist_ok=True)

    plt.figure(figsize=figsize_horizontal)

    dens_array = np.linspace(n_min, n_max, 100)

    # Plot contours for each dataset
    for data_dict, outdir_name, color in zip(data_list, outdir_names, colors):
        n, p = data_dict['densities'], data_dict['pressures']

        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        label = LABELS_DICT.get(dir_basename, dir_basename)
        print(f"  Processing contours for {label}")

        press_low = np.empty_like(dens_array)
        press_high = np.empty_like(dens_array)

        for i, dens_point in enumerate(dens_array):
            # Find pressures at this density for all EOS samples
            press_at_dens = []
            for dens_curve, press_curve in zip(n, p):
                if len(dens_curve) > 0 and np.min(dens_curve) <= dens_point <= np.max(dens_curve):
                    press_interp = np.interp(dens_point, dens_curve, press_curve)
                    press_at_dens.append(press_interp)

            if len(press_at_dens) > 10:  # Require at least 10 samples
                low, med, high = report_credible_interval(np.array(press_at_dens), hdi_prob=0.95)
                press_low[i] = med - low
                press_high[i] = med + high
            else:
                press_low[i] = np.nan
                press_high[i] = np.nan

        # Remove NaN values
        valid_mask = ~np.isnan(press_low) & ~np.isnan(press_high)
        dens_valid = dens_array[valid_mask]
        press_low_valid = press_low[valid_mask]
        press_high_valid = press_high[valid_mask]

        if len(dens_valid) > 0:
            # Plot credible interval
            plt.fill_between(dens_valid, press_low_valid, press_high_valid,
                           alpha=ALPHA, color=color, label=label)
            plt.plot(dens_valid, press_low_valid, lw=2.0, color=color)
            plt.plot(dens_valid, press_high_valid, lw=2.0, color=color)

    # Styling
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.xlim(n_min, n_max)
    plt.yscale('log')
    plt.legend()

    # Save comparison figure
    save_name = os.path.join("./figures/EOS_comparison", f"{filename_prefix}_pressure_density_contour.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Pressure-density contour plot saved to {save_name}")

def generate_filename_prefix(outdir_names: list):
    """Generate a unique filename prefix from dataset labels.

    Args:
        outdir_names: List of directory names

    Returns:
        str: Concatenated labels for filename (e.g., "Prior_radio_GW231109")
    """
    labels = []
    for outdir_name in outdir_names:
        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        label = LABELS_DICT.get(dir_basename, dir_basename)
        # Clean label for filename (remove special characters)
        clean_label = label.replace("+", "").replace(" ", "_").replace("(", "").replace(")", "")
        labels.append(clean_label)

    return "_".join(labels)

def get_colors_for_directories(directories: list):
    """Get colors for directories based on COLORS_DICT mapping.

    Args:
        directories: List of directory paths

    Returns:
        list: Colors corresponding to each directory
    """
    colors = []
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, outdir in enumerate(directories):
        dir_basename = os.path.basename(outdir.rstrip('/'))
        if dir_basename in COLORS_DICT:
            colors.append(COLORS_DICT[dir_basename])
        else:
            # Fallback to default colors if not in mapping
            colors.append(default_colors[i % len(default_colors)])
            print(f"Warning: No color mapping found for {dir_basename}, using default color {default_colors[i % len(default_colors)]}")

    return colors

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

import os

def process_given_dirs(directories, save_suffix=""):
    """Process given directories and generate comparison plots."""

    print("Money Plots Generator for Jester Inference Results")
    print("=" * 60)
    print(f"Processing {len(directories)} directories for comparison plots...")

    # Load all data
    data_list, valid_directories = load_all_data(directories)

    if len(data_list) == 0:
        print("Error: No valid data directories found!")
        return

    # Get colors for valid directories only
    valid_colors = get_colors_for_directories(valid_directories)

    # Generate filename prefix from dataset labels
    filename_prefix = generate_filename_prefix(valid_directories)
    print(f"Generated filename prefix: {filename_prefix}")

    print(f"\n{'='*60}")
    print(f"Creating comparison plots for {len(data_list)} valid datasets...")
    print(f"{'='*60}")

    # Create all comparison plots
    try:
        make_parameter_histograms(data_list, valid_directories, valid_colors, filename_prefix)
        make_mass_radius_contour_plot(data_list, valid_directories, valid_colors, filename_prefix)
        make_pressure_density_contour_plot(data_list, valid_directories, valid_colors, filename_prefix)
        print(f"\nAll comparison plots generated successfully!")
    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        return

def main():
    """Main function - configures directories and calls processing."""

    # =======================================================================
    # 1 Check GW231109
    # =======================================================================

    directories = [
        "../jester/outdir",
        "../jester/outdir_radio",
        "../jester/outdir_GW231109",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)
    
    
    # =======================================================================
    # 2 Check GW231109 vs GW190425
    # =======================================================================
    
    
    directories = [
        "../jester/outdir_radio",
        "../jester/outdir_GW231109",
        "../jester/outdir_GW190425",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)
    
    # =======================================================================
    # 3a Check GW170817 vs GW170817+GW190425
    # =======================================================================
    
    
    directories = [
        "../jester/outdir_radio",
        "../jester/outdir_GW170817",
        "../jester/outdir_GW170817_GW190425",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)
    
    # =======================================================================
    # 3a Check GW170817 vs GW170817+GW231109
    # =======================================================================
    
    
    directories = [
        "../jester/outdir_radio",
        "../jester/outdir_GW170817",
        "../jester/outdir_GW170817_GW231109",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)
    
    # =======================================================================
    # 4 Check GW231109 spins
    # =======================================================================
    
    # TODO: finish
    
    # =======================================================================
    # 5 Check GW231109 other prior choices
    # =======================================================================
    
    directories = [
        "../jester/outdir_GW231109",
        "../jester/outdir_GW231109_double_gaussian",
        "../jester/outdir_GW231109_quniv",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)
    
    # =======================================================================
    # 6 Check XP vs XAS
    # =======================================================================
    
    directories = [
        "../jester/outdir_GW231109",
        "../jester/outdir_GW231109_XAS",
    ]
    save_suffix = ""
    process_given_dirs(directories, save_suffix)


if __name__ == "__main__":
    main()
