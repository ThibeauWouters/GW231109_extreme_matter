"""
NOTE: this has to be executed on Snellius, where the Jester posterior samples are stored.

Money plots script for jester inference results on Snellius.

This script generates comparison figures from multiple jester inference output directories:
- Comparison histograms of MTOV, R14, and other EOS parameters overlaying all datasets
- Mass-radius comparison plots with different colors for each dataset
- Pressure-density (EOS) comparison plots overlaying all datasets

The script creates single plots comparing all specified directories, rather than
separate plots for each directory. All plots use the 'crest' colormap for consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tqdm
import arviz
from scipy.stats import gaussian_kde
import jesterTOV.utils as jose_utils

# Import shared EOS loading utilities
from eos_utils import load_eos_data

# Flag to use longer sampling run for GW170817+GW231109
USE_LONGER_SAMPLING = True

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
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "figure.titlesize": 12}
plt.rcParams.update(mpl_params)

# Constants
COLORS_DICT = {"prior": "gray",
               "GW170817": "orange",
               "GW231109": "teal",
               "GW231109_only": "red"}
ALPHA = 0.3
figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)
figsize_histograms = (5, 5)

LABELS_DICT = {"outdir": "Prior",
               "outdir_radio": "Heavy PSRs",
               "outdir_GW170817": "+GW170817",
               "outdir_GW231109": "+GW231109",
               "outdir_GW231109_XAS": r"+GW231109 (\texttt{XAS})",
               "outdir_GW190425": "+GW190425",
               "outdir_GW170817_GW231109": "+GW170817\n+GW231109",
               "outdir_GW170817_GW231109_longer_sampling": "+GW170817\n+GW231109",
               "outdir_GW170817_GW190425": "+GW170817\n+GW190425",
               "outdir_GW170817_GW190425_GW231109": "+GW170817\n+GW190425\n+GW231109",
               "outdir_GW231109_double_gaussian": "+GW231109 (double Gaussian)",
               "outdir_GW231109_quniv": "+GW231109 (QUR)",
               "outdir_GW231109_s025": r"+GW231109 $\chi \leq 0.25$",
               "outdir_GW231109_s040": r"+GW231109 $\chi \leq 0.4$",
               "outdir_ET_AS": "ET",
               }

COLORS_DICT = {"outdir": "darkgray",
               "outdir_radio": "dimgray",
               "outdir_GW170817": "blue",
               "outdir_GW190425": "#c7de81",
               "outdir_GW231109": "orange",
               "outdir_GW231109_XAS": "red",
               "outdir_GW170817_GW231109": "mediumslateblue",
               "outdir_GW170817_GW231109_longer_sampling": "mediumslateblue",
               "outdir_GW170817_GW190425": "#c7de81",
               "outdir_GW170817_GW190425_GW231109": "orange",
               "outdir_GW231109_double_gaussian": "mediumslateblue",
               "outdir_GW231109_quniv": "red",
               "outdir_GW231109_s025": "mediumslateblue",
               "outdir_GW231109_s040": "blue",
               "outdir_ET_AS": INJECTION_COLOR
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

def make_parameter_histograms(data_list: list,
                              outdir_names: list,
                              colors: list,
                              filename_prefix: str = "",
                              legend_outside: bool = False,
                              fontsize_legend: int = 11,
                              fontsize_labels: int = 12
                              ):
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
        'R14': {'range': (9.0, 15.0), 'xlabel': r"$R_{1.4}$ [km]"},
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
        plt.figure(figsize=figsize_histograms)

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

        plt.xlabel(config['xlabel'], fontsize=fontsize_labels)
        plt.ylabel('Probability density', fontsize=fontsize_labels)
        plt.xlim(config['range'])
        plt.ylim(bottom=0.0)
        if legend_outside:
            print(f"Putting the legend outside the plot.")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = fontsize_legend)
        else:
            plt.legend(fontsize = fontsize_legend)

        # Save comparison plot
        save_name = os.path.join("./figures/EOS_comparison", f"{filename_prefix}_{param_name}_histogram.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"  {param_name} comparison histogram saved to {save_name}")

def make_mass_radius_contour_plot(data_list: list,
                                  outdir_names: list,
                                  colors: list,
                                  filename_prefix: str = "",
                                  m_min: float = 0.6,
                                  m_max: float = 2.5,
                                  legend_outside: bool = False):
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
    if legend_outside:
        print(f"Putting the legend outside the plot.")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()

    # Save comparison figure
    save_name = os.path.join("./figures/EOS_comparison", f"{filename_prefix}_mass_radius_contour.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Mass-radius contour plot saved to {save_name}")

def make_pressure_density_contour_plot(data_list: list,
                                       outdir_names: list,
                                       colors: list,
                                       filename_prefix: str = "",
                                       n_min: float = 0.5,
                                       n_max: float = 6.0,
                                       legend_outside: bool = False):
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
    
    if legend_outside:
            print(f"Putting the legend outside the plot.")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
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
        clean_label = clean_label.replace("$", "").replace("\\", "").replace("texttt{", "").replace("}", "")
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
    default_colors = ['blue', 'red', 'green', 'orange', 'mediumslateblue', 'brown', 'pink', 'gray', 'olive', 'cyan']

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
        # Check if directory exists
        if not os.path.exists(outdir):
            print(f"Warning: Directory {outdir} does not exist. Skipping...")
            continue

        # Load data
        try:
            data = load_eos_data(outdir)
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

def process_given_dirs(directories, save_suffix="", legend_outside=False, filename_prefix=None, do_contours: bool = True):
    """Process given directories and generate comparison plots.

    Args:
        directories: List of directories to process
        save_suffix: Suffix for saved files (deprecated, kept for compatibility)
        legend_outside: Whether to place legend outside the plot
        filename_prefix: Custom prefix for saved files. If None, auto-generate from directory labels.
    """

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

    # Generate filename prefix from dataset labels if not provided
    if filename_prefix is None:
        filename_prefix = generate_filename_prefix(valid_directories)
        print(f"Generated filename prefix: {filename_prefix}")
    else:
        print(f"Using custom filename prefix: {filename_prefix}")

    print(f"\n{'='*60}")
    print(f"Creating comparison plots for {len(data_list)} valid datasets...")
    print(f"{'='*60}")

    # Create all comparison plots
    try:
        make_parameter_histograms(data_list, valid_directories, valid_colors, filename_prefix, legend_outside=legend_outside)
        if do_contours:
            print("Also plotting the contour plots...")
            make_mass_radius_contour_plot(data_list, valid_directories, valid_colors, filename_prefix, legend_outside=legend_outside)
            make_pressure_density_contour_plot(data_list, valid_directories, valid_colors, filename_prefix, legend_outside=legend_outside)
        print(f"\nAll comparison plots generated successfully!")
    except Exception as e:
        print(f"Error generating comparison plots: {e}")
        return

# FIXME: deprecate this
# def plot_injection(outdir: str):
#     print(f"Plotting the GW231109 (ET) from {outdir}...")

#     # First, load the true EOS
#     hauke_filename = "../figures/EOS_data/hauke_macroscopic.dat"
#     r, m, l, _ = np.loadtxt(hauke_filename, unpack=True)
#     R14_TARGET = np.interp(1.4, m, r)
#     print(f"  Hauke EOS R14 = {R14_TARGET:.2f} km")

#     # First, load the true EOS
#     hauke_filename = "../figures/EOS_data/hauke_microscopic.dat"
#     n, _, p, _ = np.loadtxt(hauke_filename, unpack=True)

#     # # Convert units
#     n = n / 0.16
#     print(np.min(n), np.max(n))
#     # n = n / jose_utils.fm_inv3_to_geometric / 0.16
#     # p = p / jose_utils.MeV_fm_inv3_to_geometric
#     # e = e / jose_utils.MeV_fm_inv3_to_geometric

#     P3NSAT_TARGET = np.interp(3, n, p)
#     print(f"  Hauke p3nsat = {P3NSAT_TARGET:.2f}")

#     # Load the data
#     data = load_eos_data(os.path.join("../jester", outdir))
#     masses, radii = data['masses'], data['radii']
#     n, p = data['densities'], data['pressures']

#     # Also load the prior
#     data_prior = load_eos_data(os.path.join("../jester", "outdir"))
#     masses_prior, radii_prior = data_prior['masses'], data_prior['radii']
#     n_prior, p_prior = data_prior['densities'], data_prior['pressures']

#     # Also load the Heavy PSRs
#     data_radio = load_eos_data(os.path.join("../jester", "outdir_radio"))
#     masses_radio, radii_radio = data_radio['masses'], data_radio['radii']
#     n_radio, p_radio = data_radio['densities'], data_radio['pressures']

#     # Histogram of R14
#     R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses, radii)])
#     plt.figure(figsize=figsize_horizontal)
#     kde = gaussian_kde(R14_list)
#     kde_prior = gaussian_kde(np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_prior, radii_prior)]))
#     kde_radio = gaussian_kde(np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_radio, radii_radio)]))
#     x = np.linspace(10.0, 16.0, 1000)
#     y = kde(x)
#     y_prior = kde_prior(x)
#     y_radio = kde_radio(x)
#     plt.plot(x, y_prior, color='darkgray', lw=3.0, label="Prior")
#     plt.fill_between(x, y_prior, alpha=0.3, color='darkgray')
#     plt.plot(x, y_radio, color='dimgray', lw=3.0, label="Heavy PSRs")
#     plt.fill_between(x, y_radio, alpha=0.3, color='dimgray')
#     plt.plot(x, y, color=INJECTION_COLOR, lw=3.0, label="GW231109 (ET)")
#     plt.fill_between(x, y, alpha=0.3, color=INJECTION_COLOR)

#     # Hauke:
#     plt.axvline(R14_TARGET, color='black', ls='--', lw=2.0, label="Truth")

#     plt.xlabel(r"$R_{1.4}$ [km]")
#     plt.ylabel('Density')
#     plt.xlim(10.0, 16.0)
#     plt.ylim(bottom=0.0)
#     plt.legend()
#     if "CE" in outdir:
#         save_name = os.path.join("./figures/EOS_comparison", f"ET_CE_injection_R14_histogram.pdf")
#     else:
#         save_name = os.path.join("./figures/EOS_comparison", f"ET_injection_R14_histogram.pdf")
#     plt.savefig(save_name, bbox_inches="tight")
#     plt.close()
#     print(f"  R14 histogram saved to {save_name}")

#     # Now also for p3nsat
#     p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])
#     plt.figure(figsize=figsize_horizontal)
#     kde = gaussian_kde(p3nsat_list)
#     kde_prior = gaussian_kde(np.array([np.interp(3.0, dens, press) for dens, press in zip(n_prior, p_prior)]))
#     kde_radio = gaussian_kde(np.array([np.interp(3.0, dens, press) for dens, press in zip(n_radio, p_radio)]))

#     x = np.linspace(0.1, 200.0, 1000)
#     y = kde(x)
#     y_prior = kde_prior(x)
#     y_radio = kde_radio(x)
#     plt.plot(x, y_prior, color='darkgray', lw=3.0, label="Prior")
#     plt.fill_between(x, y_prior, alpha=0.3, color='darkgray')
#     plt.plot(x, y_radio, color='dimgray', lw=3.0, label="Heavy PSRs")
#     plt.fill_between(x, y_radio, alpha=0.3, color='dimgray')
#     plt.plot(x, y, color=INJECTION_COLOR, lw=3.0, label="GW231109 (ET)")
#     plt.fill_between(x, y, alpha=0.3, color=INJECTION_COLOR)

#     # Hauke:
#     plt.axvline(P3NSAT_TARGET, color='black', ls='--', lw=2.0, label="Truth")

#     plt.xlabel(r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]")
#     plt.ylabel('Density')
#     plt.xlim(0.1, 200.0)
#     plt.ylim(bottom=0.0)
#     plt.legend()
#     if "CE" in outdir:
#         save_name = os.path.join("./figures/EOS_comparison", f"ET_CE_injection_p3nsat_histogram.pdf")
#     else:
#         save_name = os.path.join("./figures/EOS_comparison", f"ET_injection_p3nsat_histogram.pdf")
#     plt.savefig(save_name, bbox_inches="tight")
#     plt.close()
#     print(f"  p3nsat histogram saved to {save_name}")

def plot_full_injection(plot_text: bool = True,
                        plot_prior: bool = True,
                        run_dir_et: str = "outdir_GW231109_ET_jester",
                        run_dir_et_ce: str = "outdir_GW231109_ET_CE_jester",
                        target_eos: str = "jester",
                        what_prior: str = "radio"):
    """Plot both ET and ET+CE injection results together.

    Args:
        plot_text: If True, plot credible interval text in upper-right corner.
    """
    print("Plotting combined ET and ET+CE injection results...")

    # Load the true EOS (macroscopic)   
    if target_eos == "hauke":
        # Load the true NS curve (macroscopic)
        print(f"Loading target EOS information from: hauke")
        filename = "../figures/EOS_data/hauke_macroscopic.dat"
        r, m, l, _ = np.loadtxt(filename, unpack=True)
        R14_TARGET = np.interp(1.4, m, r)
        
        # Load the true EOS (microscopic)
        micro_filename = "../figures/EOS_data/hauke_microscopic.dat"
        n_hauke, _, p_hauke, _ = np.loadtxt(micro_filename, unpack=True)
        n_hauke = n_hauke / 0.16
        P3NSAT_TARGET = np.interp(3, n_hauke, p_hauke)
        print(f"  Hauke p3nsat = {P3NSAT_TARGET:.2f}")
        
    else:
        # Load the true NS curve (macroscopic)
        print(f"Loading target EOS information from: jester")
        filename = "../figures/EOS_data/jester_GW170817_maxL_EOS.npz"
        eos_data = np.load(filename)
        r, m, l = eos_data['radii'], eos_data['masses'], eos_data['Lambdas']
        R14_TARGET = np.interp(1.4, m, r)
        
        # Convert units
        n_target, p_target = eos_data['n'], eos_data['p']
        n_target = n_target / jose_utils.fm_inv3_to_geometric / 0.16
        p_target = p_target / jose_utils.MeV_fm_inv3_to_geometric
        # e_target = e_target / jose_utils.MeV_fm_inv3_to_geometric
        
        # Get target
        P3NSAT_TARGET = np.interp(3, n_target, p_target)
    
    # Print to the screen for verifications
    print(f"Showing target EOS values:")
    print(f"  R14_TARGET = {R14_TARGET:.2f} km")
    print(f"  P3NSAT_TARGET = {P3NSAT_TARGET:.2f} Mev fm^-3")

    # Load ET data -- using Anna's new runs!
    print(f"Loading jester results from ET rundir: {run_dir_et}")
    data_et = load_eos_data(os.path.join("../jester", run_dir_et))
    masses_et, radii_et = data_et['masses'], data_et['radii']
    n_et, p_et = data_et['densities'], data_et['pressures']

    # Load ET+CE data -- using Anna's new runs!
    print(f"Loading jester results from ET rundir: {run_dir_et_ce}")
    data_et_ce = load_eos_data(os.path.join("../jester", run_dir_et_ce))
    masses_et_ce, radii_et_ce = data_et_ce['masses'], data_et_ce['radii']
    n_et_ce, p_et_ce = data_et_ce['densities'], data_et_ce['pressures']

    # Load the prior
    data_prior = load_eos_data(os.path.join("../jester", "outdir"))
    masses_prior, radii_prior = data_prior['masses'], data_prior['radii']
    n_prior, p_prior = data_prior['densities'], data_prior['pressures']

    # Load the Heavy PSRs
    data_radio = load_eos_data(os.path.join("../jester", "outdir_radio"))
    masses_radio, radii_radio = data_radio['masses'], data_radio['radii']
    n_radio, p_radio = data_radio['densities'], data_radio['pressures']

    # Define colors for ET and ET+CE
    ET_COLOR = "#de8f05"
    ET_CE_COLOR = "mediumslateblue"

    # =========================================================================
    # R14 histogram
    # =========================================================================
    R14_et = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_et, radii_et)])
    R14_et_ce = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_et_ce, radii_et_ce)])
    R14_radio = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_radio, radii_radio)])

    plt.figure(figsize=figsize_histograms)
    kde_et = gaussian_kde(R14_et)
    kde_et_ce = gaussian_kde(R14_et_ce)
    kde_radio = gaussian_kde(R14_radio)

    x = np.linspace(9.0, 15.0, 1000)
    y_et = kde_et(x)
    y_et_ce = kde_et_ce(x)
    y_radio = kde_radio(x)

    # Also do prior
    if plot_prior:
        if what_prior == "radio":
            kde_prior = gaussian_kde(R14_radio)
            R14_prior = R14_radio
            prior_label = "Heavy PSRs"
        else:
            R14_prior = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(masses_prior, radii_prior)])
            kde_prior = gaussian_kde(R14_prior)
            prior_label = "Prior"
        y_prior = kde_prior(x)
    
        plt.plot(x, y_prior, color='darkgray', lw=3.0, label=prior_label)
        plt.fill_between(x, y_prior, alpha=0.3, color='darkgray')
        
    plt.plot(x, y_et, color=ET_COLOR, lw=3.0, label="ET")
    plt.fill_between(x, y_et, alpha=0.3, color=ET_COLOR)
    plt.plot(x, y_et_ce, color=ET_CE_COLOR, lw=3.0, label="ET+CE")
    plt.fill_between(x, y_et_ce, alpha=0.3, color=ET_CE_COLOR)

    # Truth line
    plt.axvline(x=R14_TARGET, color='black', ls='--', lw=2.0, label="Injection")

    # Compute 90% credible intervals
    low_et, high_et = arviz.hdi(R14_et, hdi_prob=0.90)
    med_et = np.median(R14_et)
    low_et_ce, high_et_ce = arviz.hdi(R14_et_ce, hdi_prob=0.90)
    med_et_ce = np.median(R14_et_ce)

    # Add credible intervals as text in top-right corner
    if plot_text:
        textstr_et = f"${med_et:.2f}^{{+{high_et - med_et:.2f}}}_{{-{med_et - low_et:.2f}}}$"
        textstr_et_ce = f"${med_et_ce:.2f}^{{+{high_et_ce - med_et_ce:.2f}}}_{{-{med_et_ce - low_et_ce:.2f}}}$"
        x = 0.95
        y = 0.95
        dy = 0.15
        fs = 12
        plt.text(x, y, textstr_et, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 color=ET_COLOR, fontsize=fs)
        plt.text(x, y-dy, textstr_et_ce, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 color=ET_CE_COLOR, fontsize=fs)

    plt.xlabel(r"$R_{1.4}$ [km]", fontsize=12)
    plt.ylabel('Probability density', fontsize=12)
    plt.xlim(11.0, 14.0)
    plt.ylim(bottom=0.0)
    plt.legend(fontsize=11)

    save_name = os.path.join("./figures/EOS_comparison", "ET_full_injection_R14_histogram.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  R14 histogram saved to {save_name}")
    print(f"    ET R14: {med_et:.2f} +{high_et - med_et:.2f} -{med_et - low_et:.2f} km (90% HDI)")
    print(f"    ET+CE R14: {med_et_ce:.2f} +{high_et_ce - med_et_ce:.2f} -{med_et_ce - low_et_ce:.2f} km (90% HDI)")

    # =========================================================================
    # p3nsat histogram
    # =========================================================================
    p3nsat_et = np.array([np.interp(3.0, dens, press) for dens, press in zip(n_et, p_et)])
    p3nsat_et_ce = np.array([np.interp(3.0, dens, press) for dens, press in zip(n_et_ce, p_et_ce)])
    p3nsat_prior = np.array([np.interp(3.0, dens, press) for dens, press in zip(n_prior, p_prior)])
    p3nsat_radio = np.array([np.interp(3.0, dens, press) for dens, press in zip(n_radio, p_radio)])

    plt.figure(figsize=figsize_histograms)
    kde_et = gaussian_kde(p3nsat_et)
    kde_et_ce = gaussian_kde(p3nsat_et_ce)
    kde_prior = gaussian_kde(p3nsat_prior)
    kde_radio = gaussian_kde(p3nsat_radio)

    x = np.linspace(0.1, 200.0, 1000)
    y_et = kde_et(x)
    y_et_ce = kde_et_ce(x)
    y_prior = kde_prior(x)
    y_radio = kde_radio(x)

    plt.plot(x, y_prior, color='darkgray', lw=3.0, label="Prior")
    plt.fill_between(x, y_prior, alpha=0.3, color='darkgray')
    plt.plot(x, y_radio, color='dimgray', lw=3.0, label="Heavy PSRs")
    plt.fill_between(x, y_radio, alpha=0.3, color='dimgray')
    plt.plot(x, y_et, color=ET_COLOR, lw=3.0, label="ET")
    plt.fill_between(x, y_et, alpha=0.3, color=ET_COLOR)
    plt.plot(x, y_et_ce, color=ET_CE_COLOR, lw=3.0, label="ET+CE")
    plt.fill_between(x, y_et_ce, alpha=0.3, color=ET_CE_COLOR)

    # Truth line
    print("P3NSAT_TARGET")
    print(P3NSAT_TARGET)
    plt.axvline(x=P3NSAT_TARGET, color='black', ls='--', lw=2.0, label="Injection")

    # Compute 90% credible intervals
    low_et, high_et = arviz.hdi(p3nsat_et, hdi_prob=0.90)
    med_et = np.median(p3nsat_et)
    low_et_ce, high_et_ce = arviz.hdi(p3nsat_et_ce, hdi_prob=0.90)
    med_et_ce = np.median(p3nsat_et_ce)

    # Add credible intervals as text in top-right corner
    if plot_text:
        textstr_et = f"${med_et:.1f}^{{+{high_et - med_et:.1f}}}_{{-{med_et - low_et:.1f}}}$"
        textstr_et_ce = f"${med_et_ce:.1f}^{{+{high_et_ce - med_et_ce:.1f}}}_{{-{med_et_ce - low_et_ce:.1f}}}$"
        plt.text(0.97, 0.97, textstr_et, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 color=ET_COLOR, fontsize=12)
        plt.text(0.97, 0.92, textstr_et_ce, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 color=ET_CE_COLOR, fontsize=12)

    plt.xlabel(r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]", fontsize=12)
    plt.ylabel('Probability density', fontsize=12)
    plt.xlim(0.1, 200.0)
    plt.ylim(bottom=0.0)
    plt.legend(fontsize=11)

    save_name = os.path.join("./figures/EOS_comparison", "ET_full_injection_p3nsat_histogram.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  p3nsat histogram saved to {save_name}")
    print(f"    ET p3nsat: {med_et:.1f} +{high_et - med_et:.1f} -{med_et - low_et:.1f} MeV fm^-3 (90% HDI)")
    print(f"    ET+CE p3nsat: {med_et_ce:.1f} +{high_et_ce - med_et_ce:.1f} -{med_et_ce - low_et_ce:.1f} MeV fm^-3 (90% HDI)")
    

def main():
    """Main function - configures directories and calls processing."""

    # Determine which GW170817+GW231109 directory to use
    gw170817_gw231109_dir = "../jester/outdir_GW170817_GW231109_longer_sampling" if USE_LONGER_SAMPLING else "../jester/outdir_GW170817_GW231109"

    # # =======================================================================
    # # 1 Check GW231109
    # # =======================================================================

    # directories = [
    #     "../jester/outdir",
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW231109",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix)


    # # =======================================================================
    # # 2 Check GW231109 vs GW190425
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW190425",
    #     "../jester/outdir_GW231109",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, filename_prefix="GW190425_vs_GW231109_radio")

    # # =======================================================================
    # # 2 Check GW231109 vs GW190425 -- but now with just the prior for comparison
    # # =======================================================================

    # directories = [
    #     "../jester/outdir",
    #     "../jester/outdir_GW190425",
    #     "../jester/outdir_GW231109",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, filename_prefix="GW190425_vs_GW231109_prior")

    # # =======================================================================
    # # 3a Check GW170817 vs GW170817+GW190425
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817",
    #     "../jester/outdir_GW170817_GW190425",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix)

    # # =======================================================================
    # # 3a Check GW170817 vs GW170817+GW231109
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817",
    #     gw170817_gw231109_dir,
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix)

    # # =======================================================================
    # # 3b Check GW170817+GW190425 vs GW170817+GW231109
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817_GW190425",
    #     gw170817_gw231109_dir,
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, legend_outside=False, filename_prefix="GW190425_vs_GW231109_with_GW170817")

    # # =======================================================================
    # # 3c Check GW170817+GW190425 vs GW170817+GW231109 without GW170817
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817_GW190425",
    #     gw170817_gw231109_dir,
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, legend_outside=True)

    # # =======================================================================
    # # 4 Check GW231109 spins
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW231109",
    #     "../jester/outdir_GW231109_s025",
    #     "../jester/outdir_GW231109_s040",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, legend_outside=True)

    # # =======================================================================
    # # 5 Check GW231109 other prior choices
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_GW231109",
    #     "../jester/outdir_GW231109_double_gaussian",
    #     "../jester/outdir_GW231109_quniv",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix)

    # # =======================================================================
    # # 6 Check XP vs XAS
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_GW231109",
    #     "../jester/outdir_GW231109_XAS",
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix)

    # # =======================================================================
    # # 7 Increasing constraints with more and more GW BNS
    # # =======================================================================

    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817",
    #     "../jester/outdir_GW231109",
    #     gw170817_gw231109_dir,
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, filename_prefix="final", do_contours=False)
    
    # # # Additionally also this plot where we force the original run
    
    # directories = [
    #     "../jester/outdir_radio",
    #     "../jester/outdir_GW170817",
    #     "../jester/outdir_GW231109",
    #     gw170817_gw231109_dir,
    # ]
    # save_suffix = ""
    # process_given_dirs(directories, save_suffix, filename_prefix="all_bns", do_contours=False)
    

    # =======================================================================
    # INJECTIONS
    # =======================================================================

    # Combined ET and ET+CE plot
    plot_full_injection(plot_text=False, what_prior="radio")


if __name__ == "__main__":
    main()
