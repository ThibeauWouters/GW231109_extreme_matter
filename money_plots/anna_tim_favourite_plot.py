#!/usr/bin/env python3
"""
Anna and Tim's favourite plot: GW170817 posterior with GW231109 ET injection blob.
Shows chirp mass vs lambda_tilde for both events in a single comparison plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
import seaborn as sns

np.random.seed(40)  # for reproducibility

from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.conversion import luminosity_distance_to_redshift

# ============================================================================
# FONT SIZE CONFIGURATION - Adjust these values to change all font sizes
# ============================================================================
BASE_FONTSIZE = 22          # Base font size for titles and general text
TICK_FONTSIZE = 24          # Font size for axis tick labels
AXIS_LABEL_FONTSIZE = 28    # Font size for axis labels (x and y)
LEGEND_FONTSIZE = 18        # Font size for legend text
CORNER_FONTSIZE = 20        # Font size for corner plot labels
COLORBAR_LABEL_FONTSIZE = 20  # Font size for colorbar label
COLORBAR_TICKS_FONTSIZE = 18  # Font size for colorbar tick labels
DEBUG_AXIS_FONTSIZE = 20    # Font size for debug plot axes
DEBUG_TITLE_FONTSIZE = 22   # Font size for debug plot titles
# ============================================================================

# If running on Mac, use TeX for better typography
if "Woute029" in os.getcwd():
    print(f"Updating plotting parameters for TeX")
    labelpad = 18
    rc_params = {
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.title_fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": BASE_FONTSIZE
    }
    plt.rcParams.update(rc_params)
else:
    # Default params for servers without TeX
    params = {
        "axes.grid": False,
        "font.family": "serif",
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.title_fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": BASE_FONTSIZE
    }
    plt.rcParams.update(params)

# Corner plot settings (matching final_GW_PE_figures.py)
default_corner_kwargs = dict(
    bins=40,
    smooth=0.9,
    show_titles=False,
    title_fmt=".3f",
    label_kwargs=dict(fontsize=CORNER_FONTSIZE),
    title_kwargs=dict(fontsize=CORNER_FONTSIZE),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    levels=[0.68, 0.95],  # 68% and 95% credible regions
    labelpad=0.10,
    max_n_ticks=3,
    min_n_ticks=2,
    save=False
)

# Colors (matching m1m2_overview.py for consistency)
from m1m2_overview import EVENTS
GW170817_COLOR = EVENTS['GW170817']['color']
GW231109_ET_COLOR = "#de8f05"  # Orange/gold
GW231109_ET_CE_COLOR = "#d45d01"  # Darker orange
TRUTH_COLOR = "black"

# Injection truth marker settings (for consistency between plot and legend)
INJECTION_MARKER_EDGECOLOR = "white"  # Edge color for the injection star marker
INJECTION_MARKER_EDGEWIDTH = 1.5     # Edge width for the injection star marker
INJECTION_MARKER_SIZE = 400           # Marker size (s parameter for scatter)

# Corner plot display mode
# Options:
#   "contours": Show 68% and 95% credible region contours (1σ and 2σ equivalent)
#   "scatter":  Plot raw datapoints as scatter plot (inspired by m1m2_overview.py)
CORNER_PLOT_MODE = "contour"

# Debug plots
PLOT_DEBUG_LUMINOSITY_DISTANCE = False  # If True, create histogram of luminosity_distance for all datasets
PLOT_DEBUG_REDSHIFT = False  # If True, create histogram of redshift for all datasets, NOTE: this can take a bit for the redshift conversion

# Posterior plotting flags
PLOT_GW231109_ET = True  # If True, plot GW231109 ET posterior
PLOT_GW231109_ET_CE = False  # If True, plot GW231109 ET+CE posterior

# EOS curves flags
PLOT_GW170817_EOS = True  # If True, plot EOS curves from GW170817 constraint
PLOT_JESTER_ET_EOS = False  # If True, plot EOS curves from jester ET inference
PLOT_JESTER_ET_CE_EOS = False  # If True, plot EOS curves from jester ET+CE inference


def load_eos_curves(eos_name: str):
    """
    Load the EOS curves from a file.

    Args:
        eos_name (str): Name of the EOS dataset (e.g., "GW170817", "all", "radio")

    Returns:
        tuple: (M, R, L, log_prob) arrays for masses, radii, Lambdas, and log probabilities
    """
    filename = f"../figures/EOS_data/{eos_name}.npz"
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"EOS file not found: {filename}\n"
            "Please ensure the file exists in ../figures/EOS_data/"
        )

    data = np.load(filename)
    M, R, L = data['M'], data['R'], data['L']
    log_prob = data['log_prob']
    return M, R, L, log_prob


def get_mchirp_lambda_tilde_EOS(EOS_masses: np.array,
                                EOS_Lambdas: np.array,
                                mchirp_min: float,
                                mchirp_max: float) -> tuple:
    """
    Generate an array of chirp masses (source frame) and corresponding lambda_tilde
    values based on the provided EOS masses and Lambdas.

    Args:
        EOS_masses (np.array): Array of EOS masses.
        EOS_Lambdas (np.array): Array of EOS Lambdas.
        mchirp_min (float): Minimum chirp mass (source frame).
        mchirp_max (float): Maximum chirp mass (source frame).

    Returns:
        tuple[np.array, np.array]: Arrays of chirp masses and corresponding lambda_tilde values.
    """
    mchirp_array = np.linspace(mchirp_min, mchirp_max, 100)
    q_array = np.ones_like(mchirp_array)  # Assuming mass ratio ~1 for BNS

    # These masses are in the source frame
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(mchirp_array, q_array)
    lambda_1 = np.interp(mass_1, EOS_masses, EOS_Lambdas)
    lambda_2 = np.interp(mass_2, EOS_masses, EOS_Lambdas)

    # Compute lambda tilde
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    return mchirp_array, lambda_tilde


def plot_debug_luminosity_distance(datasets: dict, save_name: str = "./figures/GW_PE/DEBUG_dL.pdf"):
    """
    Create debug histogram plot of luminosity distance for all datasets.

    Args:
        datasets (dict): Dictionary with dataset names as keys and data dictionaries as values.
                        Each data dict should have 'luminosity_distance', 'color', 'label'.
        save_name (str): Output filename for the debug plot
    """
    print("\n" + "="*60)
    print("Creating DEBUG luminosity distance histogram")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset_name, data in datasets.items():
        dL = data['luminosity_distance']
        color = data['color']
        label = data['label']

        print(f"{dataset_name}:")
        print(f"  Mean dL: {np.mean(dL):.1f} Mpc")
        print(f"  Median dL: {np.median(dL):.1f} Mpc")
        print(f"  Std dL: {np.std(dL):.1f} Mpc")
        print(f"  Range: {np.min(dL):.1f} - {np.max(dL):.1f} Mpc")

        ax.hist(dL, bins=50, alpha=0.5, color=color, label=label, density=True)

    ax.set_xlabel(r'Luminosity Distance [Mpc]', fontsize=DEBUG_AXIS_FONTSIZE)
    ax.set_ylabel('Density', fontsize=DEBUG_AXIS_FONTSIZE)
    ax.set_title('Luminosity Distance Distributions (DEBUG)', fontsize=DEBUG_TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    print(f"\nSaving debug plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.close()

    print("="*60 + "\n")


def plot_debug_redshift(datasets: dict, save_name: str = "./figures/GW_PE/DEBUG_z.pdf"):
    """
    Create debug histogram plot of redshift for all datasets.

    Args:
        datasets (dict): Dictionary with dataset names as keys and data dictionaries as values.
                        Each data dict should have 'luminosity_distance', 'color', 'label'.
        save_name (str): Output filename for the debug plot
    """
    print("\n" + "="*60)
    print("Creating DEBUG redshift histogram")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset_name, data in datasets.items():
        dL = data['luminosity_distance']
        color = data['color']
        label = data['label']

        # Convert luminosity distance to redshift
        z = luminosity_distance_to_redshift(dL)

        print(f"{dataset_name}:")
        print(f"  Mean z: {np.mean(z):.6f}")
        print(f"  Median z: {np.median(z):.6f}")
        print(f"  Std z: {np.std(z):.6f}")
        print(f"  Range: {np.min(z):.6f} - {np.max(z):.6f}")

        ax.hist(z, bins=50, alpha=0.5, color=color, label=label, density=True)

    ax.set_xlabel(r'Redshift $z$', fontsize=DEBUG_AXIS_FONTSIZE)
    ax.set_ylabel('Density', fontsize=DEBUG_AXIS_FONTSIZE)
    ax.set_title('Redshift Distributions (DEBUG)', fontsize=DEBUG_TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    print(f"\nSaving debug plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.close()

    print("="*60 + "\n")


def load_GW170817_posterior():
    """
    Load the GW170817 posterior samples for chirp mass and lambda tilde.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde, luminosity_distance) arrays
    """
    filename = "../figures/EOS_data/PE_posterior_samples_GW170817.npz"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"GW170817 posterior file not found: {filename}\n"
            "Please ensure the file exists in ../figures/EOS_data/"
        )

    data = np.load(filename)
    chirp_mass_source = data['chirp_mass_source']
    lambda_tilde = data['lambda_tilde']
    luminosity_distance = data['luminosity_distance']

    print(f"Loaded {len(chirp_mass_source)} GW170817 samples")
    return chirp_mass_source, lambda_tilde, luminosity_distance


def load_GW231109_ET_posterior():
    """
    Load the GW231109 ET injection posterior samples.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde, luminosity_distance) arrays
    """
    filename = "../posteriors/data/jester_eos_et_run_alignedspin.npz"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"GW231109 ET posterior file not found: {filename}\n"
            "Please ensure the file exists in ../posteriors/data/"
        )

    data = np.load(filename)

    # Check what parameters are available
    print(f"Available parameters in ET file: {list(data.keys())}")

    # Load source frame masses (already computed in the file)
    mass_1_source = data['mass_1_source']
    mass_2_source = data['mass_2_source']

    # Lambda tilde is already computed in the file
    lambda_tilde = data['lambda_tilde']

    # Load luminosity distance
    luminosity_distance = data['luminosity_distance']

    # Compute chirp mass from source frame masses
    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)

    print(f"Loaded {len(chirp_mass_source)} GW231109 ET samples")
    return chirp_mass_source, lambda_tilde, luminosity_distance


def load_GW231109_ET_CE_posterior():
    """
    Load the GW231109 ET+CE injection posterior samples.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde, luminosity_distance) arrays
    """
    filename = "../posteriors/data/jester_eos_et_ce_run_alignedspin.npz"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"GW231109 ET+CE posterior file not found: {filename}\n"
            "Please ensure the file exists in ../posteriors/data/"
        )

    data = np.load(filename)

    # Check what parameters are available
    print(f"Available parameters in ET+CE file: {list(data.keys())}")

    # Load source frame masses (already computed in the file)
    mass_1_source = data['mass_1_source']
    mass_2_source = data['mass_2_source']

    # Lambda tilde is already computed in the file
    lambda_tilde = data['lambda_tilde']

    # Load luminosity distance
    luminosity_distance = data['luminosity_distance']

    # Compute chirp mass from source frame masses
    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)

    print(f"Loaded {len(chirp_mass_source)} GW231109 ET+CE samples")
    return chirp_mass_source, lambda_tilde, luminosity_distance


def load_jester_eos_samples(eos_type: str):
    """
    Load the Jester EOS samples from ET or ET+CE inference.

    Args:
        eos_type (str): Either "ET" or "ET_CE" to specify which inference result to load

    Returns:
        tuple: (M, L) arrays for masses and Lambdas from the EOS samples
               Returns (None, None) if file not found
    """
    filename_map = {
        "ET": "./data/jester_eos_samples_ET.npz",
        "ET_CE": "./data/jester_eos_samples_ET_CE.npz"
    }

    if eos_type not in filename_map:
        print(f"Warning: Unknown eos_type '{eos_type}'. Must be 'ET' or 'ET_CE'")
        return None, None

    filename = filename_map[eos_type]

    if not os.path.exists(filename):
        print(f"Warning: Jester EOS file not found: {filename}")
        return None, None

    data = np.load(filename)

    # Check what keys are available
    print(f"Available keys in {eos_type} jester EOS file: {list(data.keys())}")

    # Load the masses and Lambdas
    M = data['masses_EOS']
    L = data['Lambdas_EOS']

    print(f"Loaded {len(M)} Jester {eos_type} EOS samples")
    return M, L


def load_injection_eos_curve(xlim: tuple):
    """
    Load the injection EOS curve (Jester max likelihood EOS).

    Args:
        xlim (tuple): X-axis limits for chirp mass range

    Returns:
        tuple: (chirp_mass_array, lambda_tilde_array) for the injection EOS
    """
    filename = "../figures/EOS_data/jester_GW170817_maxL_EOS.npz"

    if not os.path.exists(filename):
        print(f"Warning: Injection EOS file not found: {filename}")
        return None, None

    eos_data = np.load(filename)
    EOS_masses = eos_data['masses']
    EOS_Lambdas = eos_data['Lambdas']

    # Generate chirp mass and lambda tilde for this EOS
    mchirp_min, mchirp_max = xlim
    mchirp_array, lambda_tilde = get_mchirp_lambda_tilde_EOS(
        EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max
    )

    print(f"Loaded injection EOS curve (Jester max likelihood)")
    return mchirp_array, lambda_tilde


def get_injection_parameters():
    """
    Get the injection parameters for the GW231109 ET simulation.

    Returns:
        dict: Injection parameters including chirp_mass_source and lambda_tilde
    """
    # Injection parameters from final_GW_PE_figures.py
    injection_parameters = {
        "mass_1": 1.5879187040159342,
        "mass_2": 1.4188967691574992,
        "luminosity_distance": 168.3222418883087,
        'lambda_1': 271.02342967819004,
        'lambda_2': 553.1640516248044
    }

    # Compute lambda_tilde from component Lambdas
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
        injection_parameters['lambda_1'],
        injection_parameters['lambda_2'],
        injection_parameters['mass_1'],
        injection_parameters['mass_2']
    )

    # Compute source frame masses
    z = luminosity_distance_to_redshift(injection_parameters['luminosity_distance'])
    mass_1_source = injection_parameters['mass_1'] / (1 + z)
    mass_2_source = injection_parameters['mass_2'] / (1 + z)
    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)

    # Update with interpolated Lambdas from Jester EOS
    filename = "../figures/EOS_data/jester_GW170817_maxL_EOS.npz"
    if os.path.exists(filename):
        eos_data = np.load(filename)
        lambda_1_interp = np.interp(mass_1_source, eos_data['masses'], eos_data['Lambdas'])
        lambda_2_interp = np.interp(mass_2_source, eos_data['masses'], eos_data['Lambdas'])
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            lambda_1_interp, lambda_2_interp,
            injection_parameters['mass_1'], injection_parameters['mass_2']
        )

    injection_parameters['chirp_mass_source'] = chirp_mass_source
    injection_parameters['lambda_tilde'] = lambda_tilde

    return injection_parameters


def make_anna_tim_favourite_plot(
    save_name: str = "./figures/GW_PE/anna_tim_favourite_plot.pdf",
    overwrite: bool = False,
    show_injection_truth: bool = True,
    xlim: tuple = (1.1, 1.5),
    ylim: tuple = (0, 2000),
    eos_name: str = "GW170817",
    nb_eos_samples: int = 5_000,
    mass_min: float = 0.25,
    mass_max: float = 2.0,
    Lambda_min: float = 0.0,
    Lambda_max: float = 15000.0
):
    """
    Create Anna and Tim's favourite plot: GW170817 posterior with GW231109 ET blob.

    Args:
        save_name (str): Output filename
        overwrite (bool): Whether to overwrite existing plot
        show_injection_truth (bool): Whether to show injection truth as a star
        xlim (tuple): X-axis limits for chirp mass
        ylim (tuple): Y-axis limits for lambda tilde
        eos_name (str): Name of EOS dataset to plot in background (e.g., "GW170817")
        nb_eos_samples (int): Number of EOS curves to plot
        mass_min (float): Minimum mass for EOS curves
        mass_max (float): Maximum mass for EOS curves
        Lambda_min (float): Minimum Lambda for EOS curves
        Lambda_max (float): Maximum Lambda for EOS curves

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists, skipping...")
            return True

        print("=" * 60)
        print("Creating Anna and Tim's Favourite Plot")
        print("=" * 60)

        # Load data
        print("\nLoading GW170817 posterior...")
        chirp_mass_GW170817, lambda_tilde_GW170817, dL_GW170817 = load_GW170817_posterior()

        # Conditionally load ET posterior
        chirp_mass_ET, lambda_tilde_ET, dL_ET = None, None, None
        if PLOT_GW231109_ET:
            print("\nLoading GW231109 ET posterior...")
            chirp_mass_ET, lambda_tilde_ET, dL_ET = load_GW231109_ET_posterior()

        # Conditionally load ET+CE posterior
        chirp_mass_ET_CE, lambda_tilde_ET_CE, dL_ET_CE = None, None, None
        if PLOT_GW231109_ET_CE:
            print("\nLoading GW231109 ET+CE posterior...")
            chirp_mass_ET_CE, lambda_tilde_ET_CE, dL_ET_CE = load_GW231109_ET_CE_posterior()

        # Create debug plot for luminosity distance if enabled
        if PLOT_DEBUG_LUMINOSITY_DISTANCE:
            debug_datasets = {
                'GW170817': {
                    'luminosity_distance': dL_GW170817,
                    'color': GW170817_COLOR,
                    'label': 'GW170817 (2G)'
                }
            }
            if PLOT_GW231109_ET and dL_ET is not None:
                debug_datasets['ET'] = {
                    'luminosity_distance': dL_ET,
                    'color': GW231109_ET_COLOR,
                    'label': 'GW231109 (ET)'
                }
            if PLOT_GW231109_ET_CE and dL_ET_CE is not None:
                debug_datasets['ET+CE'] = {
                    'luminosity_distance': dL_ET_CE,
                    'color': GW231109_ET_CE_COLOR,
                    'label': 'GW231109 (ET+CE)'
                }
            plot_debug_luminosity_distance(debug_datasets)

        # Create debug plot for redshift if enabled
        if PLOT_DEBUG_REDSHIFT:
            debug_datasets = {
                'GW170817': {
                    'luminosity_distance': dL_GW170817,
                    'color': GW170817_COLOR,
                    'label': 'GW170817 (2G)'
                }
            }
            if PLOT_GW231109_ET and dL_ET is not None:
                debug_datasets['ET'] = {
                    'luminosity_distance': dL_ET,
                    'color': GW231109_ET_COLOR,
                    'label': 'GW231109 (ET)'
                }
            if PLOT_GW231109_ET_CE and dL_ET_CE is not None:
                debug_datasets['ET+CE'] = {
                    'luminosity_distance': dL_ET_CE,
                    'color': GW231109_ET_CE_COLOR,
                    'label': 'GW231109 (ET+CE)'
                }
            plot_debug_redshift(debug_datasets)

        # Get injection truth if requested
        injection_params = None
        if show_injection_truth:
            print("\nLoading injection parameters...")
            injection_params = get_injection_parameters()
            print(f"Injection chirp mass (source): {injection_params['chirp_mass_source']:.6f} Msun")
            print(f"Injection lambda tilde: {injection_params['lambda_tilde']:.2f}")

        # Create figure
        print("\nCreating plot...")
        fig = plt.figure(figsize=(8, 8))
        mchirp_min, mchirp_max = xlim

        # Initialize colormap variables for later use (even if not plotting GW170817 EOS)
        sm = None
        if PLOT_GW170817_EOS:
            # Load EOS curves
            print(f"\nLoading EOS curves from {eos_name}...")
            M, _, L, log_prob = load_eos_curves(eos_name)
            max_samples = len(log_prob)
            print(f"Loaded {max_samples} EOS curves")

            # Convert to probabilities
            prob = np.exp(log_prob)
            max_log_prob_idx = np.argmax(log_prob)

            # Downsample the samples
            indices = np.random.choice(max_samples, nb_eos_samples, replace=False)
            indices = np.append(indices, max_log_prob_idx)  # ensure max prob sample is included

            # Get colormap for probability coloring
            norm = plt.Normalize(vmin=np.min(prob), vmax=np.max(prob))
            cmap = sns.color_palette("crest", as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

            # Plot EOS curves in background
            print(f"Plotting {len(indices)} GW170817 EOS curves in background...")
            bad_counter = 0
            for i in indices:
                # Get the color for this sample
                normalized_value = norm(prob[i])
                color = cmap(normalized_value)

                if np.isnan(color).any():
                    bad_counter += 1
                    continue

                if np.any(M[i] < 0.0) or np.any(L[i] < 0.0):
                    bad_counter += 1
                    continue

                # Mask since a few datapoints might be bad
                mask_masses = (M[i] > mass_min) * (M[i] < mass_max)
                mask_Lambdas = (L[i] > Lambda_min) * (L[i] < Lambda_max)
                mask = mask_masses & mask_Lambdas

                # Get the mass-Lambda curve for this particular EOS
                EOS_masses, EOS_Lambdas = M[i][mask], L[i][mask]
                mchirp_array, lambda_tilde_eos = get_mchirp_lambda_tilde_EOS(
                    EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max
                )

                # Plot it in the background
                plt.plot(mchirp_array, lambda_tilde_eos, color=color, alpha=0.2,
                        lw=2.0, zorder=normalized_value, rasterized=True)

            print(f"Skipped {bad_counter} EOS samples due to NaNs or invalid values")

        # Plot Jester EOS curves from ET inference if enabled
        if PLOT_JESTER_ET_EOS:
            print("\nPlotting Jester ET EOS curves...")
            M_jester_et, L_jester_et = load_jester_eos_samples("ET")
            if M_jester_et is not None:
                jester_bad_counter = 0
                for i in range(len(M_jester_et)):
                    if np.any(M_jester_et[i] < 0.0) or np.any(L_jester_et[i] < 0.0):
                        jester_bad_counter += 1
                        continue

                    # Mask since a few datapoints might be bad
                    mask_masses = (M_jester_et[i] > mass_min) * (M_jester_et[i] < mass_max)
                    mask_Lambdas = (L_jester_et[i] > Lambda_min) * (L_jester_et[i] < Lambda_max)
                    mask = mask_masses & mask_Lambdas

                    # Get the mass-Lambda curve for this particular EOS
                    EOS_masses, EOS_Lambdas = M_jester_et[i][mask], L_jester_et[i][mask]
                    mchirp_array, lambda_tilde_eos = get_mchirp_lambda_tilde_EOS(
                        EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max
                    )

                    # Plot with distinct styling (ET color, slightly thicker, more opaque)
                    plt.plot(mchirp_array, lambda_tilde_eos, color=GW231109_ET_COLOR,
                            alpha=0.4, lw=2.5, zorder=1500, rasterized=True)

                print(f"Plotted {len(M_jester_et) - jester_bad_counter} Jester ET EOS curves")

        # Plot Jester EOS curves from ET+CE inference if enabled
        if PLOT_JESTER_ET_CE_EOS:
            print("\nPlotting Jester ET+CE EOS curves...")
            M_jester_et_ce, L_jester_et_ce = load_jester_eos_samples("ET_CE")
            if M_jester_et_ce is not None:
                jester_bad_counter = 0
                for i in range(len(M_jester_et_ce)):
                    if np.any(M_jester_et_ce[i] < 0.0) or np.any(L_jester_et_ce[i] < 0.0):
                        jester_bad_counter += 1
                        continue

                    # Mask since a few datapoints might be bad
                    mask_masses = (M_jester_et_ce[i] > mass_min) * (M_jester_et_ce[i] < mass_max)
                    mask_Lambdas = (L_jester_et_ce[i] > Lambda_min) * (L_jester_et_ce[i] < Lambda_max)
                    mask = mask_masses & mask_Lambdas

                    # Get the mass-Lambda curve for this particular EOS
                    EOS_masses, EOS_Lambdas = M_jester_et_ce[i][mask], L_jester_et_ce[i][mask]
                    mchirp_array, lambda_tilde_eos = get_mchirp_lambda_tilde_EOS(
                        EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max
                    )

                    # Plot with distinct styling (ET+CE color, slightly thicker, more opaque)
                    plt.plot(mchirp_array, lambda_tilde_eos, color=GW231109_ET_CE_COLOR,
                            alpha=0.4, lw=2.5, zorder=2000, rasterized=True)

                print(f"Plotted {len(M_jester_et_ce) - jester_bad_counter} Jester ET+CE EOS curves")

        # Plot the injection EOS curve prominently
        print("Plotting injection EOS curve...")
        inj_mchirp, inj_lambda_tilde = load_injection_eos_curve(xlim)
        if inj_mchirp is not None:
            plt.plot(inj_mchirp, inj_lambda_tilde, color=TRUTH_COLOR, alpha=1.0,
                    lw=3.0, zorder=200000, linestyle='--', label='Injection')

        # Set up corner kwargs based on display mode
        if CORNER_PLOT_MODE == "scatter":
            print(f"Using scatter mode for posterior plots")
            # Scatter mode: directly plot with matplotlib scatter for better zorder control
            ax = fig.gca()

            # Plot GW170817 posterior first (lower z-order)
            print("Plotting GW170817 posterior samples...")
            ax.scatter(chirp_mass_GW170817, lambda_tilde_GW170817,
                      color=GW170817_COLOR, alpha=0.3, s=1,
                      zorder=5000, rasterized=True)

            # Plot GW231109 ET posterior on top (higher z-order)
            if PLOT_GW231109_ET and chirp_mass_ET is not None:
                print("Plotting GW231109 ET posterior samples...")
                ax.scatter(chirp_mass_ET, lambda_tilde_ET,
                          color=GW231109_ET_COLOR, alpha=0.3, s=1,
                          zorder=6000, rasterized=True)

            # Plot GW231109 ET+CE posterior on top (highest z-order)
            if PLOT_GW231109_ET_CE and chirp_mass_ET_CE is not None:
                print("Plotting GW231109 ET+CE posterior samples...")
                ax.scatter(chirp_mass_ET_CE, lambda_tilde_ET_CE,
                          color=GW231109_ET_CE_COLOR, alpha=0.3, s=1,
                          zorder=7000, rasterized=True)

        else:  # "contours" mode
            print(f"Using contours mode for corner plots (68% and 95% credible regions)")
            # Contours mode: show 1σ and 2σ credible regions
            mode_kwargs = {
                'plot_datapoints': False,
                'plot_density': False,
                'plot_contours': True,
                'fill_contours': True,
                'levels': [0.68, 0.95]
            }

            # Plot GW170817 posterior first (lower z-order)
            print("Plotting GW170817 posterior...")
            corner_kwargs_gw170817 = default_corner_kwargs.copy()
            corner_kwargs_gw170817.update(mode_kwargs)
            corner_kwargs_gw170817['color'] = GW170817_COLOR
            corner_kwargs_gw170817['zorder'] = 5000
            corner.hist2d(
                chirp_mass_GW170817,
                lambda_tilde_GW170817,
                fig=fig,
                **corner_kwargs_gw170817
            )

            # Plot GW231109 ET posterior on top (higher z-order)
            if PLOT_GW231109_ET and chirp_mass_ET is not None:
                print("Plotting GW231109 ET posterior...")
                corner_kwargs_et = default_corner_kwargs.copy()
                corner_kwargs_et.update(mode_kwargs)
                corner_kwargs_et['color'] = GW231109_ET_COLOR
                corner_kwargs_et['zorder'] = 6000
                corner.hist2d(
                    chirp_mass_ET,
                    lambda_tilde_ET,
                    fig=fig,
                    **corner_kwargs_et
                )

            # Plot GW231109 ET+CE posterior on top (highest z-order)
            if PLOT_GW231109_ET_CE and chirp_mass_ET_CE is not None:
                print("Plotting GW231109 ET+CE posterior...")
                corner_kwargs_et_ce = default_corner_kwargs.copy()
                corner_kwargs_et_ce.update(mode_kwargs)
                corner_kwargs_et_ce['color'] = GW231109_ET_CE_COLOR
                corner_kwargs_et_ce['zorder'] = 7000
                corner.hist2d(
                    chirp_mass_ET_CE,
                    lambda_tilde_ET_CE,
                    fig=fig,
                    **corner_kwargs_et_ce
                )

        # Plot injection truth as star
        if show_injection_truth and injection_params is not None:
            plt.scatter(
                injection_params['chirp_mass_source'],
                injection_params['lambda_tilde'],
                color=TRUTH_COLOR,
                marker='*',
                s=INJECTION_MARKER_SIZE,
                alpha=1.0,
                zorder=300000,
                label='Injection',
                edgecolors=INJECTION_MARKER_EDGECOLOR,
                linewidths=INJECTION_MARKER_EDGEWIDTH
            )

        # Set labels and limits
        plt.xlabel(r"$\mathcal{M}_c^{\rm{src}}$ [M$_\odot$]")
        plt.ylabel(r"$\tilde{\Lambda}$")
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Add legend (only include elements that are actually plotted)
        legend_elements = [
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817 (2G)')
        ]

        if PLOT_GW231109_ET and chirp_mass_ET is not None:
            legend_elements.append(
                mpatches.Patch(facecolor=GW231109_ET_COLOR, edgecolor='k', label='GW231109 (ET)')
            )

        if PLOT_GW231109_ET_CE and chirp_mass_ET_CE is not None:
            legend_elements.append(
                mpatches.Patch(facecolor=GW231109_ET_CE_COLOR, edgecolor='k', label='GW231109 (ET+CE)')
            )

        if inj_mchirp is not None:
            legend_elements.append(
                plt.Line2D([0], [0], color=TRUTH_COLOR, lw=3, linestyle='--', label='Injection')
            )

        # if show_injection_truth:
        #     legend_elements.append(
        #         plt.Line2D([0], [0], marker='*', color='w',
        #                   markerfacecolor=TRUTH_COLOR, markersize=15,
        #                   label='Injection',
        #                   markeredgecolor=INJECTION_MARKER_EDGECOLOR,
        #                   markeredgewidth=INJECTION_MARKER_EDGEWIDTH)
        #     )

        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=LEGEND_FONTSIZE)

        # Add colorbar for EOS posterior probability (only if GW170817 EOS curves were plotted)
        if sm is not None:
            ax = fig.gca()
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03, aspect=35)
            cbar.set_label(f"EOS posterior density", fontsize=COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=COLORBAR_TICKS_FONTSIZE)

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        # Save plot
        print(f"\nSaving plot to {save_name}")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

        print("\n" + "=" * 60)
        print("✓ Successfully created Anna and Tim's favourite plot!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Failed to create plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to create the favourite plot.
    """
    # Create the plot with default settings
    success = make_anna_tim_favourite_plot(
        save_name="./figures/GW_PE/anna_tim_favourite_plot.pdf",
        overwrite=True,
        show_injection_truth=True,
        xlim=(1.101, 1.45),
        ylim=(0, 1300)
    )

    if not success:
        print("Failed to create the plot. Please check the error messages above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
