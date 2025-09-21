"""
Utility functions for the final money plots.
This module extracts common functionality from the existing plotting scripts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import json
import corner
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.conversion import luminosity_distance_to_redshift
from bilby.core.prior import PriorDict

# Add parent directory to path to import main utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils as main_utils

# Matplotlib configuration
PLOT_PARAMS = {
    "axes.grid": False,
    "text.usetex": False,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "figure.titlesize": 16
}

plt.rcParams.update(PLOT_PARAMS)

# Corner plot defaults
DEFAULT_CORNER_KWARGS = dict(
    bins=40,
    smooth=1.,
    show_titles=False,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16),
    color="blue",
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=4,
    min_n_ticks=3,
    truth_color="red",
    save=False
)

# Color scheme
GW231109_COLOR = "red"
GW190425_COLOR = "purple"
PRIOR_COLOR = "gray"
GW170817_COLOR = "orange"

# Standard comparison colors
ORANGE = "#de8f07"
BLUE = "#0472b1"
GREEN = "#019e72"

def identify_person_from_path(source_dir: str) -> str:
    """
    Identify the person who ran the analysis from the source directory path.

    Args:
        source_dir (str): Directory path containing the run

    Returns:
        str: Person identifier
    """
    if "puecher" in source_dir:
        return "anna"
    elif "dietrich6" in source_dir:
        return "tim"
    elif "wouters" in source_dir:
        return "thibeau"
    else:
        raise ValueError(f"Cannot identify person from source_dir: {source_dir}")

def generate_save_path(source_dir: str,
                      plot_type: str,
                      prefix: str = "",
                      suffix: str = ".pdf") -> str:
    """
    Generate standardized save path for plots.

    Args:
        source_dir (str): Source directory path
        plot_type (str): Type of plot (e.g., 'cornerplots', 'mass_Lambdas_figs')
        prefix (str): Optional prefix for filename
        suffix (str): File extension

    Returns:
        str: Full save path
    """
    person = identify_person_from_path(source_dir)
    run_name = source_dir.split("/")[-1]

    filename = f"{person}_{run_name}"
    if prefix:
        filename = f"{prefix}_{filename}"
    filename += suffix

    save_path = os.path.join(".", plot_type, filename)

    return save_path

def fetch_posterior_filename(source_dir: str):
    """
    Given a source_dir where a GW inference run was performed in /work/, return the path to the posterior HDF5 file if it exists,
    otherwise return None.

    Args:
        source_dir (str): Directory containing the run output. NOTE: It should have
                          outdir/final_result/ with an HDF5 posterior file.

    Returns:
        str or None: Path to the posterior file, or None if not found.
    """

    source_dir = os.path.abspath(source_dir)
    final_results_dir = os.path.join(source_dir, "outdir/final_result")

    if not os.path.exists(final_results_dir):
        return None

    posterior_files = [
        f for f in os.listdir(final_results_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    ]

    if not posterior_files:
        print(f"WARNING: No HDF5 files found in {final_results_dir}. Trying to look for JSON files now")
        final_results_dir = os.path.join(source_dir, "outdir")
        posterior_files = [
            f for f in os.listdir(final_results_dir)
            if f.endswith(".json")
        ]
        if not posterior_files:
            return None

    # Just return the first one found, there should be only one anyways
    return os.path.join(final_results_dir, posterior_files[0])

def load_posterior_samples(source_dir: str,
                          keys_to_fetch: list[str],
                          chirp_tilde: bool = False) -> tuple:
    """
    Load posterior samples from HDF5 or JSON files.

    Args:
        source_dir (str): Directory containing posterior files
        keys_to_fetch (list[str]): Parameters to extract
        chirp_tilde (bool): Whether to return chirp mass and lambda_tilde format

    Returns:
        tuple: Posterior samples as numpy arrays
    """
    posterior_file = fetch_posterior_filename(source_dir)
    if posterior_file is None:
        raise FileNotFoundError(f"No posterior file found in {source_dir}")

    if chirp_tilde:
        return main_utils.fetch_mass_Lambdas_samples(posterior_file, chirp_tilde=True)
    else:
        # Load specific keys
        if posterior_file.endswith(('.h5', '.hdf5')):
            with h5py.File(posterior_file, 'r') as f:
                posterior = f["posterior"]
                samples = np.array([posterior[key][()] for key in keys_to_fetch]).T
        else:  # JSON
            with open(posterior_file, 'r') as f:
                posterior = json.load(f)['posterior']['content']
                samples = np.array([posterior[key] for key in keys_to_fetch]).T

        return samples

def load_run_metadata(source_dir: str) -> dict:
    """
    Load run metadata including sampling time and Bayes factor.

    Args:
        source_dir (str): Directory containing the run

    Returns:
        dict: Metadata dictionary
    """
    posterior_file = fetch_posterior_filename(source_dir)
    metadata = {}

    if posterior_file and posterior_file.endswith(('.h5', '.hdf5')):
        with h5py.File(posterior_file, 'r') as f:
            if "log_bayes_factor" in f:
                metadata["log_bayes_factor"] = f["log_bayes_factor"][()]
            if "sampling_time" in f:
                metadata["sampling_time"] = f["sampling_time"][()]
                metadata["sampling_time_hrs"] = metadata["sampling_time"] / 3600.0

            # Load priors
            if "priors" in f:
                priors_bytes = f["priors"][()]
                priors_str = priors_bytes.decode()
                metadata["priors"] = json.loads(priors_str)

    return metadata

def load_priors_for_corner(source_dir: str) -> list[str]:
    """
    Load and process priors for corner plot parameter selection.

    Args:
        source_dir (str): Directory containing the run

    Returns:
        list[str]: List of parameter keys to plot
    """
    metadata = load_run_metadata(source_dir)
    priors_dict = metadata.get("priors", {})

    # Filter out unwanted keys
    prior_keys_to_skip = ['__prior_dict__', '__module__', '__name__']
    prior_keys = [k for k, v in priors_dict.items()
                  if k not in prior_keys_to_skip and "recalib" not in k]

    # Handle fixed parameters
    if "fixed_dL" in source_dir:
        prior_keys = [k for k in prior_keys if k != "luminosity_distance"]
    if "fixed_sky" in source_dir:
        prior_keys = [k for k in prior_keys if k not in ["ra", "dec"]]

    # Remove quasi-universal relation parameters
    keys_to_remove = ["lambda_symmetric", "lambda_antisymmetric", "binary_love_uniform"]
    prior_keys = [k for k in prior_keys if k not in keys_to_remove]

    return prior_keys

def load_eos_curves(eos_name: str, base_path: str = "../figures") -> tuple:
    """
    Load EOS curves from NPZ files.

    Args:
        eos_name (str): Name of the EOS
        base_path (str): Base path to EOS data directory

    Returns:
        tuple: (M, R, L, log_prob) arrays
    """
    filename = os.path.join(base_path, "EOS_data", f"{eos_name}.npz")
    data = np.load(filename)
    M, R, L = data['M'], data['R'], data['L']
    log_prob = data['log_prob']
    return M, R, L, log_prob

def get_mchirp_lambda_tilde_EOS(EOS_masses: np.array,
                               EOS_Lambdas: np.array,
                               mchirp_min: float,
                               mchirp_max: float) -> tuple[np.array, np.array]:
    """
    Convert EOS mass-Lambda curves to chirp mass - lambda_tilde space.

    Args:
        EOS_masses (np.array): Array of EOS masses
        EOS_Lambdas (np.array): Array of EOS Lambdas
        mchirp_min (float): Minimum chirp mass
        mchirp_max (float): Maximum chirp mass

    Returns:
        tuple: (mchirp_array, lambda_tilde) arrays
    """
    mchirp_array = np.linspace(mchirp_min, mchirp_max, 100)
    q_array = np.ones_like(mchirp_array)  # Assuming equal mass ratio

    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(mchirp_array, q_array)
    lambda_1 = np.interp(mass_1, EOS_masses, EOS_Lambdas)
    lambda_2 = np.interp(mass_2, EOS_masses, EOS_Lambdas)

    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    return mchirp_array, lambda_tilde

def load_GW170817_PE(chirp_tilde: bool = False, base_path: str = "../figures") -> tuple:
    """
    Load GW170817 posterior samples.

    Args:
        chirp_tilde (bool): Whether to return chirp mass format
        base_path (str): Base path to data directory

    Returns:
        tuple: Posterior sample arrays
    """
    filename = os.path.join(base_path, "EOS_data", "PE_posterior_samples_GW170817.npz")
    data = np.load(filename)

    if chirp_tilde:
        chirp_mass = data['chirp_mass_source']
        mass_ratio = data['mass_ratio']
        lambda_tilde = data['lambda_tilde']
        delta_lambda_tilde = data['delta_lambda_tilde']
        return chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde
    else:
        mass_1_source = data['mass_1_source']
        mass_2_source = data['mass_2_source']
        lambda_1 = data['lambda_1']
        lambda_2 = data['lambda_2']
        return mass_1_source, mass_2_source, lambda_1, lambda_2

def load_GW190425_PE(chirp_tilde: bool = False, base_path: str = "../figures") -> tuple:
    """
    Load GW190425 posterior samples.

    Args:
        chirp_tilde (bool): Whether to return chirp mass format
        base_path (str): Base path to data directory

    Returns:
        tuple: Posterior sample arrays
    """
    raise NotImplementedError("Need to obtain npz of posterior")

def fetch_prior_samples(source_dir: str, nb_samples: int = 10_000) -> tuple[np.array, np.array]:
    """
    Generate prior samples for chirp mass and lambda_tilde.

    Args:
        source_dir (str): Directory containing priors information
        nb_samples (int): Number of samples to generate

    Returns:
        tuple: (chirp_mass_source, lambda_tilde) arrays
    """
    priors = main_utils.initialize_priors(
        source_dir,
        ["chirp_mass", "mass_ratio", "lambda_1", "lambda_2", "luminosity_distance"]
    )
    prior = PriorDict(priors)

    samples = prior.sample(size=nb_samples)
    chirp_mass = samples["chirp_mass"]
    mass_ratio = samples["mass_ratio"]
    lambda_1 = samples["lambda_1"]
    lambda_2 = samples["lambda_2"]

    z = luminosity_distance_to_redshift(samples["luminosity_distance"])
    chirp_mass_source = chirp_mass / (1 + z)
    mass_1_source, mass_2_source = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass_source, mass_ratio
    )
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1_source, mass_2_source)

    return chirp_mass_source, lambda_tilde

def create_legend_elements(include_gw170817: bool = True,
                          include_prior: bool = False,
                          run_label: str = "GW231109") -> list:
    """
    Create legend elements for plots.

    Args:
        include_gw170817 (bool): Whether to include GW170817
        include_prior (bool): Whether to include prior
        run_label (str): Label for the main run

    Returns:
        list: Legend patch elements
    """
    elements = []

    if include_gw170817:
        elements.append(
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817')
        )

    # Determine color based on run label
    if run_label == "GW190425":
        color = GW190425_COLOR
    else:
        color = GW231109_COLOR

    elements.append(
        mpatches.Patch(facecolor=color, edgecolor='k', label=run_label)
    )

    if include_prior:
        elements.append(
            mpatches.Patch(facecolor=PRIOR_COLOR, edgecolor='k', label='Prior')
        )

    return elements

def ensure_directory_exists(filepath: str):
    """
    Ensure the directory for a filepath exists.

    Args:
        filepath (str): Full file path
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)