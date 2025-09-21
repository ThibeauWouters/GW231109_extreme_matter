#!/usr/bin/env python3
"""
Final money plots script for GW231109 extreme matter investigations.
This script creates comparison corner plots where multiple analysis runs
are overlaid for direct comparison.
"""

import os
import argparse
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner

from utils import (
    DEFAULT_CORNER_KWARGS, GW231109_COLOR, GW190425_COLOR, PRIOR_COLOR, GW170817_COLOR,
    ORANGE, BLUE, GREEN, INJECTION_COLOR, identify_person_from_path, load_posterior_samples,
    load_run_metadata, load_priors_for_corner, ensure_directory_exists
)


# If running on Mac, so we can use TeX (not on Jarvis), change some rc params
cwd = os.getcwd()
if "Woute029" in cwd:
    print(f"Updating plotting parameters for TeX")
    fs = 18
    ticks_fs = 16
    legend_fs = 40
    rc_params = {"axes.grid": False,
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"],
            "xtick.labelsize": ticks_fs,
            "ytick.labelsize": ticks_fs,
            "axes.labelsize": ticks_fs,
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

def generate_run_cache_key(source_dir: str) -> str:
    """
    Generate a unique cache key for a single run based on directory path.

    Args:
        source_dir (str): Source directory path

    Returns:
        str: Unique cache key
    """
    # Extract meaningful name from directory
    dir_name = os.path.basename(source_dir.rstrip('/'))
    # Remove common prefixes to make it cleaner
    dir_name = dir_name.replace('prod_BW_XP_s005_', '').replace('_default', '')
    return dir_name

def generate_run_cache_filename(cache_key: str, parameters: list[str]) -> str:
    """
    Generate cache filename for a single run.

    Args:
        cache_key (str): Unique key for this run
        parameters (list[str]): List of parameters

    Returns:
        str: Cache filename
    """
    params_part = "_".join(sorted(parameters))
    cache_name = f"run_{cache_key}_{params_part}.npz"
    cache_name = cache_name.replace('/', '_').replace(' ', '_')
    return f"./data/{cache_name}"

def save_run_data(cache_key: str, samples: np.ndarray, parameters: list[str]) -> bool:
    """
    Save data for a single run to cache.

    Args:
        cache_key (str): Unique key for this run
        samples (np.ndarray): Posterior samples array
        parameters (list[str]): Parameter names

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        filename = generate_run_cache_filename(cache_key, parameters)
        ensure_directory_exists(filename)

        save_dict = {'parameters': parameters}
        for i, param in enumerate(parameters):
            save_dict[param] = samples[:, i]

        np.savez(filename, **save_dict)
        print(f"Saved run data to cache: {cache_key} -> {filename}")
        return True
    except Exception as e:
        print(f"Failed to save run cache data for {cache_key}: {e}")
        return False

def load_run_data(cache_key: str, parameters: list[str]) -> np.ndarray:
    """
    Load data for a single run from cache.

    Args:
        cache_key (str): Unique key for this run
        parameters (list[str]): Desired parameter order

    Returns:
        np.ndarray: Sample array, or None if failed
    """
    try:
        filename = generate_run_cache_filename(cache_key, parameters)
        if not os.path.exists(filename):
            return None

        data = np.load(filename, allow_pickle=True)
        cached_parameters = data['parameters'].tolist()

        if not all(param in cached_parameters for param in parameters):
            print(f"Cache for {cache_key} missing required parameters. Required: {parameters}, Cached: {cached_parameters}")
            return None

        # Reconstruct samples with correct parameter order
        samples = []
        for param in parameters:
            if param in data:
                samples.append(data[param])
            else:
                raise KeyError(f"Missing parameter {param} in cache for {cache_key}")

        result = np.column_stack(samples)
        print(f"Loaded run data from cache: {cache_key}")
        return result

    except Exception as e:
        print(f"Failed to load run cache data for {cache_key}: {e}")
        return None

def load_and_cache_run(source_dir: str, parameters: list[str]) -> str:
    """
    Load a single run and cache it. Returns the cache key.

    Args:
        source_dir (str): Source directory path
        parameters (list[str]): Parameters to load

    Returns:
        str: Cache key for the run, or None if failed
    """
    try:
        cache_key = generate_run_cache_key(source_dir)

        # Check if already cached
        cached_samples = load_run_data(cache_key, parameters)
        if cached_samples is not None:
            print(f"Run {cache_key} already cached")
            return cache_key

        print(f"Loading and caching run: {source_dir} -> {cache_key}")

        # Load metadata
        metadata = load_run_metadata(source_dir)
        if "log_bayes_factor" in metadata:
            print(f"  Log Bayes factor: {metadata['log_bayes_factor']}")
        if "sampling_time_hrs" in metadata:
            print(f"  Sampling time: {metadata['sampling_time_hrs']:.2f} hours")

        # Load posterior samples
        samples = load_posterior_samples(source_dir, parameters)
        print(f"  Loaded {len(samples)} samples")

        # Save to cache
        if save_run_data(cache_key, samples, parameters):
            return cache_key
        else:
            return None

    except Exception as e:
        print(f"Failed to load and cache run {source_dir}: {e}")
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
            print(f"  Index {i}: {labels[i]} -> {colors[i]} -> z-order {zorders[i]}")

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

    return all_samples, valid_labels, valid_colors, valid_zorders

def create_comparison_cornerplot(run_keys: list[str],
                               parameters: list[str],
                               labels: list[str],
                               colors: list[str],
                               ranges: dict = None,
                               zorders: list[int] = None,
                               save_name: str = "comparison_cornerplot.pdf",
                               overwrite: bool = False,
                               dummy_normalization_keys: list[int] = None) -> bool:
    """
    Create a comparison corner plot with multiple runs overlaid.

    Args:
        run_keys (list[str]): List of unique run keys to plot
        parameters (list[str]): Parameters to include in the corner plot
        labels (list[str]): Labels for each run
        colors (list[str]): Colors for each run
        ranges (dict): Parameter ranges as {param: (min, max)} (optional)
        zorders (list[int]): Z-order for each run (higher values appear on top, optional)
        save_name (str): Output filename
        overwrite (bool): Whether to overwrite existing plots
        dummy_normalization_keys (list[int]): Indices of runs to use for each parameter
            to create a dummy normalization dataset (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists, skipping...")
            return True

        print(f"Creating comparison corner plot for {len(run_keys)} runs")
        print(f"Run keys: {run_keys}")
        print(f"Parameters: {parameters}")

        # Set default z-orders if not provided
        if zorders is None:
            zorders = list(range(len(run_keys)))

        # Load data for each run by key
        all_samples = []
        for i, run_key in enumerate(run_keys):
            print(f"Loading run: {run_key} -> {labels[i]} ({colors[i]}, z-order: {zorders[i]})")

            # Try to load from cache first
            samples = load_run_data(run_key, parameters)

            if samples is None:
                print(f"  Cache miss, need to load and cache run: {run_key}")
                # For now, return False - we'll need to load and cache runs separately
                print(f"  ERROR: Run {run_key} not found in cache!")
                return False

            all_samples.append(samples)
            print(f"  Loaded {len(samples)} samples")

        if not all_samples:
            print("No valid samples loaded!")
            return False

        # Create dummy normalization dataset if requested
        dummy_dataset = None
        if dummy_normalization_keys is not None:
            # Validate dummy_normalization_keys
            n_params = len(parameters)
            if len(dummy_normalization_keys) != n_params:
                raise ValueError(f"dummy_normalization_keys must have length {n_params} (number of parameters), got {len(dummy_normalization_keys)}")

            # Check that all keys are valid run indices
            n_runs = len(all_samples)
            for i, idx in enumerate(dummy_normalization_keys):
                if not (0 <= idx < n_runs):
                    raise ValueError(f"dummy_normalization_keys[{i}] = {idx} is not a valid run index (0 to {n_runs-1})")

            # Find minimum number of samples across all runs
            min_samples = min(len(samples) for samples in all_samples)
            print(f"Creating dummy normalization dataset using run indices: {dummy_normalization_keys}")
            print(f"Using minimum sample count: {min_samples}")

            # Create dummy dataset by selecting specified run for each parameter
            dummy_columns = []
            for param_idx, run_idx in enumerate(dummy_normalization_keys):
                samples = all_samples[run_idx]
                # Downsample to min_samples by taking first min_samples rows
                downsampled_samples = samples[:min_samples]
                dummy_columns.append(downsampled_samples[:, param_idx])

            dummy_dataset = np.column_stack(dummy_columns)
            print(f"Created dummy dataset with shape: {dummy_dataset.shape}")

        # Create the corner plot with the first dataset
        print("Creating corner plot...")

        # Set up corner kwargs
        corner_kwargs = DEFAULT_CORNER_KWARGS.copy()
        corner_kwargs["color"] = colors[0]
        corner_kwargs["zorder"] = zorders[0]

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
            # Deep copy the range to avoid modification issues
            if "range" in corner_kwargs_overlay:
                corner_kwargs_overlay["range"] = corner_kwargs_overlay["range"].copy()
            corner_kwargs_overlay["color"] = colors[i]
            corner_kwargs_overlay["zorder"] = zorders[i]
            corner_kwargs_overlay["fig"] = fig

            corner.corner(all_samples[i],
                         labels=parameter_labels,
                         **corner_kwargs_overlay)

        # Plot dummy normalization dataset last (invisible, for normalization)
        if dummy_dataset is not None:
            print("Adding dummy normalization dataset...")
            invisible_kwargs = DEFAULT_CORNER_KWARGS.copy()
            invisible_kwargs.update({
                'labels': parameter_labels,
                'hist_kwargs': {'alpha': 0, 'density': True},
                'plot_density': False,     # Disable 2D density plots
                'plot_contours': False,    # Disable 2D contours
                'plot_datapoints': False,  # Disable scatter points
                'no_fill_contours': True,  # Disable filled contours
                'color': 'black',
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

            corner.corner(dummy_dataset,
                         labels=parameter_labels,
                         **invisible_kwargs)

        # Add legend
        legend_elements = []
        for i, (label, color) in enumerate(zip(labels, colors)):
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

def generate_injection_cache_filename(filepath: str, parameters: list[str]) -> str:
    """
    Generate a cache filename for injection data based on filepath and parameters.

    Args:
        filepath (str): Path to the JSON result file
        parameters (list[str]): List of parameters

    Returns:
        str: Cache filename
    """
    import hashlib

    # Create a hash from the filepath for unique identification
    filepath_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]

    # Extract meaningful name from filepath
    filename = os.path.basename(filepath).replace('_result.json', '')

    # Create descriptive name
    params_part = "_".join(sorted(parameters))
    cache_name = f"injection_{filename}_{filepath_hash}_{params_part}.npz"

    return f"./data/{cache_name}"

def save_injection_data(filename: str, posterior_samples: np.ndarray,
                       injection_params: dict, parameters: list[str]) -> bool:
    """
    Save injection data to cache file.

    Args:
        filename (str): Cache filename
        posterior_samples (np.ndarray): Posterior samples array
        injection_params (dict): Injection parameter values
        parameters (list[str]): Parameter names for proper column mapping

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory_exists(filename)

        # Create save dictionary
        save_dict = {
            'parameters': parameters,
            'injection_params': injection_params
        }

        # Save posterior samples with parameter names
        for i, param in enumerate(parameters):
            save_dict[f'posterior_{param}'] = posterior_samples[:, i]

        np.savez(filename, **save_dict)
        print(f"Saved injection data to cache: {filename}")
        return True
    except Exception as e:
        print(f"Failed to save injection cache data: {e}")
        return False

def load_injection_data(filename: str, parameters: list[str]) -> tuple:
    """
    Load injection data from cache file.

    Args:
        filename (str): Cache filename
        parameters (list[str]): Desired parameter order

    Returns:
        tuple: (posterior_samples, injection_params) or (None, None) if failed
    """
    try:
        if not os.path.exists(filename):
            return None, None

        data = np.load(filename, allow_pickle=True)

        # Get cached parameter order
        cached_parameters = data['parameters'].tolist()

        # Check if we have all required parameters
        if not all(param in cached_parameters for param in parameters):
            print(f"Cache missing required parameters. Required: {parameters}, Cached: {cached_parameters}")
            return None, None

        # Reconstruct posterior samples with correct parameter order
        posterior_samples = []
        for param in parameters:
            key = f'posterior_{param}'
            if key in data:
                posterior_samples.append(data[key])
            else:
                raise KeyError(f"Missing parameter {param} in cache")

        posterior_samples = np.column_stack(posterior_samples)

        # Load injection parameters
        injection_params = data['injection_params'].item()

        print(f"Loaded injection data from cache: {filename}")
        return posterior_samples, injection_params

    except Exception as e:
        print(f"Failed to load injection cache data: {e}")
        return None, None

def load_injection_json(filepath: str, parameters: list[str]) -> tuple:
    """
    Load injection data from JSON file.

    Args:
        filepath (str): Path to JSON result file
        parameters (list[str]): Parameters to extract

    Returns:
        tuple: (posterior_samples, injection_params) or (None, None) if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load posterior samples
        if 'posterior' not in data or 'content' not in data['posterior']:
            raise ValueError("JSON file does not contain expected posterior structure")

        posterior = data['posterior']['content']
        samples = []
        for param in parameters:
            if param not in posterior:
                raise ValueError(f"Parameter {param} not found in posterior")
            samples.append(np.array(posterior[param]))

        posterior_samples = np.column_stack(samples)

        # Find injection parameters - prioritize injection_parameters location
        injection_params = {}

        # Check injection_parameters first (user confirmed this is the location)
        if 'injection_parameters' in data and isinstance(data['injection_parameters'], dict):
            injection_source = data['injection_parameters']
            print(f"Found injection_parameters with keys: {list(injection_source.keys())}")
        else:
            # Fallback to other locations if needed
            injection_sources = [
                data.get('meta_data', {}).get('injection_parameters', {}),
                data.get('meta_data', {}).get('injection', {}),
                data.get('injection', {}),
                data.get('truth', {})
            ]
            injection_source = {}
            for source in injection_sources:
                if isinstance(source, dict) and source:
                    injection_source = source
                    break

        # Extract injection values for our parameters
        for param in parameters:
            if param in injection_source:
                injection_params[param] = injection_source[param]
                print(f"Found injection value for {param}: {injection_source[param]}")
            else:
                print(f"Warning: Could not find injection value for parameter {param}")
                injection_params[param] = None

        return posterior_samples, injection_params

    except Exception as e:
        print(f"Failed to load injection JSON: {e}")
        return None, None

def plot_injection(filepath: str,
                  parameters: list[str] = None,
                  ranges: dict = None,
                  save_name: str = None,
                  overwrite: bool = False) -> bool:
    """
    Create corner plot for injection analysis with truth values shown in black.

    Args:
        filepath (str): Path to JSON result file (e.g., .../outdir/*result.json)
        parameters (list[str]): Parameters to plot (default: same as comparison plots)
        ranges (dict): Parameter ranges as {param: (min, max)} (optional)
        save_name (str): Output filename (default: auto-generated)
        overwrite (bool): Whether to overwrite existing plots

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use default parameters if not provided
        if parameters is None:
            parameters = [
                "chirp_mass",
                "mass_ratio",
                "chi_eff",
                "lambda_1",
                "lambda_2",
                "lambda_tilde"
            ]

        # Generate save name if not provided
        if save_name is None:
            basename = os.path.basename(filepath).replace('_result.json', '')
            save_name = f"injection_plot_{basename}.pdf"

        if os.path.exists(save_name) and not overwrite:
            print(f"File {save_name} already exists, skipping...")
            return True

        print(f"Creating injection plot for: {filepath}")
        print(f"Parameters: {parameters}")

        # Check cache first
        cache_filename = generate_injection_cache_filename(filepath, parameters)
        posterior_samples, injection_params = load_injection_data(cache_filename, parameters)

        if posterior_samples is None:
            print("Loading data from JSON file and caching...")
            posterior_samples, injection_params = load_injection_json(filepath, parameters)

            if posterior_samples is None:
                print("Failed to load injection data!")
                return False

            # Save to cache
            save_injection_data(cache_filename, posterior_samples, injection_params, parameters)
        else:
            print("Using cached injection data")

        # Create the corner plot
        print("Creating injection corner plot...")

        # Set up corner kwargs
        corner_kwargs = DEFAULT_CORNER_KWARGS.copy()
        corner_kwargs["color"] = INJECTION_COLOR

        # Prepare truth values (injection parameters) in the correct order
        truths = []
        for param in parameters:
            if injection_params.get(param) is not None:
                truths.append(injection_params[param])
            else:
                truths.append(None)  # corner will skip None values

        # Only add truths if we have at least some injection values
        if any(t is not None for t in truths):
            corner_kwargs["truths"] = truths
            corner_kwargs["truth_color"] = "black"

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

        # Create corner plot
        fig = corner.corner(posterior_samples,
                           labels=parameter_labels,
                           **corner_kwargs)

        # Save plot
        ensure_directory_exists(save_name)
        print(f"Saving injection plot to {save_name}")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

        return True

    except Exception as e:
        print(f"Failed to create injection plot: {e}")
        return False

def create_injection_comparison_plot() -> bool:
    """
    Create a comparison corner plot for the two injection analyses.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define the two injection filepaths
        injection_filepaths = [
            "/work/puecher/S231109/third_gen_runs/et_run_alignedspin/outdir/ET_gw231109_injection_alignedspin_result.json",
            "/work/puecher/S231109/third_gen_runs/et_ce_run_alignedspin/outdir/ETCE_gw231109_injectionxas_result.json"
        ]

        # Parameters to plot
        parameters = [
            "chirp_mass",
            "mass_ratio",
            "chi_eff",
            "lambda_1",
            "lambda_2",
            "lambda_tilde"
        ]

        # Labels for the two injections
        labels = [
            "ET Aligned Spin",
            "ET-CE Aligned Spin"
        ]

        # Colors for the two injections
        colors = [ORANGE, BLUE]

        # Z-orders (ET-CE on top for better visibility)
        zorders = [0, 1]

        # Common ranges for comparison
        mc = 1.306298
        eps_mc = 6e-6
        ranges = {
            "chirp_mass": (mc - eps_mc, mc + eps_mc),
            "mass_ratio": (0.80, 1.0),
            "chi_eff": (0.0290, 0.034),
            "lambda_1": (0, 750),
            "lambda_2": (0, 900),
            "lambda_tilde": (180, 400),
        }

        print("Creating injection comparison corner plot...")
        print(f"Comparing {len(injection_filepaths)} injection analyses")
        print(f"Parameters: {parameters}")

        # Load data from both injection files
        all_samples = []
        valid_labels = []
        valid_colors = []
        valid_zorders = []
        all_injection_params = []

        for i, filepath in enumerate(injection_filepaths):
            print(f"\nLoading injection data from: {filepath}")

            # Check cache first
            cache_filename = generate_injection_cache_filename(filepath, parameters)
            posterior_samples, injection_params = load_injection_data(cache_filename, parameters)

            if posterior_samples is None:
                print("Loading data from JSON file and caching...")
                posterior_samples, injection_params = load_injection_json(filepath, parameters)

                if posterior_samples is None:
                    print(f"Failed to load injection data from {filepath}")
                    continue

                # Save to cache
                save_injection_data(cache_filename, posterior_samples, injection_params, parameters)
            else:
                print("Using cached injection data")

            all_samples.append(posterior_samples)
            valid_labels.append(labels[i])
            valid_colors.append(colors[i])
            valid_zorders.append(zorders[i])
            all_injection_params.append(injection_params)

            print(f"  Loaded {len(posterior_samples)} samples")

        if not all_samples:
            print("No valid injection samples loaded!")
            return False

        # Sort by z-order for proper plotting
        sorted_data = sorted(zip(valid_zorders, all_samples, valid_labels, valid_colors, all_injection_params),
                           key=lambda x: x[0])
        valid_zorders, all_samples, valid_labels, valid_colors, all_injection_params = zip(*sorted_data)
        valid_zorders, all_samples, valid_labels, valid_colors = list(valid_zorders), list(all_samples), list(valid_labels), list(valid_colors)
        all_injection_params = list(all_injection_params)

        print("\nCreating comparison corner plot...")

        # Set up corner kwargs for first dataset
        corner_kwargs = DEFAULT_CORNER_KWARGS.copy()
        corner_kwargs["color"] = valid_colors[0]

        # Apply parameter ranges
        if ranges:
            range_list = []
            for param in parameters:
                if param in ranges:
                    range_list.append(ranges[param])
                else:
                    range_list.append(None)
            corner_kwargs["range"] = range_list

        # Prepare truth values from first injection (they should be the same)
        truths = []
        for param in parameters:
            if all_injection_params[0].get(param) is not None:
                truths.append(all_injection_params[0][param])
            else:
                truths.append(None)

        # Add truths if we have injection values
        if any(t is not None for t in truths):
            corner_kwargs["truths"] = truths
            corner_kwargs["truth_color"] = "black"

        # Create parameter labels using translation dictionary
        parameter_labels = [PARAMETER_LABELS.get(param, param) for param in parameters]

        # Create initial plot with first dataset
        fig = corner.corner(all_samples[0],
                           labels=parameter_labels,
                           **corner_kwargs)

        # Overlay additional datasets
        for i in range(1, len(all_samples)):
            corner_kwargs_overlay = corner_kwargs.copy()
            # Deep copy the range to avoid modification issues
            if "range" in corner_kwargs_overlay:
                corner_kwargs_overlay["range"] = corner_kwargs_overlay["range"].copy()
            corner_kwargs_overlay["color"] = colors[i]
            corner_kwargs_overlay["zorder"] = zorders[i]
            corner_kwargs_overlay["fig"] = fig

            corner.corner(all_samples[i],
                         labels=parameter_labels,
                         **corner_kwargs_overlay)

        # Add legend
        legend_elements = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='k', label=label)
            )

        fig.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), frameon=True)

        # Save plot
        save_name = "injection_comparison_cornerplot.pdf"
        ensure_directory_exists(save_name)
        print(f"Saving injection comparison corner plot to {save_name}")
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()

        return True

    except Exception as e:
        print(f"Failed to create injection comparison corner plot: {e}")
        return False

def main():
    """
    Main function for creating comparison corner plots.
    Edit this function to specify the directories, parameters, and settings for your comparison.
    """

    # Parameters to include in the corner plot
    parameters = [
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "lambda_1",
        "lambda_2",
        "lambda_tilde"
    ]

    # ====== COMPARISON 1: SPIN PRIOR COMPARISON ======
    print("Creating spin prior comparison corner plot...")

    # Parameter ranges
    ranges = {
        "chirp_mass": (1.3055, 1.3075),
        "mass_ratio": (0.30, 1.0),
        "chi_eff": (-0.05, 0.080),
        "lambda_1": (0, 5000),
        "lambda_2": (0, 5000),
        "lambda_tilde": (0, 5000),
    }

    # Three spin priors: chi<0.05, chi<0.025, chi<0.40
    spin_source_dirs = [
        "/work/wouters/GW231109/prod_BW_XP_s040_l5000_default/",  # chi<0.40
        "/work/wouters/GW231109/prod_BW_XP_s025_l5000_default/",  # chi<0.025
        "/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/",  # chi<0.05
    ]

    # Labels for spin comparison
    spin_labels = [
        r"$\\chi \\leq 0.40$",
        r"$\\chi \\leq 0.25$",
        r"$\\chi \\leq 0.05$",
    ]

    # Colors for spin comparison (same as before, no red)
    spin_colors = [
        ORANGE,    # "#de8f07"
        BLUE,      # "#0472b1"
        GREEN,     # "#019e72"
    ]

    # Z-orders for spin comparison
    spin_zorders = [2, 1, 0]

    # First, load and cache all runs
    print("Loading and caching spin comparison runs...")
    spin_run_keys = []
    for source_dir in spin_source_dirs:
        cache_key = load_and_cache_run(source_dir, parameters)
        if cache_key is None:
            print(f"Failed to load run from {source_dir}")
            continue
        spin_run_keys.append(cache_key)

    if len(spin_run_keys) != len(spin_source_dirs):
        print("Failed to load all spin comparison runs!")
    else:
        print(f"Successfully cached all runs with keys: {spin_run_keys}")

        # Dummy dataset: always use chi<0.05 (run index 2) for all parameters
        spin_dummy_keys = [2] * len(parameters)  # [2, 2, 2, 2, 2, 2]

        # Create the spin comparison corner plot
        spin_success = create_comparison_cornerplot(
            run_keys=spin_run_keys,
            parameters=parameters,
            labels=spin_labels,
            colors=spin_colors,
            ranges=ranges,
            zorders=spin_zorders,
            save_name="spin_comparison_cornerplot.pdf",
            overwrite=True,
            dummy_normalization_keys=spin_dummy_keys
        )

    if spin_success:
        print(f"✓ Successfully created spin comparison corner plot: spin_comparison_cornerplot.pdf")
    else:
        print("✗ Failed to create spin comparison corner plot")

    # ====== COMPARISON 2: PRIOR TYPE COMPARISON ======
    print("\nCreating prior type comparison corner plot...")
    
    # Parameter ranges
    ranges = {
        "chirp_mass": (1.3056, 1.3070),
        "mass_ratio": (0.60, 1.0),
        "chi_eff": (-0.01, 0.045),
        "lambda_1": (0, 5000),
        "lambda_2": (0, 5000),
        "lambda_tilde": (0, 5000),
    }

    # Original comparison (without the high-spin red one)
    prior_source_dirs = [
        "/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/",  # Default
        "/work/wouters/GW231109/prod_BW_XP_s005_lquniv_default/",  # Quasi universal
        "/work/wouters/GW231109/prod_BW_XP_s005_l5000_double_gaussian",  # Double Gaussian
    ]

    # Labels for prior comparison
    prior_labels = [
        "Default Prior",
        "Quasi universal relations",
        "Double Gaussian",
    ]

    # Colors for prior comparison (same as original, no red)
    prior_colors = [
        ORANGE,    # "#de8f07"
        BLUE,      # "#0472b1"
        GREEN,     # "#019e72"
    ]

    # Z-orders for prior comparison (quasi-universal on top)
    prior_zorders = [0, 2, 1]  # Default: 0, Quasi-Universal: 2 (highest), Double Gaussian: 1

    # First, load and cache all runs
    print("Loading and caching prior comparison runs...")
    prior_run_keys = []
    for source_dir in prior_source_dirs:
        cache_key = load_and_cache_run(source_dir, parameters)
        if cache_key is None:
            print(f"Failed to load run from {source_dir}")
            continue
        prior_run_keys.append(cache_key)

    if len(prior_run_keys) != len(prior_source_dirs):
        print("Failed to load all prior comparison runs!")
    else:
        print(f"Successfully cached all runs with keys: {prior_run_keys}")

        # Dummy dataset: ["Default", "Default", "Quniv", "Quniv", "Quniv"]
        # Default = run index 0, Quasi-universal = run index 1
        prior_dummy_keys = [0, 0, 1, 1, 1, 1]  # For 6 parameters: chirp_mass, mass_ratio, chi_eff, lambda_1, lambda_2, lambda_tilde

        # Create the prior comparison corner plot
        prior_success = create_comparison_cornerplot(
            run_keys=prior_run_keys,
            parameters=parameters,
            labels=prior_labels,
            colors=prior_colors,
            ranges=ranges,
            zorders=prior_zorders,
            save_name="prior_comparison_cornerplot.pdf",
            overwrite=True,
            dummy_normalization_keys=prior_dummy_keys
        )

    if prior_success:
        print(f"✓ Successfully created prior comparison corner plot: prior_comparison_cornerplot.pdf")
    else:
        print("✗ Failed to create prior comparison corner plot")

    # ====== INJECTION PLOTTING EXAMPLE ======

    mc = 1.306298
    eps_mc = 6e-6
    ranges = {
        "chirp_mass": (mc - eps_mc, mc + eps_mc),
        "mass_ratio": (0.80, 1.0),
        "chi_eff": (0.0290, 0.034),
        "lambda_1": (0, 750),
        "lambda_2": (0, 900),
        "lambda_tilde": (180, 400),
    }    

    injection_filepath = "/work/puecher/S231109/third_gen_runs/et_run_alignedspin/outdir/ET_gw231109_injection_alignedspin_result.json"
    injection_success = plot_injection(
        filepath=injection_filepath,
        ranges=ranges,
        save_name="injection_et_alignedspin.pdf",
        overwrite=True
    )
    
    if injection_success:
        print(f"✓ Successfully created injection plot")
    else:
        print("✗ Failed to create injection plot")
        
    # ====== INJECTION PLOTTING -- NOW FOR SECOND RUN ======

    mc = 1.306298
    eps_mc = 6e-6
    ranges = {
        "chirp_mass": (mc - eps_mc, mc + eps_mc),
        "mass_ratio": (0.80, 1.0),
        "chi_eff": (0.0290, 0.034),
        "lambda_1": (0, 750),
        "lambda_2": (0, 900),
        "lambda_tilde": (180, 400),
    }   
    
    ranges = None 

    injection_filepath = "/work/puecher/S231109/third_gen_runs/et_ce_run_alignedspin/outdir/ETCE_gw231109_injectionxas_result.json"
    injection_success = plot_injection(
        filepath=injection_filepath,
        ranges=ranges,
        save_name="injection_et_ce_alignedspin.pdf",
        overwrite=True
    )
    
    if injection_success:
        print(f"✓ Successfully created injection plot")
    else:
        print("✗ Failed to create injection plot")

    # ====== INJECTION COMPARISON PLOT ======

    print("\n" + "="*50)
    print("Creating injection comparison plot...")
    comparison_success = create_injection_comparison_plot()

    if comparison_success:
        print(f"✓ Successfully created injection comparison plot")
    else:
        print("✗ Failed to create injection comparison plot")

if __name__ == "__main__":
    main()