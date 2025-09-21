"""Money plots script for jester inference results on Snellius.

This script generates key figures from multiple jester inference output directories:
- Histograms of MTOV, R14, and other EOS parameters
- Mass-radius plots with posterior probability coloring
- Pressure-density (EOS) plots

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

def make_parameter_histograms(data_dict: dict, outdir_name: str, color: str = 'blue', save_suffix: str = ""):
    """Create histograms for key EOS parameters.

    Args:
        data_dict: Dictionary containing EOS data
        outdir_name: Name of the source directory (for filename)
        color: Color to use for plotting
        save_suffix: Optional suffix for filename
    """
    print(f"Creating parameter histograms for {outdir_name}...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    m, r = data_dict['masses'], data_dict['radii']
    n, p = data_dict['densities'], data_dict['pressures']

    # Calculate derived parameters
    MTOV_list = np.array([np.max(mass) for mass in m])
    R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

    parameters = {
        'MTOV': {'values': MTOV_list, 'range': (1.75, 2.75), 'xlabel': r"$M_{\rm{TOV}}$ [$M_{\odot}$]"},
        'R14': {'values': R14_list, 'range': (10.0, 16.0), 'xlabel': r"$R_{1.4}$ [km]"},
        'p3nsat': {'values': p3nsat_list, 'range': (0.1, 200.0), 'xlabel': r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]"}
    }

    for param_name, param_data in parameters.items():
        plt.figure(figsize=figsize_horizontal)

        # Create posterior KDE
        kde = gaussian_kde(param_data['values'])
        x = np.linspace(param_data['range'][0], param_data['range'][1], 1000)
        y = kde(x)

        plt.plot(x, y, color=color, lw=3.0, label='Posterior')
        plt.fill_between(x, y, alpha=0.3, color=color)

        # Add credible interval information
        low, med, high = report_credible_interval(param_data['values'])

        plt.xlabel(param_data['xlabel'])
        plt.ylabel('Density')
        plt.xlim(param_data['range'])
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.title(f'{param_name}: {med:.2f} -{low:.2f} +{high:.2f}')

        # Save to figures directory with directory name in filename
        dir_basename = os.path.basename(outdir_name.rstrip('/'))
        save_name = os.path.join("./figures", f"{dir_basename}_{param_name}_histogram{save_suffix}.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"  {param_name} histogram saved to {save_name}")

def make_mass_radius_plot(data_dict: dict, outdir_name: str, colormap: str = 'crest', save_suffix: str = ""):
    """Create mass-radius plot with posterior probability coloring.

    Args:
        data_dict: Dictionary containing EOS data
        outdir_name: Name of the source directory (for filename)
        colormap: Seaborn colormap name to use for probability coloring
        save_suffix: Optional suffix for filename
    """
    print(f"Creating mass-radius plot for {outdir_name}...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    plt.figure(figsize=(6, 12))
    m_min, m_max = 0.75, 3.5
    r_min, r_max = 6.0, 18.0

    # Plot posterior with probability coloring
    m, r, l = data_dict['masses'], data_dict['radii'], data_dict['lambdas']
    log_prob = data_dict['log_prob']
    nb_samples = np.shape(m)[0]
    print(f"  Number of samples: {nb_samples}")

    # Normalize probabilities for coloring
    log_prob = np.exp(log_prob)
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette(colormap, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    bad_counter = 0
    for i in tqdm.tqdm(range(len(log_prob))):
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

        # Get color based on probability
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)

        plt.plot(r[i], m[i],
                color=color,
                alpha=1.0,
                rasterized=True,
                zorder=1e10 + normalized_value)

    print(f"  Excluded {bad_counter} invalid samples")

    # Styling
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(r_min, r_max)
    plt.ylim(m_min, m_max)

    # Add colorbar
    fig = plt.gcf()
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Normalized posterior probability", fontsize=16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position('top')

    # Save figure to figures directory
    dir_basename = os.path.basename(outdir_name.rstrip('/'))
    save_name = os.path.join("./figures", f"{dir_basename}_mass_radius_plot{save_suffix}.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Mass-radius plot saved to {save_name}")

def make_pressure_density_plot(data_dict: dict, outdir_name: str, colormap: str = 'crest', save_suffix: str = ""):
    """Create equation of state plot (pressure vs density).

    Args:
        data_dict: Dictionary containing EOS data
        outdir_name: Name of the source directory (for filename)
        colormap: Seaborn colormap name to use for probability coloring
        save_suffix: Optional suffix for filename
    """
    print(f"Creating pressure-density plot for {outdir_name}...")

    # Ensure figures directory exists
    os.makedirs("./figures", exist_ok=True)

    plt.figure(figsize=(11, 6))

    # Plot posterior with probability coloring
    m, r, l = data_dict['masses'], data_dict['radii'], data_dict['lambdas']
    n, p = data_dict['densities'], data_dict['pressures']
    log_prob = data_dict['log_prob']

    # Normalize probabilities for coloring
    log_prob = np.exp(log_prob)
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette(colormap, as_cmap=True)

    bad_counter = 0
    for i in tqdm.tqdm(range(len(log_prob))):
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

        # Get color and plot
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)

        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(n[i][mask], p[i][mask],
                color=color,
                alpha=1.0,
                rasterized=True,
                zorder=1e10 + normalized_value)

    print(f"  Excluded {bad_counter} invalid samples")

    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.yscale('log')
    plt.xlim(0.5, 6.0)

    # Save figure to figures directory
    dir_basename = os.path.basename(outdir_name.rstrip('/'))
    save_name = os.path.join("./figures", f"{dir_basename}_pressure_density_plot{save_suffix}.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    print(f"  Pressure-density plot saved to {save_name}")

def process_directory(outdir: str, color: str = 'blue', colormap: str = 'crest', save_suffix: str = ""):
    """Process a single output directory and generate all plots."""
    print(f"\n{'='*60}")
    print(f"Processing directory: {outdir}")
    print(f"{'='*60}")

    # Check if directory exists
    if not os.path.exists(outdir):
        print(f"Warning: Directory {outdir} does not exist. Skipping...")
        return False

    # Load data
    try:
        data = load_eos_data(outdir)
        print("Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return False

    # Create all plots
    try:
        make_parameter_histograms(data, outdir, color, save_suffix)
        make_mass_radius_plot(data, outdir, colormap, save_suffix)
        make_pressure_density_plot(data, outdir, colormap, save_suffix)
        print(f"All plots generated successfully for {outdir}")
        return True
    except Exception as e:
        print(f"Error generating plots: {e}")
        return False

def main():
    """Main function with hard-coded directories and colors."""

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

    # Colors for histogram plots (one per directory)
    # Options: 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'
    colors = [
        'blue',
        'red',
        'green',
        # Add more colors as needed (should match number of directories)
    ]

    # Colormaps for mass-radius and pressure-density plots (one per directory)
    # Options: 'crest', 'viridis', 'plasma', 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples'
    colormaps = [
        'crest',
        'Reds',
        'Greens',
        # Add more colormaps as needed (should match number of directories)
    ]

    # Optional suffix for all output files
    save_suffix = ""

    # =======================================================================
    # END USER CONFIGURATION
    # =======================================================================

    print("Money Plots Generator for Jester Inference Results")
    print("=" * 60)
    print(f"Processing {len(directories)} directories...")

    # Ensure we have enough colors and colormaps
    if len(colors) < len(directories):
        print(f"Warning: Only {len(colors)} colors provided for {len(directories)} directories.")
        print("Extending with default colors...")
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        while len(colors) < len(directories):
            colors.append(default_colors[len(colors) % len(default_colors)])

    if len(colormaps) < len(directories):
        print(f"Warning: Only {len(colormaps)} colormaps provided for {len(directories)} directories.")
        print("Extending with default colormaps...")
        default_colormaps = ['crest', 'viridis', 'plasma', 'Blues', 'Reds']
        while len(colormaps) < len(directories):
            colormaps.append(default_colormaps[len(colormaps) % len(default_colormaps)])

    success_count = 0
    for i, (outdir, color, colormap) in enumerate(zip(directories, colors, colormaps), 1):
        print(f"\n[{i}/{len(directories)}] Using color='{color}', colormap='{colormap}'")
        if process_directory(outdir, color, colormap, save_suffix):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: Successfully processed {success_count}/{len(directories)} directories")
    if success_count < len(directories):
        print("Some directories failed - check output above for details")
    print(f"{'='*60}")
    print("All figures saved to ./figures/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()