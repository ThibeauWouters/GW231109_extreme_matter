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

# If running on Mac, use TeX for better typography
if "Woute029" in os.getcwd():
    print(f"Updating plotting parameters for TeX")
    fs = 18
    ticks_fs = 20
    label_fs = 24
    legend_fs = 24
    labelpad = 18
    rc_params = {
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "xtick.labelsize": ticks_fs,
        "ytick.labelsize": ticks_fs,
        "axes.labelsize": label_fs,
        "legend.fontsize": legend_fs,
        "legend.title_fontsize": legend_fs,
        "figure.titlesize": fs
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
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16
    }
    plt.rcParams.update(params)

# Corner plot settings (matching final_GW_PE_figures.py)
default_corner_kwargs = dict(
    bins=40,
    smooth=0.9,
    show_titles=False,
    title_fmt=".3f",
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    levels=[0.68, 0.95],  # 68% and 95% credible regions
    labelpad=0.10,
    max_n_ticks=3,
    min_n_ticks=2,
    save=False
)

# Colors
GW170817_COLOR = "purple"  # Purple? TODO: decide on this
GW231109_ET_COLOR = "#de8f05"  # Orange/gold
GW231109_ET_CE_COLOR = "#d45d01"  # Darker orange
TRUTH_COLOR = "black"

# Injection truth marker settings (for consistency between plot and legend)
INJECTION_MARKER_EDGECOLOR = "white"  # Edge color for the injection star marker
INJECTION_MARKER_EDGEWIDTH = 1.5     # Edge width for the injection star marker
INJECTION_MARKER_SIZE = 400           # Marker size (s parameter for scatter)


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


def load_GW170817_posterior():
    """
    Load the GW170817 posterior samples for chirp mass and lambda tilde.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde) arrays
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

    print(f"Loaded {len(chirp_mass_source)} GW170817 samples")
    return chirp_mass_source, lambda_tilde


def load_GW231109_ET_posterior():
    """
    Load the GW231109 ET injection posterior samples.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde) arrays
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

    # Compute chirp mass from source frame masses
    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)

    print(f"Loaded {len(chirp_mass_source)} GW231109 ET samples")
    return chirp_mass_source, lambda_tilde


def load_GW231109_ET_CE_posterior():
    """
    Load the GW231109 ET+CE injection posterior samples.

    Returns:
        tuple: (chirp_mass_source, lambda_tilde) arrays
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

    # Compute chirp mass from source frame masses
    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)

    print(f"Loaded {len(chirp_mass_source)} GW231109 ET+CE samples")
    return chirp_mass_source, lambda_tilde


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
        chirp_mass_GW170817, lambda_tilde_GW170817 = load_GW170817_posterior()

        print("\nLoading GW231109 ET posterior...")
        chirp_mass_ET, lambda_tilde_ET = load_GW231109_ET_posterior()

        print("\nLoading GW231109 ET+CE posterior...")
        chirp_mass_ET_CE, lambda_tilde_ET_CE = load_GW231109_ET_CE_posterior()

        # Get injection truth if requested
        injection_params = None
        if show_injection_truth:
            print("\nLoading injection parameters...")
            injection_params = get_injection_parameters()
            print(f"Injection chirp mass (source): {injection_params['chirp_mass_source']:.6f} Msun")
            print(f"Injection lambda tilde: {injection_params['lambda_tilde']:.2f}")

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

        # Create figure
        print("\nCreating plot...")
        fig = plt.figure(figsize=(12, 8))

        # Plot EOS curves in background
        print(f"Plotting {len(indices)} EOS curves in background...")
        mchirp_min, mchirp_max = xlim
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

        # Plot the injection EOS curve prominently
        print("Plotting injection EOS curve...")
        inj_mchirp, inj_lambda_tilde = load_injection_eos_curve(xlim)
        if inj_mchirp is not None:
            plt.plot(inj_mchirp, inj_lambda_tilde, color=TRUTH_COLOR, alpha=0.8,
                    lw=3.0, zorder=1000, linestyle='--', label='Injection EOS')

        # Plot GW170817 posterior first (lower z-order)
        print("Plotting GW170817 posterior contours...")
        corner_kwargs_gw170817 = default_corner_kwargs.copy()
        corner_kwargs_gw170817['color'] = GW170817_COLOR
        corner_kwargs_gw170817['zorder'] = 2000  # Above EOS curves
        corner.hist2d(
            chirp_mass_GW170817,
            lambda_tilde_GW170817,
            fig=fig,
            **corner_kwargs_gw170817
        )

        # Plot GW231109 ET posterior on top (higher z-order)
        print("Plotting GW231109 ET posterior contours...")
        corner_kwargs_et = default_corner_kwargs.copy()
        corner_kwargs_et['color'] = GW231109_ET_COLOR
        corner_kwargs_et['zorder'] = 3000  # Above GW170817
        corner.hist2d(
            chirp_mass_ET,
            lambda_tilde_ET,
            fig=fig,
            **corner_kwargs_et
        )

        # Plot GW231109 ET+CE posterior on top (highest z-order)
        print("Plotting GW231109 ET+CE posterior contours...")
        corner_kwargs_et_ce = default_corner_kwargs.copy()
        corner_kwargs_et_ce['color'] = GW231109_ET_CE_COLOR
        corner_kwargs_et_ce['zorder'] = 4000  # Above ET, highest posterior
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
                zorder=100000,
                label='Injection',
                edgecolors=INJECTION_MARKER_EDGECOLOR,
                linewidths=INJECTION_MARKER_EDGEWIDTH
            )

        # Set labels and limits
        plt.xlabel(r"$\mathcal{M}_c^{\rm{src}}$ [M$_\odot$]")
        plt.ylabel(r"$\tilde{\Lambda}$")
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817 (2G)'),
            mpatches.Patch(facecolor=GW231109_ET_COLOR, edgecolor='k', label='GW231109 (ET)'),
            mpatches.Patch(facecolor=GW231109_ET_CE_COLOR, edgecolor='k', label='GW231109 (ET+CE)')
        ]

        if inj_mchirp is not None:
            legend_elements.append(
                plt.Line2D([0], [0], color=TRUTH_COLOR, lw=3, linestyle='--',
                          label='Truth')
            )

        # if show_injection_truth:
        #     legend_elements.append(
        #         plt.Line2D([0], [0], marker='*', color='w',
        #                   markerfacecolor=TRUTH_COLOR, markersize=15,
        #                   label='Injection',
        #                   markeredgecolor=INJECTION_MARKER_EDGECOLOR,
        #                   markeredgewidth=INJECTION_MARKER_EDGEWIDTH)
        #     )

        plt.legend(handles=legend_elements, loc='upper right', frameon=True)

        # Add colorbar for EOS posterior probability
        ax = fig.gca()
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"EOS Posterior Probability ({eos_name})", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

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
        xlim=(1.15, 1.45),
        ylim=(100, 1000)
    )

    if not success:
        print("Failed to create the plot. Please check the error messages above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
