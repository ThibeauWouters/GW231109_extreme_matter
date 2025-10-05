#!/usr/bin/env python3
"""
Mass_1 vs Mass_2 overview plot for multiple gravitational wave events.
Creates a corner-style plot showing the component mass distributions
from posterior samples of different GW events.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

def ensure_directory_exists(filepath: str):
    """
    Ensure the directory for a filepath exists.

    Args:
        filepath (str): Full file path
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_mtov_cdf(filepath: str = './data/GW170817_jester_constraints.npz') -> interp1d:
    """
    Load MTOV distribution from EOS constraints and create interpolated CDF.

    Args:
        filepath (str): Path to the npz file containing masses_EOS

    Returns:
        interp1d: Interpolated CDF function that maps mass -> probability
    """
    print(f"Loading MTOV data from {filepath}...")
    data = np.load(filepath)

    # Extract MTOV values (maximum mass for each EOS)
    mtov_samples = np.max(data['masses_EOS'], axis=1)

    print(f"  Number of EOS samples: {len(mtov_samples)}")
    print(f"  MTOV range: {mtov_samples.min():.3f} - {mtov_samples.max():.3f} M_sun")
    print(f"  MTOV median: {np.median(mtov_samples):.3f} M_sun")

    # Sort samples and create empirical CDF
    mtov_sorted = np.sort(mtov_samples)
    cdf_values = np.arange(1, len(mtov_sorted) + 1) / len(mtov_sorted)

    # Create interpolated CDF function
    # Extrapolate: below minimum MTOV -> 0, above maximum MTOV -> 1
    cdf_func = interp1d(mtov_sorted, cdf_values,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(0.0, 1.0))

    return cdf_func

def plot_mtov_cdf(cdf_func: interp1d, save_name: str = None) -> bool:
    """
    Plot and save the MTOV CDF visualization.

    Args:
        cdf_func (interp1d): Interpolated CDF function
        save_name (str): Output filename (optional)

    Returns:
        bool: True if successful
    """
    if save_name is None:
        save_name = "./figures/GW_PE/mtov_cdf.pdf"

    print(f"Creating MTOV CDF plot...")

    # Create mass range for evaluation
    mass_range = np.linspace(1.0, 3.0, 1000)
    cdf_eval = cdf_func(mass_range)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot CDF
    ax.plot(mass_range, cdf_eval, linewidth=2.5, color='navy', label='MTOV CDF')
    ax.fill_between(mass_range, cdf_eval, alpha=0.3, color='navy')

    # Add reference lines
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Median')
    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='90% credible')

    # Format plot
    ax.set_xlabel(r'Mass [M$_\odot$]', fontsize=16)
    ax.set_ylabel('Cumulative Probability', fontsize=16)
    ax.set_title(r'Maximum Neutron Star Mass ($M_{\rm TOV}$) Distribution', fontsize=18)
    ax.set_xlim(1.0, 3.0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)

    # Add annotation
    median_mass = mass_range[np.argmin(np.abs(cdf_eval - 0.5))]
    ax.annotate(f'Median: {median_mass:.2f} M$_\\odot$',
                xy=(median_mass, 0.5), xytext=(median_mass + 0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=14, color='gray')

    # Save plot
    ensure_directory_exists(save_name)
    print(f"Saving MTOV CDF plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()

    return True

# User-configurable mass ranges (in solar masses)
MASS_1_RANGE = (1.1, 5) # Primary mass range
MASS_2_RANGE = (1.0, 2.5) # Secondary mass range

# User-configurable plot limits
MARGINAL_PLOT_MIN = 0.01  # Minimum value for 1D marginal plots, this is a small positive number for visibility and clipping
Q_LABEL_FONTSIZE = 16  # Font size for q-value labels on mass ratio lines

# EOS sampling configuration
USE_EOS_SAMPLING = False  # If True, use EOS sampling data for GW231109

# MTOV gradient configuration
PLOT_MTOV_GRADIENT = True  # If True, add gray gradient background showing NS-to-BH transition

# If running on Mac, so we can use TeX (not on Jarvis), change some rc params
cwd = os.getcwd()
if "Woute029" in cwd:
    print(f"Updating plotting parameters for TeX")
    fs = 20
    ticks_fs = 20
    legend_fs = 16
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

# Event information
EVENTS = {
    'GW170817': {
        'file': '../posteriors/data/GW170817.npz',
        'color': '#fab6c7',  # Light red/pink
        'label': 'GW170817'
    },
    'GW231109': {
        'file': '../posteriors/data/prod_BW_XP_s005_leos_default.npz' if USE_EOS_SAMPLING else '../posteriors/data/prod_BW_XP_s005_l5000_default.npz',
        'color': 'orange',  # Orange
        'label': 'GW231109'
    },
    'GW190425': {
        'file': '../posteriors/data/GW190425.npz',
        'color': '#c6dd81',  # Green
        'label': 'GW190425'
    },
    'GW230529': {
        'file': '../posteriors/data/GW230529.npz',
        'color': '#00707e',  # Teal
        'label': 'GW230529'
    }
}

def load_event_masses(filepath: str) -> tuple:
    """
    Load mass_1_source and mass_2_source from a .npz file.

    Args:
        filepath (str): Path to the .npz file

    Returns:
        tuple: (mass_1_source, mass_2_source) arrays
    """
    data = np.load(filepath)

    mass_1 = data['mass_1_source']
    mass_2 = data['mass_2_source']

    print(f"Loaded {len(mass_1)} samples from {filepath}")
    print(f"  Mass 1 range: {mass_1.min():.2f} - {mass_1.max():.2f} M_sun")
    print(f"  Mass 2 range: {mass_2.min():.2f} - {mass_2.max():.2f} M_sun")

    return mass_1, mass_2

def create_kde_1d(samples: np.ndarray, x_range: tuple, n_points: int = 1_000) -> tuple:
    """
    Create 1D KDE for marginal distributions.

    Args:
        samples (np.ndarray): 1D array of samples
        x_range (tuple): (min, max) range for evaluation
        n_points (int): Number of evaluation points

    Returns:
        tuple: (x_eval, kde_eval) arrays
    """
    # Filter samples to be within range
    samples_filtered = samples[(samples >= x_range[0]) & (samples <= x_range[1])]

    # Create KDE
    kde = gaussian_kde(samples_filtered)

    # Evaluate KDE
    x_eval = np.linspace(x_range[0], x_range[1], n_points)
    kde_eval = kde(x_eval)

    return x_eval, kde_eval

def plot_equal_mass_ratio_lines(ax, mass_1_range: tuple, mass_2_range: tuple):
    """
    Plot lines of constant mass ratio q = m2/m1.

    Args:
        ax: Matplotlib axis object
        mass_1_range (tuple): Range for mass_1 axis
        mass_2_range (tuple): Range for mass_2 axis
    """
    ratios = [1, 1/2, 1/3]

    for q in ratios:
        # Line: m2 = q * m1
        m1_line = np.linspace(mass_1_range[0], mass_1_range[1], 100)
        m2_line = q * m1_line

        # Only plot where m2 is within range
        valid_mask = (m2_line >= mass_2_range[0]) & (m2_line <= mass_2_range[1])

        if np.any(valid_mask):
            ax.plot(m1_line[valid_mask], m2_line[valid_mask],
                   color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=5)

            # Add rotated label along the line
            if np.any(valid_mask):
                # Find a good position for the label (position depends on q)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 1:
                    # Different positions for different q values
                    if q == 1/2:
                        label_position = 0.55  # 55% for q=1/2
                    else:
                        label_position = 0.75  # 75% along for others

                    label_idx = valid_indices[int(len(valid_indices) * label_position)]
                    label_x = m1_line[label_idx]
                    label_y = m2_line[label_idx]
                else:
                    label_x = m1_line[valid_mask][0]
                    label_y = m2_line[valid_mask][0]

                # Create label text
                if q == 1:
                    label_text = '$q = 1$'
                elif q == 1/2:
                    label_text = '$q = 1/2$'
                elif q == 1/3:
                    label_text = '$q = 1/3$'
                else:
                    label_text = f'$q = {q:.2f}$'

                # Calculate rotation angle using transform to get visual slope
                # Get the slope in data coordinates and convert to screen angle
                data_slope = q  # m2/m1

                # Convert slope to angle, accounting for axis scaling
                x_scale = (mass_1_range[1] - mass_1_range[0])
                y_scale = (mass_2_range[1] - mass_2_range[0])
                visual_slope = data_slope * (x_scale / y_scale)

                angle_deg = np.degrees(np.arctan(visual_slope))

                # Add rotated text with more spacing from the line
                # Offset the text slightly away from the line
                offset_distance = 0.05  # Small offset perpendicular to the line
                angle_rad = np.radians(angle_deg)
                offset_x = -offset_distance * np.sin(angle_rad)
                offset_y = offset_distance * np.cos(angle_rad)

                ax.text(label_x + offset_x, label_y + offset_y, label_text,
                       fontsize=Q_LABEL_FONTSIZE, color='gray', alpha=0.9,
                       rotation=angle_deg,
                       rotation_mode='anchor',
                       ha='center', va='bottom', zorder=5)

def create_m1m2_overview_plot(save_name: str = None) -> bool:
    """
    Create the main mass_1 vs mass_2 overview plot.

    Args:
        save_name (str): Output filename (optional, will be auto-generated if None)

    Returns:
        bool: True if successful, False otherwise
    """
    # Generate save name if not provided
    if save_name is None:
        base_name = "m1m2_overview"
        if USE_EOS_SAMPLING:
            base_name += "_eos_sampling"
        save_name = f"./figures/GW_PE/{base_name}.pdf"

    print("Creating mass_1 vs mass_2 overview plot...")
    print(f"EOS sampling mode: {USE_EOS_SAMPLING}")
    print(f"MTOV gradient: {PLOT_MTOV_GRADIENT}")
    print(f"Output file: {save_name}")

    # Load MTOV CDF for gradient background (if enabled)
    if PLOT_MTOV_GRADIENT:
        print("\n" + "="*60)
        mtov_cdf = load_mtov_cdf()
        plot_mtov_cdf(mtov_cdf)
        print("="*60 + "\n")
    else:
        mtov_cdf = None

    # Load data for all events
    event_data = {}
    for event_name, event_info in EVENTS.items():
        print(f"\nLoading data for {event_name}...")
        mass_1, mass_2 = load_event_masses(event_info['file'])

        # Gather in a dict for simplicity
        event_data[event_name] = {
            'mass_1': mass_1,
            'mass_2': mass_2,
            'color': event_info['color'],
            'label': event_info['label']
        }

    print(f"\nSuccessfully loaded data for {len(event_data)} events")

    # Create figure with specific layout
    fig = plt.figure(figsize=(10, 10))

    # Define grid layout: main plot with marginal histograms
    gs = fig.add_gridspec(3, 3,
                            width_ratios=[1, 4, 1],
                            height_ratios=[1, 4, 1],
                            hspace=0.05, wspace=0.05)

    # Main 2D plot
    ax_main = fig.add_subplot(gs[1, 1])

    # Marginal plots
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)

    # Add MTOV gradient background (if enabled)
    if PLOT_MTOV_GRADIENT and mtov_cdf is not None:
        # Create mesh grid for the gradient
        m1_grid = np.linspace(MASS_1_RANGE[0], MASS_1_RANGE[1], 500)
        m2_grid = np.linspace(MASS_2_RANGE[0], MASS_2_RANGE[1], 500)
        M1, M2 = np.meshgrid(m1_grid, m2_grid)

        # Evaluate CDF at each m1 position (use m1 since it's the primary mass)
        # CDF = probability that MTOV < m1
        # Low masses (< MTOV): CDF ≈ 0 → more gray (clearly NS region)
        # High masses (> MTOV): CDF ≈ 1 → more white (likely BH region)
        # So we want: gray at low masses, white at high masses
        # Use reversed gray colormap: low values (CDF=0) → dark, high values (CDF=1) → light
        alpha_grid = mtov_cdf(M1)

        # Create gray background with varying intensity based on MTOV CDF
        # Use reversed gray cmap so that high CDF (likely BH) = white, low CDF (clearly NS) = gray
        ax_main.imshow(alpha_grid,
                       extent=[MASS_1_RANGE[0], MASS_1_RANGE[1], MASS_2_RANGE[0], MASS_2_RANGE[1]],
                       origin='lower',
                       aspect='auto',
                       cmap='gray_r',  # Reversed: dark at low values, light at high values
                       alpha=0.4,  # Global alpha multiplier to keep it subtle
                       vmin=0, vmax=1,
                       zorder=-100)  # Very low zorder to stay behind everything

    # Plot 2D contours for each event
    for event_name, data in event_data.items():
        print(f"Creating 2D contours for {event_name}...")

        # Filter samples to be within specified ranges
        mask = ((data['mass_1'] >= MASS_1_RANGE[0]) & (data['mass_1'] <= MASS_1_RANGE[1]) &
                (data['mass_2'] >= MASS_2_RANGE[0]) & (data['mass_2'] <= MASS_2_RANGE[1]))

        if not np.any(mask):
            print(f"No samples for {event_name} within specified mass ranges")
            continue

        m1_filtered = data['mass_1'][mask]
        m2_filtered = data['mass_2'][mask]

        print(f"  Using {len(m1_filtered)} samples within mass ranges")

        # Create 2D histogram/contours
        corner.hist2d(m1_filtered, m2_filtered,
                      ax=ax_main,
                      color=data['color'],
                      levels=[0.5, 0.9],  # 50% and 90% credible regions
                      plot_datapoints=True,
                      plot_density=False,
                      plot_contours=False,
                      no_fill_contours=True,
                      fill_contours=True,
                      )

        # Create 1D marginal KDEs
        print(f"Creating 1D marginals for {event_name}...")

        # Top panel (mass_1 marginal)
        x_m1, kde_m1 = create_kde_1d(m1_filtered, MASS_1_RANGE)
        ax_top.plot(x_m1, kde_m1, color=data['color'], linewidth=2,
                    label=data['label'])
        ax_top.fill_between(x_m1, kde_m1, alpha=0.3, color=data['color'])

        # Right panel (mass_2 marginal)
        x_m2, kde_m2 = create_kde_1d(m2_filtered, MASS_2_RANGE)
        ax_right.plot(kde_m2, x_m2, color=data['color'], linewidth=2)
        ax_right.fill_betweenx(x_m2, kde_m2, alpha=0.3, color=data['color'])

    # Plot equal mass ratio lines
    plot_equal_mass_ratio_lines(ax_main, MASS_1_RANGE, MASS_2_RANGE)

    # Format main plot
    ax_main.set_xlabel(r'$m_1$ [M$_\odot$]', fontsize=16)
    ax_main.set_ylabel(r'$m_2$ [M$_\odot$]', fontsize=16)
    ax_main.set_xlim(MASS_1_RANGE)
    ax_main.set_ylim(MASS_2_RANGE)

    # Add legend to main plot
    legend_elements = []
    for event_name, data in event_data.items():
        legend_elements.append(
            mpatches.Patch(facecolor=data['color'], edgecolor='none', label=data['label'])
        )
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=legend_fs)

    # Format marginal plots
    ax_top.set_ylabel('Density', fontsize=14)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylim(bottom=MARGINAL_PLOT_MIN)

    ax_right.set_xlabel('Density', fontsize=14)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlim(left=MARGINAL_PLOT_MIN)

    # Remove ticks from marginal plots where appropriate
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Save plot
    ensure_directory_exists(save_name)
    print(f"Saving overview plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()

    return True

def main():
    """
    Main function to create the mass_1 vs mass_2 overview plot.
    """
    print("=" * 60)
    print("Creating Mass_1 vs Mass_2 Overview Plot")
    print("=" * 60)
    print(f"Mass 1 range: {MASS_1_RANGE[0]} - {MASS_1_RANGE[1]} M_sun")
    print(f"Mass 2 range: {MASS_2_RANGE[0]} - {MASS_2_RANGE[1]} M_sun")
    print(f"Events to include: {list(EVENTS.keys())}")

    success = create_m1m2_overview_plot()

    if success:
        print("\n✓ Successfully created mass overview plot: ./figures/GW_PE/m1m2_overview.pdf")
    else:
        print("\n✗ Failed to create mass overview plot")

if __name__ == "__main__":
    main()