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
    fig, ax = plt.subplots(figsize=MTOV_CDF_FIGSIZE)

    # Plot CDF
    ax.plot(mass_range, cdf_eval, linewidth=MTOV_CDF_LINEWIDTH, color=MTOV_CDF_COLOR, label='MTOV CDF')
    ax.fill_between(mass_range, cdf_eval, alpha=MTOV_CDF_ALPHA, color=MTOV_CDF_COLOR)

    # Add reference lines
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Median')
    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='90% credible')

    # Format plot
    ax.set_xlabel(r'Mass [M$_\odot$]', fontsize=MTOV_CDF_AXIS_FONTSIZE)
    ax.set_ylabel('Cumulative Probability', fontsize=MTOV_CDF_AXIS_FONTSIZE)
    ax.set_title(r'Maximum Neutron Star Mass ($M_{\rm TOV}$) Distribution', fontsize=MTOV_CDF_TITLE_FONTSIZE)
    ax.set_xlim(1.0, 3.0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=MTOV_CDF_ANNOTATION_FONTSIZE)

    # Add annotation
    median_mass = mass_range[np.argmin(np.abs(cdf_eval - 0.5))]
    ax.annotate(f'Median: {median_mass:.2f} M$_\\odot$',
                xy=(median_mass, 0.5), xytext=(median_mass + 0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=MTOV_CDF_ANNOTATION_FONTSIZE, color='gray')

    # Save plot
    ensure_directory_exists(save_name)
    print(f"Saving MTOV CDF plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()

    return True

# User-configurable mass ranges (in solar masses)
MASS_1_RANGE = (1.175, 5) # Primary mass range
MASS_2_RANGE = (1.0, 2.5) # Secondary mass range

# User-configurable plot limits
MARGINAL_PLOT_MIN = 0.01  # Minimum value for 1D marginal plots, this is a small positive number for visibility and clipping

# EOS sampling configuration
USE_EOS_SAMPLING = False  # If True, use EOS sampling data for GW231109

# MTOV gradient configuration
PLOT_MTOV_GRADIENT = True  # If True, add gray gradient background showing NS-to-BH transition

# ============================================================================
# PLOTTING HYPERPARAMETERS - Centralized configuration for all plot styling
# ============================================================================

# Main m1m2 overview plot
MAIN_FIGSIZE = (10, 10)  # Figure size for main overview plot
MAIN_AXIS_LABEL_FONTSIZE = 20  # Font size for m1, m2 axis labels
MAIN_TICK_LABEL_FONTSIZE = 20  # Font size for tick labels on main m1-m2 plot
MARGINAL_AXIS_LABEL_FONTSIZE = 14  # Font size for marginal density axis labels
MARGINAL_TICK_LABEL_FONTSIZE = 12  # Font size for tick labels on marginal density plots

# MTOV gradient background
MTOV_GRADIENT_COLOR = 'gray'  # Base color for MTOV gradient
MTOV_GRADIENT_ALPHA = 0.4  # Global alpha multiplier for gradient
MTOV_GRADIENT_RESOLUTION = 500  # Grid resolution for gradient

# Mass ratio lines (q = m2/m1)
Q_LINE_COLOR = 'dimgray'  # Color for constant mass ratio lines
Q_LINE_STYLE = '--'  # Line style for mass ratio lines
Q_LINE_WIDTH = 1.5  # Line width for mass ratio lines
Q_LINE_ALPHA = 0.6  # Alpha transparency for mass ratio lines
Q_LABEL_COLOR = 'black'  # Color for q-value labels
Q_LABEL_FONTSIZE = 16  # Font size for q-value labels on mass ratio lines
Q_LABEL_ALPHA = 0.9  # Alpha transparency for q-value labels

# Marginal KDE plots
MARGINAL_KDE_LINEWIDTH = 2  # Line width for 1D marginal KDE curves
MARGINAL_KDE_ALPHA = 0.3  # Alpha for filled areas under KDE curves

# Contour plots
CONTOUR_LEVELS = [0.5, 0.9]  # Credible region levels (50% and 90%)
CONTOUR_SMOOTH = 0.4  # Smoothing parameter for 2D contours

# MTOV CDF plot
MTOV_CDF_FIGSIZE = (8, 6)  # Figure size for MTOV CDF plot
MTOV_CDF_LINEWIDTH = 2.5  # Line width for CDF curve
MTOV_CDF_COLOR = 'navy'  # Color for CDF curve
MTOV_CDF_ALPHA = 0.3  # Alpha for filled area under CDF
MTOV_CDF_TITLE_FONTSIZE = 18  # Font size for plot title
MTOV_CDF_AXIS_FONTSIZE = 16  # Font size for axis labels
MTOV_CDF_ANNOTATION_FONTSIZE = 14  # Font size for annotations

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
                   color=Q_LINE_COLOR, linestyle=Q_LINE_STYLE, alpha=Q_LINE_ALPHA,
                   linewidth=Q_LINE_WIDTH, zorder=5)

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
                       fontsize=Q_LABEL_FONTSIZE, color=Q_LABEL_COLOR, alpha=Q_LABEL_ALPHA,
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
    fig = plt.figure(figsize=MAIN_FIGSIZE)

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

    # Track maximum KDE values for setting gradient heights
    max_kde_m1 = 0
    max_kde_m2 = 0

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
                      levels=CONTOUR_LEVELS,
                      plot_datapoints=True,
                      plot_density=False,
                      plot_contours=False,
                      no_fill_contours=True,
                      fill_contours=True,
                      smooth=CONTOUR_SMOOTH)

        # Create 1D marginal KDEs
        print(f"Creating 1D marginals for {event_name}...")

        # Top panel (mass_1 marginal)
        x_m1, kde_m1 = create_kde_1d(m1_filtered, MASS_1_RANGE)
        ax_top.plot(x_m1, kde_m1, color=data['color'], linewidth=MARGINAL_KDE_LINEWIDTH,
                    label=data['label'])
        ax_top.fill_between(x_m1, kde_m1, alpha=MARGINAL_KDE_ALPHA, color=data['color'])

        # Track maximum for gradient height
        max_kde_m1 = max(max_kde_m1, np.max(kde_m1))

        # Right panel (mass_2 marginal)
        x_m2, kde_m2 = create_kde_1d(m2_filtered, MASS_2_RANGE)
        ax_right.plot(kde_m2, x_m2, color=data['color'], linewidth=MARGINAL_KDE_LINEWIDTH)
        ax_right.fill_betweenx(x_m2, kde_m2, alpha=MARGINAL_KDE_ALPHA, color=data['color'])

        # Track maximum for gradient height
        max_kde_m2 = max(max_kde_m2, np.max(kde_m2))

    # Add MTOV gradient background to marginal plots (if enabled)
    # This is done after plotting KDEs so we know the correct height ranges
    if PLOT_MTOV_GRADIENT and mtov_cdf is not None:
        from matplotlib.colors import LinearSegmentedColormap
        # Reversed: white at low masses (NS), gray at high masses (BH)
        cmap = LinearSegmentedColormap.from_list('mtov_gradient',
                                                  ['white', MTOV_GRADIENT_COLOR])

        # Add some padding for whitespace (10% extra)
        padding_factor = 1.1
        max_kde_m1_with_padding = max_kde_m1 * padding_factor
        max_kde_m2_with_padding = max_kde_m2 * padding_factor

        # Top panel (m1 marginal): horizontal gradient varying with m1
        m1_grid = np.linspace(MASS_1_RANGE[0], MASS_1_RANGE[1], MTOV_GRADIENT_RESOLUTION)
        m1_cdf_vals = mtov_cdf(m1_grid)
        # Create 2D array for imshow (same gradient repeated vertically)
        m1_gradient = np.tile(m1_cdf_vals, (50, 1))

        ax_top.imshow(m1_gradient,
                     extent=[MASS_1_RANGE[0], MASS_1_RANGE[1], MARGINAL_PLOT_MIN, max_kde_m1_with_padding],
                     origin='lower',
                     aspect='auto',
                     cmap=cmap,
                     alpha=MTOV_GRADIENT_ALPHA,
                     vmin=0, vmax=1,
                     zorder=-100)

        # Right panel (m2 marginal): vertical gradient varying with m2
        m2_grid = np.linspace(MASS_2_RANGE[0], MASS_2_RANGE[1], MTOV_GRADIENT_RESOLUTION)
        m2_cdf_vals = mtov_cdf(m2_grid)
        # Create 2D array for imshow (same gradient repeated horizontally)
        m2_gradient = np.tile(m2_cdf_vals.reshape(-1, 1), (1, 50))

        ax_right.imshow(m2_gradient,
                       extent=[MARGINAL_PLOT_MIN, max_kde_m2_with_padding, MASS_2_RANGE[0], MASS_2_RANGE[1]],
                       origin='lower',
                       aspect='auto',
                       cmap=cmap,
                       alpha=MTOV_GRADIENT_ALPHA,
                       vmin=0, vmax=1,
                       zorder=-100)

        # Set the y-limits to match the gradient heights
        ax_top.set_ylim(bottom=MARGINAL_PLOT_MIN, top=max_kde_m1_with_padding)
        ax_right.set_xlim(left=MARGINAL_PLOT_MIN, right=max_kde_m2_with_padding)

    # Plot equal mass ratio lines
    plot_equal_mass_ratio_lines(ax_main, MASS_1_RANGE, MASS_2_RANGE)

    # Format main plot
    ax_main.set_xlabel(r'$m_1$ [M$_\odot$]', fontsize=MAIN_AXIS_LABEL_FONTSIZE)
    ax_main.set_ylabel(r'$m_2$ [M$_\odot$]', fontsize=MAIN_AXIS_LABEL_FONTSIZE)
    ax_main.set_xlim(MASS_1_RANGE)
    ax_main.set_ylim(MASS_2_RANGE)
    ax_main.tick_params(labelsize=MAIN_TICK_LABEL_FONTSIZE)

    # Add legend to main plot
    legend_elements = []
    for event_name, data in event_data.items():
        legend_elements.append(
            mpatches.Patch(facecolor=data['color'], edgecolor='none', label=data['label'])
        )
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=legend_fs,
                   frameon=True, facecolor='white', framealpha=1, edgecolor='black')

    # Format marginal plots
    ax_top.set_ylabel('Density', fontsize=MARGINAL_AXIS_LABEL_FONTSIZE)
    ax_top.tick_params(labelbottom=False, labelsize=MARGINAL_TICK_LABEL_FONTSIZE)
    # Set y-limits for top panel (if gradient not enabled, set based on KDE max)
    if not PLOT_MTOV_GRADIENT or mtov_cdf is None:
        padding_factor = 1.1
        ax_top.set_ylim(bottom=MARGINAL_PLOT_MIN, top=max_kde_m1 * padding_factor)

    ax_right.set_xlabel('Density', fontsize=MARGINAL_AXIS_LABEL_FONTSIZE)
    ax_right.tick_params(labelleft=False, labelsize=MARGINAL_TICK_LABEL_FONTSIZE)
    # Set x-limits for right panel (if gradient not enabled, set based on KDE max)
    if not PLOT_MTOV_GRADIENT or mtov_cdf is None:
        padding_factor = 1.1
        ax_right.set_xlim(left=MARGINAL_PLOT_MIN, right=max_kde_m2 * padding_factor)

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