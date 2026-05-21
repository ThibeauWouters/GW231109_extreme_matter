import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 18,
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})
from astropy.time import Time

# ── Layout tuning — large (9-panel) figure ───────────────────────────────────
# Fraction of figure height reserved above the panels for the legend.
# Increase to push panels down and give more space to the legend.
LARGE_LAYOUT = dict(
    figsize              = (7.5, 13),  # (width, height) in inches
    legend_top_margin    = 0.96,  # subplots top edge in figure coords (0–1)
    legend_bbox_y        = 1.01,  # legend anchor y (lower value → closer to panels)
    legend_fontsize      = 18,
    ylabel_use_figtext   = True,  # True → fig.text(); False → axes[0].set_ylabel()
    ylabel_x             = 0.01,  # x pos of rotated fig.text label (figtext mode only)
    ylabel_fontsize      = 24,
    ylabel_ypos_frac     = 0.55,  # vertical anchor for figtext (0=bottom, 1=top)
    xlabel_fontsize      = 15,    # 'Time since merger' label
    tick_labelsize       = 13,    # axis tick labels
    panel_label_fontsize = 14,    # band name text inside each panel
    wspace               = 0.05,  # horizontal gap between panels
    hspace               = 0.10,  # vertical gap between panels
    sharey               = False,
)

# ── Layout tuning — small (2-panel) figure ───────────────────────────────────
SMALL_LAYOUT = dict(
    figsize              = (8, 4),
    legend_top_margin    = 0.76,   # leave headroom for two-row legend inside the figure
    legend_bbox_y        = 0.99,   # place legend's upper edge near top of figure
    legend_fontsize      = 13,     # smaller font so 3 columns fit within the figure width
    ylabel_use_figtext   = False, # uses axes[0].set_ylabel() instead
    ylabel_x             = 0.01,  # unused in set_ylabel mode
    ylabel_fontsize      = 20,
    ylabel_ypos_frac     = 0.50,  # unused in set_ylabel mode
    xlabel_fontsize      = 18,
    tick_labelsize       = 14,
    panel_label_fontsize = 18,
    wspace               = 0.05,
    hspace               = 0.075,
    sharey               = True,
)
# ─────────────────────────────────────────────────────────────────────────────

def read_observations_file(
    filename, merger_time='2017-08-17T12:41:04'
):
    """
    Read observation data from ASCII file.
    Expected format:
    2017-08-18T00:00:00.000 ps1::g 17.41000 0.02000
    2017-08-18T00:00:00.000 ps1::r 17.56000 0.04000
    ...
    
    Parameters:
    -----------
    filename : str
        Path to observations file
    merger_time : str
        Time of merger in ISO format (YYYY-MM-DDTHH:MM:SS)
    distance_mpc : float
        Distance to source in Mpc for converting apparent to absolute magnitude
    
    Returns a dictionary with filter names as keys and DataFrames with 
    'time' and 'magnitude' columns as values.
    """
    # Parse merger time
    merger_dt = Time(merger_time, format='isot').mjd
    
    # Read the file
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                timestamp = parts[0]
                filter_name = parts[1]
                app_mag = float(parts[2])
                error = float(parts[3])
                
                # Skip infinite errors
                #if error == float('inf') or np.isinf(error):
                #    continue
                
                # Convert timestamp to datetime
                obs_dt = Time(timestamp, format='isot').mjd
                
                # Calculate time since merger in days
                time_since_merger = obs_dt - merger_dt
                
                # Convert apparent magnitude to absolute magnitude
                # M = m - 5*log10(d) - 25, where d is in Mpc
                abs_mag = app_mag
                
                data.append({
                    'filter': filter_name,
                    'time': time_since_merger,
                    'magnitude': abs_mag,
                    'error': error
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Group by filter
    obs_data = {}
    for filter_name in df['filter'].unique():
        mask = df['filter'] == filter_name
        obs_data[filter_name] = df[mask][['time', 'magnitude', 'error']].reset_index(drop=True)
    
    return obs_data

def read_lines_file(filename, filts):
    """
    Read line data from ASCII file using pandas.
    Expected format:
    time u g r i z y J K
    0.5 -16.2 -16.0 -15.8 ...
    1.0 -15.8 -15.5 -15.3 ...
    ...
    
    Returns a dictionary with filter names as keys and DataFrames with 
    'time' and 'magnitude' columns as values.
    """
    # Read the file
    df = pd.read_csv(filename, delimiter=' ', comment='#', header=0)
    
    # Get column names (first column should be time, rest are filters)
    time_col = 'sample_times'
    filter_cols = [x for x in df.columns if x != 'sample_times']
    
    # Create dictionary with filter name as key
    data = {}
    for filter_name in filts:
        data[filter_name] = pd.DataFrame({
            'time': df[time_col],
            'magnitude': df[f'{filter_name}_median'],
            'magnitude_low': df[f'{filter_name}_2p5pt'],
            'magnitude_high': df[f'{filter_name}_97p5pt'],
            'magnitude_170817': df[f'{filter_name}_170817_median'],
            'magnitude_170817_low': df[f'{filter_name}_170817_2p5pt'],
            'magnitude_170817_high': df[f'{filter_name}_170817_97p5pt'],
        })
    
    return data

FILTER_LABELS = {
    'sdssu': 'u',
    'ps1::g': 'g',
    'ps1::r': 'r',
    'ps1::i': 'i',
    'ps1::z': 'z',
    'ps1::y': 'y',
    '2massj': 'J',
    '2massh': 'H',
    '2massks': 'K',
}

# Instrument single-visit 5σ limiting magnitudes (AB) to overplot on the LC panels.
# Source: Chase, O'Connor, Fryer et al. 2022, ApJ 927, 163 (arXiv:2105.12268)
LIMITING_MAGNITUDES = {
    'ps1::g':  {'ZTF': 20.8, 'LSST': 24.7},
    'ps1::r':  {'ZTF': 20.6, 'LSST': 24.2},
    'ps1::i':  {'ZTF': 19.9, 'LSST': 23.8},
    '2massj':  {'PRIME': 19.6, 'Roman': 25.5},
}

LIMIT_STYLES = {
    'ZTF':   dict(color='#e6194b', ls='--', lw=1.5),
    'LSST':  dict(color='#3cb44b', ls='-',  lw=1.5),
    'Roman': dict(color='#4363d8', ls='-',  lw=1.5),
    'PRIME': dict(color='#9c27b0', ls='--', lw=1.5),
}

# Filters shown in the compact 2-panel figure (one optical, one nIR).
# Change these strings to switch bands without touching the rest of the code.
SMALL_PLOT_PANELS = ['ps1::g', '2massj']

# Per-panel colors for the compact figure.  Any matplotlib color string works.
# Default: borrow the same rainbow hues used in the large 9-panel figure so
# the two figures look visually consistent (H-band red for the nIR panel).
_LARGE_PANEL_ORDER = ['sdssu', 'ps1::g', 'ps1::r', 'ps1::i',
                      'ps1::z', 'ps1::y', '2massj', '2massh', '2massks']
_LARGE_COLORS = sns.color_palette("rainbow", n_colors=len(_LARGE_PANEL_ORDER))
SMALL_PLOT_COLORS = {
    'ps1::g': _LARGE_COLORS[_LARGE_PANEL_ORDER.index('ps1::g')],   # same color as large g
    '2massj': _LARGE_COLORS[_LARGE_PANEL_ORDER.index('2massh')],   # H-band red from large figure
}

def plot_light_curves(lines_file, observations_file,
                     output_filename='light_curves.pdf',
                     n_cols=2, xlim=(0.3, 4.9), ylim=(26.5, 15),
                     n_model_lines=10, show_40mpc=True,
                     panels=None, layout=None, panel_colors=None,
                     show_limits_in_legend=True, show_limit_arrows=False):
    """
    Create multi-panel plot similar to the reference figure.
    
    Parameters:
    -----------
    lines_file : str
        Path to ASCII file containing line data in columnar format
        (time, filter1, filter2, ...)
    observations_file : str
        Path to ASCII file containing observation data points
    output_filename : str
        Output filename for the figure
    n_cols : int
        Number of columns in the grid
    xlim : tuple
        X-axis limits
    ylim : tuple
        Y-axis limits
    n_model_lines : int
        Number of model lines to plot (for gradient effect)
    panels : list of str or None
        Subset of filter names to include, in the desired order.
        Pass e.g. ``['ps1::r', '2massj']`` for a compact 2-panel figure.
        ``None`` (default) shows all nine standard bands.
    layout : dict or None
        Override any layout hyperparameter defined in ``LARGE_LAYOUT``.
        ``None`` uses ``LARGE_LAYOUT`` unchanged.  Pass ``SMALL_LAYOUT``
        (or a custom dict) for the compact figure.
    panel_colors : dict or None
        Map of filter name → matplotlib color for each panel.
        Missing entries fall back to the auto rainbow palette.
        Pass ``SMALL_PLOT_COLORS`` (or a custom dict) for the compact figure.
    """
    lyt = {**LARGE_LAYOUT, **(layout or {})}

    # Default panel order — all nine standard bands
    _default_panels = [
        'sdssu', 'ps1::g', 'ps1::r', 'ps1::i',
        'ps1::z', 'ps1::y',
        '2massj', '2massh', '2massks'
    ]
    all_panels = panels if panels is not None else _default_panels
    n_panels = len(all_panels)
    # Read data from both files
    lines_data = read_lines_file(lines_file, all_panels)
    obs_data = read_observations_file(observations_file)


    # Calculate grid dimensions
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=lyt['figsize'], sharey=lyt['sharey'])
    
    # Flatten axes array for easier indexing
    if n_panels == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Per-panel colors: use panel_colors dict where provided, rainbow fallback elsewhere
    _auto_colors = sns.color_palette("rainbow", n_colors=n_panels)
    _panel_colors = panel_colors or {}

    for idx, panel_name in enumerate(all_panels):
        ax = axes[idx]
        color = _panel_colors.get(panel_name, _auto_colors[idx])

        # Plot lines (models/predictions) if available
        if panel_name in lines_data and not lines_data[panel_name].empty:
            df_line = lines_data[panel_name]
            have_obs = panel_name in obs_data
            if have_obs:
                df_obs = obs_data[panel_name]
                obs_idx = np.where(np.isfinite(df_obs['error']))[0]
                noobs_idx = np.where(~np.isfinite(df_obs['error']))[0]
            else:
                df_obs = None
                obs_idx = np.array([], dtype=int)
                noobs_idx = np.array([], dtype=int)

            ax.plot(
                df_line['time'], df_line['magnitude'], 
                color=color, linewidth=3., zorder=1
            )
            ax.fill_between(
                df_line['time'],
                df_line['magnitude_low'], 
                df_line['magnitude_high'], 
                color=color, linewidth=3., zorder=1,
                alpha=0.5
            )
            if show_40mpc:
                ax.plot(
                    df_line['time'], df_line['magnitude_170817'],
                    color=color, linewidth=3., zorder=1, linestyle='--'
                )
                ax.fill_between(
                    df_line['time'],
                    df_line['magnitude_170817_low'],
                    df_line['magnitude_170817_high'],
                    color=color, linewidth=3., zorder=1,
                    alpha=0.5
                )
            if have_obs and df_obs is not None:
                if len(obs_idx) > 0:
                    ax.errorbar(
                        df_obs['time'][obs_idx],
                        df_obs['magnitude'][obs_idx],
                        yerr=df_obs['error'][obs_idx],
                        c='k', markersize=3, fmt='o',
                        capsize=5,
                        linewidth=1.5, zorder=2)
                if len(noobs_idx) > 0:
                    ax.scatter(df_obs['time'][noobs_idx],
                               df_obs['magnitude'][noobs_idx],
                               c='white', s=20, marker='v',
                               edgecolors='black',
                               linewidth=1.5, zorder=2)
            ax.invert_yaxis()
        
        # Formatting
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # ── Instrument limiting magnitude lines ──────────────────────────────
        if panel_name in LIMITING_MAGNITUDES:
            # y_faint: the dim (large-magnitude) end of the axis
            y_faint = max(ylim)
            y_bright = min(ylim)
            y_span = abs(y_faint - y_bright)
            arrow_len = 0.05 * y_span   # arrow length in mag units (toward brighter)
            label_gap = 0.02 * y_span   # gap between arrowhead and label text

            for inst_name, limit in LIMITING_MAGNITUDES[panel_name].items():
                sty = LIMIT_STYLES.get(inst_name, dict(color='gray', ls='--', lw=1.5))
                if not (y_bright <= limit <= y_faint + 0.5):
                    continue   # skip if well outside visible range
                ax.axhline(limit, color=sty['color'], ls=sty['ls'],
                           lw=sty['lw'], alpha=0.85, zorder=4)
                if show_limit_arrows:
                    # Arrow at right edge pointing toward brighter (smaller mag = visually up)
                    x_arrow = xlim[1] - 0.08 * (xlim[1] - xlim[0])
                    ax.annotate(
                        '',
                        xy=(x_arrow, limit - arrow_len),    # arrowhead (brighter direction)
                        xytext=(x_arrow, limit + arrow_len),  # arrow tail (fainter side)
                        arrowprops=dict(arrowstyle='->', color=sty['color'],
                                       lw=sty['lw']),
                        zorder=5,
                    )
                # Label below the line (fainter side), right-aligned
                if not show_limits_in_legend:
                    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),
                            limit + label_gap,
                            inst_name, color=sty['color'],
                            fontsize=lyt['panel_label_fontsize'] - 2,
                            va='top', ha='right', zorder=5)
        # ─────────────────────────────────────────────────────────────────────

        panel_label = FILTER_LABELS.get(panel_name, panel_name.replace('_', ':'))
        ax.text(0.95, 0.95, panel_label, transform=ax.transAxes,
               fontsize=lyt['panel_label_fontsize'], fontweight='bold', va='top', ha='right')

        # Set labels only for left column and bottom row
        ax.set_yticks([15, 18, 21, 24, 27])
        ax.set_yticklabels([15, 18, 21, 24, 27])
        if idx % n_cols != 0:
            ax.tick_params(labelleft=False)

        if idx >= n_panels - n_cols:
            ax.set_xlabel('Time since merger [days]', fontsize=lyt['xlabel_fontsize'])
        else:
            ax.set_xticklabels([])

        ax.grid(False)
        ax.tick_params(labelsize=lyt['tick_labelsize'])
    
    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.subplots_adjust(wspace=lyt['wspace'], hspace=lyt['hspace'], top=lyt['legend_top_margin'])

    # Centered figure-level legend above the panels
    from matplotlib.lines import Line2D
    legend_handles_row1 = [
        plt.scatter([], [], c='k', s=10, marker='o', label='AT2017gfo'),
        Line2D([0], [0], color='k', linewidth=2., label='Estimated lightcurves'),
    ]
    if show_40mpc:
        legend_handles_row1.append(
            Line2D([0], [0], color='k', linewidth=2., linestyle='--', label='Estimation at 40Mpc')
        )

    # Collect unique instruments shown across all panels (preserving first-seen order)
    legend_handles_row2 = []
    if show_limits_in_legend:
        seen = {}
        for panel_name in all_panels:
            if panel_name not in LIMITING_MAGNITUDES:
                continue
            y_faint = max(ylim)
            y_bright = min(ylim)
            for inst_name, limit in LIMITING_MAGNITUDES[panel_name].items():
                if inst_name in seen:
                    continue
                if not (y_bright <= limit <= y_faint + 0.5):
                    continue
                sty = LIMIT_STYLES.get(inst_name, dict(color='gray', ls='--', lw=1.5))
                seen[inst_name] = True
                legend_handles_row2.append(
                    Line2D([0], [0], color=sty['color'], linewidth=sty['lw'],
                           linestyle=sty['ls'],
                           label=f'{inst_name} limit')
                )

    # Single legend box with two rows: row 1 = model curves, row 2 = instrument limits.
    # When row 2 has more entries than row 1, move the first limit up into row 1 so
    # both rows have the same length (avoids a ragged grid).
    # Matplotlib fills legends column-by-column, so interleave the two rows to guarantee
    # row 1 occupies the top row and row 2 the bottom row.
    if legend_handles_row2:
        while len(legend_handles_row2) > len(legend_handles_row1):
            legend_handles_row1.append(legend_handles_row2.pop(0))
        ncol = max(len(legend_handles_row1), len(legend_handles_row2))
        spacer = Line2D([], [], color='none', label='')
        row1_padded = legend_handles_row1 + [spacer] * (ncol - len(legend_handles_row1))
        row2_padded = legend_handles_row2 + [spacer] * (ncol - len(legend_handles_row2))
        # interleave: column-major so [col0_row0, col0_row1, col1_row0, col1_row1, ...]
        combined = []
        for j in range(ncol):
            combined.append(row1_padded[j])
            combined.append(row2_padded[j])
        fig.legend(
            handles=combined,
            loc='upper center',
            bbox_to_anchor=(0.5, lyt['legend_bbox_y']),
            ncol=ncol,
            fontsize=lyt['legend_fontsize'],
            frameon=True,
            labelspacing=0.8,   # extra vertical gap between rows
        )
    else:
        fig.legend(
            handles=legend_handles_row1,
            loc='upper center',
            bbox_to_anchor=(0.5, lyt['legend_bbox_y']),
            ncol=len(legend_handles_row1),
            fontsize=lyt['legend_fontsize'],
            frameon=True,
        )

    # Y-axis label: fig.text for the large figure, set_ylabel for the small figure
    if lyt['ylabel_use_figtext']:
        fig.text(lyt['ylabel_x'], lyt['ylabel_ypos_frac'], 'Apparent magnitude',
                 va='center', ha='center', rotation='vertical', fontsize=lyt['ylabel_fontsize'])
    else:
        axes[0].set_ylabel('Apparent magnitude', fontsize=lyt['ylabel_fontsize'])

    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
    return fig

# Example usage
if __name__ == "__main__":
    import sys
    
    print("Making plot...")
    
    if len(sys.argv) >= 2:
        lines_file = sys.argv[1]
        obser_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'light_curves.pdf'
        show_40mpc = '--no-40mpc' not in sys.argv

        # Large plot — all nine bands (tune via LARGE_LAYOUT at top of file)
        plot_light_curves(lines_file, obser_file, output_file,
                          show_40mpc=show_40mpc, layout=LARGE_LAYOUT)

        # Small plot — one optical + one nIR band (tune via SMALL_LAYOUT /
        # SMALL_PLOT_PANELS at top of file)
        small_output = output_file.replace('.pdf', '_small.pdf')
        plot_light_curves(
            lines_file, obser_file, small_output,
            n_cols=2,
            show_40mpc=show_40mpc,
            panels=SMALL_PLOT_PANELS,
            layout=SMALL_LAYOUT,
            panel_colors=SMALL_PLOT_COLORS,
        )
    else:
        print("Usage: python script.py <lines_file> [output_file]")
        print("\nLines file format (columnar):")
        print("time u g r i z y J K")
        print("0.5 -16.2 -16.0 -15.8 -15.6 -15.4 -15.2 -15.0 -14.8")
        print("1.0 -15.8 -15.5 -15.3 -15.1 -14.9 -14.7 -14.5 -14.3")
        print("...")
        print("\nOption 2 (columnar with NaN for missing data):")
        print("time u g r i")
        print("1.0 -16.0 nan -15.8 nan")
        print("2.0 nan -15.5 nan -15.2")
        print("\nExample:")
        print("python script.py lines.txt observations.txt output.pdf")

    print("Making plot... DONE")