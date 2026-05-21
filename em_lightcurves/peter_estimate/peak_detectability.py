"""
Peak magnitude detectability analysis for GW231109 kilonova.

Reads the pre-computed lightcurve posteriors (lc_data_band.dat),
finds the peak (brightest) apparent magnitude in each filter, and
compares against the single-visit limiting magnitudes of key wide-field
follow-up instruments.  Produces a summary plot and prints a detection
table to stdout.

Usage:
    python peak_detectability.py [path/to/lc_data_band.dat]

Default data file: lc_data_band.dat (same directory as this script).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# ---------------------------------------------------------------------------
# Instrument limiting magnitudes (single-visit, 5σ, AB)
# All values taken from:
#   Chase, O'Connor, Fryer et al. 2022, ApJ 927, 163 (arXiv:2105.12268)
#   "Kilonova Detectability with Wide-field Instruments"
# Explicitly quoted in em_references_discussion.tex (claude_output/EM_data/):
#   ZTF r ~ 20.4 AB; Rubin/LSST r ~ 24.7 AB;
#   Roman H ~ 26.5 AB (reaches z~1); PRIME H ~ 19–20 AB (stacking needed at 165 Mpc)
# Multi-band values for same instruments are also from Chase et al. 2022.
# ---------------------------------------------------------------------------

# Structure: {instrument_name: {filter_key: limit_mag, ...}, ...}
# Filter keys must match the column prefix in lc_data_band.dat
INSTRUMENTS = {
    'ZTF': {
        'ps1::g': 20.8,
        'ps1::r': 20.4,   # Chase et al. 2022
        'ps1::i': 19.9,
    },
    'Rubin/LSST': {
        'sdssu':  23.9,
        'ps1::g': 24.9,
        'ps1::r': 24.7,   # Chase et al. 2022
        'ps1::i': 24.0,
        'ps1::z': 23.3,
        'ps1::y': 22.4,
    },
    'Roman': {
        '2massj':  26.9,
        '2massh':  26.5,   # Chase et al. 2022 (H-band, reaches z~1)
        '2massks': 25.4,
    },
    'PRIME': {
        '2massj':  20.0,
        '2massh':  19.5,   # Chase et al. 2022: "~19–20 AB"; stacking needed at 165 Mpc
    },
}

# Display labels for filters
FILTER_LABELS = {
    'sdssu':   r'$u$',
    'ps1::g':  r'$g$',
    'ps1::r':  r'$r$',
    'ps1::i':  r'$i$',
    'ps1::z':  r'$z$',
    'ps1::y':  r'$y$',
    '2massj':  r'$J$',
    '2massh':  r'$H$',
    '2massks': r'$K_s$',
}

FILTER_ORDER = ['sdssu', 'ps1::g', 'ps1::r', 'ps1::i',
                'ps1::z', 'ps1::y', '2massj', '2massh', '2massks']

INSTRUMENT_COLORS = {
    'ZTF':        '#e6194b',
    'Rubin/LSST': '#3cb44b',
    'Roman':      '#4363d8',
    'PRIME':      '#f58231',
}

INSTRUMENT_MARKERS = {
    'ZTF':        's',
    'Rubin/LSST': 'D',
    'Roman':      '*',
    'PRIME':      '^',
}

# ---------------------------------------------------------------------------

def load_lightcurves(data_file):
    df = pd.read_csv(data_file, sep=' ', header=0)
    return df


def compute_peak_magnitudes(df, filters):
    """
    For each filter, find the peak (minimum) apparent magnitude in the
    GW231109 posterior (median, 16th, 84th percentiles of the full posterior).
    Returns a dict: {filter: {'peak_med', 'peak_lo', 'peak_hi', 'peak_time'}}
    """
    peaks = {}
    for filt in filters:
        col_med = f'{filt}_median'
        col_lo  = f'{filt}_16pt'
        col_hi  = f'{filt}_84pt'
        if col_med not in df.columns:
            continue
        # peak = minimum of the median light curve (ignoring non-finite values)
        valid = np.isfinite(df[col_med])
        if not valid.any():
            continue
        idx_peak = df.loc[valid, col_med].idxmin()
        peaks[filt] = {
            'peak_med':  df.loc[idx_peak, col_med],
            'peak_lo':   df.loc[idx_peak, col_lo],
            'peak_hi':   df.loc[idx_peak, col_hi],
            'peak_time': df.loc[idx_peak, 'sample_times'],
        }
    return peaks


def print_detection_table(peaks, instruments):
    """Print a detection summary table to stdout."""
    header = f"{'Filter':<10} {'Peak (median)':<16} {'1σ range':<20} {'Peak time':>10}   Detectable by"
    print()
    print("=" * 90)
    print("  GW231109 kilonova — peak apparent magnitude vs instrument limits")
    print("  Source for instrument limits: Chase et al. 2022, ApJ 927, 163 (arXiv:2105.12268)")
    print("  'Kilonova Detectability with Wide-field Instruments'")
    print("=" * 90)
    print()
    print("  HOW TO READ: smaller magnitude number = brighter source.")
    print("  A source is DETECTABLE if its peak magnitude < instrument limit")
    print("  (i.e. the KN peak appears ABOVE the instrument limit line in the plot).")
    print()
    print(header)
    print("-" * 90)

    for filt in FILTER_ORDER:
        if filt not in peaks:
            continue
        p = peaks[filt]
        label = FILTER_LABELS[filt].replace('$', '').replace('\\', '')
        range_str = f"[{p['peak_lo']:.1f}, {p['peak_hi']:.1f}]"
        detectable_by = []
        not_detectable_by = []
        for inst_name, inst_limits in instruments.items():
            if filt in inst_limits:
                lim = inst_limits[filt]
                if p['peak_med'] <= lim:
                    detectable_by.append(f"{inst_name} (limit {lim:.1f})")
                else:
                    not_detectable_by.append(f"{inst_name} (limit {lim:.1f}, too faint)")
        det_str = ", ".join(detectable_by) if detectable_by else "None"
        if not_detectable_by:
            det_str += "  |  NOT: " + ", ".join(not_detectable_by)
        print(f"  {label:<8} {p['peak_med']:>12.2f}   {range_str:<20} {p['peak_time']:>6.2f} d   {det_str}")

    print("=" * 90)
    print()
    print("  Instrument single-visit 5σ limiting magnitudes (AB) — all from Chase et al. 2022:")
    print()
    for inst_name, inst_limits in instruments.items():
        filt_str = ", ".join(
            f"{FILTER_LABELS[f].replace('$','').replace(chr(92),'')}={v:.1f}"
            for f, v in inst_limits.items()
            if f in peaks
        )
        if filt_str:
            print(f"    {inst_name:<15}: {filt_str}")
    print("=" * 90)
    print()


def make_detectability_plot(peaks, instruments, output_file='peak_detectability.pdf'):
    """
    Two-panel figure: left = optical, right = nIR.
    For each filter: peak magnitude as filled circle with 1σ error bars.
    Instrument limits shown as coloured dashed lines with markers.

    HOW TO READ THE PLOT
    --------------------
    The y-axis is inverted: brighter sources sit higher (smaller magnitude number).
    A kilonova is DETECTABLE by an instrument if its black data point lies ABOVE
    (higher than) the instrument's coloured limit marker.
    Conversely, if the black point is BELOW a coloured marker, the KN is too faint
    for that instrument to detect in a single visit.

    Instrument limits source: Chase et al. 2022, ApJ 927, 163 (arXiv:2105.12268),
    "Kilonova Detectability with Wide-field Instruments."
    """
    optical_filts = ['sdssu', 'ps1::g', 'ps1::r', 'ps1::i', 'ps1::z', 'ps1::y']
    nir_filts     = ['2massj', '2massh', '2massks']

    opt_avail = [f for f in optical_filts if f in peaks]
    nir_avail = [f for f in nir_filts     if f in peaks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={'width_ratios': [len(opt_avail), len(nir_avail)],
                                          'wspace': 0.07})

    # magnitude axis runs brighter (smaller) at top
    MAG_MIN, MAG_MAX = 18.0, 28.0

    for ax, filt_list, group_label in zip(
            axes,
            [opt_avail, nir_avail],
            ['Optical', 'Near-infrared']):

        x_positions = np.arange(len(filt_list))
        ax.set_xlim(-0.6, len(filt_list) - 0.4)
        ax.set_ylim(MAG_MAX, MAG_MIN)   # inverted: bright at top
        ax.set_xticks(x_positions)
        ax.set_xticklabels([FILTER_LABELS[f] for f in filt_list])
        ax.set_title(group_label, fontsize=14, pad=6)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # --- instrument limiting magnitudes ---
        for inst_name, inst_limits in instruments.items():
            inst_color  = INSTRUMENT_COLORS[inst_name]
            inst_marker = INSTRUMENT_MARKERS[inst_name]
            for xi, filt in enumerate(filt_list):
                if filt not in inst_limits:
                    continue
                lim = inst_limits[filt]
                # horizontal tick spanning ±0.35 of the cell
                ax.hlines(lim, xi - 0.35, xi + 0.35,
                          color=inst_color, linewidth=1.8,
                          linestyle='--', zorder=2)
                ax.plot(xi, lim, marker=inst_marker,
                        color=inst_color, markersize=8, zorder=3)

        # --- peak magnitudes with error bars ---
        for xi, filt in enumerate(filt_list):
            p = peaks[filt]
            err_lo = abs(p['peak_med'] - p['peak_lo'])
            err_hi = abs(p['peak_hi'] - p['peak_med'])
            ax.errorbar(xi, p['peak_med'],
                        yerr=[[err_lo], [err_hi]],
                        fmt='o', color='black',
                        markersize=8, linewidth=2,
                        capsize=5, zorder=5,
                        label='KN peak (this work)' if xi == 0 else '')

    # shared y-label on the left panel only
    axes[0].set_ylabel('Apparent magnitude [AB]', fontsize=14)
    axes[1].tick_params(labelleft=False)

    # --- legend ---
    legend_handles = [
        Line2D([0], [0], marker='o', color='black', markersize=8,
               linewidth=2, label='KN peak (this work, 1$\\sigma$)'),
    ]
    for inst_name in instruments:
        c = INSTRUMENT_COLORS[inst_name]
        m = INSTRUMENT_MARKERS[inst_name]
        legend_handles.append(
            Line2D([0], [0], marker=m, color=c, markersize=8,
                   linewidth=1.8, linestyle='--',
                   label=f'{inst_name} limit')
        )

    fig.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=11, frameon=True)

    fig.suptitle(r'GW231109 kilonova peak magnitude vs.\ instrument limits',
                 y=1.09, fontsize=14)

    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Plot saved → {output_file}")
    return fig


# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, 'lc_data_band.dat')

    data_file   = sys.argv[1] if len(sys.argv) > 1 else default_data
    output_file = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(script_dir, 'peak_detectability.pdf')

    print(f"\nLoading lightcurves from: {data_file}")
    df = load_lightcurves(data_file)

    peaks = compute_peak_magnitudes(df, FILTER_ORDER)

    print_detection_table(peaks, INSTRUMENTS)
    make_detectability_plot(peaks, INSTRUMENTS, output_file)


if __name__ == '__main__':
    main()
