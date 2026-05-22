#!/usr/bin/env python3
"""
Corner plot of m1_source, m2_source, lambda_1, lambda_2 for ET vs ET+CE runs.
Saved to figures/GW_PE/ET_vs_ET_CE_component_masses_lambdas.pdf (not copied to paper).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner

# Use TeX styling when running locally
fs = 18
ticks_fs = 20
label_fs = 24
legend_fs = 22
plt.rcParams.update({
    "axes.grid": False,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": ticks_fs,
    "ytick.labelsize": ticks_fs,
    "axes.labelsize": label_fs,
    "legend.fontsize": legend_fs,
    "legend.title_fontsize": legend_fs,
    "figure.titlesize": fs,
})

PARAMETERS = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
LABELS = [
    r"$m_1^{\rm src}\ [{\rm M}_\odot]$",
    r"$m_2^{\rm src}\ [{\rm M}_\odot]$",
    r"$\Lambda_1$",
    r"$\Lambda_2$",
]
RANGES = {
    "mass_1_source": (1.2, 2.4),
    "mass_2_source": (1.0, 1.8),
    "lambda_1": (0, 2500),
    "lambda_2": (0, 5000),
}

ET_COLOR = "#de8f05"
ET_CE_COLOR = "mediumslateblue"

FILEPATHS = [
    "../referee/jester_reruns/3G/et/jester_eos_et_run_alignedspin.npz",
    "../referee/jester_reruns/3G/et_ce/jester_eos_et_ce_run_alignedspin.npz",
]
DATASET_LABELS = ["ET", "ET+CE"]
COLORS = [ET_COLOR, ET_CE_COLOR]

SAVE_PATH = "./figures/GW_PE/ET_vs_ET_CE_component_masses_lambdas.pdf"

CORNER_KWARGS = {
    "bins": 40,
    "smooth": 0.9,
    "plot_datapoints": False,
    "plot_density": False,
    "fill_contours": True,
    "show_titles": False,
    "levels": [0.68, 0.95],
    "labelpad": 0.10,
    "max_n_ticks": 3,
    "min_n_ticks": 2,
}


def load_samples(filepath, parameters):
    data = np.load(filepath)
    missing = [p for p in parameters if p not in data]
    if missing:
        raise ValueError(f"Missing parameters {missing} in {filepath}. Available: {list(data.keys())}")
    return np.column_stack([data[p] for p in parameters])


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    ranges = [RANGES.get(p, 1.0) for p in PARAMETERS]

    all_samples = []
    for fp, label, color in zip(FILEPATHS, DATASET_LABELS, COLORS):
        samples = load_samples(fp, PARAMETERS)
        print(f"Loaded {len(samples)} samples from {fp} ({label})")
        all_samples.append(samples)

    # Use ET+CE (index 1) as the dummy normalization dataset so 1D histograms
    # share the same y-axis ceiling across both runs.
    n_min = min(len(s) for s in all_samples)
    dummy = all_samples[1][:n_min]

    fig = None
    for i, (samples, label, color) in enumerate(zip(all_samples, DATASET_LABELS, COLORS)):
        kwargs = dict(CORNER_KWARGS, color=color, labels=LABELS, range=ranges)
        if fig is None:
            fig = corner.corner(samples, **kwargs)
        else:
            corner.corner(samples, fig=fig, **kwargs)

    # Invisible dummy to normalise 1D histogram heights
    corner.corner(dummy, fig=fig, color="white", alpha=0,
                  labels=LABELS, range=ranges,
                  **{k: v for k, v in CORNER_KWARGS.items() if k != "fill_contours"},
                  fill_contours=False)

    # Add 90% CI as axis titles
    axes = np.array(fig.get_axes()).reshape(len(PARAMETERS), len(PARAMETERS))
    for col_idx, param in enumerate(PARAMETERS):
        ax = axes[col_idx, col_idx]
        for samples, color in zip(all_samples, COLORS):
            lo, med, hi = np.percentile(samples[:, col_idx], [5, 50, 95])
            current = ax.get_title()
            label_str = r"$" + f"{med:.2f}^{{+{hi-med:.2f}}}_{{-{med-lo:.2f}}}" + r"$"
            ax.set_title((current + "\n" if current else "") + label_str,
                         color=color, fontsize=14, pad=4)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, DATASET_LABELS)]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), frameon=True)

    fig.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
    print(f"\nSaved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
