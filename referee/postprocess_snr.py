"""
Postprocessing script for SNR samples (run locally).
Takes the JSON file produced by check_snr.py and prints median + 95% HDI
for H1_optimal_SNR and L1_optimal_SNR (and matched-filter equivalents) per run.
Also plots per-run histograms for visual comparison.
"""

import os
import json
import numpy as np
import arviz
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Match money_plots_jarvis.py styling when running locally
_cwd = os.getcwd()
if "Woute029" in _cwd:
    _fs = 14
    _ticks_fs = 12
    _legend_fs = 12
    plt.rcParams.update({
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "xtick.labelsize": _ticks_fs,
        "ytick.labelsize": _ticks_fs,
        "axes.labelsize": _ticks_fs,
        "legend.fontsize": _legend_fs,
        "legend.title_fontsize": _legend_fs,
        "figure.titlesize": _fs,
    })

DEFAULT_INPUT = "./data/snr_samples.json"
DEFAULT_OUTPUT = "./figures/snr_histograms.pdf"

# Colors matching money_plots_jarvis.py / utils.py
ORANGE = "#de8f07"
BLUE = "#0472b1"
PURPLE = "#9467bd"
LIGHT_BLUE = "#56B4E9"

KEYS_OF_INTEREST = [
    "H1_optimal_snr",
    "L1_optimal_snr",
    "H1_matched_filter_snr",
    "L1_matched_filter_snr",
    "network_optimal_snr",
    "network_matched_filter_snr",
]


def summarize(samples, hdi_prob=0.95):
    arr = np.array(samples)
    median = np.median(arr)
    hdi = arviz.hdi(arr, hdi_prob=hdi_prob)
    return median, hdi[0], hdi[1]


def compute_network_snr_from_components(snrs):
    """Compute network SNR as sqrt(H1**2 + L1**2) from per-detector optimal SNR samples."""
    key_map = {k.lower(): k for k in snrs}
    h1_key = key_map.get("h1_optimal_snr")
    l1_key = key_map.get("l1_optimal_snr")
    if h1_key is None or l1_key is None:
        return None
    h1 = np.array(snrs[h1_key])
    l1 = np.array(snrs[l1_key])
    return np.sqrt(h1**2 + l1**2)


def compute_network_mf_snr_from_components(snrs):
    """Compute network matched-filter SNR as sqrt(H1_mf**2 + L1_mf**2) from per-detector samples."""
    key_map = {k.lower(): k for k in snrs}
    h1_key = key_map.get("h1_matched_filter_snr")
    l1_key = key_map.get("l1_matched_filter_snr")
    if h1_key is None or l1_key is None:
        return None
    h1 = np.array(snrs[h1_key])
    l1 = np.array(snrs[l1_key])
    return np.sqrt(h1**2 + l1**2)


SUBSET_RUNS = [
    "prod_BW_XP_s005_l5000_default",
    "prod_BW_XP_s005_l5000_double_gaussian_niu",
    "prod_BW_XP_s005_leos_default",
]

# Paper-quality subset: low/high spin × Lambda-uniform/EOS-informed
PAPER_SUBSET_RUNS = [
    "prod_BW_XP_s005_l5000_default",
    "prod_BW_XP_s040_l5000_default",
    "prod_BW_XP_s005_leos_default",
    "prod_BW_XP_s040_leos_default",
]


def plot_snr_histograms(all_results, output_path, bins=50, run_filter=None):
    """Plot overlaid per-run histograms for each SNR quantity in a 2-column layout.

    Parameters
    ----------
    run_filter : list of str, optional
        If given, only include runs whose names are in this list (order preserved).
    """
    if run_filter is not None:
        all_results = {k: v for k, v in all_results.items() if k in run_filter}
        # Preserve the requested order
        all_results = {k: all_results[k] for k in run_filter if k in all_results}

    # Build the full set of keys to plot across all runs, preserving KEYS_OF_INTEREST order,
    # then append the derived network SNR.
    present_keys = []
    for key in KEYS_OF_INTEREST:
        for run_snrs in all_results.values():
            key_map = {k.lower(): k for k in run_snrs}
            if key_map.get(key.lower()) is not None:
                if key not in present_keys:
                    present_keys.append(key)
                break
    present_keys.append("network_snr_computed")
    present_keys.append("network_mf_snr_computed")

    n_panels = len(present_keys)
    ncols = 2
    nrows = (n_panels + 1) // ncols

    # Wide and tall: each panel ~4 wide, ~3 tall; legend column adds ~3 inches
    fig_width = ncols * 4 + 3
    fig_height = nrows * 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    run_names = list(all_results.keys())
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(run_names))]

    handles = []
    labels = []

    for panel_idx, key in enumerate(present_keys):
        ax = axes[panel_idx]
        for run_idx, (run_name, snrs) in enumerate(all_results.items()):
            if key == "network_snr_computed":
                samples = compute_network_snr_from_components(snrs)
                if samples is None:
                    continue
            elif key == "network_mf_snr_computed":
                samples = compute_network_mf_snr_from_components(snrs)
                if samples is None:
                    continue
            else:
                key_map = {k.lower(): k for k in snrs}
                actual_key = key_map.get(key.lower())
                if actual_key is None:
                    continue
                samples = np.array(snrs[actual_key])

            ax.hist(
                samples,
                bins=bins,
                density=True,
                histtype="step",
                color=colors[run_idx],
                linewidth=1.2,
                label=run_name,
            )
            if panel_idx == 0:
                handles.append(
                    Line2D([0], [0], color=colors[run_idx], linewidth=1.5)
                )
                labels.append(run_name)

        title = key.replace("_", " ")
        if key == "network_snr_computed":
            title = "network SNR (computed, optimal)"
        elif key == "network_mf_snr_computed":
            title = "network SNR (computed, matched-filter)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("SNR", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    # Place legend to the right of the figure
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0 - 3.0 / fig_width, 0.5),
        ncol=1,
        fontsize=8,
        title="Run",
        title_fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1.0 - 3.0 / fig_width, 1.0))
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved histogram figure to {output_path}")


def plot_snr_histograms_paper(all_results, output_path, bins=50):
    """Single-panel paper figure: network matched-filter SNR for the four paper-subset runs.

    Four distinct colors (no line-style encoding):
      Λ-uniform:    ORANGE (low-spin), BLUE (high-spin)
      EOS-informed: #2596be (low-spin), palevioletred (high-spin)
    """
    _EOS_BLUE = "#2596be"
    _RED      = "palevioletred"

    _RUN_COLOR = {
        "prod_BW_XP_s005_l5000_default": ORANGE,
        "prod_BW_XP_s040_l5000_default": BLUE,
        "prod_BW_XP_s005_leos_default":  _EOS_BLUE,
        "prod_BW_XP_s040_leos_default":  _RED,
    }
    _RUN_LABEL = {
        "prod_BW_XP_s005_l5000_default": r"$\Lambda$-uniform, $a_i \leq 0.05$",
        "prod_BW_XP_s040_l5000_default": r"$\Lambda$-uniform, $a_i \leq 0.40$",
        "prod_BW_XP_s005_leos_default":  r"EOS-informed, $a_i \leq 0.05$",
        "prod_BW_XP_s040_leos_default":  r"EOS-informed, $a_i \leq 0.40$",
    }

    runs = [r for r in PAPER_SUBSET_RUNS if r in all_results]
    if not runs:
        print("No paper-subset runs found in data; skipping paper figure.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for run_name in runs:
        snrs = all_results[run_name]
        samples = compute_network_mf_snr_from_components(snrs)
        if samples is None:
            continue
        ax.hist(
            samples,
            bins=bins,
            density=True,
            histtype="step",
            color=_RUN_COLOR.get(run_name, "black"),
            linewidth=1.6,
            label=_RUN_LABEL.get(run_name, run_name),
        )

    fs_label = 16
    ax.set_xlabel(r"Network SNR", fontsize=fs_label)
    ax.set_ylabel("Probability density", fontsize=fs_label)
    ax.legend(fontsize=12, frameon=True, loc="upper left")
    ax.set_xlim(left=7.5, right=9.7)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved paper histogram figure to {output_path}")


def print_network_snr_table(all_results, hdi_prob=0.95):
    """Print two sorted tables: network optimal SNR and network matched-filter SNR (median, high → low)."""
    optimal_rows = []
    mf_rows = []

    for run_name, snrs in all_results.items():
        net = compute_network_snr_from_components(snrs)
        if net is not None:
            median, lo, hi = summarize(net, hdi_prob=hdi_prob)
            optimal_rows.append((run_name, median, lo, hi))

        net_mf = compute_network_mf_snr_from_components(snrs)
        if net_mf is not None:
            median, lo, hi = summarize(net_mf, hdi_prob=hdi_prob)
            mf_rows.append((run_name, median, lo, hi))

    pct = int(hdi_prob * 100)

    for label, rows in [("Network SNR — optimal (sorted high → low)", optimal_rows),
                         ("Network SNR — matched-filter (sorted high → low)", mf_rows)]:
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
        col_width = max(len(r[0]) for r in rows_sorted) + 2
        header = f"{'Run':<{col_width}}  median    {pct}% HDI"
        print(f"=== {label} ===")
        print(header)
        print("-" * (len(header) + 10))
        for run_name, median, lo, hi in rows_sorted:
            print(f"{run_name:<{col_width}}  {median:.3f}     [{lo:.3f}, {hi:.3f}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="Postprocess SNR samples from check_snr.py output.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the JSON file produced by check_snr.py")
    parser.add_argument("--hdi", type=float, default=0.95, help="HDI probability (default: 0.95)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for the output histogram figure")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: 50)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} run(s) from {args.input}\n")

    for run_name, snrs in all_results.items():
        print(f"=== {run_name} ===")
        available_keys = list(snrs.keys())

        # Try to match case-insensitively
        key_map = {k.lower(): k for k in available_keys}

        printed_any = False
        for key in KEYS_OF_INTEREST:
            actual_key = key_map.get(key.lower())
            if actual_key is None:
                continue
            samples = snrs[actual_key]
            median, lo, hi = summarize(samples, hdi_prob=args.hdi)
            print(f"  {actual_key}: median={median:.3f}, {int(args.hdi*100)}% HDI=[{lo:.3f}, {hi:.3f}]")
            printed_any = True

        # Also print computed network SNRs
        net = compute_network_snr_from_components(snrs)
        if net is not None:
            median, lo, hi = summarize(net, hdi_prob=args.hdi)
            print(f"  network_snr_computed (optimal): median={median:.3f}, {int(args.hdi*100)}% HDI=[{lo:.3f}, {hi:.3f}]")
            printed_any = True
        net_mf = compute_network_mf_snr_from_components(snrs)
        if net_mf is not None:
            median, lo, hi = summarize(net_mf, hdi_prob=args.hdi)
            print(f"  network_snr_computed (matched-filter): median={median:.3f}, {int(args.hdi*100)}% HDI=[{lo:.3f}, {hi:.3f}]")
            printed_any = True

        if not printed_any:
            print(f"  No matching SNR keys found. Available: {available_keys}")
        print()

    # Sorted summary tables: network SNR median (optimal and matched-filter)
    print_network_snr_table(all_results, hdi_prob=args.hdi)

    plot_snr_histograms(all_results, args.output, bins=args.bins)

    # Subset figure: only the three comparison runs
    subset_output = args.output.replace(".pdf", "_subset.pdf")
    plot_snr_histograms(all_results, subset_output, bins=args.bins, run_filter=SUBSET_RUNS)

    # Paper-quality figure: low/high spin × Lambda-uniform/EOS-informed
    paper_output = args.output.replace(".pdf", "_paper.pdf")
    plot_snr_histograms_paper(all_results, paper_output, bins=args.bins)


if __name__ == "__main__":
    main()
