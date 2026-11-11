"""Remake ET_full_injection_R14_histogram.pdf using local jester rerun data.

Reads masses/radii directly from HDF5 files in the 3G sub-directory and
from the radio run, computes R14 via interpolation, and plots KDE histograms
together with the injection truth value from jester_GW170817_maxL_EOS.npz.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import arviz
import h5py
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

mpl_params = {
    "axes.grid": False,
    "text.usetex": True,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "figure.titlesize": 12,
}
plt.rcParams.update(mpl_params)

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_ET    = os.path.join(SCRIPT_DIR, "3G", "et",    "outdir", "results.h5")
PATH_ET_CE = os.path.join(SCRIPT_DIR, "3G", "et_ce", "outdir", "results.h5")
PATH_RADIO = os.path.join(SCRIPT_DIR, "radio", "outdir", "results.h5")
PATH_TARGET_EOS = os.path.join(SCRIPT_DIR, "3G", "jester_GW170817_maxL_EOS.npz")

ET_COLOR    = "#de8f05"
ET_CE_COLOR = "mediumslateblue"
RADIO_COLOR = "dimgray"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_r14(h5_path: str) -> np.ndarray:
    """Return R_{1.4} array from an HDF5 results file."""
    with h5py.File(h5_path, "r") as f:
        masses = f["posterior/derived_eos/masses_EOS"][:]
        radii  = f["posterior/derived_eos/radii_EOS"][:]
    return np.array([np.interp(1.4, m, r) for m, r in zip(masses, radii)])


def credible_interval(values: np.ndarray, hdi_prob: float = 0.90):
    """Return (low_err, median, high_err) for a 1-D array."""
    med = np.median(values)
    lo, hi = arviz.hdi(values, hdi_prob=hdi_prob)
    return med - lo, med, hi - med


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def main(plot_text: bool = False):
    # -- Target EOS ----------------------------------------------------------
    eos = np.load(PATH_TARGET_EOS)
    R14_TARGET = np.interp(1.4, eos["masses_EOS"], eos["radii_EOS"])
    print(f"Injection R14_TARGET = {R14_TARGET:.3f} km")

    # -- Load R14 samples ----------------------------------------------------
    R14_et    = load_r14(PATH_ET)
    R14_et_ce = load_r14(PATH_ET_CE)
    R14_radio = load_r14(PATH_RADIO)
    print(f"ET    R14 median = {np.median(R14_et):.2f} km  (n={len(R14_et)})")
    print(f"ET+CE R14 median = {np.median(R14_et_ce):.2f} km  (n={len(R14_et_ce)})")
    print(f"Radio R14 median = {np.median(R14_radio):.2f} km  (n={len(R14_radio)})")

    # -- KDE evaluation grid -------------------------------------------------
    x = np.linspace(9.0, 15.0, 1000)
    y_radio = gaussian_kde(R14_radio)(x)
    y_et    = gaussian_kde(R14_et)(x)
    y_et_ce = gaussian_kde(R14_et_ce)(x)

    # -- Credible intervals --------------------------------------------------
    low_et,    med_et,    high_et    = credible_interval(R14_et)
    low_et_ce, med_et_ce, high_et_ce = credible_interval(R14_et_ce)

    # -- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(x, y_radio, color=RADIO_COLOR, lw=3.0, label="Heavy PSRs")
    ax.fill_between(x, y_radio, alpha=0.3, color=RADIO_COLOR)

    ax.plot(x, y_et, color=ET_COLOR, lw=3.0, label="ET")
    ax.fill_between(x, y_et, alpha=0.3, color=ET_COLOR)

    ax.plot(x, y_et_ce, color=ET_CE_COLOR, lw=3.0, label="ET$+$CE")
    ax.fill_between(x, y_et_ce, alpha=0.3, color=ET_CE_COLOR)

    ax.axvline(x=R14_TARGET, color="black", ls="--", lw=2.0, label="Injection")

    if plot_text:
        textstr_et    = (f"${med_et:.2f}"
                         f"^{{+{high_et:.2f}}}_{{-{low_et:.2f}}}$")
        textstr_et_ce = (f"${med_et_ce:.2f}"
                         f"^{{+{high_et_ce:.2f}}}_{{-{low_et_ce:.2f}}}$")
        ax.text(0.95, 0.95, textstr_et,
                transform=ax.transAxes,
                va="top", ha="right", color=ET_COLOR, fontsize=12)
        ax.text(0.95, 0.80, textstr_et_ce,
                transform=ax.transAxes,
                va="top", ha="right", color=ET_CE_COLOR, fontsize=12)

    ax.set_xlabel(r"$R_{1.4}$ [km]", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_xlim(11.0, 14.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=11)

    out_path = os.path.join(SCRIPT_DIR, "ET_full_injection_R14_histogram.pdf")
    out_path = "/Users/Woute029/Documents/Code/projects/19_GW231109_referee_report/paper/ET_full_injection_R14_histogram.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  ET    R14: {med_et:.2f} +{high_et:.2f} -{low_et:.2f} km (90% HDI)")
    print(f"  ET+CE R14: {med_et_ce:.2f} +{high_et_ce:.2f} -{low_et_ce:.2f} km (90% HDI)")


if __name__ == "__main__":
    main()
