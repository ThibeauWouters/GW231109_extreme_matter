import os
import numpy as np
import bilby
import matplotlib.pyplot as plt
import scipy
from pathlib import Path

from utils import DoubleGaussian
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from bilby.gw.conversion import luminosity_distance_to_redshift
from astropy.cosmology import Planck18

from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import PLOT_PARAMS
plt.rcParams.update(PLOT_PARAMS)

bilby.core.utils.random.seed(123456)
np.random.seed(42)

# =============================================================================
# Configuration - Adjust these parameters as needed
# =============================================================================

# # Plot range configuration (set to None for automatic range based on data)
# M1_XLIM = None # full range
# M2_XLIM = None # full range

# Alternatively, uncomment these to zoom in on posteriors:
M1_XLIM = (1.2, 1.8)
M2_XLIM = (1.1, 1.8)

# Sample sizes (increase these for smoother KDEs)
PRIOR_SAMPLE_SIZE = 500_000   # Number of initial samples to draw from priors
OUTPUT_SAMPLE_SIZE = 100_000  # Number of output samples after m1 >= m2 constraint

# KDE parameters
KDE_NBINS = 10_000             # Number of points for KDE evaluation (higher = smoother)
KDE_METHOD = 'Transform'       # 'Reflection' or 'Transform' (Transform supports smooth parameter)
KDE_SMOOTH = 3.0               # Smoothing factor (higher = smoother, typically 0.5-2.0)

# Output configuration
OUTPUT_PATH = './figures/populations/populations_component_masses_comparison.pdf'
DPI = 600

# =============================================================================

if "Woute029" in os.getcwd():
    print(f"Updating plotting parameters for TeX")
    fs = 18
    ticks_fs = 20
    label_fs = 22  # Bigger labels
    legend_fs = 22  # Smaller legend
    labelpad = 15  # Bigger labelpad
    rc_params = {"axes.grid": False,
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"],
            "xtick.labelsize": ticks_fs,
            "ytick.labelsize": ticks_fs,
            "axes.labelsize": label_fs,
            "legend.fontsize": legend_fs,
            "legend.title_fontsize": legend_fs,
            "figure.titlesize": fs}
    plt.rcParams.update(rc_params)


def compute_kde(data, Nbins, xMin, xMax, bounded=False, method=None, smooth=None):
    """Compute KDE of `data` normalised across interval [`xMin`, `xMax`]
    using Nbins bins.
    If `bounded` is true, use a bounded KDE method with bounds `xMin`, `xMax`.
    """
    xPoints = np.linspace(xMin, xMax, Nbins)
    if bounded:
        if method != None:
            kernel = bounded_1d_kde(data, xlow=xMin, xhigh=xMax, method=method)
        else:
            kernel = bounded_1d_kde(data, xlow=xMin, xhigh=xMax)

        # Manually apply smoothing using set_bandwidth (smooth parameter doesn't work)
        if smooth is not None and smooth != 1.0:
            kernel.set_bandwidth(kernel.factor * smooth)

        return kernel(xPoints)
    else:
        print('gaussian kde')
        # Apply bandwidth adjustment for unbounded KDE
        kernel = scipy.stats.gaussian_kde(data)
        if smooth is not None:
            kernel.set_bandwidth(kernel.factor * smooth)
        return kernel.evaluate(xPoints) / kernel.integrate_box_1d(xMin, xMax)

def create_prior_samps(priorpts_m1,priorpts_m2,nsamp):
    
    m1_draws = np.random.choice(priorpts_m1, nsamp, replace=True)

    m2_draws = np.random.choice(priorpts_m2, nsamp, replace=True)

    m1_samps = []
    m2_samps = []

    for jj in range(0,len(m1_draws)):
    
        if m1_draws[jj] >= m2_draws[jj]:
            m1_samps.append(m1_draws[jj])
            m2_samps.append(m2_draws[jj])
        else:
            m1_samps.append(m2_draws[jj])
            m2_samps.append(m1_draws[jj])
    
    return np.array(m1_samps), np.array(m2_samps)

def compute_js(p1, p2, eps=1e-15):
    """Compute Jensen-Shannon divergence between two distributions."""
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()

    p_m = 0.5 * (p1 + p2)
    jsd = 0.5 * (entropy(p1, p_m) + entropy(p2, p_m))

    jsd_spy = jensenshannon(p1, p2)

    return np.array([jsd, jsd_spy*jsd_spy])


def generate_prior_samples(prior_config, nsamp=OUTPUT_SAMPLE_SIZE, prior_pts_size=PRIOR_SAMPLE_SIZE):
    """
    Generate prior samples for a given configuration.

    Parameters
    ----------
    prior_config : dict
        Configuration dictionary with keys:
        - 'type': 'uniform', 'gaussian', 'double_gaussian', 'default', or 'mchirp_uniform'
        - other keys depend on the type
    nsamp : int
        Number of output samples
    prior_pts_size : int
        Number of points to draw initially

    Returns
    -------
    m1_samps, m2_samps : tuple of arrays
        Component mass samples
    """
    prior_type = prior_config['type']

    if prior_type == 'uniform':
        m1_prior = bilby.core.prior.analytical.Uniform(minimum=1.0, maximum=3.0)
        m2_prior = bilby.core.prior.analytical.Uniform(minimum=1.0, maximum=3.0)
        m1_pts = np.array(m1_prior.sample(prior_pts_size))
        m2_pts = np.array(m2_prior.sample(prior_pts_size))

    elif prior_type == 'gaussian':
        m1_prior = bilby.core.prior.analytical.Gaussian(mu=1.33, sigma=0.09)
        m2_prior = bilby.core.prior.analytical.Gaussian(mu=1.33, sigma=0.09)
        m1_pts = np.array(m1_prior.sample(prior_pts_size))
        m2_pts = np.array(m2_prior.sample(prior_pts_size))

    elif prior_type == 'double_gaussian':
        m1_prior = DoubleGaussian(mu1=1.34, mu2=1.80, sigma1=0.07, sigma2=0.21, w=0.65)
        m2_prior = DoubleGaussian(mu1=1.34, mu2=1.80, sigma1=0.07, sigma2=0.21, w=0.65)
        m1_pts = np.array(m1_prior.sample(prior_pts_size))
        m2_pts = np.array(m2_prior.sample(prior_pts_size))

    elif prior_type == 'default':
        prior_chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(
            name='chirp_mass', minimum=1.29, maximum=1.32)
        prior_mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(
            name='mass_ratio', minimum=0.125, maximum=1)

        mchirp_samps_det = np.array(prior_chirp_mass.sample(prior_pts_size))
        q_samps = np.array(prior_mass_ratio.sample(prior_pts_size))

        distL = 171.27056031500612
        z = luminosity_distance_to_redshift(distL, cosmology=Planck18)
        mchirp_samps = mchirp_samps_det / (1. + z)

        m1_pts, m2_pts = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
            mchirp_samps, q_samps)

    elif prior_type == 'mchirp_uniform':
        prior_chirp_mass = bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.29, maximum=1.32)
        prior_mass_ratio = bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1)

        mchirp_samps_det = np.array(prior_chirp_mass.sample(prior_pts_size))
        q_samps = np.array(prior_mass_ratio.sample(prior_pts_size))

        distL = 171.27056031500612
        z = luminosity_distance_to_redshift(distL, cosmology=Planck18)
        mchirp_samps = mchirp_samps_det / (1. + z)

        m1_pts, m2_pts = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
            mchirp_samps, q_samps)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    return create_prior_samps(m1_pts, m2_pts, nsamp)


def load_posterior_data(data_dir='../posteriors/data/', run_names=None, nsamp=OUTPUT_SAMPLE_SIZE):
    """
    Load posterior data from NPZ files.

    Parameters
    ----------
    data_dir : str
        Directory containing NPZ files
    run_names : list of str, optional
        List of run names to load. If None, uses default runs.
    nsamp : int
        Number of samples to draw from each posterior

    Returns
    -------
    dict
        Dictionary with keys as run names and values as dicts containing m1_data and m2_data arrays
    """
    data_dir = Path(data_dir)

    if run_names is None:
        # Default runs matching the original script
        run_names = [
            'prod_BW_XP_s005_l5000_double_gaussian',
            'prod_BW_XP_s005_l5000_gaussian',
            'prod_BW_XP_s005_l5000_uniform',
            'prod_BW_XP_s005_l5000_default'
        ]

    posterior_data = {}

    for run_name in run_names:
        file_path = data_dir / f"{run_name}.npz"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        data = np.load(file_path)
        m1_data = np.random.choice(data['mass_1_source'], nsamp, replace=True)
        m2_data = np.random.choice(data['mass_2_source'], nsamp, replace=True)

        posterior_data[run_name] = {
            'm1': np.array(m1_data),
            'm2': np.array(m2_data)
        }

    return posterior_data


def compute_kdes_batch(data_dict, Nbins=KDE_NBINS, bounded=True, method=KDE_METHOD, smooth=KDE_SMOOTH):
    """
    Compute KDEs for multiple datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary with dataset names as keys and data arrays as values
    Nbins : int
        Number of bins for KDE
    bounded : bool
        Whether to use bounded KDE
    method : str
        KDE method to use
    smooth : float
        Smoothing factor for KDE

    Returns
    -------
    dict
        Dictionary with same keys as input, containing KDE evaluations and x-ranges
    """
    kde_dict = {}

    for name, data in data_dict.items():
        xMin, xMax = min(data), max(data)

        # Use consistent method for all distributions
        kde = compute_kde(data, Nbins=Nbins, xMin=xMin, xMax=xMax,
                         bounded=bounded, method=method, smooth=smooth)

        kde_dict[name] = {
            'kde': kde,
            'xMin': xMin,
            'xMax': xMax,
            'x': np.linspace(xMin, xMax, Nbins)
        }

    return kde_dict
    

def main():
    """Main execution function."""

    # =============================================================================
    # Generate prior samples
    # =============================================================================
    print("Generating prior samples...")

    prior_configs = {
        'uniform': {'type': 'uniform'},
        'gaussian': {'type': 'gaussian'},
        'double_gaussian': {'type': 'double_gaussian'},
        'default': {'type': 'default'},
        'mchirp_uniform': {'type': 'mchirp_uniform'}
    }

    prior_samples = {}
    for name, config in prior_configs.items():
        m1, m2 = generate_prior_samples(config)
        prior_samples[name] = {'m1': m1, 'm2': m2}
        print(f"  Generated {name} prior samples")


    # =============================================================================
    # Load posterior data
    # =============================================================================
    print("\nLoading posterior data...")
    posterior_data = load_posterior_data()

    # Map posterior data to prior names for easier handling
    posterior_mapping = {
        'double_gaussian': 'prod_BW_XP_s005_l5000_double_gaussian',
        'gaussian': 'prod_BW_XP_s005_l5000_gaussian',
        'uniform': 'prod_BW_XP_s005_l5000_uniform',
        'default': 'prod_BW_XP_s005_l5000_default'
    }

    # =============================================================================
    # Compute KDEs
    # =============================================================================
    print("\nComputing KDEs for prior samples...")

    # Compute KDEs for priors
    m1_prior_data = {name: data['m1'] for name, data in prior_samples.items()}
    m2_prior_data = {name: data['m2'] for name, data in prior_samples.items()}

    m1_prior_kdes = compute_kdes_batch(m1_prior_data)
    m2_prior_kdes = compute_kdes_batch(m2_prior_data)

    print("Computing KDEs for posterior samples...")

    # Compute KDEs for posteriors
    m1_posterior_data = {}
    m2_posterior_data = {}
    for prior_name, post_name in posterior_mapping.items():
        if post_name in posterior_data:
            m1_posterior_data[prior_name] = posterior_data[post_name]['m1']
            m2_posterior_data[prior_name] = posterior_data[post_name]['m2']

    m1_posterior_kdes = compute_kdes_batch(m1_posterior_data)
    m2_posterior_kdes = compute_kdes_batch(m2_posterior_data)

    # =============================================================================
    # Create plots
    # =============================================================================
    print("\nCreating plots...")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Plot configuration - same colors for prior/posterior pairs
    colors = {
        'double_gaussian': 'palevioletred',
        'gaussian': 'dodgerblue',
        'uniform': 'darkgreen',
        'default': 'darkorange'
    }

    prior_config = {
        'double_gaussian': {'color': colors['double_gaussian'], 'linestyle': '--', 'linewidth': 2},
        'gaussian': {'color': colors['gaussian'], 'linestyle': '--', 'linewidth': 2},
        'uniform': {'color': colors['uniform'], 'linestyle': '--', 'linewidth': 2},
        'default': {'color': colors['default'], 'linestyle': '--', 'linewidth': 2}
    }

    posterior_config = {
        'double_gaussian': {'color': colors['double_gaussian'], 'linestyle': '-', 'linewidth': 2},
        'gaussian': {'color': colors['gaussian'], 'linestyle': '-', 'linewidth': 2},
        'uniform': {'color': colors['uniform'], 'linestyle': '-', 'linewidth': 2},
        'default': {'color': colors['default'], 'linestyle': '-', 'linewidth': 2}
    }

    # Plot m1 priors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_prior_kdes:
            kde_data = m1_prior_kdes[name]
            ax[0].plot(kde_data['x'], kde_data['kde'], **prior_config[name])

    # Plot m1 posteriors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_posterior_kdes:
            kde_data = m1_posterior_kdes[name]
            # Plot with markers to see evaluation points
            config = posterior_config[name].copy()
            config['marker'] = 'o'
            config['markersize'] = 3
            ax[0].plot(kde_data['x'], kde_data['kde'], **config)
            ax[0].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    ax[0].set_ylim(bottom=0)
    if M1_XLIM is not None:
        ax[0].set_xlim(M1_XLIM)
    ax[0].set_xlabel(r'$m_1 \, [M_\odot]$')
    ax[0].set_ylabel('Density')

    # Plot m2 priors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_prior_kdes:
            kde_data = m2_prior_kdes[name]
            ax[1].plot(kde_data['x'], kde_data['kde'], **prior_config[name])

    # Plot m2 posteriors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_posterior_kdes:
            kde_data = m2_posterior_kdes[name]
            # Plot with markers to see evaluation points
            config = posterior_config[name].copy()
            config['marker'] = 'o'
            config['markersize'] = 3
            ax[1].plot(kde_data['x'], kde_data['kde'], **config)
            ax[1].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    # ax[1].grid(alpha=0.5, linestyle='--')
    ax[1].set_ylim(bottom=0)
    if M2_XLIM is not None:
        ax[1].set_xlim(M2_XLIM)
    ax[1].set_xlabel(r'$m_2 \, [M_\odot]$')

    # Create single unified legend
    # Line style indicators (in black)
    style_prior = Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Prior')
    style_posterior = Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Posterior')

    # Color legend entries (for different prior types)
    color_dg = Line2D([0], [0], color=colors['double_gaussian'], linewidth=2, label='Double Gauss')
    color_g = Line2D([0], [0], color=colors['gaussian'], linewidth=2, label='Gaussian')
    color_u = Line2D([0], [0], color=colors['uniform'], linewidth=2, label='Uniform')
    color_d = Line2D([0], [0], color=colors['default'], linewidth=2, label='Default')

    # Combine all handles
    all_handles = [style_prior, style_posterior, color_dg, color_g, color_u, color_d]

    # Add single legend spanning both plots
    fig.legend(handles=all_handles,
              loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=6,
              frameon=True, fontsize=10)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved to: {OUTPUT_PATH}")

    # =============================================================================
    # Compute and print Jensen-Shannon divergences
    # =============================================================================
    print("\nComputing Jensen-Shannon divergences...")

    dist_labels = ['Double Gaussian', 'Gaussian', 'Uniform', 'Default']
    dist_names = ['double_gaussian', 'gaussian', 'uniform', 'default']

    print('\n======== mass 1 JSD ========')
    for jj, name in enumerate(dist_names):
        if name in m1_prior_kdes and name in m1_posterior_kdes:
            prior_kde = m1_prior_kdes[name]['kde']
            post_kde = m1_posterior_kdes[name]['kde']
            jsd_val = compute_js(post_kde, prior_kde)
            print(f"{dist_labels[jj]:20s} === JSD entropy: {jsd_val[0]:.6f}  JSD scipy: {jsd_val[1]:.6f}")

    print('\n======== mass 2 JSD ========')
    for jj, name in enumerate(dist_names):
        if name in m2_prior_kdes and name in m2_posterior_kdes:
            prior_kde = m2_prior_kdes[name]['kde']
            post_kde = m2_posterior_kdes[name]['kde']
            jsd_val = compute_js(post_kde, prior_kde)
            print(f"{dist_labels[jj]:20s} === JSD entropy: {jsd_val[0]:.6f}  JSD scipy: {jsd_val[1]:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

