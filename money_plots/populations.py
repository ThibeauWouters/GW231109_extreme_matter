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

# Ranges used for the plotting (zoom in on the posterior to make it a bit nicer)
M1_XLIM = (1.2, 1.8)
M2_XLIM = (1.1, 1.8)

# Sample sizes (increase these for smoother KDEs)
PRIOR_SAMPLE_SIZE = 10_000   # Number of initial samples to draw from priors
OUTPUT_SAMPLE_SIZE = 10_000  # Number of output samples after m1 >= m2 constraint

# KDE parameters
KDE_NBINS = 1_000             # Number of points for KDE evaluation (higher = smoother curves)
KDE_PLOT_NBINS = 1_000        # Number of points for plotting KDEs (finer for smoother plots)
KDE_SMOOTH = 2.0              # Smoothing factor (None = auto, or set to float like 1.5 for more smoothing)
# KDE_SMOOTH = None

# Prior domain configuration
PRIOR_DOMAIN_WIDE = (0.10, 6.0)  # Wide domain for KDE evaluation

# Output configuration
OUTPUT_PATH = './figures/populations/populations_component_masses_comparison.pdf'
OUTPUT_PATH_WIDE = './figures/populations/populations_component_masses_comparison_wide.pdf'
DPI = 600

FIGSIZE = (6, 6)
AXIS_LABEL_FONTSIZE = 18

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

def create_prior_samps(priorpts_m1: np.array, priorpts_m2: np.array, nsamp: int):
    """
    Create prior samples ensuring m1 >= m2 and desired number of samples.

    Args:
        priorpts_m1 (np.array): Original m1 prior points
        priorpts_m2 (np.array): Original m2 prior points
        nsamp (in): Number of output samples to draw.

    Returns:
        m1_samps, m2_samps (np.array, np.array): Component mass samples with m1 >= m2
    """
    
    m1_draws = np.random.choice(priorpts_m1, nsamp, replace=False)
    m2_draws = np.random.choice(priorpts_m2, nsamp, replace=False)

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

def compute_js(p1, p2):
    """Compute Jensen-Shannon divergence between two distributions."""
    # Add small epsilon to avoid numerical issues with zeros
    epsilon = 1e-10
    p1 = p1 + epsilon
    p2 = p2 + epsilon

    # Normalize after adding epsilon
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()

    p_m = 0.5 * (p1 + p2)
    jsd = 0.5 * (entropy(p1, p_m, base=2) + entropy(p2, p_m, base=2))

    jsd_spy = jensenshannon(p1, p2, base=2)

    return np.array([jsd, jsd_spy*jsd_spy])


def generate_jsd_latex_table(m1_jsds, m2_jsds, output_file='./JSD_tabular.tex'):
    """
    Generate a LaTeX table of Jensen-Shannon divergences.

    Parameters
    ----------
    m1_jsds : dict of dict
        m1_jsds[posterior_name][prior_name] = (entropy_jsd, scipy_jsd) for m1
    m2_jsds : dict of dict
        m2_jsds[posterior_name][prior_name] = (entropy_jsd, scipy_jsd) for m2
    output_file : str
        Path to output LaTeX file
    """
    dist_labels = {
        'default': 'Default',
        'double_gaussian': 'Double Gaussian',
        'gaussian': 'Gaussian',
        'uniform': 'Uniform'
    }

    dist_order = ['default', 'double_gaussian', 'gaussian', 'uniform']

    # Start building the table
    lines = []
    # Tabular spec: c column for multirow label, l for row labels, then data columns
    lines.append(r'\begin{tabular}{c @{}l@{} cccc c||c cccc}')
    lines.append(r'\hline\hline')
    lines.append(r' & & \multicolumn{4}{c}{$m_1$ \textsc{Prior}} & & & \multicolumn{4}{c}{$m_2$ \textsc{Prior}} \\')
    lines.append(r'\hline')

    # Column headers
    header = r' &'  # Empty cells for multirow column and row label column
    for name in dist_order:
        header += f' & {dist_labels[name]}'
    header += ' & &'  # separator columns
    for name in dist_order:
        header += f' & {dist_labels[name]}'
    lines.append(header + r' \\')
    lines.append(r'\hline\hline')

    # Data rows
    for idx, post_name in enumerate(dist_order):
        # First column: vertical "Posterior" label on first row, empty otherwise
        if idx == 0:
            row = r'\multirow{4}{*}{\rotatebox[origin=c]{90}{\parbox[c][8mm][c]{2cm}{\centering \textsc{Posterior}}}}'
        else:
            row = ''

        # Row label
        row += r' & \multicolumn{1}{|l}{' + dist_labels[post_name] + r'}'

        # Find minimum JSD for m1 and m2 in this row
        m1_vals = {prior_name: m1_jsds[post_name][prior_name][1] for prior_name in dist_order}
        m2_vals = {prior_name: m2_jsds[post_name][prior_name][1] for prior_name in dist_order}

        min_m1_prior = min(m1_vals, key=m1_vals.get)
        min_m2_prior = min(m2_vals, key=m2_vals.get)

        # m1 JSDs (using scipy version)
        for prior_name in dist_order:
            jsd_val = m1_jsds[post_name][prior_name][1]  # scipy JSD
            if prior_name == min_m1_prior:
                row += f' & \\textbf{{{jsd_val:.2f}}}'
            else:
                row += f' & {jsd_val:.2f}'

        row += ' & &'  # separator columns

        # m2 JSDs (using scipy version)
        for prior_name in dist_order:
            jsd_val = m2_jsds[post_name][prior_name][1]  # scipy JSD
            if prior_name == min_m2_prior:
                row += f' & \\textbf{{{jsd_val:.2f}}}'
            else:
                row += f' & {jsd_val:.2f}'

        lines.append(row + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  LaTeX table saved to: {output_file}")


def generate_prior_samples(prior_config, nsamp=OUTPUT_SAMPLE_SIZE, prior_pts_size=PRIOR_SAMPLE_SIZE):
    """
    Generate prior samples for a given configuration.

    Parameters
    ----------
    prior_config : dict
        Configuration dictionary with keys:
        - 'type': 'uniform', 'gaussian', 'double_gaussian', or 'default'
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
        
        # Taken from our bilby config files
        prior_chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(
            name='chirp_mass', minimum=1.29, maximum=1.32)
        prior_mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(
            name='mass_ratio', minimum=0.125, maximum=1)
        luminosity_distance = bilby.gw.prior.UniformSourceFrame(
            name='luminosity_distance', minimum=1.0, maximum=1000.0, unit='Mpc')

        mchirp_samps_det = np.array(prior_chirp_mass.sample(prior_pts_size))
        q_samps = np.array(prior_mass_ratio.sample(prior_pts_size))
        
        # This can be a a bit slow so print status to check on that
        print(f"Sampling {prior_pts_size} points from luminosity distance prior...")
        distL = np.array(luminosity_distance.sample(prior_pts_size))
        z = luminosity_distance_to_redshift(distL)
        print(f"Sampling {prior_pts_size} points from luminosity distance prior... DONE")
        
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
        # Default run names
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


def compute_kdes_batch(data_dict, Nbins=KDE_NBINS, smooth=KDE_SMOOTH, prior_domains=(1.0, 4.0), use_method_selection=False):
    """
    Compute KDEs for multiple datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary with dataset names as keys and data arrays as values
    Nbins : int
        Number of bins for KDE
    smooth : float
        Smoothing factor for KDE
    prior_domains : dict or None
        Dictionary mapping names to (xMin, xMax) tuples for theoretical prior domains
        If None, uses min/max of data
    use_method_selection : bool
        If True, use Transform for uniform/default and Reflection for gaussian/double_gaussian
        If False, use Reflection for all (default)

    Returns
    -------
    dict
        Dictionary with same keys as input, containing KDE evaluations and x-ranges
    """
    kde_dict = {}

    for name, data in data_dict.items():
        # Use theoretical domain if provided, otherwise use data range
        xMin, xMax = prior_domains

        # Determine KDE method based on distribution type (only if use_method_selection=True)
        # Transform for default only, Reflection for gaussian, double_gaussian, and uniform
        if use_method_selection and name in ['default']:
            kde_method = 'Reflection'
            # kde_method = 'Transform' # NOTE: This seems to break things
        else:
            kde_method = 'Reflection'

        # Store the raw KDE object for later evaluation
        # kernel = scipy.stats.gaussian_kde(data) # NOTE: PESummary version is preferred!
        kernel = bounded_1d_kde(data, xlow=xMin, xhigh=xMax, method=kde_method)
        if smooth is not None:
            kernel.set_bandwidth(kernel.factor * smooth)

        kde = kernel.evaluate(np.linspace(xMin, xMax, Nbins))
        kde_dict[name] = {
            'kde': kde,        # Store the evaluated KDE values, i.e. y(x)
            'kernel': kernel,  # Store the kernel object (i.e. the KDE object)
            'data': data,      # Store original data
            'xMin': xMin,
            'xMax': xMax,
            'x': np.linspace(xMin, xMax, Nbins)
        }

    return kde_dict


def compute_js_on_common_grid(kde_dict1, kde_dict2, Nbins=KDE_NBINS):
    """
    Compute JSD between two KDEs evaluated on a common grid.

    Parameters
    ----------
    kde_dict1 : dict
        Dictionary containing 'kernel', 'xMin', 'xMax' from compute_kdes_batch
    kde_dict2 : dict
        Dictionary containing 'kernel', 'xMin', 'xMax' from compute_kdes_batch
    Nbins : int
        Number of bins for evaluation

    Returns
    -------
    array
        [entropy_jsd, scipy_jsd^2]
    """
    # Determine common grid spanning both distributions
    xMin = min(kde_dict1['xMin'], kde_dict2['xMin'])
    xMax = max(kde_dict1['xMax'], kde_dict2['xMax'])

    x_common = np.linspace(xMin, xMax, Nbins)
    print(f"JSD is going to be computed on grid from {xMin:.3f} to {xMax:.3f}")

    # Evaluate both kernels on common grid
    p1 = kde_dict1['kernel'].evaluate(x_common)
    p2 = kde_dict2['kernel'].evaluate(x_common)

    return compute_js(p1, p2)




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

    # Compute KDEs for priors with wide theoretical domain
    m1_prior_data = {name: data['m1'] for name, data in prior_samples.items()}
    m2_prior_data = {name: data['m2'] for name, data in prior_samples.items()}

    m1_prior_kdes = compute_kdes_batch(m1_prior_data, prior_domains=PRIOR_DOMAIN_WIDE, use_method_selection=True)
    m2_prior_kdes = compute_kdes_batch(m2_prior_data, prior_domains=PRIOR_DOMAIN_WIDE, use_method_selection=True)

    print("Computing KDEs for posterior samples...")

    # Compute KDEs for posteriors with wide domain
    m1_posterior_data = {}
    m2_posterior_data = {}
    for prior_name, post_name in posterior_mapping.items():
        if post_name in posterior_data:
            m1_posterior_data[prior_name] = posterior_data[post_name]['m1']
            m2_posterior_data[prior_name] = posterior_data[post_name]['m2']

    m1_posterior_kdes = compute_kdes_batch(m1_posterior_data, prior_domains=PRIOR_DOMAIN_WIDE)
    m2_posterior_kdes = compute_kdes_batch(m2_posterior_data, prior_domains=PRIOR_DOMAIN_WIDE)

    # =============================================================================
    # Compute NEW KDEs on zoomed domain for plotting
    # =============================================================================
    print("Computing new KDEs on zoomed domain for plotting...")

    # Determine zoom domain from M1_XLIM and M2_XLIM
    if M1_XLIM is not None and M2_XLIM is not None:
        xlim_min = min(M1_XLIM[0], M2_XLIM[0])
        xlim_max = max(M1_XLIM[1], M2_XLIM[1])
        zoom_domain = (xlim_min, xlim_max)
    elif M1_XLIM is not None:
        zoom_domain = M1_XLIM
    elif M2_XLIM is not None:
        zoom_domain = M2_XLIM
    else:
        # Fallback to wide domain if no zoom specified
        zoom_domain = PRIOR_DOMAIN_WIDE

    print(f"  Zoom domain: {zoom_domain[0]:.2f} to {zoom_domain[1]:.2f} solar masses")

    # Compute NEW KDEs fitted on the zoom domain with finer grid
    m1_prior_kdes_plot = compute_kdes_batch(m1_prior_data, Nbins=KDE_PLOT_NBINS, prior_domains=zoom_domain, use_method_selection=True)
    m2_prior_kdes_plot = compute_kdes_batch(m2_prior_data, Nbins=KDE_PLOT_NBINS, prior_domains=zoom_domain, use_method_selection=True)
    m1_posterior_kdes_plot = compute_kdes_batch(m1_posterior_data, Nbins=KDE_PLOT_NBINS, prior_domains=zoom_domain)
    m2_posterior_kdes_plot = compute_kdes_batch(m2_posterior_data, Nbins=KDE_PLOT_NBINS, prior_domains=zoom_domain)

    # =============================================================================
    # Create plots - Wide domain version
    # =============================================================================
    print("\nCreating wide domain plot...")

    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # Plot configuration - same colors for prior/posterior pairs
    colors = {
        'double_gaussian': 'palevioletred',
        'gaussian': 'dodgerblue',
        'uniform': 'darkgreen',
        'default': 'darkorange'
    }

    # Rescale factors for prior KDEs (to make them more visible)
    m1_prior_rescale = {
        'double_gaussian': 1.0,
        'gaussian': 1.0,
        'uniform': 1.0,
        'default': 1.0
    }

    m2_prior_rescale = {
        'double_gaussian': 1.0,
        'gaussian': 1.0,
        'uniform': 1.0,
        'default': 1.0
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

    # Plot m1 prior histograms first (so they're in the background)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_prior_kdes:
            kde_data = m1_prior_kdes[name]
            ax[0].hist(kde_data['data'], bins=50, density=True,
                      color=colors[name], alpha=0.15, range=PRIOR_DOMAIN_WIDE,
                      edgecolor=colors[name], linewidth=0.5)

    # Plot m1 priors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_prior_kdes:
            kde_data = m1_prior_kdes[name]
            ax[0].plot(kde_data['x'], kde_data['kde'] * m1_prior_rescale[name], **prior_config[name])

    # Plot m1 posterior histograms
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_posterior_kdes:
            kde_data = m1_posterior_kdes[name]
            ax[0].hist(kde_data['data'], bins=50, density=True,
                      color=colors[name], alpha=0.15, range=PRIOR_DOMAIN_WIDE,
                      edgecolor=colors[name], linewidth=0.5)

    # Plot m1 posteriors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_posterior_kdes:
            kde_data = m1_posterior_kdes[name]
            ax[0].plot(kde_data['x'], kde_data['kde'], **posterior_config[name])
            ax[0].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r'$m_1$ prob. density', fontsize=AXIS_LABEL_FONTSIZE)

    # Plot m2 prior histograms first (so they're in the background)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_prior_kdes:
            kde_data = m2_prior_kdes[name]
            ax[1].hist(kde_data['data'], bins=50, density=True,
                      color=colors[name], alpha=0.15, range=PRIOR_DOMAIN_WIDE,
                      edgecolor=colors[name], linewidth=0.5)

    # Plot m2 priors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_prior_kdes:
            kde_data = m2_prior_kdes[name]
            ax[1].plot(kde_data['x'], kde_data['kde'] * m2_prior_rescale[name], **prior_config[name])

    # Plot m2 posterior histograms
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_posterior_kdes:
            kde_data = m2_posterior_kdes[name]
            ax[1].hist(kde_data['data'], bins=50, density=True,
                      color=colors[name], alpha=0.15, range=PRIOR_DOMAIN_WIDE,
                      edgecolor=colors[name], linewidth=0.5)

    # Plot m2 posteriors
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_posterior_kdes:
            kde_data = m2_posterior_kdes[name]
            ax[1].plot(kde_data['x'], kde_data['kde'], **posterior_config[name])
            ax[1].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel(r'$m_2$ prob. density', fontsize=AXIS_LABEL_FONTSIZE)
    ax[1].set_xlabel(r'Mass $[M_\odot]$', fontsize=AXIS_LABEL_FONTSIZE)

    # Set shared x-limits to wide domain
    ax[1].set_xlim(PRIOR_DOMAIN_WIDE)

    # Two-row legend: first row for line styles, second row for colors
    # First row: line style indicators
    style_prior = Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Prior')
    style_posterior = Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Posterior')

    # Second row: color legend entries
    color_dg = Line2D([0], [0], color=colors['double_gaussian'], linewidth=3, label='Double Gaussian')
    color_g = Line2D([0], [0], color=colors['gaussian'], linewidth=3, label='Gaussian')
    color_u = Line2D([0], [0], color=colors['uniform'], linewidth=3, label='Uniform')
    color_d = Line2D([0], [0], color=colors['default'], linewidth=3, label='Default')

    # Create the handles
    all_handles = [style_prior, style_posterior, color_dg, color_g, color_u, color_d]

    # Make the legend
    fig.legend(handles=all_handles,
              loc='upper center', bbox_to_anchor=(0.5, 1.10), ncols=3,
              frameon=True, fontsize=14, columnspacing=1.25)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.05)
    plt.savefig(OUTPUT_PATH_WIDE, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Wide domain plot saved to: {OUTPUT_PATH_WIDE}")

    # =============================================================================
    # Create plots - Zoomed version (using finer KDE evaluations)
    # =============================================================================
    print("\nCreating zoomed plot...")

    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # Plot m1 priors (using fine grid)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_prior_kdes_plot:
            kde_data = m1_prior_kdes_plot[name]
            ax[0].plot(kde_data['x'], kde_data['kde'] * m1_prior_rescale[name], **prior_config[name])

    # Plot m1 posteriors (using fine grid)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m1_posterior_kdes_plot:
            kde_data = m1_posterior_kdes_plot[name]
            ax[0].plot(kde_data['x'], kde_data['kde'], **posterior_config[name])
            ax[0].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r'$m_1$ prob. density', fontsize=AXIS_LABEL_FONTSIZE)

    # Plot m2 priors (using fine grid)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_prior_kdes_plot:
            kde_data = m2_prior_kdes_plot[name]
            ax[1].plot(kde_data['x'], kde_data['kde'] * m2_prior_rescale[name], **prior_config[name])

    # Plot m2 posteriors (using fine grid)
    for name in ['double_gaussian', 'gaussian', 'uniform', 'default']:
        if name in m2_posterior_kdes_plot:
            kde_data = m2_posterior_kdes_plot[name]
            ax[1].plot(kde_data['x'], kde_data['kde'], **posterior_config[name])
            ax[1].fill_between(kde_data['x'], kde_data['kde'],
                              color=colors[name], alpha=0.3)

    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel(r'$m_2$ prob. density', fontsize=AXIS_LABEL_FONTSIZE)
    ax[1].set_xlabel(r'Mass $[M_\odot]$', fontsize=AXIS_LABEL_FONTSIZE)

    # Set shared x-limits - use the wider of M1_XLIM and M2_XLIM
    if M1_XLIM is not None and M2_XLIM is not None:
        xlim_min = min(M1_XLIM[0], M2_XLIM[0])
        xlim_max = max(M1_XLIM[1], M2_XLIM[1])
        ax[1].set_xlim(xlim_min, xlim_max)
    elif M1_XLIM is not None:
        ax[1].set_xlim(M1_XLIM)
    elif M2_XLIM is not None:
        ax[1].set_xlim(M2_XLIM)

    # Add same legend (all_handles already defined from wide plot creation)
    fig.legend(handles=all_handles,
              loc='upper center', bbox_to_anchor=(0.5, 1.08), ncols=3,
              frameon=True, fontsize=14, columnspacing=1.25)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.05)
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Zoomed plot saved to: {OUTPUT_PATH}")

    # =============================================================================
    # Compute and print Jensen-Shannon divergences
    # =============================================================================
    print("\nComputing Jensen-Shannon divergences...")

    dist_labels = ['Double Gaussian', 'Gaussian', 'Uniform', 'Default']
    dist_names = ['double_gaussian', 'gaussian', 'uniform', 'default']

    # Store JSD values for LaTeX table generation - full matrix
    m1_jsds = {}
    m2_jsds = {}

    print('\n======== mass 1 JSD ========')
    # Compute JSD for each posterior against all priors
    for post_idx, post_name in enumerate(dist_names):
        m1_jsds[post_name] = {}

        print(f"\n{dist_labels[post_idx]} posterior vs:")
        for prior_idx, prior_name in enumerate(dist_names):
            jsd_val = compute_js_on_common_grid(
                m1_posterior_kdes[post_name],
                m1_prior_kdes[prior_name]
            )
            m1_jsds[post_name][prior_name] = jsd_val
            print(f"  {dist_labels[prior_idx]:20s} === JSD entropy: {jsd_val[0]:.6f}  JSD scipy: {jsd_val[1]:.6f}")

    print('\n======== mass 2 JSD ========')
    # Compute JSD for each posterior against all priors
    for post_idx, post_name in enumerate(dist_names):
        m2_jsds[post_name] = {}

        print(f"\n{dist_labels[post_idx]} posterior vs:")
        for prior_idx, prior_name in enumerate(dist_names):
            if prior_name not in m2_prior_kdes:
                continue

            jsd_val = compute_js_on_common_grid(
                m2_posterior_kdes[post_name],
                m2_prior_kdes[prior_name]
            )
            m2_jsds[post_name][prior_name] = jsd_val
            print(f"  {dist_labels[prior_idx]:20s} === JSD entropy: {jsd_val[0]:.6f}  JSD scipy: {jsd_val[1]:.6f}")

    # =============================================================================
    # Generate LaTeX table
    # =============================================================================
    print("\nGenerating LaTeX table...")
    generate_jsd_latex_table(m1_jsds, m2_jsds, output_file='./JSD_tabular.tex')

    print("\nDone!")


if __name__ == "__main__":
    main()