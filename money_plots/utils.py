"""
Utility functions for the final money plots.
This module extracts common functionality from the existing plotting scripts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import json
import corner
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.conversion import luminosity_distance_to_redshift
from bilby.core.prior import PriorDict

import bilby
import scipy

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from bilby.gw.conversion import luminosity_distance_to_redshift
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
bilby.core.utils.random.seed(123456)
np.random.seed(42)

# Add parent directory to path to import main utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils as main_utils

# Matplotlib configuration
PLOT_PARAMS = {
    "axes.grid": False,
    "text.usetex": False,
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

plt.rcParams.update(PLOT_PARAMS)

# Corner plot defaults
DEFAULT_CORNER_KWARGS = dict(
    bins=40,
    smooth=1.,
    show_titles=False,
    label_kwargs=dict(fontsize=26),
    title_kwargs=dict(fontsize=16),
    labelpad=0.175,
    color="blue",
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=4,
    min_n_ticks=3,
    truth_color="red",
    save=False
)

# Color scheme
GW231109_COLOR = "red"
GW190425_COLOR = "purple"
PRIOR_COLOR = "gray"
GW170817_COLOR = "orange"
INJECTION_COLOR = "#cb78bd"

# Standard comparison colors
ORANGE = "#de8f07"
BLUE = "#0472b1"
GREEN = "#019e72"

def identify_person_from_path(source_dir: str) -> str:
    """
    Identify the person who ran the analysis from the source directory path.

    Args:
        source_dir (str): Directory path containing the run

    Returns:
        str: Person identifier
    """
    if "puecher" in source_dir:
        return "anna"
    elif "dietrich6" in source_dir:
        return "tim"
    elif "wouters" in source_dir:
        return "thibeau"
    else:
        raise ValueError(f"Cannot identify person from source_dir: {source_dir}")

def generate_save_path(source_dir: str,
                      plot_type: str,
                      prefix: str = "",
                      suffix: str = ".pdf") -> str:
    """
    Generate standardized save path for plots.

    Args:
        source_dir (str): Source directory path
        plot_type (str): Type of plot (e.g., 'cornerplots', 'mass_Lambdas_figs')
        prefix (str): Optional prefix for filename
        suffix (str): File extension

    Returns:
        str: Full save path
    """
    person = identify_person_from_path(source_dir)
    run_name = source_dir.split("/")[-1]

    filename = f"{person}_{run_name}"
    if prefix:
        filename = f"{prefix}_{filename}"
    filename += suffix

    save_path = os.path.join(".", plot_type, filename)

    return save_path

def fetch_posterior_filename(source_dir: str):
    """
    Given a source_dir where a GW inference run was performed in /work/, return the path to the posterior HDF5 file if it exists,
    otherwise return None.

    Args:
        source_dir (str): Directory containing the run output. NOTE: It should have
                          outdir/final_result/ with an HDF5 posterior file.

    Returns:
        str or None: Path to the posterior file, or None if not found.
    """

    source_dir = os.path.abspath(source_dir)
    final_results_dir = os.path.join(source_dir, "outdir/final_result")

    if not os.path.exists(final_results_dir):
        return None

    posterior_files = [
        f for f in os.listdir(final_results_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    ]

    if not posterior_files:
        print(f"WARNING: No HDF5 files found in {final_results_dir}. Trying to look for JSON files now")
        final_results_dir = os.path.join(source_dir, "outdir")
        posterior_files = [
            f for f in os.listdir(final_results_dir)
            if f.endswith(".json")
        ]
        if not posterior_files:
            return None

    # Just return the first one found, there should be only one anyways
    return os.path.join(final_results_dir, posterior_files[0])

def load_posterior_samples(source_dir: str,
                          keys_to_fetch: list[str],
                          chirp_tilde: bool = False) -> tuple:
    """
    Load posterior samples from HDF5 or JSON files.

    Args:
        source_dir (str): Directory containing posterior files
        keys_to_fetch (list[str]): Parameters to extract
        chirp_tilde (bool): Whether to return chirp mass and lambda_tilde format

    Returns:
        tuple: Posterior samples as numpy arrays
    """
    posterior_file = fetch_posterior_filename(source_dir)
    if posterior_file is None:
        raise FileNotFoundError(f"No posterior file found in {source_dir}")

    if chirp_tilde:
        return main_utils.fetch_mass_Lambdas_samples(posterior_file, chirp_tilde=True)
    else:
        # Load specific keys
        if posterior_file.endswith(('.h5', '.hdf5')):
            with h5py.File(posterior_file, 'r') as f:
                posterior = f["posterior"]
                samples = np.array([posterior[key][()] for key in keys_to_fetch]).T
        else:  # JSON
            with open(posterior_file, 'r') as f:
                posterior = json.load(f)['posterior']['content']
                samples = np.array([posterior[key] for key in keys_to_fetch]).T

        return samples

def load_run_metadata(source_dir: str) -> dict:
    """
    Load run metadata including sampling time and Bayes factor.

    Args:
        source_dir (str): Directory containing the run

    Returns:
        dict: Metadata dictionary
    """
    posterior_file = fetch_posterior_filename(source_dir)
    metadata = {}

    if posterior_file and posterior_file.endswith(('.h5', '.hdf5')):
        with h5py.File(posterior_file, 'r') as f:
            if "log_bayes_factor" in f:
                metadata["log_bayes_factor"] = f["log_bayes_factor"][()]
            if "sampling_time" in f:
                metadata["sampling_time"] = f["sampling_time"][()]
                metadata["sampling_time_hrs"] = metadata["sampling_time"] / 3600.0

            # Load priors
            if "priors" in f:
                priors_bytes = f["priors"][()]
                priors_str = priors_bytes.decode()
                metadata["priors"] = json.loads(priors_str)

    return metadata

def load_priors_for_corner(source_dir: str) -> list[str]:
    """
    Load and process priors for corner plot parameter selection.

    Args:
        source_dir (str): Directory containing the run

    Returns:
        list[str]: List of parameter keys to plot
    """
    metadata = load_run_metadata(source_dir)
    priors_dict = metadata.get("priors", {})

    # Filter out unwanted keys
    prior_keys_to_skip = ['__prior_dict__', '__module__', '__name__']
    prior_keys = [k for k, v in priors_dict.items()
                  if k not in prior_keys_to_skip and "recalib" not in k]

    # Handle fixed parameters
    if "fixed_dL" in source_dir:
        prior_keys = [k for k in prior_keys if k != "luminosity_distance"]
    if "fixed_sky" in source_dir:
        prior_keys = [k for k in prior_keys if k not in ["ra", "dec"]]

    # Remove quasi-universal relation parameters
    keys_to_remove = ["lambda_symmetric", "lambda_antisymmetric", "binary_love_uniform"]
    prior_keys = [k for k in prior_keys if k not in keys_to_remove]

    return prior_keys

def load_eos_curves(eos_name: str, base_path: str = "../figures") -> tuple:
    """
    Load EOS curves from NPZ files.

    Args:
        eos_name (str): Name of the EOS
        base_path (str): Base path to EOS data directory

    Returns:
        tuple: (M, R, L, log_prob) arrays
    """
    filename = os.path.join(base_path, "EOS_data", f"{eos_name}.npz")
    data = np.load(filename)
    M, R, L = data['M'], data['R'], data['L']
    log_prob = data['log_prob']
    return M, R, L, log_prob

def get_mchirp_lambda_tilde_EOS(EOS_masses: np.array,
                               EOS_Lambdas: np.array,
                               mchirp_min: float,
                               mchirp_max: float) -> tuple[np.array, np.array]:
    """
    Convert EOS mass-Lambda curves to chirp mass - lambda_tilde space.

    Args:
        EOS_masses (np.array): Array of EOS masses
        EOS_Lambdas (np.array): Array of EOS Lambdas
        mchirp_min (float): Minimum chirp mass
        mchirp_max (float): Maximum chirp mass

    Returns:
        tuple: (mchirp_array, lambda_tilde) arrays
    """
    mchirp_array = np.linspace(mchirp_min, mchirp_max, 100)
    q_array = np.ones_like(mchirp_array)  # Assuming equal mass ratio

    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(mchirp_array, q_array)
    lambda_1 = np.interp(mass_1, EOS_masses, EOS_Lambdas)
    lambda_2 = np.interp(mass_2, EOS_masses, EOS_Lambdas)

    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    return mchirp_array, lambda_tilde

def load_GW170817_PE(chirp_tilde: bool = False, base_path: str = "../figures") -> tuple:
    """
    Load GW170817 posterior samples.

    Args:
        chirp_tilde (bool): Whether to return chirp mass format
        base_path (str): Base path to data directory

    Returns:
        tuple: Posterior sample arrays
    """
    filename = os.path.join(base_path, "EOS_data", "PE_posterior_samples_GW170817.npz")
    data = np.load(filename)

    if chirp_tilde:
        chirp_mass = data['chirp_mass_source']
        mass_ratio = data['mass_ratio']
        lambda_tilde = data['lambda_tilde']
        delta_lambda_tilde = data['delta_lambda_tilde']
        return chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde
    else:
        mass_1_source = data['mass_1_source']
        mass_2_source = data['mass_2_source']
        lambda_1 = data['lambda_1']
        lambda_2 = data['lambda_2']
        return mass_1_source, mass_2_source, lambda_1, lambda_2

def load_GW190425_PE(chirp_tilde: bool = False, base_path: str = "../figures") -> tuple:
    """
    Load GW190425 posterior samples.

    Args:
        chirp_tilde (bool): Whether to return chirp mass format
        base_path (str): Base path to data directory

    Returns:
        tuple: Posterior sample arrays
    """
    raise NotImplementedError("Need to obtain npz of posterior")

def fetch_prior_samples(source_dir: str, nb_samples: int = 10_000) -> tuple[np.array, np.array]:
    """
    Generate prior samples for chirp mass and lambda_tilde.

    Args:
        source_dir (str): Directory containing priors information
        nb_samples (int): Number of samples to generate

    Returns:
        tuple: (chirp_mass_source, lambda_tilde) arrays
    """
    priors = main_utils.initialize_priors(
        source_dir,
        ["chirp_mass", "mass_ratio", "lambda_1", "lambda_2", "luminosity_distance"]
    )
    prior = PriorDict(priors)

    samples = prior.sample(size=nb_samples)
    chirp_mass = samples["chirp_mass"]
    mass_ratio = samples["mass_ratio"]
    lambda_1 = samples["lambda_1"]
    lambda_2 = samples["lambda_2"]

    z = luminosity_distance_to_redshift(samples["luminosity_distance"])
    chirp_mass_source = chirp_mass / (1 + z)
    mass_1_source, mass_2_source = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass_source, mass_ratio
    )
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1_source, mass_2_source)

    return chirp_mass_source, lambda_tilde

def create_legend_elements(include_gw170817: bool = True,
                          include_prior: bool = False,
                          run_label: str = "GW231109") -> list:
    """
    Create legend elements for plots.

    Args:
        include_gw170817 (bool): Whether to include GW170817
        include_prior (bool): Whether to include prior
        run_label (str): Label for the main run

    Returns:
        list: Legend patch elements
    """
    elements = []

    if include_gw170817:
        elements.append(
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817')
        )

    # Determine color based on run label
    if run_label == "GW190425":
        color = GW190425_COLOR
    else:
        color = GW231109_COLOR

    elements.append(
        mpatches.Patch(facecolor=color, edgecolor='k', label=run_label)
    )

    if include_prior:
        elements.append(
            mpatches.Patch(facecolor=PRIOR_COLOR, edgecolor='k', label='Prior')
        )

    return elements

def ensure_directory_exists(filepath: str):
    """
    Ensure the directory for a filepath exists.

    Args:
        filepath (str): Full file path
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

import bilby
import scipy
from bilby.core.prior import Prior,Interped

class DoubleGaussian(Prior):
    
    def __init__(self, mu1, mu2, sigma1, sigma2, w, name=None, latex_label=None, unit=None, boundary=None):
        """Double Gaussian prior with means [mu1,mu2], 
        widths [sigma1,sigma2], and weight w 
        
        Parameters
        ==========
        mu1: float
            Mean of the first Gaussian
        mu2: float
            Mean of the second Gaussian
        sigma1:
            width of the first Gaussian
        sigma2:
            width of the second Gaussian
        w:
            weight of the two first Gaussians (the second one is 1-w)
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass          
        """
        
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.w = w
        
        super(DoubleGaussian,self).__init__(name=name,latex_label=latex_label, unit=unit, boundary=boundary)
        
        self._create_inverse()
        
    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        Parameters
        ==========
        val: Union[float, int, array_like]
    
        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled
    
        ###TOFIX: add interpolation rescaling
            
        
    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
            
        gauss1 = np.exp(-(self.mu1 - val) ** 2 / (2 * self.sigma1 ** 2)) / (2 * np.pi) ** 0.5 / self.sigma1
        gauss2 = np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 ** 2)) / (2 * np.pi) ** 0.5 / self.sigma2
            
        return self.w*gauss1 + (1 - self.w) * gauss2
        
    def ln_prob(self, val):
        """Return the Log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
            
        with np.errstate(divide='ignore'):
            return np.log(self.prob(val)) 

    def cdf(self, val):
        
        cdf_gauss1 = (1 - erf((self.mu1 - val) / 2 ** 0.5 / self.sigma1)) / 2
        cdf_gauss2 = (1 - erf((self.mu2 - val) / 2 ** 0.5 / self.sigma2)) / 2
            
        return self.w * cdf_gauss1 + (1 - self.w)*cdf_gauss2 


    def _create_inverse(self):
            
        choices = np.random.choice([0, 1], size=100000, p=[self.w, 1-self.w])

        # Draw samples accordingly
        fake_samples = np.where(
        choices == 0,
                np.random.normal(self.mu1, self.sigma1, 100000),
                np.random.normal(self.mu2, self.sigma2, 100000)
        )
            

        ecdf_func = scipy.stats.ecdf(fake_samples)
        x_sorted = np.sort(fake_samples)
        y = ecdf_func.cdf.evaluate(x_sorted)
        
        
        self.inverse_cumulative_distribution = scipy.interpolate.interp1d(y, x_sorted, bounds_error=False,
                        fill_value=(x_sorted[0], x_sorted[-1]))
        

def compute_kde(data, Nbins, xMin, xMax, bounded=False, method=None, smooth=None):
    """Compute KDE of `data` normalised across interval [`xMin`, `xMax`]
    using Nbins bins.
    If `bounded` is true, use a bounded KDE method with bounds `xMin`, `xMax`.
    """
    xPoints = np.linspace(xMin, xMax, Nbins)
    if bounded:
        if method != None:
            
            if method == 'Transform':
                
                kernel = bounded_1d_kde(data,
                                    xlow=xMin, xhigh=xMax, method=method, smooth=smooth
                                )
            else:
                
                kernel = bounded_1d_kde(data,
                                    xlow=xMin, xhigh=xMax, method=method
                                )
        else:
            kernel = bounded_1d_kde(data,
                                    xlow=xMin, xhigh=xMax
                                )
        return kernel(xPoints)
    else:
        print('gaussian kde')
        kernel = scipy.stats.gaussian_kde(data)
        return kernel.evaluate(xPoints) / kernel.integrate_box_1d(xMin, xMax)
        #return kernel

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

def compute_js(p1,p2, eps=1e-15):

    '''    
    print("Any negatives?", (p1 < 0).any(), (p2 < 0).any())
    print("Sums:", p1.sum(), p2.sum())
    print("Any NaNs?", np.isnan(p1).any(), np.isnan(p2).any())
    print("Any Infs?", np.isinf(p1).any(), np.isinf(p2).any())
    '''

    p1+=eps
    p2+=eps
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    p_m = 0.5 * (p1 + p2)
    jsd = 0.5 * (entropy(p1,p_m) + entropy(p2, p_m))
    
    jsd_spy = jensenshannon(p1,p2)
    
    return np.array([jsd, jsd_spy*jsd_spy])
    
# TODO: remove this!

# ########## Uniform ################
# m1_unif = bilby.core.prior.analytical.Uniform(minimum=1.0, maximum=3.0)
# m2_unif = bilby.core.prior.analytical.Uniform(minimum=1.0, maximum=3.0)

# chirp_mass = Constraint(minimum=0.8, maximum=3.0)
# mass_ratio = Constraint(minimum=0.125, maximum=1.0)

# m1u_prior = np.array(m1_unif.sample(300000))
# m2u_prior = np.array(m2_unif.sample(300000))

# m1u_samps, m2u_samps = create_prior_samps(m1u_prior,m2u_prior, 30000)


# ########## Double Gaussian ###############
# m1_doublegauss = DoubleGaussian(mu1 = 1.34, mu2 = 1.80, sigma1 = 0.07, sigma2 = 0.21, w=0.65)
# m2_doublegauss = DoubleGaussian(mu1 = 1.34, mu2 = 1.80, sigma1 = 0.07, sigma2 = 0.21, w=0.65)
# chirp_mass = Constraint(minimum=0.8, maximum=3.0)
# mass_ratio = Constraint(minimum=0.125, maximum=1.0)

# m1dg_prior = np.array(m1_doublegauss.sample(300000))
# m2dg_prior = np.array(m2_doublegauss.sample(300000))

# m1dg_samps, m2dg_samps = create_prior_samps(m1dg_prior,m2dg_prior, 30000)

# ######### Gaussian ##################

# m1_gauss = bilby.core.prior.analytical.Gaussian(mu=1.33, sigma=0.09)
# m2_gauss = bilby.core.prior.analytical.Gaussian(mu=1.33, sigma=0.09)

# chirp_mass = Constraint(minimum=0.8, maximum=3.0)
# mass_ratio = Constraint(minimum=0.125, maximum=1.0)

# m1g_prior = np.array(m1_gauss.sample(300000))
# m2g_prior = np.array(m2_gauss.sample(300000))

# m1g_samps, m2g_samps = create_prior_samps(m1g_prior,m2g_prior, 30000)

# ########### Default #####################

# from bilby.gw.conversion import luminosity_distance_to_redshift
# from astropy.cosmology import Planck18

# prior_chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=1.29, maximum=1.32)
# prior_mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)

# mchirp_samps_det = np.array(prior_chirp_mass.sample(300000))
# q_samps = np.array(prior_mass_ratio.sample(300000))

# distL = 171.27056031500612
# z = luminosity_distance_to_redshift(distL, cosmology=Planck18)
# print('redshift', z)
# mchirp_samps = mchirp_samps_det / (1. + z)

# m1def_prior, m2def_prior = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(mchirp_samps,q_samps)

# m1def_samps, m2def_samps = create_prior_samps(m1def_prior,m2def_prior, 30000)

# ########### Mchirp Uniform ################

# prior_chirp_mass = bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.29, maximum=1.32)
# prior_mass_ratio = bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1)

# mchirp_samps_det_unif = np.array(prior_chirp_mass.sample(300000))
# q_samps_unif = np.array(prior_mass_ratio.sample(300000))

# distL = 171.27056031500612
# z = luminosity_distance_to_redshift(distL, cosmology=Planck18)
# print('redshift', z)
# mchirp_samps_unif = mchirp_samps_det_unif / (1. + z)

# m1_mcu_prior, m2_mcu_prior = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(mchirp_samps_unif,q_samps_unif)

# m1_mcu_samps, m2_mcu_samps = create_prior_samps(m1_mcu_prior,m2_mcu_prior, 30000)


# ### Load data, compute kdes and plot

# resfiles = ['/work/wouters/GW231109/prod_BW_XP_s005_l5000_double_gaussian/outdir/result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5', '/work/wouters/GW231109/prod_BW_XP_s005_l5000_gaussian/outdir/result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5', '/work/wouters/GW231109/prod_BW_XP_s005_l5000_uniform/outdir/result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5', '/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/outdir/result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5', '/work/puecher/S231109/eos_sampling/prod_BW_XP_s005_leos_default/outdir/result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5']

# m1_data = []
# m2_data = []

# for samp_file in resfiles:

#     name = samp_file.split('/')[4]
#     print(name)
#     #'xas_default_l5000_s005.hdf5'

#     data = h5py.File(samp_file, 'r')
#     samp = data['posterior']

#     m1pos = np.random.choice(samp['mass_1_source'], 30000, replace=True)
#     m2pos = np.random.choice(samp['mass_2_source'], 30000, replace=True)

#     #m1_data.append(np.array(m1_data_tmp))
#     #m2_data.append(np.array(m2_data_tmp))

#     ############################
#     ### Compute and print jsds
#     ############################

#     print('========== For run {} ============'.format(name))
#     dist_labels = ['Double Gaussian', 'Gaussian', 'Uniform', 'Default']

#     print('======== mass 1 JSD ========')

#     xmin1 = min(min(m1dg_samps), min(m1g_samps), min(m1u_samps), min(m1def_samps), min(m1pos))
#     xmax1 = max(max(m1dg_samps), max(m1g_samps), max(m1u_samps), max(m1def_samps), max(m1pos))

#     xmin2 = min(min(m2dg_samps), min(m2g_samps), min(m2u_samps), min(m2def_samps), min(m2pos))
#     xmax2 = max(max(m2dg_samps), max(m2g_samps), max(m2u_samps), max(m2def_samps), max(m2pos))

#     print('xmin:', 'dg', min(m1dg_samps), 'g', min(m1g_samps), 'unif', min(m1u_samps), 'def', min(m1def_samps), 'data', min(m1pos), 'tot', xmin1)
#     #print(lala)

#     for jj, samps in enumerate([m1dg_samps, m1g_samps, m1u_samps, m1def_samps]):


#         #xmin = min(min(samps),min(m1pos))
#         #xmax = max(max(samps), max(m1pos))
    
#         if jj > 1:
#             method_prior = 'Transform'
#         else:
#              method_prior = 'Reflection'
#         prior_kde = compute_kde(samps, Nbins=5000, xMin = xmin1, xMax=xmax1,
#                       bounded=True, method=method_prior)
#         pos_kde = compute_kde(m1pos, Nbins=5000, xMin=xmin1, xMax=xmax1,
#                         bounded=True, method='Reflection')
#         '''
#         plt.plot(np.linspace(xmin1, xmax1,1000), prior_kde/prior_kde.sum(), label='prior')
#         plt.plot(np.linspace(xmin1, xmax1,1000), pos_kde/pos_kde.sum(), label='posterior')
#         med = 0.5 * (prior_kde/prior_kde.sum() + pos_kde/pos_kde.sum())
#         plt.plot(np.linspace(xmin1, xmax1,1000), med, label='med')
#         plt.legend()
#         plt.savefig('jsd_stuff_{}_{}.png'.format(jj,name))
#         plt.close()
#         '''
#         jsd_val = compute_js(pos_kde, prior_kde)
    
#         print(dist_labels[jj], '===', 'JSD entropy:', jsd_val[0], 'JSD scipy:', jsd_val[1])


#     #print(lala)
#     print('======== mass 2 JSD ========')

#     for jj, samps in enumerate([m2dg_samps, m2g_samps, m2u_samps, m2def_samps]):
        
#         #xmin = min(min(samps),min(m2pos))
#         #xmax = max(max(samps), max(m2pos))

#         if jj > 1:
#             method_prior = 'Transform'
#         else:
#              method_prior = 'Reflection'
#         prior_kde = compute_kde(samps, Nbins=5000, xMin = xmin2, xMax=xmax2,
#                       bounded=True, method=method_prior)
#         pos_kde = compute_kde(m1pos, Nbins=5000, xMin=xmin2, xMax=xmax2,
#                         bounded=True, method='Reflection')
        
#         jsd_val = compute_js(pos_kde, prior_kde)
    
#         print(dist_labels[jj], '===', 'JSD entropy:', jsd_val[0], 'JSD scipy:', jsd_val[1])
     

