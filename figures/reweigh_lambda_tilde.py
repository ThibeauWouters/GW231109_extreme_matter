import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py

from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate

from bilby.core.prior import Uniform, Constraint
from bilby.core.prior import PriorDict
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio

from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.gw.conversion import binary_love_lambda_symmetric_to_lambda_1_lambda_2_manual_marginalisation

def reflected_kde(samples, boundary=0, eval_point=0):
    """Reflect samples across boundary for better boundary estimation"""
    # Reflect samples across the boundary
    samples = np.asarray(samples)
    reflected_samples = np.concatenate([samples, 2*boundary - samples])
    
    # Fit KDE to reflected data
    kde = gaussian_kde(reflected_samples)
    
    # # Evaluate only on original domain
    # return kde(eval_point)[0]
    return kde

params = {"axes.grid": False,
        # "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

from make_presentation_figs import file_paths, load_hdf5, load_json

def boundary_corrected_kde(data, boundary=0, bw='scott'):
    """KDE with reflection boundary correction"""
    kde = KDEUnivariate(data)
    kde.fit(bw=bw, boundary=boundary, fft=False)
    return kde

plt.figure(figsize=(6,4))
for run_label, filepath in file_paths.items():
    
    # # FIXME: this is more complicated than I thought
    # # First load the prior samples
    # if filepath.endswith(".hdf5") or filepath.endswith(".h5"):
    #     prior_samples, lambda_tilde_prior_samples, _ = load_hdf5(filepath, load_prior=True)
    # else:
    #     prior_samples, lambda_tilde_prior_samples, _ = load_json(filepath, load_prior=True)
    
    # First make the prior KDE?
    if "lquniv" not in filepath:
        chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=1.29, maximum=1.32, unit='$M_{\odot}$')
        mass_ratio = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
        lambda_1 = Uniform(name='lambda_1', minimum=0, maximum=5000.0)
        lambda_2 = Uniform(name='lambda_2', minimum=0, maximum=5000.0)
        
        priors = dict(chirp_mass=chirp_mass, mass_ratio=mass_ratio, lambda_1=lambda_1, lambda_2=lambda_2)
        prior = PriorDict(priors)
        
        # Generate samples from the prior
        prior_samples = prior.sample(10_000)
    
    else:
        chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=1.29, maximum=1.32, unit='$M_{\odot}$')
        mass_ratio = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
        
        lambda_1 = Constraint(name = 'lambda_1', minimum = 0, maximum = 5000)
        lambda_2 = Constraint(name = 'lambda_2', minimum = 0, maximum = 5000)
        lambda_symmetric = Uniform(0, 5000, name='lambda_symmetric', latex_label='$\Lambda_S$')
        binary_love_uniform = Uniform(0.0, 1.0, name='binary_love_uniform', latex_label='$\mathrm{BL}_{uni}$')
        lambda_antisymmetric = Constraint(name='lambda_antisymmetric', minimum=0, maximum=5000, latex_label='$\Lambda_A$')
        
        priors = dict(chirp_mass=chirp_mass,
                      mass_ratio=mass_ratio,
                      lambda_1=lambda_1,
                      lambda_2=lambda_2,
                      lambda_symmetric=lambda_symmetric,
                      binary_love_uniform=binary_love_uniform,
                      lambda_antisymmetric=lambda_antisymmetric
                      )
        prior = PriorDict(priors)
        
        # Generate samples from the prior
        prior_samples = prior.sample(10_000)
        print(list(prior_samples.keys()))
        
        prior_samples["lambda_1"], prior_samples["lambda_2"] =\
            binary_love_lambda_symmetric_to_lambda_1_lambda_2_manual_marginalisation(
                prior_samples['binary_love_uniform'],
                prior_samples['lambda_symmetric'],
                prior_samples['mass_ratio'])

    
    chirp_mass_samples = prior_samples["chirp_mass"]
    mass_ratio_samples = prior_samples["mass_ratio"]
    lambda_1_samples = prior_samples["lambda_1"]
    lambda_2_samples = prior_samples["lambda_2"]
    
    # Generate lambda tilde
    mass_1_samples, mass_2_samples = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_samples, mass_ratio_samples)
    lambda_tilde_prior_samples = lambda_1_lambda_2_to_lambda_tilde(lambda_1_samples, lambda_2_samples, mass_1_samples, mass_2_samples)
    
    # Build the KDE
    # prior_kde = gaussian_kde(lambda_tilde_prior_samples)
    prior_kde = reflected_kde(lambda_tilde_prior_samples)
    x = np.linspace(0.0, 10_000.0, 1_000)
    plt.plot(x, prior_kde(x), color='C1', label='KDE', linewidth=2)

    if filepath.endswith(".hdf5") or filepath.endswith(".h5"):
        samples, lambda_tilde_posterior_samples, luminosity_distance = load_hdf5(filepath)
    else:
        samples, lambda_tilde_posterior_samples, luminosity_distance = load_json(filepath)
        
    # posterior_kde = reflected_kde(lambda_tilde_posterior_samples)
    posterior_kde = gaussian_kde(lambda_tilde_posterior_samples)
    x = np.linspace(0.0, 10_000.0, 1000)
    posterior_kde_values = posterior_kde(x)
    prior_kde_values = prior_kde(x)
    
    reweigh_factor = posterior_kde_values / prior_kde_values
    reweigh_factor /= np.trapz(reweigh_factor, x)
    
    reweighted_samples = np.random.choice(x, size=10_000, p=reweigh_factor/np.sum(reweigh_factor))
    
    # TODO: compute Savage-Dickey ratio here?
    eps = 1e-3
    savage_dickey_ratio = posterior_kde(eps) / prior_kde(eps)
    savage_dickey_ratio = float(savage_dickey_ratio)
    ln_BF = np.log(savage_dickey_ratio)
    print(f"{run_label} Savage-Dickey ratio: {savage_dickey_ratio}")
    print(f"{run_label} ln BF: {ln_BF}")
    
    # Make KDE
    reweighted_kde = reflected_kde(reweighted_samples)
    plt.figure(figsize=(12, 6))
    fs = 12
    fs_label = 22
    fs_legend = 16
    lw = 3
    plt.plot(x, prior_kde_values, color='gray', label='Prior', linewidth=lw)
    plt.plot(x, posterior_kde_values, color='C0', label='Posterior', linewidth=lw)
    plt.plot(x, reweighted_kde(x), color='C3', label=r'Re-weighted posterior (flat $\tilde{\Lambda}$ prior)', linewidth=lw)
    plt.xlabel(r'$\tilde{\Lambda}$', fontsize=fs_label)
    plt.ylabel('Density', fontsize=fs_label)
    plt.title(f'Savage-Dickey ratio = {savage_dickey_ratio:.2f} (ln BF = {ln_BF:.2f})', fontsize=fs_label)
    
    if "lquniv" in filepath:
        plt.xlim(0.0, 5_000.0)
    else:
        plt.xlim(0.0, 8_000.0)
    plt.ylim(0.0)
    plt.legend(fontsize=fs_legend)
    plt.savefig(f"./presentation/savage_dickey/{run_label}.png", dpi=300)
    plt.close()