import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py

from scipy.stats import gaussian_kde

from bilby.core.prior import Uniform
from bilby.core.prior import PriorDict
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio

from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde

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

from make_presentation_figs import file_paths

# First make the prior KDE?
chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=0.86912851, maximum=0.88912851, unit='$M_{\odot}$')
mass_ratio = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
lambda_1 = Uniform(name='lambda_1', minimum=0, maximum=5000.0)
lambda_2 = Uniform(name='lambda_2', minimum=0, maximum=5000.0)
priors = dict(chirp_mass=chirp_mass, mass_ratio=mass_ratio, lambda_1=lambda_1, lambda_2=lambda_2)
prior = PriorDict(priors)

# Generate samples from the prior
prior_samples = prior.sample(10_000)

chirp_mass_samples = prior_samples["chirp_mass"]
mass_ratio_samples = prior_samples["mass_ratio"]
lambda_1_samples = prior_samples["lambda_1"]
lambda_2_samples = prior_samples["lambda_2"]

# Generate lambda tilde
mass_1_samples, mass_2_samples = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_samples, mass_ratio_samples)
lambda_tilde_prior_samples = lambda_1_lambda_2_to_lambda_tilde(lambda_1_samples, lambda_2_samples, mass_1_samples, mass_2_samples)

# Make a histogram of prior
plt.figure(figsize=(6,4))
plt.hist(lambda_tilde_prior_samples, bins=50, density=True, histtype='step', color='black', label='Prior', linewidth=2)

# Build the KDE
prior_kde = gaussian_kde(lambda_tilde_prior_samples)
x = np.linspace(0.0, 10_000.0, 1000)
plt.plot(x, prior_kde(x), color='C1', label='KDE', linewidth=2)

plt.legend(fontsize=14)
plt.xlabel(r'$\tilde{\Lambda}$', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Prior on '+r'$\tilde{\Lambda}$', fontsize=16)
plt.xlim(0.0, 10_000.0)
plt.savefig("./figures/prior_lambda_tilde.png", dpi=300)
plt.close()

for run_label, filepath in file_paths.items():
    with h5py.File(filepath, "r") as f:
        lambda_tilde_posterior_samples = f["posterior"]["lambda_tilde"][:]
        
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
    reweighted_kde = gaussian_kde(reweighted_samples)
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
    plt.xlim(0.0, 8_000.0)
    plt.ylim(0.0)
    plt.legend(fontsize=fs_legend)
    plt.savefig(f"./figures/{run_label}_reweigh_lambda_tilde.png", dpi=300)
    plt.close()