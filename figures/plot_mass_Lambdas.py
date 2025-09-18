"""
Plot the mass-Lambdas contours from PE inference on top of mass-Lambdas EOS curves obtained from Jester inferences.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
import seaborn as sns

from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.conversion import luminosity_distance_to_redshift

import utils # utilities from the utils.py file in this directory
from bilby.core.prior import PriorDict

params = {"axes.grid": False,
        # "text.usetex" : True, # does not work on Jarvis
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"], # remove annoying warnings on Jarvis
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
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

GW231109_COLOR = "red"
G1124251_COLOR = "purple"
PRIOR_COLOR = "gray"
GW170817_COLOR = "orange"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mass-Lambda contours with EOS curves."
    )
    parser.add_argument(
        "--nb-samples", type=int, default=10_000,
        help="Number of EOS samples to plot (default: 5000)"
    )
    parser.add_argument(
        "--mass-min", type=float, default=0.25,
        help="Minimum mass cutoff in solar masses (default: 0.25)"
    )
    parser.add_argument(
        "--mass-max", type=float, default=2.0,
        help="Maximum mass cutoff in solar masses (default: 2.0)"
    )
    parser.add_argument(
        "--Lambda-min", type=float, default=0.0,
        help="Minimum tidal deformability cutoff (default: 0.0)"
    )
    parser.add_argument(
        "--Lambda-max", type=float, default=15_000.0,
        help="Maximum tidal deformability cutoff (default: 15_000.0)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite existing plots (default: False)"
    )
    return parser.parse_args()

def load_eos_curves(eos_name: str):
    """
    Load the EOS curves from a file. 
    """
    filename = f"EOS_data/{eos_name}.npz"
    data = np.load(filename)
    M, R, L = data['M'], data['R'], data['L']
    log_prob = data['log_prob']
    return M, R, L, log_prob

def get_mchirp_lambda_tilde_EOS(EOS_masses: np.array,
                                EOS_Lambdas: np.array,
                                mchirp_min: np.array,
                                mchirp_max: np.array) -> tuple[np.array, np.array]:
    """
    Generate an array of chirp masses (source fame) and corresponding lambda_tilde values based on the provided EOS masses and Lambdas.

    Args:
        EOS_masses (np.array): Array of EOS masses.
        EOS_Lambdas (np.array): Array of EOS Lambdas.
        mchirp_min (float): Minimum chirp mass (source frame).
        mchirp_max (float): Maximum chirp mass (source frame).

    Returns:
        tuple[np.array, np.array]: Tuple of arrays containing chirp masses and corresponding lambda_tilde values.
    """
    
    mchirp_array = np.linspace(mchirp_min, mchirp_max, 100)
    q_array = np.ones_like(mchirp_array) # TODO: change this later to use chirp mass and lambda_tilde, but approximately equal to 1 for now
    
    # These masses are in the source frame, 
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(mchirp_array, q_array)
    lambda_1 = np.interp(mass_1, EOS_masses, EOS_Lambdas)
    lambda_2 = np.interp(mass_2, EOS_masses, EOS_Lambdas)
    
    # Now get lambda tilde
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
    
    return mchirp_array, lambda_tilde

def load_GW170817_PE(chirp_tilde: bool = False):
    """
    Load the GW170817 posterior samples for mass and Lambda.
    """
    filename = "EOS_data/PE_posterior_samples_GW170817.npz"
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
    

    
def fetch_prior_samples(source_dir: str,
                        nb_samples: int = 10_000) -> tuple[np.array, np.array]:
    
        # Get the priors, and turn it into a PriorDict object
        priors = utils.initialize_priors(source_dir, ["chirp_mass", "mass_ratio", "lambda_1", "lambda_2", "luminosity_distance"])
        prior = PriorDict(priors)
        
        # Now generate sample and fetch them for convenience
        samples = prior.sample(size=nb_samples)
        chirp_mass = samples["chirp_mass"]
        mass_ratio = samples["mass_ratio"]
        lambda_1 = samples["lambda_1"]
        lambda_2 = samples["lambda_2"]
        
        # Then transform
        z = luminosity_distance_to_redshift(samples["luminosity_distance"])
        chirp_mass_source = chirp_mass / (1+z)
        mass_1_source, mass_2_source = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_source, mass_ratio)
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1_source, mass_2_source)

        return chirp_mass_source, lambda_tilde
    

def make_plot_components(args: argparse.Namespace, eos_name: str, source_dir: str, plot_GW170817_PE: bool = True):
    """
    Main function to plot the mass-Lambda contours, with the component masses and Lambdas.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        eos_name (str): Name of the EOS to load.
        source_dir (str): Directory containing the posterior file.
    """
    
    raise NotImplementedError("This function was implemented but it has not been used in a while, so it might be broken. Please use make_plot_chirp_tilde or be careful when using this function.")
    
    # Check if the posterior file exists
    posterior_file = utils.fetch_posterior_filename(source_dir)
    
    # Fetch the mass and Lambda samples from the posterior file
    print(f"Fetching the mass and Lambda samples from {posterior_file}")
    mass_1_source, mass_2_source, lambda_1, lambda_2 = utils.fetch_mass_Lambdas_samples(posterior_file)
    
    # Load the EOS curves
    M, _, L, log_prob = load_eos_curves(eos_name)
    max_samples = len(log_prob)
    print(f"Loaded {len(log_prob)} EOS curves from eos_name = {eos_name}.")

    # Convert to probabilities
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    max_log_prob_idx = np.argmax(log_prob)
    
    # Downsample the samples
    indices = np.random.choice(max_samples, args.nb_samples, replace=False)
    indices = np.append(indices, max_log_prob_idx) # ensure the max prob sample is included, for consistent normalization

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig = plt.figure(figsize=(8, 8))
    
    # Plot all the EOS curves
    print(f"Plotting EOS curves")
    for i in indices:
        # Get the color for this sample
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        
        # Mask since a few datapoints might be bad
        mask_masses = (M[i] > args.mass_min) * (M[i] < args.mass_max)
        mask_Lambdas = (L[i] > args.Lambda_min) * (L[i] < args.Lambda_max)
        mask = mask_masses & mask_Lambdas
        
        plt.plot(L[i][mask], M[i][mask], color=color, alpha=0.2, lw=2.0, zorder=normalized_value, rasterized=True)
        
    # Plot the mass-lambda contours on top
    print(f"Plotting PE contours")
    corner.hist2d(lambda_1, mass_1_source, fig=fig, color=GW231109_COLOR, **default_corner_kwargs)
    corner.hist2d(lambda_2, mass_2_source, fig=fig, color=GW231109_COLOR, **default_corner_kwargs)
    
    if plot_GW170817_PE:
        print(f"Plotting GW170817 PE contours")
        mass_1_source_GW170817, mass_2_source_GW170817, lambda_1_GW170817, lambda_2_GW170817 = load_GW170817_PE()
        corner.hist2d(lambda_1_GW170817, mass_1_source_GW170817, fig=fig, color=GW170817_COLOR, **default_corner_kwargs)
        corner.hist2d(lambda_2_GW170817, mass_2_source_GW170817, fig=fig, color=GW170817_COLOR, **default_corner_kwargs)
        
    # Add labels
    fs = 16
    plt.xlabel(r"$\Lambda$", fontsize=fs)
    plt.ylabel(r"$M$ [M$_\odot$]", fontsize=fs)
    plt.xlim(right=args.Lambda_max)
    plt.ylim(args.mass_min, args.mass_max)
    
    # Use the run name from the source_dir for the plot title
    run_name = source_dir.split("/")[-1]
    plt.title(f"PE = {run_name}", fontsize=fs)
    
    if plot_GW170817_PE:
        # Define legend entries
        legend_elements = [
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817'),
            mpatches.Patch(facecolor=GW231109_COLOR, edgecolor='k', label='GW231109')
        ]

        # Add the legend manually
        plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    # Add colorbar
    ax = fig.gca()
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"Posterior EOS probability: {eos_name}", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs-2)
    
    # Use the name of person to easily identify the source of the run again
    if "puecher" in source_dir:
        person = "anna"
    elif "dietrich6" in source_dir:
        person = "tim"
    elif "wouters" in source_dir:
        person = "thibeau"
    else:
        raise ValueError(f"Who ran this? I don't know the person from the source_dir {source_dir}!")
        
    save_name = f"./mass_Lambdas_figs/components/EOS_{eos_name}_{person}_{run_name}.pdf"
    print(f"Saving figure to {save_name}")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    

def make_plot_chirp_tilde(args: argparse.Namespace,
                          eos_name: str,
                          source_dir: str,
                          show_priors: bool = True,
                          plot_GW170817_PE: bool = True):
    """
    Main function to plot the mass-Lambda contours, with the component masses and Lambdas.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        eos_name (str): Name of the EOS to load.
        source_dir (str): Directory containing the posterior file.
    """
    
    # First of all, generate the save_name
    
    # Use the name of person to easily identify the source of the run again
    if "puecher" in source_dir:
        person = "anna"
    elif "dietrich6" in source_dir:
        person = "tim"
    elif "wouters" in source_dir:
        person = "thibeau"
    else:
        raise ValueError(f"Who ran this? I don't know the person from the source_dir {source_dir}!")
    
    run_name = source_dir.split("/")[-1]
    
    if plot_GW170817_PE:
        save_name = f"./mass_Lambdas_figs/chirp_tilde/with_GW170817_posterior/EOS_{eos_name}_{person}_{run_name}.pdf"
    else:
        save_name = f"./mass_Lambdas_figs/chirp_tilde/without_GW170817_posterior/EOS_{eos_name}_{person}_{run_name}.pdf"
        
    # Save the G1124251 runs in a separate folder for now. Bit hacky but whatever
    if "G1124251" in source_dir:
        save_name = save_name.replace("mass_Lambdas_figs/", "G1124251/mass_Lambdas_figs/")
        PE_color = G1124251_COLOR
        run_label = "G1124251"
    else:
        PE_color = GW231109_COLOR
        run_label = "GW231109"
        
    # Check if the file already exists, and if so, then skip
    if os.path.exists(save_name) and args.overwrite is False:
        print(f"File {save_name} already exists, skipping...")
        return
    
    # FIXME: just putting these here, but put them under argparse later
    mchirp_min = 1.1
    mchirp_max = 1.5
    lambda_tilde_max = 2_000 # TODO: determine them from the PE samples eg 99% quantile?
    
    # Check if the posterior file exists
    posterior_file = utils.fetch_posterior_filename(source_dir)
    if posterior_file is None:
        print(f"No posterior file found in {source_dir}, skipping...")
        return
    
    # Fetch the mass and Lambda samples from the posterior file
    print(f"Fetching the mass and Lambda samples from {posterior_file}")
    chirp_mass_source_GW231109, _, lambda_tilde_GW231109, _ = utils.fetch_mass_Lambdas_samples(posterior_file, chirp_tilde=True)
    
    fig = plt.figure(figsize=(12, 6))
    
    if eos_name == "tabular":
        if not os.path.exists("./eos_tables"):
            raise FileNotFoundError("The eos_tables directory does not exist. You might have to unzip the eos_tables.tar.gz file first.")
        
        print(f"Reading EOSs from the tabular and plotting those")
        for filename in os.listdir("./eos_tables"):
            if filename.startswith("BHBlp"):
                print(f"Anna told me to skip this one, so we are skipping this one.")
                continue
        
            full_filename = os.path.join("./eos_tables", filename)
            df = pd.read_csv(full_filename,
                comment="#",
                sep=r"\s+",
                names=["rho_c", "R_iso", "M", "Mb", "R", "Lambda", "e_c", "R_inv"]
            )
            EOS_masses = np.array(df["M"].values)
            EOS_Lambdas = np.array(df["Lambda"].values)
            
            mchirp_array, lambda_tilde = get_mchirp_lambda_tilde_EOS(EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max)
            
            # TODO: get the colors based on R1.4
            
            # Plot it
            eos_label = filename.split("_")[0]
            plt.plot(mchirp_array, lambda_tilde, label=eos_label, lw=3.0, zorder=1, rasterized=True)
    else:
        # Load the EOS curves
        M, _, L, log_prob = load_eos_curves(eos_name)
        max_samples = len(log_prob)
        print(f"Loaded {len(log_prob)} EOS curves from eos_name = {eos_name}.")

        # Convert to probabilities
        log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
        max_log_prob_idx = np.argmax(log_prob)
        
        # Downsample the samples
        np.random.seed(42)  # for reproducibility
        indices = np.random.choice(max_samples, args.nb_samples, replace=False)
        indices = np.append(indices, max_log_prob_idx) # ensure the max prob sample is included, for consistent normalization

        # Get a colorbar for log prob, but normalized
        norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
        cmap = sns.color_palette("crest", as_cmap=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        print(f"Plotting EOS curves from Jester")
        for i in indices:
            # Get the color for this sample
            normalized_value = norm(log_prob[i])
            color = cmap(normalized_value)
            
            # Mask since a few datapoints might be bad
            mask_masses = (M[i] > args.mass_min) * (M[i] < args.mass_max)
            mask_Lambdas = (L[i] > args.Lambda_min) * (L[i] < args.Lambda_max)
            mask = mask_masses & mask_Lambdas
            
            # Get the mass-Lambdas curve for this particular EOS
            EOS_masses, EOS_Lambdas = M[i][mask], L[i][mask]
            mchirp_array, lambda_tilde = get_mchirp_lambda_tilde_EOS(EOS_masses, EOS_Lambdas, mchirp_min, mchirp_max)
            
            # Then plot it
            plt.plot(mchirp_array, lambda_tilde, color=color, alpha=0.2, lw=2.0, zorder=normalized_value, rasterized=True)
            
    # Plot the mass-lambda contours on top
    print(f"Plotting PE posterior contours")
    corner.hist2d(chirp_mass_source_GW231109, lambda_tilde_GW231109, fig=fig, color=PE_color, **default_corner_kwargs)
    
    if show_priors:
        print(f"Plotting PE prior contours")
        chirp_mass_source_prior, lambda_tilde_prior = fetch_prior_samples(source_dir)
        corner.hist2d(chirp_mass_source_prior, lambda_tilde_prior, fig=fig, color=PRIOR_COLOR, **default_corner_kwargs)
    
    
    if plot_GW170817_PE:
        print(f"Plotting GW170817 PE contours")
        chirp_mass_source_GW170817, _, lambda_tilde_GW170817, _ = load_GW170817_PE(chirp_tilde=True)
        corner.hist2d(chirp_mass_source_GW170817, lambda_tilde_GW170817, fig=fig, color=GW170817_COLOR, **default_corner_kwargs)
        
    # Add labels
    fs = 16
    plt.xlabel(r"$\mathcal{M}_c^{\rm{source}}$ [M$_\odot$]", fontsize=fs)
    plt.ylabel(r"$\tilde{\Lambda}$", fontsize=fs)
    plt.xlim(1.1, 1.5)
    plt.ylim(top=lambda_tilde_max)
    
    # Use the run name from the source_dir for the plot title
    run_name = source_dir.split("/")[-1]
    plt.title(f"PE = {run_name}", fontsize=fs)
    
    if plot_GW170817_PE and eos_name != "tabular":
        # In this case, we only have GW231109 and GW170817, so we can add a legend by hand
        legend_elements = [
            mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817'),
            mpatches.Patch(facecolor=GW231109_COLOR, edgecolor='k', label=run_label)
        ]
        
        if show_priors:
            legend_elements += [mpatches.Patch(facecolor=PRIOR_COLOR, edgecolor='k', label='Prior')]

        # Add the legend manually
        plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    else:
        # First get the default legend from the plotted objects
        handles, labels = plt.gca().get_legend_handles_labels()

        # Add GW170817 and GW231109 patches
        if plot_GW170817_PE:
            handles += [
                mpatches.Patch(facecolor=GW170817_COLOR, edgecolor='k', label='GW170817'),
                mpatches.Patch(facecolor=PE_color, edgecolor='k', label=run_label)
            ]
        else:
            handles += [
                mpatches.Patch(facecolor=PE_color, edgecolor='k', label=run_label)
            ]
        
        if show_priors:
            handles += [mpatches.Patch(facecolor=PRIOR_COLOR, edgecolor='k', label='Prior')]

        plt.legend(handles=handles, loc='upper right', frameon=True)
    
    # Add colorbar
    if eos_name != "tabular":
        ax = fig.gca()
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"Posterior EOS probability: {eos_name}", fontsize=fs)
        cbar.ax.tick_params(labelsize=fs-2)
    
    print(f"Saving figure to {save_name}")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    
def main():
    args = parse_args()
    
    # List of all the jester EOS constraints to loop over
    eos_name_list = ["GW170817",
                    #  "all",
                    #  "radio",
                     ]
    
    # # List of base dirs to loop over
    base_dir_list = ["/work/wouters/GW231109/",
                     "/work/puecher/S231109/bw_runs_debug/",
                     ]
    
    for eos_name in eos_name_list:
        for base_dir in base_dir_list:
            source_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ["outdir", "data"]]
            print(f"For base directory {base_dir}, found the source directories:")
            print(f"    {source_dirs}")
            
            for source_dir in source_dirs:
                source_dir = os.path.join(base_dir, source_dir)
                print(f" =============== Plotting EOS = {eos_name}, source_dir = {source_dir} ===============")
                # make_plot_components(args, eos_name, source_dir)
                make_plot_chirp_tilde(args, eos_name, source_dir, show_priors=False, plot_GW170817_PE=True)
                # make_plot_chirp_tilde(args, eos_name, source_dir, show_priors=False, plot_GW170817_PE=False)
                
if __name__ == "__main__":
    main()