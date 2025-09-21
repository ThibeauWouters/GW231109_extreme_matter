import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py
import json

from scipy.stats import gaussian_kde

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
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=3, 
                        min_n_ticks=2,
                        labelpad = 0.075,
                        truth_color = "red",
                        save=False)

def load_hdf5(filename: str, load_prior: bool = False):
    if load_prior:
        print(f"WARNING: Loading the prior")
    with h5py.File(filename, "r") as f:
        posterior = f["prior"] if load_prior else f["posterior"]
        
        chirp_mass = posterior["chirp_mass"][:]
        total_mass = posterior["total_mass"][:]
        mass_ratio = posterior["mass_ratio"][:]
        m1 = posterior["mass_1_source"][:]
        m2 = posterior["mass_2_source"][:]
        lambda_tilde = posterior["lambda_tilde"][:]
        luminosity_distance = posterior["luminosity_distance"][:]
            
        samples = np.vstack((chirp_mass, total_mass, mass_ratio, m1, m2, lambda_tilde, luminosity_distance)).T
        
        return samples, lambda_tilde, luminosity_distance
        
def load_json(filename: str, load_prior: bool = False):
    if load_prior:
        print(f"WARNING: Loading the prior")
    with open(filename, "r") as f:
        result = json.load(f)
        print("list(result.keys())")
        print(list(result.keys()))
        posterior = result["prior"]["content"] if load_prior else result["posterior"]["content"]
        
        chirp_mass = np.array(posterior["chirp_mass"])
        total_mass = np.array(posterior["total_mass"])
        mass_ratio = np.array(posterior["mass_ratio"])
        m1 = np.array(posterior["mass_1_source"])
        m2 = np.array(posterior["mass_2_source"])
        lambda_tilde = np.array(posterior["lambda_tilde"])
        luminosity_distance = np.array(posterior["luminosity_distance"])
        
    samples = np.vstack((chirp_mass, total_mass, mass_ratio, m1, m2, lambda_tilde, luminosity_distance)).T
    
    return samples, lambda_tilde, luminosity_distance
    

###### Setup
file_paths = {
  "quniv": "/work/wouters/GW231109/prod_BW_XP_s005_lquniv_default/outdir/final_result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5",
  "flagship": "/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/outdir/final_result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5",
#   "ET": "/work/puecher/S231109/third_gen_runs/et_run_alignedspin/outdir/ET_gw231109_injection_alignedspin_result.json",
}
run_labels_list = ["Flagship", "Quniv"]

# Colors for different runs
colors = {"flagship": "blue",
          "quniv": "orange",
          }
alphas = {"flaghsip": 0.3,
          "quniv": 0.3,
          }
###### Setup

def main():
    # Store data for all runs
    all_data = {}

    # Load all data
    for run_label, filepath in file_paths.items():
        if filepath.endswith(".hdf5") or filepath.endswith(".h5"):
            samples, lambda_tilde, luminosity_distance = load_hdf5(filepath)
        else:
            samples, lambda_tilde, luminosity_distance = load_json(filepath)
            
        all_data[run_label] = {
            'samples': samples,
            'lambda_tilde': lambda_tilde,
            'luminosity_distance': luminosity_distance
        }
        
    # Create corner plot with both runs overlaid
    # ranges = [[0.87875, 0.8799], [1.99, 2.3], [0.25, 1.0], [0, 8000], [50, 600]] # TODO: broken

    ranges = None
    labels = [r"$\mathcal{M}_c$ [M$_\odot$]",
            r"$M$ [M$_\odot$]",
            r"$q$",
            r"$m_1$ [M$_\odot$]",
            r"$m_2$ [M$_\odot$]",
            r"$\tilde{\Lambda}$",
            r"$d_L$ [Mpc]"]

    # Plot first run
    first_run = list(all_data.keys())[0]
    fig = corner.corner(all_data[first_run]['samples'], labels=labels, density=True, range=ranges, 
                    color=colors[first_run], **default_corner_kwargs)

    # Overlay second run
    for i, run_label in enumerate(list(all_data.keys())[1:], 1):
        corner.corner(all_data[run_label]['samples'], labels=labels, density=True, range=ranges, 
                    color=colors[run_label], fig=fig, **default_corner_kwargs)

    # Add legend to corner plot
    legend_elements = [plt.Line2D([0], [0], color=colors[run], lw=2, label=label) 
                    for run, label in zip(all_data.keys(), run_labels_list)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.975), fontsize=18)

    plt.savefig("./presentation/combined_corner.png", bbox_inches="tight", dpi=300)
    plt.close()

    # # Combined lambda_tilde plot
    # plt.figure(figsize=(8, 6))
    # fs = 12

    # for run_label in all_data.keys():
    #     lambda_tilde = all_data[run_label]['lambda_tilde']
    #     lambda_tilde_kde = gaussian_kde(lambda_tilde)
    #     min_lambda_tilde = np.min(lambda_tilde)
    #     max_lambda_tilde = np.max(lambda_tilde)
    #     lambda_tilde_space = np.linspace(min_lambda_tilde, max_lambda_tilde, 5_000)
    #     lambda_tilde_pdf = lambda_tilde_kde(lambda_tilde_space)
        
    #     plt.plot(lambda_tilde_space, lambda_tilde_pdf, color=colors[run_label], 
    #             label=run_labels_list[list(all_data.keys()).index(run_label)], linewidth=2)
    #     plt.fill_between(lambda_tilde_space, 0, lambda_tilde_pdf, 
    #                     color=colors[run_label], alpha=alphas[run_label])

    # plt.xlabel(r"$\tilde{\Lambda}$", fontsize=fs)
    # plt.ylabel("Probability Density", fontsize=fs)
    # plt.yscale("log")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("./presentation/combined_lambda_tilde.png", bbox_inches="tight", dpi=300)
    # plt.close()

    # # Combined luminosity distance plot ### NOTE: not so interesting for this event
    # plt.figure(figsize=(8, 6))

    # for run_label in all_data.keys():
    #     luminosity_distance = all_data[run_label]['luminosity_distance']
    #     lum_dist_kde = gaussian_kde(luminosity_distance)
    #     min_lum_dist = np.min(luminosity_distance)
    #     max_lum_dist = np.max(luminosity_distance)
    #     lum_dist_space = np.linspace(min_lum_dist, max_lum_dist, 5_000)
    #     lum_dist_pdf = lum_dist_kde(lum_dist_space)

    #     plt.plot(lum_dist_space, lum_dist_pdf, color=colors[run_label], 
    #              label=run_labels_list[list(all_data.keys()).index(run_label)], linewidth=2)
    #     plt.fill_between(lum_dist_space, 0, lum_dist_pdf, 
    #                      color=colors[run_label], alpha=alphas[run_label])

    # plt.xlabel(r"$d_L$ [Mpc]", fontsize=fs)
    # plt.ylabel("Probability Density", fontsize=fs)
    # plt.yscale("log")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("./presentation/combined_luminosity_distance.png", bbox_inches="tight", dpi=300)
    # plt.close()

if __name__ == "__main__":
    main()
