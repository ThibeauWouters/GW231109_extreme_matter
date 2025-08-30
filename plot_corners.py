"""
Plot the mass-Lambdas contours from PE inference on top of mass-Lambdas EOS curves obtained from Jester inferences.
"""

import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
import seaborn as sns

import utils # utilities from the utils.py file in this directory

params = {"axes.grid": False,
        "text.usetex" : False,
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
    
def make_cornerplot(source_dir: str,
                    keys_to_add: list[str] = ["chi_eff", "chi_p"],
                    overwrite: bool = False):
    """
    Make a corner plot of the mass-Lambda contours from PE inference on top of mass-Lambda EOS curves.
    
    Parameters:
    - source_dir: Directory containing the posterior samples and EOS curves.
    """
    
    # First of all, generate the save_name
    # TODO: move this to a utils function
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
    
    save_name = f"./cornerplots/{person}_{run_name}_cornerplot.pdf"
    
    # Save the G1124251 runs in a separate folder for now. Bit hacky but whatever
    if "G1124251" in source_dir:
        save_name = save_name.replace("cornerplots/", "G1124251/cornerplots/")
    
    if os.path.exists(save_name) and overwrite is False:
        print(f"File {save_name} already exists, skipping...")
        return
    
    # Check if the posterior file exists
    posterior_file = utils.fetch_posterior_filename(source_dir)
    if posterior_file is None:
        print(f"No posterior file found in {source_dir}. Skipping.")
        return
    
    with h5py.File(posterior_file, 'r') as f:
        
        # For general info, print ln bayes factor and sampling time
        log_bayes_factor = f["log_bayes_factor"][()]
        print(f"Log Bayes factor: {log_bayes_factor}")
        
        sampling_time = f["sampling_time"][()]
        sampling_time_hrs = sampling_time / 3600.0
        print(f"Sampling time: {sampling_time_hrs:.2f} hours")
        
        # Load the priors used
        prior_keys_to_skip = ['__prior_dict__', '__module__', '__name__']
        priors_bytes = f["priors"][()]        # bytes
        priors_str = priors_bytes.decode()    # bytes → string
        priors_dict = json.loads(priors_str)  # string → dict
        priors_dict = {k: v for k, v in priors_dict.items() if k not in prior_keys_to_skip and "recalib" not in k}
        prior_keys = list(priors_dict.keys())
        print(f"Prior dict keys: {prior_keys}")
        
        precessing = ("a_1" in prior_keys) and ("a_2" in prior_keys) # TODO: this is maybe not the best way to check for precession
        if precessing and "chi_p" in keys_to_add:
            print(f"Precessing run detected, removing 'chi_p' from the user-defined keys to add.")
            keys_to_add.remove("chi_p")
            
        if "fixed_dL" in source_dir:
            print(f"Found 'fixed' key in source_dir name. Assuming run with fixed ra and dec and removing it from the list of prior keys to plot.")
            prior_keys.remove("luminosity_distance")
        if "fixed_sky" in source_dir:
            prior_keys.remove("ra")
            prior_keys.remove("dec")
            
        # Load the posterior samples of the prior dict keys, and any other keys added by user
        keys_to_fetch = prior_keys + keys_to_add
        posterior = f["posterior"]
        posterior_samples = np.array([posterior[key][()] for key in keys_to_fetch]).T
        
    print(f"Creating corner plot . . .")
    corner.corner(posterior_samples,
                labels=keys_to_fetch,
                **default_corner_kwargs)
    
    print(f"Saving corner plot to {save_name}")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    
def main():
    # List of base dirs to loop over
    base_dir_list = ["/work/wouters/GW231109/"]
    
    for base_dir in base_dir_list:
        source_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ["outdir", "data"]]
        print(f"For base directory {base_dir}, found the source directories:")
        print(f"    {source_dirs}")
        
        for source_dir in source_dirs:
            source_dir = os.path.join(base_dir, source_dir)
            print(f"============ Processing source directory: {source_dir} ============")
            make_cornerplot(source_dir)
            
            
            print(f"===================================================================\n\n\n\n")
                
if __name__ == "__main__":
    main()