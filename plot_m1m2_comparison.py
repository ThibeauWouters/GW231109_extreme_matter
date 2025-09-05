"""
Plot 2D corner plots comparing mass_1_source and mass_2_source from PE inference across multiple runs.
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner

import utils  # utilities from the utils.py file in this directory

params = {"axes.grid": False,
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
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Corner plot kwargs for 2D histogram
corner_kwargs = dict(bins=40, 
                    smooth=1., 
                    show_titles=False,
                    label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16), 
                    plot_density=True, 
                    plot_datapoints=False, 
                    fill_contours=True,
                    max_n_ticks=4, 
                    min_n_ticks=3,
                    save=False)

# Colors for different runs
COLORS = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

def make_m1m2_comparison_plot(source_dirs: list[str], 
                             base_dir: str,
                             identifier: str,
                             overwrite: bool = False):
    """
    Create a 2D corner plot comparing mass_1_source and mass_2_source across multiple runs.
    
    Parameters:
    - source_dirs: List of subdirectories containing posterior samples
    - base_dir: Base directory containing the source directories
    - identifier: Unique identifier for the plot filename
    - overwrite: Whether to overwrite existing plots
    """
    
    save_name = f"./figures/m1m2/{identifier}.pdf"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    
    if os.path.exists(save_name) and not overwrite:
        print(f"File {save_name} already exists, skipping...")
        return
    
    fig = plt.figure(figsize=(12, 8))
    
    legend_elements = []
    valid_runs = 0
    
    # Collect all mass data to determine plot limits
    all_mass_1 = []
    all_mass_2 = []
    mass_data = []  # Store (mass_1, mass_2, color, label) for plotting
    
    for i, source_dir_name in enumerate(source_dirs):
        source_dir = os.path.join(base_dir, source_dir_name)
        
        if not os.path.isdir(source_dir):
            print(f"Directory {source_dir} does not exist, skipping...")
            continue
            
        # Check if the posterior file exists
        posterior_file = utils.fetch_posterior_filename(source_dir)
        if posterior_file is None:
            print(f"No posterior file found in {source_dir}. Skipping.")
            continue
        
        try:
            with h5py.File(posterior_file, 'r') as f:
                posterior = f["posterior"]
                mass_1_source = posterior["mass_1_source"][()]
                mass_2_source = posterior["mass_2_source"][()]
                
            # Get color for this run - use gray for test runs
            if "test" in source_dir_name.lower():
                color = "gray"
            else:
                color = COLORS[i % len(COLORS)]
            
            # Store data for plotting
            mass_data.append((mass_1_source, mass_2_source, color, source_dir_name))
            
            # Collect all masses for limit calculation
            all_mass_1.extend(mass_1_source)
            all_mass_2.extend(mass_2_source)
            
            valid_runs += 1
            print(f"Successfully loaded data for {source_dir_name}")
            
        except Exception as e:
            print(f"Failed to process {source_dir_name}: {e}")
            continue
    
    if valid_runs == 0:
        print("No valid runs found, not creating plot.")
        plt.close(fig)
        return
    
    # Calculate plot limits based on quantiles
    all_mass_1 = np.array(all_mass_1)
    all_mass_2 = np.array(all_mass_2)
    
    eps = 0.001
    m1_min, m1_max = np.quantile(all_mass_1, [eps, 1-eps])
    m2_min, m2_max = np.quantile(all_mass_2, [eps, 1-eps])
    
    # Add small margin for better visualization
    m1_margin = (m1_max - m1_min) * 0.10
    m2_margin = (m2_max - m2_min) * 0.10
    
    m1_min -= m1_margin
    m1_max += m1_margin
    m2_min -= m2_margin
    m2_max += m2_margin
    
    print(f"Plot limits: m1 = [{m1_min:.3f}, {m1_max:.3f}], m2 = [{m2_min:.3f}, {m2_max:.3f}]")
    
    # Now plot all the data
    for mass_1_source, mass_2_source, color, source_dir_name in mass_data:
        # Plot the 2D corner plot
        corner.hist2d(mass_1_source, mass_2_source, 
                     fig=fig, 
                     color=color,
                     **corner_kwargs)
        
        # Add to legend
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='k', label=source_dir_name)
        )
        
        print(f"Successfully plotted {source_dir_name}")
    
    # Add labels
    fs = 16
    plt.xlabel(r"$m_1^{\rm{source}}$ [M$_\odot$]", fontsize=fs)
    plt.ylabel(r"$m_2^{\rm{source}}$ [M$_\odot$]", fontsize=fs)
    
    # Set limits based on data quantiles
    plt.xlim(m1_min, m1_max)
    plt.ylim(m2_min, m2_max)
    
    # Add title
    plt.title(f"Mass comparison: {identifier}", fontsize=fs)
    
    # Add legend to the right of the plot
    if len(legend_elements) > 0:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    print(f"Saving figure to {save_name}")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Make m1-m2 comparison plots for GW inferences.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing plots if they already exist.")
    args = parser.parse_args()
    
    # Base directory
    base_dir = "/work/wouters/GW231109/"
    
    comparison_sets = {
        "neural_priors_double_gaussian": [
            "test_precessing_timefix",
            "neural_priors_double_gaussian_radio",
            "neural_priors_double_gaussian_radio_chiEFT", 
            "neural_priors_double_gaussian_radio_NICER"
        ],
        "neural_priors_gaussian": [
            "test_precessing_timefix",
            "neural_priors_gaussian_radio",
            "neural_priors_gaussian_radio_chiEFT", 
            "neural_priors_gaussian_radio_NICER"
        ],
        "neural_priors_uniform": [
            "test_precessing_timefix",
            "neural_priors_uniform_radio",
            "neural_priors_uniform_radio_chiEFT", 
            "neural_priors_uniform_radio_NICER"
        ],
        "neural_priors_radio": [
            "test_precessing_timefix",
            "neural_priors_uniform_radio",
            "neural_priors_gaussian_radio",
            "neural_priors_double_gaussian_radio",
        ],
        "neural_priors_radio_chiEFT": [
            "test_precessing_timefix",
            "neural_priors_uniform_radio_chiEFT",
            "neural_priors_gaussian_radio_chiEFT",
            "neural_priors_double_gaussian_radio_chiEFT",
        ],
        "neural_priors_radio_NICER": [
            "test_precessing_timefix",
            "neural_priors_uniform_radio_NICER",
            "neural_priors_gaussian_radio_NICER",
            "neural_priors_double_gaussian_radio_NICER",
        ],
    }
    
    # If no specific runs are hardcoded, use all available directories
    if not any(comparison_sets.values()):
        print("No specific comparison sets defined. Using all available directories...")
        if os.path.exists(base_dir):
            all_dirs = [d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) 
                       and d not in ["outdir", "data"]]
            comparison_sets = {"all_runs": all_dirs}
        else:
            print(f"Base directory {base_dir} does not exist!")
            return
    
    # Make plots for each comparison set
    for identifier, source_dirs in comparison_sets.items():
        if not source_dirs:  # Skip empty lists
            continue
            
        print(f"\n============ Creating comparison plot for: {identifier} ============")
        print(f"Comparing runs: {source_dirs}")
        
        make_m1m2_comparison_plot(source_dirs, base_dir, identifier, overwrite=args.overwrite)
        
        print(f"===================================================================\n")

if __name__ == "__main__":
    main()