"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tqdm
import arviz
from scipy.stats import gaussian_kde

np.random.seed(2)
import jesterTOV.utils as jose_utils

mpl_params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)

COLORS_DICT = {"prior": "gray",
               "GW170817": "orange",
               "GW231109": "teal",
               "GW231109_only": "red"}
ALPHA = 0.3

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

figsize_vertical = (6, 8)
figsize_horizontal = (8, 6)

def report_credible_interval(values: np.array, 
                             hdi_prob: float = 0.95,
                             verbose: bool = False) -> None:
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob = hdi_prob)
    
    low = med - low
    high = high - med
    
    if verbose:
        print(f"\n\n\n{med:.2f}-{low:.2f}+{high:.2f} (at {hdi_prob} HDI prob)\n\n\n")
    return low, med, high
    
def make_masterplots(outdir: str):
    filename = os.path.join(outdir, "eos_samples.npz")
    
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, _ = data["n"], data["p"], data["e"], data["cs2"]
    
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric
    
    nb_samples = np.shape(m)[0]
    print(f"Number of samples: {nb_samples}")

    # Plotting
    samples_kwargs = {"color": "black",
                      "alpha": 1.0,
                      "rasterized": True}

    plt.subplots(1, 1, figsize=(11, 6))

    m_min, m_max = 0.51, 3.25
    r_min, r_max = 8.0, 16.0
    
    # Sample requested number of indices randomly:
    log_prob = data["log_prob"]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    
    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    print("Creating NS plot . . .")
    bad_counter = 0
    for i in tqdm.tqdm(range(len(log_prob))):

        # Get color
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        samples_kwargs["color"] = color
        samples_kwargs["zorder"] = 1e10 + normalized_value
        
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue
    
        if any(l[i] < 0):
            bad_counter += 1
            continue
        
        if any((m[i] > 1.0) * (r[i] > 20.0)):
            bad_counter += 1
            continue
        
        # Mass-radius plot
        plt.plot(r[i], m[i], **samples_kwargs)
        
    print(f"Bad counter: {bad_counter}")
    # Beautify the plots a bit
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(r_min, r_max)
    plt.ylim(m_min, m_max)
    
    # Add the colorbar
    fig = plt.gcf()
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Normalized posterior probability", fontsize = 16)
    cbar.set_ticks([])
    cbar.ax.xaxis.labelpad = 5
    cbar.ax.tick_params(labelsize=0, length=0)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    cbar.set_label(r"Normalized posterior probability")

    # Save the figure
    save_name = os.path.join("./figures", f"{outdir}_postprocessing_NS.png")
    plt.savefig(save_name, bbox_inches = "tight", dpi=300)
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()
    print("Creating NS plot . . . DONE")
    
    plt.subplots(1, 1, figsize=(11, 6))
    print("Creating EOS plots . . .")
    for i in tqdm.tqdm(range(len(log_prob))):

        # Get color
        normalized_value = norm(log_prob[i])
        color = cmap(normalized_value)
        samples_kwargs["color"] = color
        samples_kwargs["zorder"] = 1e10 + normalized_value
        
        if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
            bad_counter += 1
            continue
    
        if any(l[i] < 0):
            bad_counter += 1
            continue
        
        if any((m[i] > 1.0) * (r[i] > 20.0)):
            bad_counter += 1
            continue
        
        # Mass-radius plot
        mask = (n[i] > 0.5) * (n[i] < 6.0)
        plt.plot(n[i][mask], p[i][mask], **samples_kwargs)
        
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.yscale('log')
    plt.xlim(0.5, 6.0)
    
    # # Add the colorbar
    # fig = plt.gcf()
    # sm.set_array([])
    # cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])  # [left, bottom, width, height]
    # cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label("Normalized posterior probability", fontsize = 16)
    # cbar.set_ticks([])
    # cbar.ax.xaxis.labelpad = 5
    # cbar.ax.tick_params(labelsize=0, length=0)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.get_offset_text().set_visible(False)
    # cbar.set_label(r"Normalized posterior probability")

    # Save the figure
    save_name = os.path.join("./figures", f"{outdir}_postprocessing_EOS.png")
    plt.savefig(save_name, bbox_inches = "tight", dpi=300)
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()
    print("Creating NS plot . . . DONE")
    
def make_comparison_plot(include_only: bool = False):
    """Make a comparison plot between different runs.
    
    Args:
        include_only (bool): Whether to include the GW231109_only run
    """
    
    if include_only:
        outdirs_list = ["prior", "GW170817", "GW231109_only", "GW231109"]
        filename_suffix = "_with_only"
    else:
        outdirs_list = ["prior", "GW170817", "GW231109"]
        filename_suffix = ""
        
    outdirs_dict = {"prior": "outdir_prior", 
                    "GW170817": "outdir_GW170817", 
                    "GW231109_only": "outdir_GW231109_only",
                    "GW231109": "outdir_GW231109"}
    translation_dict = {"prior": "Prior", 
                        "GW170817": "GW170817", 
                        "GW231109_only": "GW231109",
                        "GW231109": "GW170817+GW231109"}
    
    data_dict = {"MTOV": {},
                 "R14": {},
                 "p3nsat": {}}
    
    for outdir in outdirs_list:
        print(f"Gathering the information for outdir = {outdir} . . .")
        
        # Fetch the data that we want
        filename = os.path.join(outdirs_dict[outdir], "eos_samples.npz")
        data = np.load(filename)
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        logpc_EOS = data["logpc_EOS"]
        n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # Now make some postprocessing information
        MTOV_list = np.array([np.max(mass) for mass in m])
        R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
        p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])
        
        # Save into the dict
        data_dict["MTOV"][outdir] = MTOV_list
        data_dict["R14"][outdir] = R14_list
        data_dict["p3nsat"][outdir] = p3nsat_list
        
        
    ranges = {"MTOV": (1.75, 2.75),
              "R14": (10.0, 16.0),
              "p3nsat": (0.1, 200.0)}
    xlabels = {"MTOV": r"$M_{\rm{TOV}}$ [$M_{\odot}$]",
               "R14": r"$R_{1.4}$ [km]",
               "p3nsat": r"$p(3n_{\rm{sat}})$ [MeV fm$^{-3}$]"}
    
    # Now make the plots
    for key in data_dict.keys():
        plt.subplots(1, 1, figsize=figsize_horizontal)
        
        for i, outdir in enumerate(outdirs_list):
            # Make KDE
            kde = gaussian_kde(data_dict[key][outdir])
            x = np.linspace(ranges[key][0], ranges[key][1], 1000)
            y = kde(x)
            
            plt.plot(x, y,
                     label=translation_dict[outdir],
                     color=COLORS_DICT[outdir],
                     lw=3.0,
                     )
            
        xlabel = xlabels[key]
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.xlim(ranges[key][0], ranges[key][1])
        plt.ylim(bottom=0.0)
        plt.ylabel("Density")
        plt.legend()
        
        save_name = os.path.join("./figures", f"{key}{filename_suffix}.png")
        print(f"Saving to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight", dpi=300)
        plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
        plt.close()
        

def make_contour_plot_radii(m_min: float = 0.6,
                            m_max: float = 2.1,
                            include_only: bool = False,
                            use_l10000: bool = True,
                            ):
    """Make a contour plot of the posterior distribution of radii
    
    Args:
        m_min (float): Minimum mass for the plot
        m_max (float): Maximum mass for the plot
        include_only (bool): Whether to include the GW231109_only run
    """
    
    if include_only:
        outdirs_list = ["prior", "GW170817", "GW231109_only", "GW231109"]
        filename_suffix = "_with_only"
    else:
        outdirs_list = ["prior", "GW170817", "GW231109"]
        filename_suffix = ""
        
    outdirs_dict = {"prior": "outdir_prior", 
                    "GW170817": "outdir_GW170817", 
                    "GW231109_only": "outdir_GW231109_only",
                    }
    
    if use_l10000:
        outdirs_dict["GW231109"] = "outdir_GW231109_l10000"
        suffix = "_l10000"
    else:
        outdirs_dict["GW231109"] = "outdir_GW231109"
        suffix = ""
        
    filename_suffix += suffix
        
    translation_dict = {"prior": "Prior", 
                        "GW170817": "GW170817", 
                        "GW231109_only": "GW231109",
                        "GW231109": "GW170817+GW231109"}
    
    
    plt.figure(figsize=figsize_vertical)
    for outdir in outdirs_list:
        print(f"Gathering the information for outdir = {outdir} . . .")
        
        # Fetch the data that we want
        filename = os.path.join(outdirs_dict[outdir], "eos_samples.npz")
        data = np.load(filename)
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        
        # Radii only now
        masses_array = np.linspace(m_min, m_max, 100)
        radii_low = np.empty_like(masses_array)
        radii_high = np.empty_like(masses_array)
        
        for i, mass_point in tqdm.tqdm(enumerate(masses_array)):
            # Gather all the radii at this mass point
            radii_at_mass = []
            for mass, radius in zip(m, r):
                radii_at_mass.append(np.interp(mass_point, mass, radius))
            radii_at_mass = np.array(radii_at_mass)
            
            # Construct 95% credible interval
            low, med, high = report_credible_interval(radii_at_mass, hdi_prob=0.95)

            # Save
            radii_low[i] = med - low
            radii_high[i] = med + high
            
        # Now to plotting
        plt.fill_betweenx(masses_array,
                          radii_low,
                          radii_high,
                          alpha=ALPHA,
                          label=translation_dict[outdir],
                          color=COLORS_DICT[outdir],
                          )
        
        # Also plot the outer lines
        plt.plot(radii_low, masses_array, lw=2.0, color=COLORS_DICT[outdir])
        plt.plot(radii_high, masses_array, lw=2.0, color=COLORS_DICT[outdir])
        
        # Make nice
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.xlim(8.0, 16.0)
        plt.ylim(m_min, m_max)
        plt.legend()
        
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        
    save_name = os.path.join("./figures", f"contour_radii{filename_suffix}.png")
    print(f"Saving to {save_name}")
    plt.savefig(save_name, bbox_inches = "tight", dpi=300)
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()
    
def make_contour_plot_pressures(n_min: float = 0.5,
                                n_max: float = 6.0,
                                include_only: bool = False,
                                use_l10000: bool = True):
    """Make a contour plot of the posterior distribution of pressures
    
    Args:
        n_min (float): Minimum density for the plot
        n_max (float): Maximum density for the plot
        include_only (bool): Whether to include the GW231109_only run
    """
    
    if include_only:
        outdirs_list = ["prior", "GW170817", "GW231109_only", "GW231109"]
        filename_suffix = "_with_only"
    else:
        outdirs_list = ["prior", "GW170817", "GW231109"]
        filename_suffix = ""
        
    outdirs_dict = {"prior": "outdir_prior", 
                    "GW170817": "outdir_GW170817", 
                    "GW231109_only": "outdir_GW231109_only",
                    }
    
    if use_l10000:
        outdirs_dict["GW231109"] = "outdir_GW231109_l10000"
        suffix = "_l10000"
    else:
        outdirs_dict["GW231109"] = "outdir_GW231109"
        suffix = ""
        
    filename_suffix += suffix
    
    translation_dict = {"prior": "Prior", 
                        "GW170817": "GW170817", 
                        "GW231109_only": "GW231109",
                        "GW231109": "GW170817+GW231109"}
    
    plt.figure(figsize=figsize_horizontal)
    for outdir in outdirs_list:
        print(f"Gathering the information for outdir = {outdir} . . .")
        
        # Fetch the data that we want
        filename = os.path.join(outdirs_dict[outdir], "eos_samples.npz")
        data = np.load(filename)
        
        # Now load n and p
        n, p = data["n"], data["p"]
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        
        # Radii only now
        dens_array = np.linspace(n_min, n_max, 100)
        press_low = np.empty_like(dens_array)
        press_high = np.empty_like(dens_array)
        
        for i, dens in tqdm.tqdm(enumerate(dens_array)):
            # Gather all the radii at this mass point
            press_at_dens = []
            for mass, radius in zip(n, p):
                press_at_dens.append(np.interp(dens, mass, radius))
            press_at_dens = np.array(press_at_dens)
            
            # Construct 95% credible interval
            low, med, high = report_credible_interval(press_at_dens, hdi_prob=0.95)

            # Save
            press_low[i] = med - low
            press_high[i] = med + high
            
        # Now to plotting
        plt.fill_between(dens_array,
                         press_low,
                         press_high,
                         alpha=ALPHA,
                         label=translation_dict[outdir],
                         color=COLORS_DICT[outdir],
                         )
        
        # Also plot the outer lines
        plt.plot(dens_array, press_low, lw=2.0, color=COLORS_DICT[outdir])
        plt.plot(dens_array, press_high, lw=2.0, color=COLORS_DICT[outdir])
        
        # Make nice
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
        plt.xlim(n_min, n_max)
        # plt.ylim(0.1, 500.0)
        plt.yscale('log')
        
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        
    save_name = os.path.join("./figures", f"contour_pressures{filename_suffix}.png")
    print(f"Saving to {save_name}")
    plt.savefig(save_name, bbox_inches = "tight", dpi=300)
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()


if __name__ == "__main__":
    # Generate plots without the "only" run (original behavior)
    print("Generating plots without GW231109_only run...")
    make_contour_plot_radii(include_only=False)
    make_contour_plot_pressures(include_only=False)
    make_comparison_plot(include_only=False)
    
    # Generate plots with the "only" run (new behavior)
    print("Generating plots with GW231109_only run...")
    make_contour_plot_radii(include_only=True)
    make_contour_plot_pressures(include_only=True)
    make_comparison_plot(include_only=True)
    
    # Generate master plots for all runs -- choose below which outdirs
    
    # make_masterplots("outdir_GW170817")
    # make_masterplots("outdir_GW231109")
    make_masterplots("outdir_GW231109_l10000")