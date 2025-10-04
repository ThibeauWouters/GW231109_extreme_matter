import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 18,
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})
from astropy.time import Time

def read_observations_file(
    filename, merger_time='2017-08-17T12:41:04'
):
    """
    Read observation data from ASCII file.
    Expected format:
    2017-08-18T00:00:00.000 ps1::g 17.41000 0.02000
    2017-08-18T00:00:00.000 ps1::r 17.56000 0.04000
    ...
    
    Parameters:
    -----------
    filename : str
        Path to observations file
    merger_time : str
        Time of merger in ISO format (YYYY-MM-DDTHH:MM:SS)
    distance_mpc : float
        Distance to source in Mpc for converting apparent to absolute magnitude
    
    Returns a dictionary with filter names as keys and DataFrames with 
    'time' and 'magnitude' columns as values.
    """
    # Parse merger time
    merger_dt = Time(merger_time, format='isot').mjd
    
    # Read the file
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                timestamp = parts[0]
                filter_name = parts[1]
                app_mag = float(parts[2])
                error = float(parts[3])
                
                # Skip infinite errors
                #if error == float('inf') or np.isinf(error):
                #    continue
                
                # Convert timestamp to datetime
                obs_dt = Time(timestamp, format='isot').mjd
                
                # Calculate time since merger in days
                time_since_merger = obs_dt - merger_dt
                
                # Convert apparent magnitude to absolute magnitude
                # M = m - 5*log10(d) - 25, where d is in Mpc
                abs_mag = app_mag
                
                data.append({
                    'filter': filter_name,
                    'time': time_since_merger,
                    'magnitude': abs_mag,
                    'error': error
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Group by filter
    obs_data = {}
    for filter_name in df['filter'].unique():
        mask = df['filter'] == filter_name
        obs_data[filter_name] = df[mask][['time', 'magnitude', 'error']].reset_index(drop=True)
    
    return obs_data

def read_lines_file(filename, filts):
    """
    Read line data from ASCII file using pandas.
    Expected format:
    time u g r i z y J K
    0.5 -16.2 -16.0 -15.8 ...
    1.0 -15.8 -15.5 -15.3 ...
    ...
    
    Returns a dictionary with filter names as keys and DataFrames with 
    'time' and 'magnitude' columns as values.
    """
    # Read the file
    df = pd.read_csv(filename, delimiter=' ', comment='#', header=0)
    
    # Get column names (first column should be time, rest are filters)
    time_col = 'sample_times'
    filter_cols = [x for x in df.columns if x != 'sample_times']
    
    # Create dictionary with filter name as key
    data = {}
    for filter_name in filts:
        data[filter_name] = pd.DataFrame({
            'time': df[time_col],
            'magnitude': df[f'{filter_name}_median'],
            'magnitude_low': df[f'{filter_name}_2p5pt'],
            'magnitude_high': df[f'{filter_name}_97p5pt'],
            'magnitude_170817': df[f'{filter_name}_170817_median'],
            'magnitude_170817_low': df[f'{filter_name}_170817_2p5pt'],
            'magnitude_170817_high': df[f'{filter_name}_170817_97p5pt'],
        })
    
    return data

def plot_light_curves(lines_file, observations_file,
                     output_filename='light_curves.pdf',
                     n_cols=2, figsize=(7.5, 9), xlim=(0.3, 4.9), ylim=(25, 15),
                     n_model_lines=10):
    """
    Create multi-panel plot similar to the reference figure.
    
    Parameters:
    -----------
    lines_file : str
        Path to ASCII file containing line data in columnar format
        (time, filter1, filter2, ...)
    observations_file : str
        Path to ASCII file containing observation data points
    output_filename : str
        Output filename for the figure
    n_cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size (width, height)
    xlim : tuple
        X-axis limits
    ylim : tuple
        Y-axis limits
    n_model_lines : int
        Number of model lines to plot (for gradient effect)
    """
    # Get all unique panel names from both files
    all_panels = [
        'sdssu', 'ps1::g', 'ps1::r', 'ps1::i',
        'ps1::z', #'ps1::y',
        '2massj', #'2massh', '2massks'
    ]
    n_panels = len(all_panels)
    # Read data from both files
    lines_data = read_lines_file(lines_file, all_panels)
    obs_data = read_observations_file(observations_file)


    # Calculate grid dimensions
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_panels == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Create multiple lines with color gradient (simulating model ensemble)
    import seaborn as sns
    colors = sns.color_palette("rainbow", n_colors=n_panels)
    
    for idx, panel_name in enumerate(all_panels):
        ax = axes[idx]
        
        if idx == 0:
            ax.scatter(
                10., 10.,
                c='k', s=10, marker='o', 
                label='AT2017gfo')
            ax.plot(
                [10., 10.],
                [10., 10.], 
                color='k', linewidth=2., zorder=1,
                label='Estimation'
            )
            ax.plot(
                [10., 10.],
                [10., 10.], 
                color='k', linewidth=2., zorder=1,
                linestyle='--',
                label='Estimation at 40Mpc'
            )
            ax.legend(
                bbox_to_anchor=(-0.1, 1.02),
                loc='lower left', ncols=3,
                fontsize=14
            )
        
        # Plot lines (models/predictions) if available
        if panel_name in lines_data and not lines_data[panel_name].empty:
            df_line = lines_data[panel_name] 
            df_obs = obs_data[panel_name]
            obs_idx = np.where(np.isfinite(df_obs['error']))[0]
            noobs_idx = np.where(~np.isfinite(df_obs['error']))[0]

            ax.plot(
                df_line['time'], df_line['magnitude'], 
                color=colors[idx], linewidth=3., zorder=1
            )
            ax.fill_between(
                df_line['time'],
                df_line['magnitude_low'], 
                df_line['magnitude_high'], 
                color=colors[idx], linewidth=3., zorder=1,
                alpha=0.5
            )
            ax.plot(
                df_line['time'], df_line['magnitude_170817'],
                color=colors[idx], linewidth=3., zorder=1, linestyle='--'
            )
            ax.fill_between(
                df_line['time'],
                df_line['magnitude_170817_low'], 
                df_line['magnitude_170817_high'], 
                color=colors[idx], linewidth=3., zorder=1,
                alpha=0.5
            )
            if len(obs_idx) > 0:
                ax.errorbar(
                    df_obs['time'][obs_idx],
                    df_obs['magnitude'][obs_idx],
                    yerr=df_obs['error'][obs_idx],
                    c='k', markersize=3, fmt='o', 
                    capsize=5,
                    linewidth=1.5, zorder=2)
            if len(noobs_idx) > 0:
                ax.scatter(df_obs['time'][noobs_idx],
                           df_obs['magnitude'][noobs_idx],
                           c='white', s=20, marker='v', 
                           edgecolors='black', 
                           linewidth=1.5, zorder=2)
            ax.invert_yaxis()
        
        # Formatting
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(0.95, 0.95, panel_name.replace('_',':'), transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')
        
        # Set labels only for left column and bottom row
        if idx % n_cols == 0 and panel_name == 'ps1::z':
            ax.set_ylabel('Apparent magnitude', fontsize=20)
            ax.yaxis.set_label_coords(-0.15, 1.6)
            ax.set_yticks([15, 18, 21, 24])
            ax.set_yticklabels([15, 18, 21, 24])
        elif idx % n_cols == 0:
            ax.set_yticks([15, 18, 21, 24])
            ax.set_yticklabels([15, 18, 21, 24])
        else:
            ax.set_yticklabels([])
        
        if idx >= n_panels - n_cols:
            ax.set_xlabel('Time since merger (days)', fontsize=15)
        else:
            ax.set_xticklabels([])
        
        ax.grid(False)
        ax.tick_params(labelsize=13)
    
    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
    return fig

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        lines_file = sys.argv[1]
        obser_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'light_curves.pdf'
        
        plot_light_curves(lines_file, obser_file, output_file)
    else:
        print("Usage: python script.py <lines_file> [output_file]")
        print("\nLines file format (columnar):")
        print("time u g r i z y J K")
        print("0.5 -16.2 -16.0 -15.8 -15.6 -15.4 -15.2 -15.0 -14.8")
        print("1.0 -15.8 -15.5 -15.3 -15.1 -14.9 -14.7 -14.5 -14.3")
        print("...")
        print("\nOption 2 (columnar with NaN for missing data):")
        print("time u g r i")
        print("1.0 -16.0 nan -15.8 nan")
        print("2.0 nan -15.5 nan -15.2")
        print("\nExample:")
        print("python script.py lines.txt observations.txt output.pdf")
