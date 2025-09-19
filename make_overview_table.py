"""
Automate creating an overview of the runs performed
"""

import os
import copy
import json
import pandas as pd
from datetime import datetime
import pytz

import utils

DEFAULT_PRIORS_TO_KEEP_ALIGNED = ["chirp_mass",
                                  "mass_ratio",
                                  "chi_1",
                                  "chi_2",
                                  "lambda_1",
                                  "lambda_2"
                                  ]

DEFAULT_PRIORS_TO_KEEP_PRECESSING = copy.deepcopy(DEFAULT_PRIORS_TO_KEEP_ALIGNED)
DEFAULT_PRIORS_TO_KEEP_PRECESSING[DEFAULT_PRIORS_TO_KEEP_PRECESSING.index("chi_1")] = "a_1"
DEFAULT_PRIORS_TO_KEEP_PRECESSING[DEFAULT_PRIORS_TO_KEEP_PRECESSING.index("chi_2")] = "a_2"

def fetch_table_information_single_run(source_dir: list[str]) -> dict:
    """
    Get the relevant information we need out of a single run
    """
    
    information_dict = {}
    
    # Sampling time
    sampling_time = utils.fetch_sampling_time(source_dir)
    information_dict["sampling_time"] = sampling_time
    
    # Log bayes factor
    log_bayes_factor = utils.fetch_log_bayes_factor(source_dir)
    information_dict["log_bayes_factor"] = log_bayes_factor
    
    waveform_approximant = utils.fetch_waveform_approximant(source_dir)
    information_dict["waveform_approximant"] = waveform_approximant
    
    # Get the full path and append it to the dictionary
    source_dir = os.path.abspath(source_dir)
    posterior_filename = utils.fetch_posterior_filename(source_dir)
    print(f"Fetching information from {posterior_filename}")
    information_dict["source_dir"] = source_dir
    
    # Get all the priors
    priors = utils.fetch_raw_priors(source_dir)
    
    # TODO: now we are assuming we never want this list to be user-defined, that sounds reasonable, but then this might eb made nicer?
    if "phi_jl" in priors:
        priors_to_keep = DEFAULT_PRIORS_TO_KEEP_PRECESSING
    else:
        priors_to_keep = DEFAULT_PRIORS_TO_KEEP_ALIGNED
    
    # Only keep the priors that we were told to keep
    priors = {k: v for k, v in priors.items() if k in priors_to_keep}
    information_dict["priors"] = priors
    
    # The values that were not saved as priors were fixed and therefore not found, fetch them using a different function
    fixed_params_keys = [k for k in priors_to_keep if k not in list(priors.keys())]
    fixed_params_dict = utils.fetch_fixed_parameters(source_dir, fixed_params_keys)
    information_dict["fixed_parameters"] = fixed_params_dict
    
    return information_dict

def extract_prior_info(prior_dict):
    """Extract prior name and bounds from a prior dictionary"""
    if not prior_dict:
        return "N/A"
    
    prior_name = prior_dict.get("__name__", "Unknown")
    kwargs = prior_dict.get("kwargs", {})
    
    if "minimum" in kwargs and "maximum" in kwargs:
        min_val = kwargs["minimum"]
        max_val = kwargs["maximum"]
        return f"{prior_name} [{min_val:.3f}, {max_val:.3f}]"
    elif "peak" in kwargs:  # For DeltaFunction priors
        peak_val = kwargs["peak"]
        return f"{prior_name} (peak={peak_val:.1f})"
    else:
        return f"{prior_name}"

def get_parameter_info(run_data, param_name, fixed_param_name=None):
    """Get parameter info from either priors or fixed parameters"""
    priors = run_data.get("priors", {})
    fixed_params = run_data.get("fixed_parameters", {})
    
    # Check priors first
    if param_name in priors:
        return extract_prior_info(priors[param_name])
    
    # Check fixed parameters if provided
    if fixed_param_name and fixed_param_name in fixed_params:
        value = fixed_params[fixed_param_name]
        return f"Fixed: {value:.3f}"
    
    return "N/A"

def get_spin_parameter_info(run_data):
    """Get spin parameter info, handling both chi_1 and a_1 naming"""
    priors = run_data.get("priors", {})
    
    # Check for chi_1 first, then a_1
    if "chi_1" in priors:
        return extract_prior_info(priors["chi_1"])
    elif "a_1" in priors:
        return extract_prior_info(priors["a_1"])
    else:
        return "N/A"

def get_lambda_parameter_info(run_data):
    """Get lambda parameter info, combining lambda_1 and lambda_2 bounds"""
    priors = run_data.get("priors", {})
    
    lambda_1 = priors.get("lambda_1", {})
    lambda_2 = priors.get("lambda_2", {})
    
    if not lambda_1 or not lambda_2:
        return "N/A"
    
    # Both should be uniform, so extract bounds
    kwargs_1 = lambda_1.get("kwargs", {})
    kwargs_2 = lambda_2.get("kwargs", {})
    
    if "minimum" in kwargs_1 and "maximum" in kwargs_1:
        min_val = kwargs_1["minimum"]
        max_val = kwargs_1["maximum"]
        prior_name = lambda_1.get("__name__", "Uniform")
        return f"{prior_name} [{min_val:.0f}, {max_val:.0f}]"
    
    return "N/A"

def shorten_directory_name(full_path):
    """Shorten directory name for display"""
    # Extract just the run name from the full path
    return full_path.split('/')[-1]

def create_markdown_table(df):
    """Convert DataFrame to markdown table format"""
    # Create a copy for display
    df_display = df.copy()
    
    # Reorder columns for better display, including index and full directory
    display_cols = ['index', 'waveform', 'chirp_mass_prior', 'mass_ratio_prior', 
                   'a_1_prior', 'lambda_prior', 'log_bayes_factor', 'sampling_time_hrs', 'directory']
    df_display = df_display[display_cols]
    
    # Rename columns for better display
    df_display.columns = ['#', 'Waveform', 'Chirp Mass Prior', 'Mass Ratio Prior',
                         'Spin Prior', 'Lambda Prior', 'Log Bayes Factor', 'Sampling Time (hrs)', 'Directory']
    
    # Convert to markdown
    markdown_table = df_display.to_markdown(index=False, tablefmt='github')
    return markdown_table

def create_readme(df):
    """Create a README.md file with the formatted table"""
    
    # Get current timestamp in CEST
    cest = pytz.timezone('Europe/Berlin')  # CEST timezone
    current_time = datetime.now(cest)
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S CEST")
    
    markdown_content = f"""# S250818k investigations

## Overview of runs

Generated automatically on: {timestamp}

## Run Overview Table

{create_markdown_table(df)}
"""

    readme_path = "./README.md"
    with open(readme_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Created README with formatted table: {readme_path}")
        
def get_all_runs(base_dir_list: list[str],
                 save_JSON: bool = True,
                 only_prod: bool = True):
    all_runs_information_dict = {}
    for base_dir in base_dir_list:
        source_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ["outdir", "data"]]
        print(f"For base directory {base_dir}, found the source directories:")
        print(f"    {source_dirs}")
        
        if only_prod:
            source_dirs = [d for d in source_dirs if "prod" in d]
            print(f"Only source dirs with prod in them:")
            print(f"    {source_dirs}")
        
        for source_dir in source_dirs:
            source_dir = os.path.join(base_dir, source_dir)
            if utils.fetch_posterior_filename(source_dir) is None:
                print(f"Skipping {source_dir}, no posterior file found")
                continue
            information_dict = fetch_table_information_single_run(source_dir)
            all_runs_information_dict[source_dir] = information_dict
            
    if save_JSON:
        # Use json to save the dictionary
        with open("./overview/all_runs_information.json", "w") as f:
            json.dump(all_runs_information_dict, f, indent=4)
            
    return all_runs_information_dict

def make_overview_table(all_runs_information_dict: dict = None,
                        output: str = "./overview/overview_table.csv"):
    
    """
    Make an overview table from the information dictionary
    """
    # If no data provided, load from JSON file
    if all_runs_information_dict is None:
        json_path = "./overview/all_runs_information.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find {json_path}")
        
        with open(json_path, 'r') as f:
            all_runs_information_dict = json.load(f)
    
    # Prepare data for DataFrame
    table_data = []
    
    for source_dir, run_data in all_runs_information_dict.items():
        # Convert sampling time to hours (assuming it's in seconds)
        sampling_time_raw = run_data.get('sampling_time', 0)
        sampling_time_hrs = sampling_time_raw / 3600.0 if sampling_time_raw else 0
        
        row = {
            'waveform': run_data.get('waveform_approximant', 'N/A'),
            'chirp_mass_prior': get_parameter_info(run_data, 'chirp_mass'),
            'mass_ratio_prior': get_parameter_info(run_data, 'mass_ratio'),
            'a_1_prior': get_spin_parameter_info(run_data),
            'lambda_prior': get_lambda_parameter_info(run_data),
            'log_bayes_factor': run_data.get('log_bayes_factor', 'N/A'),
            'sampling_time_hrs': f"{sampling_time_hrs:.2f}" if sampling_time_hrs > 0 else "N/A",
            'directory': source_dir
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by directory for consistent ordering
    df = df.sort_values('directory')
    
    # Add index column as first column
    df.insert(0, 'index', range(1, len(df) + 1))
    
    # Save to CSV
    df.to_csv(output, index=False)
    
    # Create README with formatted table
    create_readme(df)
    
    print(f"Successfully created overview table with {len(df)} runs")
    print(f"Saved CSV to: {output}")
    
    return df

    
def main():
    base_dir_list = ["/work/wouters/GW231109/",
                     "/work/puecher/S231109/bw_runs_debug/",
                     "/work/puecher/S231109/third_gen_runs/"
                     ]
    
    # First, get all runs and save to JSON
    all_runs_information_dict = get_all_runs(base_dir_list, only_prod=False)
    
    # Then, create the CSV overview table and README
    make_overview_table(all_runs_information_dict)
    
if __name__ == "__main__":
    main()