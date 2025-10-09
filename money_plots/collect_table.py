"""Collect EOS parameters from multiple jester inference results and generate tables.

This script collects MTOV, R14, and p(3nsat) parameters from multiple inference
output directories, calculates 90% credible intervals, and outputs results as
both JSON and LaTeX table formats.

Inspired by money_plots_snellius.py parameter calculation functions.

Usage:
    Modify the main() function to specify directories, then run:
    python collect_table.py
"""

import numpy as np
import os
import json
import arviz

# Import shared EOS loading utilities
from eos_utils import load_eos_data

# Labels and colors from money_plots_snellius.py
LABELS_DICT = {"outdir": "Prior",
               "outdir_radio": "Heavy PSRs",
               "outdir_GW170817": "GW170817",
               "outdir_GW190425": "GW190425",
               "outdir_GW231109": "GW231109 (default)",
               "outdir_GW231109_gaussian": "GW231109 (Gaussian)",
               "outdir_GW231109_double_gaussian": "GW231109 (double Gaussian)",
               "outdir_GW231109_quniv": "GW231109 (QUR)",
               "outdir_GW231109_s040": "GW231109 ($\\chi_i \\leq 0.4$)",
               "outdir_GW231109_XAS": "GW231109 (\\texttt{XAS})",
               "outdir_GW170817_GW231109": "GW170817+GW231109",
               "outdir_GW170817_GW190425": "GW170817+GW190425",
               "outdir_GW170817_GW190425_GW231109": "GW170817+GW190425+GW231109",
               "outdir_ET_AS": "ET",
               }

def report_credible_interval(values: np.array,
                             hdi_prob: float = 0.90,
                             verbose: bool = False) -> tuple:
    """Calculate credible intervals for given values."""
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob=hdi_prob)

    low_err = med - low
    high_err = high - med

    if verbose:
        print(f"{med:.2f}-{low_err:.2f}+{high_err:.2f} (at {hdi_prob} HDI prob)")
    return low_err, med, high_err

def calculate_eos_parameters(data_dict: dict) -> dict:
    """Calculate MTOV, R14, and p(3nsat) from EOS data.

    Args:
        data_dict: Dictionary containing EOS data

    Returns:
        dict: Dictionary with parameter arrays
    """
    m, r = data_dict['masses'], data_dict['radii']
    n, p = data_dict['densities'], data_dict['pressures']

    # Calculate derived parameters
    MTOV_list = np.array([np.max(mass) for mass in m])
    R14_list = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    p3nsat_list = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])

    return {
        'MTOV': MTOV_list,
        'R14': R14_list,
        'p3nsat': p3nsat_list
    }

def collect_parameters_from_directories(directories: list, hdi_prob: float = 0.90, add_prior: bool = False) -> dict:
    """Collect EOS parameters from multiple directories and calculate credible intervals.

    Args:
        directories: List of directory paths containing EOS samples
        hdi_prob: Credible interval probability (default 0.90)
        add_prior: If False, skip the "outdir" (Prior) directory (default False)

    Returns:
        dict: Nested dictionary with results for each directory and parameter
    """
    results = {}

    # Filter out "outdir" if add_prior is False
    if not add_prior:
        directories = [d for d in directories if not os.path.basename(d.rstrip('/')) == 'outdir']

    for outdir in directories:
        # Check if directory exists
        if not os.path.exists(outdir):
            print(f"Warning: Directory {outdir} does not exist. Skipping...")
            continue

        try:
            # Load data
            data = load_eos_data(outdir)

            # Calculate parameters
            parameters = calculate_eos_parameters(data)

            # Get directory basename and map to label
            dir_basename = os.path.basename(outdir.rstrip('/'))
            label = LABELS_DICT.get(dir_basename, dir_basename)

            print(f"Processing {label} ({dir_basename})...")

            # Calculate credible intervals for each parameter
            param_results = {}
            for param_name, param_values in parameters.items():
                low_err, med, high_err = report_credible_interval(param_values, hdi_prob=hdi_prob, verbose=True)
                width = high_err + low_err
                param_results[param_name] = {
                    'median': float(med),
                    'lower_error': float(low_err),
                    'upper_error': float(high_err),
                    'width': float(width),
                    'credible_interval': f"{med:.2f}^{{+{high_err:.2f}}}_{{-{low_err:.2f}}}"
                }
                print(f"  {param_name}: {param_results[param_name]['credible_interval']}")

            results[dir_basename] = {
                'label': label,
                'parameters': param_results
            }

        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {outdir}: {e}")
            continue

    return results

def save_results_to_json(results: dict, filename: str = "eos_parameters_table.json"):
    """Save results to JSON file.

    Args:
        results: Results dictionary from collect_parameters_from_directories
        filename: Output JSON filename
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def json_to_latex_table(json_filename: str, output_filename: str = "eos_parameters_table.tex", add_prior: bool = False):
    """Convert JSON results to LaTeX table format.

    Formats entries with the smallest width (uncertainty) in bold for each parameter.
    Organizes results into groups with whitespace separation.

    Args:
        json_filename: Input JSON filename
        output_filename: Output LaTeX filename
        add_prior: If False, skip the "outdir" (Prior) directory (default False)
    """
    # Load JSON data
    with open(json_filename, 'r') as f:
        results = json.load(f)

    # Define group organization
    if add_prior:
        group_order = [
            # Group 1: Prior and radio timing
            ["outdir", "outdir_radio"],
            # Group 2: GW231109 variations
            ["outdir_GW231109", "outdir_GW231109_gaussian", "outdir_GW231109_double_gaussian",
             "outdir_GW231109_quniv", "outdir_GW231109_s040", "outdir_GW231109_XAS"],
            # Group 3: Individual GW events
            ["outdir_GW170817", "outdir_GW190425"],
            # Group 4: Two-event combinations
            ["outdir_GW170817_GW231109", "outdir_GW170817_GW190425"],
            # Group 5: Three-event combination
            ["outdir_GW170817_GW190425_GW231109"]
        ]
    else:
        group_order = [
            # Group 1: Radio timing only
            ["outdir_radio"],
            # Group 2: GW231109 variations
            ["outdir_GW231109", "outdir_GW231109_gaussian", "outdir_GW231109_double_gaussian",
             "outdir_GW231109_quniv", "outdir_GW231109_s040", "outdir_GW231109_XAS"],
            # Group 3: Individual GW events
            ["outdir_GW170817", "outdir_GW190425"],
            # Group 4: Two-event combinations
            ["outdir_GW170817_GW231109", "outdir_GW170817_GW190425"],
            # Group 5: Three-event combination
            ["outdir_GW170817_GW190425_GW231109"]
        ]

    # Find entries with minimum width for each parameter
    min_widths = {}
    for param in ['MTOV', 'R14', 'p3nsat']:
        min_width = float('inf')
        min_dir = None
        for dir_basename, data in results.items():
            width = data['parameters'][param]['width']
            if width < min_width:
                min_width = width
                min_dir = dir_basename
        min_widths[param] = min_dir

    # Start LaTeX table
    latex_content = []
    latex_content.append("\\begin{tabular}{l@{\\hspace{1.5cm}}c@{\\hspace{1.5cm}}c@{\\hspace{1.5cm}}c}")
    latex_content.append("\\toprule\\toprule")
    latex_content.append("Dataset & $M_{\\mathrm{TOV}}$ [$M_{\\odot}$] & $R_{1.4}$ [km] & $p(3n_{\\mathrm{sat}})$ [MeV fm$^{-3}$] \\\\")
    latex_content.append("\\midrule")

    # Add data rows organized by groups
    for group_idx, group in enumerate(group_order):
        for item_idx, dir_basename in enumerate(group):
            if dir_basename not in results:
                continue  # Skip if directory wasn't processed

            data = results[dir_basename]
            label = data['label']
            params = data['parameters']

            # Format each parameter with credible interval, bold if minimum width
            mtov = params['MTOV']['credible_interval']
            r14 = params['R14']['credible_interval']
            p3nsat = params['p3nsat']['credible_interval']

            # Escape special characters for LaTeX
            label_escaped = label.replace('+', '$+$')

            latex_content.append(f"{label_escaped} & ${mtov}$ & ${r14}$ & ${p3nsat}$ \\\\")

        # Add spacing and horizontal line between groups (except after the last group)
        if group_idx < len(group_order) - 1:
            latex_content.append("\\addlinespace")
            latex_content.append("\\hline")
            latex_content.append("\\addlinespace")

    # End table
    latex_content.append("\\bottomrule\\bottomrule")
    latex_content.append("\\end{tabular}")

    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"LaTeX table saved to {output_filename}")

def json_to_latex_table_r14_only(json_filename: str, output_filename: str = "eos_r14_table.tex"):
    """Convert JSON results to LaTeX table showing only R1.4 for selected datasets.

    Includes: GW231109 (default), GW170817, GW190425, and their combinations.

    Args:
        json_filename: Input JSON filename
        output_filename: Output LaTeX filename
    """
    # Load JSON data
    with open(json_filename, 'r') as f:
        results = json.load(f)

    # Define specific datasets to include in order
    selected_datasets = [
        # Individual events
        "outdir_GW231109",
        "outdir_GW170817",
        "outdir_GW190425",
        # Two-event combinations
        "outdir_GW170817_GW231109",
        "outdir_GW170817_GW190425",
        # Three-event combination
        "outdir_GW170817_GW190425_GW231109"
    ]

    # Start LaTeX table
    latex_content = []
    latex_content.append("\\begin{tabular}{l@{\\hspace{1.5cm}}c}")
    latex_content.append("\\toprule\\toprule")
    latex_content.append("Dataset & $R_{1.4}$ [km] \\\\")
    latex_content.append("\\midrule")

    # Add data rows
    for dir_basename in selected_datasets:
        if dir_basename not in results:
            continue  # Skip if directory wasn't processed

        data = results[dir_basename]
        label = data['label']
        r14 = data['parameters']['R14']['credible_interval']

        # Escape special characters for LaTeX
        label_escaped = label.replace('+', '$+$')

        latex_content.append(f"{label_escaped} & ${r14}$ \\\\")
        latex_content.append("\\addlinespace")

    # End table
    latex_content.append("\\bottomrule\\bottomrule")
    latex_content.append("\\end{tabular}")

    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"R1.4-only LaTeX table saved to {output_filename}")

def main(add_prior: bool = False):
    """Main function - configure directories and generate tables.

    Args:
        add_prior: If False, skip the "outdir" (Prior) directory (default False)
    """

    # =======================================================================
    # Configure directories to process
    # =======================================================================

    # Organize directories into logical groups
    directories = [
        # Group 1: Prior and radio timing
        "../jester/outdir",
        "../jester/outdir_radio",

        # Group 2: Individual GW events
        "../jester/outdir_GW170817",
        "../jester/outdir_GW190425",

        # Group 3: GW231109 variations
        "../jester/outdir_GW231109",
        "../jester/outdir_GW231109_gaussian",
        "../jester/outdir_GW231109_double_gaussian",
        "../jester/outdir_GW231109_quniv",
        "../jester/outdir_GW231109_s040",
        "../jester/outdir_GW231109_XAS",

        # Group 4: Combinations
        "../jester/outdir_GW170817_GW231109",
        "../jester/outdir_GW170817_GW190425",
        "../jester/outdir_GW170817_GW190425_GW231109",
    ]

    print("EOS Parameter Table Generator")
    print("=" * 50)
    if add_prior:
        print(f"Processing {len(directories)} directories (including Prior)...")
    else:
        print(f"Processing {len(directories)} directories (excluding Prior)...")

    # Collect parameters and calculate credible intervals
    results = collect_parameters_from_directories(directories, hdi_prob=0.90, add_prior=add_prior)

    if len(results) == 0:
        print("Error: No valid results found!")
        return

    # Save to JSON
    json_filename = "eos_parameters_table.json"
    save_results_to_json(results, json_filename)

    # Convert to LaTeX table
    latex_filename = "eos_parameters_table.tex"
    json_to_latex_table(json_filename, latex_filename, add_prior=add_prior)

    # Generate R1.4-only table for selected datasets
    r14_latex_filename = "eos_r14_table.tex"
    json_to_latex_table_r14_only(json_filename, r14_latex_filename)

    print(f"\nProcessing complete!")
    print(f"JSON output: {json_filename}")
    print(f"LaTeX output: {latex_filename}")
    print(f"R1.4-only LaTeX output: {r14_latex_filename}")

if __name__ == "__main__":
    main()