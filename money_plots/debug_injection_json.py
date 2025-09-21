#!/usr/bin/env python3
"""
Temporary debug script to investigate JSON structure for injection parameters.
This will be deleted once we understand the JSON format.
"""

import json
import os

def investigate_json_structure(filepath):
    """
    Load and print the structure of a JSON result file to understand
    where injection parameters are stored.
    """
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return

    print(f"Investigating JSON structure for: {filepath}")
    print("=" * 80)

    with open(filepath, 'r') as f:
        data = json.load(f)

    print("Top-level keys:")
    for key in data.keys():
        print(f"  - {key}")

    print("\nDetailed structure:")

    # Look for injection/truth parameters
    if 'injection_parameters' in data:
        print("\nFound 'injection_parameters':")
        injection_params = data['injection_parameters']
        if isinstance(injection_params, dict):
            for key, value in injection_params.items():
                print(f"  - {key}: {value}")
        else:
            print(f"  Type: {type(injection_params)}")
            print(f"  Value: {injection_params}")

    # Look for meta data
    if 'meta_data' in data:
        print("\nFound 'meta_data':")
        meta_data = data['meta_data']
        if isinstance(meta_data, dict):
            for key in meta_data.keys():
                print(f"  - {key}")
                if 'injection' in key.lower():
                    print(f"    -> {meta_data[key]}")

    # Look in posterior structure
    if 'posterior' in data:
        print("\nFound 'posterior':")
        posterior = data['posterior']
        if isinstance(posterior, dict):
            for key in posterior.keys():
                print(f"  - {key}")
                if key == 'content' and isinstance(posterior[key], dict):
                    print("    posterior content keys:")
                    for param_key in list(posterior[key].keys())[:10]:  # Show first 10
                        print(f"      - {param_key}")
                    if len(posterior[key].keys()) > 10:
                        print(f"      ... and {len(posterior[key].keys()) - 10} more")

    # Check all top-level keys for injection-related content
    print("\nScanning all keys for 'injection' or 'truth':")
    for key, value in data.items():
        if 'injection' in key.lower() or 'truth' in key.lower():
            print(f"  Found: {key}")
            if isinstance(value, dict) and len(value) < 20:
                for sub_key, sub_value in value.items():
                    print(f"    - {sub_key}: {sub_value}")
            else:
                print(f"    Type: {type(value)}, Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")

# Example usage - this would be run on the remote machine
if __name__ == "__main__":
    # Example path (will work when run on remote)
    example_path = "/work/puecher/S231109/third_gen_runs/et_run_alignedspin/outdir/ET_gw231109_injection_alignedspin_result.json"

    print("This debug script should be run on the remote machine with access to:")
    print(f"  {example_path}")
    print("\nTo run on remote:")
    print("  python debug_injection_json.py")

    # If running locally, will show the expected structure
    print("\nNote: This file will be deleted once we understand the JSON structure.")