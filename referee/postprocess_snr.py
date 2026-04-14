"""
Postprocessing script for SNR samples (run locally).
Takes the JSON file produced by check_snr.py and prints median + 95% HDI
for H1_optimal_SNR and L1_optimal_SNR (and matched-filter equivalents) per run.
"""

import json
import numpy as np
import arviz
import argparse

DEFAULT_INPUT = "./data/snr_samples.json"

KEYS_OF_INTEREST = [
    "H1_optimal_snr",
    "L1_optimal_snr",
    "H1_matched_filter_snr",
    "L1_matched_filter_snr",
    "network_optimal_snr",
    "network_matched_filter_snr",
]

def summarize(samples, hdi_prob=0.95):
    arr = np.array(samples)
    median = np.median(arr)
    hdi = arviz.hdi(arr, hdi_prob=hdi_prob)
    return median, hdi[0], hdi[1]

def main():
    parser = argparse.ArgumentParser(description="Postprocess SNR samples from check_snr.py output.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the JSON file produced by check_snr.py")
    parser.add_argument("--hdi", type=float, default=0.95, help="HDI probability (default: 0.95)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} run(s) from {args.input}\n")

    for run_name, snrs in all_results.items():
        print(f"=== {run_name} ===")
        available_keys = list(snrs.keys())

        # Try to match case-insensitively
        key_map = {k.lower(): k for k in available_keys}

        printed_any = False
        for key in KEYS_OF_INTEREST:
            actual_key = key_map.get(key.lower())
            if actual_key is None:
                continue
            samples = snrs[actual_key]
            median, lo, hi = summarize(samples, hdi_prob=args.hdi)
            print(f"  {actual_key}: median={median:.3f}, {int(args.hdi*100)}% HDI=[{lo:.3f}, {hi:.3f}]")
            printed_any = True

        if not printed_any:
            print(f"  No matching SNR keys found. Available: {available_keys}")
        print()

if __name__ == "__main__":
    main()
