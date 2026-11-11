"""
Postprocessing script for log Bayes factor results (run locally).
Takes the JSON file produced by check_logbf.py and prints the ln B^signal_noise
values sorted from highest to lowest, together with a brief summary.
"""

import json
import numpy as np
import argparse

DEFAULT_INPUT = "./data/logbf_results.json"


def print_table(results):
    """Print results sorted by log Bayes factor (descending)."""
    sorted_items = sorted(results.items(), key=lambda kv: kv[1], reverse=True)

    col_width = max(len(k) for k in results) + 2
    header = f"{'Run':<{col_width}}  ln B^signal_noise"
    print(header)
    print("-" * len(header))
    for run_name, lbf in sorted_items:
        print(f"{run_name:<{col_width}}  {lbf:.4f}")
    print()

    values = list(results.values())
    print(f"  N runs : {len(values)}")
    print(f"  Min    : {min(values):.4f}")
    print(f"  Max    : {max(values):.4f}")
    print(f"  Median : {np.median(values):.4f}")
    print(f"  Mean   : {np.mean(values):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess log Bayes factor results from check_logbf.py output."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the JSON file produced by check_logbf.py",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    if not results:
        print(f"No results found in {args.input}.")
        return

    print(f"Loaded {len(results)} run(s) from {args.input}\n")
    print_table(results)


if __name__ == "__main__":
    main()
