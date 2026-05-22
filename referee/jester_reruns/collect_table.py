"""Collect EOS parameters from jester inference results and generate LaTeX tables.

Adapted from money_plots/collect_table.py for the referee jester reruns.
Results are loaded from <run_dir>/outdir/results.h5 via InferenceResult.

Usage:
    python collect_table.py
"""

import json
import os
import sys

import arviz
import numpy as np

sys.path.insert(0, "/home/twouters2/projects/jester_review/jester")
from jesterTOV.inference.result import InferenceResult
import jesterTOV.utils as jose_utils

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABELS_DICT = {
    "prior": "Prior",
    "radio": "Heavy PSRs",
    "GW170817": "GW170817",
    "GW190425": "GW190425",
    "GW231109": "GW231109 (default)",
    "GW231109_gaussian": "GW231109 (Gaussian)",
    "GW231109_double_gaussian": "GW231109 (double Gaussian)",
    "GW231109_double_gaussian_niu": "GW231109 (double Gaussian, Niu)",
    "GW231109_double_gaussian_niu_v2": "GW231109 (double Gaussian, Niu v2)",
    "GW231109_uniform": "GW231109 (uniform)",
    "GW231109_lquniv": "GW231109 (QUR)",
    "GW231109_highspin": r"GW231109 ($a_i \leq 0.4$)",
    "GW231109_XAS": r"GW231109 (\texttt{XAS})",
    "GW170817_GW190425": "GW170817+GW190425",
    "GW170817_GW231109": "GW170817+GW231109",
    "GW190425_GW231109": "GW190425+GW231109",
    "GW170817_GW190425_GW231109": "GW170817+GW190425+GW231109",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eos_data(run_dir: str) -> dict:
    """Load EOS data from <run_dir>/outdir/results.h5.

    Converts n and p from geometric units and removes NaN-containing samples.

    Returns dict with keys: masses, radii, lambdas, densities, pressures.
    """
    filepath = os.path.join(run_dir, "outdir", "results.h5")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    print(f"Loading {filepath}")
    result = InferenceResult.load(filepath)

    m = result.posterior["masses_EOS"]
    r = result.posterior["radii_EOS"]
    l = result.posterior["Lambdas_EOS"]
    n = result.posterior["n"]
    p = result.posterior["p"]

    # Remove NaN-containing samples
    valid = (
        ~np.any(np.isnan(m), axis=1)
        & ~np.any(np.isnan(r), axis=1)
        & ~np.any(np.isnan(l), axis=1)
    )
    n_dropped = len(m) - valid.sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} NaN samples ({100 * n_dropped / len(m):.1f}%)")
    m, r, l, n, p = m[valid], r[valid], l[valid], n[valid], p[valid]

    # Convert units: geometric → physical
    n = n / jose_utils.fm_inv3_to_geometric / 0.16   # → n/n_sat
    p = p / jose_utils.MeV_fm_inv3_to_geometric       # → MeV fm^{-3}

    return {"masses": m, "radii": r, "lambdas": l, "densities": n, "pressures": p}


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def format_to_sig_figs(value: float, sig_figs: int = 2) -> tuple:
    if value == 0:
        return "0", 0
    magnitude = int(np.floor(np.log10(abs(value))))
    decimal_places = max(0, sig_figs - magnitude - 1)
    return f"{value:.{decimal_places}f}", decimal_places


def report_credible_interval(values: np.ndarray, hdi_prob: float = 0.90) -> tuple:
    med = np.median(values)
    low, high = arviz.hdi(values, hdi_prob=hdi_prob)
    return med - low, med, high - med


def calculate_eos_parameters(data: dict) -> dict:
    m, r = data["masses"], data["radii"]
    n, p = data["densities"], data["pressures"]
    MTOV = np.array([mass.max() for mass in m])
    R14  = np.array([np.interp(1.4, mass, radius) for mass, radius in zip(m, r)])
    p3n  = np.array([np.interp(3.0, dens, press) for dens, press in zip(n, p)])
    return {"MTOV": MTOV, "R14": R14, "p3nsat": p3n}


# ---------------------------------------------------------------------------
# Collection + formatting
# ---------------------------------------------------------------------------

def collect_parameters(run_dirs: list, hdi_prob: float = 0.90) -> dict:
    results = {}
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir.rstrip("/"))
        try:
            data = load_eos_data(run_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {run_name}: {e}")
            continue
        except Exception as e:
            print(f"  Error processing {run_name}: {e}")
            continue

        label = LABELS_DICT.get(run_name, run_name)
        print(f"Processing {label} ...")
        params = calculate_eos_parameters(data)

        param_results = {}
        for name, values in params.items():
            low_err, med, high_err = report_credible_interval(values, hdi_prob)
            width = low_err + high_err

            if name == "R14":
                ci = f"{med:.2f}^{{+{high_err:.2f}}}_{{-{low_err:.2f}}}"
            elif name == "p3nsat":
                med_s, nd = format_to_sig_figs(med, 2)
                ci = f"{med_s}^{{+{high_err:.{nd}f}}}_{{-{low_err:.{nd}f}}}"
            else:  # MTOV
                ci = f"{med:.2f}^{{+{high_err:.2f}}}_{{-{low_err:.2f}}}"

            print(f"  {name}: {ci}")
            param_results[name] = {
                "median": float(med),
                "lower_error": float(low_err),
                "upper_error": float(high_err),
                "width": float(width),
                "credible_interval": ci,
            }

        results[run_name] = {"label": label, "parameters": param_results}

    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def save_json(results: dict, filename: str = "eos_parameters_table.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {filename}")


def make_full_table(json_file: str, out_file: str = "eos_parameters_table.tex"):
    with open(json_file) as f:
        results = json.load(f)

    group_order = [
        # Prior and heavy pulsars
        ["prior", "radio"],
        # GW231109 variations
        ["GW231109", "GW231109_gaussian", "GW231109_double_gaussian",
         "GW231109_double_gaussian_niu", "GW231109_double_gaussian_niu_v2",
         "GW231109_uniform", "GW231109_lquniv", "GW231109_highspin", "GW231109_XAS"],
        # Individual established events
        ["GW170817", "GW190425"],
        # Two-event combinations
        ["GW170817_GW190425", "GW170817_GW231109", "GW190425_GW231109"],
        # Three-event combination
        ["GW170817_GW190425_GW231109"],
    ]

    lines = [
        r"\begin{tabular}{l@{\hspace{1.5cm}}c@{\hspace{1.5cm}}c@{\hspace{1.5cm}}c}",
        r"\toprule\toprule",
        r"Dataset & $M_{\mathrm{TOV}}$ [$M_{\odot}$] & $R_{1.4}$ [km] & $p(3n_{\mathrm{sat}})$ [MeV fm$^{-3}$] \\",
        r"\midrule",
    ]

    for g_idx, group in enumerate(group_order):
        for run_name in group:
            if run_name not in results:
                continue
            data = results[run_name]
            label = data["label"].replace("+", "$+$")
            p = data["parameters"]
            lines.append(
                f"{label} & ${p['MTOV']['credible_interval']}$ & "
                f"${p['R14']['credible_interval']}$ & "
                f"${p['p3nsat']['credible_interval']}$ \\\\"
            )
        if g_idx < len(group_order) - 1:
            lines += [r"\addlinespace", r"\hline", r"\addlinespace"]

    lines += [r"\bottomrule\bottomrule", r"\end{tabular}"]
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved full table: {out_file}")


def make_r14_table(json_file: str, out_file: str = "eos_r14_table.tex"):
    with open(json_file) as f:
        results = json.load(f)

    selected = [
        "prior",
        "radio",
        "GW170817",
        "GW190425",
        "GW231109",
        "GW170817_GW190425",
        "GW170817_GW231109",
        "GW190425_GW231109",
        "GW170817_GW190425_GW231109",
    ]

    lines = [
        r"\begin{tabular}{l@{\hspace{1.5cm}}c}",
        r"\toprule\toprule",
        r"Dataset & $R_{1.4}$ [km] \\",
        r"\midrule",
    ]

    for run_name in selected:
        if run_name not in results:
            continue
        data = results[run_name]
        label = data["label"].replace("+", "$+$")
        r14 = data["parameters"]["R14"]["credible_interval"]
        lines.append(f"{label} & ${r14}$ \\\\")
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule\bottomrule", r"\end{tabular}"]
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved R1.4 table: {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_dirs = [
        # Prior and radio timing
        os.path.join(script_dir, "prior"),
        os.path.join(script_dir, "radio"),
        # Individual events
        os.path.join(script_dir, "GW170817"),
        os.path.join(script_dir, "GW190425"),
        # GW231109 variations
        os.path.join(script_dir, "GW231109"),
        os.path.join(script_dir, "GW231109_gaussian"),
        os.path.join(script_dir, "GW231109_double_gaussian"),
        os.path.join(script_dir, "GW231109_double_gaussian_niu"),
        os.path.join(script_dir, "GW231109_double_gaussian_niu_v2"),
        os.path.join(script_dir, "GW231109_uniform"),
        os.path.join(script_dir, "GW231109_lquniv"),
        os.path.join(script_dir, "GW231109_highspin"),
        os.path.join(script_dir, "GW231109_XAS"),
        # Combined events
        os.path.join(script_dir, "GW170817_GW190425"),
        os.path.join(script_dir, "GW170817_GW231109"),
        os.path.join(script_dir, "GW190425_GW231109"),
        os.path.join(script_dir, "GW170817_GW190425_GW231109"),
    ]

    print("=" * 60)
    print("EOS Parameter Table Generator (jester reruns)")
    print("=" * 60)

    results = collect_parameters(run_dirs, hdi_prob=0.90)

    if not results:
        print("No valid results found — exiting.")
        return

    json_file = os.path.join(script_dir, "eos_parameters_table.json")
    save_json(results, json_file)
    make_full_table(json_file, os.path.join(script_dir, "eos_parameters_table.tex"))
    make_r14_table(json_file, os.path.join(script_dir, "eos_r14_table.tex"))

    print("\nDone.")


if __name__ == "__main__":
    main()
