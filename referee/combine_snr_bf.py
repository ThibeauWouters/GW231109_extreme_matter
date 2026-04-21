"""
Combined table: ln B^signal_noise, network optimal SNR, network matched-filter SNR.
Sorted by log BF descending.
"""

import json
import numpy as np
import arviz

SNR_INPUT  = "./data/snr_samples.json"
BF_INPUT   = "./data/logbf_results.json"
HDI_PROB   = 0.95
USE_LOG10  = False   # True → divide ln B by ln(10) and show log10(B); False → keep ln(B)


def main():
    print("Combined SNR and Bayes factor results: \n\n\n")
    with open(SNR_INPUT, "r") as f:
        snr_data = json.load(f)
    with open(BF_INPUT, "r") as f:
        bf_data = json.load(f)

    all_runs  = sorted(set(snr_data) | set(bf_data))
    pct       = int(HDI_PROB * 100)
    bf_label  = "log10(B)" if USE_LOG10 else "ln(B)"
    bf_factor = 1.0 / np.log(10) if USE_LOG10 else 1.0

    rows = []
    for run in all_runs:
        lbf  = bf_data.get(run)
        if lbf is not None:
            lbf = lbf * bf_factor
        snrs = snr_data.get(run, {})
        km   = {k.lower(): k for k in snrs}

        h1o = km.get("h1_optimal_snr")
        l1o = km.get("l1_optimal_snr")
        h1m = km.get("h1_matched_filter_snr")
        l1m = km.get("l1_matched_filter_snr")

        opt_med = opt_lo = opt_hi = None
        mf_med  = mf_lo  = mf_hi  = None

        if h1o and l1o:
            arr = np.sqrt(np.array(snrs[h1o])**2 + np.array(snrs[l1o])**2)
            opt_med = np.median(arr)
            opt_lo, opt_hi = arviz.hdi(arr, hdi_prob=HDI_PROB)

        mf_max = None
        if h1m and l1m:
            arr = np.sqrt(np.array(snrs[h1m])**2 + np.array(snrs[l1m])**2)
            mf_med = np.median(arr)
            mf_lo, mf_hi = arviz.hdi(arr, hdi_prob=HDI_PROB)
            mf_max = np.max(arr)

        rows.append((run, lbf, opt_med, opt_lo, opt_hi, mf_med, mf_lo, mf_hi, mf_max))

    rows.sort(key=lambda r: r[1] if r[1] is not None else -np.inf, reverse=True)

    best_snr_idx = None
    best_snr_val = -np.inf
    for i, r in enumerate(rows):
        if r[8] is not None and r[8] > best_snr_val:
            best_snr_val = r[8]
            best_snr_idx = i

    col = max(len(r[0]) for r in rows) + 2

    def fb(v):
        return f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"

    def fs(med, lo, hi):
        if med is None:
            return f"{'N/A':<26}"
        return f"{med:6.3f}-{med-lo:.3f}+{hi-med:.3f}"

    def fmax(v):
        return f"{v:>10.3f}" if v is not None else f"{'N/A':>10}"

    snr_col_hdr = f"net SNR opt (med [{pct}% HDI])"
    header = f"{'Run':<{col}}  {bf_label:>10}  {snr_col_hdr:<26}  {'net SNR mf  (med [' + str(pct) + '% HDI])':<26}  {'mf SNR max':>10}  "
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for i, (run, lbf, om, ol, oh, mm, ml, mh, mx) in enumerate(rows):
        arrow = " <--" if i == best_snr_idx else ""
        print(f"{run:<{col}}  {fb(lbf)}  {fs(om, ol, oh)}  {fs(mm, ml, mh)}  {fmax(mx)}{arrow}")
    print(sep)
    print(f"  {len(rows)} run(s)   |   N/A = not present in that file")


if __name__ == "__main__":
    main()
