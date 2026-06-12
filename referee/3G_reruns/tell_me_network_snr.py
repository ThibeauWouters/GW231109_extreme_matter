import os
import re
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

def parse_snrs(log_path):
    """Return list of (ifo_name, optimal_snr) pairs from a bilby injection log."""
    ifo_pattern = re.compile(r"Injected signal in ([^:]+):")
    snr_pattern = re.compile(r"optimal SNR\s*=\s*([\d.]+)")

    results = []
    current_ifo = None
    with open(log_path) as f:
        for line in f:
            m = ifo_pattern.search(line)
            if m:
                current_ifo = m.group(1).strip()
                continue
            m = snr_pattern.search(line)
            if m and current_ifo is not None:
                results.append((current_ifo, float(m.group(1))))
                current_ifo = None
    return results

def main():
    for network in ["et", "et_2l", "et_ce", "et_2l_ce"]:
        log_path = os.path.join(BASE, network, "logs", f"{network}_xasinj_gw231109.err")
        if not os.path.exists(log_path):
            print(f"[{network}] log not found: {log_path}")
            continue

        ifo_snrs = parse_snrs(log_path)
        if not ifo_snrs:
            print(f"[{network}] no SNR entries found")
            continue

        print(f"\n=== {network} ===")
        for ifo, snr in ifo_snrs:
            print(f"  {ifo}: optimal SNR = {snr:.2f}")

        network_snr = np.sqrt(sum(snr**2 for _, snr in ifo_snrs))
        print(f"  Network SNR = {network_snr:.2f}")

if __name__ == "__main__":
    main()