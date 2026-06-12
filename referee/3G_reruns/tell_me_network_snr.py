import os
import numpy as np

def main():
    for network in ["et", "et_2l", "et_ce", "et_2l_ce"]:
        
        outdir = f"./{network}/outdir/"
        npz_files = [f for f in os.listdir(outdir) if (f.endswith(".npz") and "posterior" in f)]
        npz_file = npz_files[0]
        
        print(f"Processing {network} with file {npz_file}")
        
        data = np.load(os.path.join(outdir, npz_file))
        keys = list(data.keys())
        print(f"keys found: {keys}")
        
        snr_keys = [key for key in keys if "snr" in key]
        print(f"SNR keys found: {snr_keys}")

if __name__ == "__main__":
    main()