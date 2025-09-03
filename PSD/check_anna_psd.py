import numpy as np
import matplotlib.pyplot as plt

psd_H1 = "/work/wouters/GW231109/data/H1_psd.txt"
psd_L1 = "/work/wouters/GW231109/data/L1_psd.txt"

plt.figure()
for psd_file in [psd_H1, psd_L1]:
    freqs, psd = np.loadtxt(psd_file, unpack=True)
    
    plt.loglog(freqs, psd)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [1/Hz]")
    
plt.savefig("check_anna_psd.png")
plt.close()