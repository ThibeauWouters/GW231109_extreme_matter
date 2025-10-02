import arviz
import numpy as np

# Get the true EOS

hauke_filename = "../figures/EOS_data/hauke_macroscopic.dat"
r, m, l, _ = np.loadtxt(hauke_filename, unpack=True)
# print("r")
# print(r)

# print("m")
# print(m)

r14_true = np.interp(1.4, m, r)
print(f"R14 true: {r14_true:.2f} km")
print(f"=====================================================================\n\n\n")


filenames_dict = {"ET": "../jester/outdir_GW231109_ET_AS/eos_samples.npz",
                  "ET-CE": "../jester/outdir_GW231109_ET_CE/eos_samples.npz",
                  }
for ifo_network, filename in filenames_dict.items():
    data = np.load(filename)
    masses, radii = data["masses_EOS"], data["radii_EOS"]

    r14_list = np.array([np.interp(1.4, m, r) for m, r in zip(masses, radii)])
    median = np.median(r14_list)
    low, high = arviz.hdi(r14_list, hdi_prob=0.9)
    low = median - low
    high = high - median

    print(f"Filename: {filename}")
    print(f"R14 CI: {median:.2f}-{low:.2f}+{high:.2f}")

    print(f"=====================================================================")