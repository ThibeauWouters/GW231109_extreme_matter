import arviz
import numpy as np

filename = "../jester/outdir_GW231109_ET_AS/eos_samples.npz"
data = np.load(filename)
masses, radii = data["masses_EOS"], data["radii_EOS"]

r14_list = np.array([np.interp(1.4, m, r) for m, r in zip(masses, radii)])
median = np.median(r14_list)
low, high = arviz.hdi(r14_list, hdi_prob=0.9)
low = median - low
high = high - median

print(f"Filename: {filename}")
print(f"R14 CI: {median:.2f}-{low:.2f}+{high:.2f}")