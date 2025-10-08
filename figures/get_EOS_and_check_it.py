import numpy as np
import matplotlib.pyplot as plt
import os

# Get the EOS:
filename = "/home/twouters2/projects/GW231109/GW231109_extreme_matter/jester/outdir_radio/eos_samples.npz"
data = np.load(filename)
keys = list(data.keys())

print("Keys in the data file:", keys)

masses_EOS, radii_EOS, Lambdas_EOS, log_prob = data['masses_EOS'], data['radii_EOS'], data['Lambdas_EOS'], data['log_prob']
all_n, all_p, all_e = data['n'], data['p'], data['e']

# Make a quick histogram of the log_prob to see its distribution
plt.hist(log_prob, bins=50, density=True, color='blue')
plt.xlabel("Log Probability")
plt.ylabel("Density")
plt.title("Histogram of Log Probability of EOS Samples")
plt.savefig("./figures/log_prob_histogram.png")
plt.close()

# Get the maximum likelihood EOS index:
my_index = np.argmax(log_prob)

# # Instead, get the median EOS index:
# median_log_prob = np.median(log_prob)
# my_index = np.argmin(np.abs(log_prob - median_log_prob)) + 5

# Extract the corresponding masses and Lambdas:
print("Index:", my_index)
masses = masses_EOS[my_index]
radii = radii_EOS[my_index]
Lambdas = Lambdas_EOS[my_index]
n = all_n[my_index]
p = all_p[my_index]
e = all_e[my_index]

save_dict = {
    "n": n,
    "p": p,
    "e": e,
    "masses": masses,
    "radii": radii,
    "Lambdas": Lambdas
}   

# Save to ./EOS_data
output_dir = "./EOS_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.savez(os.path.join(output_dir, "jester_GW170817_maxL_EOS.npz"), **save_dict)

# For the plots, truncate a bit
TRUNCATE_MASS = 0.9
mask = (masses > TRUNCATE_MASS)
masses, radii, Lambdas = masses[mask], radii[mask], Lambdas[mask]

# Also load Hauke's EOS for comparison
filename = "./EOS_data/hauke_macroscopic.dat"
hauke_r, hauke_m, hauke_Lambda, _ = np.loadtxt(filename, unpack=True)
mask_hauke = (hauke_m > TRUNCATE_MASS)
hauke_r, hauke_m, hauke_Lambda = hauke_r[mask_hauke], hauke_m[mask_hauke], hauke_Lambda[mask_hauke]

# Also plot it
plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
plt.subplot(1, 2, 1)
# M vs R
plt.plot(radii, masses, color='blue', label = 'Jester')
plt.plot(hauke_r, hauke_m, color='red', linestyle = "--", label = 'Hauke')
plt.xlabel("Radius (km)")
plt.ylabel("Mass (Msun)")

# M vs Lambda
plt.subplot(1, 2, 2)
plt.plot(masses, Lambdas, color='blue', label = 'Jester')
plt.plot(hauke_m, hauke_Lambda, color='red', linestyle = "--", label = 'Hauke')
plt.yscale('log')
plt.xlabel("Mass (Msun)")
plt.ylabel("Lambda")
plt.legend()

plt.savefig("./figures/jester_GW170817_maxL_EOS_MR_and_MLambda.pdf", bbox_inches='tight')
plt.close()