"""
The original fileformat is not using the correct names for the parameters, so we need to convert them to the correct names.
"""

import numpy as np

# Names mapping
names_mapping = {"masses": "masses_EOS",
                 "radii": "radii_EOS",
                 "Lambdas": "Lambdas_EOS"
                 }

filename = "old_jester_GW170817_maxL_EOS.npz"
data = np.load(filename)

output_dict = {}
for key, value in data.items():
    if key in names_mapping:
        output_dict[names_mapping[key]] = value
    else:
        output_dict[key] = value
        
output_filename = "jester_GW170817_maxL_EOS.npz"
np.savez(output_filename, **output_dict)
print(f"Converted file saved as {output_filename}")