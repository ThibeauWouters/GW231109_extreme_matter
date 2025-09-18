"""
The posterior samples from GW170817 PE inference only have redshifted chirp mass, mass ratio, and tidal deformabilities.
This is a small script to convert masses to also store mass_1_source and mass_2_source there.
"""

import numpy as np
from bilby.gw.conversion import luminosity_distance_to_redshift
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde

filename = "./PE_posterior_samples_GW170817.npz"
data = np.load(filename)
data = dict(data)

chirp_mass, mass_ratio = data['chirp_mass'], data['mass_ratio']
luminosity_distance = data['luminosity_distance']

z = luminosity_distance_to_redshift(luminosity_distance)
chirp_mass_source = chirp_mass / (1 + z)
mass_1_source, mass_2_source = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_source, mass_ratio)

# Store all old data and the new masses there
data['chirp_mass_source'] = chirp_mass_source
data['mass_1_source'] = mass_1_source
data['mass_2_source'] = mass_2_source

# Also convert component lambdas to lambda_tilde
lambda_1 = data['lambda_1']
lambda_2 = data['lambda_2']

lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1_source, mass_2_source)
delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1_source, mass_2_source)
data['lambda_tilde'] = lambda_tilde
data['delta_lambda_tilde'] = delta_lambda_tilde

np.savez(filename, **data)
print(f"Updated {filename} with new parameters.")