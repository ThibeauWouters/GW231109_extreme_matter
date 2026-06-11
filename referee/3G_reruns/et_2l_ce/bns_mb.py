import sys
sys.path.insert(0, '/work/wouters/projects/19_GW231109_referee/bilby')

import numpy as np
import bilby

from bilby.gw.conversion import luminosity_distance_to_redshift

# Set True to verify setup without running the sampler
DRY_RUN = False

outdir = "outdir"
label = "ET2LCE_gw231109_injection_alignedspin"

bilby.core.utils.random.seed(13599876)

minimum_frequency = 5.
reference_frequency = 5.
sampling_frequency = 4096

injection_parameters = {"mass_1": 1.5879187040159342, "mass_2": 1.4188967691574992, "geocent_time": 1383609314.0505133, "a_1": 0.0305182034770227, "a_2": 0.028570123024914268, "phi_12": 0.0, "phi_jl": 0.0, "psi": 1.5591681494817768, "theta_jn": 2.5287713998365304, "ra": 0.1778415509774547, "dec": -0.602827165369817, "tilt_1": 0.0, "tilt_2": 0.0, "phase": 3.139490467696903, "luminosity_distance": 168.3222418883087, "lambda_1": 226.15368890383624, "lambda_2": 470.4950264665735}

###############
### THIBEAU ###
###############

# Convert to source frame masses
z = luminosity_distance_to_redshift(injection_parameters["luminosity_distance"])
mass_1_source = injection_parameters["mass_1"] / (1 + z)
mass_2_source = injection_parameters["mass_2"] / (1 + z)

print(f"Original Lambdas:")
print(f"Lambda 1: {injection_parameters['lambda_1']}")
print(f"Lambda 2: {injection_parameters['lambda_2']}")

print(f"mass_1_source: {mass_1_source}")
print(f"mass_2_source: {mass_2_source}")

print(f"Overwriting with the jester EOS:")
jester_eos_filename = '/work/wouters/projects/19_GW231109_referee/GW231109_extreme_matter/referee/jester_reruns/3G/jester_GW170817_maxL_EOS.npz'
jester_data = np.load(jester_eos_filename)
jester_masses = jester_data['masses_EOS']  # in solar masses
jester_lambdas = jester_data['Lambdas_EOS']

# Interpolate to find the Lambdas for the source frame masses
lambda_1 = np.interp(mass_1_source, jester_masses, jester_lambdas)
lambda_2 = np.interp(mass_2_source, jester_masses, jester_lambdas)

injection_parameters['lambda_1'] = lambda_1
injection_parameters['lambda_2'] = lambda_2

print(f"NEW Lambdas:")
print(f"Lambda 1: {injection_parameters['lambda_1']}")
print(f"Lambda 2: {injection_parameters['lambda_2']}")

####################
### THIBEAU DONE ###
####################

CHIEFF = (injection_parameters['a_1'] * injection_parameters['mass_1'] + injection_parameters['a_2'] * injection_parameters['mass_2']) / (injection_parameters['mass_1'] + injection_parameters['mass_2'])
duration = bilby.gw.utils.calculate_time_to_merger(frequency=minimum_frequency, mass_1=injection_parameters['mass_1'], mass_2=injection_parameters['mass_2'], chi=CHIEFF, safety=1.1)
duration = int(duration + 1.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomXAS_NRTidalv3", reference_frequency=reference_frequency, minimum_frequency=minimum_frequency,
    ),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
)

# ET 2L: ET_EMR (Meuse-Rhine) + ET_Sar (Sardinia), both 15 km + CE at LIGO-H site (CE_psd.txt)
ifos = bilby.gw.detector.InterferometerList(["ET_EMR", "ET_Sar", "CE"])

for ifo in ifos:
    if ifo.name == "CE":
        ifo.minimum_frequency = 10.
    else:
        ifo.minimum_frequency = minimum_frequency

ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                   duration=duration,
                                                   start_time=injection_parameters["geocent_time"] - duration + 2)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomXP_NRTidalv3", reference_frequency=reference_frequency, minimum_frequency=minimum_frequency,
    ),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
)

priors = bilby.core.prior.PriorDict(filename='bns.prior')

likelihood = bilby.gw.likelihood.MBGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=search_waveform_generator,
    priors=priors,
    reference_chirp_mass=priors["chirp_mass"].minimum,
    distance_marginalization=True,
    phase_marginalization=True,
    time_reference="H1",
    reference_frame="H1L1",
    with_eos=False,
    Neos=4145,
    eos_path='/work/puecher/S231109/eos_sampling/MRL_sorted',
    eos_weight_path='/work/puecher/S231109/eos_sampling/eos_pos_setA_sorted.txt'
)

if DRY_RUN:
    print("DRY RUN complete — setup finished without errors. Exiting before sampling.")
    sys.exit(0)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=2000,
    naccept=60,
    npool=192,
    check_point_plot=True,
    check_point_delta_t=1800,
    print_method='interval-60',
    sample='acceptance-walk',
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label
)

result.plot_corner()
