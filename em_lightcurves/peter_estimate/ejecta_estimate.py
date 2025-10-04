import numpy as np
import pandas as pd
import h5py

def dynamic_mass_fitting_KrFo(
    mass_1,
    mass_2,
    compactness_1,
    compactness_2,
    a=-9.3335,
    b=114.17,
    c=-337.56,
    n=1.5465,
):
    """
    See https://arxiv.org/pdf/2002.07728.pdf
    """
    
    mdyn = mass_1 * (
        a / compactness_1 + b * np.power(mass_2 / mass_1, n) + c * compactness_1
    )
    mdyn += mass_2 * (
        a / compactness_2 + b * np.power(mass_1 / mass_2, n) + c * compactness_2
    )
    mdyn *= 1e-3
    
    mdyn = np.maximum(0.0, mdyn)
    
    return mdyn

def log10_disk_mass_fitting(
    total_mass,
    mass_ratio,
    MTOV,
    R16,
    a0=-1.725,
    delta_a=-2.337,
    b0=-0.564,
    delta_b=-0.437,
    c=0.958,
    d=0.057,
    beta=5.879,
    q_trans=0.886,
):

    k = -3.606 * MTOV / R16 + 2.38
    threshold_mass = k * MTOV

    xi = 0.5 * np.tanh(beta * (mass_ratio - q_trans))

    a = a0 + delta_a * xi
    b = b0 + delta_b * xi

    log10_mdisk = a * (1 + b * np.tanh((c - total_mass / threshold_mass) / d))
    log10_mdisk = np.maximum(-3.0, log10_mdisk)

    return log10_mdisk

# EOS directory
EOS_path = '../anna_posterior_eos/MRL_sorted/'
# load the posterior samples
data = h5py.File(
    '../anna_posterior_eos/eossamplingGW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5'
)
posterior = data['posterior']

# now start to prepare the needed posterior samples into a pandas dataframe
key_needed = [
    'mass_1_source', 'mass_2_source', 'EOS', 'luminosity_distance', 'theta_jn'
]
posterior = pd.DataFrame.from_dict(
    {key: np.array(posterior[key]) for key in key_needed}
)
# adjust the EOS a bit
posterior['EOS'] = posterior['EOS'].to_numpy().astype(int) + 1
# also calculate additional parameters for the universal relation fittings
R16 = []
MTOV = []
C1 = []
C2 = []
for m1src, m2src, eos in zip(
    posterior['mass_1_source'], posterior['mass_2_source'], posterior['EOS']
):
    m_eos, r_eos = np.loadtxt(
        f'../anna_posterior_eos/MRL_sorted/{eos}.dat',
        usecols=[1, 0],
        unpack=True
    ) 
    R16.append(np.interp(1.6, m_eos, r_eos))
    MTOV.append(m_eos[-1])
    C1.append(m1src * 1.477/ np.interp(m1src, m_eos, r_eos))
    C2.append(m2src * 1.477/ np.interp(m2src, m_eos, r_eos))

posterior['compactness_1'] = np.array(C1)
posterior['compactness_2'] = np.array(C2)
posterior['MTOV'] = np.array(MTOV)
posterior['R16'] = np.array(R16)

posterior['log10_mdyn'] = np.log10(
    dynamic_mass_fitting_KrFo(
        posterior['mass_1_source'],
        posterior['mass_2_source'],
        posterior['compactness_1'],
        posterior['compactness_2']
    )
)
posterior['log10_mdisk'] = log10_disk_mass_fitting(
    posterior['mass_1_source'] + posterior['mass_2_source'],
    posterior['mass_2_source'] / posterior['mass_1_source'],
    posterior['MTOV'],
    posterior['R16'] * 1.477 # convert this into solar mass
)
# assuming 30% of the disk mass are ejected as wind
posterior['log10_mwind'] = np.log10(0.3) + posterior['log10_mdisk']

posterior.to_csv('posterior_samples_with_ejecta_mass.dat', sep=' ', index=False)
