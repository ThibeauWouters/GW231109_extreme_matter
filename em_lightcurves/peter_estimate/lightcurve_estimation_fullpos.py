import numpy as np
import pandas as pd
from nmma.em.model import SVDLightCurveModel

# load the posterior
posterior = pd.read_csv(
    './posterior_samples_with_ejecta_mass.dat', delimiter=' ', header=0
)
# load the kilonova model
model_name = 'Bu2019lm'
model_path = '/home/kingu/apuecher/S23109ci/svmodels_downloads/nmma-models/models/'
n_coeff = 10
tini, tmax, dt = 0.1, 5.0, 0.05
sample_times = np.arange(tini, tmax, dt)
interpolation_type = "tensorflow"
filts = [
    "sdssu",
    "ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y",
    "2massj", "2massh", "2massks"
]

light_curve_model = SVDLightCurveModel(
        model_name,
        sample_times,
        svd_path='/home/kingu/apuecher/S23109ci/svmodels_downloads/nmma-models/models/',
        interpolation_type=interpolation_type,
        filters=filts
)
# viewing angle adjustment
em_view_angle = np.minimum(posterior['theta_jn'], np.pi - posterior['theta_jn']) * 180 / np.pi 
# now loop over the samples and generate all lc
np.random.seed(42)
import tqdm
mag_full = {}
for i in tqdm.tqdm(range(0, len(posterior['luminosity_distance']))):
    parameters = {
        "log10_mej_dyn": posterior['log10_mdyn'][i], #log10_mej_dyn, 
        "log10_mej_wind": posterior['log10_mwind'][i], #log10_mej_wind, 
        "KNphi": np.random.uniform(15., 75.),
        "KNtheta": em_view_angle[i],
        "luminosity_distance": posterior['luminosity_distance'][i]
    }
    _, mag = light_curve_model.generate_lightcurve(sample_times, parameters)
    mag_full[i] = mag
# store the lc
mag_final = {}
mag_final['sample_times'] = sample_times
for key, inner_dict in mag_full.items():
    for letter, array in inner_dict.items():
        if letter not in mag_final:
            mag_final[letter] = []
        mag_final[letter].append(array)
# Convert lists to numpy arrays
for letter in mag_final:
    mag_final[letter] = np.array(mag_final[letter])

mag_df = pd.DataFrame()
mag_df['sample_times'] = mag_final['sample_times']
dist_reshape = np.tile(posterior['luminosity_distance'], (len(sample_times), 1)).T
dist_mod = 5 * np.log10(dist_reshape * 1e5)
dist_mod_170817 = 5 * np.log10(np.ones(dist_mod.shape) * 40 * 1e5)
for filt in filts:
    mag_per_filt = mag_final[filt]
    mag_per_filt_app = mag_per_filt + dist_mod
    mag_per_filt_170817 = mag_per_filt + dist_mod_170817
    filt = filt.replace('_',':')
    mag_df[f'{filt}_median'] = np.median(mag_per_filt_app, axis=0)
    mag_df[f'{filt}_2p5pt'] = np.quantile(mag_per_filt_app, axis=0, q=0.025) 
    mag_df[f'{filt}_97p5pt'] = np.quantile(mag_per_filt_app, axis=0, q=0.975) 
    mag_df[f'{filt}_16pt'] = np.quantile(mag_per_filt_app, axis=0, q=0.16) 
    mag_df[f'{filt}_84pt'] = np.quantile(mag_per_filt_app, axis=0, q=0.84) 
    mag_df[f'{filt}_170817_median'] = np.median(mag_per_filt_170817, axis=0)
    mag_df[f'{filt}_170817_2p5pt'] = np.quantile(mag_per_filt_170817, axis=0, q=0.025) 
    mag_df[f'{filt}_170817_97p5pt'] = np.quantile(mag_per_filt_170817, axis=0, q=0.975) 
    mag_df[f'{filt}_170817_16pt'] = np.quantile(mag_per_filt_170817, axis=0, q=0.16) 
    mag_df[f'{filt}_170817_84pt'] = np.quantile(mag_per_filt_170817, axis=0, q=0.84) 
mag_df.to_csv('lc_data_band.dat', sep=' ', index=False)
