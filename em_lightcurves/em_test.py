import numpy as np
import bilby
from nmma.joint.conversion import BNSEjectaFitting, EOS2Parameters, MultimessengerConversionWithLambdas
import h5py
import lal
from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel
import os,glob


#general modules
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os, sys, time, glob
from astropy.time import Time
import copy
import scipy
import nmma.em.io as io
#from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel
from nmma.em import training, utils, model_parameters
from sncosmo.bandpasses import _BANDPASSES



posfile = 'GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5'

conversion = MultimessengerConversionWithLambdas(binary_type='BNS', with_ejecta=True) #(eos_data_path ='/home/kingu/apuecher/S23109ci/NMMA/EOS/chiralEFT_MTOV', Neos=5000, binary_type='BNS')

pneeded = ['mass_1_source', 'mass_2_source', 'chirp_mass_source', 'lambda_tilde', 'lambda_1', 'lambda_2', 'theta_jn', 'luminosity_distance', 'redshift']

### MTOV hard coded from Hauke's maxl eos because I have no clue how to find it otherwise
### maybe to be consistent we should alos generate lambdas with this eos but who knows

mtov = 2.427536751340168
alpha_min, alpha_max = 1e-2, 2e-2
log10zeta_min, log10zeta_max = -3, 0
alpha = 0. #np.random.uniform(alpha_min, alpha_max)
ratio_zeta = 0.3 #10 ** np.random.uniform(log10zeta_min, log10zeta_max)

print('alpha',alpha, 'ratio zeta', ratio_zeta)

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


pmed = {}

with h5py.File(posfile, 'r') as f:

    pos = f["posterior"]
    #print(pos.keys())
    for p in pneeded:

        pmed[p] = np.quantile(pos[p], 0.5)

log_lambda_1 = np.log(pmed['lambda_1'])
log_lambda_2 = np.log(pmed['lambda_2'])

compactness_1 = (
    0.371 - 0.0391 * log_lambda_1 + 0.001056 * log_lambda_1 * log_lambda_1
)
compactness_2 = (
    0.371 - 0.0391 * log_lambda_2 + 0.001056 * log_lambda_2 * log_lambda_2
)

radius_1 = (
    pmed['mass_1_source'] / compactness_1 * lal.MRSUN_SI / 1e3
        )
radius_2 = (
    pmed['mass_2_source'] / compactness_2 * lal.MRSUN_SI / 1e3
        )

#chirp_mass_source = component_masses_to_chirp_mass(pmed['mass_1_source'], pmed['mass_2_source'])
#lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
#            lambda_1, lambda_2, mass_1_source, mass_2_source
#        )

R_16= (
    pmed['chirp_mass_source']
            * np.power(pmed['lambda_tilde'] / 0.0042, 1.0 / 6.0)
            * lal.MRSUN_SI
            / 1e3
        )


#theta_jn = converted_parameters["theta_jn"]
KNtheta = (
    180 / np.pi * np.minimum(pmed['theta_jn'], np.pi - pmed['theta_jn'])
            )
inclination_EM = (
                KNtheta * np.pi / 180.0
            )

total_mass = pmed['mass_1_source'] + pmed['mass_2_source']
mass_ratio = pmed['mass_2_source'] / pmed['mass_1_source']


mdyn_fit = dynamic_mass_fitting_KrFo(pmed['mass_1_source'], pmed['mass_2_source'], compactness_1, compactness_2)

log10_mdisk_fit = log10_disk_mass_fitting(
            total_mass,
            mass_ratio,
            mtov,
            R_16 * 1e3 / lal.MRSUN_SI,
        )


mej_dyn = mdyn_fit + alpha
log10_mej_dyn = np.log10(mej_dyn)

log10_mej_dyn = log10_mej_dyn
log10_mej_wind = np.log10(ratio_zeta) + log10_mdisk_fit
log10_mej_wind = log10_mej_wind
# total eject mass
total_ejeta_mass = 10**log10_mej_dyn + 10**log10_mej_wind
log10_mej = np.log10(total_ejeta_mass)
log10_mej = log10_mej

print(log10_mej_dyn, log10_mej_wind)
print(lala)

model_name = "Bu2019lm"
n_coeff = 3
# The array of times we'll use to examine each lightcurve
tini, tmax, dt = 0.1, 5.0, 0.2
tt = np.arange(tini, tmax + dt, dt)

# The filters we'll be focusing on
filts = ["sdssu","ztfg","ztfr","ztfi","ps1__z","ps1__y","2massj","2massh","2massks"] # We will focus on these two bands; all available: ["u","g","r","i","z","y","J","H","K"]
#filts = ["u","g","r","i","z","y","J","H","K"]

print(os.getcwd())
dataDir = "../nmma/tests/data/bulla" ## Example absolute path: "/Users/fabioragosta/nmma/nmma/tests/data/bulla"
ModelPath = "/home/kingu/apuecher/S23109ci/svmodels_downloads/nmma-models/models/" #"../svdmodels" ## Example absolute path: "/Users/fabioragosta/nmma/svdmodels"
filenames = glob.glob("%s/*.dat" % dataDir)

#data = io.read_photometry_files(filenames, filters=filts)
# Loads the model data
#training_data, parameters = model_parameters.Bu2019lm_sparse(data)

#two differen interpolation types are possible "sklearn_gp" or "tensorflow"
interpolation_type = "sklearn_gp"

tmin=0.1
tmax=20.0
deltat=0.1
sample_times = np.arange(tmin, tmax, deltat)

'''
training_model=training.SVDTrainingModel(
    model_name,
    copy.deepcopy(training_data),
    parameters,
    tt,
    filts,
    svd_path=ModelPath,
    n_coeff=n_coeff,
    interpolation_type=interpolation_type,
    n_epochs=100
)
'''
'''
light_curve_model = SVDLightCurveModel(
        model_name,
        sample_times,
        svd_path=ModelPath,
        interpolation_type=interpolation_type,
        #model_parameters=training_model.model_parameters,
        #filters=filts,
    )
'''

model_name = "Bu2019lm"
n_coeff = 3
# The array of times we'll use to examine each lightcurve
tini, tmax, dt = 0.1, 5.0, 0.2
tt = np.arange(tini, tmax + dt, dt)  

# The filters we'll be focusing on
filts = ["sdssu","ztfg","ztfr","ztfi","ps1__z","ps1__y","2massj","2massh","2massks"] # We will focus on these two bands; all available: ["u","g","r","i","z","y","J","H","K"]
#filts = ["u","g","r","i","z","y","J","H","K"]

print(os.getcwd())
dataDir = "../nmma/tests/data/bulla" ## Example absolute path: "/Users/fabioragosta/nmma/nmma/tests/data/bulla"
#ModelPath = "../svdmodels" ## Example absolute path: "/Users/fabioragosta/nmma/svdmodels"
filenames = glob.glob("%s/*.dat" % dataDir)

data = io.read_photometry_files(filenames, filters=filts)
# Loads the model data
training_data, parameters = model_parameters.Bu2019lm_sparse(data)

#two differen interpolation types are possible "sklearn_gp" or "tensorflow"
interpolation_type = "sklearn_gp"
training_model=training.SVDTrainingModel(
    model_name,
    copy.deepcopy(training_data),
    parameters,
    tt,
    filts,
    svd_path=ModelPath,
    n_coeff=n_coeff,
    interpolation_type=interpolation_type,
    n_epochs=100
)

light_curve_model = SVDLightCurveModel(
        model_name,
        sample_times,
        svd_path=ModelPath,
        interpolation_type=interpolation_type,
        model_parameters=training_model.model_parameters,
        filters=filts,
    )

#light_curve_model = SVDLightCurveModel(model='Bu2019lm', sample_times = sample_times, svd_path = ModelPath, parameter_conversion=None, mag_ncoeff=3,  lbol_ncoeff=3)
print(light_curve_model)
'''
data = {
  'log10_mej_dyn': log10_mej_dyn,
  'log10_mej_wind': log10_mej_wind,
  #'KNphi': KNtheta,
  'inclination_EM': inclination_EM,
  'redshift': pmed['redshift']}
'''

print('modelkeys', training_data.keys())
modelkeys = list(training_data.keys())
for mk in modelkeys:
    training_ = training_data[mk]
    parameters = training_model.model_parameters
    data = {param: training_[param] for param in parameters}
    print(data)

modelkeys = list(training_data.keys())
training = training_data[modelkeys[0]]
parameters = training_model.model_parameters
data = {param: training[param] for param in parameters}
data["redshift"] = 0
data['inclination_EM']=0
print('data',data)
lbol, mag = light_curve_model.generate_lightcurve(sample_times, data)
#lbol, mag= light_curve_model.generate_lightcurve(sample_times, data)

#n_coeff = 3

print('filters', mag.keys())
filts = mag.keys()
rows = int(np.ceil(len(filts) / ncols))
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

for ii, filt in enumerate(filts):
    loc_x, loc_y = np.divmod(ii, nrows)
    loc_x, loc_y = int(loc_x), int(loc_y)
    ax = fig.add_subplot(gs[loc_y, loc_x])

    plt.plot(training['t'], training[filt], "k--", label="grid")
    plt.plot(sample_times, mag[filt], "b-", label="interpolated")

    ax.set_xlim([0, 14])
    ax.set_ylim([-12, -18])
    ax.set_ylabel(filt, fontsize=30, rotation=0, labelpad=14)

    if ii == 0:
        ax.legend(fontsize=16)

    if ii == len(filts) - 1:
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_yticks([-18, -16, -14, -12])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.grid(which="both", alpha=0.5)

fig.text(0.45, 0.05, "Time [days]", fontsize=30)
fig.text(
    0.01,
    0.5,
    "Absolute Magnitude",
    va="center",
    rotation="vertical",
    fontsize=30,
)

plt.tight_layout()
#plt.show()
plt.savefig('test_lightcurve.png')
#plt.plot(sample_times, mag[filt], "b-", label="interpolated")

'''
tmin=0.1
tmax=20.0
deltat=0.1
t = np.arange(tmin, tmax, deltat)
lc_model = SVDLightCurveModel(model='Bu2019lm', sample_times = t, svd_path = "nmma/svdmodels/", parameter_conversion=None, mag_ncoeff=None,  lbol_ncoeff=None)
'''

'''
# GRB afterglow energy
log10_E0_MSUN = (
            np.log10(converted_parameters["ratio_epsilon"])
            + np.log10(1.0 - converted_parameters["ratio_zeta"])
            + log10_mdisk_fit
)
log10_E0_erg = log10_E0_MSUN + np.log10(
    lal.MSUN_SI * scipy.constants.c * scipy.constants.c * 1e7
)
converted_parameters["log10_E0"] = log10_E0_erg
'''

'''
    params = {
            "luminosity_distance": pos['luminosity_distance'],
            "chirp_mass": pos['chirp_mass'],
            "ratio_epsilon": 1e-20,
            "theta_jn": pos['theta_jn'],
            "a_1": pos['a1'],
            "a_2": pos['a2'],
            "mass_1": pos['mass_1'],
            "mass_2": pos['mass_2'],
            "EOS": idx,
            "cos_tilt_1": pos['cos_tilt_1'],
            "cos_tilt_2": pos['cos_tilt_2'],
            "KNphi": 30,
        }

    alpha_min, alpha_max = 1e-2, 2e-2
    log10zeta_min, log10zeta_max = -3, 0
    alpha = np.random.uniform(alpha_min, alpha_max)
    zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)

    params = {
                **params,
                "alpha": alpha,
                "ratio_zeta": zeta,
            }

    complete_parameters, _ = parameter_conversion(pos)

    #tc_gps = time.Time(args.gps, format="gps")
    #trigger_time = tc_gps.mjd

    #complete_parameters["kilonova_trigger_time"] = trigger_time

    print(complete_parameters.keys())
    data = create_light_curve_data(
        complete_parameters, args, doAbsolute=args.absolute
    )

'''
