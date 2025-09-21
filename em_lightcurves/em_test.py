import numpy as np
import bilby
from nmma.joint.conversion import BNSEjectaFitting, EOS2Parameters, MultimessengerConversionWithLambdas
import h5py
import lal
from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel
import os,glob, io

### LOAD HAUKE's EOS
print(f"Loading Hauke's EOS . . .")
hauke_eos_filename = "../figures/EOS_data/hauke_macroscopic.dat"
R_HAUKE, M_HAUKE, L_HAUKE, PC_HAUKE = np.loadtxt(hauke_eos_filename, unpack=True)
C_HAUKE = M_HAUKE/R_HAUKE
mtov = np.max(M_HAUKE)
print(f'MTOV: {mtov}')
print(f"Loading Hauke's EOS . . . DONE")

### LOAD POSTERIOR DATASET

pneeded = ['mass_1_source',
           'mass_2_source',
           'chirp_mass_source',
           'luminosity_distance',
           'lambda_tilde',
           'lambda_1',
           'lambda_2',
           'theta_jn']

posfile = '/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/outdir/final_result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5'
print(f"Loading posterior from {posfile} . . .")

# Get median values of some posterior parameters
pmed = {}
with h5py.File(posfile, 'r') as f:
    pos = f["posterior"]
    for p in pneeded:
        pmed[p] = np.quantile(pos[p], 0.5)

log_lambda_1 = np.log([np.interp(_m, M_HAUKE, L_HAUKE) for _m in [pmed['mass_1_source']]])[0]
log_lambda_2 = np.log([np.interp(_m, M_HAUKE, L_HAUKE) for _m in [pmed['mass_2_source']]])[0]

print("log_lambda_1")
print(log_lambda_1)

print("log_lambda_2")
print(log_lambda_2)

print(f"Loading posterior from {posfile} . . . DONE")

conversion = MultimessengerConversionWithLambdas(binary_type='BNS', with_ejecta=True) # FIXME: needed or copied over here?

alpha_min, alpha_max = 1e-2, 2e-2
log10zeta_min, log10zeta_max = -3, 0
alpha = np.random.uniform(alpha_min, alpha_max)
ratio_zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)

print('alpha', alpha, 'ratio zeta', ratio_zeta)

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

print(f"Computing aux data for fits . . .")

compactness_1 = np.interp(pmed['mass_1_source'], M_HAUKE, C_HAUKE)
compactness_2 = np.interp(pmed['mass_2_source'], M_HAUKE, C_HAUKE)

radius_1 = np.interp(pmed['mass_1_source'], M_HAUKE, R_HAUKE)
radius_2 = np.interp(pmed['mass_2_source'], M_HAUKE, R_HAUKE)

R_16 = np.interp(1.6, M_HAUKE, R_HAUKE) 

KNtheta = (
    180 / np.pi * np.minimum(pmed['theta_jn'], np.pi - pmed['theta_jn'])
            )
inclination_EM = (
                KNtheta * np.pi / 180.0
            )

total_mass = pmed['mass_1_source'] + pmed['mass_2_source']
mass_ratio = pmed['mass_2_source'] / pmed['mass_1_source']
dL = pmed['luminosity_distance']

mdyn_fit = dynamic_mass_fitting_KrFo(pmed['mass_1_source'], pmed['mass_2_source'], compactness_1, compactness_2)

log10_mdisk_fit = log10_disk_mass_fitting(
            total_mass,
            mass_ratio,
            mtov,
            R_16 * 1e3 / lal.MRSUN_SI,
        )
print(f"Computing aux data for fits . . . DONE")

print(f"Computing ejecta masses . . .")
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
print(f"Computing ejecta masses . . . DONE")

print(f"Loading LC model . . .")
model_name = "Bu2019lm"
n_coeff = 10
tini, tmax, dt = 0.1, 5.0, 0.2
tt = np.arange(tini, tmax + dt, dt)
filts = ["sdssu","ztfg","ztfr","ztfi","ps1__z","ps1__y","2massj","2massh","2massks"]
interpolation_type = "tensorflow"
sample_times = np.arange(tini, tmax, dt)

light_curve_model = SVDLightCurveModel(
        model_name,
        sample_times,
        svd_path=None,
        interpolation_type=interpolation_type,
        filters=filts
    )
print(f"Loading LC model . . . DONE")

parameters = {"log10_mej_dyn": log10_mej_dyn, 
              "log10_mej_wind": log10_mej_wind, 
              "KNphi": 30.0, # FIXME: need to change this
              "KNtheta": KNtheta,
              "luminosity_distance": dL,
              }

print(f"Going to try and generate LC with the following parameters")
for key, value in parameters.items():
    print(f"    {key}: {value}")
lc = light_curve_model.generate_lightcurve(sample_times, parameters)

print("lc")
print(lc)

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
