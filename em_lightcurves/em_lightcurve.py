import numpy as np
import bilby
from nmma.joint.conversion import BNSEjectaFitting, EOS2Parameters, MultimessengerConversionWithLambdas
import h5py
import lal
from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel
import os,glob, io
import scipy

'''
### LOAD HAUKE's EOS
print(f"Loading Hauke's EOS . . .")
hauke_eos_filename = "../figures/EOS_data/hauke_macroscopic.dat"
R_HAUKE, M_HAUKE, L_HAUKE, PC_HAUKE = np.loadtxt(hauke_eos_filename, unpack=True)
C_HAUKE = M_HAUKE/R_HAUKE
mtov = np.max(M_HAUKE)
print(f'MTOV: {mtov}')
print(f"Loading Hauke's EOS . . . DONE")
'''

### LOAD POSTERIOR DATASET

pneeded = ['mass_1_source',
           'mass_2_source',
           'chirp_mass_source',
           'luminosity_distance',
           'lambda_tilde',
           'lambda_1',
           'lambda_2',
           'theta_jn']

#posfile = '/work/wouters/GW231109/prod_BW_XP_s005_l5000_default/outdir/final_result/GW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5'
resfile = "eossamplingGW231109_data0_1383609314-056813_analysis_H1L1_result.hdf5"
#print(f"Loading posterior from {posfile} . . .")

#alpha = np.random.uniform(alpha_min, alpha_max)
#ratio_zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)
alpha = 0. #np.random.uniform(alpha_min, alpha_max)
ratio_zeta = 0.3 #10 ** np.random.uniform(log10zeta_min, log10zeta_max)

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

with h5py.File(resfile, 'r') as f:

    pos = f["posterior"]
    print(pos.keys())
    #r = pos['radius_1']


    m1= pos['mass_1_source']
    m2 = pos['mass_2_source']
    eosp = pos['EOS']
    #print(np.array(eosid))
    lambda1 = []
    lambda2 =[]
    lambda_tilde = []
    KNthetapos = []
    dL = np.quantile(pos['luminosity_distance'], 0.5)
    log10_mej_dyn_pos = []
    log10_mej_wind_pos = []

    for jj in range(0,len(m1)):

        EOSID = pos["EOS"][jj].astype(int) + 1
        eosfile = 'MRL_sorted/{}.dat'.format(EOSID)
        rad, mass, lam = np.loadtxt(eosfile, usecols = [0,1,2]).T
        mtov = max(mass)
        f_m_lam = scipy.interpolate.interp1d(mass, lam)
        lam1 = f_m_lam(m1[jj])
        lambda1.append(lam1)
        lam2 = f_m_lam(m2[jj])
        lambda2.append(lam2)
        lamT = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lam1, lam2, m1[jj], m2[jj])
        #lambda_tilde.append(lamT)

        log_lambda_1 = np.log(lam1)
        log_lambda_2 = np.log(lam2)

        compactness_1 = (
        0.371 - 0.0391 * log_lambda_1 + 0.001056 * log_lambda_1 * log_lambda_1
        )
        compactness_2 = (
        0.371 - 0.0391 * log_lambda_2 + 0.001056 * log_lambda_2 * log_lambda_2
        )

        radius_1 = (
            m1[jj] / compactness_1 * lal.MRSUN_SI / 1e3
            )
        radius_2 = (
            m2[jj] / compactness_2 * lal.MRSUN_SI / 1e3
            )

        R_16= (
            pos['chirp_mass_source'][jj]
                * np.power(lamT / 0.0042, 1.0 / 6.0)
                * lal.MRSUN_SI
                / 1e3
            )


        #theta_jn = converted_parameters["theta_jn"]
        KNtheta = (
            180 / np.pi * np.minimum(pos['theta_jn'][jj], np.pi - pos['theta_jn'][jj])
            )
        KNthetapos.append(KNtheta)
        inclination_EM = (
                KNtheta * np.pi / 180.0
            )

        total_mass = m1[jj] + m2[jj]
        mass_ratio = m2[jj] / m1[jj]


        mdyn_fit = dynamic_mass_fitting_KrFo(m1[jj], m2[jj], compactness_1, compactness_2)

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

        #print(log10_mej_dyn, log10_mej_wind)

        log10_mej_dyn_pos.append(log10_mej_dyn)
        log10_mej_wind_pos.append(log10_mej_wind)


pos_mej_dyn = np.array(log10_mej_dyn_pos)
pos_mej_wind = np.array(log10_mej_wind_pos)

med_dyn = np.quantile(pos_mej_dyn, 0.5)
med_wind = np.quantile(pos_mej_wind, 0.5)

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

parameters = {"log10_mej_dyn": med_dyn, #log10_mej_dyn, 
              "log10_mej_wind": med_wind, #log10_mej_wind, 
              "KNphi": 30.0, # FIXME: need to change this
              "KNtheta": np.quantile(KNthetapos,0.5),
              "luminosity_distance": dL, #np.quantile(pos['luminosity_distance'], 0.5),
              }

print(f"Going to try and generate LC with the following parameters")
for key, value in parameters.items():
    print(f"    {key}: {value}")
lbol, mag = light_curve_model.generate_lightcurve(sample_times, parameters)

#print("lc")



import matplotlib.pyplot as plt


fig = plt.figure(figsize=(16, 18))

ncols = 1
nrows = int(np.ceil(len(filts) / ncols))
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

for ii, filt in enumerate(filts):
    loc_x, loc_y = np.divmod(ii, nrows)
    loc_x, loc_y = int(loc_x), int(loc_y)
    ax = fig.add_subplot(gs[loc_y, loc_x])

    #plt.plot(training['t'], training[filt], "k--", label="grid")
    plt.scatter(sample_times, mag[filt], "b-", label="interpolated")

    ax.set_xlim([0, 14])
    ax.set_ylim([-12, -18])
    ax.set_ylabel(filt, fontsize=30, rotation=0, labelpad=14)

    #if ii == 0:
    #    ax.legend(fontsize=16)

    if ii == len(filts) - 1:
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_yticks([-18, -16, -14, -12])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.grid(which="both", alpha=0.5)

fig.text(0.45, 0.05, "Time [days]", fontsize=30)

plt.savefig('lightcurve_scatter_test.png')
