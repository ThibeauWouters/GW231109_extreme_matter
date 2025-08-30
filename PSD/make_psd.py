"""
Using PyCBC
"""

import numpy as np
import matplotlib.pyplot as plt
import pycbc

from pycbc import frame, psd
from lalframe.utils import frtools

trigger_time = 1383609314.05

framefile_h1 = '/work/wouters/GW231109/H-H1_GWOSC_O4a_4KHZ_R1-1383608320-4096.gwf'
framefile_l1 = '/work/wouters/GW231109/L-L1_GWOSC_O4a_4KHZ_R1-1383608320-4096.gwf'

data_h1 = frame.read_frame(framefile_h1, 'H1:GWOSC-4KHZ_R1_STRAIN')
data_l1 = frame.read_frame(framefile_l1, 'L1:GWOSC-4KHZ_R1_STRAIN')

# Check if identical?
print(data_h1 == data_l1)

psd_h1=data_h1.trim_zeros().psd(8)
psd_l1=data_l1.trim_zeros().psd(8)

plt.loglog(psd_h1.sample_frequencies, psd_h1**0.5, alpha = 0.5, label='H1-PSD')
plt.loglog(psd_l1.sample_frequencies, psd_l1**0.5, alpha = 0.5, label='L1-PSD')
plt.xlim(20, 2000)
plt.ylim(1e-24,1e-21)
plt.savefig('psd.png', bbox_inches='tight', dpi=300)
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD [strain/$\sqrt{Hz}$]')
plt.legend()
plt.close()

# Save files
f_samples = psd_h1.sample_frequencies.data

data_psd_h1 = np.array([psd_h1.sample_frequencies.data[(f_samples<2048) & (f_samples>10)], psd_h1.data[(f_samples<2048) & (f_samples>10)]]).T
data_psd_l1 = np.array([psd_l1.sample_frequencies.data[(f_samples<2048) & (f_samples>10)], psd_l1.data[(f_samples<2048) & (f_samples>10)]]).T

# Save the PSD to the text file:
np.savetxt('H1_psd.txt', data_psd_h1)
np.savetxt('L1_psd.txt', data_psd_l1)