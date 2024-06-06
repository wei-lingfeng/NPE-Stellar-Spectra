import os
import smart
import pickle
import numpy as np
import apogee_tools as ap
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.table import Table
from itertools import compress

order = 'all'
modelset = 'phoenix-aces-agss-cond-2011'
instrument = 'apogee'
apogee_type = 'aspcap'

if apogee_type == 'apstar':
    prefix = 'apStar'
elif apogee_type == 'apvisit':
    prefix = 'apVisit'
elif apogee_type == 'aspcap':
    prefix = 'aspcapStar'

user_path = os.path.expanduser('~')
data_path = f'{user_path}/ML/Group7-Project/data'
spec_path = f'{data_path}/spec'
apogee_table = Table.read(f'{data_path}/apogee_table.csv')


# Get the LSF
if not os.path.exists(f'{data_path}/lsf.npy'):
    xlsf = np.linspace(-7.,7.,43)
    lsf  = ap.apogee_hack.spec.lsf.eval(xlsf)
    with open(f'{data_path}/lsf.npy', 'wb') as file:
        np.save(file, xlsf)
        np.save(file, lsf)
else:
    with open(f'{data_path}/lsf.npy', 'rb') as file:
        xlsf = np.load(file)
        lsf = np.load(file)


N_stars = len(apogee_table)

# crval1 = 4.179
# cdelt1 = 6e-6
# wave = np.power(10, crval1 + cdelt1 * np.arange(8575))
spec = smart.Spectrum(name=apogee_table['ID'][0], path=f"{spec_path}/{prefix}-dr17-{apogee_table['ID'][0]}.fits", instrument=instrument, apply_sigma_mask=True, datatype=apogee_type, applytell=True)
wave = spec.oriWave
# Save wavelength
with open(f'{data_path}/wavelength.npy', 'wb') as file:
    np.save(file, wave)

fluxes = []
for i in tqdm(range(len(apogee_table['ID']))):
    apogee_id = apogee_table['ID'][i]
    teff = apogee_table['Teff'][i]
    rv = apogee_table['HRV'][i]
    metal = apogee_table['[M/H]'][i]
    logg = apogee_table['logg'][i]
    vsini = apogee_table['Vsini'][i]
    spec = smart.Spectrum(name=apogee_id, path=f'{spec_path}/{prefix}-dr17-{apogee_id}.fits', instrument=instrument, apply_sigma_mask=True, datatype=apogee_type, applytell=True)
    
    spec.wave = np.ma.array(spec.oriWave, mask=spec.mask)
    spec.flux = np.ma.array(spec.oriFlux, mask=spec.mask)
    spec.noise = np.ma.array(spec.oriNoise, mask=spec.mask)
    # model = smart.makeModel(teff=teff, logg=logg, metal=metal, vsini=vsini, rv=rv, instrument=instrument, order=order, modelset=modelset, data=spec, lsf=lsf, xlsf=xlsf)
    fluxes.append(spec.flux)

same_length = np.array([len(_) for _ in fluxes])==7514
fluxes = np.shape(np.array(list(compress(fluxes, same_length))))
apogee_table = apogee_table[same_length]
apogee_table.write(f'{user_path}/ML/Group7-Project/data/apogee_table.csv', overwrite=True)

spectra = {}
for key in apogee_table.keys():
    spectra[key] = apogee_table[key].data

spectra['wave'] = wave
spectra['flux'] = fluxes
spectra['lsf']  = lsf
spectra['xlsf'] = np.linspace(-7.,7.,43)

with open(f'{data_path}/spectra.pkl', 'wb') as file:
    pickle.dump(spectra, file)