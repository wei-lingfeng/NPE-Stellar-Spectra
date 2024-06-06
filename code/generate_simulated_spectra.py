import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import smart
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

test = False
n_params = 5
assert n_params in [3, 5], 'n_params must be either 3(teff, rv, vsini) or 5(teff, rv, vsini, logg, metal)'

user_path = os.path.expanduser('~')
data_path = f'{user_path}/ML/Group7-Project/data'
save_path = f'/stow/weilingfeng/data/apogee/simulated_spectra_{n_params}_params.pkl'

with open(f'{data_path}/wavelength.npy', 'rb') as file:
    wave = np.load(file)

with open(f'{data_path}/lsf.npy', 'rb') as file:
    xlsf = np.load(file)
    lsf = np.load(file)

if not test:
    np.random.seed(0)

if test:
    size=1
else:
    size = 50000

teff    = np.random.uniform(low=2300., high=7000., size=size)
rv      = np.random.uniform(low=-200., high=200., size=size)
vsini   = np.random.uniform(low=0., high=100., size=size)
if n_params == 3:
    logg    = 4. * np.ones(size)
    metal   = 0. * np.ones(size)
elif n_params == 5:
    logg    = np.random.uniform(low=2.5, high=6., size=size)
    metal   = np.random.uniform(low=-2.12, high=0.5, size=size)

params = list(zip(teff, rv, vsini, logg, metal))

instrument = 'apogee'
order = 'all'
modelset = 'phoenix-aces-agss-cond-2011'

def simulate_spectra(params):
    teff, rv, vsini, logg, metal = params
    model = smart.makeModel(teff=teff, rv=rv, vsini=vsini, logg=logg, metal=metal, instrument=instrument, order=order, modelset=modelset, lsf=lsf, xlsf=xlsf)
    model.flux = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=wave))
    return model.flux

def noise(x):
    rand = np.random.normal(0, 1, size=x.shape[0])

    # let's say x_err is drawn from [0, 0.02]
    x_err = np.random.uniform() * 0.02
    x_noise = x + rand * x_err * x
    return x_noise

with Pool(32) as pool:
    # need to keep list to print progress bar
    fluxes = np.array(list(tqdm(pool.imap(simulate_spectra, params), total=size)))

# Add noise
fluxes_with_noise = np.empty_like(fluxes)
for i in range(size):
    fluxes_with_noise[i] = noise(fluxes[i])

if test:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wave, fluxes_with_noise[0], lw=0.7)
    ax.plot(wave, fluxes[0], lw=0.7)
    ax.set_xlabel('Wavelength ($\AA$)')
    ax.set_ylabel('Normalized Flux')
    plt.show()
else:
    simulated_spectra = {
        'params': ['teff', 'vsini', 'rv'] if n_params==3 else ['teff', 'vsini', 'rv', 'logg', 'metal'],
        'n_params': n_params,
        'wave': wave,
        'flux': fluxes,
        'flux_with_noise': fluxes_with_noise,
        'teff': teff,
        'logg': logg[0] if n_params==3 else logg,
        'metal': metal[0] if n_params==3 else metal,
        'vsini': vsini,
        'rv': rv,
        'order': order,
        'modelset': modelset,
        'lsf': lsf,
        'xlsf': xlsf,
    }

    with open(save_path, 'wb') as file:
        pickle.dump(simulated_spectra, file)

    if not os.path.exists(f"{data_path}/{save_path.split('/')[-1]}"):
        os.symlink(save_path, f"{data_path}/{save_path.split('/')[-1]}")