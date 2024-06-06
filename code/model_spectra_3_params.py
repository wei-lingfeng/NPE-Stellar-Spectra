import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    # prevent numpy from multithreading
os.environ['MKL_NUM_THREADS'] = '1'         # prevent numpy from multithreading
import nbi
import torch
import smart
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import uniform
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

user_path = os.path.expanduser('~')
data_path = '/stow/weilingfeng/data/apogee'
save_path = f'{data_path}/nbi_3_params'

new_run = True

# Read simulated spectra
with open(f'{data_path}/simulated_spectra_3_params.pkl', 'rb') as file:
    simulated_spectra = pickle.load(file)

wave = simulated_spectra['wave']
flux = simulated_spectra['flux_with_noise']
params = np.array([simulated_spectra[_] for _ in ['teff', 'rv', 'vsini']]).T

# np.random.seed(0)

# simulator
logg = simulated_spectra['logg']
metal = simulated_spectra['metal']
instrument = 'apogee'
order = 'all'
modelset = 'phoenix-aces-agss-cond-2011'
lsf  = simulated_spectra['lsf']
xlsf = simulated_spectra['xlsf']

def simulate_spectra(params):
    teff, rv, vsini = params
    model = smart.makeModel(teff=teff, rv=rv, vsini=vsini, logg=logg, metal=metal, instrument=instrument, order=order, modelset=modelset, lsf=lsf, xlsf=xlsf)
    model.flux = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=wave))
    return model.flux

def noise(x, y):
    rand = np.random.normal(0, 1, size=x.shape[0])

    # let's say x_err is drawn from [0, 0.05]
    x_err = np.random.uniform() * 0.05
    x_noise = x + rand * x_err * x
    return x_noise, y


prior = {
    'teff':     uniform(loc=2300, scale=7000-2300),  # U(2300, 7000) K
    'rv':       uniform(loc=-200, scale=200-(-200)), # U(-200, 200) km/s
    'vsini':    uniform(loc=0, scale=100),           # U(0, 100) km/s
    # 'logg':     uniform(loc=2.5, scale=6-2.5),       # U(2.5, 6)
    # 'metal':    uniform(loc=-2.12, scale=0.5+2.12)   # U(-2.12, 0.5)
}

labels = list(prior.keys())
priors = [prior[k] for k in labels]

if new_run:
    # clear cuda cache
    torch.cuda.empty_cache()
    
    # the NBI package provides the "ResNet-GRU" network as the default
    # featurizer network for sequential data
    featurizer = {
        'type': 'resnet-gru',
        'norm': 'weight_norm',
        'dim_in': 1,            # Number of channels for input data
        'dim_out': 256,         # Output feature vector dimension
        'dim_conv_max': 512,    # Maximum hidden dimension for CNN
        'depth': 8              # Number of 1D ResNet layers
    }

    flow = {
        'n_dims': len(labels),  # dimension of parameter space
        'flow_hidden': 256,
        'num_cond_inputs': 256,
        'num_blocks': 15,
        'n_mog': 4              # Number of Mixture of Gaussian as base density
    }

    # initialize NBI engine
    engine = nbi.NBI(
        flow=flow,
        featurizer=featurizer,
        # simulator=simulate_spectra,
        priors=priors,
        labels=labels,
        device='cuda',
        path=save_path
    )

    engine.fit(
        x=flux,
        y=params,
        n_rounds=1,
        n_epochs=10,
        batch_size=256,
        lr=0.0001,
        early_stop_patience=20,
        noise=noise  # this can also be an array if fixed noise
    )
    
    best_model = engine.best_params
    with open(f'{save_path}/best_model_path.txt', 'w') as file:
        file.write(best_model)

else:
    with open(f'{save_path}/best_model_path.txt', 'r') as file:
        best_model = file.read()
    
    engine = nbi.NBI(
        state_dict=best_model,
        simulator=simulate_spectra,
        priors=priors,
        labels=labels,
        device='cuda',
        path=save_path
    )

plt.figure(figsize=(12, 3))
np.random.seed(4)

# draw random parameter from prior
y_true = [var.rvs(1)[0] for var in priors]
x_err = 0.01
x_obs = simulate_spectra(y_true) + np.random.normal(size=len(wave)) * x_err

dim = len(wave)
y_pred = engine.predict(x_obs, y_true=y_true, n_samples=12800, corner=True, corner_reweight=True, seed=0)
# y_pred, weights = engine.predict(x_obs, x_err=np.array([0.1]*dim), y_true=y_true, n_samples=12800, corner=True, corner_reweight=True, seed=0)
plt.show()

y_pred_med = np.median(y_pred, axis=0)


def plot_spectrum(wave, obs_spec, model_spec, lw=0.7, alpha=0.8, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(wave, obs_spec, color='C0', alpha=alpha, lw=lw)
    ax1.plot(wave, model_spec, color='C3', alpha=alpha, lw=lw)
    ax2.plot(wave, obs_spec - model_spec, color='C7', alpha=alpha, lw=lw*0.8)

    ax1.minorticks_on()
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='both', labelsize=12, labeltop=False)  # don't put tick labels at the top
    ax1.set_ylabel('Normalized Flux', fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()

    ax2.axhline(y=0, color='k', linestyle='--', dashes=(8, 2), alpha=alpha, lw=lw)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
    ax2.set_ylabel('Residual', fontsize=15)
    h2, l2 = ax2.get_legend_handles_labels()

    legend_elements = [
        Line2D([], [], color='C0', alpha=alpha, lw=1.2, label='Data'),
        Line2D([], [], color='C3', lw=1.2, label='Model'),
        Line2D([], [], color='C7', alpha=alpha, lw=1.2, label='Residual')
    ]

    ax2.legend(handles=legend_elements, frameon=True, loc='lower left', bbox_to_anchor=(1, -0.08), fontsize=12, borderpad=0.5)
    fig.align_ylabels((ax1, ax2))
    # ax1.set_title(f'APOGEE {apogee_id}, Teff={teff:.2f}±{e_teff:.2}, RV={rv:.2f}±{e_rv:.2f}, vsini={vsini:.2f}')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


plot_spectrum(wave, x_obs, simulate_spectra(y_pred_med), save_path=f'{user_path}/ML/Group7-Project/figure/Model Spectrum.pdf')