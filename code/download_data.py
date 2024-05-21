import os
import wget
import smart
import numpy as np
import apogee_tools as ap
from tqdm import tqdm
from astropy.io import fits
from astroquery.vizier import Vizier

instrument = 'apogee'
apogee_type = 'aspcap'
user_path = os.path.expanduser('~')
data_path = f'{user_path}/ML/data'
spec_path = f'{data_path}/spec'

if apogee_type == 'apstar':
    prefix = 'apStar'
elif apogee_type == 'apvisit':
    prefix = 'apVisit'
elif apogee_type == 'aspcap':
    prefix = 'aspcapStar'

apogee_table = Vizier( 
    columns=['ID', 'Loc', 'RAJ2000', 'DEJ2000', 'SNR', 'HRV', 'e_HRV', 'Teff', 'e_Teff', 'logg', 'e_logg', 'Vsini', '[M/H]', 'e_[M/H]'], 
    column_filters={'Teff': '2500..4000', 'Vsini': '>0'},
    row_limit=-1
).get_catalogs('III/284/allstars')[0]

apogee_table.rename_columns(
    ['__M_H_', 'e__M_H_'],
    ['[M/H]', 'e_[M/H]'],
)

hdulist = fits.open('~/Software/apogee_data/allStar-dr17-synspec_rev1.fits')
apogee_id_list = hdulist[1].data['APOGEE_ID']

print('Searching for location id...')
idxs = np.zeros(len(apogee_table), dtype=int)
for i in tqdm(range(len(apogee_table))):
    idxs[i] = np.where(apogee_id_list==apogee_table['ID'][i])[0][0]

telescope   = hdulist[1].data['TELESCOPE'][idxs]
field       = hdulist[1].data['FIELD'][idxs]

constraint = telescope == 'apo25m'
apogee_table = apogee_table[constraint]
telescope = telescope[constraint]
field = field[constraint]

apogee_table.write(f'{data_path}/apogee_table.csv', overwrite=True)

with open(f'{user_path}/ML/data/apogee_download.txt', 'w') as file:
    for i in range(len(apogee_table)):
        file.write(f"{field[i]}/apStar-dr17-{apogee_table['ID'][i]}.fits\n")

save_path = f'{data_path}/spec/'
# Download data
for i in tqdm(range(len(apogee_table))):
    main_url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/apo25m/{field[i]}/{prefix}-dr17-{apogee_table['ID'][i]}.fits"
    try:
        wget.download(main_url, save_path)
    except:
        print(f'{main_url} Not Found!')
        continue


# Save download successful table
download_successful = np.ones(len(apogee_table), dtype=bool)
for i in tqdm(range(len(apogee_table))):
    apogee_id = apogee_table['ID'][i]
    object_path = f'{spec_path}/{prefix}-dr17-{apogee_id}.fits'
    if not os.path.exists(object_path):
        download_successful[i] = 0
        continue
    # spec = smart.Spectrum(name=apogee_id, path=object_path, instrument=instrument, apply_sigma_mask=True, datatype=apogee_type, applytell=True)

apogee_table = apogee_table[download_successful]
field = field[download_successful]
apogee_table['Field'] = field
apogee_table.write(f'{data_path}/apogee_table.csv', overwrite=True)