import os
import os.path as osp
from functools import cached_property
import numpy as np
from numpy.random import default_rng
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from magnificat import drw_utils
import magnificat.input_data as input_data


def load_sdss_dr7_catalogs(bandpasses):
    drw_path = os.path.join(input_data.__path__[0],
                            'sdss_dr7_s82', 's82drw_{:s}.dat')
    shen_path = os.path.join(input_data.__path__[0],
                             'sdss_dr7_s82', 'DB_QSO_S82.dat')
    # Normalizing wavelength
    wavelength_norm = 4000.0  # in angstroms, from MacLeod et al 2010
    # Central wavelength (value) for each SDSS bandpass (key) used to
    # estimate the rest-frame wavelength
    wavelength_center = {'u': 3520.0,
                         'g': 4800.0,
                         'r': 6250.0,
                         'i': 7690.0,
                         'z': 9110.0}  # in angstroms, from MacLeod et al 2010
    # Cols of s82drew_*.dat
    # http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_drw.html
    drw_columns = ['SDR5ID', 'ra', 'dec', 'redshift', 'M_i', 'log_mass_BH', 'chi2']
    drw_columns += ['log_tau', 'log_tau_lowlim', 'log_tau_uplim']
    drw_columns += ['log_sighat', 'log_sfhat_lowlim', 'log_sfhat_uplim']
    drw_columns += ['edge', 'll_model', 'll_noise', 'll_inf', 'mu', 'N_obs']
    # Dictionary of pandas dataframes, each dataframe representing the band
    drw_dict = {}
    for bp in bandpasses:
        drw_bp = pd.read_csv(drw_path.format(bp),
                             index_col=False, sep=r'\s+',
                             skiprows=3, names=drw_columns)
        # z correction
        z_corr = 1.0/(1.0 + drw_bp['redshift'].values)
        # normalized rest-frame wavelength
        drw_bp['rf_wavelength'] = wavelength_center[bp]*z_corr/wavelength_norm
        drw_bp['log_rf_wavelength'] = np.log10(drw_bp['rf_wavelength'].values)
        # log of rest-frame tau in days
        drw_bp['log_rf_tau'] = drw_bp['log_tau'].values - np.log10(1.0 + drw_bp['redshift'].values)
        # log of SF_inf in mag (with the proper unit conversion, see Note (4) in schema)
        drw_bp['log_sf_inf'] = drw_bp['log_sighat'].values + 0.5*drw_bp['log_rf_tau'] - 0.5*np.log10(365.0)
        drw_bp['bandpass'] = bp
        # can't use SDR5ID b/c collapses new QSOs as -1 (not unique)
        drw_bp['id'] = np.arange(len(drw_bp))
        drw_dict[bp] = drw_bp
    # Concatenate across bands
    drw_all = pd.concat(drw_dict.values(), axis=0)
    # http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_DB.html
    shen = clean_shen_et_al_2008_catalog(shen_path, bandpasses)
    merged = drw_all.merge(shen, on=['ra', 'dec'],
                           how='inner', suffixes=('', '_shen'))
    np.testing.assert_array_almost_equal(merged['M_i'].values,
                                         merged['M_i_shen'].values)
    np.testing.assert_array_almost_equal(merged['redshift'].values,
                                         merged['redshift_shen'].values)
    return merged


def clean_shen_et_al_2008_catalog(in_path, bandpasses, out_path=None):
    """Clean the catalog in
    http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_DB.html

    """
    from astropy.io import ascii
    # Reading the raw data file
    DATA = ascii.read(in_path)
    # Extracting useful columns from the DATA file
    dbID = DATA.field('col1')
    SDSS_id = DATA.field('col4')
    RA = DATA.field('col2')
    DEC = DATA.field('col3')
    M_i = DATA.field('col5')
    redshift = DATA.field('col7')
    BH_MASS = DATA.field('col8')
    mags = dict()
    mags['u'] = DATA.field('col10')
    mags['g'] = DATA.field('col11')
    mags['r'] = DATA.field('col12')
    mags['i'] = DATA.field('col13')
    mags['z'] = DATA.field('col14')
    # Converting the ID to an integer
    dbID = [int(i) for i in dbID]
    # Generating columns for the cleaned Stripe 82 data
    shen_df = pd.DataFrame(dbID, columns=['dbID'])
    shen_df['SDSS_id'] = SDSS_id
    shen_df['ra'] = RA
    shen_df['dec'] = DEC
    shen_df['M_i'] = M_i
    shen_df['redshift'] = redshift
    shen_df['BH_mass'] = BH_MASS
    for bp in bandpasses:
        shen_df[bp] = mags[bp]
    # Remove AGN without BH mass measurements
    shen_df = shen_df[shen_df['BH_mass'] > 0.1]
    if out_path is None:
        return shen_df
    else:
        shen_df.to_csv(out_path, index=False)


class SDSSDR7Dataset(Dataset):
    """
    Dataset of SDSS DR7 light curves with metadata

    Parameters
    ----------
    max_x : float
        Maximum observer-frame time in days
    delta_x : float
        Resolution of DRW rendering in observer-frame days

    """
    bp_to_int = dict(zip(list('ugriz'), range(5)))

    def __init__(self,
                 agn_params,
                 bp_params,
                 bandpasses,
                 out_dir,
                 num_samples,
                 rescale_x=1.0/(3339*0.5)*4.0,
                 shift_x=-3339*0.5,
                 metadata_kwargs=dict(keep_agn_mode='max_obs'),
                 light_curve_kwargs=dict(),):
        self.agn_params = agn_params
        self.bp_params = bp_params
        self.bandpasses = bandpasses
        assert 'i' in self.bandpasses
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'processed'), exist_ok=True)
        self.num_samples = num_samples
        self.rescale_x = rescale_x
        self.shift_x = shift_x
        # For standardizing params
        self.mean_params = None
        self.std_params = None
        self.slice_params = None
        self.cleaned_metadata_path = os.path.join(self.out_dir,
                                                  'processed',
                                                  'metadata.dat')
        self.metadata_kwargs = metadata_kwargs
        self.light_curve_kwargs = light_curve_kwargs
        self._process_metadata(**self.metadata_kwargs)
        self.lc_ids = np.sort(np.unique(self.metadata['dbID'].values))
        self.int_to_lc_id = dict(zip(np.arange(len(self.lc_ids)), self.lc_ids))
        self._process_light_curves(**self.light_curve_kwargs)

    def _process_metadata(self, keep_agn_mode):
        """Process the metadata storing the target labels

        """
        if os.path.exists(self.cleaned_metadata_path):
            self.metadata = pd.read_csv(self.cleaned_metadata_path,
                                        index_col=None)
        else:
            self._load_metadata()
            self._apply_selection()
            self._set_keep_agn(keep_agn_mode)
            self._save_metadata()

    def _load_metadata(self):
        """Read in the DRW and BH metadata

        """
        self.metadata = load_sdss_dr7_catalogs(self.bandpasses)

    def _apply_selection(self):
        """Apply the MacLeod et al 2010 selection

        """
        # Apply selection
        merged = self.metadata.copy()
        before_cut = len(self.metadata)
        # 1. Number of observations
        # merged = merged.query('(log_tau != -10.0) & (log_sighat != -10.0)') # same result as below
        merged = merged.query('N_obs >= 10')
        after_1 = len(merged)
        print("N_obs selection removed: ", before_cut - after_1)
        print("and left: ", after_1)
        # 2. Model likelihood
        merged['del_noise'] = merged['ll_model'] - merged['ll_noise']
        merged = merged.query('del_noise > 2.0')
        after_2 = len(merged)
        print("Model likelihood selection removed: ", after_1 - after_2)
        print("and left: ", after_2)
        # 3. Tau is not a lower limit
        merged['del_inf'] = merged['ll_model'] - merged['ll_inf']
        merged = merged.query('del_inf > 0.05')
        after_3 = len(merged)
        print("Tau lower limit selection removed: ", after_2 - after_3)
        print("and left: ", after_3)
        # 4. Edge
        merged = merged.query('edge == 0')
        after_edgecut = len(merged)
        print("Edge selection removed: ", after_3 - after_edgecut)
        print("and left: ", after_edgecut)
        for bp in list('ugriz'):
            print(bp, merged[merged['bandpass'] == bp].shape[0])
        self.selected = merged

    def _set_keep_agn(self, keep_agn_mode):
        """Set which AGN IDs to keep

        """
        n_obs = np.empty(len(self.bandpasses))
        for bp_i, bp in enumerate(self.bandpasses):
            n_obs[bp_i] = self.selected[self.selected['bandpass'] == bp].shape[0]
        if keep_agn_mode == 'max_obs':
            max_obs_i = np.argmax(n_obs)
            max_bp = self.bandpasses[max_obs_i]
            keep_id = self.selected[self.selected['bandpass'] == max_bp]['dbID'].unique()
            self.metadata = self.metadata[self.metadata['dbID'].isin(keep_id)].copy()
        else:
            raise NotImplementedError

    def _save_metadata(self):
        self.metadata['tau'] = 10.0**(self.metadata['log_tau'].values)
        self.metadata['SF_inf'] = 10.0**(self.metadata['log_sf_inf'].values)
        self.metadata.to_csv(self.cleaned_metadata_path, index=None)

    @cached_property
    def cadence(self):
        # Unique IDs of AGN we kept
        lc_ids = np.sort(np.unique(self.metadata['dbID'].values))
        min_mjds = []
        max_mjds = []
        # LC columns in the catalog (order is important)
        lc_columns = []
        for bp in list('ugriz'):
            lc_columns += [f'mjd_{bp}', bp, f'{bp}_err']
        lc_columns += ['ra', 'dec']
        for lc_id in lc_ids:
            lc_path = os.path.join(input_data.__path__[0],
                                   'sdss_dr7_s82', 'QSO_S82', str(lc_id))
            lc = pd.read_csv(lc_path, sep=' ', header=None,
                             names=lc_columns)
            for bp in self.bandpasses:
                mjd = lc[f'mjd_{bp}'].values
                flux = lc[f'{bp}'].values
                # We discard the nonsensical MJDs < 10 days
                is_good = np.logical_and((flux > 0.0), (mjd > 10.0))
                mjd = mjd[is_good]
                flux = flux[is_good]
                if np.min(mjd) < 50000:
                    print("mjd < 50000 anomaly: ", lc_id)
                # = lc[[f'mjd_{bp}', f'{bp}']].values
                min_mjds.append(np.min(mjd))
                max_mjds.append(np.max(mjd))
        cadence = dict(global_min_mjd=np.min(min_mjds),
                       global_max_mjd=np.max(max_mjds),
                       survey_length=np.max(max_mjds) - np.min(min_mjds))
        return cadence

    def _process_light_curves(self):
        """Process the light curves, which are the NN input

        """
        # LC columns in the catalog (order is important)
        lc_columns = []
        for bp in list('ugriz'):
            lc_columns += [f'mjd_{bp}', bp, f'{bp}_err']
        lc_columns += ['ra', 'dec']
        for agn_i, lc_id in tqdm(enumerate(self.lc_ids), total=len(self.lc_ids)):
            if os.path.exists(os.path.join(self.out_dir, 'processed',
                                           f'x_{agn_i}.npy')):
                continue
            lc_path = os.path.join(input_data.__path__[0],
                                   'sdss_dr7_s82', 'QSO_S82', str(lc_id))
            lc = pd.read_csv(lc_path, sep=' ', header=None, names=lc_columns)
            # Figure out maximum number of common observations
            n_obs_per_bp = []
            for bp in self.bandpasses:
                # Populate asynchronous mjd, flux pairs per band
                mjd = lc[f'mjd_{bp}'].values
                flux = lc[f'{bp}'].values
                flux_err = lc[f'{bp}_err'].values
                is_good = np.logical_and((flux > 0.0), (mjd > 10.0))
                mjd = mjd[is_good] - self.cadence['global_min_mjd']
                flux = flux[is_good]
                n_obs_per_bp.append(len(mjd))
            n_obs = min(n_obs_per_bp)
            # Initialize light curve arrays
            x = np.empty([n_obs, len(self.bandpasses)])  # times for all bands
            y = np.empty([n_obs, len(self.bandpasses)])  # fluxes for all bands
            y_err = np.empty([n_obs, len(self.bandpasses)])  # flux errors for all bands
            # Populate light curve info one band at a time
            for bp in self.bandpasses:
                bp_i = self.bp_to_int[bp]
                # Populate asynchronous mjd, flux pairs per band
                mjd = lc[f'mjd_{bp}'].values
                flux = lc[f'{bp}'].values
                flux_err = lc[f'{bp}_err'].values
                is_good = np.logical_and((flux > 0.0), (mjd > 10.0))
                mjd = mjd[is_good] - self.cadence['global_min_mjd']
                flux = flux[is_good]
                flux_err = flux_err[is_good]
                # Discard observations beyond n_obs... (TODO: don't do this)
                x[:, bp_i] = mjd[:n_obs]
                y[:, bp_i] = flux[:n_obs]
                y_err[:, bp_i] = flux_err[:n_obs]
            # Save x, y, y_err to disk
            np.save(os.path.join(self.out_dir, 'processed',
                                 f'x_{agn_i}.npy'),
                    x, allow_pickle=True)
            np.save(os.path.join(self.out_dir, 'processed',
                                 f'y_{agn_i}.npy'),
                    y, allow_pickle=True)
            np.save(os.path.join(self.out_dir, 'processed',
                                 f'y_err_{agn_i}.npy'),
                    y_err, allow_pickle=True)

    def __getitem__(self, index):
        # [n_obs, n_filters]
        x = torch.from_numpy(np.load(os.path.join(self.out_dir, 'processed',
                                                  f'x_{index}.npy'),
                                     allow_pickle=True)).float()
        y = torch.from_numpy(np.load(os.path.join(self.out_dir, 'processed',
                                                  f'y_{index}.npy'),
                                     allow_pickle=True)).float()
        y_err = torch.from_numpy(np.load(os.path.join(self.out_dir, 'processed',
                                                      f'y_err_{index}.npy'),
                                         allow_pickle=True)).float()
        y_err = torch.minimum(y_err, torch.ones_like(y_err)*0.01)
        # Rescale x
        x += self.shift_x
        x *= self.rescale_x
        # Add noise
        y += torch.randn_like(y_err)
        y -= 20.0  # arbitrarily subtract 20 to get flux in a nice range
        # Collect params
        params = []
        sub_meta = self.metadata[self.metadata['dbID'] == self.int_to_lc_id[index]]
        agn_row = sub_meta.iloc[[0]]  # single row for querying AGN params
        params += agn_row[self.agn_params].values.tolist()[0]
        for bp in self.bandpasses:
            params += sub_meta.loc[sub_meta['bandpass'] == bp, self.bp_params].values.tolist()[0]
        params = torch.tensor(params).float()

        # Standardize params
        if self.mean_params is not None:
            params -= self.mean_params
            params /= self.std_params
        return x, y, params

    def get_normalizing_metadata(self, dataloader):
        mean_params = 0.0
        std_params = 0.0
        for i, (_, _, _, _, params) in enumerate(dataloader):
            # params ~ list of batch_size tensors, each of shape [n_params]
            stacked_params = torch.cat(params, dim=0).to(torch.device('cpu'))
            mean_params += (torch.mean(stacked_params, dim=0) - mean_params)/(i+1)
            std_params += (torch.std(stacked_params, dim=0) - std_params)/(i+1)
        self.mean_params = mean_params
        self.std_params = std_params

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    train_dataset = SDSSDR7Dataset(out_dir='sdss_dr7',
                                   agn_params=['M_i', 'BH_mass', 'redshift'],
                                   bp_params=['log_rf_tau', 'log_sf_inf'],
                                   bandpasses=list('ugriz'),
                                   num_samples=10,
                                   rescale_x=1.0/(3339*0.5)*4.0,
                                   shift_x=-3650*0.5,
                                   metadata_kwargs=dict(keep_agn_mode='max_obs'),
                                   light_curve_kwargs=dict(),)
    print(train_dataset.mean_params, train_dataset.std_params)
    x, y, params = train_dataset[0]
    print(x.shape, y.shape, params.shape)



