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
    # Cols of s82drw_*.dat
    # http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_drw.html
    drw_columns = ['SDR5ID', 'ra', 'dec', 'redshift', 'M_i', 'log_mass_BH', 'chi2']
    drw_columns += ['log_tau', 'log_sighat', 'log_tau_lowlim', 'log_tau_uplim']
    drw_columns += ['log_sfhat_lowlim', 'log_sfhat_uplim']
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
    merged.drop(['M_i_shen', 'redshift_shen'], axis=1, inplace=True)
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


class S82Sampler:
    """Sampler of AGN parameters from S82. Parameters can be those used to
    render the light curves or those to be inferred by the network.

    """
    bp_to_int = dict(zip(list('ugriz'), range(5)))

    def __init__(self,
                 agn_params,  # ['BH_mass', 'redshift', 'M_i']
                 bp_params,  # ['log_rf_tau', 'log_sf_inf']
                 bandpasses,
                 out_dir,
                 seed=123):
        self.agn_params = agn_params
        self.bp_params = bp_params
        self.bandpasses = bandpasses
        assert 'i' in self.bandpasses
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'processed'), exist_ok=True)
        # For standardizing params
        self.mean_params = None
        self.std_params = None
        self.slice_params = None
        self.cleaned_metadata_path = os.path.join(self.out_dir,
                                                  'processed',
                                                  'metadata.dat')
        self.rng = np.random.default_rng(seed)
        self._idx = None  # init

    def process_metadata(self, keep_agn_mode='max_obs'):
        """Process the metadata storing the target labels

        """
        if os.path.exists(self.cleaned_metadata_path):
            self.metadata = pd.read_csv(self.cleaned_metadata_path,
                                        index_col=None)
        else:
            self._load_metadata()
            self._apply_selection()
            self._set_keep_agn(keep_agn_mode)
            self._collapse_bandpasses()
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

    def _collapse_bandpasses(self):
        """Collapse rows for bandpasses belonging to one AGN to a single row

        """
        self.metadata = self.metadata.pivot(['id'] + self.agn_params,
                                            'bandpass',
                                            self.bp_params)
        self.metadata.columns = ['_'.join(col).strip() for col in \
                                 self.metadata.columns.values]
        self.metadata = self.metadata.reset_index()
        self.metadata['id'] = np.arange(self.metadata.shape[0])

    def _save_metadata(self):
        # self.metadata['tau'] = 10.0**(self.metadata['log_rf_tau'].values)
        # self.metadata['SF_inf'] = 10.0**(self.metadata['log_sf_inf'].values)
        self.metadata.to_csv(self.cleaned_metadata_path, index=None)

    @property
    def idx(self):
        """Indices of AGNs to sample from"""
        return self._idx

    @idx.setter
    def idx(self, idx):
        """Indices of AGNs to sample from"""
        if max(idx) > len(self) - 1:
            raise ValueError("Indices can't exceed size of dataset, "
                             f"which is {len(self)}.")
        self._idx = idx

    def sample(self):
        i = self.rng.choice(self.idx, replace=True)
        return self.metadata.iloc[i]

    def __len__(self):
        if hasattr(self, 'metadata'):
            return self.metadata.shape[0]
        else:
            return 0


if __name__ == '__main__':
    sampler = S82Sampler(agn_params=['BH_mass', 'redshift', 'M_i'],
                         bp_params=['log_rf_tau', 'log_sf_inf'],
                         bandpasses=list('ugriz'),
                         out_dir='s82_sampler_testing',
                         seed=123)
    sampler.process_metadata()
    sampler.idx = [0, 1]
    params = sampler.sample()
    print(params)
    params = sampler.sample()
    print(params)
    assert params['redshift'].dtype == np.float64
