import os
import numpy as np
import pandas as pd


class DC2Sampler:

    def __init__(self, seed=123, bandpasses=list('ugrizy'), idx=None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.bandpasses = bandpasses
        self.zp = -2.5*np.log10(3631)
        self.to_Jy = 1.e-9  # nanoJy to Jy
        self.idx = idx
        import magnificat.input_data as in_data
        self.in_data = in_data.__path__[0]
        self.flux_cols = [f'flux_{bp}' for bp in self.bandpasses]
        self.mag_cols = [f'mag_{bp}' for bp in self.bandpasses]
        self._load_catalog()
        self.count = 0

    def _load_param_cat(self):
        df = pd.read_csv(os.path.join(self.in_data, 'joined_0.csv'))
        cols = ['blackHoleAccretionRate', 'blackHoleEddingtonRatio',
                'redshift_x', 'galaxy_id', 'blackHoleMass',
                'M_i',
                'agn_sf_u', 'agn_sf_g', 'agn_sf_r', 'agn_sf_i', 'agn_sf_z', 'agn_sf_y',
                'agn_tau_u', 'agn_tau_g', 'agn_tau_r', 'agn_tau_i', 'agn_tau_z', 'agn_tau_y']
        df = df[cols].copy()
        rename_dict = {'blackHoleAccretionRate': 'accretion_rate',
                       'blackHoleEddingtonRatio': 'edd_ratio',
                       'redshift_x': 'redshift',
                       'blackHoleMass': 'BH_mass'}
        for bp in list('ugrizy'):
            # Log DRW params, since logged is closer to normal
            df[f'log_sf_inf_{bp}'] = np.log10(df[f'agn_sf_{bp}'].values)
            df[f'log_rf_tau_{bp}'] = np.log10(df[f'agn_tau_{bp}'].values)
            df.drop([f'agn_sf_{bp}', f'agn_tau_{bp}'],
                    columns=True, inplace=True)
        df.rename(rename_dict, inplace=True, axis=1)
        return df

    def _load_truth_cat(self):
        truth_cols = ['host_galaxy'] + self.flux_cols
        truth = pd.read_csv(os.path.join(self.in_data, 'dc2_truth.csv'),
                            index_col=None,
                            usecols=truth_cols
                            )
        # Convert flux to mag
        flux = truth[self.flux_cols].values
        truth[self.mag_cols] = self.flux_to_mag(flux*self.to_Jy)
        return truth[['host_galaxy'] + self.mag_cols]

    def _load_catalog(self):
        params = self._load_param_cat()
        truth = self._load_truth_cat()
        cat = params.merge(truth, how='inner', suffixes=['', '_duplicate'],
                           left_on='galaxy_id', right_on='host_galaxy',
                           validate='one_to_one')
        master_cols = ['galaxy_id', 'redshift', 'BH_mass', 'M_i']  # 'accretion_rate', 'edd_ratio'
        master_cols += self.mag_cols
        master_cols += [f'SF_inf_{bp}' for bp in self.bandpasses]
        master_cols += [f'tau_{bp}' for bp in self.bandpasses]
        cat.reset_index(drop=True, inplace=True)
        self.cat = cat[master_cols].iloc[self.idx].copy().reset_index(drop=True)
        self.cat['BH_mass'] = np.log10(self.cat['BH_mass'].values)
        self.size = self.cat.shape[0]

    def sample(self):
        # i = self.rng.integers(low=0, high=self.size)
        sample = self.cat.iloc[self.count]
        self.count += 1
        return sample

    def flux_to_mag(self, flux):
        mag = -2.5*np.log10(flux) - self.zp
        return mag


if __name__ == '__main__':
    sampler = DC2Sampler(123, ['i'], idx=[5, 6, 8])
    params = sampler.sample()
    print(params)
    params = sampler.sample()
    print(params)
    assert params['redshift'].dtype == np.float64
