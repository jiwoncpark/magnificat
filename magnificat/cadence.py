import os
import os.path as osp
import math
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np
import magnificat.observation_utils as obs_utils


class LSSTCadence:
    min_mjd = 59853.01679398148
    nside_in = 32
    nested = False
    fov_radius = 1.75  # deg, field of view

    def __init__(self, out_dir, seed: int = 1234):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.seed = seed
        cols = ['expMJD', 'visitExpTime', 'obsHistID']
        cols += ['descDitheredRA', 'descDitheredDec', 'fiveSigmaDepth']
        cols += ['filtSkyBrightness']
        cols += ['filter', 'FWHMgeom', 'FWHMeff']
        self.cols = cols
        import magnificat.input_data as in_data
        self.in_data = in_data.__path__[0]
        # List of DC2 healpixes (NSIDE=32, ring scheme)
        with open(osp.join(self.in_data, 'healpix_list_dc2.txt'), 'r') as f:
            hp_list = [int(line.rstrip()) for line in f.readlines()]
        self.hp_list = hp_list

    def get_pointings_single_hp(self, hp: int, n_pointings: int):
        """
        Get pointing positions from a single healpix, upgrading it if necessary

        Parameters
        ----------
        hp : int
            a single healpix id in DC2 (NSIDE=32, nested)
        n_pointings : int
            how many pointings to get from this healpix

        Returns
        -------
        tuple
            ra, dec of pointings

        """
        nside_out = obs_utils.get_target_nside(n_pointings, self.nside_in)
        # Upsampled healpix IDs in nested scheme
        hp_ids = obs_utils.upgrade_healpix(hp, self.nested,
                                           self.nside_in, nside_out)
        ra, dec = obs_utils.get_healpix_centers(hp_ids, nside_out, nest=True)
        return ra, dec

    def get_pointings(self, n_pointings: int):
        """
        Get pointing positions all over the DC2 field

        Parameters
        ----------
        n_pointings : int
            how many pointings to get

        Returns
        -------
        tuple
            ra, dec of pointings

        """
        pointings_per_hp = math.ceil(n_pointings/len(self.hp_list))
        ra = np.empty(n_pointings)
        dec = np.empty(n_pointings)
        for i, hp in enumerate(self.hp_list):
            r, d = self.get_pointings_single_hp(hp, pointings_per_hp)
            ra[i*pointings_per_hp:(i+1)*pointings_per_hp] = r
            dec[i*pointings_per_hp:(i+1)*pointings_per_hp] = d
        return ra[:n_pointings], dec[:n_pointings]

    def get_obs_info(self, ra: np.ndarray, dec: np.ndarray):
        opsim = self.load_opsim_db()
        for i, (r, d) in tqdm(enumerate(zip(ra, dec)), total=len(ra)):
            obs_info_i = pd.DataFrame()
            dist, _, _ = obs_utils.get_distance(ra_f=opsim['ra'].values,
                                                dec_f=opsim['dec'].values,
                                                ra_i=r,
                                                dec_i=d)
            opsim['dist'] = dist  # deg
            obs_info_i = obs_info_i.append(opsim[opsim['dist'] < self.fov_radius],
                                           ignore_index=True)
            obs_info_i.to_csv(osp.join(self.out_dir, f'obs_{i}.csv'), index=None)
            shifted_mjd = (obs_info_i['expMJD'].values - self.min_mjd)
            np.save(osp.join(self.out_dir, f'mjd_{i}.npy'), shifted_mjd)

    def get_mjd_single_pointing(self, i: int, rounded: bool):
        mjd = np.load(osp.join(self.out_dir, f'mjd_{i}.npy'))
        if rounded:
            mjd = np.round(mjd)
        return mjd

    def load_opsim_db(self):
        """Load the OpSim database with relevant columns as an iterator

        """
        con = sqlite3.connect(osp.join(self.in_data, 'minion_1016_desc_dithered_v4_trimmed.db'))
        db = pd.read_sql_query("SELECT {:s} FROM Summary".format(', '.join(self.cols)),
                               con)
        db['ra'] = np.rad2deg(db['descDitheredRA'].values)
        db['dec'] = np.rad2deg(db['descDitheredDec'].values)
        return db


if __name__ == '__main__':
    cadence_obj = LSSTCadence('obs', 1234)
    ra, dec = cadence_obj.get_pointings(100)
    print(ra.shape)
    cadence_obj.get_obs_info(ra, dec)
