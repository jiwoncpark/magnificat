import os
import os.path as osp
import math
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import magnificat.observation_utils as obs_utils


class LSSTCadence:
    min_mjd = 59580.139555
    nside_in = 32
    nested = False
    fov_radius = 1.75  # deg, field of view
    bp_to_int = dict(zip(list('ugrizy'), range(6)))
    int_to_bp = dict(zip(range(6), list('ugrizy')))
    bp = list('ugrizy')

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
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

    def get_pointings_single_hp(self, hp: int, n_pointings_init: int):
        """
        Get pointing positions from a single healpix, upgrading it if necessary

        Parameters
        ----------
        hp : int
            a single healpix id in DC2 (NSIDE=32, nested)
        n_pointings_init : int
            how many pointings to get from this healpix

        Returns
        -------
        tuple
            ra, dec of pointings

        """
        nside_out = obs_utils.get_target_nside(n_pointings_init, self.nside_in)
        # Upsampled healpix IDs in nested scheme
        hp_ids = obs_utils.upgrade_healpix(hp, self.nested,
                                           self.nside_in, nside_out)
        ra, dec = obs_utils.get_healpix_centers(hp_ids, nside_out, nest=True)
        return ra[:n_pointings_init], dec[:n_pointings_init]

    def get_pointings(self, n_pointings_init: int):
        """
        Get pointing positions all over the DC2 field

        Parameters
        ----------
        n_pointings_init : int
            how many pointings to get

        Returns
        -------
        tuple
            ra, dec of pointings

        """
        pointings_per_hp = math.ceil(n_pointings_init/len(self.hp_list))
        ra = np.empty(pointings_per_hp*len(self.hp_list))
        dec = np.empty(pointings_per_hp*len(self.hp_list))
        for i, hp in enumerate(self.hp_list):
            r, d = self.get_pointings_single_hp(hp, pointings_per_hp)
            ra[i*pointings_per_hp:(i+1)*pointings_per_hp] = r
            dec[i*pointings_per_hp:(i+1)*pointings_per_hp] = d
        return ra[:n_pointings_init], dec[:n_pointings_init]

    def get_obs_info(self, ra: np.ndarray, dec: np.ndarray,
                     skip_existing=True,
                     min_visits=0, skip_ddf=True):
        """Loop through pointings and query visits that fall inside FOV

        Note
        ----
        After DDF rejection, we might end up with a final `n_pointings`
        different from `n_pointings_init`

        """
        opsim = self.load_opsim_db()
        n_pointings = 0  # init
        for i, (r, d) in tqdm(enumerate(zip(ra, dec)), total=len(ra)):
            if skip_existing:
                if os.path.exists(osp.join(self.out_dir, f'mask_{i}.npy')):
                    continue
            obs_info_i = pd.DataFrame()
            # Get distance from each visit row to pointing, (r, d)
            dist, _, _ = obs_utils.get_distance(ra_f=opsim['ra'].values,
                                                dec_f=opsim['dec'].values,
                                                ra_i=r,
                                                dec_i=d)
            opsim['dist'] = dist  # deg, include in obs_info_i to store
            # Misc obs info
            obs_info_i = obs_info_i.append(opsim[dist < self.fov_radius],
                                           ignore_index=True)
            # Get filter assignment in int, for convenience
            filters = obs_info_i['filter'].values
            filters = np.array(list(map(self.bp_to_int.get, filters)))
            # Store MJD for easy access
            mjd = (obs_info_i['expMJD'].values - self.min_mjd)
            mjd.sort()
            # Skip DDF visits, which have more than 1500 visits
            if len(mjd) < min_visits:
                continue
            if skip_ddf:
                if len(mjd) > 1500:
                    continue
            # Store observation mask, 1 where observed in filter else 0
            mask = np.zeros([len(mjd), 6]).astype(bool)  # [n_obs, 6]
            for bp_i in range(6):
                mask[:, bp_i] = (filters == bp_i)
            # Store obs info, MJD, and mask for later access
            obs_info_i.to_csv(osp.join(self.out_dir,
                                       f'obs_{n_pointings}.csv'),
                              index=None)
            np.save(osp.join(self.out_dir,
                             f'mjd_{n_pointings}.npy'), mjd)
            np.save(osp.join(self.out_dir,
                             f'mask_{n_pointings}.npy'), mask)
            n_pointings += 1
        self.n_pointings = n_pointings

    def bin_by_day(self, skip_existing=True):
        """Bin the observations by day

        Parameters
        ----------
        skip_existing : bool, optional
            whether to skip operations for pointings already saved
            to disk

        """
        # Mask indicating if a time was observed at least once in any filter
        full_mjd = np.arange(3650)
        observed = np.zeros(3650).astype(bool)  # init as all unobserved
        # Iterate through all pointings to identify days to trim
        for p in range(self.n_pointings):
            mjd = self.get_mjd_single_pointing(p, rounded=True)  # [n_obs,]
            observed[mjd] = True  # mask observed
        np.save(osp.join(self.out_dir,
                         'observed_mask.npy'), observed)
        mjd_trimmed = full_mjd[observed]  # [trimmed_T,]
        print(f"Trimmed MJD has {len(mjd_trimmed)} out of 3650 days.")
        np.save(osp.join(self.out_dir,
                         'mjd_trimmed.npy'), mjd_trimmed)
        # Compile binned mask
        for p in range(self.n_pointings):
            if skip_existing:
                if os.path.exists(osp.join(self.out_dir,
                                           f'trimmed_mask_{p}.npy')):
                    continue
            # 10-year full mask in all filters
            full = np.zeros([3650, 6]).astype(bool)  # init, all unobserved
            # 10-year observed days combined across filters for this pointing
            mjd = self.get_mjd_single_pointing(p, rounded=True)  # [n_obs,]
            mask = self.get_mask_single_pointing(p)  # [n_obs, 6]
            for bp_i in range(6):
                full[mjd, bp_i] = mask[:, bp_i]
            trimmed = full[observed, :]  # trim so only relevant times remain
            np.save(osp.join(self.out_dir,
                             f'trimmed_mask_{p}.npy'), trimmed)

    def get_observed_mask(self):
        """Get trimmed MJD that is observed in at least one band at all
        times

        """
        return np.load(osp.join(self.out_dir, 'observed_mask.npy'))

    def get_trimmed_mjd(self):
        """Get trimmed MJD that is observed in at least one band at all
        times

        """
        return np.load(osp.join(self.out_dir, 'mjd_trimmed.npy'))

    def get_trimmed_mask(self, i: int, as_tensor=False):
        """Get trimmed mask that has times corresponding to trimmed_mjd

        """
        mask = np.load(osp.join(self.out_dir, f'trimmed_mask_{i}.npy'))
        mask = mask.astype(bool)
        if as_tensor:
            mask = torch.from_numpy(mask).to(torch.bool)
        return mask

    def get_mjd_single_pointing(self, i: int, rounded: bool):
        mjd = np.load(osp.join(self.out_dir, f'mjd_{i}.npy'))
        if rounded:
            mjd = np.round(mjd).astype(np.int64)
        return mjd

    def get_mjd_i_band_pointing(self, i: int, rounded: bool):
        mjd = np.load(osp.join(self.out_dir, f'mjd_{i}.npy'))
        mask = np.load(osp.join(self.out_dir, f'mask_{i}.npy')).astype(bool)
        i_band_int = mjd[mask[:, 3]]
        return i_band_int

    def get_mask_single_pointing(self, i: int):
        mask = np.load(osp.join(self.out_dir, f'mask_{i}.npy')).astype(bool)
        return mask

    def load_opsim_db(self):
        """Load the OpSim database with relevant columns as an iterator

        """
        con = sqlite3.connect(osp.join(self.in_data, 'minion_1016_desc_dithered_v4_trimmed.db'))
        db = pd.read_sql_query("SELECT {:s} FROM Summary".format(', '.join(self.cols)),
                               con)
        db['ra'] = np.rad2deg(db['descDitheredRA'].values)
        db['dec'] = np.rad2deg(db['descDitheredDec'].values)
        # print("min MJD: ", db['expMJD'].min())
        return db


if __name__ == '__main__':
    cadence_obj = LSSTCadence('obs')
    ra, dec = cadence_obj.get_pointings(100)
    cadence_obj.get_obs_info(ra, dec)
    mjd = cadence_obj.get_mjd_single_pointing(0, rounded=False)
    print("mjd: ", mjd.shape)
    mask = cadence_obj.get_mask_single_pointing(0)
    print("mask: ", mask.shape)
    print(ra.shape)
    cadence_obj.bin_by_day()

