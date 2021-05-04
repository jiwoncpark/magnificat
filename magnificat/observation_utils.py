import math
import numpy as np
import healpy as hp
import sqlite3
import pandas as pd

__all__ = ['get_pointings', 'upgrade_healpix', 'get_healpix_centers']
__all__ += ['get_target_nside']


def load_opsim_db(db_path):
    # Create your connection.
    cnx = sqlite3.connect(db_path)
    obs_hist = pd.read_sql_query("SELECT * FROM ObsHistory", cnx)
    cols = ['observationStartMJD', 'observationId', 'numExposures', 'filter']
    cols += ['seeingFwhmGeom', 'seeingFwhmEff', 'seeingFwhm500']
    cols += ['fiveSigmaDepth', 'skyBrightness', 'ra', 'dec']
    obs_hist = obs_hist[cols]
    return obs_hist


def get_pointings(n_pointings, healpix_in, nside_in, seed):
    rng = np.random.default_rng(seed=seed)
    target_nside = get_target_nside(n_pointings, nside_in=nside_in)
    sightline_ids = upgrade_healpix(healpix_in, False,
                                    nside_in, target_nside)
    ra_grid, dec_grid = get_healpix_centers(sightline_ids, target_nside,
                                            nest=True)
    # Randomly choose number of sightlines requested
    rand_i = rng.choice(np.arange(len(ra_grid)),
                        size=n_pointings,
                        replace=False)
    ra_grid, dec_grid = ra_grid[rand_i], dec_grid[rand_i]
    return ra_grid, dec_grid


def upgrade_healpix(pix_id, nested, nside_in, nside_out):
    """Upgrade (superresolve) a healpix into finer ones

    Parameters
    ----------
    pix_id : int
        coarse healpix ID to upgrade
    nested : bool
        whether `pix_id` is given in NESTED scheme
    nside_in : int
        NSIDE of `pix_id`
    nside_out : int
        desired NSIDE of finer healpix

    Returns
    -------
    np.array
        the upgraded healpix IDs in the NESTED scheme

    """
    if not nested:
        pix_id = hp.ring2nest(nside_in, pix_id)
    order_diff = np.log2(nside_out) - np.log2(nside_in)
    factor = 4**order_diff
    upgraded_ids = pix_id*factor + np.arange(factor)
    return upgraded_ids.astype(int)


def get_healpix_centers(pix_id, nside, nest):
    """Get the ra, dec corresponding to centers of the healpixels with given IDs

    Parameters
    ----------
    pix_id : int or array-like
        IDs of healpixels to evaluate centers. Must be in NESTED scheme
    nside_in : int
        NSIDE of `pix_id`

    """
    theta, phi = hp.pix2ang(nside, pix_id, nest=nest)
    ra, dec = np.degrees(phi), -np.degrees(theta-0.5*np.pi)
    return ra, dec


def get_target_nside(n_pix, nside_in=2**5):
    """Get the NSIDE corresponding to the number of sub-healpixels

    Parameters
    ----------
    n_pix : int
        desired number of pixels
    nside_in : int
        input NSIDE to subsample

    """
    order_in = int(np.log2(nside_in))
    order_diff = math.ceil(np.log(n_pix)/np.log(4.0))  # round up log4(n_pix)
    order_out = order_diff + order_in
    nside_out = int(2**order_out)
    return nside_out


def get_distance(ra_i, dec_i, ra_f, dec_f):
    """Compute the distance between two angular positions given in degrees

    """
    ra_diff = (ra_f - ra_i)*np.cos(np.deg2rad(dec_f))
    dec_diff = (dec_f - dec_i)
    return np.linalg.norm(np.vstack([ra_diff, dec_diff]), axis=0), ra_diff, dec_diff