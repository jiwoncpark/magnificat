import unittest
import numpy as np
import matplotlib.pyplot as plt
import magnificat.drw_utils as drw_utils


def get_drw_dc2(t_obs, tau, z, SF_inf, rng):
    """
    Return the delta mag_norm values wrt the infinite-time average
    mag_norm for the provided AGN light curve parameters.  mag_norm is
    the object's un-reddened monochromatic magnitude at 500nm.

    Taken with minor modification from
    https://github.com/LSSTDESC/sims_GCRCatSimInterface/blob/320ddc07432bcaa05723944738a6e02b6841b69e/python/desc/sims/GCRCatSimInterface/Variability.py#L169

    Parameters
    ----------
    expmjd: np.array
        Times at which to evaluate the light curve delta flux values
        in MJD.  Observer frame.
    tau: float
        Variability time scale in days.
    z: float
        redshift
    SF_inf: float
        Structure function parameter, i.e., asymptotic rms variability on
        long time scales.
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator

    Returns
    -------
    np.array of delta mag_norm values.

    Notes
    -----
    This code is based on/stolen from
    https://github.com/astroML/astroML/blob/master/astroML/time_series/generate.py

    """
    # mjds = np.array(expmjd)

    # agn_walk_start_date = 58580.0
    # if min(mjds) < agn_walk_start_date:
    # raise RuntimeError(f'mjds must start after {agn_walk_start_date}')

    # t_obs = np.arange(agn_walk_start_date, max(mjds + 1), dtype=float)
    t_rest = t_obs/(1.0 + z)/tau

    N = len(t_rest)
    steps = rng.normal(0, 1, N)
    delta_mag_norm = np.zeros(N)
    delta_mag_norm[0] = steps[0]*SF_inf
    for i in range(1, N):
        dt = t_rest[i] - t_rest[i - 1]
        delta_mag_norm[i] = (delta_mag_norm[i - 1]*(1. - dt)
                             + np.sqrt(2*dt)*SF_inf*steps[i])
    # dm_out = np.interp(mjds, t_obs, delta_mag_norm)
    return delta_mag_norm


class TestDRWUtils(unittest.TestCase):
    """Tests for `drw_utils.py`

    """
    @classmethod
    def setUpClass(cls):
        cls.seed = 123
        cls.z = 2.0

    def test_drw_dc2_vs_magnify(self):
        tau_rest = 300.0  # days
        SF_inf = 0.3  # mag

        t_obs = np.arange(200)  # days
        t_rest = t_obs/(1 + self.z)  # days

        # DC2 accepts rest-frame tau and t_obs
        drw_utils.dc2 = get_drw_dc2(t_obs,
                                    tau=tau_rest, SF_inf=SF_inf,
                                    z=self.z,
                                    rng=np.random.default_rng(self.seed))
        # magnify accepts rest-frame tau and t_rest
        drw_utils.magnify = drw_utils.get_drw(t_rest,
                                              tau=tau_rest, SF_inf=SF_inf,
                                              z=self.z, xmean=0.0,
                                              rng=np.random.default_rng(self.seed))

        plt.scatter(t_rest, drw_utils.dc2, marker='.', label='DC2')
        plt.scatter(t_rest, drw_utils.magnify, marker='.', label='magnify')
        plt.xlabel('t_rest')
        plt.ylabel('DRW LC')
        plt.legend()
        plt.savefig('drw_utils.dc2_vs_magnify.png')
        np.testing.assert_array_almost_equal(drw_utils.dc2, drw_utils.magnify, 6)


if __name__ == '__main__':
    unittest.main()
