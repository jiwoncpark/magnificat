import unittest
import shutil
import numpy as np
from magnificat import cadence


class TestLSSTCadence(unittest.TestCase):
    """Tests for LSSTCadence class

    """
    @classmethod
    def setUpClass(cls):
        cls.out_dir = 'obs_testing'

    def test_get_pointings(self):
        """Test input and output shapes of get_pointings
        """
        cadence_obj = cadence.LSSTCadence(self.out_dir)
        ra, dec = cadence_obj.get_pointings(100)
        np.testing.assert_equal(len(ra), 100)
        np.testing.assert_equal(len(dec), 100)

    def test_get_obs_info(self):
        """Test queried visits of get_obs_info

        """
        cadence_obj = cadence.LSSTCadence(self.out_dir)
        ra, dec = cadence_obj.get_pointings(100)
        cadence_obj.get_obs_info(ra, dec, skip_ddf=True)
        # Final n_pointings should be <= requested 100
        # if there were DDF pointings
        assert cadence_obj.n_pointings <= 100
        for p in range(cadence_obj.n_pointings):
            # Number of visits should be < 1500 if DDF was skipped
            mjd = cadence_obj.get_mjd_single_pointing(p, rounded=False)
            assert len(mjd) < 1500

    def test_get_mjd_single_pointing(self):
        """Test `get_mjd_single_pointing`

        """
        cadence_obj = cadence.LSSTCadence(self.out_dir)
        ra, dec = cadence_obj.get_pointings(100)
        cadence_obj.get_obs_info(ra, dec, skip_ddf=True)
        mjd = cadence_obj.get_mjd_single_pointing(0, rounded=True)
        assert mjd.dtype is np.dtype(np.int64)
        assert np.min(mjd) >= 0
        assert np.max(mjd) <= 3649

    def test_get_mask_single_pointing(self):
        """Test `get_mask_single_pointing`

        """
        cadence_obj = cadence.LSSTCadence(self.out_dir)
        ra, dec = cadence_obj.get_pointings(100)
        cadence_obj.get_obs_info(ra, dec, skip_ddf=True)
        mjd = cadence_obj.get_mjd_single_pointing(0, rounded=True)
        T = len(mjd)
        mask = cadence_obj.get_mask_single_pointing(0)
        np.testing.assert_equal(mask.shape, [T, 6])
        assert mask.dtype is np.dtype(bool)

    def test_bin_by_day(self):
        """Test `bin_by_day`

        """
        cadence_obj = cadence.LSSTCadence(self.out_dir)
        ra, dec = cadence_obj.get_pointings(100)
        cadence_obj.get_obs_info(ra, dec, skip_ddf=True)
        cadence_obj.bin_by_day()
        obs_mask = cadence_obj.get_observed_mask()
        assert obs_mask.dtype is np.dtype(bool)
        trimmed_mjd = cadence_obj.get_trimmed_mjd()
        trimmed_T = len(trimmed_mjd)
        assert trimmed_T == sum(obs_mask)
        trimmed_mask = cadence_obj.get_trimmed_mask(0)
        np.testing.assert_equal(trimmed_mask.shape, [trimmed_T, 6])
        assert trimmed_mask.dtype is np.dtype(bool)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)


if __name__ == '__main__':
    unittest.main()
