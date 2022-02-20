import os
import unittest
import shutil
import numpy as np
from magnificat.drw_dataset import DRWDataset
from magnificat.samplers.s82_sampler import S82Sampler


class TestDRWDatasetS82S82Sampler(unittest.TestCase):
    """Tests for LSSTCadence class

    """
    def setUp(self):
        self.out_dir = 'drw_data_s82_testing'
        self.obs_dir = 'obs_testing'
        self.sampler_dir = 's82_sampler_testing'
        os.makedirs(self.out_dir, exist_ok=True)

    def test_constructor(self):
        """Test input and output shapes of get_pointings
        """
        agn_params = ['BH_mass', 'redshift', 'M_i', 'u', 'g', 'r', 'i', 'z']
        sampler = S82Sampler(agn_params=agn_params,
                             bp_params=['log_rf_tau', 'log_sf_inf'],
                             bandpasses=list('ugriz'),
                             out_dir=self.sampler_dir,
                             seed=123)
        sampler.process_metadata()
        sampler.idx = [0, 1]
        obs_kwargs = dict(n_pointings_init=3,
                          obs_dir=self.obs_dir,
                          bandpasses=list('ugriz'))
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 is_training=True,
                                 transform_x_func=lambda x: x,
                                 transform_y_func=lambda x: x,
                                 prestored_bandpasses=list('ugriz'),
                                 seed=123,
                                 obs_kwargs=obs_kwargs)

    def test_seeding(self):
        """Test seeding of sightlines
        """
        # Run 0
        agn_params = ['BH_mass', 'redshift', 'M_i', 'u', 'g', 'r', 'i', 'z']
        sampler = S82Sampler(agn_params=agn_params,
                             bp_params=['log_rf_tau', 'log_sf_inf'],
                             bandpasses=list('ugriz'),
                             out_dir=self.sampler_dir,
                             seed=123)
        sampler.process_metadata()
        sampler.idx = [0, 1]
        obs_kwargs = dict(n_pointings_init=3,
                          obs_dir=self.obs_dir,
                          bandpasses=list('ugriz'))
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 is_training=True,
                                 transform_x_func=lambda x: x,
                                 transform_y_func=lambda x: x,
                                 prestored_bandpasses=list('ugriz'),
                                 seed=123,
                                 obs_kwargs=obs_kwargs)
        n_pointings_run0 = drw_dataset.cadence_obj.n_pointings
        # Run 1
        agn_params = ['BH_mass', 'redshift', 'M_i', 'u', 'g', 'r', 'i', 'z']
        sampler = S82Sampler(agn_params=agn_params,
                             bp_params=['log_rf_tau', 'log_sf_inf'],
                             bandpasses=list('ugriz'),
                             out_dir=self.sampler_dir,
                             seed=123)
        sampler.process_metadata()
        sampler.idx = [0, 1]
        obs_kwargs = dict(n_pointings_init=3,
                          obs_dir=self.obs_dir,
                          bandpasses=list('ugriz'))
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 is_training=True,
                                 transform_x_func=lambda x: x,
                                 transform_y_func=lambda x: x,
                                 prestored_bandpasses=list('ugriz'),
                                 seed=123,
                                 obs_kwargs=obs_kwargs)
        n_pointings_run1 = drw_dataset.cadence_obj.n_pointings
        np.testing.assert_equal(n_pointings_run0, n_pointings_run1)

    def test_getitem(self):
        """Test `__getitem__`
        """
        agn_params = ['BH_mass', 'redshift', 'M_i', 'u', 'g', 'r', 'i', 'z']
        sampler = S82Sampler(agn_params=agn_params,
                             bp_params=['log_rf_tau', 'log_sf_inf'],
                             bandpasses=list('ugriz'),
                             out_dir=self.sampler_dir,
                             seed=123)
        sampler.process_metadata()
        sampler.idx = [0, 1]
        obs_kwargs = dict(n_pointings_init=3,
                          obs_dir=self.obs_dir,
                          bandpasses=list('ugriz'))
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 is_training=True,
                                 transform_x_func=lambda x: x,
                                 transform_y_func=lambda x: x,
                                 prestored_bandpasses=list('ugriz'),
                                 seed=123,
                                 obs_kwargs=obs_kwargs)
        data = drw_dataset[0]
        # bandpasses
        np.testing.assert_array_equal(drw_dataset.bandpasses,
                                      list('ugriz'))
        np.testing.assert_array_equal(drw_dataset.bandpasses_int,
                                      [0, 1, 2, 3, 4])
        # x
        assert len(data['x']) == drw_dataset.trimmed_T
        # y
        np.testing.assert_array_equal(data['y'].shape,
                                      [drw_dataset.trimmed_T, 5])
        assert not (data['y'] < -50).any()  # can't be -99
        # param
        assert len(data['params']) == len(drw_dataset.param_names)
        # trimmed_mask
        np.testing.assert_array_equal(data['trimmed_mask'].shape,
                                      [drw_dataset.trimmed_T, 5])

    def test_getitem_singleband(self):
        """Test `__getitem__`
        """
        agn_params = ['BH_mass', 'redshift', 'M_i', 'i']
        sampler = S82Sampler(agn_params=agn_params,
                             bp_params=['log_rf_tau', 'log_sf_inf'],
                             bandpasses=list('i'),
                             out_dir=self.sampler_dir,
                             seed=123)
        sampler.process_metadata()
        sampler.idx = [0, 1]
        obs_kwargs = dict(n_pointings_init=3,
                          obs_dir=self.obs_dir,
                          bandpasses=list('i'))
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 is_training=True,
                                 transform_x_func=lambda x: x,
                                 transform_y_func=lambda x: x,
                                 prestored_bandpasses=list('i'),
                                 seed=123,
                                 obs_kwargs=obs_kwargs)
        data = drw_dataset[0]
        # bandpasses
        np.testing.assert_array_equal(drw_dataset.bandpasses,
                                      list('i'))
        np.testing.assert_array_equal(drw_dataset.bandpasses_int,
                                      [3])
        # x
        assert len(data['x']) == drw_dataset.trimmed_T
        # y
        np.testing.assert_array_equal(data['y'].shape,
                                      [drw_dataset.trimmed_T, 1])
        assert not (data['y'] < -50).any()  # can't be -99
        # param
        assert len(data['params']) == len(drw_dataset.param_names)
        # trimmed_mask
        np.testing.assert_array_equal(data['trimmed_mask'].shape,
                                      [drw_dataset.trimmed_T, 1])

    def tearDown(self):
        shutil.rmtree(self.out_dir)
        shutil.rmtree(self.obs_dir)
        shutil.rmtree(self.sampler_dir)


if __name__ == '__main__':
    unittest.main()
