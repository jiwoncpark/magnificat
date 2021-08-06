import unittest
import shutil
import numpy as np
from magnificat.drw_dataset import DRWDataset


class Sampler:
    def __init__(self, seed, bandpasses):
        np.random.seed(seed)
        self.bandpasses = bandpasses
        self.idx = []

    def sample(self):
        sample_dict = dict()
        for bp in self.bandpasses:
            SF_inf = np.maximum(np.random.randn()*0.05 + 0.2, 0.2)
            # SF_inf = 10**(np.random.randn(N)*(0.25) + -0.8)
            # SF_inf = np.ones(N)*0.15
            # tau = 10.0**np.maximum(np.random.randn(N)*0.5 + 2.0, 0.1)
            tau = np.maximum(np.random.randn()*50.0 + 200.0, 10.0)
            # mag = np.maximum(np.random.randn(N) + 19.0, 17.5)
            mag = 0.0
            # z = np.maximum(np.random.randn(N) + 2.0, 0.5)
            sample_dict[f'tau_{bp}'] = tau
            sample_dict[f'SF_inf_{bp}'] = SF_inf
            sample_dict[f'mag_{bp}'] = mag
        sample_dict['redshift'] = 2.0
        sample_dict['M_i'] = -16.0
        sample_dict['BH_mass'] = 10.0
        return sample_dict


class TestDRWDataset(unittest.TestCase):
    """Tests for LSSTCadence class

    """
    def setUp(self):
        self.out_dir = 'drw_data_testing'

    def test_constructor(self):
        """Test input and output shapes of get_pointings
        """
        bandpasses = ['g', 'r', 'i']
        sampler = Sampler(123, bandpasses)
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=2,
                                 n_pointings_init=10,
                                 is_training=True)

    def test_getitem(self):
        """Test `__getitem__`
        """
        bandpasses = ['g', 'i', 'r']
        sampler = Sampler(123, bandpasses)
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=3,
                                 n_pointings_init=10,
                                 is_training=True)
        data = drw_dataset[0]
        # bandpasses
        np.testing.assert_array_equal(drw_dataset.bandpasses,
                                      ['g', 'r', 'i'])
        np.testing.assert_array_equal(drw_dataset.bandpasses_int,
                                      [1, 2, 3])
        # x
        assert len(data['x']) == drw_dataset.trimmed_T
        # y
        np.testing.assert_array_equal(data['y'].shape,
                                      [drw_dataset.trimmed_T, len(bandpasses)])
        assert not (data['y'] < -50).any()  # can't be -99
        # param
        assert len(data['params']) == len(drw_dataset.param_names)
        # trimmed_mask
        np.testing.assert_array_equal(data['trimmed_mask'].shape,
                                      [drw_dataset.trimmed_T, len(bandpasses)])

    def test_get_normalizing_metadata(self):
        """Test `get_normalizing_metadata`
        """
        bandpasses = ['g', 'r', 'i']
        sampler = Sampler(123, bandpasses)
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=3,
                                 n_pointings_init=10,
                                 is_training=False,
                                 err_y=0.0)
        param0 = drw_dataset[0]['params']
        param1 = drw_dataset[1]['params']
        param2 = drw_dataset[2]['params']
        params = np.stack([param0, param1, param2], axis=0)
        drw_dataset.get_normalizing_metadata(set_metadata=True)
        assert len(drw_dataset.mean_params) == len(drw_dataset.param_names)
        assert len(drw_dataset.std_params) == len(drw_dataset.param_names)
        np.testing.assert_array_almost_equal(drw_dataset.mean_params,
                                             np.mean(params, axis=0))
        np.testing.assert_array_almost_equal(drw_dataset.std_params,
                                             np.std(params, axis=0))

    def test_transform(self):
        """Test log and slice transforms
        """
        bandpasses = ['g', 'r', 'i']
        sampler = Sampler(123, bandpasses)
        drw_dataset = DRWDataset(sampler,
                                 self.out_dir,
                                 num_samples=3,
                                 n_pointings_init=10,
                                 is_training=False,
                                 err_y=0.0)
        # Without log
        params = np.stack([drw_dataset[i]['params'] for i in range(3)],
                          axis=0)
        # With log
        log_param_names = ['SF_inf_i', 'tau_g']
        to_log = [drw_dataset.param_names.index(p) for p in log_param_names]
        drw_dataset.log_params = to_log
        params_log = np.stack([drw_dataset[i]['params'] for i in range(3)],
                              axis=0)
        params[:, to_log] = np.log10(params[:, to_log])
        np.testing.assert_array_almost_equal(params, params_log)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
