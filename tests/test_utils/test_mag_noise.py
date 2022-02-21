import unittest
import numpy as np
import torch
from magnificat.utils.mag_noise import MagNoise, MagNoiseTorch


class TestMagNoise(unittest.TestCase):
    """Tests for `drw_utils.py`

    """
    @classmethod
    def setUpClass(cls):
        cls.seed = 123
        cls.mags_np = np.random.randn(100, 5)*1 + 20  # random simulated mags
        cls.mags_torch = torch.tensor(cls.mags_np)  # random simulated mags

    def test_mag_noise_np_ugriz(self):
        """Check if noised mags are within reasonable range"""
        mag_noise_np = MagNoise(mag_idx=[0, 1, 2, 3, 4],
                                which_bands=list('ugriz'),
                                override_kwargs=None,
                                depth=10,
                                airmass=1.15304)
        noised = mag_noise_np(self.mags_np)
        np.testing.assert_array_less(noised, np.ones_like(noised)*25)
        np.testing.assert_array_less(np.ones_like(noised)*15, noised)

    def test_mag_noise_torch_ugriz(self):
        """Check if noised mags are within reasonable range"""
        mag_noise_torch = MagNoiseTorch(mag_idx=[0, 1, 2, 3, 4],
                                        which_bands=list('ugriz'),
                                        override_kwargs=None,
                                        depth=10,
                                        airmass=1.15304)
        noised = mag_noise_torch(self.mags_torch)
        np.testing.assert_array_less(noised, np.ones_like(noised)*25)
        np.testing.assert_array_less(np.ones_like(noised)*15, noised)


if __name__ == '__main__':
    unittest.main()
