import os
import os.path as osp
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from magnificat import drw_utils, cadence


class DRWDataset(Dataset):
    """
    Dataset of DRW light curves

    Parameters
    ----------
    num_samples : int
        Number of light curves (size of dataset)
    err_y : float
        Gaussian sigma of flux uncertainty in mag
    max_x : float
        Maximum observer-frame time in days
    delta_x : float
        Resolution of DRW rendering in observer-frame days

    """
    def __init__(self,
                 SF_inf_tau_mean_z_sampler,
                 out_dir,
                 num_samples,
                 rescale_x=0.001,
                 shift_x=0.0,
                 delta_x=1.0,
                 max_x=3650.0,
                 err_y=0.01,
                 seed=123):
        self.SF_inf_tau_mean_z_sampler = SF_inf_tau_mean_z_sampler
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.num_samples = num_samples
        self.seed = seed
        self.rescale_x = rescale_x
        self.shift_x = shift_x
        self.delta_x = delta_x
        self.max_x = max_x
        self.err_y = err_y
        self.n_points = len(torch.arange(0, self.max_x+self.delta_x, self.delta_x))
        self.x = np.arange(0, self.max_x+self.delta_x, self.delta_x)
        self.x += self.shift_x
        self.x *= self.rescale_x
        # Fixed attributes
        self.min_max = (-1, 1)
        self.x_dim = 1
        self.y_dim = 1
        # For standardizing params
        self._generate_light_curves()

    def _generate_light_curves(self):
        """Generate and store fully observed DRW light curves

        """
        for index in tqdm(range(self.num_samples)):
            if osp.exists(osp.join(self.out_dir, f'drw_{index}.pt')):
                continue
            else:
                # rng = default_rng(int(str(self.seed) + str(index)))
                torch.manual_seed(int(str(self.seed) + str(index)))
                params = self.SF_inf_tau_mean_z_sampler.sample(1)[0]
                SF_inf, tau, mean, z = params
                # Shifted rest-frame times
                t_obs = torch.arange(0, self.max_x+self.delta_x, self.delta_x)
                t_rest = t_obs/(1.0 + z)
                # DRW flux
                y = drw_utils.get_drw_torch(t_rest, tau, z, SF_inf,
                                            xmean=mean)  # [n_points,]
                x = t_obs.unsqueeze(1)  # [n_points, 1]
                y = y.unsqueeze(1)  # [n_points, 1]
                # x = x[..., np.newaxis]  # [n_points, 1]
                # y = y[..., np.newaxis]  # [n_points, 1]
                params = torch.from_numpy(params)
                torch.save((x, y, params),
                           osp.join(self.out_dir, f'drw_{index}.pt'))

    def __getitem__(self, index):
        # Load fully observed light curve
        x, y, params = torch.load(osp.join(self.out_dir, f'drw_{index}.pt'))
        # x = torch.from_numpy(x).unsqueeze(1)
        # y = torch.from_numpy(y).unsqueeze(1)
        # y = np.interp(x, t_obs_full, y_full)
        # Rescale x
        x += self.shift_x
        x *= self.rescale_x
        # Add noise and rescale flux to [-1, 1]
        y += torch.randn_like(y)*0.01
        # y = (y - torch.min(y))/(torch.max(y) - torch.min(y))*2.0 - 1.0
        # Standardize params
        params = params[self.slice_params]
        params = (params - self.mean_params)/self.std_params
        return x, y, params

    def __len__(self):
        return self.num_samples
