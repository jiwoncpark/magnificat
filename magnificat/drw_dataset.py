import os
import os.path as osp
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from magnificat import drw_utils


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
    bp_to_int = dict(zip(list('ugrizy'), range(6)))

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
        # Figure out how many bandpasses are sampled
        sample = self.SF_inf_tau_mean_z_sampler.sample(1)
        self.bandpasses = list(sample.keys())
        self.bandpasses_int = [self.bp_to_int[bp] for bp in self.bandpasses]
        self.bandpasses_int.sort()
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
        self.mean_params = None
        self.std_params = None
        self.slice_params = None
        self._generate_light_curves_multi_filter()
        self.get_normalizing_metadata()

    def _generate_light_curves_multi_filter(self):
        """Generate and store fully observed DRW light curves

        """
        for bp in self.bandpasses:
            bp_int = self.bp_to_int[bp]
            for index in tqdm(range(self.num_samples), desc=bp):
                if osp.exists(osp.join(self.out_dir, f'drw_{bp_int}_{index}.pt')):
                    continue
                else:
                    self._generate_light_curves(index, bp)

    def _generate_light_curves(self, index, bandpass):
        # rng = default_rng(int(str(self.seed) + str(index)))
        bp_int = self.bp_to_int[bandpass]
        torch.manual_seed(int(str(self.seed) + str(index)))
        params = self.SF_inf_tau_mean_z_sampler.sample(1)[bandpass][0]
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
        params = torch.from_numpy(params).unsqueeze(1)  # [n_params, 1]
        torch.save((x, y, params),
                   osp.join(self.out_dir, f'drw_{bp_int}_{index}.pt'))

    def __getitem__(self, index):
        # Load fully observed light curve
        x_to_concat = []
        y_to_concat = []
        params_to_concat = []
        for bp in self.bandpasses_int:
            x, y, params = torch.load(osp.join(self.out_dir, f'drw_{bp}_{index}.pt'))
            x_to_concat.append(x)
            y_to_concat.append(y)
            params_to_concat.append(params)
        # Concat along filter dimension
        x = torch.cat(x_to_concat, dim=1)  # [N_t, 6]
        y = torch.cat(y_to_concat, dim=1)  # [N_t, 6]
        params = torch.cat(params_to_concat, dim=1)  # [4, 6]
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
        if self.slice_params is not None:
            params = params[self.slice_params, :]
        if self.mean_params is not None:
            params -= self.mean_params[self.slice_params, :]
            params /= self.std_params[self.slice_params, :]
        return x, y, params

    def get_normalizing_metadata(self):
        loader = DataLoader(self,
                            batch_size=100,
                            shuffle=False,
                            drop_last=False)
        mean_params = torch.zeros([4, len(self.bandpasses)])
        std_params = torch.zeros([4, len(self.bandpasses)])
        for i, (_, _, params) in enumerate(loader):
            mean_params += (torch.mean(params, dim=0) - mean_params)/(i+1)
            std_params += (torch.std(params, dim=0) - std_params)/(i+1)
        self.mean_params = mean_params
        self.std_params = std_params

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import random

    class Sampler:
        def __init__(self, seed):
            random.seed(seed)
            np.random.seed(seed)

        def sample(self, N):
            sample = dict()
            for bp in list('ugrizy'):
                SF_inf = np.maximum(np.random.randn(N)*0.05 + 0.2, 0.2)
                # SF_inf = 10**(np.random.randn(N)*(0.25) + -0.8)
                # SF_inf = np.ones(N)*0.15
                # tau = 10.0**np.maximum(np.random.randn(N)*0.5 + 2.0, 0.1)
                tau = np.maximum(np.random.randn(N)*50.0 + 200.0, 10.0)
                # mag = np.maximum(np.random.randn(N) + 19.0, 17.5)
                mag = np.zeros(N)
                # z = np.maximum(np.random.randn(N) + 2.0, 0.5)
                z = np.ones(N)*2.0
                sample[bp] = np.stack([SF_inf, tau, mag, z], axis=-1)  # [N, 4]
            return sample

    train_seed = 123
    sampler = Sampler(train_seed)

    train_dataset = DRWDataset(sampler, 'train_drw',
                               num_samples=10,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01)
    train_dataset.slice_params = [0, 1]
    print(train_dataset.mean_params, train_dataset.std_params)
    x, y, params = train_dataset[0]
    print(x.shape, y.shape, params.shape)



