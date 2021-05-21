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
    int_to_bp = dict(zip(range(6), list('ugrizy')))

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
        bandpasses = self.SF_inf_tau_mean_z_sampler.bandpasses
        self.bandpasses_int = [self.bp_to_int[bp] for bp in bandpasses]
        self.bandpasses_int.sort()
        self.bandpasses = [self.int_to_bp[bp_i] for bp_i in self.bandpasses_int]
        param_names = ['BH_mass', 'M_i']
        param_names += [f'SF_inf_{bp}' for bp in self.bandpasses]
        param_names += [f'mag_{bp}' for bp in self.bandpasses]
        param_names += ['redshift']
        param_names += [f'tau_{bp}' for bp in self.bandpasses]
        self.param_names = param_names
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
        for index in tqdm(range(self.num_samples), desc="light curves"):
            if osp.exists(osp.join(self.out_dir, f'drw_{index}.pt')):
                continue
            params_dict = self.SF_inf_tau_mean_z_sampler.sample()
            z = params_dict['redshift']
            x_to_concat = []
            y_to_concat = []
            for bp_int, bp in enumerate(self.bandpasses):
                tau = params_dict[f'tau_{bp}']
                SF_inf = params_dict[f'SF_inf_{bp}']
                mean_mag = params_dict[f'mag_{bp}']
                x, y = self._generate_light_curves(index, tau, SF_inf, mean_mag, z)
                x_to_concat.append(x)
                y_to_concat.append(y)
            # Sort params in predetermined ordering
            params = torch.tensor([params_dict[n] for n in self.param_names])  # [n_params]
            # Concat along filter dimension in predetermined filter ordering
            x = torch.cat(x_to_concat, dim=1)  # [N_t, N_filters]
            y = torch.cat(y_to_concat, dim=1)  # [N_t, N_filters]
            torch.save((x, y, params), osp.join(self.out_dir, f'drw_{index}.pt'))

    def _generate_light_curves(self, index, tau, SF_inf, mean, z):
        # rng = default_rng(int(str(self.seed) + str(index)))
        torch.manual_seed(int(str(self.seed) + str(index)))
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
        return x, y

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
        if self.slice_params is not None:
            params = params[self.slice_params]
        if self.mean_params is not None:
            params -= self.mean_params[self.slice_params]
            params /= self.std_params[self.slice_params]
        return x, y, params

    def get_normalizing_metadata(self):
        loader = DataLoader(self,
                            batch_size=100,
                            shuffle=False,
                            drop_last=False)
        mean_params = torch.zeros([len(self.param_names)])
        std_params = torch.zeros([len(self.param_names)])
        for i, (_, _, params) in enumerate(loader):
            print(mean_params.shape, params.shape)
            mean_params += (torch.mean(params, dim=0) - mean_params)/(i+1)
            std_params += (torch.std(params, dim=0) - std_params)/(i+1)
        self.mean_params = mean_params
        self.std_params = std_params

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import random

    class Sampler:
        def __init__(self, seed, bandpasses):
            random.seed(seed)
            np.random.seed(seed)
            self.bandpasses = list('ugrizy')

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

    train_seed = 123
    sampler = Sampler(train_seed, bandpasses=list('ugrizy'))

    train_dataset = DRWDataset(sampler, 'train_drw',
                               num_samples=3,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in ['tau_i', 'SF_inf_i']]
    print(train_dataset.slice_params)
    print(train_dataset.mean_params, train_dataset.std_params)
    x, y, params = train_dataset[0]
    print(x.shape, y.shape, params.shape)



