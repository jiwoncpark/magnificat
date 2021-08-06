import os
import os.path as osp
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from magnificat import drw_utils
from magnificat.cadence import LSSTCadence


class DRWDataset(Dataset):

    bp_to_int = dict(zip(list('ugrizy'), range(6)))
    int_to_bp = dict(zip(range(6), list('ugrizy')))

    def __init__(self,
                 SF_inf_tau_mean_z_sampler,
                 out_dir,
                 num_samples,
                 n_pointings_init,
                 is_training,
                 rescale_x=0.001,
                 shift_x=0.0,
                 err_y=0.01,
                 seed=123):
        """Dataset of DRW light curves

        Parameters
        ----------
        SF_inf_tau_mean_z_sampler : flexible
            Any sampler that has a `sample()` method returning a dict
            of `self.param_names` (see below) and has an attribute
            `bandpasses` which is a list of strings indicating which
            LSST bands, and `idx` which is list of indices if sampler
            is associated with a catalog
        out_dir : str
            Output directory for this dataset
        num_samples : int
            Number of AGNs in this dataset
        n_pointings_init : int
            How many pointings to request
        is_training : bool
            whether this is the training set
        rescale_x : float, optional
            Rescaling factor for the times x, useful if the ML model is
            sensitive to the absolute scale of time
        shift_x : float, optional
            Additive shift for the times x, useful if the ML model is
            sensitive to the absolute scale of time
        err_y : float, optional
            1-sigma scatter in the photometric error, in mag
        seed : int, optional
            Random seed relevant for sampling pointings
        """
        self.SF_inf_tau_mean_z_sampler = SF_inf_tau_mean_z_sampler
        # Figure out which bandpasses are sampled
        bandpasses = self.SF_inf_tau_mean_z_sampler.bandpasses
        self.bandpasses_int = [self.bp_to_int[bp] for bp in bandpasses]
        self.bandpasses_int.sort()
        self.bandpasses = [self.int_to_bp[bp_i] for bp_i in self.bandpasses_int]
        # Compile list of parameters, both bp-dependent and otherwise
        param_names = ['BH_mass', 'M_i']
        param_names += [f'SF_inf_{bp}' for bp in self.bandpasses]
        param_names += [f'mag_{bp}' for bp in self.bandpasses]
        param_names += ['redshift']
        param_names += [f'tau_{bp}' for bp in self.bandpasses]
        self.param_names = param_names
        # Create output directory for this dataset
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.num_samples = num_samples
        self.n_pointings_init = n_pointings_init
        self.is_training = is_training
        self.seed = seed
        self.rescale_x = rescale_x
        self.shift_x = shift_x
        self.delta_x = 1.0  # 1-day interval
        self.max_x = 3650.0  # LSST 10-year
        self.err_y = err_y
        # Preview of untrimmed times
        self.x_grid = np.arange(0, self.max_x, self.delta_x)
        self.x_grid += self.shift_x
        self.x_grid *= self.rescale_x
        self.n_points = len(self.x_grid)
        # For standardizing params
        self.mean_params = None
        self.std_params = None
        self.log_params = None
        self.slice_params = None
        # Load observation strategy
        self.load_obs_strat()
        # Generate and prestore light curves
        self._generate_x_y_params()
        np.savetxt(os.path.join(out_dir, 'cat_idx.txt'),
                   self.SF_inf_tau_mean_z_sampler.idx, fmt='%i')

    def load_obs_strat(self):
        """Load observation strategies

        """
        self.cadence_obj = LSSTCadence(osp.join(self.out_dir,
                                                f'obs_{self.seed}'))
        ra, dec = self.cadence_obj.get_pointings(self.n_pointings_init)
        self.cadence_obj.get_obs_info(ra, dec, skip_ddf=True,
                                      min_visits=500)
        self.cadence_obj.bin_by_day()
        obs_mask = self.cadence_obj.get_observed_mask()  # [trimmed_T,]
        self.trimmed_T = sum(obs_mask)
        self.obs_mask = torch.from_numpy(obs_mask).to(torch.bool)
        self.rng = np.random.default_rng(self.seed)  # for sampling pointings

    def get_t_obs(self):
        """Get full 10-year times in observed frame

        """
        return torch.arange(0, self.max_x, self.delta_x)

    def _generate_x_y_params(self):
        """Generate and store fully observed DRW light curves and params

        """
        # Save times first, since it's the same for all AGNs in dataset
        x = self.get_t_obs()[self.obs_mask]  # [trimmed_T]
        torch.save(x, osp.join(self.out_dir, 'x.pt'))
        for index in tqdm(range(self.num_samples), desc="y, params"):
            if osp.exists(osp.join(self.out_dir, f'drw_{index}.pt')):
                continue
            # Sample params
            params_dict = self.SF_inf_tau_mean_z_sampler.sample()
            z = params_dict['redshift']
            y_concat = torch.ones([self.n_points, 6])*(-99)  # [3650, 6]
            # Render LC for each filter
            for bp in self.bandpasses:
                bp_int = self.bp_to_int[bp]
                tau = params_dict[f'tau_{bp}']
                SF_inf = params_dict[f'SF_inf_{bp}']
                mean_mag = params_dict[f'mag_{bp}']
                y = self._generate_light_curve(index, tau, SF_inf,
                                               mean_mag, z)
                y_concat[:, bp_int] = y
            # Sort params in predetermined ordering
            params = torch.tensor([params_dict[n] for n in self.param_names])  # [n_params]
            # Concat along filter dimension in predetermined filter ordering
            y_concat = y_concat[self.obs_mask, :]  # [trimmed_T, N_filters]
            torch.save((y_concat, params),
                       osp.join(self.out_dir, f'drw_{index}.pt'))

    def _generate_light_curve(self, index, tau, SF_inf, mean, z):
        """Generate a single light curve in a given filter.
        Rendering is done in the rest frame, with the input params
        assumed to be in the rest frame.

        Parameters
        ----------
        index : int
            index within the dataset
        tau : float
            rest-frame timescale
        SF_inf : float
            rest-frame asymptotic amplitude
        mean : float
            mean static magnitude
        z : float
            redshift

        Returns
        -------
        tuple
            single-filter light curve of shape [n_points, 1]
        """
        torch.manual_seed(int(str(self.seed) + str(index)))
        # Shifted rest-frame times
        t_rest = self.get_t_obs()/(1.0 + z)
        # DRW flux
        y = drw_utils.get_drw_torch(t_rest, tau, z, SF_inf,
                                    xmean=mean)  # [T,]
        return y

    def __getitem__(self, index):
        # Load trimmed times
        x = torch.load(osp.join(self.out_dir, 'x.pt'))  # [trimmed_T,]
        # Load fully observed light curve at trimmed times
        y, params = torch.load(osp.join(self.out_dir,
                                        f'drw_{index}.pt'))  # [trimmed_T, 6]
        # Slice relevant bandpasses
        y = y[:, self.bandpasses_int]
        # Rescale x for numerical stability of ML model
        x += self.shift_x
        x *= self.rescale_x
        # Add noise and rescale flux to [-1, 1]
        y += torch.randn_like(y)*self.err_y
        # y = (y - torch.min(y))/(torch.max(y) - torch.min(y))*2.0 - 1.0
        if self.slice_params is not None:
            params = params[self.slice_params]
        if self.log_params is not None:
            params[self.log_params] = torch.log10(params[self.log_params])
        if self.mean_params is not None:
            params -= self.mean_params
            params /= self.std_params
        # Sample observation mask
        if self.is_training:
            # Randomly drawn pointing index
            p = self.rng.integers(low=0, high=self.cadence_obj.n_pointings)
        else:
            # Do not shuffle pointing for validation set
            p = index
        trimmed_mask = self.cadence_obj.get_trimmed_mask(p,
                                                         as_tensor=True)
        trimmed_mask = trimmed_mask[:, self.bandpasses_int]

        data = dict(x=x,
                    y=y,
                    params=params,
                    trimmed_mask=trimmed_mask
                    )
        return data

    def get_normalizing_metadata(self, set_metadata=True):
        loader = DataLoader(self,
                            batch_size=100,
                            shuffle=False,
                            drop_last=False)
        mean_params = 0.0
        var_params = 0.0
        # Compute mean, std
        for i, data in enumerate(loader):
            params = data['params']
            new_mean = params.mean(dim=0)
            new_var = params.var(dim=0, unbiased=False)
            var_params += (new_var - var_params)/(i+1)
            var_params += (i/(i+1)**2.0)*(mean_params - new_mean)**2.0
            mean_params += (new_mean - mean_params)/(i+1)
        std_params = var_params**0.5
        if set_metadata:
            self.mean_params = mean_params
            self.std_params = std_params
        return mean_params, std_params

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import random

    class Sampler:
        def __init__(self, seed, bandpasses):
            random.seed(seed)
            np.random.seed(seed)
            self.bandpasses = bandpasses

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
    sampler = Sampler(train_seed, bandpasses=['i'])

    train_dataset = DRWDataset(sampler, 'train_drw',
                               num_samples=3,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in ['tau_i', 'SF_inf_i', 'M_i']]
    train_dataset.log_params = [True, True, False]
    train_dataset.get_normalizing_metadata()
    print(train_dataset.mean_params, train_dataset.std_params)
    x, y, params = train_dataset[0]
    print(x.shape, y.shape, params.shape)



