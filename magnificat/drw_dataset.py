import os
import os.path as osp
import numpy as np
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
                 params_sampler,
                 out_dir,
                 num_samples,
                 is_training,
                 transform_x_func=lambda x: x,
                 transform_y_func=lambda x: x,
                 prestored_bandpasses=list('ugrizy'),
                 seed=123,
                 obs_kwargs={}):
        """Dataset of DRW light curves

        Parameters
        ----------
        params_sampler : flexible
            Any sampler that has a `sample()` method returning a dict
            of `self.param_names` (see below) and has an attribute
            `bandpasses` which is a list of strings indicating which
            LSST bands, and `idx` which is list of indices if sampler
            is associated with a catalog
        out_dir : str
            Output directory for this dataset
        num_samples : int
            Number of AGNs in this dataset
        is_training : bool
            whether this is the training set
        transform_x_func : callable, optional
            Transform function for the times x, useful if the ML model is
            sensitive to the absolute scale of time. Default: identity function
        prestored_bandpasses : TYPE, optional
            Description
        seed : int, optional
            Random seed relevant for generating DRW light curves
        obs_kwargs: dict
            Parameters defining pointings. Includes as keys 'n_pointings_init'
            (number of pointings to request), 'obs_dir' (directory
            containing observation conditions), 'seed' (random seed for
            sampling observation conditions for each light curve, defaults to
            `seed`), 'bandpasses' (list of bandpasses to include in trimming)
        """
        self.params_sampler = params_sampler
        # Figure out which bandpasses are sampled
        bandpasses = self.params_sampler.bandpasses
        self.bandpasses_int = [self.bp_to_int[bp] for bp in bandpasses]
        self.bandpasses_int.sort()
        self.bandpasses = [self.int_to_bp[bp_i] for bp_i in self.bandpasses_int]
        # Compile list of parameters, both bp-dependent and otherwise
        # Determined at data generation time
        param_names = ['BH_mass', 'M_i']
        param_names += [f'log_sf_inf_{bp}' for bp in prestored_bandpasses]
        param_names += [f'{bp}' for bp in prestored_bandpasses]
        param_names += ['redshift']
        param_names += [f'log_rf_tau_{bp}' for bp in prestored_bandpasses]
        self.param_names = param_names
        # Create output directory for this dataset
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.num_samples = num_samples
        self.obs_kwargs = obs_kwargs
        self.is_training = is_training
        self.seed = seed
        self.transform_x_func = transform_x_func
        self.transform_y_func = transform_y_func
        self.delta_x = 1.0  # 1-day interval
        self.max_x = 3650.0  # LSST 10-year
        # Preview of untrimmed times
        self.x_grid = np.arange(0, self.max_x, self.delta_x)
        self.x_grid = self.transform_x_func(self.x_grid)
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
                   self.params_sampler.idx, fmt='%i')
        self._fully_obs = False  # init property
        self._add_noise = True  # init property

    def get_sliced_params(self):
        return np.array(self.param_names)[np.array(self.slice_params)]

    def load_obs_strat(self):
        """Load observation strategies

        """
        self.cadence_obj = LSSTCadence(self.obs_kwargs['obs_dir'])
        ra, dec = self.cadence_obj.get_pointings(self.obs_kwargs['n_pointings_init'])
        self.cadence_obj.get_obs_info(ra, dec, skip_ddf=True,
                                      min_visits=50)
        self.cadence_obj.bin_by_day(bandpasses=self.obs_kwargs['bandpasses'])
        obs_mask = self.cadence_obj.get_observed_mask()  # [3650,]
        self.trimmed_T = sum(obs_mask)
        self.obs_mask = torch.from_numpy(obs_mask).to(torch.bool)
        self.rng = np.random.default_rng(self.obs_kwargs.get('seed', self.seed))  # for sampling pointings

    def get_t_obs(self):
        """Get full 10-year times in observed frame

        """
        return torch.arange(0, self.max_x, self.delta_x)

    def _generate_x_y_params(self):
        """Generate and store fully observed DRW light curves and params

        """
        # Save times first, since it's the same for all AGNs in dataset
        x = self.get_t_obs()  # [3651]
        torch.save(self.obs_mask, osp.join(self.out_dir, 'obs_mask.pt'))
        torch.save(x, osp.join(self.out_dir, 'x.pt'))
        for index in tqdm(range(self.num_samples), desc="y, params"):
            if osp.exists(osp.join(self.out_dir, f'drw_{index}.pt')):
                continue
            # Sample params
            params_dict = self.params_sampler.sample()
            z = params_dict['redshift']
            y_concat = torch.ones([self.n_points, 6])*(-99)  # [3650, 6]
            # Render LC for each filter
            for bp in self.bandpasses:
                bp_int = self.bp_to_int[bp]
                log_rf_tau = params_dict[f'log_rf_tau_{bp}']
                log_sf_inf = params_dict[f'log_sf_inf_{bp}']
                mean_mag = params_dict[f'{bp}']
                y = self._generate_light_curve(index, log_rf_tau, log_sf_inf,
                                               mean_mag, z)  # [3650,]
                y_concat[:, bp_int] = y
            # Sort params in predetermined ordering
            params = torch.tensor([params_dict[n] for n in self.param_names])  # [n_params]
            # Concat along filter dimension in predetermined filter ordering
            # y_concat = y_concat[self.obs_mask, :]  # [trimmed_T, N_filters]
            # Save y_concat without obs_mask
            # y_concat ~ [3651, N_filters]
            torch.save((y_concat, params),
                       osp.join(self.out_dir, f'drw_{index}.pt'))

    def _generate_light_curve(self, index, log_rf_tau, log_sf_inf, mean, z):
        """Generate a single light curve in a given filter.
        Rendering is done in the rest frame, with the input params
        assumed to be in the rest frame.

        Parameters
        ----------
        index : int
            index within the dataset
        log_rf_tau : float
            log10 of rest-frame timescale in days
        log_sf_inf : float
            log10 of rest-frame asymptotic amplitude in mag
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
        tau = 10**log_rf_tau
        sf_inf = 10**log_sf_inf
        y = drw_utils.get_drw_torch(t_rest, tau, z, sf_inf,
                                    xmean=mean)  # [T,]
        return y

    @property
    def fully_obs(self):
        return self._fully_obs

    @fully_obs.setter
    def fully_obs(self, val):
        self._fully_obs = val

    @property
    def add_noise(self):
        return self._add_noise

    @add_noise.setter
    def add_noise(self, val):
        self._add_noise = val

    def __getitem__(self, index):
        # Load fully observed light curve at fully obs times
        y, params = torch.load(osp.join(self.out_dir,
                                        f'drw_{index}.pt'))  # [T=3650, 6]
        if self.fully_obs:
            obs_mask = slice(None)
        else:
            obs_mask = self.obs_mask
        # Trim the times
        x = torch.load(osp.join(self.out_dir, 'x.pt'))[obs_mask]  # [trimmed_T,]
        y = y[obs_mask, :]
        # Slice relevant bandpasses
        y = y[:, self.bandpasses_int]
        # Rescale x for numerical stability of ML model
        x = self.transform_x_func(x)
        # Add noise and rescale flux to [-1, 1]
        y = self.transform_y_func(y)
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
            p = 0
        trimmed_mask = self.cadence_obj.get_trimmed_mask(p,
                                                         as_tensor=True)
        # trimmed_mask = trimmed_mask[:, self.bandpasses_int]

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
        print("Computing normalizing metadata...")
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
                log_sf_inf = np.maximum(np.random.randn()*0.05 + 0.2, 0.2)
                # log_sf_inf = 10**(np.random.randn(N)*(0.25) + -0.8)
                # log_sf_inf = np.ones(N)*0.15
                # tau = 10.0**np.maximum(np.random.randn(N)*0.5 + 2.0, 0.1)
                tau = np.maximum(np.random.randn()*50.0 + 200.0, 10.0)
                # mag = np.maximum(np.random.randn(N) + 19.0, 17.5)
                mag = 0.0
                # z = np.maximum(np.random.randn(N) + 2.0, 0.5)
                sample_dict[f'log_rf_tau_{bp}'] = tau
                sample_dict[f'log_sf_inf_{bp}'] = log_sf_inf
                sample_dict[f'{bp}'] = mag
            sample_dict['redshift'] = 2.0
            sample_dict['M_i'] = -16.0
            sample_dict['BH_mass'] = 10.0
            return sample_dict

    train_seed = 123
    sampler = Sampler(train_seed, bandpasses=['i'])

    train_dataset = DRWDataset(sampler, 'train_drw_s82',
                               num_samples=3,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in ['log_rf_taui', 'log_sf_inf_i', 'M_i']]
    train_dataset.log_params = [True, True, False]
    train_dataset.get_normalizing_metadata()
    print(train_dataset.mean_params, train_dataset.std_params)
    x, y, params = train_dataset[0]
    print(x.shape, y.shape, params.shape)



