import numpy as np
from magnificat.samplers.dc2_sampler import DC2Sampler


class DC2SubsetSampler:
    def __init__(self, seed, bandpasses, std_factor=0.5):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.bandpasses = bandpasses
        self.std_factor = std_factor
        dc2_sampler = DC2Sampler(self.seed, bandpasses)
        cat = dc2_sampler.cat.copy().drop('galaxy_id', axis=1)
        self.mean = cat.mean()
        self.std = cat.std()*self.std_factor
        self.n_params = len(self.mean)

    def sample(self):
        eps = self.rng.standard_normal(self.n_params)
        return eps*self.std + self.mean


if __name__ == '__main__':
    sampler = DC2SubsetSampler(123, list('ugri'))
    s = sampler.sample()
    print(s)
    s = sampler.sample()
    print(s)