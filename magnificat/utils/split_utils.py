import numpy as np
from scipy import stats


def subsample_dist(values, sub_dist, n_samples, seed,
                   return_kde_fit=False):
    """Subsample from a given array following a specified distribution

    Parameters
    ----------
    values : np.ndarray
        Values to subsample from
    sub_dist : scipy.stats._continuous_distns object
        Subsampling distribution. Must have a `pdf` method for evaluating PDF
    n_samples : int
        Number of samples
    seed : int
    return_kde_fit : bool
        Whether to return the KDE fit on `values`

    Returns
    -------
    np.ndarray
        Indices of subsamples within `values`

    """
    kde = stats.gaussian_kde(values, bw_method='scott')
    rng = np.random.default_rng(seed)
    sub_p = sub_dist.pdf(values)/kde.pdf(values)
    sub_p = sub_p/sub_p.sum()
    samples_i = rng.choice(np.arange(len(values)),
                           n_samples,
                           p=sub_p,
                           replace=False)
    if return_kde_fit:
        return samples_i, kde
    else:
        return samples_i


def random_split(indices, frac_val, seed):
    """Randomly split a list

    Parameters
    ----------
    indices : np.ndarray
        List of indices
    frac_val : float
        Fraction of validation points
    seed : int

    Returns
    -------
    tuple
        Training indices, validation indices

    """
    rng = np.random.default_rng(seed)
    val_i = rng.choice(indices,
                       size=int(len(indices)*frac_val))
    train_i = list(set(indices) - set(val_i))
    return train_i, val_i
