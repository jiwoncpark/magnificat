import numpy as np
from sklearn.utils import check_random_state
import torch


def get_drw(t_rest, tau, z, SF_inf, xmean=0, random_state=None):
    """Generate a damped random walk light curve
    This uses a damped random walk model to generate a light curve similar
    to that of a QSO [1]_.
    Taken with minor modification from
    https://github.com/astroML/astroML/blob/main/astroML/time_series/generate.py

    Parameters
    ----------
    t_rest : array_like
        rest-frame time.  Should be in increasing order
    tau : float
        relaxation time
    z : float
        redshift
    xmean : float (optional)
        mean value of random walk; default=0
    SF_inf : float (optional
        Structure function at infinity; default=0.3
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator

    Returns
    -------
    x : ndarray
        the sampled values corresponding to times t_rest

    Notes
    -----
    The differential equation is (with t = time/tau):
        dX = -X(t) * dt + sigma * sqrt(tau) * e(t) * sqrt(dt) + b * tau * dt
    where e(t) is white noise with zero mean and unit variance, and
        Xmean = b * tau
        SFinf = sigma * sqrt(tau / 2)
    so
        dX(t) = -X(t) * dt + sqrt(2) * SFint * e(t) * sqrt(dt) + Xmean * dt

    References
    ----------
    .. [1] Kelly, B., Bechtold, J. & Siemiginowska, A. (2009)
           Are the Variations in Quasar Optical Flux Driven by Thermal
           Fluctuations? ApJ 698:895 (2009)

    """
    #  Xmean = b * tau
    #  SFinf = sigma * sqrt(tau / 2)
    rng = check_random_state(random_state)
    N = len(t_rest)

    tau_obs = tau*(z+1)
    t_obs = t_rest*(z+1)
    t = t_obs / tau_obs  # tau-adjusted times

    x = np.zeros(N)
    E = rng.normal(0, 1, N)
    x[0] = E[0]*SF_inf + xmean

    for i in range(1, N):
        dt = t[i] - t[i - 1]
        x[i] = (x[i - 1]
                - dt * (x[i - 1] - xmean)
                + np.sqrt(2 * dt) * SF_inf * E[i])
    return x


def get_drw_torch(t_rest, tau, z, SF_inf, xmean=0):
    """Generate a damped random walk light curve
    This uses a damped random walk model to generate a light curve similar
    to that of a QSO [1]_.
    Taken with minor modification from
    https://github.com/astroML/astroML/blob/main/astroML/time_series/generate.py

    Parameters
    ----------
    t_rest : array_like
        rest-frame time.  Should be in increasing order
    tau : float
        relaxation time
    z : float
        redshift
    xmean : float (optional)
        mean value of random walk; default=0
    SF_inf : float (optional
        Structure function at infinity; default=0.3
    random_state : int
        random seed or random number generator

    Returns
    -------
    x : ndarray
        the sampled values corresponding to times t_rest

    Notes
    -----
    The differential equation is (with t = time/tau):
        dX = -X(t) * dt + sigma * sqrt(tau) * e(t) * sqrt(dt) + b * tau * dt
    where e(t) is white noise with zero mean and unit variance, and
        Xmean = b * tau
        SFinf = sigma * sqrt(tau / 2)
    so
        dX(t) = -X(t) * dt + sqrt(2) * SFint * e(t) * sqrt(dt) + Xmean * dt

    References
    ----------
    .. [1] Kelly, B., Bechtold, J. & Siemiginowska, A. (2009)
           Are the Variations in Quasar Optical Flux Driven by Thermal
           Fluctuations? ApJ 698:895 (2009)

    """
    #  Xmean = b * tau
    #  SFinf = sigma * sqrt(tau / 2)
    N = len(t_rest)

    tau_obs = tau*(z+1)
    t_obs = t_rest*(z+1)
    t = t_obs / tau_obs  # tau-adjusted times

    x = torch.zeros(N)
    E = torch.randn(N)
    x[0] = E[0]*SF_inf + xmean

    for i in range(1, N):
        dt = t[i] - t[i - 1]
        x[i] = (x[i - 1]
                - dt * (x[i - 1] - xmean)
                + np.sqrt(2 * dt) * SF_inf * E[i])
    return x
