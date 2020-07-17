import numpy as np
from scipy.stats import rv_histogram


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    try:
        stacked_posterior = np.log(posterior_density.stack(
            z=['x_position', 'y_position']))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior_density.position[
            np.log(posterior_density).argmax('position')]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def sample_posterior(posterior, place_bin_edges, n_samples=1000):
    hist_dist = rv_histogram((posterior, place_bin_edges))
    return hist_dist.rvs(size=n_samples)
