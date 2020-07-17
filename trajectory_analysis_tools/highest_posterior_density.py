import numpy as np


def highest_posterior_density(posterior_density, coverage=0.95):
    """
    Same as credible interval
    https://stats.stackexchange.com/questions/240749/how-to-find-95-credible-interval

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_position_bins) or
        shape (n_time, n_x_bins, n_y_bins)
    coverage : float, optional

    Returns
    -------
    threshold : ndarray, shape (n_time,)

    """
    try:
        posterior_density = posterior_density.stack(
            z=["x_position", "y_position"]
        ).values
    except KeyError:
        posterior_density = posterior_density.values
    const = np.sum(posterior_density, axis=1, keepdims=True)
    sorted_norm_posterior = np.sort(posterior_density, axis=1)[:, ::-1] / const
    posterior_less_than_coverage = np.cumsum(
        sorted_norm_posterior, axis=1) >= coverage
    crit_ind = np.argmax(posterior_less_than_coverage, axis=1)
    # Handle case when there are no points in the posterior less than coverage
    crit_ind[posterior_less_than_coverage.sum(axis=1) == 0] = (
        posterior_density.shape[1] - 1
    )

    n_time = posterior_density.shape[0]
    threshold = sorted_norm_posterior[(
        np.arange(n_time), crit_ind)] * const.squeeze()
    return threshold
