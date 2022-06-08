from functools import lru_cache
from pathlib import Path

from astropy import log
from ...utils.cosmology import luminosity_distance
import numpy as np
from scipy import optimize
from scipy.special import erf

log.setLevel("ERROR")


@lru_cache(maxsize=None)
def compute_corr_dz(model_z, obs):
    """The mass-dependent physical properties are computed assuming the
    redshift of the model. However because we round the observed redshifts to
    two decimals, there can be a difference of 0.005 in redshift between the
    models and the actual observation. This causes two issues. First there is a
    difference in the luminosity distance. At low redshift, this can cause a
    discrepancy in the mass-dependent physical properties: ~0.35 dex at z=0.010
    vs 0.015 for instance. In addition, the 1+z dimming will be different.
    We compute here the correction factor for these two effects.

    Parameters
    ----------
    model_z: float
        Redshift of the model.
    obs: instance of the Observation class
        Object containing the distance and redshift of an object

    """
    return (
        (obs.distance / luminosity_distance(model_z)) ** 2.0
        * (1.0 + model_z)
        / (1.0 + obs.redshift)
    )


def _compute_scaling(models, obs, corr_dz, wz):
    """Compute the scaling factor to be applied to the model fluxes to best fit
    the observations. Note that we look over the bands to avoid the creation of
    an array of the same size as the model_fluxes array. Because we loop on the
    bands and not on the models, the impact on the performance should be small.

    Parameters
    ----------
    models: ModelsManagers class instance
        Contains the models (fluxes, intensive, and extensive properties).
    obs: Observation class instance
        Contains the fluxes, intensive properties, extensive properties and
        their errors, for a sigle observation.
    corr_dz: float
        Correction factor to scale the extensive properties to the right
        distance
    wz: slice
        Selection of the models at the redshift of the observation or all the
        redshifts in photometric-redshift mode.

    Returns
    -------
    scaling: array
        Scaling factors minimising the χ²
    """

    _ = list(models.flux.keys())[0]
    num = np.zeros_like(models.flux[_][wz])
    denom = np.zeros_like(models.flux[_][wz])

    for band, flux in obs.flux.items():
        # Multiplications are faster than divisions, so we directly use the
        # inverse error
        inv_err2 = 1.0 / obs.flux_err[band] ** 2.0
        model = models.flux[band][wz]
        num += model * (flux * inv_err2)
        denom += model ** 2.0 * inv_err2

    for name, prop in obs.extprop.items():
        # Multiplications are faster than divisions, so we directly use the
        # inverse error
        inv_err2 = 1.0 / obs.extprop_err[name] ** 2.0
        model = models.extprop[name][wz]
        num += model * (prop * inv_err2 * corr_dz)
        denom += model ** 2.0 * (inv_err2 * corr_dz ** 2.0)

    return num / denom

def weighted_param(param, weights):
    """Compute the weighted mean and standard deviation of an array of data.
    Note that here we assume that the sum of the weights is normalised to 1.
    This simplifies and accelerates the computation.

    Parameters
    ----------
    param: array
        Values of the parameters for the entire grid of models
    weights: array
        Weights by which to weigh the parameter values

    Returns
    -------
    mean: float
        Weighted mean of the parameter values
    std: float
        Weighted standard deviation of the parameter values

    """

    mean = np.einsum("i, i", param, weights)
    delta = param - mean
    std = np.sqrt(np.einsum("i, i, i", weights, delta, delta))

    return (mean, std)
