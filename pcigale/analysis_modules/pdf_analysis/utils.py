from functools import lru_cache
from pathlib import Path

from astropy import log
from ...utils.cosmology import luminosity_distance
import numpy as np
from scipy import optimize
from scipy.special import erf

log.setLevel('ERROR')


def save_chi2(obs, variable, models, chi2, values):
    """Save the chi² and the associated physocal properties

    """
    fname = Path('out') / f"{obs.id}_{variable.replace('/', '_')}_chi2-block-" \
        f"{models.iblock}.npy"
    data = np.memmap(fname, dtype=np.float64, mode='w+',
                     shape=(2, chi2.size))
    data[0, :] = chi2
    data[1, :] = values


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
    return (obs.distance / luminosity_distance(model_z)) ** 2. * \
           (1. + model_z) / (1. + obs.redshift)


def dchi2_over_ds2(s, obsdata, obsdata_err, obslim, obslim_err, moddata,
                   modlim):
    """Function used to estimate the normalization factor in the SED fitting
    process when upper limits are included in the dataset to fit (from Eq. A11
    in Sawicki M. 2012, PASA, 124, 1008).

    Parameters
    ----------
    s: Float
        Contains value onto which we perform minimization = normalization
        factor
    obsdata: array
        Fluxes and extensive properties
    obsdata_err: array
        Errors on the fluxes and extensive properties
    obslim: array
        Fluxes and extensive properties upper limits
    obslim_err: array
        Errors on the fluxes and extensive properties upper limits
    moddata: array
        Model fluxes and extensive properties
    modlim: array
        Model fluxes and extensive properties for upper limits

   Returns
    -------
    func: Float
        Eq. A11 in Sawicki M. 2012, PASA, 124, 1008).

    """
    # We enter into this function if lim_flag = True.

    # The mask "data" selects the filter(s) for which measured fluxes are given
    # i.e., when obs_fluxes is >=0. and obs_errors >=0.
    # The mask "lim" selects the filter(s) for which upper limits are given
    # i.e., when obs_errors < 0
    sqrt2 = np.sqrt(2)
    dchi2_over_ds_data = np.sum((obsdata - s * moddata) *
                                moddata / obsdata_err**2.)

    dchi2_over_ds_lim = np.sqrt(2. / np.pi) * np.sum(
        modlim * np.exp(
            -np.square((obslim - s * modlim) / (sqrt2 * obslim_err))
        ) / (
            obslim_err * (1. + erf((obslim - s * modlim) / (sqrt2 * obslim_err)))
        ))
    func = dchi2_over_ds_data - dchi2_over_ds_lim

    return func


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
        inv_err2 = 1. / obs.flux_err[band] ** 2.
        model = models.flux[band][wz]
        num += model * (flux * inv_err2)
        denom += model ** 2. * inv_err2

    for name, prop in obs.extprop.items():
        # Multiplications are faster than divisions, so we directly use the
        # inverse error
        inv_err2 = 1. / obs.extprop_err[name] ** 2.
        model = models.extprop[name][wz]
        num += model * (prop * inv_err2 * corr_dz)
        denom += model ** 2. * (inv_err2 * corr_dz ** 2.)

    return num / denom


def _correct_scaling_ul(scaling, mod, obs, wz):
    """Correct the scaling factor when one or more fluxes and/or properties are
    upper limits.

    Parameters
    ----------
    scaling: array
        Contains the scaling factors of all the models
    mod: ModelsManager
        Contains the models
    obs: ObservationsManager
        Contains the observations
    wz: slice
        Selection of the models at the redshift of the observation or all the
        redshifts in photometric-redshift mode.
    """
    # Store the keys so we always read them in the same order
    fluxkeys = obs.flux.keys()
    fluxulkeys = obs.flux_ul.keys()
    extpropkeys = obs.extprop.keys()
    extpropulkeys = obs.extprop_ul.keys()

    # Put the fluxes and extensive properties in the same array for simplicity
    obsdata = [obs.flux[k] for k in fluxkeys]
    obsdata += [obs.extprop[k] for k in extpropkeys]
    obsdata_err = [obs.flux_err[k] for k in fluxkeys]
    obsdata_err += [obs.extprop_err[k] for k in extpropkeys]
    obslim = [obs.flux_ul[k] for k in fluxulkeys]
    obslim += [obs.extprop_ul[k] for k in extpropulkeys]
    obslim_err = [obs.flux_ul_err[k] for k in fluxulkeys]
    obslim_err += [obs.extprop_ul_err[k] for k in extpropulkeys]
    obsdata = np.array(obsdata)
    obsdata_err = np.array(obsdata_err)
    obslim = np.array(obslim)
    obslim_err = np.array(obslim_err)

    # We store views models at the right redshifts to avoid having SharedArray
    # recreate a numpy array for each model
    modflux = {k: mod.flux[k][wz] for k in mod.flux.keys()}
    modextprop = {k: mod.extprop[k][wz] for k in mod.extprop.keys()}
    for imod in np.where(np.isfinite(scaling))[0]:
        moddata = [modflux[k][imod] for k in fluxkeys]
        moddata += [modextprop[k][imod] for k in extpropkeys]
        modlim = [modflux[k][imod] for k in fluxulkeys]
        modlim += [modextprop[k][imod] for k in extpropulkeys]
        moddata = np.array(moddata)
        modlim = np.array(modlim)
        scaling[imod] = optimize.root(dchi2_over_ds2, scaling[imod],
                                      args=(obsdata, obsdata_err,
                                            obslim, obslim_err,
                                            moddata, modlim)).x


def compute_chi2(models, obs, corr_dz, wz, lim_flag):
    """Compute the χ² of observed fluxes with respect to the grid of models. We
    take into account upper limits if need be. Note that we look over the bands
    to avoid the creation of an array of the same size as the model_fluxes
    array. Because we loop on the bands and not on the models, the impact on
    the performance should be small.

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
    lim_flag: boolean
        Boolean indicating whether upper limits should be treated (True) or
        discarded (False)

    Returns
    -------
    chi2: array
        χ² for all the models in the grid
    scaling: array
        scaling of the models to obtain the minimum χ²
    """
    limits = lim_flag and (len(obs.flux_ul) > 0 or len(obs.extprop_ul) > 0)
    scaling = _compute_scaling(models, obs, corr_dz, wz)

    # Some observations may not have flux values in some filter(s), but
    # they can have upper limit(s).
    if limits is True:
        _correct_scaling_ul(scaling, models, obs, wz)

    # χ² of the comparison of each model to each observation.
    chi2 = np.zeros_like(scaling)

    # Computation of the χ² from fluxes
    for band, flux in obs.flux.items():
        # Multiplications are faster than divisions, so we directly use the
        # inverse error
        inv_flux_err = 1. / obs.flux_err[band]
        model = models.flux[band][wz]
        chi2 += ((model * scaling - flux) * inv_flux_err) ** 2.

    # Computation of the χ² from intensive properties
    for name, prop in obs.intprop.items():
        model = models.intprop[name][wz]
        chi2 += ((model - prop) * (1. / obs.intprop_err[name])) ** 2.

    # Computation of the χ² from extensive properties
    for name, prop in obs.extprop.items():
        # Multiplications are faster than divisions, so we directly use the
        # inverse error
        inv_prop_err = 1. / obs.extprop_err[name]
        model = models.extprop[name][wz]
        chi2 += (((scaling * model) * corr_dz - prop) * inv_prop_err) ** 2.

    # Finally take the presence of upper limits into account
    if limits is True:
        for band, obs_error in obs.flux_ul_err.items():
            model = models.flux[band][wz]
            chi2 -= 2. * np.log(.5 *
                                (1. + erf(((obs.flux_ul[band] -
                                 model * scaling) / (np.sqrt(2.) * obs_error)))))
        for band, obs_error in obs.extprop_ul_err.items():
            model = models.extprop[band][wz]
            chi2 -= 2. * np.log(.5 *
                                (1. + erf(((obs.extprop_ul[band] -
                                 model * scaling) / (np.sqrt(2.) * obs_error)))))

    return chi2, scaling


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

    mean = np.einsum('i, i', param, weights)
    delta = param - mean
    std = np.sqrt(np.einsum('i, i, i', delta, delta, weights))

    return (mean, std)
