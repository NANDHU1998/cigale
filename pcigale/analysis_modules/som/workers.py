import numpy as np

from ...warehouse import SedWarehouse
from pcigale.analysis_modules.som.som import UnsupervisedSOM, SupervisedSOM
from .utils import weighted_param


def init_sed(models, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    counter: Counter class object
        Counter for the number of models computed

    """
    global gbl_warehouse, gbl_models, gbl_counter

    gbl_warehouse = SedWarehouse()

    gbl_models = models
    gbl_counter = counter


def init_som(models, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    models: ModelsManagers
        Manages the storage of the computed models (fluxes and properties).
    counter: Counter class object
        Counter for the number of objects analysed

    """
    global gbl_models, gbl_counter

    gbl_models = models
    gbl_counter = counter


def init_fit(som, results, counter):
    """Initializer of the pool of processes to share variables between workers.

    Parameters
    ----------
    som: SupervisedSOM
        Trained self-organizing map
    results: ResultsManager
        Contains the estimates and errors on the properties.
    counter: Counter class object
        Counter for the number of objects analysed

    """
    global gbl_som, gbl_results, gbl_counter

    gbl_som = som
    gbl_results = results
    gbl_counter = counter


def sed(idx, midx):
    """Worker process to retrieve a SED and affect the relevant data to an
    instance of ModelsManager.

    Parameters
    ----------
    idx: int
        Index of the model within the current block of models.
    midx: int
        Global index of the model.

    """
    sed = gbl_warehouse.get_sed(
        gbl_models.params.modules, gbl_models.params.from_index(midx)
    )

    # The redshift is the fastest varying variable but we want to store it
    # as the slowest one so that models at a given redshift are contiguous
    idx = (idx % gbl_models.nz) * gbl_models.nm + idx // gbl_models.nz
    if "sfh.age" in sed.info and sed.info["sfh.age"] > sed.info["universe.age"]:
        for band in gbl_models.flux:
            gbl_models.flux[band][idx] = np.nan
        for prop in gbl_models.extprop:
            gbl_models.extprop[prop][idx] = np.nan
        for prop in gbl_models.intprop:
            gbl_models.intprop[prop][idx] = np.nan
    else:
        for band in gbl_models.flux:
            gbl_models.flux[band][idx] = sed.compute_fnu(band)
        for prop in gbl_models.extprop:
            gbl_models.extprop[prop][idx] = sed.info[prop]
        for prop in gbl_models.intprop:
            gbl_models.intprop[prop][idx] = sed.info[prop]
    gbl_models.index[idx] = midx

    gbl_counter.inc()


def som():
    usom = UnsupervisedSOM(shape=(30, 30), n=10000, periodic=True)
    usom.train(gbl_models.flux)

    som = SupervisedSOM(usom, n=10000)
    som.train(gbl_models.flux, gbl_models.extprop | gbl_models.intprop)

    return som


def somz(iz, z):
    if z == np.nan:
        usom = UnsupervisedSOM(shape=(30, 30), n=10000, periodic=True)
        usom.train(gbl_models.flux)

        ssom = SupervisedSOM(usom, n=10000)
        ssom.train(gbl_models.flux, gbl_models.extprop | gbl_models.intprop)
    else:
        s = slice(iz * gbl_models.nm, (iz + 1) * gbl_models.nm)

        flux = {k: v[s] for k, v in gbl_models.flux.items()}
        prop = {k: v[s] for k, v in (gbl_models.extprop | gbl_models.intprop).items()}

        usom = UnsupervisedSOM(shape=(30, 30), n=10000, periodic=True)
        usom.train(flux)

        ssom = SupervisedSOM(usom, n=10000)
        ssom.train(flux, prop)

    gbl_counter.inc()
    return (z, ssom)


def fit(idx, obs):
    """Worker process to analyse the PDF and estimate parameters values and
    store them in an instance of ResultsManager.

    Parameters
    ----------
    idx: int
        Index of the observation. This is necessary to put the computed values
        at the right location in the ResultsManager.
    obs: row
        Input data for an individual object

    """
    np.seterr(invalid="ignore")

    if len(obs.flux) >= 2:
        z = np.array(list(gbl_som.keys()))
        som = gbl_som[z[np.abs(obs.redshift - z).argmin()]]

        likelihood, alpha = som.usom.map.likelihood(obs)
        bmu = np.unravel_index(np.argmax(likelihood), likelihood.shape)

        likelihood *= 1.0 / np.sum(likelihood)
        likelihood = likelihood.ravel()

        for k in gbl_results.bayes.intmean:
            mean, std = weighted_param(som.map.weights[k].ravel(), likelihood)
            gbl_results.bayes.intmean[k][idx] = mean
            gbl_results.bayes.interror[k][idx] = std

        for k in gbl_results.bayes.extmean:
            mean, std = weighted_param((som.map.weights[k] * alpha).ravel(), likelihood)
            gbl_results.bayes.extmean[k][idx] = mean
            gbl_results.bayes.exterror[k][idx] = std

        for k in gbl_results.bayes.fluxmean:
            mean, std = weighted_param((som.map.weights[k] * alpha).ravel(), likelihood)
            gbl_results.bayes.fluxmean[k][idx] = mean
            gbl_results.bayes.fluxerror[k][idx] = std

        for k in gbl_results.bayes.intmean:
            gbl_results.best.intprop[k][idx] = som.map.weights[k][bmu]

        for k in gbl_results.bayes.extmean:
            gbl_results.best.extprop[k][idx] = som.map.weights[k][bmu] * alpha[bmu]

        for k in gbl_results.bayes.fluxmean:
            gbl_results.best.flux[k][idx] = som.map.weights[k][bmu] * alpha[bmu]

    gbl_counter.inc()
