"""These classes manage the results from the analysis. The main class
ResultsManager contains instances of BayesResultsManager and BestResultsManager
that contain the bayesian and best-fit estimates of the physical properties
along with the names of the parameters, which are proportional to the mass,
etc. Each of these classes contain a merge() method that allows to combine
results of the analysis with different blocks of models.
"""
import ctypes
from pathlib import Path

from astropy.table import Table, Column
from astropy.units import Unit, LogUnit
import numpy as np

from pcigale.utils.console import console, INFO, ERROR
from .utils import SharedArray


class BayesResultsManager:
    """This class contains the results of the bayesian estimates of the
    physical properties of the analysed objects. It is constructed from a
    ModelsManager instance, which provides the required information on the
    shape of the arrays. Because it can contain the partial results for only
    one block of models, we also store the sum of the model weights (that is
    the likelihood) so we can merge different instances to compute the combined
    estimates of the physical properties.

    """
    def __init__(self, models):
        nobs = len(models.obs)
        self.propertiesnames = models.allpropnames
        extpropnames = [prop for prop in models.obs.conf['analysis_params']['variables']
                        if (prop in models.allextpropnames or
                            prop[:-4] in models.allextpropnames)]
        intpropnames = [prop for prop in models.obs.conf['analysis_params']['variables']
                        if (prop in models.allintpropnames or
                            prop[:-4] in models.allintpropnames)]
        fluxnames = [name for name in models.conf['analysis_params']['bands']]
        self.nproperties = len(intpropnames) + len(extpropnames)

        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        self.intmean = {prop: SharedArray(nobs) for prop in intpropnames}
        self.interror = {prop: SharedArray(nobs) for prop in intpropnames}
        self.extmean = {prop: SharedArray(nobs) for prop in extpropnames}
        self.exterror = {prop: SharedArray(nobs) for prop in extpropnames}
        self.fluxmean = {band: SharedArray(nobs) for band in fluxnames}
        self.fluxerror = {band: SharedArray(nobs) for band in fluxnames}
        self.weight = SharedArray(nobs)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of BayesResultsManager instances
            List of the partial results to be merged

        """
        if not (isinstance(results, list) and
                all(isinstance(result, BayesResultsManager)
                    for result in results)):
            raise TypeError("A list of BayesResultsManager is required.")

        merged = results[0]
        intmean = {prop: np.array([result.intmean[prop]
                                   for result in results])
                   for prop in merged.intmean}
        interror = {prop: np.array([result.interror[prop]
                                    for result in results])
                    for prop in merged.interror}
        extmean = {prop: np.array([result.extmean[prop]
                                   for result in results])
                   for prop in merged.extmean}
        exterror = {prop: np.array([result.exterror[prop]
                                    for result in results])
                    for prop in merged.exterror}
        fluxmean = {band: np.array([result.fluxmean[band]
                                   for result in results])
                    for band in merged.fluxmean}
        fluxerror = {band: np.array([result.fluxerror[band]
                                    for result in results])
                     for band in merged.fluxerror}
        weight = np.array([result.weight for result in results])

        totweight = np.nansum(weight, axis=0)

        for prop in merged.intmean:
            merged.intmean[prop][:] = np.nansum(
                intmean[prop] * weight, axis=0) / totweight

            # We compute the merged standard deviation by combining the
            # standard deviations for each block. See
            # http://stats.stackexchange.com/a/10445 where the number of
            # datapoints has been substituted with the weights. In short we
            # exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
            merged.interror[prop][:] = np.sqrt(np.nansum(
                weight * (interror[prop]**2. + (intmean[prop] - merged.intmean[prop])**2), axis=0) / totweight)

        for prop in merged.extmean:
            merged.extmean[prop][:] = np.nansum(
                extmean[prop] * weight, axis=0) / totweight

            # We compute the merged standard deviation by combining the
            # standard deviations for each block. See
            # http://stats.stackexchange.com/a/10445 where the number of
            # datapoints has been substituted with the weights. In short we
            # exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
            merged.exterror[prop][:] = np.sqrt(np.nansum(
                weight * (exterror[prop]**2. + (extmean[prop] - merged.extmean[prop])**2), axis=0) / totweight)

        for prop in merged.extmean:
            if prop.endswith('_log'):
                merged.exterror[prop][:] = \
                    np.maximum(0.02, merged.exterror[prop])
            else:
                merged.exterror[prop][:] = \
                    np.maximum(0.05 * merged.extmean[prop],
                               merged.exterror[prop])

        for band in merged.fluxmean:
            merged.fluxmean[band][:] = np.nansum(
                fluxmean[band] * weight, axis=0) / totweight

            # We compute the merged standard deviation by combining the
            # standard deviations for each block. See
            # http://stats.stackexchange.com/a/10445 where the number of
            # datapoints has been substituted with the weights. In short we
            # exploit the fact that Var(X) = E(Var(X)) + Var(E(X)).
            merged.fluxerror[band][:] = np.sqrt(np.nansum(
                weight * (fluxerror[band]**2. + (fluxmean[band] - merged.fluxmean[band])**2), axis=0) / totweight)

        return merged


class BestResultsManager:
    """This class contains the physical properties of the best fit of the
    analysed objects. It is constructed from a ModelsManager instance, which
    provides the required information on the shape of the arrays. Because it
    can contain the partial results for only one block of models, we also store
    the index so we can merge different instances to compute the best fit.

    """

    def __init__(self, models):
        self.obs = models.obs
        nobs = len(models.obs)
        # Arrays where we store the data related to the models. For memory
        # efficiency reasons, we use RawArrays that will be passed in argument
        # to the pool. Each worker will fill a part of the RawArrays. It is
        # important that there is no conflict and that two different workers do
        # not write on the same section.
        fluxnames = list(dict.fromkeys(models.obs.bands +
                                       models.conf['analysis_params']['bands']))
        self.flux = {band: SharedArray(nobs) for band in fluxnames}
        allintpropnames = models.allintpropnames
        allextpropnames = models.allextpropnames
        self.intprop = {prop: SharedArray(nobs)
                        for prop in allintpropnames}
        self.extprop = {prop: SharedArray(nobs)
                        for prop in allextpropnames}
        self.chi2 = SharedArray(nobs)
        self.scaling = SharedArray(nobs)
        self.index = SharedArray(nobs, ctypes.c_uint32)

    @property
    def flux(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._flux

    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @property
    def intprop(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._intprop

    @intprop.setter
    def intprop(self, intprop):
        self._intprop = intprop

    @property
    def extprop(self):
        """Returns a shared array containing the fluxes of the best fit for
        each observation.

        """
        return self._extprop

    @extprop.setter
    def extprop(self, extprop):
        self._extprop = extprop

    @property
    def chi2(self):
        """Returns a shared array containing the raw chi² of the best fit for
        each observation.

        """
        return self._chi2

    @chi2.setter
    def chi2(self, chi2):
        self._chi2 = chi2

    @property
    def index(self):
        """Returns a shared array containing the index of the best fit for each
        observation.

        """
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def scaling(self):
        """Returns a shared array containing the scaling of the best fit for each
        observation.

        """
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of BestResultsManager instances
            List of the partial results to be merged

        """
        if not (isinstance(results, list) and
                all(isinstance(result, BestResultsManager)
                    for result in results)):
            raise TypeError("A list of BestResultsManager is required.")

        if len(results) == 1:
            return results[0]

        chi2 = np.array([result.chi2 for result in results])
        merged = results[0]
        for iobs, bestidx in enumerate(np.argsort(chi2, axis=0)[0, :]):
            if np.isfinite(bestidx):
                for band in merged.flux:
                    merged.flux[band][iobs] = \
                        results[bestidx].flux[band][iobs]
                for prop in merged.intprop:
                    merged.intprop[prop][iobs] = \
                        results[bestidx].intprop[prop][iobs]
                for prop in merged.extprop:
                    merged.extprop[prop][iobs] = \
                        results[bestidx].extprop[prop][iobs]
                merged.chi2[iobs] = results[bestidx].chi2[iobs]
                merged.scaling[iobs] = results[bestidx].scaling[iobs]
                merged.index[iobs] = results[bestidx].index[iobs]

        return merged

    def analyse_chi2(self):
        """Function to analyse the best chi^2 and find out what fraction of
         objects seems to be overconstrainted.

        """
        # If no best model has been found, it means none could be properly
        # fitted. We warn the user in that case
        bad = [str(id_) for id_ in self.obs.table['id'][np.isnan(self.chi2)]]
        if len(bad) > 0:
            console.print("{ERROR} No suitable model found for "
                          f"{', '.join(bad)}. It may be that models are older "
                          "than the universe or that the χ² are very large.")

        obs = [self.obs.table[obs].data for obs in self.obs.tofit]
        nobs = np.count_nonzero(np.isfinite(obs), axis=0)
        chi2_red = self.chi2 / (nobs - 1)
        # If low values of reduced chi^2, it means that the data are overfitted
        # Errors might be under-estimated or not enough valid data.
        console.print(f"{INFO} {np.round(100. * (chi2_red < 1e-12).sum() / chi2_red.size, 1)}%"
                      " of the objects have χ²_red~0 and "
                      f"{np.round(100. * (chi2_red < 0.5).sum() / chi2_red.size, 1)}% "
                      "χ²_red<0.5.")


class ResultsManager:
    """This class contains the physical properties (best fit and bayesian) of
    the analysed objects. It is constructed from a ModelsManager instance,
    which provides the required information to initialise the instances of
    BestResultsManager and BayesResultsManager that store the results.

    """

    def __init__(self, models):
        self.conf = models.conf
        self.obs = models.obs
        self.params = models.params
        self.unit = models.unit

        self.bayes = BayesResultsManager(models)
        self.best = BestResultsManager(models)

    @staticmethod
    def merge(results):
        """Merge a list of partial results computed on individual blocks of
        models.

        Parameters
        ----------
        results: list of ResultsManager instances
            List of the partial results to be merged

        """
        merged = results[0]
        merged.bayes = BayesResultsManager.merge([result.bayes
                                                  for result in results])
        merged.best = BestResultsManager.merge([result.best
                                                for result in results])

        return merged

    @staticmethod
    def fluxunit(band):
        """Help function to determine whether the band is a line or a filter
        and returns the appropriate unit.

        Parameters
        ----------
        band: str
            Name of the band. For a line it must start with ".line".

        Returns
        -------
        unit: astropy.unit.Unit
            Unit of the band flux.
        """
        if band.startswith('line.') or band.startswith('linefilter.'):
            return Unit('W/m^2')
        return Unit('mJy')

    def save(self, filename):
        """Save the estimated values derived from the analysis of the PDF and
        the parameters associated with the best fit. A simple text file and a
        FITS file are generated.

        Parameters
        ----------
        filename:

        """
        table = Table()

        table.add_column(Column(self.obs.table['id'], name="id"))

        for prop in sorted(self.bayes.intmean):
            if prop.endswith('_log'):
                unit = LogUnit(self.unit[prop[:-4]])
            else:
                unit = Unit(self.unit[prop])
            table.add_column(Column(self.bayes.intmean[prop],
                                    name="bayes." + prop, unit=unit))
            table.add_column(Column(self.bayes.interror[prop],
                                    name="bayes." + prop + "_err", unit=unit))
        for prop in sorted(self.bayes.extmean):
            if prop.endswith('_log'):
                unit = LogUnit(self.unit[prop[:-4]])
            else:
                unit = Unit(self.unit[prop])
            table.add_column(Column(self.bayes.extmean[prop],
                                    name="bayes." + prop, unit=unit))
            table.add_column(Column(self.bayes.exterror[prop],
                                    name="bayes." + prop + "_err", unit=unit))
        for band in sorted(self.bayes.fluxmean):
            unit = self.fluxunit(band)
            table.add_column(Column(self.bayes.fluxmean[band],
                                    name="bayes." + band, unit=unit))
            table.add_column(Column(self.bayes.fluxerror[band],
                                    name="bayes." + band + "_err",
                                    unit=unit))

        table.add_column(Column(self.best.chi2, name="best.chi_square"))
        obs = [self.obs.table[obs].data for obs in self.obs.tofit]
        nobs = np.count_nonzero(np.isfinite(obs), axis=0)
        table.add_column(Column(self.best.chi2 / (nobs - 1),
                                name="best.reduced_chi_square"))

        for prop in sorted(self.best.intprop):
            table.add_column(Column(self.best.intprop[prop],
                                    name="best." + prop,
                                    unit=Unit(self.unit[prop])))
        for prop in sorted(self.best.extprop):
            table.add_column(Column(self.best.extprop[prop],
                                    name="best." + prop,
                                    unit=Unit(self.unit[prop])))

        for band in self.best.flux:
            table.add_column(Column(self.best.flux[band],
                                    name="best." + band,
                                    unit=self.fluxunit(band)))

        out = Path('out')
        table.write(out / f"{filename}.txt", format='ascii.fixed_width',
                    delimiter=None)
        table.write(out / f"{filename}.fits", format='fits')
