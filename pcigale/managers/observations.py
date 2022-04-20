from astropy.table import Column
import numpy as np
from pathlib import Path
from scipy.constants import parsec

from pcigale.utils.cosmology import luminosity_distance
from pcigale.utils.io import read_table
from .utils import get_info

from pcigale.utils.console import console, WARNING


class ObservationsManager:
    """Class to abstract the handling of the observations and provide a
    consistent interface for the rest of cigale to deal with observations.

    An ObservationsManager is in charge of reading the input file to memory,
    check the consistency of the data, replace invalid values with NaN, etc.

    """
    def __new__(cls, config, params=None, **kwargs):
        if config['data_file']:
            return ObservationsManagerPassbands(config, params, **kwargs)
        else:
            return ObservationsManagerVirtual(config, **kwargs)


class ObservationsManagerPassbands:
    """Class to generate a manager for data files providing fluxes in
    passbands.

    A class instance can be used as a sequence. In that case a row is returned
    at each iteration.
    """

    def __init__(self, config, params, defaulterror=0.1, modelerror=0.1,
                 threshold=-9990.):

        self.conf = config
        self.params = params
        self.allpropertiesnames, _, self.massproportional = get_info(self)
        self.table = read_table(config['data_file'])
        self.bands = [band for band in config['bands']
                      if not band.endswith('_err')]
        self.bands_err = [band for band in config['bands']
                          if band.endswith('_err')]
        self.intprops = [prop for prop in config['properties']
                         if (prop not in self.massproportional
                             and not prop.endswith('_err'))]
        self.intprops_err = [prop for prop in config['properties']
                             if (prop.endswith('_err')
                                 and prop[:-4] not in self.massproportional)]
        self.extprops = [prop for prop in config['properties']
                         if (prop in self.massproportional
                             and not prop.endswith('_err'))]
        self.extprops_err = [prop for prop in config['properties']
                             if (prop.endswith('_err')
                                 and prop[:-4] in self.massproportional)]
        self.tofit = self.bands + self.intprops + self.extprops
        self.tofit_err = self.bands_err + self.intprops_err + self.extprops_err

        # Sanitise the input
        self._check_id()
        self._check_filters()
        self._check_errors(defaulterror)
        self._check_invalid(config['analysis_params']['lim_flag'],
                            threshold)
        self._add_model_error(config["additionalerror"])

        # Rebuild the quantities to fit after vetting them
        self.tofit = self.bands + self.intprops + self.extprops
        self.tofit_err = self.bands_err + self.intprops_err + self.extprops_err

        self.observations = list([Observation(row, self)
                                  for row in self.table])

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        self.idx = 0
        self.max = len(self.observations)

        return self

    def __next__(self):
        if self.idx < self.max:
            obs = self.observations[self.idx]
            self.idx += 1
            return obs
        raise StopIteration

    def _check_id(self):
        """Check that the id of each object is unique. If not trigger an
        exception as it may cause issues down the road.

        """
        values, counts = np.unique(self.table['id'], return_counts=True)
        duplicates = values[counts > 1].data

        if duplicates.size > 0:
            raise Exception("The input file has the following duplicated id: " +
                            ", ".join(duplicates.astype(str)) +
                            ". The id must be unique.")

    def _check_filters(self):
        """Check whether the list of filters and poperties makes sense.

        Two situations are checked:
        * If a filter or property to be included in the fit is missing from
        the data file, an exception is raised.
        * If a filter or property is given in the input file but is not to be
        included in the fit, a warning is displayed

        """
        for item in self.tofit + self.tofit_err:
            if item not in self.table.colnames:
                raise Exception(f"{item} to be taken in the fit but not present"
                                f" in the observation table.")

        for item in self.table.colnames:
            if (item != 'id' and item != 'redshift' and item != 'distance' and
                    item not in self.tofit + self.tofit_err):
                self.table.remove_column(item)
                console.print(f"{WARNING} [b]{item}[/b] in the input file but "
                              "not to be taken into account in the fits.")

    def _check_errors(self, defaulterror=0.1):
        """Check whether the error columns are present. If not, add them.

        This happens in two cases. Either when the error column is not in the
        list of bands to be analysed or when the error column is not present
        in the data file. Note that an error cannot be included explicitly if
        it is not present in the input file. Such a case would be ambiguous
        and will have been caught by self._check_filters().

        We take the error as defaulterror × flux, so by default 10% of the
        flux. The absolute value of the flux is taken in case it is negative.

        Parameters
        ----------
        defaulterror: float
            Fraction of the flux to take as the error when the latter is not
            given in input. By default: 10%.

        """
        if defaulterror < 0.:
            raise ValueError("The relative default error must be positive.")

        for item in self.tofit:
            error = item + '_err'
            if item in self.intprops:
                if (error not in self.intprops_err or
                        error not in self.table.colnames):
                    raise ValueError("Intensive properties errors must be in "
                                     "input file.")
            elif (error not in self.tofit_err or
                  error not in self.table.colnames):
                colerr = Column(data=np.fabs(self.table[item] * defaulterror),
                                name=error)
                self.table.add_column(colerr,
                                      index=self.table.colnames.index(item) + 1)
                console.print(f"{WARNING} {defaulterror * 100}% of {item} "
                              "taken as errors.")

    def _check_invalid(self, upperlimits="none", threshold=-9990.):
        """Check whether invalid data are correctly marked as such.

        This happens in two cases:
        * Data are marked as upper limits (negative error) but the upper
        limit flag is set to False.
        * Data or errors are lower than -9990.

        We mark invalid data as np.nan. In case the entire column is made of
        invalid data, we remove them from the table

        Parameters
        ----------
        threshold: float
            Threshold under which the data are consisdered invalid.

        """
        allinvalid = []

        for item in self.bands + self.extprops:
            error = item + '_err'
            if upperlimits == "none":
                w = np.where((self.table[item] < threshold) |
                             (self.table[error] <= 0.))
            else:
                w = np.where((self.table[item] < threshold) |
                             (self.table[error] == 0.))
            self.table[item][w] = np.nan
            self.table[error][w] = np.nan
            if np.all(~np.isfinite(self.table[item])):
                allinvalid.append(item)

        for item in allinvalid:
            if item in self.bands:
                self.bands.remove(item)
                self.bands_err.remove(item + '_err')
            elif item in self.extprops:
                self.extprops.remove(item)
                self.extprops_err.remove(item + '_err')
            self.table.remove_columns([item, item + '_err'])
            console.print(f"{WARNING} {allinvalid} removed as no valid data "
                          "was found.")

    def _add_model_error(self, modelerror=0.1):
        """Add in quadrature the error of the model to the input error.

        Parameters
        ----------
        modelerror: float
            Relative error of the models relative to the flux (or property). By
            default 10%.

        """
        if modelerror < 0.:
            raise ValueError("The relative model error must be positive.")

        for item in self.bands + self.extprops:
            error = item + '_err'
            w = np.where(self.table[error] >= 0.)
            self.table[error][w] = np.sqrt(self.table[error][w]**2. + (
                self.table[item][w] * modelerror)**2.)

    def generate_mock(self, fits):
        """Replaces the actual observations with a mock catalogue. It is
        computed from the best fit fluxes of a previous run. For each object
        and each band, we randomly draw a new flux from a Gaussian distribution
        centered on the best fit flux and with a standard deviation identical
        to the observed one.

        Parameters
        ----------
        fits: ResultsManager
            Best fit fluxes of a previous run.

        """
        for band in self.bands:
            err = band + '_err'
            self.table[band] = np.random.normal(fits.best.flux[band],
                                                np.fabs(self.table[err]))
        for prop in self.intprops:
            err = prop + '_err'
            self.table[prop] = np.random.normal(fits.best.intprop[prop],
                                                np.fabs(self.table[err]))
        for prop in self.extprops:
            err = prop + '_err'
            self.table[prop] = np.random.normal(fits.best.extprop[prop],
                                                np.fabs(self.table[err]))

        self.observations = list([Observation(row, self)
                                  for row in self.table])

    def save(self, filename):
        """Saves the observations as seen internally by the code so it is easy
        to see what fluxes are actually used in the fit. Files are saved in
        FITS and ASCII formats.

        Parameters
        ----------
        filename: str
            Root of the filename where to save the observations.

        """
        out = Path('out')
        self.table.write(out / f'{filename}.fits')
        self.table.write(out / f'{filename}.txt', format='ascii.fixed_width',
                         delimiter=None)


class ObservationsManagerVirtual:
    """Virtual observations manager when there is no observations file given
    as input. In that case we only use the list of bands given in the
    pcigale.ini file.
    """

    def __init__(self, config, **kwargs):
        self.bands = [band for band in config['bands'] if not
                      band.endswith('_err')]

        if len(self.bands) != len(config['bands']):
            console.print(f"{WARNING} error bars were given in the list of "
                          "bands.")
        elif len(self.bands) == 0:
            console.print(f"{WARNING} no band was given.")

        # We set the other class members to None as they do not make sense in
        # this situation
        self.bands_err = None
        self.table = None
        self.intprops = set()
        self.extprops = set()

    def __len__(self):
        """As there is no observed object the length is naturally 0.
        """
        return 0


class Observation:
    """Class to take one row of the observations table and extract the list of
    fluxes, intensive properties, extensive properties and their errors, that
    are going to be considered in the fit.
    """

    def __init__(self, row, cls):
        self.redshift = row['redshift']
        self.id = row['id']
        if 'distance' in row.colnames and np.isfinite(row['distance']):
            self.distance = row['distance'] * parsec * 1e6
        else:
            if self.redshift >= 0.:
                self.distance = luminosity_distance(self.redshift)
            else:
                self.distance = np.nan
        self.flux = {k: row[k] for k in cls.bands
                     if np.isfinite(row[k]) and row[k + '_err'] > 0.}
        self.flux_ul = {k: row[k] for k in cls.bands
                        if np.isfinite(row[k]) and row[k + '_err'] <= 0.}
        self.flux_err = {k: row[k + '_err'] for k in self.flux.keys()}
        self.flux_ul_err = {k: -row[k + '_err'] for k in self.flux_ul.keys()}

        self.extprop = {k: row[k] for k in cls.extprops
                        if np.isfinite(row[k]) and row[k + '_err'] > 0.}
        self.extprop_ul = {k: row[k] for k in cls.extprops
                           if np.isfinite(row[k]) and row[k + '_err'] <= 0.}
        self.extprop_err = {k: row[k + '_err'] for k in self.extprop.keys()}
        self.extprop_ul_err = {k: -row[k + '_err']
                               for k in self.extprop_ul.keys()}

        self.intprop = {k: row[k] for k in cls.intprops}
        self.intprop_err = {k: row[k + '_err'] for k in cls.intprops}
