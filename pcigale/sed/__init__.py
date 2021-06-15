"""
This class represents a Spectral Energy Distribution (SED) as used by pcigale.
Such SED is characterised by:

- sfh: the Star Formation History of the galaxy.

- modules: a list of tuples (module name, parameter dictionary) containing all
  the pcigale modules the SED 'went through'.

- wavelength_grid: the grid of wavelengths [nm] used for the luminosities.

- luminosities: a dictionary containing arrays of all the luminosity density
  [W/nm] contributions to the SED.

- info: a dictionary containing various information about the SED.

- mass_proportional_info: the set of keys in the info dictionary whose value
  is proportional to the galaxy mass.

"""

import numpy as np
from numpy.core.multiarray import interp  # Compiled version
from scipy.constants import parsec

from . import utils
from .io.vo import save_sed_to_vo
from .io.fits import save_sed_to_fits
from ..data import SimpleDatabase as Database


class SED:
    """Spectral Energy Distribution with associated information
    """

    # We declare the filters cache here as to be efficient it needs to be
    # shared between different objects.
    cache_filters = {}

    def __init__(self, sfh=None):
        """Create a new SED

        Parameters
        ----------
        sfh: (numpy.array, numpy.array)
            Star Formation History: tuple of two numpy array, the first is the
            time in Myr and the second is the Star Formation Rate in Msun/yr.
            If no SFH is given, it's set to None.

        """
        self.sfh = sfh
        self.modules = []
        self.wavelength_grid = None
        self.luminosity = None
        self.luminosities = None
        self.lines = dict()
        self.info = dict()
        self.mass_proportional_info = set()
        self.unit = dict()

    @property
    def sfh(self):
        """Return a copy of the star formation history
        """
        if self._sfh is None:
            return None

        return np.copy(self._sfh)

    @sfh.setter
    def sfh(self, value):

        # The SFH can be set multiple times. Maybe it's better to make is
        # settable only once and then provide an update_sfh method for when
        # it's needed.
        self._sfh = value

        if value is not None:
            sfh_sfr = value
            self._sfh = value
            self.add_info("sfh.sfr", sfh_sfr[-1], True, force=True,
                          unit='solMass/yr')
            self.add_info("sfh.sfr10Myrs", np.mean(sfh_sfr[-10:]), True,
                          force=True, unit='solMass/yr')
            self.add_info("sfh.sfr100Myrs", np.mean(sfh_sfr[-100:]), True,
                          force=True, unit='solMass/yr')
            self.add_info("sfh.age", sfh_sfr.size, False, force=True,
                          unit='Myr')

    @property
    def fnu(self):
        """Total Fν flux density of the SED

        Return the total Fν density vector, i.e the total luminosity converted
        to Fν flux in mJy.
        """

        # Fλ flux density in W/m²/nm
        if 'universe.luminosity_distance' in self.info:
            f_lambda = utils.luminosity_to_flux(self.luminosity,
                                                self.info
                                                ['universe.luminosity_distance'])
        else:
            f_lambda = utils.luminosity_to_flux(self.luminosity, 10. * parsec)

        # Fν flux density in mJy
        f_nu = utils.lambda_flambda_to_fnu(self.wavelength_grid, f_lambda)

        return f_nu

    def add_info(self, key, value, mass_proportional=False, force=False,
                 unit=''):
        """
        Add a key / value to the information dictionary

        If the key is present in the dictionary, it will raise an exception.
        Use this method (instead of direct value assignment ) to avoid
        overriding an already present information.

        Parameters
        ----------
        key: any immutable
           The key used to retrieve the information.
        value: anything
           The information.
        mass_proportional: boolean
           If True, the added variable is set as proportional to the
           mass.
        force: boolean
           If false (default), adding a key that already exists in the info
           dictionary will raise an error. If true, doing this will update
           the associated value.

        """
        if (key not in self.info) or force:
            self.info[key] = value
            self.unit[key] = unit
            if mass_proportional:
                self.mass_proportional_info.add(key)
        else:
            raise KeyError("The information %s is already present "
                           "in the SED. " % key)

    def add_module(self, module_name, module_conf):
        """Add a new module information to the SED.

        Parameters
        ----------
        module_name: string
            Name of the module. This name can be suffixed with anything
            using a dot.
        module_conf: dictionary
            Dictionary containing the module parameters.

        TODO: Complete the parameter dictionary with the default values from
              the module if they are not present.

        """
        self.modules.append((module_name, module_conf))

    def add_contribution(self, contribution_name, results_wavelengths,
                         results_lumin):
        """
        Add a new luminosity contribution to the SED.

        The luminosity contribution of the module is added to the contribution
        table doing an interpolation between the current wavelength grid and
        the grid of the module contribution. During the interpolation,
        everything that is outside of the concerned wavelength domain has its
        luminosity set to 0. Also, the name of the contribution is added to
        the contribution names array.

        Parameters
        ----------
        contribution_name: string
            Name of the contribution added. This name is used to retrieve the
            luminosity contribution and allows one module to add more than
            one contribution.

        results_wavelengths: array of floats
            The vector of the wavelengths of the module results (in nm).

        results_lumin: array of floats
            The vector of the Lλ luminosities (in W/nm) of the module results.

        """
        # If the SED luminosity table is empty, then there is nothing to
        # compute.
        if self.luminosity is None:
            self.wavelength_grid = results_wavelengths
            self.luminosities = {contribution_name: results_lumin}
            # Make a copy so that changing the luminosity does not affect the
            # component
            self.luminosity = results_lumin.copy()
        else:
            # If the added luminosity contribution changes the SED wavelength
            # grid, we interpolate everything on a common wavelength grid.
            if (results_wavelengths.size != self.wavelength_grid.size or
                    (np.count_nonzero(results_wavelengths != self.wavelength_grid) > 0)):
                # Interpolate each luminosity component to the new wavelength
                # grid setting everything outside the wavelength domain to 0.
                self.wavelength_grid, self.luminosities = \
                    utils.interpolate_lumin(self.wavelength_grid,
                                            self.luminosities,
                                            results_wavelengths,
                                            results_lumin,
                                            contribution_name)

                self.luminosity = np.sum(list(self.luminosities.values()),
                                         axis=0)
            else:

                self.luminosities[contribution_name] = results_lumin
                self.luminosity += results_lumin

    def compute_fnu(self, filter_name):
        """
        Compute the Fν flux density in a given filter

        The filters are stored in the database in such a way that after
        integration and conversion from luminosity to flux we directly get the
        latter in units of mJy. If the SED spectrum does not cover all the
        filter response table, NaN is returned.

        Parameters
        ----------
        filter_name: string
            Name of the filter to integrate into. It must be presnt in the
            database.

        Return
        ------
        fnu: float
            The integrated Fν density in mJy.
        """

        wavelength = self.wavelength_grid

        # First we try to fetch the filter's wavelength, transmission and
        # pivot wavelength from the cache. The two keys are the size of the
        # spectrum wavelength grid and the name of the filter. The first key is
        # necessary because different spectra may have different sampling. To
        # avoid having the resample the filter every time on the optimal grid
        # (spectrum+filter), we store the resampled filter. That way we only
        # have to resample to spectrum.
        if 'universe.redshift' in self.info:
            if 'nebular.lines_width' in self.info:
                key = (wavelength.size, filter_name,
                       self.info['nebular.lines_width'],
                       self.info['universe.redshift'])
            else:
                key = (wavelength.size, filter_name,
                       self.info['universe.redshift'])
            dist = self.info['universe.luminosity_distance']
        else:
            if 'nebular.lines_width' in self.info:
                key = (wavelength.size, filter_name,
                       self.info['nebular.lines_width'], 0.)
            else:
                key = (wavelength.size, filter_name, 0.)
            dist = 10. * parsec

        if filter_name.startswith('line.'):
            lum = 0
            for name in filter_name.split('+'):
                line = self.lines[name.split('.', maxsplit=1)[1]]
                lum += line[1] + line[2]  # Young and old components
            return utils.luminosity_to_flux(lum, dist)

        if key in self.cache_filters:
            wavelength_r, transmission_r, lambda_piv = self.cache_filters[key]
        else:
            with Database("filters") as db:
                filter_ = db.get(name=filter_name)
            wl = filter_.wl
            tr = filter_.tr
            lambda_piv = filter_.pivot
            lambda_min = wl[0]
            lambda_max = wl[-1]
            if filter_name.startswith('linefilter.'):
                if 'universe.redshift' in self.info:
                    zp1 = 1. + self.info['universe.redshift']
                else:
                    zp1 = 1.
                wl *= zp1
                lambda_piv *= zp1
                lambda_min *= zp1
                lambda_max *= zp1

            # Test if the filter covers all the spectrum extent. If not then
            # the flux is not defined
            if ((wavelength[0] > lambda_min) or (wavelength[-1] < lambda_max)):
                return np.nan

            # We regrid both spectrum and filter to the best wavelength grid
            # to avoid interpolating a high wavelength density curve to a low
            # density one. Also, we limit the work wavelength domain to the
            # filter one.
            w = np.where((wavelength >= lambda_min) &
                         (wavelength <= lambda_max))
            wavelength_r = utils.best_grid(wavelength[w], wl, key)
            transmission_r = interp(wavelength_r, wl, tr)

            self.cache_filters[key] = (wavelength_r, transmission_r,
                                       lambda_piv)

        l_lambda_r = interp(wavelength_r, wavelength, self.luminosity)

        # We compute directly Fν from ∫T×Fλ×dλ/∫T×c/λ²×dλ. The filter bandpass
        # in the database is already normalised so that we do not need to
        # compute the denominator (it is a constant that does not depend on the
        # spectrum) and the normalisation is such that the results we obtain
        # are directly in mJy.
        f_nu = utils.luminosity_to_flux(
            utils.flux_trapz(transmission_r * l_lambda_r, wavelength_r, key),
            dist)

        return f_nu

    def to_votable(self, filename, mass=1.):
        """
        Save the SED to a VO-table file

        Parameters
        ----------
        filename: string
            Name of the VO-table file
        mass: float
            Galaxy mass in solar mass. When need, the saved data will be
            multiplied by this mass.

        """
        save_sed_to_vo(self, filename, mass)

    def to_fits(self, prefix, mass=1.):
        """
        Save the SED to FITS files.

        Parameters
        ----------
        prefix: string
            Prefix of the fits file containing the path and the id of the model
        mass: float
            Galaxy mass in solar masses. When needed, the data will be scaled
            to this mass

        """
        save_sed_to_fits(self, prefix, mass)

    def copy(self):
        """
        Create a new copy of the object. This is done manually rather than
        using copy.deepcopy() for speed reasons. As we know the structure of
        the object, we can do a better job.

        """
        sed = SED()
        if self._sfh is not None:
            sed._sfh = self._sfh
        sed.modules = self.modules[:]
        if self.wavelength_grid is not None:
            sed.wavelength_grid = self.wavelength_grid.copy()
            sed.luminosity = self.luminosity.copy()
            # Copy only the dictionary but not the content. Individual
            # components are considered immutable.
            sed.luminosities = self.luminosities.copy()

        sed.lines = self.lines.copy()
        sed.info = self.info.copy()
        sed.unit = self.unit  # No need to copy, the units will not change
        sed.mass_proportional_info = self.mass_proportional_info.copy()

        return sed
