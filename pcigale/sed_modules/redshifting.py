"""
Redshifting module
==================

This module implements the redshifting of a SED. The SED must be rest-frame
or the module will raise en exception when processing it.

Note that this module, contrary to the other SED creation modules, actually
changes the individual luminosity contributions as it redshifts everyone.
Also note that doing this, this module does not rely on the SED object
interface but on its inner implementations. That means that if the SED object
is changed, this module may need to be adapted.

"""

import numpy as np
from scipy.special import factorial
from ..utils.cosmology import age, luminosity_distance

from . import SedModule


def igm_transmission(wavelength, redshift):
    """Intergalactic transmission (Meiksin, 2006)

    Compute the intergalactic transmission as described in Meiksin, 2006.

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength(s) in nm.
    redshift: float
        The redshift. Must be strictly positive.

    Returns
    -------
    igm_transmission: numpy array of floats
        The intergalactic transmission at each input wavelength.

    """
    n_transitions_low = 10
    n_transitions_max = 31
    gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    lambda_limit = 91.2  # Lyman limit in nm

    lambda_n = np.empty(n_transitions_max)
    z_n = np.empty((n_transitions_max, len(wavelength)))
    for n in range(2, n_transitions_max):
        lambda_n[n] = lambda_limit / (1. - 1. / float(n * n))
        z_n[n, :] = (wavelength / lambda_n[n]) - 1.

    # From Table 1 in Meiksin (2006), only n >= 3 are relevant.
    # fact has a length equal to n_transitions_low.
    fact = np.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373,
                     0.0283])

    # First, tau_alpha is the mean Lyman alpha transmitted flux,
    # Here n = 2 => tau_2 = tau_alpha
    tau_n = np.zeros((n_transitions_max, len(wavelength)))
    if redshift <= 4:
        tau_a = 0.00211 * np.power(1. + redshift, 3.7)
        tau_n[2, :] = 0.00211 * np.power(1. + z_n[2, :], 3.7)
    elif redshift > 4:
        tau_a = 0.00058 * np.power(1. + redshift, 4.5)
        tau_n[2, :] = 0.00058 * np.power(1. + z_n[2, :], 4.5)

    # Then, tau_n is the mean optical depth value for transitions
    # n = 3 - 9 -> 1
    for n in range(3, n_transitions_max):
        if n <= 5:
            w = np.where(z_n[n, :] < 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 3.)))
            w = np.where(z_n[n, :] >= 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 6.)))
        elif 5 < n <= 9:
            tau_n[n, :] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, :]), (1. / 3.)))
        else:
            tau_n[n, :] = (tau_n[9, :] * 720. /
                           (float(n) * (float(n * n - 1.))))

    for n in range(2, n_transitions_max):
        # If z_n>=redshift or z_n<0, the photon cannot be absorbed by Lyman n→1
        w = np.where((z_n[n, :] >= redshift) | (z_n[n, :] < 0))
        tau_n[n, w] = 0.

    z_l = wavelength / lambda_limit - 1.
    w = slice(None, np.searchsorted(z_l, redshift))

    tau_l_igm = np.zeros_like(wavelength)
    tau_l_igm[w] = (0.805 * np.power(1. + z_l[w], 3) *
                    (1. / (1. + z_l[w]) - 1. / (1. + redshift)))

    term1 = gamma - np.exp(-1.)

    n = np.arange(n_transitions_low - 1)
    term2 = np.sum(np.power(-1., n) / (factorial(n) * (2 * n - 1)))

    term3 = ((1. + redshift) * np.power(wavelength[w] / lambda_limit, 1.5) -
             np.power(wavelength[w] / lambda_limit, 2.5))

    term4 = np.sum(np.array(
        [((2. * np.power(-1., n) / (factorial(n) * ((6 * n - 5) * (2 * n - 1)))) *
          ((1. + redshift) ** (2.5 - (3 * n)) *
           (wavelength[w] / lambda_limit) ** (3 * n) -
           (wavelength[w] / lambda_limit) ** 2.5))
         for n in np.arange(1, n_transitions_low)]), axis=0)

    tau_l_lls = np.zeros_like(wavelength)
    tau_l_lls[w] = n0 * ((term1 - term2) * term3 - term4)

    # Reset for short wavelength (z_l<0)
    w = slice(None, np.searchsorted(z_l, 0.))

    # Get the normalization factor at z_l=0
    tau_norm_l_igm = np.interp(0, z_l, tau_l_igm)
    tau_norm_l_lls = np.interp(0, z_l, tau_l_lls)

    # Calculate tau_l_igm & tau_l_lls, assuming cross section ~λ^2.75
    # from (O'Meara et al. 2013)
    damp_factor = (z_l[w] + 1.) ** 2.75
    tau_l_igm[w] = tau_norm_l_igm * damp_factor
    tau_l_lls[w] = tau_norm_l_lls * damp_factor

    tau_taun = np.sum(tau_n[2:n_transitions_max, :], axis=0)

    return np.exp(- tau_taun - tau_l_igm - tau_l_lls)


class Redshifting(SedModule):
    """Redshift a SED

    This module redshift a rest-frame SED. If the SED is already redshifted, an
    exception is raised.

    """

    parameter_list = {
        "redshift": (
            "cigale_list(minvalue=0.)",
            "Redshift of the objects. Leave empty to use the redshifts from the"
            " input file.",
            None
        )
    }

    def _init_code(self):
        """Compute the age of the Universe at a given redshift
        """
        self.redshift = float(self.parameters["redshift"])

        # Raise an error when applying a negative redshift. This module is
        # not for blue-shifting.
        if self.redshift < 0.:
            raise Exception(f"The redshift provided is negative "
                            f"({self.redshift}).")

        self.universe_age = age(self.redshift)
        self.luminosity_distance = luminosity_distance(self.redshift)
        # We do not define the values of the IGM attenuation component yet.
        # This is because we need the wavelength grid for that first. This
        # will be assigned on the first call.
        self.igm_attenuation = {}

    def process(self, sed):
        """Redshift the SED

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        redshift = self.redshift

        # If the SED is already redshifted, raise an error.
        if ('universe.redshift' in sed.info and
                sed.info['universe.redshift'] > 0.):
            raise Exception(f"The SED is already redshifted (z="
                            f"{sed.info['universe.redshift']}).")

        if redshift > 0.:
            # We redshift directly the SED wavelength grid
            sed.wavelength_grid *= 1. + redshift

            # We modify each luminosity contribution to keep energy constant.
            # Because the luminosities are immutable, we need to create new
            # arrays to replace them.
            invzp1 = 1. / (1. + redshift)
            sed.luminosities = {name: sed.luminosities[name] * invzp1
                                for name in sed.luminosities}
            sed.luminosity *= invzp1

        sed.add_info("universe.redshift", redshift)
        sed.add_info("universe.luminosity_distance", self.luminosity_distance,
                     unit='m')
        sed.add_info("universe.age", self.universe_age, unit='Myr')

        # We identify the right grid from the length of the wavelength array.
        # It is not completely foolproof but it is good enough. We need to do
        # that in case two models do not have the same wavelength sampling.
        # This is the case for instance if some but not all models have an AGN
        # fraction of 0.
        key = sed.wavelength_grid.size
        if key not in self.igm_attenuation:
            self.igm_attenuation[key] = igm_transmission(sed.wavelength_grid,
                                                         redshift) - 1.
        sed.add_contribution('igm', sed.wavelength_grid,
                             self.igm_attenuation[key] * sed.luminosity)
        sed.add_module(self.name, self.parameters)


# SedModule to be returned by get_module
Module = Redshifting
