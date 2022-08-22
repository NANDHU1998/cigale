from functools import lru_cache

from astropy.cosmology import WMAP7 as _cosmo
from scipy.constants import parsec


@lru_cache(maxsize=None)
def luminosity_distance(redshift):
    """Computes the luminosity distance in m for a given redshift. If the
    redshift is 0, then we assume a distance of 10 pc.

    Parameter
    ---------
    redshift: scalar
        Redshift of the object

    Returns
    -------
    luminosity_distance: scalar
        Luminosity distance to the object in m
    """
    if redshift > 0:
        return _cosmo.luminosity_distance(redshift).value * 1e6 * parsec
    return 10.0 * parsec


@lru_cache(maxsize=None)
def age(redshift):
    """Computes the age of the universe in Myr at a given redshift.

    Parameter
    ---------
    redshift: scalar
        Redshift of the object

    Returns
    -------
    age: float
        Age of the universe in Myr at the corresponding redshift
    """
    return _cosmo.age(redshift).value * 1000.0
