import numpy as np
from numpy.core.multiarray import interp  # Compiled version
from scipy.constants import c, pi

# Cache dictionaries
dx_cache = {}
best_grid_cache = {}
x_cache = {}


def lambda_to_nu(wavelength):
    """Convert wavelength (nm) to frequency (Hz)

    Parameters
    ----------
    wavelength: float or array of floats
        The wavelength(s) in nm.

    Returns
    -------
    nu: float or array of floats
        The frequency(ies) in Hz.

    """
    return c / (wavelength * 1.e-9)


def nu_to_lambda(frequency):
    """Convert frequency (Hz) to wavelength (nm)

    Parameters
    ----------
    frequency: float or numpy.array of floats
        The frequency(ies) in Hz.

    Returns
    -------
    wavelength: float or numpy.array of floats
        The wavelength(s) in nm.

    """
    return 1.e-9 * c / frequency


def luminosity_to_flux(luminosity, dist):
    """
    Convert a luminosity (or luminosity density) to a flux (or flux density).

    F = L / (4πDl2)

    Parameters
    ----------
    luminosity: float or array of floats
        Luminosity (typically in W) or luminosity density (W/nm or W/Hz).
    dist: float
        Luminosity distance of the object in metres

    Returns
    -------
    flux: float or array of floats
        The flux (typically in W/m²) of flux density (W/m²/nm or W/m²/Hz).

    """

    return luminosity * (1. / (4. * pi * dist * dist))


def lambda_flambda_to_fnu(wavelength, flambda):
    """
    Convert a Fλ vs λ spectrum to Fν vs λ

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in nm.
    flambda: list-like of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    Returns
    -------
    fnu: array of floats
        The Fν flux density in mJy (or the Lν luminosity density in
        1.e-29 W/Hz).

    """
    wavelength = np.array(wavelength, dtype=float)
    flambda = np.array(flambda, dtype=float)

    # Factor 1e+29 is to switch from W/m²/Hz to mJy
    # Factor 1e-9 is to switch from nm to m (only one because the other nm
    # wavelength goes with the Fλ in W/m²/nm).
    fnu = 1e+29 * 1e-9 * flambda * wavelength * wavelength / c

    return fnu


def lambda_fnu_to_flambda(wavelength, fnu):
    """
    Convert a Fν vs λ spectrum to Fλ vs λ

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in nm.
    fnu: list-like of floats
        The Fν flux density in mJy (of the  Lν luminosity density in
        1.e-29 W/Hz).

    Returns
    -------
    flambda: array of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    """
    wavelength = np.array(wavelength, dtype=float)
    fnu = np.array(fnu, dtype=float)

    # Factor 1e-29 is to switch from Jy to W/m²/Hz
    # Factor 1e+9 is to switch from m to nm
    flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c

    return flambda


def redshift_spectrum(wavelength, flux, redshift, is_fnu=False):
    """Redshit a spectrum

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength in nm.
    flux: array like of floats
        The flux or luminosity density.
    redshift: float
        The redshift.
    is_fnu: boolean
        If false (default) the flux is a Fλ density in W/m²/nm (or a Lλ
        luminosity density in W/nm). If true, the flux is a Fν density in mJy
        (or a Lν luminosity density in 1.e-29 W/Hz).

    Results
    -------
    wavelength, flux: tuple of numpy arrays of floats
        The redshifted spectrum with the same kind of flux (or luminosity)
        density as the input.

    """
    wavelength = np.array(wavelength, dtype=float)
    flux = np.array(flux, dtype=float)
    redshift = float(redshift)

    if redshift < 0:
        redshift_factor = 1. / (1. - redshift)
    else:
        redshift_factor = 1. + redshift

    if is_fnu:
        # Switch to Fλ
        flux = lambda_fnu_to_flambda(wavelength, flux)

    wavelength *= redshift_factor
    flux /= redshift_factor

    if is_fnu:
        # Switch back to Fλ
        flux = lambda_flambda_to_fnu(wavelength, flux)

    return wavelength, flux


def memoise_1var(f):
    """
    Memoisation helper for a function taking 1 numpy array as input. As it is
    not hashable, we cannot use the standard memoisation function. Here we
    we use the array size, minimum, and maximum values as hashes.

    Parameters
    ----------
    f: function
        Function to memoise.

    Returns
    -------
    memoise_helper: function
        Meomoised best_grid function

    """
    memo = {}

    def memoise_helper(x):
        sx = x.size
        minx = x[0]
        maxx = x[-1]
        if (sx, minx, maxx) not in memo:
            memo[(sx, minx, maxx)] = f(x)
        return memo[(sx, minx, maxx)]
    return memoise_helper


def memoise_2var(f):
    """
    Memoisation helper for a function taking 2 numpy arrays as input. As they
    are not hashable, we cannot use the standard memoisation function. Here
    we use the array sizes, minimum, and maximum values as hashes.

    Parameters
    ----------
    f: function
        Function to memoise.

    Returns
    -------
    memoise_helper: function
        Meomoised best_grid function

    """
    memo = {}

    def memoise_helper(x, y):
        sx = x.size
        minx = x[0]
        maxx = x[-1]
        sy = y.size
        miny = y[0]
        maxy = y[-1]
        if (sx, minx, maxx, sy, miny, maxy) not in memo:
            memo[(sx, minx, maxx, sy, miny, maxy)] = f(x, y)
        return memo[(sx, minx, maxx, sy, miny, maxy)]
    return memoise_helper


def best_grid(wavelengths1, wavelengths2, key):
    """
    Return the best wavelength grid to regrid to arrays

    Considering the two wavelength grids passed in parameters, this function
    compute the best new grid that will be used to regrid the two spectra
    before combining them. We do not use np.unique as it is much slowe than
    finding the unique elements by hand.

    Parameters
    ----------
    wavelengths1, wavelengths2: array of floats
        The wavelength grids to be 'regridded'.
    key: tuple
        Key to key the results in cache.

    Returns
    -------
    new_grid: array of floats
        Array containing all the wavelengths found in the input arrays.

    """

    if key in best_grid_cache:
        return best_grid_cache[key]
    wl = np.concatenate((wavelengths1, wavelengths2))
    wl.sort(kind='mergesort')
    flag = np.ones(len(wl), dtype=bool)
    np.not_equal(wl[1:], wl[:-1], out=flag[1:])
    best_grid_cache[key] = wl[flag]
    return wl[flag]


@memoise_2var
def diff_wl(wavelengths1, wavelengths2):
    """
    Memoised version of numpy.setdiff1d. It is used to find the values that
    are in wavelengths1 but not in wavelengths2.

    Parameters
    ----------
    wavelengths1, wavelengths2: array of floats
        The wavelength grids to be compared

    Returns
    -------
    setdiff1d: array
        Array containing the wavelengths in wavelengths1 but not in
        wavelengths2.

    """

    return np.setdiff1d(wavelengths1, wavelengths2, assume_unique=True)


@memoise_1var
def argsort_wl(wavelengths):
    """
    Memoised version of numpy.argsort. It is used to sort the wavelengths.

    Parameters
    ----------
    wavelengths: array
        Array containing the wavelengths to be sorted.

    Returns
    -------
    argsort: array
        Array containing the sorting indices.

    """

    return np.argsort(wavelengths,
                      kind='mergesort')


def interpolate_lumin(wl, lumin, wl_new, lumin_new, name_new):
    """
    Procedure to interpolate the luminosity components given the wavelengths
    grid and the luminosity of a new component.

    Parameters
    ----------
    wl: 1D array
        Wavelengths grid of the luminosity components.
    lumin: 2D array
        Luminosity components. The first index gives a given component and the
        second gives the wavelength.
    wl_new: 1D array
        Wavelengths grid of he new component.
    lumin_new: 1D array
        Luminosity of the new component.

    Returns
    -------
    wl_best: 1D array
        The best merged wavelengths grid
    lumin_out: 2D array
        New luminosities array resampled if necessary to the best merge
        wavelengths grid, and including the new component.

    """
    # First we remove from the wavelengths grid of the new component the
    # wavelengths that are already present for the other components. We do
    # that to avoid having the interpolate on a wavelength for which we
    # already know the answer.
    wl_unique = diff_wl(wl_new, wl)

    # If the new component has non already existing wavelengths then we
    # interpolate on these new wavelengths and only those ones. The first
    # lumin.shape[0] elements are the same as the input luminosity components.
    # We only carry out the interpolation of the new components. Finally, we
    # reorder the output array with increasing wavelength.
    if wl_unique.size > 0:
        # Output 2D array so we interpolate all components at once.
        lumin_out = np.zeros((len(lumin), wl.size + wl_unique.size))

        for i, component in enumerate(lumin):
            lumin_out[i, :wl.size] = lumin[component]

        # We interpolate only on the wavelengths where the components are
        # already defined.
        w = np.where((wl_unique > wl[0]) & (wl_unique < wl[-1]))
        lumin_out[:, wl.size + w[0]] = quick_interp_lum(wl_unique[w], wl,
                                                        lumin_out)

        wl_best = np.concatenate((wl, wl_unique))
        s = argsort_wl(wl_best)
        wl_best = wl_best[s]
        lumin_out = np.take(lumin_out, s, axis=-1)
        luminosities = {name: lumin_out[i, :] for i, name in enumerate(lumin)}
    else:
        wl_best = wl
        luminosities = lumin.copy()

    # We interpolate the new component on the new merged wavelength grid.
    luminosities[name_new] = interp(wl_best, wl_new, lumin_new, left=0.,
                                    right=0.)

    return (wl_best, luminosities)


def flux_trapz(y, x, key):
    """
    Light weight trapezoidal rule used to compute the flux in filters. It
    assumes that all the x and y input arrays are already numpy arrays to save
    time. Also the width between x elements are saved in cache using the "key"
    parameter as for a given x sampling it will always be the same. We do not
    compute they key ourselves because we do not have a proper way to hash it.
    Using x[0], x[-1], and x.size is not sufficient as if the redshift is small
    it is very likely that not of these elements will change. However the
    calling function has this knowledge and actually uses them to get the
    resampled filters from the cache. That way the two cache are sure to remain
    consistent.

    Parameters
    ----------
    y: 1D array
        The values to be integrated, typically fluxes
    x: 1D array
        Sampling of y, typically wavelengths
    key: tuple
        Contains the size of the x array, the name of the filter, and the
        redshift. Those elements point to a unique x grid. This is used to
        cache some computations are the x sampling will not change.

    Returns
    -------
    flux_trapz: float
        integral(y, dx)
    """
    if key in dx_cache:
        dx = dx_cache[key]
    else:
        dx = np.diff(x)
        dx_cache[key] = dx
    return np.dot(dx, y[1:] + y[:-1]) * .5


def quick_interp_lum(x_new, x, y):
    """
    Light weight interpolation function to interpolate luminosities on a new
    wavelength grid. It assumes that all vallues are within the range of
    definition. Also a number of quantities are cached in a dictionary in
    order to avoid recomputing them at each call. The key is a tuple of the
    size of the interpolation point, the size of the original x array and the
    shape of the original y array. It should not be called for any other
    purpose at it relies on strong assumptions that may be wrong interwise.

    Parameters
    ----------
    x_new : 1D array
        Wavelengths where the luminosities array must be interpolated
    x: 1D array
        Wavelengths of the luminosities array
    y: 2D array
        Luminosities array

    Returns
    -------
    y_new: 2D array
        Interpolated luminosities array
    """
    key = (x_new.size, x.size, y.shape)
    if key in x_cache:
        lo, hi, frac_x = x_cache[key]
    else:
        hi = np.searchsorted(x, x_new)
        # Clip them so that they are at least 1.
        # Removes mis-interpolation of x_new[n] = x[0]
        hi = hi.clip(1, len(x) - 1).astype(int)
        lo = hi - 1
        frac_x = (x_new - x[lo]) / (x[hi] - x[lo])
        x_cache[key] = (lo, hi, frac_x)

    y_lo = np.take(y, lo, axis=-1)
    y_hi = np.take(y, hi, axis=-1)

    return y_lo + (y_hi - y_lo) * frac_x
