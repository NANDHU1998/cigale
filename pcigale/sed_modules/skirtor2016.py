"""
SKIRTOR 2016 (Stalevski et al., 2016) AGN dust torus emission module
==================================================

This module implements the SKIRTOR 2016 models.

"""
from functools import lru_cache

from astropy.table import Table
import numpy as np
import pkg_resources
import scipy.constants as cst
from scipy.interpolate import interp1d

from pcigale.data import SimpleDatabase as Database
from . import SedModule

@lru_cache
def k_ext_short(ext_law):
    if ext_law == 0:
        ext_file = 'extFun_SMC.dat'
    elif ext_law == 1:
        ext_file = 'extFun_MRN.dat'
    elif ext_law == 2:
        ext_file = 'extFun_Gaskel04.dat'
    else:
        raise ValueError(f"Exinction law {ext_law} unknown.")
    ext_file = pkg_resources.resource_filename(__name__, f"curves/{ext_file}")
    curve = Table.read(ext_file, header_start=5, format='ascii')
    wl = curve['lambda'] * 1e3
    AB, AV = np.interp([440., 550.], wl, curve['ext'])

    return interp1d(wl, curve['ext'] / (AB - AV))

def k_ext(wavelength, ext_law):
    """
    Compute k(λ)=A(λ)/E(B-V) for a specified extinction law

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    ext_law: the extinction law
             0=SMC, 1=Calzetti2000, 2=Gaskell2004

    Returns
    -------
    a numpy array of floats

    """
    if ext_law == 0:
        # SMC, from Bongiorno+2012
        k = 1.39 * (wavelength * 1e-3) ** -1.2
    elif ext_law == 1:
        # Calzetti2000, from dustatt_calzleit.py
        result = np.zeros(len(wavelength))
        # Attenuation between 120 nm and 630 nm
        mask = (wavelength < 630)
        result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                                0.198e6 / wavelength[mask] ** 2 +
                                0.011e9 / wavelength[mask] ** 3) + 4.05
        # Attenuation between 630 nm and 2200 nm
        mask = (wavelength >= 630)
        result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05
        k = result
    elif ext_law == 2:
        # Gaskell+2004, from the appendix of that paper
        x = 1e3 / wavelength
        Alam_Av = np.zeros(len(wavelength))
        # Attenuation for x = 1.6 -- 3.69
        mask = (x < 3.69)
        Alam_Av[mask] = -0.8175 + 1.5848 * x[mask] - 0.3774 * x[mask]**2 \
            + 0.0296 * x[mask]**3
        # Attenuation for x = 3.69 -- 8
        mask = (x >= 3.69)
        Alam_Av[mask] = 1.3468 + 0.0087 * x[mask]
        # Set negative values to zero
        Alam_Av[Alam_Av < 0.] = 0.
        # Convert A(λ)/A(V) to A(λ)/E(B-V)
        # assuming A(B)/A(V) = 1.182 (Table 3 of Gaskell+2004)
        k = Alam_Av / 0.182
    else:
        raise KeyError("Extinction law is different from the expected ones")

    mask = np.where(wavelength < 100.)
    if mask[0].size > 0:
        k_short = k_ext_short(ext_law)(wavelength[mask])
        k[mask] = k_short * (k[mask][-1] / k_short[-1])

    return k

def disk(wl, limits, coefs):
    ss = np.searchsorted(wl, limits)
    wpl = [slice(lo, hi) for lo, hi in zip(ss[:-1], ss[1:])]

    norms = np.ones_like(coefs)
    for idx in range(1, coefs.size):
        norms[idx] = norms[idx - 1] * \
            limits[idx] ** (coefs[idx - 1] - coefs[idx])

    spectrum = np.zeros_like(wl)
    for w, coef, norm in zip(wpl, coefs, norms):
        spectrum[w] = wl[w]**coef * norm

    return spectrum * (1. / np.trapz(spectrum, x=wl))


def schartmann2005_disk(wl, delta=0.):
    limits = np.array([8., 50., 125., 10000., 1e6])
    coefs = np.array([1.0, -0.2, -1.5 + delta, -4.0])

    return disk(wl, limits, coefs)


def skirtor_disk(wl, delta=0.):
    limits = np.array([8., 10., 100., 5000., 1e6])
    coefs = np.array([0.2, -1.0, -1.5 + delta, -4.0])

    return disk(wl, limits, coefs)


class SKIRTOR2016(SedModule):
    """SKIRTOR 2016 (Stalevski et al., 2016) AGN dust torus emission


    The relative normalization of these components is handled through a
    parameter which is the fraction of the total IR luminosity due to the AGN
    so that: L_AGN = fracAGN * L_IRTOT, where L_AGN is the AGN luminosity,
    fracAGN is the contribution of the AGN to the total IR luminosity
    (L_IRTOT), i.e. L_Starburst+L_AGN.

    """

    parameter_list = {
        't': (
            "cigale_list(options=3 & 5 & 7 & 9 & 11)",
            "Average edge-on optical depth at 9.7 micron; the actual one along"
            "the line of sight may vary depending on the clumps distribution. "
            "Possible values are: 3, 5, 7, 9, and 11.",
            7
        ),
        'pl': (
            "cigale_list(options=0. & .5 & 1. & 1.5)",
            "Power-law exponent that sets radial gradient of dust density."
            "Possible values are: 0., 0.5, 1., and 1.5.",
            1.0
        ),
        'q': (
            "cigale_list(options=0. & .5 & 1. & 1.5)",
            "Index that sets dust density gradient with polar angle."
            "Possible values are:  0., 0.5, 1., and 1.5.",
            1.0
        ),
        'oa': (
            'cigale_list(options=10 & 20 & 30 & 40 & 50 & 60 & 70 & 80)',
            "Angle measured between the equatorial plan and edge of the torus. "
            "Half-opening angle of the dust-free cone is 90°-oa. "
            "Possible values are: 10, 20, 30, 40, 50, 60, 70, and 80",
            40
        ),
        'R': (
            'cigale_list(options=10 & 20 & 30)',
            "Ratio of outer to inner radius, R_out/R_in."
            "Possible values are: 10, 20, and 30",
            20
        ),
        'Mcl': (
            'cigale_list(options=0.97)',
            "fraction of total dust mass inside clumps. 0.97 means 97% of "
            "total mass is inside the clumps and 3% in the interclump dust. "
            "Possible values are: 0.97.",
            0.97
        ),
        'i': (
            'cigale_list(options=0 & 10 & 20 & 30 & 40 & 50 & 60 & 70 & 80 & 90)',
            "inclination, i.e. viewing angle, position of the instrument "
            "w.r.t. the AGN axis. i=[0, 90°-oa): face-on, type 1 view; "
            "i=[90°-oa, 90°]: edge-on, type 2 view. "
            "Possible values are: 0, 10, 20, 30, 40, 50, 60, 70, 80, and 90.",
            30
        ),
        'disk_type': (
            'integer(min=0, max=1)',
            "Disk spectrum: 0 for the regular Skirtor spectrum, 1 for the "
            "Schartmann (2005) spectrum.",
            0
        ),
        'delta': (
            'cigale_list()',
            "Power-law of index δ modifying the optical slop of the disk. "
            "Negative values make the slope steeper where as positive values "
            "make it shallower.",
            0.
        ),
        'fracAGN': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "AGN fraction.",
            0.1
        ),
        'lambda_fracAGN': (
            'string()',
            'Wavelength range in microns where to compute the AGN fraction. '
            'Note that it includes all the components, not just dust emission. '
            'To use the the total dust luminosity set to 0/0.',
            "0/0"
        ),
        'law': (
            'cigale_list(dtype=int, options=0 & 1 & 2)',
            "Extinction law of the polar dust: "
            "0 (SMC), 1 (Calzetti 2000), or 2 (Gaskell et al. 2004)",
            0
        ),
        'EBV': (
            'cigale_list(minvalue=0.)',
            "E(B-V) for the extinction in the polar direction in magnitudes.",
            0.1
        ),
        'temperature': (
            'cigale_list(minvalue=0.)',
            "Temperature of the polar dust in K.",
            100.
        ),
        "emissivity": (
            "cigale_list(minvalue=0.)",
            "Emissivity index of the polar dust.",
            1.6
        )
    }

    def _init_code(self):
        """Get the template set out of the database"""
        self.t = int(self.parameters["t"])
        self.pl = float(self.parameters["pl"])
        self.q = float(self.parameters["q"])
        self.oa = int(self.parameters["oa"])
        self.R = int(self.parameters["R"])
        self.Mcl = float(self.parameters["Mcl"])
        self.i = int(self.parameters["i"])
        self.disk_type = int(self.parameters["disk_type"])
        self.delta = float(self.parameters["delta"])
        self.fracAGN = float(self.parameters["fracAGN"])
        if self.fracAGN == 1.:
            raise ValueError("AGN fraction is exactly 1. Behaviour undefined.")
        lambda_fracAGN = str(self.parameters["lambda_fracAGN"]).split('/')
        self.lambdamin_fracAGN = float(lambda_fracAGN[0]) * 1e3
        self.lambdamax_fracAGN = float(lambda_fracAGN[1]) * 1e3
        if (self.lambdamin_fracAGN < 0
                or self.lambdamin_fracAGN > self.lambdamax_fracAGN):
            raise ValueError("lambda_fracAGN incorrect. Constrain "
                             f"0 < {self.lambdamin_fracAGN} < "
                             f"{self.lambdamax_fracAGN} not respected.")
        self.law = int(self.parameters["law"])
        self.EBV = float(self.parameters["EBV"])
        self.temperature = float(self.parameters["temperature"])
        self.emissivity = float(self.parameters["emissivity"])

        with Database("skirtor2016") as db:
            self.SKIRTOR2016 = db.get(t=self.t, pl=self.pl, q=self.q,
                                      oa=self.oa, R=self.R, Mcl=self.Mcl,
                                      i=self.i)
            AGN1 = db.get(t=self.t, pl=self.pl, q=self.q, oa=self.oa, R=self.R,
                          Mcl=self.Mcl, i=0)
        # Re-normalize AGN1, to be consistent with the intrinsic AGN
        # luminosity of SKIRTOR2016
        AGN1.disk *= AGN1.norm / self.SKIRTOR2016.norm

        # We offer the possibility to modify the change the disk spectrum.
        # To ensure the conservation of the energy we first normalize the new
        # spectrum to that of an AGN 1 from skirtor. Then we multiply by the
        # ratio of the emission spectrum of the AGN model to that of an AGN 1.
        # This is done so that the “absorption curve” is reproduced. The exact
        # distribution of the energy does not appear to have a strong effect on
        # the actual absorbed luminosity, probably because very little radiation
        # can escape the torus
        if self.disk_type == 0:
            disk = skirtor_disk(self.SKIRTOR2016.wl, delta=self.delta)
        elif self.disk_type == 1:
            disk = schartmann2005_disk(self.SKIRTOR2016.wl, delta=self.delta)
        else:
            raise ValueError("The parameter disk_type must be 0 or 1.")
        disk *= np.trapz(AGN1.disk, x=AGN1.wl)

        self.SKIRTOR2016.disk = np.nan_to_num(disk * self.SKIRTOR2016.disk /
                                              AGN1.disk)
        AGN1.disk = disk

        # Calculate the extinction
        ext_fac = 10**(-.4 * k_ext(self.SKIRTOR2016.wl, self.law) * self.EBV)

        # Calculate the new AGN SED shape after extinction
        # The direct and scattered components (line-of-sight) are extincted for
        # type-1 AGN
        # Keep the direct and scatter components for type-2
        if self.i <= (90. - self.oa):
            self.SKIRTOR2016.disk *= ext_fac

        # Calculate the total extincted luminosity averaged over all directions
        # The computation is non-trivial as the disk emission is anisotropic:
        # L(θ, λ) = A(λ)×cosθ×(1+2×cosθ). Given that L(θ=0, λ) = A(λ)×3, we get
        # A = L(θ=0, λ)/3. Then Lpolar = 2×∭L(θ, λ)×(1-ext_fact(λ))×sinθ dφdθdλ,
        # woth φ between 0 and 2π, θ between 0 and π/2-OA, and λ the wavelengths
        # Integrating over φ,
        # Lpolar = 4π/3×∬L(θ=0, λ)×(1-ext_fact(λ))×cosθ×(1+2×cosθ)×sinθ dθdλ.
        # Now doing the usual integration over θ,
        # Lpolar = 4π/3×∫L(θ=0, λ)×(1-ext_fact(λ))×[7/6-1/2×sin²OA-2/3×sin³OA] dλ.
        # Now a critical point is that the SKIRTOR models are provided in flux
        # and are multiplied by 4πd². Because the emission is anisotropic, we
        # need to redivide by 4π to get the correct luminosity for a given θ,
        # hence Lpolar = [7/18-1/6×sin²OA-2/9×sin³OA]×∫L(θ=0, λ)×(1-ext_fact(λ)) dλ.
        # Integrating over λ gives the bolometric luminosity
        sin_oa = np.sin(np.deg2rad(self.oa))
        l_ext = (7. / 18. - sin_oa**2. / 6. - sin_oa**3 * 2. / 9.) * \
            np.trapz(AGN1.disk * (1. - ext_fac), x=AGN1.wl)

        # Casey (2012) modified black body model
        c = cst.c * 1e9
        lambda_0 = 200e3
        conv = c / self.SKIRTOR2016.wl ** 2.
        hc_lkt = cst.h * c / (self.SKIRTOR2016.wl * cst.k * self.temperature)
        err_settings = np.seterr(over='ignore')  # ignore exp overflow
        blackbody = conv * \
            (1. - np.exp(-(lambda_0 / self.SKIRTOR2016.wl) ** self.emissivity)) * \
            (c / self.SKIRTOR2016.wl) ** 3. / (np.exp(hc_lkt) - 1.)
        np.seterr(**err_settings)  # Restore the previous settings
        blackbody *= l_ext / np.trapz(blackbody, x=self.SKIRTOR2016.wl)

        # Add the black body to dust thermal emission
        self.SKIRTOR2016.dust += blackbody

        # Normalize direct, scatter, and thermal components
        norm = 1. / np.trapz(self.SKIRTOR2016.dust, x=self.SKIRTOR2016.wl)
        self.SKIRTOR2016.dust *= norm
        self.SKIRTOR2016.disk *= norm

        # Integrate AGN luminosity for different components
        self.lumin_disk = np.trapz(self.SKIRTOR2016.disk, x=self.SKIRTOR2016.wl)

        if self.lambdamin_fracAGN < self.lambdamax_fracAGN:
            w = np.where((self.SKIRTOR2016.wl >= self.lambdamin_fracAGN) &
                         (self.SKIRTOR2016.wl <= self.lambdamax_fracAGN))
            wl = np.hstack([self.lambdamin_fracAGN, self.SKIRTOR2016.wl[w],
                            self.lambdamax_fracAGN])
            spec = np.interp(wl, self.SKIRTOR2016.wl,
                             self.SKIRTOR2016.dust + self.SKIRTOR2016.disk)
            self.AGNlumin = np.trapz(spec, x=wl)
        elif (self.lambdamin_fracAGN == 0.) & (self.lambdamax_fracAGN == 0.):
            self.AGNlumin = 1.
        elif self.lambdamin_fracAGN == self.lambdamax_fracAGN:
            self.AGNlumin = np.interp(self.lambdamin_fracAGN,
                                      self.SKIRTOR2016.wl,
                                      self.SKIRTOR2016.dust +
                                      self.SKIRTOR2016.disk)
        # Store the SED wavelengths
        self.wl = None

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True, unit='W')
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info('agn.t', self.t)
        sed.add_info('agn.pl', self.pl)
        sed.add_info('agn.q', self.q)
        sed.add_info('agn.oa', self.oa, unit='deg')
        sed.add_info('agn.R', self.R)
        sed.add_info('agn.Mcl', self.Mcl)
        sed.add_info('agn.i', self.i, unit='deg')
        sed.add_info('agn.fracAGN', self.fracAGN)
        sed.add_info('agn.law', self.law)
        sed.add_info('agn.EBV', self.EBV, unit='mag')
        sed.add_info('agn.temperature', self.temperature, unit='K')
        sed.add_info('agn.emissivity', self.emissivity)
        sed.add_info('agn.disk_type', self.disk_type)
        sed.add_info('agn.delta', self.delta)

        # Compute the AGN luminosity
        if self.lambdamin_fracAGN < self.lambdamax_fracAGN:
            if self.wl is None:
                w = np.where((sed.wavelength_grid >= self.lambdamin_fracAGN) &
                             (sed.wavelength_grid <= self.lambdamax_fracAGN))
                self.wl = np.hstack([self.lambdamin_fracAGN,
                                     sed.wavelength_grid[w],
                                     self.lambdamax_fracAGN])
            spec = np.interp(self.wl, sed.wavelength_grid, sed.luminosity)
            scale = np.trapz(spec, x=self.wl) / self.AGNlumin
        elif (self.lambdamin_fracAGN == 0.) and (self.lambdamax_fracAGN == 0.):
            scale = luminosity
        elif self.lambdamin_fracAGN == self.lambdamax_fracAGN:
            scale = np.interp(self.lambdamin_fracAGN, sed.wavelength_grid,
                              sed.luminosity) / self.AGNlumin

        agn_power = scale * (1. / (1. - self.fracAGN) - 1.)
        lumin_dust = agn_power
        lumin_disk = agn_power * np.trapz(self.SKIRTOR2016.disk,
                                          x=self.SKIRTOR2016.wl)

        sed.add_info('agn.dust_luminosity', lumin_dust, True, unit='W')
        sed.add_info('agn.disk_luminosity', lumin_disk, True, unit='W')
        sed.add_info('agn.luminosity', lumin_dust + lumin_disk, True, unit='W')

        sed.add_contribution('agn.SKIRTOR2016_dust', self.SKIRTOR2016.wl,
                             agn_power * self.SKIRTOR2016.dust)
        sed.add_contribution('agn.SKIRTOR2016_disk', self.SKIRTOR2016.wl,
                             agn_power * self.SKIRTOR2016.disk)


# SedModule to be returned by get_module
Module = SKIRTOR2016
