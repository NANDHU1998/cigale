"""
Fritz et al. (2006) AGN dust torus emission module
==================================================

This module implements the Fritz et al. (2006) models.

"""
from functools import lru_cache

from astropy.table import Table
import numpy as np
import pkg_resources
import scipy.constants as cst
from scipy.interpolate import interp1d

from pcigale.data import SimpleDatabase as Database
from . import SedModule

__category__ = "AGN"


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


class Fritz2006(SedModule):
    """Fritz et al. (2006) AGN dust torus emission

    The AGN emission is computed from the library of Fritz et al. (2006) from
    which all of the models are available. They take into account two emission
    components linked to the AGN. The first one is the isotropic emission of
    the central source, which is assumed to be point-like. This emission is a
    composition of power laws with variable indices, in the wavelength range of
    0.001-20 microns. The second one is the thermal and scattering dust torus
    emission. The conservation of the energy is always verified within 1% for
    typical solutions, and up to 10% in the case of very high optical depth and
    non-constant dust density. We refer the reader to Fritz et al. (2006) for
    more information on the library.

    The relative normalization of these components is handled through a
    parameter which is the fraction of the total IR luminosity due to the AGN
    so that: L_AGN = fracAGN * L_IRTOT, where L_AGN is the AGN luminosity,
    fracAGN is the contribution of the AGN to the total IR luminosity
    (L_IRTOT), i.e. L_Starburst+L_AGN.

    """

    parameter_list = {
        'r_ratio': (
            "cigale_list(options=10. & 30. & 60. & 100. & 150.)",
            "Ratio of the maximum to minimum radii of the dust torus. "
            "Possible values are: 10, 30, 60, 100, 150.",
            60.
        ),
        'tau': (
            "cigale_list(options=0.1 & 0.3 & 0.6 & 1.0 & 2.0 & 3.0 & 6.0 & "
            "10.0)",
            "Optical depth at 9.7 microns. "
            "Possible values are: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.",
            1.0
        ),
        'beta': (
            "cigale_list(options=-1.00 & -0.75 & -0.50 & -0.25 & 0.00)",
            "Beta. Possible values are: -1.00, -0.75, -0.50, -0.25, 0.00.",
            -0.50
        ),
        'gamma': (
            'cigale_list(options=0.0 & 2.0 & 4.0 & 6.0)',
            "Gamma. Possible values are: 0.0, 2.0, 4.0, 6.0.",
            4.0
        ),
        'opening_angle': (
            'cigale_list(options=60. & 100. & 140.)',
            "Full opening angle of the dust torus (Fig 1 of Fritz 2006). "
            "Possible values are: 60., 100., 140.",
            100.
        ),
        'psy': (
            'cigale_list(options=0.001 & 10.100 & 20.100 & 30.100 & 40.100 & '
            '50.100 & 60.100 & 70.100 & 80.100 & 89.990)',
            "Angle between equatorial axis and line of sight. "
            "Psy = 90◦ for type 1 and Psy = 0° for type 2. Possible values "
            "are: 0.001, 10.100, 20.100, 30.100, 40.100, 50.100, 60.100, "
            "70.100, 80.100, 89.990.",
            50.100
        ),
        'disk_type': (
            'integer(min=0, max=1)',
            "Disk spectrum: 0 for the regular Skirtor spectrum, 1 for the "
            "Schartmann (2005) spectrum.",
            1
        ),
        'delta': (
            'cigale_list()',
            "Power-law of index δ modifying the optical slop of the disk. "
            "Negative values make the slope steeper where as positive values "
            "make it shallower.",
            -0.36
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
            0.03
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
        self.r_ratio = float(self.parameters["r_ratio"])
        self.tau = float(self.parameters["tau"])
        self.beta = float(self.parameters["beta"])
        self.gamma = float(self.parameters["gamma"])
        self.opening_angle = (180. - self.parameters["opening_angle"]) / 2.
        self.psy = float(self.parameters["psy"])
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

        with Database("fritz2006") as db:
            self.fritz2006 = db.get(r_ratio=self.r_ratio, tau=self.tau,
                                    beta=self.beta, gamma=self.gamma,
                                    opening_angle=self.opening_angle,
                                    psy=self.psy)
            AGN1 = db.get(r_ratio=self.r_ratio, tau=self.tau, beta=self.beta,
                          gamma=self.gamma, opening_angle=self.opening_angle,
                          psy=89.990)
        # Re-normalize AGN1, to be consistent with the intrinsic AGN
        # luminosity of fritz2006
        AGN1.disk *= AGN1.norm / self.fritz2006.norm

        # We offer the possibility to modify the change the disk spectrum.
        # To ensure the conservation of the energy we first normalize the new
        # spectrum to that of an AGN 1 from skirtor. Then we multiply by the
        # ratio of the emission spectrum of the AGN model to that of an AGN 1.
        # This is done so that the “absorption curve” is reproduced. The exact
        # distribution of the energy does not appear to have a strong effect on
        # the actual absorbed luminosity, probably because very little radiation
        # can escape the torus
        if self.disk_type == 0:
            disk = skirtor_disk(self.fritz2006.wl, delta=self.delta)
        elif self.disk_type == 1:
            disk = schartmann2005_disk(self.fritz2006.wl, delta=self.delta)
        else:
            raise ValueError("The parameter disk_type must be 0 or 1.")
        disk *= np.trapz(AGN1.disk, x=AGN1.wl)

        err_settings = np.seterr(invalid='ignore')  # ignore 0/0 warning
        self.fritz2006.disk = np.nan_to_num(disk * self.fritz2006.disk /
                                              AGN1.disk)
        np.seterr(**err_settings)  # Restore the previous settings
        AGN1.disk = disk

        # Calculate the extinction
        ext_fac = 10**(-.4 * k_ext(self.fritz2006.wl, self.law) * self.EBV)

        # Calculate the new AGN SED shape after extinction
        # The direct and scattered components (line-of-sight) are extincted for
        # type-1 AGN
        # Keep the direct and scatter components for type-2
        if self.psy > (90 - self.opening_angle):
            self.fritz2006.disk *= ext_fac

        # Calculate the total extincted luminosity averaged over all directions
        l_ext = (1. - np.cos(np.deg2rad(self.opening_angle))) * \
            np.trapz(AGN1.disk * (1. - ext_fac), x=AGN1.wl)

        # Casey (2012) modified black body model
        c = cst.c * 1e9
        lambda_0 = 200e3
        conv = c / self.fritz2006.wl ** 2.
        hc_lkt = cst.h * c / (self.fritz2006.wl * cst.k * self.temperature)
        err_settings = np.seterr(over='ignore')  # ignore exp overflow
        blackbody = conv * \
            (1. - np.exp(-(lambda_0 / self.fritz2006.wl) ** self.emissivity)) * \
            (c / self.fritz2006.wl) ** 3. / (np.exp(hc_lkt) - 1.)
        np.seterr(**err_settings)  # Restore the previous settings
        blackbody *= l_ext / np.trapz(blackbody, x=self.fritz2006.wl)

        # Add the black body to dust thermal emission
        self.fritz2006.dust += blackbody

        # Normalize direct, scatter, and thermal components
        norm = 1. / np.trapz(self.fritz2006.dust, x=self.fritz2006.wl)
        self.fritz2006.dust *= norm
        self.fritz2006.polar_dust = blackbody * norm
        self.fritz2006.disk *= norm

        # Integrate AGN luminosity for different components
        self.lumin_disk = np.trapz(self.fritz2006.disk, x=self.fritz2006.wl)
        self.lumin_polar_dust = np.trapz(self.fritz2006.polar_dust,
                                         x=self.fritz2006.wl)

        # Intrinsic (de-reddened) AGN luminosity from the central source
        self.lumin_intrin_disk = np.trapz(AGN1.disk, x=AGN1.wl) * norm

        # Calculate Lλ(2500 Å) and convert to Lν
        self.l_agn_2500A = np.interp(250, AGN1.wl, AGN1.disk) * norm
        self.l_agn_2500A *= 250.0 ** 2.0 / c

        if self.lambdamin_fracAGN < self.lambdamax_fracAGN:
            w = np.where((self.fritz2006.wl >= self.lambdamin_fracAGN) &
                         (self.fritz2006.wl <= self.lambdamax_fracAGN))
            wl = np.hstack([self.lambdamin_fracAGN, self.fritz2006.wl[w],
                            self.lambdamax_fracAGN])
            spec = np.interp(wl, self.fritz2006.wl,
                             self.fritz2006.dust + self.fritz2006.disk)
            self.AGNlumin = np.trapz(spec, x=wl)
        elif (self.lambdamin_fracAGN == 0.) & (self.lambdamax_fracAGN == 0.):
            self.AGNlumin = 1.
        elif self.lambdamin_fracAGN == self.lambdamax_fracAGN:
            self.AGNlumin = np.interp(self.lambdamin_fracAGN,
                                      self.fritz2006.wl,
                                      self.fritz2006.dust +
                                      self.fritz2006.disk)
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
        sed.add_info('agn.r_ratio', self.r_ratio)
        sed.add_info('agn.tau', self.tau)
        sed.add_info('agn.beta', self.beta)
        sed.add_info('agn.gamma', self.gamma)
        sed.add_info('agn.opening_angle', self.parameters["opening_angle"],
                     unit='deg')
        sed.add_info('agn.psy', self.psy, unit='deg')
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
        lumin_disk = agn_power * self.lumin_disk
        lumin_polar_dust = agn_power * self.lumin_polar_dust
        lumin_torus = agn_power - lumin_polar_dust

        # power_accretion is the intrinsic disk luminosity
        power_accretion = agn_power * self.lumin_intrin_disk
        l_agn_2500A = agn_power * self.l_agn_2500A

        sed.add_info('agn.total_dust_luminosity', lumin_dust, True, unit='W')
        sed.add_info('agn.polar_dust_luminosity', lumin_polar_dust, True, unit='W')
        sed.add_info('agn.torus_dust_luminosity', lumin_torus, True, unit='W')
        sed.add_info('agn.disk_luminosity', lumin_disk, True, unit='W')
        sed.add_info('agn.luminosity', lumin_dust + lumin_disk, True, unit='W')
        sed.add_info('agn.accretion_power', power_accretion, True, unit='W')
        sed.add_info('agn.intrin_Lnu_2500A_30deg', l_agn_2500A, True, unit='W/Hz')

        sed.add_contribution('agn.fritz2006_torus', self.fritz2006.wl,
                             agn_power * (self.fritz2006.dust -
                                          self.fritz2006.polar_dust))
        sed.add_contribution('agn.fritz2006_polar_dust', self.fritz2006.wl,
                             agn_power * (self.fritz2006.polar_dust) )
        sed.add_contribution('agn.fritz2006_disk', self.fritz2006.wl,
                             agn_power * self.fritz2006.disk)


# SedModule to be returned by get_module
Module = Fritz2006
