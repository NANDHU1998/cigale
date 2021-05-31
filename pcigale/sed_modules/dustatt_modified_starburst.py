"""
Calzetti et al. (2000) and Leitherer et al. (2002) attenuation module
=====================================================================

This module implements the Calzetti et al. (2000) and  Leitherer et al. (2002)
attenuation formulae, adding an UV-bump and a power law.

"""

import numpy as np

from . import SedModule


def k_calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000). This formula
    is given for wavelengths between 120 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = np.zeros(len(wavelength))

    # Attenuation between 120 nm and 630 nm
    mask = (wavelength < 630)
    result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                            0.198e6 / wavelength[mask] ** 2 +
                            0.011e9 / wavelength[mask] ** 3) + 4.05

    # Attenuation between 630 nm and 2200 nm
    mask = (wavelength >= 630)
    result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05

    return result


def k_leitherer2002(wavelength):
    """Compute the Leitherer et al. (2002) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Leitherer at al. (2002). This formula
    is given for wavelengths between 91.2 nm and 180 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)

    return result


def uv_bump(wavelength, central_wave, gamma, ebump):
    """Compute the Lorentzian-like Drude profile.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    central_wave: float
        Central wavelength of the bump in nm.
    gamma: float
        Width (FWHM) of the bump in nm.
    ebump: float
        Amplitude of the bump.

    Returns
    -------
    a numpy array of floats

    """
    return (ebump * wavelength ** 2 * gamma ** 2 /
            ((wavelength ** 2 - central_wave ** 2) ** 2 +
             wavelength ** 2 * gamma ** 2))


def power_law(wavelength, delta):
    """Power law 'centered' on 550 nm..

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid in nm.
    delta: float
        The slope of the power law.

    Returns
    -------
    array of floats

    """
    return (wavelength / 550) ** delta


def a_vs_ebv(wavelength, bump_wave, bump_width, bump_ampl, power_slope):
    """Compute the complete attenuation curve A(λ)/E(B-V)*

    The Leitherer et al. (2002) formula is used between 91.2 nm and 150 nm, and
    the Calzetti et al. (2000) formula is used after 150 (we do an
    extrapolation after 2200 nm). When the attenuation becomes negative, it is
    kept to 0. This continuum is multiplied by the power law and then the UV
    bump is added.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.
    power_slope: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/E(B-V)* attenuation at each wavelength of the grid.

    """
    attenuation = np.zeros(len(wavelength))

    # Leitherer et al.
    mask = (wavelength > 91.2) & (wavelength < 150)
    attenuation[mask] = k_leitherer2002(wavelength[mask])
    # Calzetti et al.
    mask = (wavelength >= 150)
    attenuation[mask] = k_calzetti2000(wavelength[mask])
    # We set attenuation to 0 where it becomes negative
    mask = (attenuation < 0)
    attenuation[mask] = 0
    # Power law
    attenuation *= power_law(wavelength, power_slope)
    # UV bump
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl)

    # As the powerlaw slope changes E(B-V), we correct this so that the curve
    # always has the same E(B-V) as the starburst curve. This ensures that the
    # E(B-V) requested by the user is the actual E(B-V) of the curve.
    wl_BV = np.array([440., 550.])
    EBV_calz = ((k_calzetti2000(wl_BV) * power_law(wl_BV, 0.)) +
                uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    EBV = ((k_calzetti2000(wl_BV) * power_law(wl_BV, power_slope)) +
           uv_bump(wl_BV, bump_wave, bump_width, bump_ampl))
    attenuation *= (EBV_calz[1] - EBV_calz[0]) / (EBV[1] - EBV[0])

    return attenuation


def ccm(wave, Rv):
    """ Compute the complete attenuation curve A(λ)/E(B-V) for emission lines

    Using Cardelli, Clayton & Mathis (1989) MW extinction function.
    Valid from 1250 Angstroms to 3.3 microns.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths in nm.
    Rv : float
        Ratio of total to selective extinction, A_V / E(B-V).
    """

    x = 1e3 / wave

    # In the paper the condition is 0.3 < x < 1.1.
    # However setting just x <1.1 avoids to have an artificial break at 0.3
    # with something positive above 0.3 and 0 below.
    cond1 = x < 1.1
    cond2 = (x >= 1.1) & (x < 3.3)
    cond3 = (x >= 3.3) & (x < 5.9)
    cond4 = (x >= 5.9) & (x < 8.0)
    cond5 = (x >= 8.0) & (x <= 11.)
    fcond1 = lambda wn: Rv * .574 * wn**1.61 - .527 * wn**1.61
    fcond2 = lambda wn: 1.0 * (Rv * np.polyval([-.505, 1.647, -0.827, -1.718,
                                                1.137, .701, -0.609, 0.104, 1.],
                                               wn - 1.82) +
                               np.polyval([3.347, -10.805, 5.491, 11.102,
                                           -7.985, -3.989, 2.908, 1.952, 0.],
                                          wn - 1.82))
    fcond3 = lambda wn: 1.0 * (Rv * (1.752 - 0.316 * wn -
                               (0.104 / ((wn - 4.67)**2 + 0.341))) +
                               (-3.090 + 1.825 * wn +
                               (1.206 / ((wn - 4.62)**2 + 0.263))))
    fcond4 = lambda wn: 1.0 * (Rv * (1.752 - 0.316 * wn -
                                     (0.104 / ((wn - 4.67)**2 + 0.341)) +
                                     np.polyval([-0.009779, -0.04473, 0., 0.],
                                                wn - 5.9)) +
                               (-3.090 + 1.825 * wn +
                                (1.206 / ((wn - 4.62)**2 + 0.263)) +
                                np.polyval([0.1207, 0.2130, 0., 0.],
                                           wn - 5.9)))
    fcond5 = lambda wn: 1.0 * (Rv * (np.polyval([-0.070, 0.137, -0.628, -1.073],
                                                wn - 8.)) +
                               np.polyval([0.374, -0.420, 4.257, 13.670],
                                          wn - 8.))

    return np.piecewise(x, [cond1, cond2, cond3, cond4, cond5],
                        [fcond1, fcond2, fcond3, fcond4, fcond5])


def Pei92(wave, Rv=None, law='mw'):
    """ Compute the extinction curve A(λ)/E(B-V) for emission lines

    Using Pei92 (1989) MW,LMC,SMC extinction function.
    Valid from 912 Angstroms to 25 microns.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths in nm.
    Rv : float
        Ratio of total to selective extinction, A_V / E(B-V).
    law : string
          name of the extinction curve to use: 'mw','lmc' or 'smc'
    """
    wvl = wave * 1e-3
    if law.lower() == 'smc':
        if Rv is None:
            Rv = 2.93
        a_coeff = np.array([185, 27, 0.005, 0.010, 0.012, 0.03])
        wvl_coeff = np.array([0.042, 0.08, 0.22, 9.7, 18, 25])
        b_coeff = np.array([90, 5.50, -1.95, -1.95, -1.80, 0.0])
        n_coeff = np.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])

    elif law.lower() == 'lmc':
        if Rv is None:
            Rv = 3.16
        a_coeff = np.array([175, 19, 0.023, 0.005, 0.006, 0.02])
        wvl_coeff = np.array([0.046, 0.08, 0.22, 9.7, 18, 25])
        b_coeff = np.array([90, 5.5, -1.95, -1.95, -1.8, 0.0])
        n_coeff = np.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])

    elif law.lower() == 'mw':
        if Rv is None:
            Rv = 3.08
        a_coeff = np.array([165, 14, 0.045, 0.002, 0.002, 0.012])
        wvl_coeff = np.array([0.046, 0.08, 0.22, 9.7, 18, 25])
        b_coeff = np.array([90, 4.0, -1.95, -1.95, -1.8, 0.0])
        n_coeff = np.array([2.0, 6.5, 2.0, 2.0, 2.0, 2.0])

    Alambda_over_Ab = np.zeros(len(wvl))
    for a, wv, b, n in zip(a_coeff, wvl_coeff, b_coeff, n_coeff):
        Alambda_over_Ab += a / ((wvl / wv)**n + (wv / wvl)**n + b)

    # Normalise with Av
    Alambda_over_Av = (1 / Rv + 1) * Alambda_over_Ab

    # set 0 for wvl < 91.2nm
    Alambda_over_Av[wvl < 91.2 * 1e-3] = 0

    # Set 0 for wvl > 30 microns
    Alambda_over_Av[wvl > 30] = 0

    # Transform Alambda_over_Av into Alambda_over_E(B-V)
    Alambda_over_Ebv = Rv * Alambda_over_Av

    return Alambda_over_Ebv


class ModStarburstAtt(SedModule):
    """Calzetti + Leitherer attenuation module

    This module computes the dust attenuation using the
    formulae from Calzetti et al. (2000) and Leitherer et al. (2002).

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.

    """

    parameter_list = {
        "E_BV_lines": (
            "cigale_list(minvalue=0.)",
            "E(B-V)l, the colour excess of the nebular lines light for "
            "both the young and old population.",
            0.3
        ),
        "E_BV_factor": (
            "cigale_list(minvalue=0., maxvalue=1.)",
            "Reduction factor to apply on E_BV_lines to compute E(B-V)s "
            "the stellar continuum attenuation. Both young and old population "
            "are attenuated with E(B-V)s. ",
            0.44
        ),
        "uv_bump_wavelength": (
            "cigale_list(minvalue=0.)",
            "Central wavelength of the UV bump in nm.",
            217.5
        ),
        "uv_bump_width": (
            "cigale_list()",
            "Width (FWHM) of the UV bump in nm.",
            35.
        ),
        "uv_bump_amplitude": (
            "cigale_list(minvalue=0.)",
            "Amplitude of the UV bump. For the Milky Way: 3.",
            0.
        ),
        "powerlaw_slope": (
            "cigale_list()",
            "Slope delta of the power law modifying the attenuation curve.",
            0.
        ),
        "Ext_law_emission_lines": (
            "cigale_list(dtype=int, options=1 & 2 & 3)",
            "Extinction law to use for attenuating the emissio  n lines flux. "
            "Possible values are: 1, 2, 3. 1: MW, 2: LMC, 3: SMC. MW is "
            "modelled using CCM89, SMC and LMC using Pei92.",
            1
        ),
        "Rv": (
            "cigale_list()",
            "Ratio of total to selective extinction, A_V / E(B-V), "
            "for the extinction curve applied to emission lines."
            "Standard value is 3.1 for MW using CCM89, but can be changed."
            "For SMC and LMC using Pei92 the value is automatically set to "
            "2.93 and 3.16 respectively, no matter the value you write.",
            3.1
        ),
        "filters": (
            "string()",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "B_B90 & V_B90 & FUV"
        )
    }

    def _init_code(self):
        """Get the filters from the database"""

        self.ebvl = float(self.parameters["E_BV_lines"])
        self.ebv_factor = float(self.parameters["E_BV_factor"])
        self.ebvs = self.ebv_factor * self.ebvl
        self.uv_bump_wavelength = float(self.parameters["uv_bump_wavelength"])
        self.uv_bump_width = float(self.parameters["uv_bump_width"])
        self.uv_bump_amplitude = float(self.parameters["uv_bump_amplitude"])
        self.powerlaw_slope = float(self.parameters["powerlaw_slope"])
        self.ext_law_emLines = int(self.parameters["Ext_law_emission_lines"])
        self.Rv = float(self.parameters["Rv"])
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        # We cannot compute the attenuation until we know the wavelengths. Yet,
        # we reserve the object.
        self.contatt = None
        self.lineatt = {}

    def process(self, sed):
        """Add the modified Calzetti attenuation law to stellar continuum
        for old and young stellar populations. Add the MW, LM or SMC extincton
        curve to to the nebular lines and continuum for young and
        stellar populations

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wl = sed.wavelength_grid

        # Fλ fluxes (only from continuum) in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute stellar attenuation curve
        if self.contatt is None:
            self.contatt = 10 ** (-.4 * a_vs_ebv(wl, self.uv_bump_wavelength,
                                                 self.uv_bump_width,
                                                 self.uv_bump_amplitude,
                                                 self.powerlaw_slope) *
                                  self.ebvs)

        # Compute nebular attenuation curves
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            if self.ext_law_emLines == 1:
                self.lineatt['nebular'] = ccm(wl, self.Rv)
                for name, att in zip(names, ccm(linewl, self.Rv)):
                    self.lineatt[name] = att
            elif self.ext_law_emLines == 2:
                self.lineatt['nebular'] = Pei92(wl, law='smc')
                for name, att in zip(names, Pei92(linewl, law='smc')):
                    self.lineatt[name] = att
            elif self.ext_law_emLines == 3:
                self.lineatt['nebular'] = Pei92(wl, law='lmc')
                for name, att in zip(names, Pei92(linewl, law='lmc')):
                    self.lineatt[name] = att
            for k, v in self.lineatt.items():
                self.lineatt[k] = 10. ** (-.4 * v * self.ebvl)

        dust_lumin = 0.
        contribs = [contrib for contrib in sed.luminosities if
                    'absorption' not in contrib]

        for contrib in contribs:
            luminosity = sed.luminosities[contrib]
            if 'nebular' in contrib:
                attenuation_spec = luminosity * (self.lineatt['nebular'] - 1.)
            else:
                attenuation_spec = luminosity * (self.contatt - 1.)
            dust_lumin -= np.trapz(attenuation_spec, wl)

            sed.add_module(self.name, self.parameters)
            sed.add_contribution("attenuation." + contrib, wl,
                                 attenuation_spec)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (linewl, old * self.lineatt[name],
                               young * self.lineatt[name])

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + dust_lumin, True, True,
                         unit='W')
        else:
            sed.add_info("dust.luminosity", dust_lumin, True, unit='W')

        # Fλ fluxes (only from continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]),
                         unit='mag')

        sed.add_info('attenuation.E_BV_lines', self.ebvl, unit='mag')
        sed.add_info('attenuation.E_BVs', self.ebvs, unit='mag')
        sed.add_info('attenuation.E_BV_factor', self.ebv_factor)
        sed.add_info('attenuation.uv_bump_wavelength', self.uv_bump_wavelength,
                     unit='nm')
        sed.add_info('attenuation.uv_bump_width', self.uv_bump_width, unit='nm')
        sed.add_info('attenuation.uv_bump_amplitude', self.uv_bump_amplitude)
        sed.add_info('attenuation.powerlaw_slope', self.powerlaw_slope)


# SedModule to be returned by get_module
Module = ModStarburstAtt
