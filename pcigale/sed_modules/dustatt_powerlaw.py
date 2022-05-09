"""
Charlot and Fall (2000) power law attenuation module
====================================================

This module implements the attenuation based on a power law as defined
in Charlot and Fall (2000) with a UV bump added.

"""

import numpy as np

from . import SedModule

__category__ = "dust attenuation"


def power_law(wavelength, delta):
    """Compute the power law (λ / λv)^δ

    Parameters
    ----------
    wavelength: array of float
        Wavelength grid in nm.
    delta: float
        Power law slope.

    Returns
    -------
    a numpy array of floats

    """
    wave = np.array(wavelength)
    return (wave / 550) ** delta


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


def alambda_av(wavelength, delta, bump_wave, bump_width, bump_ampl):
    """Compute the complete attenuation curve A(λ)/Av

    The continuum is a power law (λ / λv) ** δ to which is added a UV bump.
    Over the Lyman continuum, there is no attenuation.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    delta: float
        Slope of the power law.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/Av attenuation at each wavelength of the grid.

    """
    wave = np.array(wavelength)

    attenuation = power_law(wave, delta)
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl)

    # Lyman continuum not attenuated.
    attenuation[wavelength <= 91.2] = 0.

    return attenuation


class PowerLawAtt(SedModule):
    """Power law attenuation module

    This module computes the attenuation using a power law
    as defined in Charlot and Fall (2000).

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution.

    """

    parameter_list = {
        "Av_young": (
            "cigale_list(minvalue=0.)",
            "V-band attenuation of the young population.",
            1.
        ),
        "Av_old_factor": (
            "cigale_list(minvalue=0., maxvalue=1.)",
            "Reduction factor for the V-band attenuation of the old "
            "population compared to the young one (<1).",
            0.44
        ),
        "uv_bump_wavelength": (
            "cigale_list(minvalue=0.)",
            "Central wavelength of the UV bump in nm.",
            217.5
        ),
        "uv_bump_width": (
            "cigale_list(minvalue=0.)",
            "Width (FWHM) of the UV bump in nm.",
            35.
        ),
        "uv_bump_amplitude": (
            "cigale_list(minvalue=0.)",
            "Amplitude of the UV bump. For the Milky Way: 0.75",
            0.
        ),
        "powerlaw_slope": (
            "cigale_list()",
            "Slope delta of the power law continuum.",
            -0.7
        ),
        "filters": (
            "string()",
            "Filters for which the attenuation will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "V_B90 & FUV"
        )
    }

    def _init_code(self):
        self.av = {}
        self.av['young'] = float(self.parameters["Av_young"])
        self.av_old_factor = float(self.parameters["Av_old_factor"])
        self.av['old'] = self.av_old_factor * self.av['young']
        self.uv_bump_wavelength = float(self.parameters["uv_bump_wavelength"])
        self.uv_bump_width = float(self.parameters["uv_bump_width"])
        self.uv_bump_amplitude = float(self.parameters["uv_bump_amplitude"])
        self.powerlaw_slope = float(self.parameters["powerlaw_slope"])

        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        # We cannot compute the attenuation until we know the wavelengths. Yet,
        # we reserve the object.
        self.contatt = {}
        self.lineatt = {}

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wavelength = sed.wavelength_grid

        # Fλ fluxes (only from continuum)) in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute attenuation curve
        if len(self.contatt) == 0:
            att = alambda_av(wavelength, self.powerlaw_slope,
                             self.uv_bump_wavelength, self.uv_bump_width,
                             self.uv_bump_amplitude)
            for age in ['old', 'young']:
                self.contatt[age] = 10. ** (-.4 * self.av[age] * att)

        # Compute the attenuation curves on the line wavelength grid
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            att = alambda_av(linewl, self.powerlaw_slope,
                             self.uv_bump_wavelength, self.uv_bump_width,
                             self.uv_bump_amplitude)
            old_curve = 10 ** (-.4 * self.av['old'] * att)
            young_curve = 10 ** (-.4 * self.av['young'] * att)

            for name, old, young in zip(names, old_curve, young_curve):
                self.lineatt[name] = (old, young)

        dust_lumin = 0.
        contribs = [contrib for contrib in sed.luminosities if
                    'absorption' not in contrib]
        for contrib in contribs:
            age = contrib.split('.')[-1].split('_')[-1]
            luminosity = sed.luminosities[contrib]

            attenuation_spectrum = luminosity * (self.contatt[age] - 1.)
            dust_lumin -= np.trapz(attenuation_spectrum, wavelength)

            sed.add_module(self.name, self.parameters)
            sed.add_info("attenuation.Av." + contrib, self.av[age], unit='mag')
            sed.add_contribution("attenuation." + contrib, wavelength,
                                 attenuation_spectrum)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (linewl, old * self.lineatt[name][0],
                               young * self.lineatt[name][1])

        sed.add_info("attenuation.av_old_factor", self.av_old_factor)
        sed.add_info('attenuation.uv_bump_wavelength', self.uv_bump_wavelength,
                     unit='nm')
        sed.add_info('attenuation.uv_bump_width', self.uv_bump_width, unit='nm')
        sed.add_info("attenuation.uv_bump_amplitude", self.uv_bump_amplitude)
        sed.add_info("attenuation.powerlaw_slope", self.powerlaw_slope)

        # Total attenuation
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + dust_lumin, True, True,
                         unit='W')
        else:
            sed.add_info("dust.luminosity", dust_lumin, True, unit='W')

        # Fλ fluxes (only in continuum) in each filter after attenuation.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]),
                         unit='mag')


# SedModule to be returned by get_module
Module = PowerLawAtt
