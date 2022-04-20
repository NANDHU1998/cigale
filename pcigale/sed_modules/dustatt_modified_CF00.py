"""
Modified Charlot & Fall 2000 dust attenuation module
===================================

This module implements an attenuation law combining the birth cloud (BC)
attenuation and the interstellar medium (ISM) attenuation, each one modelled by
a power law. The young star emission is attenuated by the BC and the ISM
attenuations whereas the old star emission is only affected by the ISM. This
simple model was proposed by Charlot & Fall (2000). This module is described in
more detail in Lo Faro et al. (2017).

Parameters available for analysis
---------------------------------

- attenuation.Av_ISM: Av attenuation in the interstellar medium
- attenuation.mu: Av_ISM / (Av_BV+Av_ISM)
- attenuation.slope_BC: slope of the power law in the birth clouds
- attenuation.slope_ISM: slope of the power law in the ISM
- attenuation.<NAME>: amount of total attenuation in the luminosity
    contribution <NAME>
- attenuation.<FILTER>: total attenuation in the filter
"""

import numpy as np

from . import SedModule

__category__ = "dust attenuation"


def alambda_av(wl, delta):
    """Compute the complete attenuation curve A(λ)/Av

    Attenuation curve of the form (λ / 550 nm) ** δ. The Lyman continuum is not
    attenuated.

    Parameters
    ----------
    wl: array of floats
        The wavelength grid in nm.
    delta: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/Av attenuation at each wavelength of the grid.

    """
    attenuation = (wl / 550.) ** delta

    # Lyman continuum not attenuated.
    attenuation[wl <= 91.2] = 0.

    return attenuation


class ModCF00Att(SedModule):
    """Two power laws attenuation module

    Attenuation module combining the birth cloud (BC) attenuation and the
    interstellar medium (ISM) one.

    The attenuation can be computed on the whole spectrum or on a specific
    contribution and is added to the SED as a negative contribution. See
    Lo Faro et al. (2017).

    """

    parameter_list = {
        "Av_ISM": (
            "cigale_list(minvalue=0)",
            "V-band attenuation in the interstellar medium.",
            1.
        ),
        "mu": (
            "cigale_list(minvalue=.0001, maxvalue=1.)",
            "Av_ISM / (Av_BC+Av_ISM)",
            0.44
        ),
        "slope_ISM": (
            "cigale_list()",
            "Power law slope of the attenuation in the ISM.",
            -0.7
        ),
        "slope_BC": (
            "cigale_list()",
            "Power law slope of the attenuation in the birth clouds.",
            -1.3
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
        self.Av_ISM = float(self.parameters['Av_ISM'])
        self.mu = float(self.parameters['mu'])
        self.slope_ISM = float(self.parameters['slope_ISM'])
        self.slope_BC = float(self.parameters['slope_BC'])
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        self.Av_BC = self.Av_ISM * (1. - self.mu) / self.mu
        self.contatt = {}
        self.lineatt = {}

    def process(self, sed):
        """Add the dust attenuation to the SED.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        wl = sed.wavelength_grid

        # Compute the attenuation curves on the continuum wavelength grid
        if len(self.contatt) == 0:
            self.contatt['old'] = 10. ** (-.4 * alambda_av(wl, self.slope_ISM) *
                                          self.Av_ISM)
            # Emission from the young population is attenuated by both
            # components
            self.contatt['young'] = 10. ** (-.4 * alambda_av(wl, self.slope_BC) *
                                            self.Av_BC) * self.contatt['old']

        # Compute the attenuation curves on the line wavelength grid
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            old_curve = 10.**(-.4 * alambda_av(linewl, self.slope_ISM) *
                              self.Av_ISM)
            young_curve = 10.**(-.4 * alambda_av(linewl, self.slope_BC) *
                                self.Av_BC) * old_curve

            for name, old, young in zip(names, old_curve, young_curve):
                self.lineatt[name] = (old, young)

        # Fλ fluxes in each filter before attenuation.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        dust_lumin = 0.
        contribs = [contrib for contrib in sed.luminosities if
                    'absorption' not in contrib]

        for contrib in contribs:
            age = contrib.split('.')[-1].split('_')[-1]
            luminosity = sed.luminosities[contrib]

            attenuation_spectrum = luminosity * (self.contatt[age] - 1.)
            dust_lumin -= np.trapz(attenuation_spectrum, wl)

            sed.add_module(self.name, self.parameters)
            sed.add_contribution("attenuation." + contrib, wl,
                                 attenuation_spectrum)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (linewl, old * self.lineatt[name][0],
                               young * self.lineatt[name][1])

        sed.add_info('attenuation.Av_ISM', self.Av_ISM, unit='mag')
        sed.add_info('attenuation.Av_BC', self.Av_BC, unit='mag')
        sed.add_info('attenuation.mu', self.mu)
        sed.add_info('attenuation.slope_BC', self.slope_BC)
        sed.add_info('attenuation.slope_ISM', self.slope_ISM)

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
            att = -2.5 * np.log10(flux_att[filt] / flux_noatt[filt])
            sed.add_info("attenuation." + filt, max(0., att), unit='mag')


# CreationModule to be returned by get_module
Module = ModCF00Att
