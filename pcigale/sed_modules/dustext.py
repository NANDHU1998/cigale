"""
Simple screen extinction module
=====================================================================

This module implements various screen extinction laws, including NW, LMC, and
SMC laws.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule

def ccm(wave, Rv):
    """ Computes the MW extinction law using the Cardelli, Clayton & Mathis
    (1989) curve valid from 1250 Angstroms to 3.3 microns.

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
                                                wn-8.)) +
                               np.polyval([0.374, -0.420, 4.257, 13.670],
                                          wn - 8.))

    return np.piecewise(x, [cond1, cond2, cond3, cond4, cond5],
                        [fcond1, fcond2, fcond3, fcond4, fcond5])


def Pei92(wave, Rv=None, law='mw'):
    """ Compute the extinction law using the Pei92 (1989) MW, LMC, and SMC
    curves valid from 912 Angstroms to 25 microns.

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


class DustExtinction(SedModule):
    """Screen extinction law

    This module computes the screen extinction from various classical
    extinctions laws for the MW, the LMC, and the SMC

    The extinction is computed for all the components and is added to the SED as
    a negative contribution.

    """

    parameter_list = OrderedDict([
        ("E_BV", (
            "cigale_list(minvalue=0.)",
            "E(B-V), the colour excess.",
            0.3
        )),
        ("Rv", (
            "cigale_list(minvalue=0.)",
            "Ratio of total to selective extinction, A_V / E(B-V). The "
            "standard value is 3.1 for MW using CCM89. For SMC and LMC using "
            "Pei92 the values should be 2.93 and 3.16.",
            3.1
        )),
        ("law", (
            "cigale_list(dtype=int, options=0 & 1 & 2)",
            "Extinction law to apply. The values are 0 for CCM, 1 for SMC, and "
            "2 for LCM.",
            0
        )),
        ("filters", (
            "string()",
            "Filters for which the extinction will be computed and added to "
            "the SED information dictionary. You can give several filter "
            "names separated by a & (don't use commas).",
            "B_B90 & V_B90 & FUV"
        ))
    ])

    def _init_code(self):
        """Get the filters from the database"""

        self.ebv = float(self.parameters["E_BV"])
        self.law = int(self.parameters["law"])
        self.Rv = float(self.parameters["Rv"])
        self.filter_list = [item.strip() for item in
                            self.parameters["filters"].split("&")]
        # We cannot compute the extinction until we know the wavelengths. Yet,
        # we reserve the object.
        self.att = None
        self.lineatt = {}

    def process(self, sed):
        """Add the extinction component to each of the emission components.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wl = sed.wavelength_grid

        # Fλ fluxes (only from continuum) in each filter before extinction.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute stellar extinction curve
        if self.att is None:
            if self.law == 0:
                self.att = 10 ** (-.4 * ccm(wl, self.Rv) * self.ebv)
            elif self.law == 1:
                self.att = 10 ** (-.4 * Pei92(wl, Rv=self.Rv, law='smc') *
                                  self.ebv)
            elif self.law == 2:
                self.att = 10 ** (-.4 * Pei92(wl, self.Rv, law='lmc') *
                                  self.ebv)

        # Compute nebular extinction curves
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            if self.law == 0:
                self.lineatt['nebular'] = ccm(wl, self.Rv)
                for name,  att in zip(names, ccm(linewl, self.Rv)):
                    self.lineatt[name] = att
            elif self.law == 1:
                self.lineatt['nebular'] = Pei92(wl, law='smc')
                for name,  att in zip(names, Pei92(linewl, law='smc')):
                    self.lineatt[name] = att
            elif self.law == 2:
                self.lineatt['nebular'] = Pei92(wl, law='lmc')
                for name,  att in zip(names, Pei92(linewl, law='lmc')):
                    self.lineatt[name] = att
            for k, v in self.lineatt.items():
                self.lineatt[k] = 10. ** (-.4 * v * self.ebv)

        dust_lumin = 0.
        contribs = [contrib for contrib in sed.luminosities if
                    'absorption' not in contrib]

        for contrib in contribs:
            luminosity = sed.luminosities[contrib]
            if 'nebular' in contrib:
                extinction_spec = luminosity * (self.lineatt['nebular'] - 1.)
            else:
                extinction_spec = luminosity * (self.att - 1.)
            dust_lumin -= np.trapz(extinction_spec, wl)

            sed.add_module(self.name, self.parameters)
            sed.add_contribution("attenuation." + contrib, wl,
                                 extinction_spec)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (linewl, old * self.lineatt[name],
                               young * self.lineatt[name])

        # Total extinction
        if 'dust.luminosity' in sed.info:
            sed.add_info("dust.luminosity",
                         sed.info["dust.luminosity"] + dust_lumin, True, True)
        else:
            sed.add_info("dust.luminosity", dust_lumin, True)

        # Fλ fluxes (only from continuum) in each filter after extinction.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info("attenuation." + filt,
                         -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]))

        sed.add_info('attenuation.E_BV', self.ebv)


# SedModule to be returned by get_module
Module = DustExtinction
