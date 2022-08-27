# -*- coding: utf-8 -*-
# Copyright (C) 2017 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
BPASS v2 stellar emission module
==================================================

This module implements the BPASS v2 Single Stellar Populations.

"""

import numpy as np

from . import SedModule
from pcigale.data import SimpleDatabase

__category__ = "SSP"


class BPASSv2(SedModule):
    """BPASS v2 stellar emission module

    This SED creation module convolves the SED star formation history with a
    BPASS v2 single stellar population to add a stellar component to the SED.

    """

    parameters = {
        "imf": (
            "cigale_list(dtype=int, options=0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8)",
            "Initial mass function: 0 (-1.30 between 0.1 to 0.5Msun and -2.00 "
            "from 0.5 to 300Msun), 1 (-1.30 between 0.1 to 0.5Msun and -2.00 "
            "from 0.5 to 100Msun), 2 (-2.35 from 0.1 to 100Msun), 3 (-1.30 "
            "between 0.1 to 0.5Msun and -2.35 from 0.5 to 300Msun), 4 (-1.30 "
            "between 0.1 to 0.5Msun and -2.35 from 0.5 to 100Msun), 5 (-1.30 "
            "between 0.1 to 0.5Msun and -2.70 from 0.5 to 300Msun), 6 (-1.30 "
            "between 0.1 to 0.5Msun and -2.70 from 0.5 to 100Msun), 7 ("
            "Chabrier up to 100Msun), 8 (Chabrier up to 300Msun).",
            2
        ),
        "metallicity": (
            "cigale_list(options=0.00001 & 0.0001 & 0.001 & 0.002 & 0.003 & "
            "0.004 & 0.006 & 0.008 & 0.010 & 0.014 & 0.020 & 0.030 & 0.040)",
            "Metalicity. Possible values are: 0.00001, 0.0001, 0.001, 0.002, "
            "0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040.",
            0.02
        ),
        "binary": (
            "cigale_list(options=0 & 1)",
            "Single (0) or binary (1) stellar populations.",
            0
        ),
        "separation_age": (
            "cigale_list(dtype=int, minvalue=0)",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ),
    }

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.metallicity = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])
        self.binary = bool(self.parameters["binary"])

        with SimpleDatabase("bpassv2") as db:
            self.ssp = db.get(imf=self.imf, Z=self.metallicity,
                              binary=self.binary)

        self.wave = self.ssp.wl
        self.w_lymanc = np.where(self.wave <= 91.1)

    def convolve(self, sfh):
        """Convolve the SSP with a Star Formation History

        Given an SFH, this method convolves the info table and the SSP
        luminosity spectrum.

        Parameters
        ----------
        sfh: array of floats
            Star Formation History in Msun/yr.

        Returns
        -------
        spec_young: array of floats
            Spectrum in W/nm of the young stellar populations.
        spec_old: array of floats
            Same as spec_young but for the old stellar populations.
        info_young: dictionary
            Dictionary containing various information from the *.?color tables
            for the young stellar populations:
            * "m_star": Total mass in stars in Msun
            * "m_gas": Mass returned to the ISM by evolved stars in Msun
            * "n_ly": rate of H-ionizing photons (s-1)
        info_old : dictionary
            Same as info_young but for the old stellar populations.
        info_all: dictionary
            Same as info_young but for the entire stellar population. Also
            contains "age_mass", the stellar mass-weighted age

        """
        # We cut the SSP to the maximum age considered to simplify the
        # computation. We take only the first three elements from the
        # info table as the others do not make sense when convolved with the
        # SFH (break strength).
        info = self.ssp.info[:, :sfh.size]
        spec = self.ssp.spec[:, :sfh.size]

        # The convolution is just a matter of reverting the SFH and computing
        # the sum of the data from the SSP one to one product. This is done
        # using the dot product. The 1e6 factor is because the SFH is in solar
        # mass per year.
        info_young = 1e6 * np.dot(info[:, :self.separation_age],
                                  sfh[-self.separation_age:][::-1])
        spec_young = 1e6 * np.dot(spec[:, :self.separation_age],
                                  sfh[-self.separation_age:][::-1])

        info_old = 1e6 * np.dot(info[:, self.separation_age:],
                                sfh[:-self.separation_age][::-1])
        spec_old = 1e6 * np.dot(spec[:, self.separation_age:],
                                sfh[:-self.separation_age][::-1])

        info_all = info_young + info_old

        info_young = dict(zip(["m_star", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "n_ly"], info_all))

        info_all['age_mass'] = np.average(self.ssp.t[:sfh.size],
                                          weights=info[0, :] * sfh[::-1])

        return spec_young, spec_old, info_young, info_old, info_all


    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        out = self.convolve(sed.sfh)
        spec_young, spec_old, info_young, info_old, info_all = out

        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave_lymanc = self.wave[self.w_lymanc]
        lum_lyc_young = np.trapz(spec_young[self.w_lymanc], wave_lymanc)
        lum_lyc_old = np.trapz(spec_old[self.w_lymanc], wave_lymanc)

        # We do similarly for the total stellar luminosity
        lum_young, lum_old = np.trapz([spec_young, spec_old], self.wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.binary", float(self.binary))

        sed.add_info("stellar.m_star_young", info_young["m_star"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all["m_star"], True)
        sed.add_info("stellar.n_ly", info_all["n_ly"], True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)

        sed.add_contribution("stellar.old", self.wave, spec_old)
        sed.add_contribution("stellar.young", self.wave, spec_young)


# SedModule to be returned by get_module
Module = BPASSv2
