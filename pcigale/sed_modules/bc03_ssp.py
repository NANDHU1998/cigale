"""
Bruzual and Charlot (2003) stellar emission module for an SSP
=============================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule
from ..data import SimpleDatabase as Database
from pcigale.sed_modules import get_module


class BC03SSP(SedModule):

    parameter_list = OrderedDict([
        ("imf", (
            "cigale_list(dtype=int, options=0. & 1.)",
            "Initial mass function: 0 (Salpeter) or 1 (Chabrier).",
            0
        )),
        ("metallicity", (
            "cigale_list(options=0.0001 & 0.0004 & 0.004 & 0.008 & 0.02 & "
            "0.05)",
            "Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, "
            "0.02, 0.05.",
            0.02
        )),
        ("separation_age", (
            "cigale_list(dtype=int, minvalue=0)",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ))
    ])

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.metallicity = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])

        with Database("bc03_SSP") as db:
            if self.imf == 0:
                self.ssp = db.get(imf='salp', Z=self.metallicity)
            elif self.imf == 1:
                self.ssp = db.get(imf='chab', Z=self.metallicity)
            else:
                raise Exception("IMF #{} unknown".format(self.imf))

        self.rf = get_module("restframe_parameters")

    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        if 'ssp.index' in sed.info:
            index = sed.info['ssp.index']
        else:
            raise Exception('The stellar models do not correspond to pure SSP.')

        if self.ssp.t[index] <= self.separation_age:
            spec_young = self.ssp.spec[:, index]
            info_young = self.ssp.info[:, index]
            spec_old = np.zeros_like(spec_young)
            info_old = np.zeros_like(info_young)
        else:
            spec_old = self.ssp.spec[:, index]
            info_old = self.ssp.info[:, index]
            spec_young = np.zeros_like(spec_old)
            info_young = np.zeros_like(info_old)
        info_all = info_young + info_old

        info_young = dict(zip(["m_star", "m_gas", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "m_gas", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "m_gas", "n_ly"], info_all))
        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave = self.ssp.wl
        w = np.where(wave <= 91.1)
        lum_lyc_young, lum_lyc_old = np.trapz([spec_young[w], spec_old[w]],
                                              wave[w])

        # We do similarly for the total stellar luminosity
        lum_young, lum_old = np.trapz([spec_young, spec_old], wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)
        sed.add_info("stellar.age", self.ssp.t[index])

        sed.add_info("stellar.m_star_young", info_young["m_star"], True)
        sed.add_info("stellar.m_gas_young", info_young["m_gas"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.m_gas_old", info_old["m_gas"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all["m_star"], True)
        sed.add_info("stellar.m_gas", info_all["m_gas"], True)
        sed.add_info("stellar.n_ly", info_all["n_ly"], True)
        sed.add_info("stellar.lum_ly", lum_lyc_young + lum_lyc_old, True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)

        sed.add_info("stellar.beta0", self.rf.calz94(sed)[0])

# SedModule to be returned by get_module
Module = BC03SSP
